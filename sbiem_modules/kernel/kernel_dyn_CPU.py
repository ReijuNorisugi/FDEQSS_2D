# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260309.
"""

# This is the module for elasto-dynamic kernel computation.
# This is specialized for CPU computation.

import numpy as np
import torch as tr
import torch_dct as dct
from scipy.special import j0 as j0
from scipy.special import j1 as j1
from scipy.special import struve
from scipy.special import sici
import sys
import os
from concurrent.futures import ThreadPoolExecutor

class Kernel():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.mode     = Conditions['mode']     # Flag of rupture mode.
        self.qd       = Conditions['qd']       # Flag for quasi-dynamic.
        self.num_GPU  = Conditions['num_GPU']  # Number of GPU.
        self.rmPB     = Conditions['rmPB']     # Flag for removing periodic boundaries.
        self.mirror   = Conditions['mirror']   # Flag for mirror.
        self.Stepper  = Conditions['Stepper']  # Flag for time marching scheme.
        self.outerror = Conditions['outerror']  # Flag for error monitoring.
        self.lam   = Medium['lam']      # Computational size.
        self.hcell = Medium['hcell']    # Cell size.
        self.kn    = Medium['kn']       # Wavenumber.
        self.mu    = Medium['mu']       # Rigidity.
        self.nu    = Medium['nu']       # Poisson's ratio.
        self.Tw    = Medium['Tw']       # Truncation window.
        self.Nele  = Medium['Nele']     # System size.
        self.ncell = Medium['ncell']    # Number of cells.
        self.nconv = self.Nele          # Number of cells for convolution.
        self.nperi = Medium['nperi']    # Replication period.
        self.dtmin = Medium['dtmin']    # Minimum time step.
        self.device = Devices['device_1']

        self.plot_kernel = False    # Flag to visalize the convolutional kernel.

        self.Cs = Medium['Cs']   # S-wave velocity.
        self.Cp = Medium['Cp']   # P-wave velocity.
        if self.mode == 'III' or self.mode == 'IV':
            self.C = self.Cs
        elif self.mode == 'II':
            self.C = self.Cp
        self.alpha = self.Cp / self.Cs    # Coefficient.
        self.inv_alpha = (self.Cs / self.Cp)**2 # Coefficient.
        self.alpha2 = self.alpha**2       # Alpha square.

        # Coefficients for statick kernel with periodic boundary.
        self.coef = 0.5 * self.mu * np.abs(self.kn)
        self.coef = tr.as_tensor(self.coef, dtype=tr.float64, device='cpu')
        if self.rmPB:
            # Coefficients for statick kernel without periodic boundary.
            if not self.mirror:
                self.coef_sinc = 2. * self.mu * np.arange(0, (0.5*self.Nele)+1, 1) / self.lam
                self.coef_sinc = tr.as_tensor(self.coef_sinc, dtype=tr.float64, device='cpu')
            else:
                self.coef_sinc = self.mu * np.arange(0, self.Nele, 1) / self.lam
                self.coef_sinc = tr.as_tensor(self.coef_sinc, dtype=tr.float64, device='cpu')

        # Remove periodic boundary condition if rmPB is True.
        if not self.rmPB:
            # 2-dimensional array of wavenumber times c_{S} for broadcasting.
            self.ck = np.array([np.abs(self.kn) * self.Cs])
        else:
            # Lw includes fault domain and extended zero-padding region to construct clean dynamic kernel.
            self.Lw = self.lam*0.5 + self.C * self.Tw
            # Number of grids.
            self.n_ext = np.arange(0, self.Lw / self.hcell + 1, 1)
            # Angular wavenumber.
            if not self.mirror:
                self.kn_ext = 2. * np.pi * self.n_ext[0:int(self.n_ext[-1]/2)+1] / self.Lw
            else: # For mirror, the system size includes mirror image.
                self.kn_ext = np.pi * self.n_ext / self.Lw
            # 2-dimensional array of wavenumber times c_{S} for broadcasting.
            self.ck = np.array([np.abs(self.kn_ext) * self.Cs])
            # Coefficient for static kernel with extended cordinate.
            self.coef_ext = 0.5 * self.mu * np.abs(self.kn_ext)

        # Set minimum time step for temporal convolution.
        if self.Stepper == 'LR':
            self.min_step = self.dtmin
        elif self.Stepper == 'LR_mdf':
            self.min_step = self.dtmin
        elif self.Stepper == 'RO':
            # If stepper is RO, we use half of dtmin for double-half step solution.
            self.min_step = self.dtmin * 0.5
        elif self.Stepper == 'CS':
            self.min_step = self.dtmin
        elif self.outerror:
            self.min_step = self.dtmin * 0.5


        self.num = int(self.Tw // self.min_step)  # Index for history.
        self.thist = np.array([np.arange(0, self.num) * self.min_step + self.min_step * 0.5]).T # 2-dimensional array for broadcasting.
        self.thist = np.flipud(self.thist) # Flip for delay time.
        self.num_splits = 32                # Number of split for parallel kernel computation.

        # Defining functions in advance.
        self.tr_sum = tr.sum
        self.tr_irfft = tr.fft.irfft
        self.tr_rfft = tr.fft.rfft
        self.tr_dct = dct.dct
        self.tr_idct = dct.idct
        self.tr_cat = tr.cat
        self.tr_chunk = tr.chunk

        self.isnan = np.isnan
        self.pi = np.pi

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def reset_kernel(self):
        if not self.qd:
            self.kernel_st = None
            self.kernel_sum = None
            self.kernel = None

    # Integral( TJ1(T)dT ).
    def j1_int(self, tim):
        s0 = struve(0, tim)
        s1 = struve(1, tim)
        id_nan = self.isnan(s0)
        id_nan_ = self.isnan(s1)
        # I struggled to find this error for 2 weeks.
        s0[id_nan] = 0.
        s1[id_nan_] = 0.
        return 0.5*self.pi*tim*(s0*j1(tim) - s1*j0(tim))


    # Integral( J0(T)dT ).
    def j0_int(self, tim):
        return tim*j0(tim) + self.j1_int(tim)


    # Integral( TW(T)dT ).
    def W_int(self, tim):
        return 0.5*((tim**2)*self.W(tim) + self.j1_int(tim))


    # 1 - Integral( J1(T)/T dT ) = W(T).
    def W(self, tim):
        return 1. - tim * j0(tim) + j1(tim) - self.j1_int(tim)


    def check_nan(self):
        if np.any(self.isnan(self.kernel)):
            print('!!!!!!!!!!!!!!!!!   The kernel contains NAN   !!!!!!!!!!!!!!!!!!!')
            sys.exit()
        else:
            pass

    # Functions to prepare convolutional kernel.
    # This is processed in parallel for wavenumber.
    def KI(self):
        arg_chunks = np.array_split(self.arg, self.num_splits, axis=1)
        arg__chunks = np.array_split(self.alpha * self.arg, self.num_splits, axis=1)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.KI_chunk, a, a_) for a, a_ in zip(arg_chunks, arg__chunks)]
            results = [f.result() for f in futures]
        self.kernel = np.concatenate(results, axis=1)

    def KI_chunk(self, arg_chunk, arg__chunk):
        return (2 * (1 - self.inv_alpha) - self.alpha2 * (1 - self.W(arg__chunk))
                - ( 4 * (self.W_int(arg_chunk) - self.inv_alpha * self.W_int(arg__chunk)) )
                - ( (4 - self.inv_alpha) * self.j0_int(arg__chunk) - 4 * self.j0_int(arg_chunk) )
                )

    def KII(self):
        arg_chunks = np.array_split(self.arg, self.num_splits, axis=1)
        arg__chunks = np.array_split(self.alpha * self.arg, self.num_splits, axis=1)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.KII_chunk, a, a_) for a, a_ in zip(arg_chunks, arg__chunks)]
            results = [f.result() for f in futures]
        self.kernel = np.concatenate(results, axis=1)

    def KII_chunk(self, arg_chunk, arg__chunk):
        return (2 * (1 - self.inv_alpha) - (1. - self.W(arg_chunk))
                - 4. * ( self.inv_alpha * self.W_int(arg__chunk) - self.W_int(arg_chunk) )
                - ( 3. * self.j0_int(arg_chunk) - 4. * self.inv_alpha * self.j0_int(arg__chunk) )
                )

    def KIII(self):
        arg_chunk = np.array_split(self.arg, self.num_splits, axis=1)
        with ThreadPoolExecutor(max_workers=self.num_splits) as executor:
            results = list(executor.map(self.W, arg_chunk))
        self.kernel = np.concatenate(results, axis=1)

    # Prepare dynamic convolutional kernel with periodic boundary.
    # Mode I, II, III, and IV are available now.
    def comp_kernel(self):
        self.arg = self.ck * self.thist    # Argment for kernel preparation.
        if self.mode == 'II':
            if not self.qd:
                # Compute dynamic kernel.
                self.KII()
            # Compute static kernel.
            self.kernel_st = (- self.coef / (1 - self.nu)).to(self.device)
        elif self.mode == 'III' or self.mode == 'IV':
            if not self.qd:
                self.KIII()
            self.kernel_st = (- self.coef).to(self.device)
        elif self.mode == 'I':
            if not self.qd:
                self.KI()
            self.kernel_st = (- self.coef / (1 - self.nu)).to(self.device)
        if self.qd:
            self.arg = None
            return
        else:
            self.kernel = tr.from_numpy(np.ascontiguousarray(self.kernel))
            self.kernel = self.coef * self.kernel * self.min_step
            if self.plot_kernel:
                self.plot()
            #  Dynamic kernel when dt > Tw.
            self.kernel_sum = tr.sum(self.kernel, axis=0)
            self.arg = None


    # Compute dynamic kernel without periodic boundary.
    # Mode I, II, are III are available now.
    def comp_kernel_without_pb(self):
        self.arg = self.ck * self.thist # Argment for kernel preparation.
        if self.mode == 'II':
            if not self.qd:
                # Compute dynamic kernel with periodic boundary using extended space.
                # This is clean dynamic kernel which is not affected by periodicity.
                self.KII()
                self.kernel = self.coef_ext * self.kernel
                # Static kernel with periodic boundary.
                self.kernel_st_pd = (- self.coef_ext / (1 - self.nu))
            if not self.mirror:
                self.sici = tr.tensor(sici(np.pi * np.arange(0, (0.5*self.Nele)+1, 1))[0],
                                      dtype=tr.float64, device='cpu')
            else:
                self.sici = tr.tensor(sici(2. * np.pi * np.arange(0, self.Nele, 1))[0],
                                      dtype=tr.float64, device='cpu')
            # Static kernel without periodic boundary.
            self.kernel_st = - self.coef_sinc * self.sici / (1. - self.nu)
        elif self.mode == 'III' or self.mode == 'IV':
            if not self.qd:
                self.KIII()
                self.kernel = self.coef_ext * self.kernel
                self.kernel_st_pd = - self.coef_ext
            if not self.mirror:
                self.sici = tr.tensor(sici(np.pi * np.arange(0, (0.5*self.Nele)+1, 1))[0],
                                      dtype=tr.float64, device='cpu')
            else:
                self.sici = tr.tensor(sici(np.pi * np.arange(0, self.Nele, 1))[0],
                                      dtype=tr.float64, device='cpu')
            self.kernel_st = - self.coef_sinc * self.sici
        elif self.mode == 'I':
            if not self.qd:
                self.KI()
                self.kernel = self.coef_ext * self.kernel
                self.kernel_st_pd = (- self.coef_ext / (1 - self.nu))
            if not self.mirror:
                self.sici = tr.tensor(sici(np.pi * np.arange(0, (0.5*self.Nele)+1, 1))[0],
                                      dtype=tr.float64, device='cpu')
            else:
                self.sici = tr.tensor(sici(np.pi * np.arange(0, self.Nele, 1))[0],
                                      dtype=tr.float64, device='cpu')
            self.kernel_st = - self.coef_sinc * self.sici / (1. - self.nu)

        if not self.qd:
            # K_dyn(clean) + K_st with periodic boundary.
            self.kernel += self.kernel_st_pd
            self.kernel = self.kernel * np.ones(self.kernel.shape[1], dtype=np.float64) * self.min_step

            # Synmetric spatial representation of K_dyn + K_st.
            if not self.mirror:
                self.kernel = np.fft.irfft(self.kernel, n=int(self.n_ext[-1]), axis=1)
                self.kernel = self.kernel[:, 0:int(0.5*self.Nele)+1]
            else:
                # This should include mirror image.
                self.kernel = np.fft.irfft(self.kernel, n=int(2*self.n_ext[-1]), axis=1)
                self.kernel = self.kernel[:, 0:self.Nele]
            self.kernel_ = np.flip(self.kernel[:, 1:-1], axis=1)
            self.kernel = np.hstack([self.kernel, self.kernel_])
            self.kernel_ = None

            # Back to frequency domain.
            self.kernel = tr.tensor(np.real(np.fft.rfft(self.kernel, axis=1)), dtype=tr.float64, device='cpu')
            # Remove K_st without peridoci boundary.
            # self.kernel is dynamic kernel for convolution without periodic boundary.
            self.kernel -= self.kernel_st * self.min_step
            self.kernel_sum = tr.sum(self.kernel, axis=0)

            # You can visualize the kernel. Figures are stored in /Check.
            if self.plot_kernel:
                self.plot()
        # Remove memory-consuming variable.
        self.arg = None

    def chunk_tensor(self):
        pass

    def send_kernel(self):
        pass

    # Visalize kernel if needed.
    def plot(self):
        if self.plot_kernel:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            os.makedirs('Check', exist_ok=True)

            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(self.kernel.shape[1]):
                if i < 10:
                    ax.plot(self.thist,
                            (self.kernel[:, i]-tr.min(self.kernel[:, i]))/(tr.max(self.kernel[:, i])-tr.min(self.kernel[:, i])),
                            lw=3, color=cm.managua(i/10))
            ax.set_xlabel('Delay time (sec)', fontsize=13)
            ax.set_ylabel('Normalized kernel', fontsize=13)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            if self.rmPB:
               plt.savefig('Check/Kernel_low_{}_noPD.png'.format(self.mode), dpi=600)
            else:
                plt.savefig('Check/Kernel_low_{}_PD.png'.format(self.mode), dpi=600)
            plt.show()
            plt.close()

            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(self.kernel.shape[1]):
                if i < 50 and i >= 10:
                    plt.plot(self.thist,
                             (self.kernel[:, i]-tr.min(self.kernel[:, i]))/(tr.max(self.kernel[:, i])-tr.min(self.kernel[:, i])),
                             color=cm.managua(i/40))
            ax.set_xlabel('Delay time (sec)', fontsize=13)
            ax.set_ylabel('Normalized kernel', fontsize=13)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            if self.rmPB:
               plt.savefig('Check/Kernel_middle_{}_noPD.png'.format(self.mode), dpi=600)
            else:
                plt.savefig('Check/Kernel_middle_{}_PD.png'.format(self.mode), dpi=600)
            plt.show()
            plt.close()

            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(self.kernel.shape[1]):
                if i >= 50:
                    plt.plot(self.thist,
                             (self.kernel[:, i]-tr.min(self.kernel[:, i]))/(tr.max(self.kernel[:, i])-tr.min(self.kernel[:, i])),
                             color=cm.managua(i/462))
            ax.set_xlabel('Delay time (sec)', fontsize=13)
            ax.set_ylabel('Normalized kernel', fontsize=13)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            if self.rmPB:
               plt.savefig('Check/Kernel_high_{}_noPD.png'.format(self.mode), dpi=600)
            else:
                plt.savefig('Check/Kernel_high_{}_PD.png'.format(self.mode), dpi=600)
            plt.show()
            plt.close()
