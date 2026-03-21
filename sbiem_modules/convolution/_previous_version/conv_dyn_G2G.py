# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20250928.
"""

# This is the module for elasto-dynamic kernel computation and dynamic convolution.
# This is specialilzed for communication with multiple GPUs when they are directly connected.

import sbiem_modules.Load_pickle as load
import numpy as np
import scipy.fft as fft
import torch as tr
#import torch_dct as dct
from scipy.special import j0 as j0
from scipy.special import j1 as j1
from scipy.special import struve
from scipy.special import sici
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Class for static and dynamic convolution.
class Convolution():
    def __init__(self):
        # Load dictionaries.
        fname = load.load('fname.pkl')['fname']
        Medium = load.load('Output{}Medium.pkl'.format(fname))
        Conditions = load.load('Output{}Conditions.pkl'.format(fname))
        Devices = load.load('Output{}Devices.pkl'.format(fname))
        
        self.mode     = Conditions['mode']     # Flag of rupture mode.
        self.qd       = Conditions['qd']       # Flag for quasi-dynamic.
        self.num_GPU  = Conditions['num_GPU']  # Number of GPU.
        self.rmPB     = Conditions['rmPB']     # Flag for removing periodic boundaries.
        self.act_res  = Conditions['act_res']  # Flag ro activate restart saveing function.
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
        self.alpha = self.Cp / self.Cs          # Coefficient.
        self.inv_alpha = (self.Cs / self.Cp)**2 # Coefficient.
        self.alpha2 = self.alpha**2             # Alpha square.
        
        # Coefficients for static kernel.
        self.coef = 0.5 * self.mu * np.abs(self.kn)
        self.coef = tr.as_tensor(self.coef, dtype=tr.float64, device='cpu')
        if not self.mirror:
            self.coef_sinc = 2. * self.mu * np.arange(0, (0.5*self.Nele)+1, 1) / self.lam
        else:
            print('not yet')
            sys.exit()
            self.coef_sinc = self.mu * np.arange(0, self.Nele, 1) / self.lam
        self.coef_sinc = tr.as_tensor(self.coef_sinc, dtype=tr.float64, device='cpu')
        
        # Remove periodic boundary condition if rmPB is True.
        if not self.rmPB:
            self.ck = np.array([np.abs(self.kn) * self.Cs]) # 2-dimensional array for broadcasting.
        else:
            self.Lw = self.lam*0.5 + self.C * self.Tw
            if not self.mirror:
                self.n_ext = np.arange(0, self.Lw / self.hcell + 1, 1)
                self.kn_ext = 2. * np.pi * self.n_ext[0:int(self.n_ext[-1]/2)+1] / self.Lw
            else:
                self.n_ext = np.arange(0, self.Lw / self.hcell, 1)
                self.kn_ext = np.pi * self.n_ext / self.Lw
            
            self.coef_ext = 0.5 * self.mu * np.abs(self.kn_ext)
            self.ck = np.array([np.abs(self.kn_ext) * self.Cs])
            self.V_rel = np.zeros(self.n_ext.shape, dtype=np.float64)
            self.V_rel[0] = 1.
            if not self.mirror:
                self.V_rel = np.fft.rfft(self.V_rel)
            else:
                self.V_rel = fft.dct(self.V_rel)

        if self.Stepper == 'LR':
            self.min_step = self.dtmin
        elif self.Stepper == 'LR_mdf':
            self.min_step = self.dtmin
        elif self.Stepper == 'RO': # If stepper is RO, we use half of dtmin for double-half step solutions.
            self.min_step = self.dtmin * 0.5
        elif self.Stepper == 'CS':
            self.min_step = self.dtmin
        elif self.outerror:
            self.min_step = self.dtmin * 0.5
        
        self.num = int(self.Tw // self.min_step)  # Index for history.
        self.thist = np.array([np.arange(0, self.num) * self.min_step + self.min_step * 0.5]).T # 2-dimensional array for broadcasting.
        self.thist = np.flipud(self.thist) # Flip for delay time.
        self.num_splits = 32                # Number of aplit for parallel kernel computation.
        
        if not self.mirror:
            self.dtype_np = np.complex128 # You should explicitly choose dtype.
            self.dtype_tr = tr.complex128 # You should explicitly choose dtype.
        else:
            self.dtype_np = np.float64 # You should explicitly choose dtype.
            self.dtype_tr = tr.float64 # You should explicitly choose dtype.
            # For zero-padding.
            self.pad = tr.zeros(int(self.Nele*(self.nperi-1)/self.nperi), dtype=self.dtype_tr, device=self.device)
        
        self.inst = self.num  # Number of steps convolved instantaneously.
        self.latest = 0       # Current index.
        self.index = 0        # Index for efficient computation.
        self.full_size = len(self.thist)
        self.previously_dt_ge_Tw = False

        # Save dynamic kernel after first time interation for efficiency.
        self.K = tr.zeros(len(self.kn), dtype=self.dtype_tr, device=self.device)
        
        if self.num_GPU == 0:
            self.dDell_hist = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
        elif self.num_GPU == 1:
            self.dDell_hist = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cuda:0')
        elif self.num_GPU == 2:
            self.dDell_hist_1 = tr.zeros(len(self.thist), int(len(self.kn)//2)+1, dtype=self.dtype_tr, device='cuda:0')
            self.dDell_hist_2 = tr.zeros(len(self.thist), int(len(self.kn)//2), dtype=self.dtype_tr, device='cuda:1')
        elif self.num_GPU == 4:
            self.dDell_hist_1 = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:0')
            self.dDell_hist_2 = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:1')
            self.dDell_hist_3 = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:2')
            self.dDell_hist_4 = tr.zeros(len(self.thist), int(len(self.kn)//4)-2, dtype=self.dtype_tr, device='cuda:3')
        
        
        # Set streams for paralell convolution.
        if self.num_GPU == 2:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
        elif self.num_GPU == 4:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.stream_3 = tr.cuda.Stream(device='cuda:2')
            self.stream_4 = tr.cuda.Stream(device='cuda:3')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
            self.event_3 = tr.cuda.Event()
            self.event_4 = tr.cuda.Event()
        
        # Defining functions in advance.
        self.tr_sum = tr.sum
        self.tr_irfft = tr.fft.irfft
        self.tr_rfft = tr.fft.rfft
        #self.tr_dct = dct.dct
        #self.tr_idct = dct.idct
        self.tr_cat = tr.cat
        self.tr_zeros = tr.zeros
        self.tr_div = tr.div
        self.tr_einsum = tr.einsum
        self.tr_int = tr.int64
        self.tr_chunk = tr.chunk
        
        self.isnan = np.isnan
        self.pi = np.pi
        
        self.cuda_stream = tr.cuda.stream
        self.synchronize = tr.cuda.synchronize
        
        del fname, Medium, Conditions, Devices


    def __getstate__(self):
        state = self.__dict__.copy()
        if self.num_GPU == 2:
            state['stream_1'] = None
            state['stream_2'] = None
            state['event_1'] = None
            state['event_2'] = None
        elif self.num_GPU == 4:
            state['stream_1'] = None
            state['stream_2'] = None
            state['stream_3'] = None
            state['stream_4'] = None
            state['event_1'] = None
            state['event_2'] = None
            state['event_3'] = None
            state['event_4'] = None
        return state
        
        
    def __setstate__(self, state):
        self.__dict__ = state
        if self.num_GPU == 2:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
        elif self.num_GPU == 4:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.stream_3 = tr.cuda.Stream(device='cuda:2')
            self.stream_4 = tr.cuda.Stream(device='cuda:3')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
            self.event_3 = tr.cuda.Event()
            self.event_4 = tr.cuda.Event()
    
    
    def reset_kernel(self):
        if not self.qd:
            self.kernel_st = None
            self.kernel_sum = None
            self.kernel = None
    
    def reset_history(self):
        if not self.qd:
            if self.num_GPU == 0 or self.num_GPU == 1:
                self.dDell_hist = None
            elif self.num_GPU == 2:
                self.kernel_1 = None
                self.kenrle_2 = None
                self.dDell_hist_1 = None
                self.dDell_hist_2 = None
            elif self.num_GPU ==4 :
                self.kernel_1 = None
                self.kernel_2 = None
                self.kernel_3 = None
                self.kernel_4 = None
                self.dDell_hist_1 = None
                self.dDell_hist_2 = None
                self.dDell_hist_3 = None
                self.dDell_hist_4 = None
            

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
    # This is processed in parallel.
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
    
    
    # Prepare dynamic convolutional kernel here, mode I, II, III, and IV are available.
    def comp_kernel(self):
        self.arg = self.ck * self.thist    # Argment for kernel preparation.
        if self.mode == 'II':
            if not self.qd:
                self.KII()
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
            self.kernel_sum = tr.sum(self.kernel, axis=0)   # For instantaneous convolution when the time step is larger than Tw.
            if self.num_GPU != 0:
                self.send_kernel()
            self.arg = None
        
    
    # Prepare dynamic convolutional kernel without periodic boundaries.
    # Mode I, II, and III are available now.
    def comp_kernel_without_pb(self):
        self.arg = self.ck * self.thist    # Argment for kernel preparation.
        if self.mode == 'II':
            if not self.qd:
                self.KII()
                self.kernel = self.coef_ext * self.kernel
                self.kernel_st_pd = (- self.coef_ext / (1 - self.nu))
            if not self.mirror:
                self.sici = tr.tensor(sici(np.pi * np.arange(0, (0.5*self.Nele)+1, 1))[0], 
                                      dtype=tr.float64, device='cpu')
            else:
                self.sici = tr.tensor(sici(2. * np.pi * np.arange(0, self.Nele, 1))[0], 
                                      dtype=tr.float64, device='cpu')
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
            self.kernel += self.kernel_st_pd
            self.kernel = self.kernel * self.V_rel * self.min_step
        
            if not self.mirror:
                self.kernel = np.fft.irfft(self.kernel, n=int(self.n_ext[-1]), axis=1)
            else:
                self.kernel = fft.idct(self.kernel, n=int(self.n_ext[-1]), axis=1)
            self.kernel = self.kernel[:, 0:int(0.5*self.Nele)+1]
            self.kernel_ = np.flip(self.kernel[:, 1:-1], axis=1)
            self.kernel = np.hstack([self.kernel, self.kernel_])
            self.kernel_ = None
            if not self.mirror:
                self.kernel = tr.tensor(np.real(np.fft.rfft(self.kernel, axis=1)), dtype=tr.float64, device='cpu')
            else:
                self.kernel = tr.tensor(np.real(fft.dct(self.kernel, axis=1)), dtype=tr.float64, device='cpu')
            self.kernel -= self.kernel_st * self.min_step
            if self.plot_kernel:
                self.plot()
            self.kernel_sum = tr.sum(self.kernel, axis=0) # For instantaneous convolution when the time step is larger than Tw.
        
        if self.num_GPU != 0:
            self.kernel_st = self.kernel_st.to('cuda:0')
        
        self.arg = None
    
    
    def chunk_tensor(self):
        if self.num_GPU == 2:
            self.kernel_1, self.kernel_2 = tr.chunk(self.kernel, 2, dim=1)
        elif self.num_GPU == 4:
            self.kernel_1, self.kernel_2, self.kernel_3, self.kernel_4 = tr.chunk(self.kernel, 4, dim=1)
            
    
    def send_kernel(self):
        if not self.qd:
            if self.num_GPU == 1:
                self.kernel = self.kernel.to('cuda:0')
                self.kernel_sum = self.kernel_sum.to('cuda:0')
                self.kernel_st = self.kernel_st.to('cuda:0')
                self.synchronize()
            elif self.num_GPU == 2:
                self.kernel_1 = self.kernel_1.to('cuda:0', non_blocking=True)
                self.kernel_2 = self.kernel_2.to('cuda:1', non_blocking=True)
                self.kernel_sum = self.kernel_sum.to('cuda:0', non_blocking=True)
                self.kernel_st = self.kernel_st.to('cuda:0', non_blocking=True)
                self.synchronize()
            elif self.num_GPU == 4:
                self.kernel_1 = self.kernel_1.to('cuda:0', non_blocking=True)
                self.kernel_2 = self.kernel_2.to('cuda:1', non_blocking=True)
                self.kernel_3 = self.kernel_3.to('cuda:2', non_blocking=True)
                self.kernel_4 = self.kernel_4.to('cuda:3', non_blocking=True)
                self.kernel_sum = self.kernel_sum.to('cuda:0', non_blocking=True)
                self.kernel_st = self.kernel_st.to('cuda:0', non_blocking=True)
                self.synchronize()
    
    
    # Note that torch.chunk impose overflowed component to the first tensor.
    # And it split tensors by same sizes as possible.
    def send_history(self):
        if not self.qd:
            if self.num_GPU == 1:
                self.dDell_hist = self.dDell_hist.to(device='cuda:0', non_blocking=True)
            elif self.num_GPU == 2:
                self.dDell_hist_1 = self.dDell_hist_1.to(device='cuda:0', non_blocking=True)
                self.dDell_hist_2 = self.dDell_hist_2.to(device='cuda:1', non_blocking=True)
            elif self.num_GPU == 4:
                self.dDell_hist_1 = self.dDell_hist_1.to(device='cuda:0', non_blocking=True)
                self.dDell_hist_2 = self.dDell_hist_2.to(device='cuda:1', non_blocking=True)
                self.dDell_hist_3 = self.dDell_hist_3.to(device='cuda:2', non_blocking=True)
                self.dDell_hist_4 = self.dDell_hist_4.to(device='cuda:3', non_blocking=True)
    
    
    def sendback_history(self):
        if not self.qd:
            if self.num_GPU == 1:
                self.dDell_hist = self.dDell_hist.to(device='cpu', non_blocking=True)
            elif self.num_GPU == 2:
                self.dDell_hist_1 = self.dDell_hist_1.to(device='cpu', non_blocking=True)
                self.dDell_hist_2 = self.dDell_hist_2.to(device='cpu', non_blocking=True)
            elif self.num_GPU == 4:
                self.dDell_hist_1 = self.dDell_hist_1.to(device='cpu', non_blocking=True)
                self.dDell_hist_2 = self.dDell_hist_2.to(device='cpu', non_blocking=True)
                self.dDell_hist_3 = self.dDell_hist_3.to(device='cpu', non_blocking=True)
                self.dDell_hist_4 = self.dDell_hist_4.to(device='cpu', non_blocking=True)
                
            
    # Visalize kernel if needed.
    def plot(self):
        if self.plot_kernel:
            import matplotlib.pyplot as plt
            import cmocean as cmo
            os.makedirs('Check', exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(self.kernel.shape[1]):
                if i < 10:
                    ax.plot(self.thist, 
                            (self.kernel[:, i]-tr.min(self.kernel[:, i]))/(tr.max(self.kernel[:, i])-tr.min(self.kernel[:, i])),
                            lw=3, color=cmo.cm.thermal(i/10))
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
                             color=cmo.cm.thermal(i/40))
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
                             color=cmo.cm.thermal(i/462))
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
    
    
    def create_hist_on_cpu(self):
        if self.num_GPU == 0 or self.num_GPU == 1:
            self.dDell_hist = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
        elif self.num_GPU == 2:
            self.dDell_hist_1 = tr.zeros(len(self.thist), int(len(self.kn)//2)+1, dtype=self.dtype_tr, device='cpu')
            self.dDell_hist_2 = tr.zeros(len(self.thist), int(len(self.kn)//2), dtype=self.dtype_tr, device='cpu')
        elif self.num_GPU == 4:
            self.dDell_hist_1 = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
            self.dDell_hist_2 = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
            self.dDell_hist_3 = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
            self.dDell_hist_4 = tr.zeros(len(self.thist), int(len(self.kn)//4)-2, dtype=self.dtype_tr, device='cpu')
            
    
    def copy_all_hist(self):
        if self.num_GPU == 1 or self.num_GPU == 0:
            self.dDell_hist.copy_(self.kkeep_dDell.expand_as(self.dDell_hist))
        elif self.num_GPU == 2:
            self.kkeep_dDell_1, self.kkeep_dDell_2 = self.tr_chunk(self.kkeep_dDell, 2, dim=0)
            self.kkeep_dDell_2 = self.kkeep_dDell_2.to('cuda:1')
            self.synchronize()

            with self.cuda_stream(self.stream_1):
                self.dDell_hist_1.copy_(self.kkeep_dDell_1.expand_as(self.dDell_hist_1))
                self.event_1.record()
            with self.cuda_stream(self.stream_2):
                self.dDell_hist_2.copy_(self.kkeep_dDell_2.expand_as(self.dDell_hist_2))
                self.event_2.record()
            self.event_1.synchronize()
            self.event_2.synchronize()
        elif self.num_GPU == 4:
            self.kkeep_dDell_1, self.kkeep_dDell_2, self.kkeep_dDell_3, self.kkeep_dDell_4 = self.tr_chunk(self.kkeep_dDell, 4, dim=0)
            self.kkeep_dDell_2 = self.kkeep_dDell_2.to('cuda:1', non_blocking=True)
            self.kkeep_dDell_3 = self.kkeep_dDell_3.to('cuda:2', non_blocking=True)
            self.kkeep_dDell_4 = self.kkeep_dDell_4.to('cuda:3', non_blocking=True)
            self.synchronize()

            with self.cuda_stream(self.stream_1):
                self.dDell_hist_1.copy_(self.kkeep_dDell_1.expand_as(self.dDell_hist_1))
                self.event_1.record()
            with self.cuda_stream(self.stream_2):
                self.dDell_hist_2.copy_(self.kkeep_dDell_2.expand_as(self.dDell_hist_2))
                self.event_2.record()
            with self.cuda_stream(self.stream_3):
                self.dDell_hist_3.copy_(self.kkeep_dDell_3.expand_as(self.dDell_hist_3))
                self.event_3.record()
            with self.cuda_stream(self.stream_4):
                self.dDell_hist_4.copy_(self.kkeep_dDell_4.expand_as(self.dDell_hist_4))
                self.event_4.record()
            self.event_1.synchronize()
            self.event_2.synchronize()
            self.event_3.synchronize()
            self.event_4.synchronize()
        self.previously_dt_ge_Tw = False


    # Keep Fourier velocity for convolution.
    def store_dDell(self, dt, store):
        if store:
            self.inst = int(dt / self.min_step)
            if dt > self.Tw:
                self.kkeep_dDell = self.keep_dDell.clone()
                self.latest = self.num - 1
                self.previously_dt_ge_Tw = True

            elif dt <= self.Tw:
                if not self.previously_dt_ge_Tw:
                    pass
                else: # Repeating this process when dt > Tw is time-consuming.
                    self.copy_all_hist()
                        
                if self.num_GPU == 2:
                    self.keep_dDell_1, self.keep_dDell_2 = self.tr_chunk(self.keep_dDell, 2, dim=0)
                    self.keep_dDell_2 = self.keep_dDell_2.to('cuda:1')
                    self.synchronize()
                elif self.num_GPU == 4:
                    self.keep_dDell_1, self.keep_dDell_2, self.keep_dDell_3, self.keep_dDell_4 = self.tr_chunk(self.keep_dDell, 4, dim=0)
                    self.keep_dDell_2 = self.keep_dDell_2.to('cuda:1', non_blocking=True)
                    self.keep_dDell_3 = self.keep_dDell_3.to('cuda:2', non_blocking=True)
                    self.keep_dDell_4 = self.keep_dDell_4.to('cuda:3', non_blocking=True)
                    self.synchronize()
                else:
                    pass
                
                self.index = self.latest + self.inst - self.num + 1 # For efficient computation.
                if self.latest == self.num - 1:
                    if self.num_GPU == 1 or self.num_GPU == 0:
                        self.dDell_hist[0:self.inst, :].copy_(self.keep_dDell.expand_as(self.dDell_hist[0:self.inst, :]))
                    elif self.num_GPU == 2:
                        with self.cuda_stream(self.stream_1):
                            self.dDell_hist_1[0:self.inst, :].copy_(self.keep_dDell_1.expand_as(self.dDell_hist_1[0:self.inst, :]))
                            self.event_1.record()
                        with self.cuda_stream(self.stream_2):
                            self.dDell_hist_2[0:self.inst, :].copy_(self.keep_dDell_2.expand_as(self.dDell_hist_2[0:self.inst, :]))
                            self.event_2.record()
                        self.event_1.synchronize()
                        self.event_2.synchronize()
                    elif self.num_GPU == 4:
                        with self.cuda_stream(self.stream_1):
                            self.dDell_hist_1[0:self.inst, :].copy_(self.keep_dDell_1).expand_as(self.dDell_hist_1[0:self.inst, :])
                            self.event_1.record()
                        with self.cuda_stream(self.stream_2):
                            self.dDell_hist_2[0:self.inst, :].copy_(self.keep_dDell_2.expand_as(self.dDell_hist_2[0:self.inst, :]))
                            self.event_2.record()
                        with self.cuda_stream(self.stream_3):
                            self.dDell_hist_3[0:self.inst, :].copy_(self.keep_dDell_3.expand_as(self.dDell_hist_3[0:self.inst, :]))
                            self.event_3.record()
                        with self.cuda_stream(self.stream_4):
                            self.dDell_hist_4[0:self.inst, :].copy_(self.keep_dDell_4.expand_as(self.dDell_hist_4[0:self.inst, :]))
                            self.event_4.record()
                        self.event_1.synchronize()
                        self.event_2.synchronize()
                        self.event_3.synchronize()
                        self.event_4.synchronize()
                    self.latest = self.inst - 1

                elif self.index <= 0:
                    self.index_ = self.latest + 1
                    self.index__ = self.index_ + self.inst
                    if self.num_GPU == 1 or self.num_GPU == 0:
                        self.dDell_hist[self.index_:self.index__, :].copy_(
                            self.keep_dDell.expand_as(self.dDell_hist[self.index_:self.index__, :]))
                    elif self.num_GPU == 2:
                        with self.cuda_stream(self.stream_1):
                            self.dDell_hist_1[self.index_:self.index__, :].copy_(
                                self.keep_dDell_1.expand_as(self.dDell_hist_1[self.index_:self.index__, :]))
                            self.event_1.record()
                        with self.cuda_stream(self.stream_2):
                            self.dDell_hist_2[self.index_:self.index__, :].copy_(
                                self.keep_dDell_2.expand_as(self.dDell_hist_2[self.index_:self.index__, :]))
                            self.event_2.record()
                        self.event_1.synchronize()
                        self.event_2.synchronize()
                    elif self.num_GPU == 4:
                        with self.cuda_stream(self.stream_1):
                            self.dDell_hist_1[self.index_:self.index__, :].copy_(
                                self.keep_dDell_1.expand_as(self.dDell_hist_1[self.index_:self.index__, :]))
                            self.event_1.record()
                        with self.cuda_stream(self.stream_2):
                            self.dDell_hist_2[self.index_:self.index__, :].copy_(
                                self.keep_dDell_2.expand_as(self.dDell_hist_2[self.index_:self.index__, :]))
                            self.event_2.record()
                        with self.cuda_stream(self.stream_3):
                            self.dDell_hist_3[self.index_:self.index__, :].copy_(
                                self.keep_dDell_3.expand_as(self.dDell_hist_3[self.index_:self.index__, :]))
                            self.event_3.record()
                        with self.cuda_stream(self.stream_4):
                            self.dDell_hist_4[self.index_:self.index__, :].copy_(
                                self.keep_dDell_4.expand_as(self.dDell_hist_4[self.index_:self.index__, :]))
                            self.event_4.record()
                        self.event_1.synchronize()
                        self.event_2.synchronize()
                        self.event_3.synchronize()
                        self.event_4.synchronize()
                    self.latest += self.inst

                else:
                    self.index_ = self.latest + 1
                    if self.num_GPU == 1 or self.num_GPU == 0:
                        self.dDell_hist[self.index_:self.num, :].copy_(
                            self.keep_dDell.expand_as(self.dDell_hist[self.index_:self.num, :]))
                        self.dDell_hist[0:self.index, :].copy_(
                            self.keep_dDell.expand_as(self.dDell_hist[0:self.index, :]))
                    elif self.num_GPU == 2:
                        with self.cuda_stream(self.stream_1):
                            self.dDell_hist_1[self.index_:self.num, :].copy_(
                                self.keep_dDell_1.expand_as(self.dDell_hist_1[self.index_:self.num, :]))
                            self.dDell_hist_1[0:self.index, :].copy_(
                                self.keep_dDell_1.expand_as(self.dDell_hist_1[0:self.index, :]))
                            self.event_1.record()
                        with self.cuda_stream(self.stream_2):
                            self.dDell_hist_2[self.index_:self.num, :].copy_(
                                self.keep_dDell_2.expand_as(self.dDell_hist_2[self.index_:self.num, :]))
                            self.dDell_hist_2[0:self.index, :].copy_(
                                self.keep_dDell_2.expand_as(self.dDell_hist_2[0:self.index, :]))
                            self.event_2.record()
                        self.event_1.synchronize()
                        self.event_2.synchronize()
                    elif self.num_GPU == 4:
                        with self.cuda_stream(self.stream_1):
                            self.dDell_hist_1[self.index_:self.num, :].copy_(
                                self.keep_dDell_1.expand_as(self.dDell_hist_1[self.index_:self.num, :]))
                            self.dDell_hist_1[0:self.index, :].copy_(
                                self.keep_dDell_1.expand_as(self.dDell_hist_1[0:self.index, :]))
                            self.event_1.record()
                        with self.cuda_stream(self.stream_2):
                            self.dDell_hist_2[self.index_:self.num, :].copy_(
                                self.keep_dDell_2.expand_as(self.dDell_hist_2[self.index_:self.num, :]))
                            self.dDell_hist_2[0:self.index, :].copy_(
                                self.keep_dDell_2.expand_as(self.dDell_hist_2[0:self.index, :]))
                            self.event_2.record()
                        with self.cuda_stream(self.stream_3):
                            self.dDell_hist_3[self.index_:self.num, :].copy_(
                                self.keep_dDell_3.expand_as(self.dDell_hist_3[self.index_:self.num, :]))
                            self.dDell_hist_3[0:self.index, :].copy_(
                                self.keep_dDell_3.expand_as(self.dDell_hist_3[0:self.index, :]))
                            self.event_3.record()
                        with self.cuda_stream(self.stream_4):
                            self.dDell_hist_4[self.index_:self.num, :].copy_(
                                self.keep_dDell_4.expand_as(self.dDell_hist_4[self.index_:self.num, :]))
                            self.dDell_hist_4[0:self.index, :].copy_(
                                self.keep_dDell_4.expand_as(self.dDell_hist_4[0:self.index, :]))
                            self.event_4.record()
                        self.event_1.synchronize()
                        self.event_2.synchronize()
                        self.event_3.synchronize()
                        self.event_4.synchronize()
                    self.latest += self.inst - self.num
        else:
            self.inst = int(dt / self.min_step)
            if dt > self.Tw:
                self.previously_dt_ge_Tw = True
                self.flag_copy = 0
                self.init_copy = 0
                self.end_copy = self.full_size
            elif dt <= self.Tw:
                if not self.previously_dt_ge_Tw:
                    pass
                else:
                    self.previously_dt_ge_Tw = False
                    self.flag_copy = 0
                    self.init_copy = 0
                    self.end_copy = self.full_size
                    return
                self.index = self.latest + self.inst - self.num + 1
                if self.latest == self.num - 1:
                    self.flag_copy = 0
                    self.init_copy = 0
                    self.end_copy = self.inst
                elif self.index <= 0:
                    self.index_ = self.latest + 1
                    self.index__ = self.index_ + self.inst
                    self.flag_copy = 0
                    self.init_copy = self.index_
                    self.end_copy = self.index__
                else:
                    self.index_ = self.latest + 1
                    self.flag_copy = 1
                    self.init_copy = self.index_
                    self.end_copy = self.index
                    

    # Static convolution.
    def conv_st(self, Dell):
        return self.kernel_st * Dell


    # Instant convolution (should be done at every time step).
    def conv_inst(self, d_Dell, dt0):
        if dt0 >= self.Tw:
            self.inst = self.num
            return self.kernel_sum * d_Dell
        else:
            self.inst = int(dt0 / self.min_step)
            if self.num_GPU == 1 or self.num_GPU == 0:
                return self.kernel[self.num-self.inst:self.num, :].sum(dim=0) * d_Dell
            elif self.num_GPU == 2:
                with self.cuda_stream(self.stream_1):
                    self.K_inst_1 = self.kernel_1[self.num-self.inst:self.num, :].sum(dim=0)
                    self.event_1.record()
                with self.cuda_stream(self.stream_2):
                    self.K_inst_2 = self.kernel_2[self.num-self.inst:self.num, :].sum(dim=0)
                    self.K_inst_2 = self.K_inst_2.to('cuda:0', non_blocking=True)
                    self.event_2.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
                self.K_inst = self.tr_cat((self.K_inst_1, self.K_inst_2), dim=0) * d_Dell
                return self.K_inst
            elif self.num_GPU == 4:
                with self.cuda_stream(self.stream_1):
                    self.K_inst_1 = self.kernel_1[self.num-self.inst:self.num, :].sum(dim=0)
                    self.event_1.record()
                with self.cuda_stream(self.stream_2):
                    self.K_inst_2 = self.kernel_2[self.num-self.inst:self.num, :].sum(dim=0)
                    self.K_inst_2 = self.K_inst_2.to('cuda:0', non_blocking=True)
                    self.event_2.record()
                with self.cuda_stream(self.stream_3):
                    self.K_inst_3 = self.kernel_3[self.num-self.inst:self.num, :].sum(dim=0)
                    self.K_inst_3 = self.K_inst_3.to('cuda:0', non_blocking=True)
                    self.event_3.record()
                with self.cuda_stream(self.stream_4):
                    self.K_inst_4 = self.kernel_4[self.num-self.inst:self.num, :].sum(dim=0)
                    self.K_inst_4 = self.K_inst_4.to('cuda:0', non_blocking=True)
                    self.event_4.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
                self.event_3.synchronize()
                self.event_4.synchronize()
                self.K_inst = self.tr_cat((self.K_inst_1, self.K_inst_2,
                                           self.K_inst_3, self.K_inst_4), dim=0) * d_Dell
                return self.K_inst

    # Dynamic kernel convolution (availabe at second step).
    def conv(self, dt):
        self.inst = int(dt / self.min_step)
        if dt > self.Tw:
            self.K.zero_()
            return self.K
        else:
            self.index = self.latest + self.inst - self.num + 1
            if self.index >= 0:
                if self.num_GPU == 1 or self.num_GPU == 0:
                    self.K = (self.kernel[0:self.num-self.inst, :] * self.dDell_hist[self.index:self.latest+1, :]).sum(dim=0)
                elif self.num_GPU == 2:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (self.kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1[self.index:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (self.kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    self.event_1.synchronize()
                    self.event_2.synchronize()
                    self.K = self.tr_cat((self.K1, self.K2), dim=0)
                    self.synchronize()
                elif self.num_GPU == 4:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (self.kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1[self.index:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (self.kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    with self.cuda_stream(self.stream_3):
                        self.K3 = (self.kernel_3[0:self.num-self.inst, :] * self.dDell_hist_3[self.index:self.latest+1, :]).sum(dim=0)
                        self.K3 = self.K3.to('cuda:0', non_blocking=True)
                        self.event_3.record()
                    with self.cuda_stream(self.stream_4):
                        self.K4 = (self.kernel_4[0:self.num-self.inst, :] * self.dDell_hist_4[self.index:self.latest+1, :]).sum(dim=0)
                        self.K4 = self.K4.to('cuda:0', non_blocking=True)
                        self.event_4.record()
                    self.event_1.synchronize()
                    self.event_2.synchronize()
                    self.event_3.synchronize()
                    self.event_4.synchronize()
                    self.K = self.tr_cat((self.K1, self.K2,
                                          self.K3, self.K4), dim=0)
                    self.synchronize()
                return self.K
            else:
                if self.num_GPU == 1 or self.num_GPU == 0:
                    self.K = (self.kernel[0:-self.index, :] * self.dDell_hist[self.index+self.num:self.num, :]).sum(dim=0)
                    self.K += (self.kernel[-self.index:self.num-self.inst, :] * self.dDell_hist[0:self.latest+1, :]).sum(dim=0)
                elif self.num_GPU == 2:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (self.kernel_1[0:-self.index, :] * self.dDell_hist_1[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1 += (self.kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1[0:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (self.kernel_2[0:-self.index, :] * self.dDell_hist_2[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2 += (self.kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2[0:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    self.event_1.synchronize()
                    self.event_2.synchronize()
                    self.K = self.tr_cat((self.K1, self.K2), dim=0)
                    self.synchronize()
                elif self.num_GPU == 4:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (self.kernel_1[0:-self.index, :] * self.dDell_hist_1[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1 += (self.kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1[0:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (self.kernel_2[0:-self.index, :] * self.dDell_hist_2[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2 += (self.kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2[0:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    with self.cuda_stream(self.stream_3):
                        self.K3 = (self.kernel_3[0:-self.index, :] * self.dDell_hist_3[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K3 += (self.kernel_3[-self.index:self.num-self.inst, :] * self.dDell_hist_3[0:self.latest+1, :]).sum(dim=0)
                        self.K3 = self.K3.to('cuda:0', non_blocking=True)
                        self.event_3.record()
                    with self.cuda_stream(self.stream_4):
                        self.K4 = (self.kernel_4[0:-self.index, :] * self.dDell_hist_4[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K4 += (self.kernel_4[-self.index:self.num-self.inst, :] * self.dDell_hist_4[0:self.latest+1, :]).sum(dim=0)
                        self.K4 = self.K4.to('cuda:0', non_blocking=True)
                        self.event_4.record()
                    self.event_1.synchronize()
                    self.event_2.synchronize()
                    self.event_3.synchronize()
                    self.event_4.synchronize()
                    self.K = self.tr_cat((self.K1, self.K2,
                                          self.K3, self.K4), dim=0)
                    self.synchronize()
                return self.K


    # Execute convolutions.
    def exe_conv(self, Dell, d_Dell, dt, first):
        if not self.qd: # Fully-dynamic.
            if not self.mirror:
                if first:
                    tmp = self.conv_st(Dell) + self.conv_inst(d_Dell, dt) + self.conv(dt)
                    return self.tr_irfft(tmp, self.nconv)[:self.ncell]
                else:
                    tmp = self.conv_st(Dell) + self.K + self.conv_inst(d_Dell, dt)
                    return self.tr_irfft(tmp, self.nconv)[:self.ncell]
            else:
                if first:
                    return self.tr_idct( self.conv_st(Dell) + self.conv(dt) + self.conv_inst(d_Dell, dt))[:self.ncell]
                else:
                    return self.tr_idct( self.conv_st(Dell) + self.K + self.conv_inst(d_Dell, dt))[:self.ncell]
        else: # Quasi-dynamic.
            if not self.mirror:
                tmp = self.conv_st(Dell)
                return self.tr_irfft( tmp, n=self.nconv )[:self.ncell]
            else:
                tmp = self.conv_st(Dell)
                return self.tr_idct( tmp )[:self.ncell]


    # FFT (back slip).
    def vfft(self, VB):
        if not self.mirror:
            return self.tr_rfft(VB, n=self.nconv)
        else:
            return self.tr_dct(self.tr_cat((VB, self.pad), dim=0))

    # FFT (slip dificit).
    def delfft(self, dB):
        if not self.mirror:
            return self.tr_rfft(dB, n=self.nconv)
        else:
            return self.tr_dct(self.tr_cat((dB, self.pad), dim=0))
        
    # Update Dell (first step).
    def upt_Dell_first(self, ini, dt):
        ini.Dell += ini.d_Dell * dt
        return ini
    
    # Update Dell (second step).
    def upt_Dell_second(self, ini, dt):
        self.keep_dDell = (ini.d_Dell_prv + ini.d_Dell) * 0.5
        ini.Dell = ini.Dell_prv + self.keep_dDell * dt
        return ini