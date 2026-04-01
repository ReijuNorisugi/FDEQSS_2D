# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20250928.
"""

# This is the module for computing stress transfer functional.
# This is specialilzed for communication with multiple GPUs when they are directly connected.

import numpy as np
import torch as tr
import torch_dct as dct
import sys

class Convolution():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
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
        
        self.split = True
        if self.mirror:
            self.split = False

        self.Cs = Medium['Cs']   # S-wave velocity.
        self.Cp = Medium['Cp']   # P-wave velocity.
        if self.mode == 'III' or self.mode == 'IV':
            self.C = self.Cs
        elif self.mode == 'II':
            self.C = self.Cp
        self.alpha = self.Cp / self.Cs          # Coefficient.
        self.inv_alpha = (self.Cs / self.Cp)**2 # Coefficient.
        self.alpha2 = self.alpha**2             # Alpha square.

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
        self.len_thist = len(self.thist)
        self.num_splits = 32                # Number of aplit for parallel kernel computation.
        
        if not self.mirror:
            if self.split:
                self.dtype_np = np.float64
                self.dtype_tr = tr.float64
            else:
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
        if not self.mirror:
            self.K = tr.zeros(len(self.kn), dtype=tr.complex128, device=self.device)
        else:
            self.K = tr.zeros(len(self.kn), dtype=tr.float64, device=self.device)
        
        if not self.qd:
            if self.split:
                if self.num_GPU == 0:
                    self.dDell_hist_r = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
                    self.dDell_hist_i = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
                elif self.num_GPU == 1:
                    self.dDell_hist_r = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cuda:0')
                    self.dDell_hist_i = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cuda:0')
                elif self.num_GPU == 2:
                    self.dDell_hist_1r = tr.zeros(len(self.thist), int(len(self.kn)//2)+1, dtype=self.dtype_tr, device='cuda:0')
                    self.dDell_hist_2r = tr.zeros(len(self.thist), int(len(self.kn)//2), dtype=self.dtype_tr, device='cuda:1')
                    self.dDell_hist_1i = tr.zeros(len(self.thist), int(len(self.kn)//2)+1, dtype=self.dtype_tr, device='cuda:0')
                    self.dDell_hist_2i = tr.zeros(len(self.thist), int(len(self.kn)//2), dtype=self.dtype_tr, device='cuda:1')
                elif self.num_GPU == 4:
                    self.dDell_hist_1r = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:0')
                    self.dDell_hist_2r = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:1')
                    self.dDell_hist_3r = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:2')
                    self.dDell_hist_4r = tr.zeros(len(self.thist), int(len(self.kn)//4)-2, dtype=self.dtype_tr, device='cuda:3')
                    self.dDell_hist_1i = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:0')
                    self.dDell_hist_2i = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:1')
                    self.dDell_hist_3i = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cuda:2')
                    self.dDell_hist_4i = tr.zeros(len(self.thist), int(len(self.kn)//4)-2, dtype=self.dtype_tr, device='cuda:3')
            else:
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
        self.tr_dct = dct.dct
        self.tr_idct = dct.idct
        self.tr_cat = tr.cat
        self.tr_chunk = tr.chunk
        self.tr_complex = tr.complex
        
        self.cuda_stream = tr.cuda.stream
        self.synchronize = tr.cuda.synchronize


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
    
    def reset_history(self):
        if not self.qd:
            if self.split:
                if self.num_GPU == 0 or self.num_GPU == 1:
                    self.dDell_hist_r = None
                    self.dDell_hist_i = None
                elif self.num_GPU == 2:
                    self.dDell_hist_1r = None
                    self.dDell_hist_2r = None
                    self.dDell_hist_1i = None
                    self.dDell_hist_2i = None
                elif self.num_GPU ==4 :
                    self.dDell_hist_1r = None
                    self.dDell_hist_2r = None
                    self.dDell_hist_3r = None
                    self.dDell_hist_4r = None
                    self.dDell_hist_1i = None
                    self.dDell_hist_2i = None
                    self.dDell_hist_3i = None
                    self.dDell_hist_4i = None
            else:
                if self.num_GPU == 0 or self.num_GPU == 1:
                    self.dDell_hist = None
                elif self.num_GPU == 2:
                    self.dDell_hist_1 = None
                    self.dDell_hist_2 = None
                elif self.num_GPU ==4 :
                    self.dDell_hist_1 = None
                    self.dDell_hist_2 = None
                    self.dDell_hist_3 = None
                    self.dDell_hist_4 = None
    
    # Note that torch.chunk impose overflowed component to the first tensor.
    # And it split tensors by same sizes as possible.
    def send_history(self):
        if not self.qd:
            if self.split:
                if self.num_GPU == 1:
                    self.dDell_hist_r = self.dDell_hist_r.to(device='cuda:0', non_blocking=True)
                    self.dDell_hist_i = self.dDell_hist_i.to(device='cuda:0', non_blocking=True)
                elif self.num_GPU == 2:
                    self.dDell_hist_1r = self.dDell_hist_1r.to(device='cuda:0', non_blocking=True)
                    self.dDell_hist_2r = self.dDell_hist_2r.to(device='cuda:1', non_blocking=True)
                    self.dDell_hist_1i = self.dDell_hist_1i.to(device='cuda:0', non_blocking=True)
                    self.dDell_hist_2i = self.dDell_hist_2i.to(device='cuda:1', non_blocking=True)
                elif self.num_GPU == 4:
                    self.dDell_hist_1r = self.dDell_hist_1r.to(device='cuda:0', non_blocking=True)
                    self.dDell_hist_2r = self.dDell_hist_2r.to(device='cuda:1', non_blocking=True)
                    self.dDell_hist_3r = self.dDell_hist_3r.to(device='cuda:2', non_blocking=True)
                    self.dDell_hist_4r = self.dDell_hist_4r.to(device='cuda:3', non_blocking=True)
                    self.dDell_hist_1i = self.dDell_hist_1i.to(device='cuda:0', non_blocking=True)
                    self.dDell_hist_2i = self.dDell_hist_2i.to(device='cuda:1', non_blocking=True)
                    self.dDell_hist_3i = self.dDell_hist_3i.to(device='cuda:2', non_blocking=True)
                    self.dDell_hist_4i = self.dDell_hist_4i.to(device='cuda:3', non_blocking=True)
            else:
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
            if self.split:
                if self.num_GPU == 1:
                    self.dDell_hist_r = self.dDell_hist_r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_i = self.dDell_hist_i.to(device='cpu', non_blocking=True)
                elif self.num_GPU == 2:
                    self.dDell_hist_1r = self.dDell_hist_1r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_2r = self.dDell_hist_2r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_1i = self.dDell_hist_1i.to(device='cpu', non_blocking=True)
                    self.dDell_hist_2i = self.dDell_hist_2i.to(device='cpu', non_blocking=True)
                elif self.num_GPU == 4:
                    self.dDell_hist_1r = self.dDell_hist_1r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_2r = self.dDell_hist_2r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_3r = self.dDell_hist_3r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_4r = self.dDell_hist_4r.to(device='cpu', non_blocking=True)
                    self.dDell_hist_1i = self.dDell_hist_1i.to(device='cpu', non_blocking=True)
                    self.dDell_hist_2i = self.dDell_hist_2i.to(device='cpu', non_blocking=True)
                    self.dDell_hist_3i = self.dDell_hist_3i.to(device='cpu', non_blocking=True)
                    self.dDell_hist_4i = self.dDell_hist_4i.to(device='cpu', non_blocking=True)
            else:
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
    
    
    def create_hist_on_cpu(self):
        if self.split:
            if self.num_GPU == 0 or self.num_GPU == 1:
                self.dDell_hist_r = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_i = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
            elif self.num_GPU == 2:
                self.dDell_hist_1r = tr.zeros(len(self.thist), int(len(self.kn)//2)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_2r = tr.zeros(len(self.thist), int(len(self.kn)//2), dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_1i = tr.zeros(len(self.thist), int(len(self.kn)//2)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_2i = tr.zeros(len(self.thist), int(len(self.kn)//2), dtype=self.dtype_tr, device='cpu')
            elif self.num_GPU == 4:
                self.dDell_hist_1r = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_2r = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_3r = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_4r = tr.zeros(len(self.thist), int(len(self.kn)//4)-2, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_1i = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_2i = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_3i = tr.zeros(len(self.thist), int(len(self.kn)//4)+1, dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_4i = tr.zeros(len(self.thist), int(len(self.kn)//4)-2, dtype=self.dtype_tr, device='cpu')
        else:
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
        if self.qd:
            return
        else:
            pass
        
        if self.split:
            if self.num_GPU == 1 or self.num_GPU == 0:
                self.dDell_hist_r.copy_((self.kkeep_dDell.real).expand_as(self.dDell_hist_r))
                self.dDell_hist_i.copy_((self.kkeep_dDell.imag).expand_as(self.dDell_hist_i))
            elif self.num_GPU == 2:
                self.kkeep_dDell_1, self.kkeep_dDell_2 = self.tr_chunk(self.kkeep_dDell, 2, dim=0)
                self.kkeep_dDell_2 = self.kkeep_dDell_2.to('cuda:1')
                self.synchronize()

                with self.cuda_stream(self.stream_1):
                    self.dDell_hist_1r.copy_((self.kkeep_dDell_1.real).expand_as(self.dDell_hist_1r))
                    self.dDell_hist_1i.copy_((self.kkeep_dDell_1.imag).expand_as(self.dDell_hist_1i))
                    self.event_1.record()
                with self.cuda_stream(self.stream_2):
                    self.dDell_hist_2r.copy_((self.kkeep_dDell_2.real).expand_as(self.dDell_hist_2r))
                    self.dDell_hist_2i.copy_((self.kkeep_dDell_2.imag).expand_as(self.dDell_hist_2i))
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
                    self.dDell_hist_1r.copy_((self.kkeep_dDell_1.real).expand_as(self.dDell_hist_1r))
                    self.dDell_hist_1i.copy_((self.kkeep_dDell_1.imag).expand_as(self.dDell_hist_1i))
                    self.event_1.record()
                with self.cuda_stream(self.stream_2):
                    self.dDell_hist_2r.copy_((self.kkeep_dDell_2.real).expand_as(self.dDell_hist_2r))
                    self.dDell_hist_2i.copy_((self.kkeep_dDell_2.imag).expand_as(self.dDell_hist_2i))
                    self.event_2.record()
                with self.cuda_stream(self.stream_3):
                    self.dDell_hist_3r.copy_((self.kkeep_dDell_3.real).expand_as(self.dDell_hist_3r))
                    self.dDell_hist_3i.copy_((self.kkeep_dDell_3.imag).expand_as(self.dDell_hist_3i))
                    self.event_3.record()
                with self.cuda_stream(self.stream_4):
                    self.dDell_hist_4r.copy_((self.kkeep_dDell_4.real).expand_as(self.dDell_hist_4r))
                    self.dDell_hist_4i.copy_((self.kkeep_dDell_4.imag).expand_as(self.dDell_hist_4i))
                    self.event_4.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
                self.event_3.synchronize()
                self.event_4.synchronize()
        else:
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
    
    
    def align_hist(self, tar, retain):
        if self.qd:
            return
        else:
            pass
        
        if self.previously_dt_ge_Tw:
            self.latest = tar.latest
            return
        else:
            if not retain:
                if self.flag_copy == 1:
                    self.init = self.init_copy
                    self.end = self.len_thist
                    self.init_ = 0
                    self.end_ = self.end_copy
                else:
                    self.init = self.init_copy
                    self.end = self.end_copy
            else:
                self.flag_copy = tar.flag_copy
                if self.flag_copy == 1:
                    self.init = tar.init_copy
                    self.end = self.len_thist
                    self.init_ = 0
                    self.end_ = tar.end_copy
                else:
                    self.init = tar.init_copy
                    self.end = tar.end_copy
            
            if self.num_GPU == 0 or self.num_GPU == 1:
                if self.split:
                    self.dDell_hist_r[self.init:self.end, :].copy_(tar.dDell_hist_r[self.init:self.end, :])
                    self.dDell_hist_i[self.init:self.end, :].copy_(tar.dDell_hist_i[self.init:self.end, :])
                else:
                    self.dDell_hist[self.init:self.end, :].copy_(tar.dDell_hist[self.init:self.end, :])
                if self.flag_copy == 1:
                    if self.split:
                        self.dDell_hist_r[self.init_:self.end_, :].copy_(tar.dDell_hist_r[self.init_:self.end_, :])
                        self.dDell_hist_i[self.init_:self.end_, :].copy_(tar.dDell_hist_i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist[self.init_:self.end_, :].copy_(tar.dDell_hist[self.init_:self.end_, :])
            elif self.num_GPU == 2:
                with self.cuda_stream(self.stream_1):
                    if self.split:
                        self.dDell_hist_1r[self.init:self.end, :].copy_(tar.dDell_hist_1r[self.init:self.end, :])
                        self.dDell_hist_1i[self.init:self.end, :].copy_(tar.dDell_hist_1i[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_1r[self.init_:self.end_, :].copy_(tar.dDell_hist_1r[self.init_:self.end_, :])
                            self.dDell_hist_1i[self.init_:self.end_, :].copy_(tar.dDell_hist_1i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist_1[self.init:self.end, :].copy_(tar.dDell_hist_1[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_1[self.init_:self.end_, :].copy_(tar.dDell_hist_1[self.init_:self.end_, :])
                    self.event_1.record()
                with self.cuda_stream(self.stream_2):
                    if self.split:
                        self.dDell_hist_2r[self.init:self.end, :].copy_(tar.dDell_hist_2r[self.init:self.end, :])
                        self.dDell_hist_2i[self.init:self.end, :].copy_(tar.dDell_hist_2i[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_2r[self.init_:self.end_, :].copy_(tar.dDell_hist_2r[self.init_:self.end_, :])
                            self.dDell_hist_2i[self.init_:self.end_, :].copy_(tar.dDell_hist_2i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist_2[self.init:self.end, :].copy_(tar.dDell_hist_2[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_2[self.init_:self.end_, :].copy_(tar.dDell_hist_2[self.init_:self.end_, :])
                    self.event_2.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
            elif self.num_GPU == 4:
                with self.cuda_stream(self.stream_1):
                    if self.split:
                        self.dDell_hist_1r[self.init:self.end, :].copy_(tar.dDell_hist_1r[self.init:self.end, :])
                        self.dDell_hist_1i[self.init:self.end, :].copy_(tar.dDell_hist_1i[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_1r[self.init_:self.end_, :].copy_(tar.dDell_hist_1r[self.init_:self.end_, :])
                            self.dDell_hist_1i[self.init_:self.end_, :].copy_(tar.dDell_hist_1i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist_1[self.init:self.end, :].copy_(tar.dDell_hist_1[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_1[self.init_:self.end_, :].copy_(tar.dDell_hist_1[self.init_:self.end_, :])
                    self.event_1.record()
                with self.cuda_stream(self.stream_2):
                    if self.split:
                        self.dDell_hist_2r[self.init:self.end, :].copy_(tar.dDell_hist_2r[self.init:self.end, :])
                        self.dDell_hist_2i[self.init:self.end, :].copy_(tar.dDell_hist_2i[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_2r[self.init_:self.end_, :].copy_(tar.dDell_hist_2r[self.init_:self.end_, :])
                            self.dDell_hist_2i[self.init_:self.end_, :].copy_(tar.dDell_hist_2i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist_2[self.init:self.end, :].copy_(tar.dDell_hist_2[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_2[self.init_:self.end_, :].copy_(tar.dDell_hist_2[self.init_:self.end_, :])
                    self.event_2.record()
                with self.cuda_stream(self.stream_3):
                    if self.split:
                        self.dDell_hist_3r[self.init:self.end, :].copy_(tar.dDell_hist_3r[self.init:self.end, :])
                        self.dDell_hist_3i[self.init:self.end, :].copy_(tar.dDell_hist_3i[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_3r[self.init_:self.end_, :].copy_(tar.dDell_hist_3r[self.init_:self.end_, :])
                            self.dDell_hist_3i[self.init_:self.end_, :].copy_(tar.dDell_hist_3i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist_3[self.init:self.end, :].copy_(tar.dDell_hist_3[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_3[self.init_:self.end_, :].copy_(tar.dDell_hist_3[self.init_:self.end_, :])
                    self.event_3.record()
                with self.cuda_stream(self.stream_4):
                    if self.split:
                        self.dDell_hist_4r[self.init:self.end, :].copy_(tar.dDell_hist_4r[self.init:self.end, :])
                        self.dDell_hist_4i[self.init:self.end, :].copy_(tar.dDell_hist_4i[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_4r[self.init_:self.end_, :].copy_(tar.dDell_hist_4r[self.init_:self.end_, :])
                            self.dDell_hist_4i[self.init_:self.end_, :].copy_(tar.dDell_hist_4i[self.init_:self.end_, :])
                    else:
                        self.dDell_hist_4[self.init:self.end, :].copy_(tar.dDell_hist_4[self.init:self.end, :])
                        if self.flag_copy == 1:
                            self.dDell_hist_4[self.init_:self.end_, :].copy_(tar.dDell_hist_4[self.init_:self.end_, :])
                    self.event_4.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
                self.event_3.synchronize()
                self.event_4.synchronize()
            self.latest = tar.latest


    # Keep Fourier velocity for convolution.
    def store_dDell(self, dt, store):
        if self.qd:
            return
        else:
            pass
        
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
                    if self.split:
                        if self.num_GPU == 1 or self.num_GPU == 0:
                            self.dDell_hist_r[0:self.inst, :].copy_((self.keep_dDell.real).expand_as(self.dDell_hist_r[0:self.inst, :]))
                            self.dDell_hist_i[0:self.inst, :].copy_((self.keep_dDell.imag).expand_as(self.dDell_hist_i[0:self.inst, :]))
                        elif self.num_GPU == 2:
                            with self.cuda_stream(self.stream_1):
                                self.dDell_hist_1r[0:self.inst, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[0:self.inst, :]))
                                self.dDell_hist_1i[0:self.inst, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[0:self.inst, :]))
                                self.event_1.record()
                            with self.cuda_stream(self.stream_2):
                                self.dDell_hist_2r[0:self.inst, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[0:self.inst, :]))
                                self.dDell_hist_2i[0:self.inst, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[0:self.inst, :]))
                                self.event_2.record()
                            self.event_1.synchronize()
                            self.event_2.synchronize()
                        elif self.num_GPU == 4:
                            with self.cuda_stream(self.stream_1):
                                self.dDell_hist_1r[0:self.inst, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[0:self.inst, :]))
                                self.dDell_hist_1i[0:self.inst, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[0:self.inst, :]))
                                self.event_1.record()
                            with self.cuda_stream(self.stream_2):
                                self.dDell_hist_2r[0:self.inst, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[0:self.inst, :]))
                                self.dDell_hist_2i[0:self.inst, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[0:self.inst, :]))
                                self.event_2.record()
                            with self.cuda_stream(self.stream_3):
                                self.dDell_hist_3r[0:self.inst, :].copy_(
                                    (self.keep_dDell_3.real).expand_as(self.dDell_hist_3r[0:self.inst, :]))
                                self.dDell_hist_3i[0:self.inst, :].copy_(
                                    (self.keep_dDell_3.imag).expand_as(self.dDell_hist_3i[0:self.inst, :]))
                                self.event_3.record()
                            with self.cuda_stream(self.stream_4):
                                self.dDell_hist_4r[0:self.inst, :].copy_(
                                    (self.keep_dDell_4.real).expand_as(self.dDell_hist_4r[0:self.inst, :]))
                                self.dDell_hist_4i[0:self.inst, :].copy_(
                                    (self.keep_dDell_4.imag).expand_as(self.dDell_hist_4i[0:self.inst, :]))
                                self.event_4.record()
                            self.event_1.synchronize()
                            self.event_2.synchronize()
                            self.event_3.synchronize()
                            self.event_4.synchronize()
                    else:
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
                    
                    if self.split:
                        if self.num_GPU == 1 or self.num_GPU == 0:
                            self.dDell_hist_r[self.index_:self.index__, :].copy_(
                                (self.keep_dDell.real).expand_as(self.dDell_hist_r[self.index_:self.index__, :]))
                            self.dDell_hist_i[self.index_:self.index__, :].copy_(
                                (self.keep_dDell.imag).expand_as(self.dDell_hist_i[self.index_:self.index__, :]))
                        elif self.num_GPU == 2:
                            with self.cuda_stream(self.stream_1):
                                self.dDell_hist_1r[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[self.index_:self.index__, :]))
                                self.dDell_hist_1i[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[self.index_:self.index__, :]))
                                self.event_1.record()
                            with self.cuda_stream(self.stream_2):
                                self.dDell_hist_2r[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[self.index_:self.index__, :]))
                                self.dDell_hist_2i[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[self.index_:self.index__, :]))
                                self.event_2.record()
                            self.event_1.synchronize()
                            self.event_2.synchronize()
                        elif self.num_GPU == 4:
                            with self.cuda_stream(self.stream_1):
                                self.dDell_hist_1r[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[self.index_:self.index__, :]))
                                self.dDell_hist_1i[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[self.index_:self.index__, :]))
                                self.event_1.record()
                            with self.cuda_stream(self.stream_2):
                                self.dDell_hist_2r[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[self.index_:self.index__, :]))
                                self.dDell_hist_2i[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[self.index_:self.index__, :]))
                                self.event_2.record()
                            with self.cuda_stream(self.stream_3):
                                self.dDell_hist_3r[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_3.real).expand_as(self.dDell_hist_3r[self.index_:self.index__, :]))
                                self.dDell_hist_3i[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_3.imag).expand_as(self.dDell_hist_3i[self.index_:self.index__, :]))
                                self.event_3.record()
                            with self.cuda_stream(self.stream_4):
                                self.dDell_hist_4r[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_4.real).expand_as(self.dDell_hist_4r[self.index_:self.index__, :]))
                                self.dDell_hist_4i[self.index_:self.index__, :].copy_(
                                    (self.keep_dDell_4.imag).expand_as(self.dDell_hist_4i[self.index_:self.index__, :]))
                                self.event_4.record()
                            self.event_1.synchronize()
                            self.event_2.synchronize()
                            self.event_3.synchronize()
                            self.event_4.synchronize()
                    else:
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
                    
                    if self.split:
                        if self.num_GPU == 1 or self.num_GPU == 0:
                            self.dDell_hist_r[self.index_:self.num, :].copy_(
                                (self.keep_dDell.real).expand_as(self.dDell_hist_r[self.index_:self.num, :]))
                            self.dDell_hist_r[0:self.index, :].copy_(
                                (self.keep_dDell.real).expand_as(self.dDell_hist_r[0:self.index, :]))
                            self.dDell_hist_i[self.index_:self.num, :].copy_(
                                (self.keep_dDell.imag).expand_as(self.dDell_hist_i[self.index_:self.num, :]))
                            self.dDell_hist_i[0:self.index, :].copy_(
                                (self.keep_dDell.imag).expand_as(self.dDell_hist_i[0:self.index, :]))
                        elif self.num_GPU == 2:
                            with self.cuda_stream(self.stream_1):
                                self.dDell_hist_1r[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[self.index_:self.num, :]))
                                self.dDell_hist_1r[0:self.index, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[0:self.index, :]))
                                self.dDell_hist_1i[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[self.index_:self.num, :]))
                                self.dDell_hist_1i[0:self.index, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[0:self.index, :]))
                                self.event_1.record()
                            with self.cuda_stream(self.stream_2):
                                self.dDell_hist_2r[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[self.index_:self.num, :]))
                                self.dDell_hist_2r[0:self.index, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[0:self.index, :]))
                                self.dDell_hist_2i[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[self.index_:self.num, :]))
                                self.dDell_hist_2i[0:self.index, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[0:self.index, :]))
                                self.event_2.record()
                            self.event_1.synchronize()
                            self.event_2.synchronize()
                        elif self.num_GPU == 4:
                            with self.cuda_stream(self.stream_1):
                                self.dDell_hist_1r[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[self.index_:self.num, :]))
                                self.dDell_hist_1r[0:self.index, :].copy_(
                                    (self.keep_dDell_1.real).expand_as(self.dDell_hist_1r[0:self.index, :]))
                                self.dDell_hist_1i[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[self.index_:self.num, :]))
                                self.dDell_hist_1i[0:self.index, :].copy_(
                                    (self.keep_dDell_1.imag).expand_as(self.dDell_hist_1i[0:self.index, :]))
                                self.event_1.record()
                            with self.cuda_stream(self.stream_2):
                                self.dDell_hist_2r[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[self.index_:self.num, :]))
                                self.dDell_hist_2r[0:self.index, :].copy_(
                                    (self.keep_dDell_2.real).expand_as(self.dDell_hist_2r[0:self.index, :]))
                                self.dDell_hist_2i[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[self.index_:self.num, :]))
                                self.dDell_hist_2i[0:self.index, :].copy_(
                                    (self.keep_dDell_2.imag).expand_as(self.dDell_hist_2i[0:self.index, :]))
                                self.event_2.record()
                            with self.cuda_stream(self.stream_3):
                                self.dDell_hist_3r[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_3.real).expand_as(self.dDell_hist_3r[self.index_:self.num, :]))
                                self.dDell_hist_3r[0:self.index, :].copy_(
                                    (self.keep_dDell_3.real).expand_as(self.dDell_hist_3r[0:self.index, :]))
                                self.dDell_hist_3i[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_3.imag).expand_as(self.dDell_hist_3i[self.index_:self.num, :]))
                                self.dDell_hist_3i[0:self.index, :].copy_(
                                    (self.keep_dDell_3.imag).expand_as(self.dDell_hist_3i[0:self.index, :]))
                                self.event_3.record()
                            with self.cuda_stream(self.stream_4):
                                self.dDell_hist_4r[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_4.real).expand_as(self.dDell_hist_4r[self.index_:self.num, :]))
                                self.dDell_hist_4r[0:self.index, :].copy_(
                                    (self.keep_dDell_4.real).expand_as(self.dDell_hist_4r[0:self.index, :]))
                                self.dDell_hist_4i[self.index_:self.num, :].copy_(
                                    (self.keep_dDell_4.imag).expand_as(self.dDell_hist_4i[self.index_:self.num, :]))
                                self.dDell_hist_4i[0:self.index, :].copy_(
                                    (self.keep_dDell_4.imag).expand_as(self.dDell_hist_4i[0:self.index, :]))
                                self.event_4.record()
                            self.event_1.synchronize()
                            self.event_2.synchronize()
                            self.event_3.synchronize()
                            self.event_4.synchronize()
                    else:
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
    def conv_st(self, kernel_st, Dell):
        return kernel_st * Dell


    # Instant convolution (should be done at every time step).
    def conv_inst_1(self, kernel_sum, kernel, d_Dell, dt0):
        if dt0 >= self.Tw:
            self.inst = self.num
            return kernel_sum * d_Dell
        else:
            self.inst = int(dt0 / self.min_step)
            return kernel[self.num-self.inst:self.num, :].sum(dim=0) * d_Dell
            
    def conv_inst_2(self, kernel_sum, kernel_1, kernel_2, d_Dell, dt0):
        if dt0 >= self.Tw:
            self.inst = self.num
            return kernel_sum * d_Dell
        else:
            self.inst = int(dt0 / self.min_step)
            with self.cuda_stream(self.stream_1):
                self.K_inst_1 = kernel_1[self.num-self.inst:self.num, :].sum(dim=0)
                self.event_1.record()
            with self.cuda_stream(self.stream_2):
                self.K_inst_2 = kernel_2[self.num-self.inst:self.num, :].sum(dim=0)
                self.K_inst_2 = self.K_inst_2.to('cuda:0', non_blocking=True)
                self.event_2.record()
            self.event_1.synchronize()
            self.event_2.synchronize()
            return self.tr_cat((self.K_inst_1, self.K_inst_2), dim=0) * d_Dell
    
    def conv_inst_4(self, kernel_sum, kernel_1, kernel_2, kernel_3, kernel_4, d_Dell, dt0):
        if dt0 >= self.Tw:
            self.inst = self.num
            return kernel_sum * d_Dell
        else:
            self.inst = int(dt0 / self.min_step)
            with self.cuda_stream(self.stream_1):
                self.K_inst_1 = kernel_1[self.num-self.inst:self.num, :].sum(dim=0)
                self.event_1.record()
            with self.cuda_stream(self.stream_2):
                self.K_inst_2 = kernel_2[self.num-self.inst:self.num, :].sum(dim=0)
                self.K_inst_2 = self.K_inst_2.to('cuda:0', non_blocking=True)
                self.event_2.record()
            with self.cuda_stream(self.stream_3):
                self.K_inst_3 = kernel_3[self.num-self.inst:self.num, :].sum(dim=0)
                self.K_inst_3 = self.K_inst_3.to('cuda:0', non_blocking=True)
                self.event_3.record()
            with self.cuda_stream(self.stream_4):
                self.K_inst_4 = kernel_4[self.num-self.inst:self.num, :].sum(dim=0)
                self.K_inst_4 = self.K_inst_4.to('cuda:0', non_blocking=True)
                self.event_4.record()
            self.event_1.synchronize()
            self.event_2.synchronize()
            self.event_3.synchronize()
            self.event_4.synchronize()
            return self.tr_cat((self.K_inst_1, self.K_inst_2,
                                self.K_inst_3, self.K_inst_4), dim=0) * d_Dell


    # Dynamic kernel convolution.
    def conv_1(self, kernel, dt):
        self.inst = int(dt / self.min_step)
        if dt > self.Tw:
            self.K.zero_()
            return self.K
        else:
            self.index = self.latest + self.inst - self.num + 1
            if self.split:
                if self.index >= 0:
                    self.Kr = (kernel[0:self.num-self.inst, :] * self.dDell_hist_r[self.index:self.latest+1, :]).sum(dim=0)
                    self.Ki = (kernel[0:self.num-self.inst, :] * self.dDell_hist_i[self.index:self.latest+1, :]).sum(dim=0)
                    self.K = self.tr_complex(self.Kr, self.Ki)
                    return self.K
                else:
                    self.Kr = (kernel[0:-self.index, :] * self.dDell_hist_r[self.index+self.num:self.num, :]).sum(dim=0)
                    self.Kr += (kernel[-self.index:self.num-self.inst, :] * self.dDell_hist_r[0:self.latest+1, :]).sum(dim=0)
                    self.Ki = (kernel[0:-self.index, :] * self.dDell_hist_i[self.index+self.num:self.num, :]).sum(dim=0)
                    self.Ki += (kernel[-self.index:self.num-self.inst, :] * self.dDell_hist_i[0:self.latest+1, :]).sum(dim=0)
                    self.K = self.tr_complex(self.Kr, self.Ki)
                    return self.K
            else:
                if self.index >= 0:
                    self.K = (kernel[0:self.num-self.inst, :] * self.dDell_hist[self.index:self.latest+1, :]).sum(dim=0)
                    return self.K
                else:
                    self.K = (kernel[0:-self.index, :] * self.dDell_hist[self.index+self.num:self.num, :]).sum(dim=0)
                    self.K += (kernel[-self.index:self.num-self.inst, :] * self.dDell_hist[0:self.latest+1, :]).sum(dim=0)
                    return self.K
    
    def conv_2(self, kernel_1, kernel_2, dt):
        self.inst = int(dt / self.min_step)
        if dt > self.Tw:
            self.K.zero_()
            return self.K
        else:
            self.index = self.latest + self.inst - self.num + 1
            if self.index >= 0:
                if self.split:
                    with self.cuda_stream(self.stream_1):
                        self.K1r = (kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1r[self.index:self.latest+1, :]).sum(dim=0)
                        self.K1i = (kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1i[self.index:self.latest+1, :]).sum(dim=0)
                        self.K1 = self.tr_complex(self.K1r, self.K1i)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2r = (kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2r[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2i = (kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2i[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.tr_complex(self.K2r, self.K2i)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                else:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1[self.index:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
                self.K = self.tr_cat((self.K1, self.K2), dim=0)
                self.synchronize()
                return self.K
            else:
                if self.split:
                    with self.cuda_stream(self.stream_1):
                        self.K1r = (kernel_1[0:-self.index, :] * self.dDell_hist_1r[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1r += (kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1r[0:self.latest+1, :]).sum(dim=0)
                        self.K1i = (kernel_1[0:-self.index, :] * self.dDell_hist_1i[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1i += (kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1i[0:self.latest+1, :]).sum(dim=0)
                        self.K1 = self.tr_complex(self.K1r, self.K1i)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2r = (kernel_2[0:-self.index, :] * self.dDell_hist_2r[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2r += (kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2r[0:self.latest+1, :]).sum(dim=0)
                        self.K2i = (kernel_2[0:-self.index, :] * self.dDell_hist_2i[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2i += (kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2i[0:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.tr_complex(self.K2r, self.K2i)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                else:    
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (kernel_1[0:-self.index, :] * self.dDell_hist_1[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1 += (kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1[0:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (kernel_2[0:-self.index, :] * self.dDell_hist_2[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2 += (kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2[0:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                self.event_1.synchronize()
                self.event_2.synchronize()
                self.K = self.tr_cat((self.K1, self.K2), dim=0)
                self.synchronize()
                return self.K
    
    def conv_4(self, kernel_1, kernel_2, kernel_3, kernel_4, dt):
        self.inst = int(dt / self.min_step)
        if dt > self.Tw:
            self.K.zero_()
            return self.K
        else:
            self.index = self.latest + self.inst - self.num + 1
            if self.index >= 0:
                if self.split:
                    with self.cuda_stream(self.stream_1):
                        self.K1r = (kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1r[self.index:self.latest+1, :]).sum(dim=0)
                        self.K1i = (kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1i[self.index:self.latest+1, :]).sum(dim=0)
                        self.K1 = self.tr_complex(self.K1r, self.K1i)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2r = (kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2r[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2i = (kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2i[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.tr_complex(self.K2r, self.K2i)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    with self.cuda_stream(self.stream_3):
                        self.K3r = (kernel_3[0:self.num-self.inst, :] * self.dDell_hist_3r[self.index:self.latest+1, :]).sum(dim=0)
                        self.K3i = (kernel_3[0:self.num-self.inst, :] * self.dDell_hist_3i[self.index:self.latest+1, :]).sum(dim=0)
                        self.K3 = self.tr_complex(self.K3r, self.K3i)
                        self.K3 = self.K3.to('cuda:0', non_blocking=True)
                        self.event_3.record()
                    with self.cuda_stream(self.stream_4):
                        self.K4r = (kernel_4[0:self.num-self.inst, :] * self.dDell_hist_4r[self.index:self.latest+1, :]).sum(dim=0)
                        self.K4i = (kernel_4[0:self.num-self.inst, :] * self.dDell_hist_4i[self.index:self.latest+1, :]).sum(dim=0)
                        self.K4 = self.tr_complex(self.K4r, self.K4i)
                        self.K4 = self.K4.to('cuda:0', non_blocking=True)
                        self.event_4.record()
                else:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (kernel_1[0:self.num-self.inst, :] * self.dDell_hist_1[self.index:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (kernel_2[0:self.num-self.inst, :] * self.dDell_hist_2[self.index:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    with self.cuda_stream(self.stream_3):
                        self.K3 = (kernel_3[0:self.num-self.inst, :] * self.dDell_hist_3[self.index:self.latest+1, :]).sum(dim=0)
                        self.K3 = self.K3.to('cuda:0', non_blocking=True)
                        self.event_3.record()
                    with self.cuda_stream(self.stream_4):
                        self.K4 = (kernel_4[0:self.num-self.inst, :] * self.dDell_hist_4[self.index:self.latest+1, :]).sum(dim=0)
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
                if self.split:
                    with self.cuda_stream(self.stream_1):
                        self.K1r = (kernel_1[0:-self.index, :] * self.dDell_hist_1r[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1r += (kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1r[0:self.latest+1, :]).sum(dim=0)
                        self.K1i = (kernel_1[0:-self.index, :] * self.dDell_hist_1i[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1i += (kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1i[0:self.latest+1, :]).sum(dim=0)
                        self.K1 = self.tr_complex(self.K1r, self.K1i)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2r = (kernel_2[0:-self.index, :] * self.dDell_hist_2r[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2r += (kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2r[0:self.latest+1, :]).sum(dim=0)
                        self.K2i = (kernel_2[0:-self.index, :] * self.dDell_hist_2i[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2i += (kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2i[0:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.tr_complex(self.K2r, self.K2i)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    with self.cuda_stream(self.stream_3):
                        self.K3r = (kernel_3[0:-self.index, :] * self.dDell_hist_3r[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K3r += (kernel_3[-self.index:self.num-self.inst, :] * self.dDell_hist_3r[0:self.latest+1, :]).sum(dim=0)
                        self.K3i = (kernel_3[0:-self.index, :] * self.dDell_hist_3i[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K3i += (kernel_3[-self.index:self.num-self.inst, :] * self.dDell_hist_3i[0:self.latest+1, :]).sum(dim=0)
                        self.K3 = self.tr_complex(self.K3r, self.K3i)
                        self.K3 = self.K3.to('cuda:0', non_blocking=True)
                        self.event_3.record()
                    with self.cuda_stream(self.stream_4):
                        self.K4r = (kernel_4[0:-self.index, :] * self.dDell_hist_4r[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K4r += (kernel_4[-self.index:self.num-self.inst, :] * self.dDell_hist_4r[0:self.latest+1, :]).sum(dim=0)
                        self.K4i = (kernel_4[0:-self.index, :] * self.dDell_hist_4i[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K4i += (kernel_4[-self.index:self.num-self.inst, :] * self.dDell_hist_4i[0:self.latest+1, :]).sum(dim=0)
                        self.K4 = self.tr_complex(self.K4r, self.K4i)
                        self.K4 = self.K4.to('cuda:0', non_blocking=True)
                        self.event_4.record()
                else:
                    with self.cuda_stream(self.stream_1):
                        self.K1 = (kernel_1[0:-self.index, :] * self.dDell_hist_1[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K1 += (kernel_1[-self.index:self.num-self.inst, :] * self.dDell_hist_1[0:self.latest+1, :]).sum(dim=0)
                        self.event_1.record()
                    with self.cuda_stream(self.stream_2):
                        self.K2 = (kernel_2[0:-self.index, :] * self.dDell_hist_2[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K2 += (kernel_2[-self.index:self.num-self.inst, :] * self.dDell_hist_2[0:self.latest+1, :]).sum(dim=0)
                        self.K2 = self.K2.to('cuda:0', non_blocking=True)
                        self.event_2.record()
                    with self.cuda_stream(self.stream_3):
                        self.K3 = (kernel_3[0:-self.index, :] * self.dDell_hist_3[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K3 += (kernel_3[-self.index:self.num-self.inst, :] * self.dDell_hist_3[0:self.latest+1, :]).sum(dim=0)
                        self.K3 = self.K3.to('cuda:0', non_blocking=True)
                        self.event_3.record()
                    with self.cuda_stream(self.stream_4):
                        self.K4 = (kernel_4[0:-self.index, :] * self.dDell_hist_4[self.index+self.num:self.num, :]).sum(dim=0)
                        self.K4 += (kernel_4[-self.index:self.num-self.inst, :] * self.dDell_hist_4[0:self.latest+1, :]).sum(dim=0)
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
    def exe_conv(self, ker, Dell, d_Dell, dt, first):
        if not self.qd: # Fully-dynamic.
            if first:
                if self.num_GPU == 0 or self.num_GPU == 1:
                    tmp = (self.conv_st(ker.kernel_st, Dell) + 
                           self.conv_1(ker.kernel, dt) +
                           self.conv_inst_1(ker.kernel_sum, ker.kernel, d_Dell, dt)
                           )
                elif self.num_GPU == 2:
                    tmp = (self.conv_st(ker.kernel_st, Dell) + 
                           self.conv_2(ker.kernel_1, ker.kernel_2, dt) +
                           self.conv_inst_2(ker.kernel_sum, ker.kernel_1, ker.kernel_2, d_Dell, dt)
                           )
                elif self.num_GPU == 4:
                    tmp = (self.conv_st(ker.kernel_st, Dell) + 
                           self.conv_4(ker.kernel_1, ker.kernel_2, 
                                       ker.kernel_3, ker.kernel_4, dt) +
                           self.conv_inst_4(ker.kernel_sum, ker.kernel_1, ker.kernel_2, 
                                            ker.kernel_3, ker.kernel_4, d_Dell, dt)
                           )
            else:
                if self.num_GPU == 0 or self.num_GPU == 1:
                    tmp = (self.conv_st(ker.kernel_st, Dell) + 
                            self.K + 
                            self.conv_inst_1(ker.kernel_sum, ker.kernel, d_Dell, dt)
                            )
                elif self.num_GPU == 2:
                    tmp = (self.conv_st(ker.kernel_st, Dell) + 
                            self.K + 
                            self.conv_inst_2(ker.kernel_sum, ker.kernel_1, ker.kernel_2, d_Dell, dt)
                            )
                elif self.num_GPU == 4:
                    tmp = (self.conv_st(ker.kernel_st, Dell) + 
                            self.K + 
                            self.conv_inst_4(ker.kernel_sum, ker.kernel_1, ker.kernel_2, 
                                            ker.kernel_3, ker.kernel_4, d_Dell, dt)
                            )
            if not self.mirror:
                return self.tr_irfft(tmp, self.nconv)[:self.ncell]
            else:
                return self.tr_idct(tmp)[:self.ncell]
        else: # Quasi-dynamic.
            tmp = self.conv_st(ker.kernel_st, Dell)
            if not self.mirror:
                return self.tr_irfft( tmp, n=self.nconv )[:self.ncell]
            else:
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
        return ini.Dell
    
    # Update Dell (second step).
    def upt_Dell_second(self, ini, dt):
        self.keep_dDell = (ini.d_Dell_prv + ini.d_Dell) * 0.5
        ini.Dell = ini.Dell_prv + self.keep_dDell * dt
        return ini.Dell