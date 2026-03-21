# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20251001.
"""

# This is the module for computing stress transfer functional.
# This is specialilzed for CPU computation.

import numpy as np
import torch as tr
import torch_dct as dct

class Convolution():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.qd       = Conditions['qd']       # Flag for quasi-dynamic.
        self.mirror   = Conditions['mirror']   # Flag for mirror.
        self.Stepper  = Conditions['Stepper']  # Flag for time marching scheme.
        self.outerror = Conditions['outerror']  # Flag for error monitoring.
        self.kn    = Medium['kn']       # Wavenumber.
        self.Tw    = Medium['Tw']       # Truncation window.
        self.Nele  = Medium['Nele']     # System size.
        self.ncell = Medium['ncell']    # Number of cells.
        self.nconv = self.Nele          # Number of cells for convolution.
        self.nperi = Medium['nperi']    # Replication period.
        self.dtmin = Medium['dtmin']    # Minimum time step.
        self.device = Devices['device_1']
        
        self.plot_kernel = False    # Flag to visalize the convolutional kernel.
        
        self.split = True  # Split real and imaginary part of slip rate.
        if self.mirror:
            self.split = False
        
        if self.Stepper == 'LR':
            self.min_step = self.dtmin
        elif self.Stepper == 'LR_mdf':
            self.min_step = self.dtmin
        elif self.Stepper == 'RO': # If stepper is RO, we use half of dtmin for double-half step solution.
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
                self.dtype_np = np.complex128
                self.dtype_tr = tr.complex128
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
            self.K = tr.zeros(len(self.kn), dtype=self.dtype_tr, device=self.device)
        
        # History of Fourier slip rate.
        if not self.qd:
            if self.split:
                self.dDell_hist_r = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
                self.dDell_hist_i = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
            else:
                self.dDell_hist = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
            
        # Defining functions in advance.
        self.tr_sum = tr.sum
        self.tr_irfft = tr.fft.irfft
        self.tr_rfft = tr.fft.rfft
        self.tr_dct = dct.dct
        self.tr_idct = dct.idct
        self.tr_cat = tr.cat
        self.tr_chunk = tr.chunk
        self.tr_complex = tr.complex
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state
        
    def __setstate__(self, state):
        self.__dict__ = state
    
    def reset_history(self):
        if not self.qd:
            if self.split:
                self.dDell_hist_r = None
                self.dDell_hist_i = None
            else:
                self.dDell_hist = None
    
    # Note that torch.chunk impose overflowed component to the first tensor.
    # And it split tensors by same sizes as possible.
    def send_history(self):
        if not self.qd:
            pass
    
    def sendback_history(self):
        if not self.qd:
            pass
    
    def create_hist_on_cpu(self):
        if self.split:
            self.dDell_hist_r = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
            self.dDell_hist_i = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
        else:
            self.dDell_hist = tr.zeros(len(self.thist), len(self.kn), dtype=self.dtype_tr, device='cpu')
            
    
    def copy_all_hist(self):
        if self.qd:
            return
        else:
            pass
        
        if self.split:
            self.dDell_hist_r.copy_((self.kkeep_dDell.real).expand_as(self.dDell_hist_r))
            self.dDell_hist_i.copy_((self.kkeep_dDell.imag).expand_as(self.dDell_hist_i))
        else:
            self.dDell_hist.copy_(self.kkeep_dDell.expand_as(self.dDell_hist))
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
                if self.flag_copy == 1:
                    self.init = tar.init_copy
                    self.end = self.len_thist
                    self.init_ = 0
                    self.end_ = tar.end_copy
                else:
                    self.init = tar.init_copy
                    self.end = tar.end_copy
                
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
                
                self.index = self.latest + self.inst - self.num + 1 # For efficient computation.
                if self.latest == self.num - 1:
                    if self.split:
                        self.dDell_hist_r[0:self.inst, :].copy_((self.keep_dDell.real).expand_as(self.dDell_hist_r[0:self.inst, :]))
                        self.dDell_hist_i[0:self.inst, :].copy_((self.keep_dDell.imag).expand_as(self.dDell_hist_i[0:self.inst, :]))
                    else:
                        self.dDell_hist[0:self.inst, :].copy_(self.keep_dDell.expand_as(self.dDell_hist[0:self.inst, :]))
                    self.latest = self.inst - 1

                elif self.index <= 0:
                    self.index_ = self.latest + 1
                    self.index__ = self.index_ + self.inst
                    if self.split:
                        self.dDell_hist_r[self.index_:self.index__, :].copy_(
                            (self.keep_dDell.real).expand_as(self.dDell_hist_r[self.index_:self.index__, :]))
                        self.dDell_hist_i[self.index_:self.index__, :].copy_(
                            (self.keep_dDell.imag).expand_as(self.dDell_hist_i[self.index_:self.index__, :]))
                    else:
                        self.dDell_hist[self.index_:self.index__, :].copy_(
                            self.keep_dDell.expand_as(self.dDell_hist[self.index_:self.index__, :]))
                    self.latest += self.inst

                else:
                    self.index_ = self.latest + 1
                    if self.split:
                        self.dDell_hist_r[self.index_:self.num, :].copy_(
                            (self.keep_dDell.real).expand_as(self.dDell_hist_r[self.index_:self.num, :]))
                        self.dDell_hist_r[0:self.index, :].copy_(
                            (self.keep_dDell.real).expand_as(self.dDell_hist_r[0:self.index, :]))
                        self.dDell_hist_i[self.index_:self.num, :].copy_(
                            (self.keep_dDell.imag).expand_as(self.dDell_hist_i[self.index_:self.num, :]))
                        self.dDell_hist_i[0:self.index, :].copy_(
                            (self.keep_dDell.imag).expand_as(self.dDell_hist_i[0:self.index, :]))
                    else:
                        self.dDell_hist[self.index_:self.num, :].copy_(
                            self.keep_dDell.expand_as(self.dDell_hist[self.index_:self.num, :]))
                        self.dDell_hist[0:self.index, :].copy_(
                            self.keep_dDell.expand_as(self.dDell_hist[0:self.index, :]))
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
    def conv_inst(self, kernel_sum, kernel, d_Dell, dt0):
        if dt0 >= self.Tw:
            self.inst = self.num
            return kernel_sum * d_Dell
        else:
            self.inst = int(dt0 / self.min_step)
            return sum_jit_comp(kernel[self.num-self.inst:self.num, :]) * d_Dell

    # Dynamic kernel convolution (availabe at second step).
    def conv(self, kernel, dt):
        self.inst = int(dt / self.min_step)
        if dt > self.Tw:
            self.K.zero_()
            return self.K
        else:
            self.index = self.latest + self.inst - self.num + 1
            if self.index >= 0:
                if self.split:
                    self.Kr = conv_jit_comp(kernel[0:self.num-self.inst, :], 
                                            self.dDell_hist_r[self.index:self.latest+1, :])
                    self.Ki = conv_jit_comp(kernel[0:self.num-self.inst, :], 
                                            self.dDell_hist_i[self.index:self.latest+1, :])
                    self.K = self.tr_complex(self.Kr, self.Ki)
                else:
                    self.K = conv_jit_comp(kernel[0:self.num-self.inst, :], 
                                           self.dDell_hist[self.index:self.latest+1, :])
                return self.K
            else:
                if self.split:
                    self.Kr = conv_jit_comp(kernel[0:-self.index, :],
                                            self.dDell_hist_r[self.index+self.num:self.num, :])
                    self.Kr += conv_jit_comp(kernel[-self.index:self.num-self.inst, :],
                                             self.dDell_hist_r[0:self.latest+1, :])
                    self.Ki = conv_jit_comp(kernel[0:-self.index, :],
                                            self.dDell_hist_i[self.index+self.num:self.num, :])
                    self.Ki += conv_jit_comp(kernel[-self.index:self.num-self.inst, :],
                                             self.dDell_hist_i[0:self.latest+1, :])
                    self.K = self.tr_complex(self.Kr, self.Ki)
                else:
                    self.K = conv_jit_comp(kernel[0:-self.index, :],
                                           self.dDell_hist[self.index+self.num:self.num, :])
                    self.K += conv_jit_comp(kernel[-self.index:self.num-self.inst, :],
                                            self.dDell_hist[0:self.latest+1, :])
                return self.K


    # Execute convolutions.
    def exe_conv(self, ker, Dell, d_Dell, dt, first):
        if not self.qd: # Fully-dynamic.
            if first:
                tmp = (self.conv_st(ker.kernel_st, Dell) + 
                       self.conv(ker.kernel, dt) +
                       self.conv_inst(ker.kernel_sum, ker.kernel, d_Dell, dt)
                       )
            else:
                tmp = (self.conv_st(ker.kernel_st, Dell) + 
                       self.K + 
                       self.conv_inst(ker.kernel_sum, ker.kernel, d_Dell, dt))
            if not self.mirror:
                return self.tr_irfft(tmp, self.nconv)[:self.ncell]
            else:
                return self.tr_idct(tmp)[:self.ncell]
        else: # Quasi-dynamic.
            tmp = self.conv_st(ker.kernel_st, Dell)
            if not self.mirror:
                return self.tr_irfft(tmp, n=self.nconv)[:self.ncell]
            else:
                return self.tr_idct(tmp)[:self.ncell]


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


# JIT compile of dynamic convolution.
# Torch.jit.script provides great efficiency when computing on CPU.
@tr.jit.script
def conv_jit_comp(kernel: tr.Tensor, hist: tr.Tensor) -> tr.Tensor:
    return (kernel * hist).sum(dim=0)

@tr.jit.script
def sum_jit_comp(kernel: tr.Tensor) -> tr.Tensor:
    return kernel.sum(dim=0)