# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260323.
"""

# This is the module which summarizes the functions for single time step.

import torch as tr

# Class for one time step.
class Heun():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        # Main device for computation.
        self.device = Devices['device_1']
        
        # Current variables.
        self.delta   = FieldVariables['delta'].clone()   # Displacement.
        self.state   = FieldVariables['state'].clone()   # State variables.
        self.V       = FieldVariables['V'].clone()       # Slip rate.
        self.tau     = FieldVariables['tau'].clone()     # Shear stress.
        self.phi     = FieldVariables['phi'].clone()     # Shear strength.
        self.Dell    = FieldVariables['Dell'].clone()    # Fourier displacement.
        self.d_Dell  = FieldVariables['d_Dell'].clone()  # Fourier slip rate.
        self.f       = FieldVariables['f'].clone()       # Stress transfer functional.
        self.keep_V  = FieldVariables['V'].clone()       # Keep slip rate.
        self.tau_ini = FieldVariables['tau_ini'].clone() # tau0.
        
        # Preserve previous step informations.
        self.delta_prv   = FieldVariables['delta'].clone()
        self.state_prv   = FieldVariables['state'].clone()
        self.V_prv       = FieldVariables['V'].clone()
        self.tau_prv     = FieldVariables['tau'].clone()
        self.Dell_prv    = FieldVariables['Dell'].clone()
        self.d_Dell_prv  = FieldVariables['d_Dell'].clone()
        self.f_prv       = FieldVariables['f'].clone()
        self.tau_ini_prv = FieldVariables['tau_ini'].clone()
        
        # Constants.
        self.Vpl = Medium['Vpl']
        self.tauc = FaultParams['tauc']
        self.A = FaultParams['a'] * FaultParams['sigma']
        self.C = FaultParams['c'] * FaultParams['sigma']
        self.k2 = FaultParams['k']**2
        self.V0 = Medium['V0']
        
        # Define functions in advnace.
        self.tr_sum = tr.sum
        self.tr_log = tr.log
        self.tr_sqrt = tr.sqrt
        self.synchronize = tr.cuda.synchronize

    # Function for one time step evolution, 2nd-order accuracy.
    def guess(self, update, ker, cv, pr, ti, dt, dt_, first, second, store):
        # Update delta, state, Dell, and tau_ini.
        self = update.first(self, dt)
        self = cv.upt_Dell_first(self, dt)
        
        # Convolution.
        self.f = cv.exe_conv(ker, self.Dell, self.d_Dell, dt, first)
        
        # Seek solution of V.
        self.V = pr.Halley(self)
        
        # FFT(V - Vpl).
        self.d_Dell = cv.vfft(self.V-self.Vpl)

        ########################################

        # Update V to average value.
        self = update.ave(self)
        
        # Update delta, state, Dell, and tau_ini.
        self = update.second(self, dt)
        self = cv.upt_Dell_second(self, dt)
        
        # Convolution, utilize the first convolution.
        self.f = cv.exe_conv(ker, self.Dell, cv.keep_dDell, dt, second)
        
        # Seek solution of V.
        self.V = pr.Halley(self)
        
        # FFT (delta - Vpl*t).
        self.Dell = cv.delfft(self.delta - self.Vpl*((dt_ + ti.t_split) + ti.id_unit*ti.T_unit))
        
        # FFT (V - Vpl).
        self.d_Dell = cv.vfft(self.V - self.Vpl)

        # Store d_Dell history when step is half size.
        cv.store_dDell(dt, store)
        
        # Evaluate traction.
        self.eval_traction()


    def eval_traction(self):
        self.phi = self.tauc + self.C*self.tr_sqrt(self.tr_sum(self.k2 * (self.state**2), axis=1))
        self.tau = self.A * self.tr_log(self.V / self.V0) + self.phi
            

    # Update physical variables when time steps forward.
    def upt_prv(self, upt):
        
        self.delta_prv.copy_(upt.delta)
        self.state_prv.copy_(upt.state)
        self.V_prv.copy_(upt.V)
        self.Dell_prv.copy_(upt.Dell)
        self.d_Dell_prv.copy_(upt.d_Dell)
        self.f_prv.copy_(upt.f)
        self.tau_ini_prv.copy_(upt.tau_ini)
        self.tau_prv.copy_(upt.tau)
        
        if self.device != 'cpu':
            self.synchronize()
        
    def upt_prs(self, upt):
        self.delta.copy_(upt.delta)
        self.state.copy_(upt.state)
        self.V.copy_(upt.V)
        self.Dell.copy_(upt.Dell)
        self.d_Dell.copy_(upt.d_Dell)
        self.f.copy_(upt.f)
        self.tau_ini.copy_(upt.tau_ini)
        self.tau.copy_(upt.tau)
        
        if self.device != 'cpu':
            self.synchronize()
        
    # Rollback tensors if time step is invalid.
    def rollback_tensors(self, prv):
        self.delta.copy_(prv.delta_prv)
        self.state.copy_(prv.state_prv)
        self.V.copy_(prv.V_prv)
        self.tau.copy_(prv.tau_prv)
        self.tau_ini.copy_(prv.tau_ini_prv)
        self.Dell.copy_(prv.Dell_prv)
        self.d_Dell.copy_(prv.d_Dell_prv)
        self.f.copy_(prv.f_prv)
        
        self.delta_prv.copy_(prv.delta_prv)
        self.state_prv.copy_(prv.state_prv)
        self.V_prv.copy_(prv.V_prv)
        self.tau_prv.copy_(prv.tau_prv)
        self.tau_ini_prv.copy_(prv.tau_ini_prv)
        self.Dell_prv.copy_(prv.Dell_prv)
        self.d_Dell_prv.copy_(prv.d_Dell_prv)
        self.f_prv.copy_(prv.f_prv)
        
        if self.device != 'cpu':
            self.synchronize()