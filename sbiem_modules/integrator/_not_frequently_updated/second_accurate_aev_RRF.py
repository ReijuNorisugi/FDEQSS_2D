# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
"""

# This is the module for second-order accurate solution.

import torch as tr
import numpy as np
import copy
import pickle

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

fname = load_pickle('fname.pkl')['fname']
Conditions = load_pickle('Output{}Conditions.pkl'.format(fname))
Medium = load_pickle('Output{}Medium.pkl'.format(fname))
FieldVariables = load_pickle('Output{}FieldVariables.pkl'.format(fname))
FaultParams = load_pickle('Output{}FaultParams.pkl'.format(fname))
Devices = load_pickle('Output{}Devices.pkl'.format(fname))


#Class for one time step.
class Onetimestep():
    def __init__(self, delta=FieldVariables['delta'], state=FieldVariables['state'], V=FieldVariables['V'], 
                 Dell=FieldVariables['Dell'], d_Dell=FieldVariables['d_Dell'], f=FieldVariables['f'], tau=FieldVariables['tau'],
                 Vpl=Medium['Vpl'], eta=Medium['eta'], tau_ini=FaultParams['tau_ini'], tau_rate=Medium['tau_rate'],
                 ncell=Medium['ncell'], nperi=Medium['nperi'], mirror=Conditions['mirror'], Cutoff=Conditions['Cutoff'],
                 tauc=FaultParams['tauc'], sigma=FaultParams['sigma'], a=FaultParams['a'], c=FaultParams['c'],
                 k=FaultParams['k'], Vc=FaultParams['Vc'], phi=FieldVariables['phi'], V0=Medium['V0'], 
                 a_dyn=FaultParams['a_dyn'], alpha=FaultParams['alpha'], beta=FaultParams['beta'],
                 aev=Conditions['aev'], Vf=FaultParams['Vf'], exponent=FaultParams['exponent']):
        self.delta = delta.clone()            # Displacement.
        self.state = state.clone()            # State variables.
        self.V = V.clone()                    # Slip rate.
        self.tau = tau.clone()                # Shear traction.
        self.phi = phi.clone()                # Shear strength.
        self.tau_ini = tau_ini.clone()        # Tau0.
        self.Dell = Dell.clone()              # Fourier displacement
        self.d_Dell = d_Dell.clone()          # Fourier velocity
        self.f = f.clone()                    # Fourier shear stress
        self.keep_dDell = self.d_Dell.clone() # instant dDell
        self.keep_V = self.V.clone()          # instant slip rate
        
        # Preserve previous step informations.
        self.delta_prv   = delta.clone()
        self.state_prv   = state.clone()
        self.V_prv       = V.clone()
        self.tau_prv     = tau.clone()
        self.tau_ini_prv = tau_ini.clone()
        self.Dell_prv    = Dell.clone()
        self.d_Dell_prv  = d_Dell.clone()
        self.f_prv       = f.clone()
        
        self.Vpl = Vpl                        # Loading rate.
        self.eta = eta                        # Radiation dumping.
        self.tau_rate = tau_rate              # Tau_rate: stress loading.
        self.tauc = tauc
        self.A = a * sigma
        self.C = c * sigma
        self.k2 = k**2
        self.Vc = Vc
        self.V0 = V0
        self.Vf = Vf
        self.exponent = exponent
        self.A_dyn = a_dyn * sigma
        self.ncell = ncell                    # Number of cells.
        self.nperi = nperi                    # Fault replication.
        self.nconv = self.ncell * self.nperi  # Number of cells for convolution.
        self.mirror = mirror                  # Flag of mirror.
        self.Cutoff = Cutoff
        self.aev = aev
        
        # Define functions in advnace.
        self.deepcopy = copy.deepcopy
        self.tr_sum = tr.sum
        self.tr_log = tr.log
        self.tr_sqrt = tr.sqrt
        self.synchronize = tr.cuda.synchronize


    # Function for one time step evolution, 2nd-order accuracy.
    def guess(self, update, cv, pr, ti, dt, dt_, first, second, store):
        # Update delta, state, Dell, and tau_ini.
        self = update.first(self, dt)
        self = cv.upt_Dell_first(self, dt)
        # Convolution.
        self.f = cv.exe_conv(self.Dell, self.d_Dell, dt, first)
        # Seek solution of V.
        self.V = pr.Halley(self)
        # FFT(V).
        self.d_Dell = cv.vfft(self.V-self.Vpl)
        
        ########################################

        # Update V to average value.
        self = update.ave(self)
        # Update delta, state, Dell, and tau_ini.
        self = update.second(self, dt)
        self = cv.upt_Dell_second(self, dt)
        # Convolution, utilize the first convolution.
        self.f = cv.exe_conv(self.Dell, cv.keep_dDell, dt, second)
        # Seek solution of V.
        self.V = pr.Halley(self)
        # FFT (delta).
        self.Dell = cv.delfft(self.delta - self.Vpl*((dt_ + ti.t_split) + ti.id_unit*ti.T_unit))
        # FFT (V).
        self.d_Dell = cv.vfft(self.V - self.Vpl)
        # Store d_Dell history when step is half size.
        cv.store_dDell(dt, store)


    # Update physical variables when time steps forward.
    def upt_prv(self, upt):
        
        self.delta_prv.copy_(upt.delta)
        self.state_prv.copy_(upt.state)
        self.V_prv.copy_(upt.V)
        self.Dell_prv.copy_(upt.Dell)
        self.d_Dell_prv.copy_(upt.d_Dell)
        self.f_prv.copy_(upt.f)
        self.tau_ini_prv.copy_(upt.tau_ini)
        
        self.phi = self.tauc + self.C*self.tr_sqrt(self.tr_sum(self.k2 * (upt.state**2), axis=1))
        self.tau = (self.A_dyn + (self.A - self.A_dyn) / (1. + (upt.V / self.Vf)**self.exponent)) * self.tr_log(upt.V / self.V0) + self.phi
        self.tau_prv.copy_(self.tau)
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
        self.synchronize()
        
    # Clone tensors if time step is invalid.
    def clone_tensors(self, prv):
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
        self.synchronize()