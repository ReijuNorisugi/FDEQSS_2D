# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
"""

#This is the module for V prediction and variable update with RRF law.

import torch as tr
import pickle

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


fname = load_pickle('fname.pkl')['fname']
Conditions = load_pickle('Output{}Conditions.pkl'.format(fname))
Medium = load_pickle('Output{}Medium.pkl'.format(fname))
FaultParams = load_pickle('Output{}FaultParams.pkl'.format(fname))
Devices = load_pickle('Output{}Devices.pkl'.format(fname))


# Class for Nweton-Raphson method
class NR():
    def __init__(self, eta=Medium['eta'], Vpl = Medium['Vpl'], V0=Medium['V0'], L=FaultParams['L'], state_size=FaultParams['state_size'],
                 a=FaultParams['a'], b_RMSF=FaultParams['b_RMSF'], tau0=FaultParams['tau0'], f0=FaultParams['f0'],
                 sigma=FaultParams['sigma'], tau_ini=FaultParams['tau_ini'], Devices=Devices):
        self.p = 0.          # log(V/V0)
        self.calc = 0.       # calculation unit
        self.ep = 0.         # calculation unit
        self.T = 0.          # Objective funtion T
        self.dT = 0.         # derivative of objective function T
        self.tol = 0.        # torlelance
        self.max_rep = 10**3 # Max iteration for Newton-Raphson.
    
        self.eta = eta       # radiation dumping coefficient
        self.Vpl = Vpl       # loading late
        self.f0 = f0         # reference friction
        self.V0 = V0         # reference velocity
        self.log_V0 = tr.log(tr.tensor([V0], dtype=tr.float64, device=Devices['device_1']))
        self.sigma = sigma   # normal stress
        self.A = a * sigma   # direct effect coefficient
        self.B = b_RMSF * tr.reshape(sigma, (-1, 1))   # shear strength coefficient
        self.L = L
        self.tau0 = tau0     # reference stress (gouge effect)
        self.tau_ini = tau_ini             # tau0
        self.etaA = eta / self.A    # coefficient
        self.etaVpl = eta * Vpl
        self.state_size = state_size
        self.etaVpl = tr.tensor(self.etaVpl, dtype=tr.float64, device=Devices['device_1'])
        self.epsilon = 10**(-14)           # criteria
        self.dtype = tr.float64
        
        # define functions in advance
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_max = tr.max
        self.tr_abs = tr.abs
        self.tr_sqrt = tr.sqrt
        self.tr_sum = tr.sum
        self.tr_any = tr.any
        self.tr_cat = tr.cat
        self.tr_reshape = tr.reshape
        self.tr_clamp = tr.clamp


    # get masimum absolute value which contributes error of NR search
    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))


    # Newton-Raphson method. Note using for loop significantly slows your code
    def NR_search(self, ini):
        self.p = self.tr_log( ini.V )
        self.calc = ( self.tau0 - ini.tau_ini - ini.f + self.tr_sum(ini.state, axis=1) - self.etaVpl ) / self.A - self.log_V0
        self.ep = self.etaA * self.tr_exp(self.p)
        self.T = self.calc + self.p + self.ep
        self.dT = 1. + self.ep

        for _ in range(self.max_rep):
            self.p -= (self.T / self.dT)
            self.p = self.tr_clamp(self.p, max=7.)
            self.ep = self.etaA * self.tr_exp(self.p)
            self.T = self.calc + self.p + self.ep
            
            self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.p, self.ep), dim=0))
            if self.tr_any(self.tr_abs(self.T) >= self.tol):
                self.dT = 1. + self.ep
                continue
            else:
                break
        return self.ep / self.etaA
  


# Class for updating physical variables
class Update():
    def __init__(self, ncell=Medium['ncell'], a=FaultParams['a'], b_RMSF=FaultParams['b_RMSF'],
                 L=FaultParams['L'], tau_rate=Medium['tau_rate'], sigma=FaultParams['sigma'], V0=Medium['V0'],
                 TW=Conditions['TW'], tc=FaultParams['tc']):
        self.decay = 0.                                           # decay rate for state variable evolution
        self.state_ss = 0.                                        # steady-state state variable
        self.a = a
        self.b_RMSF = b_RMSF
        self.B_RMSF = b_RMSF * tr.reshape(sigma, (-1, 1))
        self.L = L
        self.ncell = ncell                                        # number of cells
        self.tau_rate = tau_rate
        self.TW = TW
        self.tc = tc
        self.V0 = V0
    
        # define functions in advance
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1
        self.tr_reshape = tr.reshape
        self.tr_log = tr.log


    # first time evolution step
    def first(self, ini, dt):
         # State and L_dyn are (ncell, state_size) tensors.
        self.re_V = tr.reshape(ini.V, (-1, 1))  # reshape velocity to (ncell, 1)
        if self.TW:
            ini.L_dyn = self.L + self.re_V * self.tc
        else:
            pass
        self.state_ss = - self.B_RMSF * self.tr_log(self.re_V / self.V0)
        self.decay = - ( self.re_V / ini.L_dyn ) * dt
    
        ini.delta += ini.V * dt
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.tau_ini += self.tau_rate * dt
        return ini
  
  
    # second time evolution step, 2nd-order accuracy
    def second(self, ini, dt): # Note input dt is full time step, not a half.
        self.re_V = tr.reshape(ini.keep_V, (-1, 1))  # reshape velocity to (ncell, 1)
        if self.TW:
            ini.L_dyn = self.L + self.re_V * self.tc
        else:
            pass
        ini.delta = ini.delta_prv + ini.V * dt
        self.decay = self.decay * 0.5
        ini.state = ini.state_prv * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
    
        self.state_ss = - self.B_RMSF * self.tr_log(self.re_V / self.V0)
        self.decay = - ( self.re_V / ini.L_dyn ) * dt * 0.5
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.tau_ini += self.tau_rate * dt
        return ini


    # take average of velocity
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini