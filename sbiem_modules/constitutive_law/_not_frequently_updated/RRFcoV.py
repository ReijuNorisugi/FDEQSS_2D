# -*- coding: utf-8 -*-
"""Fully_dynamic_torch.py
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


# Class for Nweton-Raphson method
class NR():
    def __init__(self, eta=Medium['eta'], Vpl = Medium['Vpl'], Vc=FaultParams['Vc'],
                 a=FaultParams['a'], c=FaultParams['c'], tauc=FaultParams['tauc'], f0=FaultParams['f0'],
                 k=FaultParams['k'], alpha=FaultParams['alpha'], beta=FaultParams['beta'],
                 sigma=FaultParams['sigma']):
        self.p = 0.          # log(V/V0).
        self.calc = 0.       # Calculation unit.
        self.ep = 0.         # Calculation unit.
        self.T = 0.          # Objective funtion T.
        self.dT = 0.         # Derivative of objective function T.
        self.tol = 0.        # Torlelance.
        self.max_rep = 10**6
    
        self.eta = eta       # Radiation dumping coefficient.
        self.Vpl = Vpl       # Loading late.
        self.f0 = f0         # Reference friction.
        self.Vc = Vc         # Cut-off velocity.
        self.sigma = sigma   # Normal stress.
        self.A = a * sigma   # Direct effect coefficient.
        self.C = c * sigma   # Shear strength coefficient.
        self.k2 = k**2       # Wavenumber square.
        self.alpha = alpha   # Abrasoin coefficient.
        self.beta = beta     # Adhesion coefficient.
        self.tauc = tauc     # Reference stress (gouge effect).
        self.etaA = eta / self.A    # Coefficient.
        self.etaVpl = eta * Vpl     # Coefficient.
        self.epsilon = 10**(-14)    # Criteria.
        self.dtype = tr.float64

        # Define functions in advance.
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1
        self.tr_max = tr.max
        self.tr_abs = tr.abs
        self.tr_sqrt = tr.sqrt
        self.tr_sum = tr.sum
        self.tr_any = tr.any
        self.tr_cat = tr.cat
        self.tr_clamp = tr.clamp


    # Get masimum absolute value which contributes error of NR search.
    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))


    # Newton-Raphson method. Note using "for" loop for each cell significantly slows your code.
    def NR_search(self, ini):
        self.p = self.tr_log(ini.V)
        self.ep = ini.V
        self.calc = (-self.etaVpl - ini.tau_ini - ini.f + self.tauc + self.C * self.tr_sqrt(self.tr_sum(self.k2 * (ini.state**2), axis=1))) / self.A
        self.arg = self.p - self.tr_log(ini.V + self.Vc)
        self.arg_ = self.etaA * ini.V
        self.T = self.calc + self.arg + self.arg_
        self.dT = 1. + self.arg_ - ini.V / (ini.V + self.Vc)

        for _ in range(self.max_rep):
            self.p -= self.T / self.dT
            self.p = self.tr_clamp(self.p, max=7.)
            self.ep = self.tr_exp(self.p)
            self.arg = self.p - self.tr_log(self.ep + self.Vc)
            self.arg_ = self.etaA * self.ep
            self.T = self.calc + self.arg + self.arg_
            
            self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.arg, self.arg_), dim=0))
            if self.tr_any(self.tr_abs(self.T) >= self.tol):
                self.dT = 1. + self.arg_ - self.ep / (self.ep + self.Vc)
                continue
            else:
                break
        return self.ep


# Class for updating physical variables.
class Update():
    def __init__(self, k=FaultParams['k'], alpha=FaultParams['alpha'], beta=FaultParams['beta'],
                Ybar=FaultParams['Ybar'], ncell=Medium['ncell'], tau_rate=Medium['tau_rate']):
        self.decay = 0.                                           # Decay rate for state variable evolution.
        self.state_ss = 0.                                        # Steady-state state variable.
        self.alpha = tr.reshape(alpha, (ncell, 1))                # Abrasion coefficient, reshape is for broadcasting.
        self.beta = beta                                          # Adhesions coefficient.
        self.betak = k * tr.reshape(beta, (ncell, 1))             # Coefficient.
        self.betakYbar = k * Ybar * tr.reshape(beta, (ncell, 1))  # Coefficient.
        self.betakkYbar = k * self.betakYbar                      # Coefficient.
        self.ncell = ncell                                        # Number of cells.
        self.tau_rate = tau_rate                                  # Stressing rate.
    
        # Define functions in advance.
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1
        self.tr_reshape = tr.reshape


    # First time evolution step.
    def first(self, ini, dt):
        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt

        ini.delta += ini.V * dt
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.tau_ini += self.tau_rate * dt
        return ini
  
  
    # Second time evolution step, 2nd-order accuracy.
    def second(self, ini, dt): # Note input dt is full time step, not a half.
        ini.delta = ini.delta_prv + ini.V * dt
        self.decay = self.decay * 0.5
        ini.state = ini.state_prv * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)

        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.keep_V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt * 0.5
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.tau_ini += self.tau_rate * dt
        return ini


    # Take average of velocity.
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini