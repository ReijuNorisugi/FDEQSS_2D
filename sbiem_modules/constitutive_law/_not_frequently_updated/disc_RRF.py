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


# Class for Nweton-Raphson method.
class NR():
    def __init__(self, eta=Medium['eta'], Vpl = Medium['Vpl'], V0=Medium['V0'],
                 a=FaultParams['a'], c=FaultParams['c'], tauc=FaultParams['tauc'], k=FaultParams['k'],
                 sigma=FaultParams['sigma'], n_array=FaultParams['n_array'], device=Devices['device_1']):
        self.max_rep = 10**3     # Max iteration for Newton-Raphson.
        self.epsilon = 10**(-14) # Convergence Criteria
        self.atol = 10**(-20)    # Absolute error.
    
        self.A = a * sigma   # direct effect coefficient
        self.C = c * sigma   # shear strength coefficient.
        self.etaA = eta / self.A
        self._ = tr.tensor(2., dtype=tr.float64, device=device)
        self.k3 = (k**2)*n_array*tr.log(self._)
        self.V0 = tr.tensor(V0, dtype=tr.float64, device=device)
        self.arg = - tr.log(self.V0) + (tauc - eta * Vpl) / self.A

        # define functions in advance
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_max = tr.max
        self.tr_abs = tr.abs
        self.tr_sqrt = tr.sqrt
        self.tr_sum = tr.sum
        self.tr_any = tr.any
        self.tr_clamp = tr.clamp
        self.tr_cat = tr.cat
        self.tr_stack = tr.stack


    # get masimum absolute value which contributes error of NR search
    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))


    # Newton-Raphson method. Note using for loop significantly slows your code
    def Halley(self, ini):
        p = self.tr_log( ini.V )
        calc = self.arg + ( self.C * self.tr_sqrt(self.tr_sum((self.k3)*(ini.state**2), axis=1)) - ini.tau_ini - ini.f ) / self.A
        ep = self.etaA * ini.V
        T = calc + p + ep
        dT1 = 1. + ep
        dT2 = ep

        for _ in range(self.max_rep):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = self.tr_clamp(p, max=7.)
            ep = self.etaA * self.tr_exp(p)
            T = calc + p + ep
            
            tol = self.epsilon * self.absmax(self.tr_cat((calc, p, ep), dim=0))
            if self.tr_any(self.tr_abs(T) >= tol):
                dT1 = 1. + ep
                dT2 = ep
                continue
            else:
                break
        return ep / self.etaA
  


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

        self.tmp = self.tr_expm1(self.decay)
        ini.state = ini.state * (self.tmp + 1) - self.state_ss * self.tmp
        
        ini.delta += ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini

  
    # Second time evolution step, 2nd-order accuracy.
    def second(self, ini, dt): # Note input dt is full time step, not a half.
        self.decay = self.decay * 0.5
        self.tmp = self.tr_expm1(self.decay)
        ini.state = ini.state_prv * (self.tmp + 1.) - self.state_ss * self.tmp

        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.keep_V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt * 0.5
        self.tmp = self.tr_expm1(self.decay)
        ini.state = ini.state * (self.tmp + 1.) - self.state_ss * self.tmp
        
        ini.delta = ini.delta_prv + ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini


    # Take average of velocity.
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini