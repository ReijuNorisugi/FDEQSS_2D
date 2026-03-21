# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
"""

#This is the module for V prediction and variable update with RRF law.

import torch as tr
import numpy as np
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
                 a=FaultParams['a'], c=FaultParams['c'], tauc=FaultParams['tauc'], f0=FaultParams['f0'],
                 k=FaultParams['k'], alpha=FaultParams['alpha'], beta=FaultParams['beta'],
                 sigma=FaultParams['sigma'], Vc=FaultParams['Vc'], a_dyn=FaultParams['a_dyn'], 
                 gamma=FaultParams['gamma'], Vc_=FaultParams['Vc_'], Devices=Devices):
        self.max_rep = 10**5 # Max iteration for Newton-Raphson.
    
        self.eta = eta       # radiation dumping coefficient
        self.Vpl = Vpl       # loading late
        self.f0 = f0         # reference friction
        self.V0 = V0         # reference velocity
        self.log_V0 = tr.log(tr.tensor([V0], dtype=tr.float64, device=Devices['device_1']))
        self.sigma = sigma   # normal stress
        self.A = a * sigma   # direct effect coefficient
        self.C = c * sigma   # shear strength coefficient
        self.A_dyn = a_dyn * sigma
        self.k2 = k**2  # wavenumber square
        self.alpha = alpha   # abrasoin coefficient
        self.beta = beta     # adhesion coefficient
        self.tauc = tauc     # reference stress (gouge effect)
        self.etaA = eta / self.A    # coefficient
        self.etaVpl = eta * Vpl
        self.Vc = Vc
        self.Vc_ = Vc_
        self.gamma = gamma
        self.coef = self.Vc_ * self.gamma / self.A
        self.etaVpl = tr.tensor(self.etaVpl, dtype=tr.float64, device=Devices['device_1'])
        self.one = tr.tensor(1, dtype=tr.float64, device=Devices['device_1'])
        self.epsilon = 10**(-14)           # criteria
        self.epsilon2 = 10**(-1)
        self.decay = tr.ones_like(self.A, dtype=tr.float64, device=Devices['device_1'])
        #self.decay = np.sqrt(2.)
        self.gamma = tr.ones_like(self.A, dtype=tr.float64, device=Devices['device_1'])
        self.dtype = tr.float64
        self.ones = tr.ones_like(self.A, dtype=tr.float64, device=Devices['device_1'])

        # define functions in advance
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_max = tr.max
        self.tr_min = tr.min
        self.tr_abs = tr.abs
        self.tr_sqrt = tr.sqrt
        self.tr_sum = tr.sum
        self.tr_any = tr.any
        self.tr_clamp = tr.clamp
        self.tr_cat = tr.cat
        self.tr_sign = tr.sign
        self.tr_autograd = tr.autograd.grad
        
        self.i = 0


    # get masimum absolute value which contributes error of NR search
    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))
    
    def absmin(self, tensor):
        return self.tr_min(self.tr_abs(tensor))
    
    def prep(self, ini):
        self.p = self.tr_log( ini.V )
        self.calc = (( self.tauc - ini.tau_ini - ini.f + self.C * self.tr_sqrt(self.tr_sum((self.k2)*(ini.state**2), axis=1)) 
                      - self.etaVpl ) / self.A
        )
        self.ep = self.tr_exp(self.p)
        self.arg = self.tr_log(self.ep / (self.ep + self.Vc)) + self.coef / (self.ep + self.Vc_)
        self.ep_ = self.etaA * self.ep
        self.T = self.calc + self.arg + self.ep_
    
    def itr(self):
        self.p = self.tr_clamp(self.p, max=30.)
        self.ep = self.tr_exp(self.p)
        self.arg = self.tr_log(self.ep / (self.ep + self.Vc)) + self.coef / (self.ep + self.Vc_)
        self.ep_ = self.etaA * self.ep
        self.T = self.calc + self.arg + self.ep_

    # Newton-Raphson method. Note using for loop significantly slows your code
    def Halley(self, ini):
        self.prep(ini)
        self.dT1 = 1. - self.ep * ( 1. / (self.ep + self.Vc_) + self.coef / (self.ep + self.coef)**2 - self.etaA )
        self.dT2 = self.ep * (
                   (self.etaA - 1. / (self.ep + self.Vc_)) + 
                   ( self.ep + self.coef * (2. * self.ep / (self.ep + self.Vc_) - 1.) ) * self.ep / (self.ep + self.Vc_)**2
        )

        for _ in range(self.max_rep):
            #self.p -= 2. * self.T * self.dT1 / (2. * (self.dT1**2) - self.T * self.dT2)
            self.p -= self.T / self.dT1
            self.itr()
            
            self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.arg, self.ep_), dim=0))
            if self.tr_any(self.tr_abs(self.T) >= self.tol):
                self.dT1 = 1. - self.ep * ( 1. / (self.ep + self.Vc_) + self.coef / (self.ep + self.coef)**2 - self.etaA )
                self.dT2 = self.ep * (
                           (self.etaA - 1. / (self.ep + self.Vc_)) + 
                           ( self.ep + self.coef * (2. * self.ep / (self.ep + self.Vc_) - 1.) ) * self.ep / (self.ep + self.Vc_)**2
                )
                continue
            else:
                #print(_)
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