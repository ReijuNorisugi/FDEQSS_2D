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
    def __init__(self, eta=Medium['eta'], Vpl = Medium['Vpl'], V0=Medium['V0'], delta0=Medium['delta0'], ncell=Medium['ncell'],
                 a=FaultParams['a'], c=FaultParams['c'], tauc=FaultParams['tauc'], f0=FaultParams['f0'],
                 k=FaultParams['k'], alpha=FaultParams['alpha'], beta=FaultParams['beta'],
                 sigma=FaultParams['sigma'], tau_ini=FaultParams['tau_ini']):
        self.p = 0.          # log(V/V0)
        self.calc = 0.       # calculation unit
        self.ep = 0.         # calculation unit
        self.T = 0.          # Objective funtion T
        self.dT = 0.         # derivative of objective function T
        self.tol = 0.        # torlelance
    
        self.eta = eta       # radiation dumping coefficient
        self.Vpl = Vpl       # loading late
        self.f0 = f0         # reference friction
        self.V0 = V0         # reference velocity
        self.delta0 = delta0 / k # initial discrepancy
        self.ncell = ncell   # number of cells
        self.sigma = sigma   # normal stress
        self.A = a * sigma   # direct effect coefficient
        self.C = c * sigma   # shear strength coefficient
        self.k_half = 0.5 * k
        self.k2 = k**2       # wavenumber square
        self.alpha = alpha   # abrasoin coefficient
        self.beta = beta     # adhesion coefficient
        self.tauc = tauc     # reference stress (gouge effect)
        self.tau_ini = tau_ini             # tau0
        self.etaV0A = eta * V0 / self.A    # coefficient
        self.epsilon = 10**(-14)           # criteria
        self.dtype = tr.float64

        # define functions in advance
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_max = tr.max
        self.tr_abs = tr.abs
        self.tr_tensor = tr.tensor
        self.tr_sqrt = tr.sqrt
        self.tr_sum = tr.sum
        self.tr_sin = tr.sin
        self.tr_reshape = tr.reshape


    # get masimum absolute value which contributes error of NR search
    def get_absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))


    # Newton-Raphson method. Note using for loop significantly slows your code
    def NR_search(self, ini):
        self.p = self.tr_log( ini.V / self.V0 )
        self.calc = ( ( ( self.tauc - self.tau_ini - ini.f ) / self.A ) + 
                      (self.C/self.A) * self.tr_sqrt(self.tr_sum((self.k2)*(ini.state**2)*
                                                                 ((self.tr_sin(self.k_half * (self.tr_reshape(ini.delta, (self.ncell, 1)) + self.delta0)))**2), axis=1)) -
                      ( self.eta * self.Vpl / self.A )
        )
        self.ep = self.etaV0A * self.tr_exp(self.p)
        self.T = self.calc + self.p + self.ep
        self.dT = 1 + self.ep
        self.tol = self.epsilon * self.get_absmax((self.tr_tensor([self.get_absmax(self.calc), self.get_absmax(self.p), self.get_absmax(self.ep)], dtype=self.dtype)))
        while self.get_absmax(self.T) >= self.tol:
            self.p -= (self.T / self.dT)
            self.T = self.calc + self.p + self.etaV0A * self.tr_exp(self.p)
            self.dT = 1 + self.etaV0A * self.tr_exp(self.p)
        return self.V0 * self.tr_exp(self.p)
  


# Class for updating physical variables
class Update():
    def __init__(self, k=FaultParams['k'], alpha=FaultParams['alpha'], beta=FaultParams['beta'], Ybar=FaultParams['Ybar'], ncell=Medium['ncell']):
        self.decay = 0.                                           # decay rate for state variable evolution
        self.state_ss = 0.                                        # steady-state state variable
        self.alpha = tr.reshape(alpha, (ncell, 1))                # abrasion coefficient, reshape is for broadcasting
        self.beta = beta                                          # adhesions coefficient
        self.betak = k * tr.reshape(beta, (ncell, 1))             # coefficient
        self.betakYbar = k * Ybar * tr.reshape(beta, (ncell, 1))  # coefficient
        self.betakkYbar = k * self.betakYbar                      # coefficient
        self.ncell = ncell                                        # number of cells
    
        # define functions in advance
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1
        self.tr_reshape = tr.reshape


    # first time evolution step
    def first(self, ini, dt):
        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt

        ini.delta += ini.V * dt
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        return ini
  
  
    # second time evolution step, 2nd-order accuracy
    def second(self, bef, ini, dt): # Note input dt is full time step, not a half.
        ini.delta = bef.delta + ini.V * dt
        self.decay = self.decay * 0.5
        ini.state = bef.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)

        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.keep_V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt * 0.5
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        return ini


    # take average of velocity
    def ave(self, bef, ini):
        ini.keep_V = ini.V
        ini.V = (bef.V + ini.V) * 0.5
        return ini