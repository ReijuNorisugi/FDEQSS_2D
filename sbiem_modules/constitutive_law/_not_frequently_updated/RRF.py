# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
"""

#This is the module for V prediction and variable update with RRF law.

import torch as tr
import numpy as np
import pickle
from typing import Tuple
from packaging import version

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
                 sigma=FaultParams['sigma'], device=Devices['device_1']):
        self.max_rep = 10**4     # Max iteration for Newton-Raphson.
        self.epsilon = 10**(-14) # Convergence criteria.
        self.atol = 10**(-20)    # Absolute error.
    
        self.A = a * sigma
        self.C = c * sigma
        self.k2 = k**2
        self.etaA = eta / self.A
        self.V0 = tr.tensor(V0, dtype=tr.float64, device=device)
        self.etaVpl = tr.tensor(eta * Vpl, dtype=tr.float64, device=device)
        self.arg = - tr.log(self.V0) + (- self.etaVpl + tauc) / self.A

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
        return self.tr_max(self.tr_abs(tensor), dim=0).values


    # Newton-Raphson method. Note using for loop significantly slows your code
    def Halley(self, ini):
        #self.T, self.dT1, self.dT2, self.calc, self.p = step_1(ini.V, self.tauc, ini.tau_ini, ini.f, self.C,
        #                                                       ini.state, self.k2, self.etaVpl, self.A, self.log_V0, self.etaA)
        #return step_2(self.p, self.T, self.dT1, self.dT2, self.calc, self.etaA, self.epsilon)
        
        p = self.tr_log( ini.V )
        calc = self.arg + ( self.C * self.tr_sqrt(self.tr_sum((self.k2)*(ini.state**2), dim=1)) - ini.tau_ini - ini.f ) / self.A
        ep = self.etaA * ini.V
        T = calc + p + ep
        dT1 = 1. + ep
        dT2 = ep
        
        for _ in range(self.max_rep):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = self.tr_clamp(p, max=7.)
            ep = self.etaA * self.tr_exp(p)
            T = calc + p + ep
            
            tol = self.epsilon * (self.absmax(self.tr_stack([calc, p, ep], dim=0)) + self.atol)
            if self.tr_any(self.tr_abs(T) >= tol):
                dT1 = 1. + ep
                dT2 = ep
                continue
            else:
                break
        return ep / self.etaA

if version.parse(tr.__version__) >= version.parse("2.0.0"):
    @tr.compile(dynamic=False, mode='max-autotune-no-cudagraphs')
    def step_1(V: tr.Tensor, tauc: tr.Tensor, tau_ini: tr.Tensor,
            f: tr.Tensor, C: tr.Tensor, state: tr.Tensor, 
            k2: tr.Tensor, etaVpl: tr.Tensor, A: tr.Tensor, 
            log_V0: tr.Tensor, etaA: tr.Tensor) -> Tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor]:
        p = tr.log(V)
        calc = (( tauc - tau_ini - f + C * tr.sqrt(tr.sum((k2)*(state**2), dim=1)) 
                - etaVpl ) / A - log_V0
        )
        ep = etaA * tr.exp(p)
        T = calc + p + ep
        dT1 = 1. + ep
        dT2 = ep
        return T, dT1, dT2, calc, p

    @tr.compile(dynamic=False, mode='max-autotune-no-cudagraphs')
    def step_2(p: tr.Tensor, T: tr.Tensor, dT1: tr.Tensor, dT2: tr.Tensor,
            calc: tr.Tensor, etaA: tr.Tensor, epsilon: tr.Tensor) -> tr.Tensor:
        for _ in range(10000):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = tr.clamp(p, max=7.)
            ep = etaA * tr.exp(p)
            T = calc + p + ep
            
            tol = epsilon * tr.max(tr.max(tr.max(tr.abs(calc)), tr.max(tr.abs(p))), tr.max(tr.abs(ep)))
            if tr.any(tr.abs(T) >= tol):
                dT1 = 1. + ep
                dT2 = ep
                continue
            else:
                break
        return tr.exp(p)
else:
    @tr.jit.script
    def step_1(V: tr.Tensor, tauc: tr.Tensor, tau_ini: tr.Tensor,
            f: tr.Tensor, C: tr.Tensor, state: tr.Tensor, 
            k2: tr.Tensor, etaVpl: tr.Tensor, A: tr.Tensor, 
            log_V0: tr.Tensor, etaA: tr.Tensor) -> Tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor]:
        p = tr.log(V)
        calc = (( tauc - tau_ini - f + C * tr.sqrt(tr.sum((k2)*(state**2), dim=1)) 
                - etaVpl ) / A - log_V0
        )
        ep = etaA * tr.exp(p)
        T = calc + p + ep
        dT1 = 1. + ep
        dT2 = ep
        return T, dT1, dT2, calc, p

    @tr.jit.script
    def step_2(p: tr.Tensor, T: tr.Tensor, dT1: tr.Tensor, dT2: tr.Tensor,
            calc: tr.Tensor, etaA: tr.Tensor, epsilon: tr.Tensor) -> tr.Tensor:
        for _ in range(10000):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = tr.clamp(p, max=7.)
            ep = etaA * tr.exp(p)
            T = calc + p + ep
            
            tol = epsilon * tr.max(tr.max(tr.max(tr.abs(calc)), tr.max(tr.abs(p))), tr.max(tr.abs(ep)))
            if tr.any(tr.abs(T) >= tol):
                dT1 = 1. + ep
                dT2 = ep
                continue
            else:
                break
        return tr.exp(p)
    


# Class for updating physical variables.
class Update():
    def __init__(self, k=FaultParams['k'], alpha=FaultParams['alpha'], beta=FaultParams['beta'], 
                 Ybar=FaultParams['Ybar'], ncell=Medium['ncell'], tau_rate=Medium['tau_rate'], device=Devices):
        self.decay = 0.                                           # Decay rate for state variable evolution.
        self.state_ss = 0.                                        # Steady-state state variable.
        self.alpha = tr.reshape(alpha, (ncell, 1))                # Abrasion coefficient, reshape is for broadcasting.
        self.beta = beta                                          # Adhesions coefficient.
        self.betak = k * tr.reshape(beta, (ncell, 1))             # Coefficient.
        self.betakYbar = k * Ybar * tr.reshape(beta, (ncell, 1))  # Coefficient.
        self.betakkYbar = k * self.betakYbar                      # Coefficient.
        self.ncell = ncell                                        # Number of cells.
        self.tau_rate = tau_rate                                  # Stressing rate.
        self.tmp = tr.zeros_like(self.betakYbar, dtype=tr.float64, device=device['device_1'])
    
        # Define functions in advance.
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1
        self.tr_reshape = tr.reshape


    # First time evolution step.
    def first(self, ini, dt):
        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt

        tmp = self.tr_expm1(self.decay)
        ini.state = ini.state * (tmp + 1.) - self.state_ss * tmp
        
        ini.delta += ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini

  
    # Second time evolution step, 2nd-order accuracy.
    def second(self, ini, dt): # Note input dt is full time step, not a half.
        self.decay = self.decay * 0.5
        tmp = self.tr_expm1(self.decay)
        ini.state = ini.state_prv * (tmp + 1.) - self.state_ss * tmp

        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.keep_V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt * 0.5
        tmp = self.tr_expm1(self.decay)
        ini.state = ini.state * (tmp + 1.) - self.state_ss * tmp
        
        ini.delta = ini.delta_prv + ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini


    # Take average of velocity.
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini