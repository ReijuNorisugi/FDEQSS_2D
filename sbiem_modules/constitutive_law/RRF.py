# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20251009.
"""

# This is the module for V prediction and variable update with RRF law.

import sbiem_modules.utils.Load_pickle as load
import torch as tr

# Class for Nweton-Raphson method.
class Solver():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        # Main device for computation.
        self.device = Devices['device_1']
        
        # Conditions for iteration.
        self.max_rep = 10**4     # Max iteration for Newton-Raphson.
        self.epsilon = 10**(-14) # Convergence criteria.
        self.epsilon = tr.tensor(self.epsilon, dtype=tr.float64, device=self.device)

        # Constants.
        self.A = FaultParams['a'].clone() * FaultParams['sigma'].clone()
        self.C = FaultParams['c'].clone() * FaultParams['sigma'].clone()
        self.k2 = FaultParams['k'].clone()**2
        self.etaA = Medium['eta'] / self.A
        self.etaVpl = tr.tensor(Medium['eta'] * Medium['Vpl'], dtype=tr.float64, device=self.device)
        self.V0     = tr.tensor(Medium['V0'], dtype=tr.float64, device=self.device)
        self.arg    = - tr.log(self.V0) + (- self.etaVpl + FaultParams['tauc'].clone()) / self.A

        # Define functions in advance.
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

    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor), dim=0).values

    # Halley's method.
    def Halley(self, ini):
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
            
            tol = self.epsilon * (self.absmax(self.tr_stack([calc, p, ep], dim=0)))
            if self.tr_any(self.tr_abs(T) >= tol):
                dT1 = 1. + ep
                dT2 = ep
                continue
            else:
                break
        return ep / self.etaA


# Class for updating physical variables.
class Update():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.ncell = Medium['ncell']      # Number of cells.
        self.alpha = tr.reshape(FaultParams['alpha'], (self.ncell, 1)) # Abrasion coefficient, reshape is for broadcasting.
        self.beta = FaultParams['beta']                                # Adhesions coefficient.
        self.betak = FaultParams['k'] * tr.reshape(FaultParams['beta'], (self.ncell, 1))
        self.betakYbar = FaultParams['k'] * FaultParams['Ybar'] * tr.reshape(FaultParams['beta'], (self.ncell, 1))
        self.betakkYbar = FaultParams['k'] * self.betakYbar
        self.tau_rate = Medium['tau_rate']  # Stressing rate.
    
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