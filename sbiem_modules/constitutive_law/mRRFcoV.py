# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20251011.
"""

# This is the module for V prediction and variable update with RRF law.

import torch as tr

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
        self.etaA = Medium['eta'] / self.A
        self.etaVpl = tr.tensor(Medium['eta'] * Medium['Vpl'], dtype=tr.float64, device=self.device)
        self.V0     = tr.tensor(Medium['V0'], dtype=tr.float64, device=self.device)
        self.Vc     = FaultParams['Vc']
        self.arg    = (- self.etaVpl + FaultParams['tauc'].clone()) / self.A
        
        _ = tr.tensor(2., dtype=tr.float64, device=Devices['device_1'])
        self.k3 = (FaultParams['k']**2)*FaultParams['n_array']*tr.log(_)   # Wavenumber square times correction.

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

    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))

    # Halley's method.
    def Halley(self, ini):
        p = self.tr_log(ini.V)
        ep = ini.V
        calc = self.arg + (self.C * self.tr_sqrt(self.tr_sum(self.k3 * (ini.state**2), axis=1)) - ini.tau_ini - ini.f) / self.A
        arg_ = p - self.tr_log(ini.V + self.Vc)
        arg__ = self.etaA * ini.V
        T = calc + arg_ + arg__
        dT = 1. + arg__ - ini.V / (ini.V + self.Vc)

        for _ in range(self.max_rep):
            p -= T / dT
            p = self.tr_clamp(p, max=7.)
            ep = self.tr_exp(p)
            arg_ = p - self.tr_log(ep + self.Vc)
            arg__ = self.etaA * ep
            T = calc + arg_ + arg__
            
            tol = self.epsilon * self.absmax(self.tr_cat((calc, arg_, arg__), dim=0))
            if self.tr_any(self.tr_abs(T) >= tol):
                dT = 1. + arg__ - ep / (ep + self.Vc)
                continue
            else:
                break
        return ep



# Class for updating physical variables.
class Update():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        # Constants.
        self.ncell = Medium['ncell']
        self.alpha = tr.reshape(FaultParams['alpha'], (self.ncell, 1)) # Abrasion coefficient, reshape is for broadcasting.
        self.beta = FaultParams['beta']                                # Adhesions coefficient.
        self.betak = FaultParams['k'] * tr.reshape(self.beta, (self.ncell, 1))
        self.betakYbar = FaultParams['k'] * FaultParams['Ybar'] * tr.reshape(self.beta, (self.ncell, 1))
        self.betakkYbar = FaultParams['k'] * self.betakYbar
        self.tau_rate = Medium['tau_rate'] # Stressing rate.
    
        # Define functions in advance.
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1
        self.tr_reshape = tr.reshape

    # First time evolution step.
    def first(self, ini, dt):
        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt

        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.delta += ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini.delta, ini.state, ini.tau_ini

  
    # Second time evolution step, 2nd-order accuracy.
    def second(self, ini, dt): # Note input dt is full time step, not a half.
        self.decay = self.decay * 0.5
        ini.state = ini.state_prv * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)

        self.state_ss = self.betakYbar / (self.alpha*self.tr_reshape(ini.keep_V, (self.ncell, 1))+self.betak)
        self.decay = - (self.betakkYbar / self.state_ss) * dt * 0.5
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.delta = ini.delta_prv + ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini.delta, ini.state, ini.tau_ini


    # Take average of velocity.
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini.keep_V, ini.V