# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

# This is the module for V prediction and variable update with RSF-aging law.

import torch as tr

# Class for iterative solvers of the non-linear equation.
class Solver():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        device = Devices['device_1']
        
        # Conditions for iteration.
        self.max_rep = int(1.e4)      # Max iteration of solvers.
        self.epsilon = 1.e-14  # Convergence criteria.
        self.epsilon = tr.tensor(self.epsilon, dtype=tr.float64, device=device)
        
        # Constants.
        self.A      = FaultParams['a'].clone() * FaultParams['sigma'].clone()
        self.B      = FaultParams['b'].clone() * FaultParams['sigma'].clone()
        self.L      = FaultParams['L'].clone()
        self.etaA   = Medium['eta'] / self.A
        self.etaVpl = tr.tensor(Medium['eta'] * Medium['Vpl'], dtype=tr.float64, device=device)
        self.V0     = tr.tensor(Medium['V0'], dtype=tr.float64, device=device)
        self.arg    = - tr.log(self.V0) + (- self.etaVpl + Medium['f0'] * FaultParams['sigma'].clone()) / self.A
        
        self.W = tr.zeros_like(self.A, dtype=tr.float64, device=device)
        
        # Define functions in advance.
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_abs = tr.abs
        self.tr_all = tr.all
        self.tr_any = tr.any
        self.tr_clamp = tr.clamp
        self.tr_where = tr.where
        self.tr_max = tr.maximum
    
    # Halley method. This will be faster.
    def Halley(self, ini):
        p = self.tr_log( ini.V )
        calc = self.arg + (self.B * self.tr_log(self.V0 * ini.state / self.L) - ini.tau_ini - ini.f) / self.A
        ep = self.etaA * ini.V
        T = calc + p + ep
        dT1 = ep + 1.
        dT2 = ep
        
        for _ in range(self.max_rep):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = self.tr_clamp(p, max=7.)
            ep = (self.etaA * self.tr_exp(p))
            T = (calc + p + ep)
            tol = self.epsilon * self.tr_max(self.tr_max(calc.abs().max(), p.abs().max()), ep.abs().max())
            mask = self.tr_abs(T) >= tol
            if self.tr_any(mask):
                dT1 = ep + 1.
                dT2 = ep
                continue
            else:
                break
        return ep / self.etaA


# Class for updating physical variables.
class Update():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        # Constants.
        self.L        = FaultParams['L']   # Reference characteristic weakening distance.
        self.tau_rate = Medium['tau_rate'] # Stressing rate.
        
        # Define functions in advance.
        self.tr_exp   = tr.exp
        self.tr_expm1 = tr.expm1


    # First time step evolution.
    def first(self, ini, dt):
        self.state_ss = self.L / ini.V
        self.decay = - dt / self.state_ss
        
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.delta += ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini
  

    # Second time step evolution.
    def second(self, ini, dt): # Note input dt is a given full step, not a half.
        self.decay = self.decay * 0.5
        ini.state = ini.state_prv * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        self.state_ss = self.L / ini.keep_V
        self.decay = - (dt * 0.5) / self.state_ss
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.delta = ini.delta_prv + ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini


    # Take average of velocity.
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini