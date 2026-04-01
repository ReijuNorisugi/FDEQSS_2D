# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

# This is the module for V prediction and variable update with regularized RSF-aging law.

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
        self.a   = FaultParams['a'].clone()
        self.b   = FaultParams['b'].clone()
        self.L   = FaultParams['L'].clone()
        self.A   = FaultParams['sigma'].clone() * self.a
        self.f0  = tr.tensor(Medium['f0'], dtype=tr.float64, device=device)
        self.V0  = tr.tensor(Medium['V0'], dtype=tr.float64, device=device)
        
        self.etaA = Medium['eta'] * self.V0 / self.A
        self.arg  = tr.tensor(Medium['eta'] * Medium['Vpl'], dtype=tr.float64, device=device) / self.A
        
        self.W = tr.zeros_like(self.A, dtype=tr.float64, device=device)
        
        # Define functions in advance.
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_abs = tr.abs
        self.tr_all = tr.all
        self.tr_any = tr.any
        self.tr_clamp = tr.clamp
        self.tr_where = tr.where
        self.tr_asinh = tr.asinh
        self.tr_sqrt = tr.sqrt
        self.tr_max = tr.maximum
    
    # Halley method.
    def Halley(self, ini):
        p = self.tr_log( ini.V / self.V0 )
        calc = 0.5 * self.tr_exp( (self.f0 + self.b * self.tr_log(self.V0 * ini.state / self.L) ) / self.a )
        F = (ini.tau_ini + ini.f) / self.A + self.arg
        ep = self.etaA * self.tr_exp(p)
        arg = self.tr_exp(p) * calc
        T = self.tr_asinh(arg) + ep - F
        dT1 = arg / self.tr_sqrt(1. + arg**2) + ep
        dT2 = - arg / (1. + arg**2)**(3./2.) + ep
        
        for _ in range(self.max_rep):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = self.tr_clamp(p, max=26.)
            ep = self.etaA * self.tr_exp(p)
            arg = self.tr_exp(p) * calc
            arg_ = self.tr_asinh(arg)
            T = arg_ + ep - F
            tol = self.epsilon * self.tr_max(self.tr_max(arg_.abs().max(), F.abs().max()), ep.abs().max())
            mask = self.tr_abs(T) >= tol
            if self.tr_any(mask):
                dT1 = arg / self.tr_sqrt(1. + arg**2) + ep
                dT2 = - arg / (1. + arg**2)**(3./2.) + ep
                continue
            else:
                break
        return self.V0 * self.tr_exp(p)


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
        return ini.delta, ini.state, ini.tau_ini


    # Second time step evolution.
    def second(self, ini, dt): # Note input dt is a given full step, not a half.
        self.decay = self.decay * 0.5
        ini.state = ini.state_prv * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        self.state_ss = self.L / ini.keep_V
        self.decay = - (dt * 0.5) / self.state_ss
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        
        ini.delta = ini.delta_prv + ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini.delta, ini.state, ini.tau_ini


    # Take average of velocity.
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini.keep_V, ini.V