# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20250117.
"""

#This is the module for V prediction and variable update with RSF-aging law.

import sbiem_modules.Load_pickle as load
import torch as tr
import torch.optim as optimizers
import torch.optim.lr_scheduler as lr_scheduler
import time
from typing import Tuple
from packaging import version

fname = load.load('fname.pkl')['fname']
Conditions = load.load('Output{}Conditions.pkl'.format(fname))
Medium = load.load('Output{}Medium.pkl'.format(fname))
FaultParams = load.load('Output{}FaultParams.pkl'.format(fname))
Devices = load.load('Output{}Devices.pkl'.format(fname))

# Class for Newton-Raphson method.
# I tested Lambert Omega function, but it is not that fast on GPU.
class NR():
    def __init__(self, eta=Medium['eta'], Vpl = Medium['Vpl'], f0=Medium['f0'], V0=Medium['V0'],
                 a=FaultParams['a'], b=FaultParams['b'], Devices=Devices,
                 sigma=FaultParams['sigma']):
        #self.p       = tr.ones_like(a, dtype=tr.float64, device=device, requires_grad=True)   # log (V).
        self.calc    = 0.                  # Calculation unit.
        self.ep      = 0.                  # Calculation unit.
        self.T       = 0.                  # Objective funtion.
        self.dT      = 0.                  # Derivative of objective function T.
        self.max_rep = 10**8               # For Newton-Raphson.
        self.device = Devices
        
        self.eta     = eta                 # Radiation dumping coefficient.
        self.Vpl     = Vpl                 # Loading rate.
        self.f0      = f0                  # Reference friction.
        self.V0      = V0                  # Reference velocity.
        self.V0      = tr.tensor(self.V0, dtype=tr.float64, device=self.device['device_1'])
        self.Sigma   = f0 * sigma          # Reference stress.
        self.A       = a * sigma           # Direct effect coefficient.
        self.B       = b * sigma           # Shear strength coefficient.
        self.etaA    = eta / self.A        # Coefficient.
        self.etaVpl  = eta * Vpl           # Coefficient.
        self.etaVpl  = tr.tensor(self.etaVpl, dtype=tr.float64, device=self.device['device_1'])
        self.log_V0  = tr.log(self.V0)
        self.arg     = - self.log_V0 + (- self.etaVpl + self.Sigma) / self.A
        self.one     = tr.tensor(1, dtype=tr.float64, device=self.device['device_1'])
        self.epsilon = 10**(-14)           # Convergence criteria
        self.epsilon = tr.tensor(self.epsilon, dtype=tr.float64, device=self.device['device_1'])
        self.tol     = self.epsilon * (10**4)
        
        self.W = tr.zeros_like(self.A, dtype=tr.float64, device=self.device['device_1'])
        self.mask = tr.ones_like(self.A, dtype=tr.bool, device=self.device['device_1'])
        
        
        # Define functions in advance.
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_max = tr.max
        self.tr_min = tr.min
        self.tr_abs = tr.abs
        self.tr_all = tr.all
        self.tr_any = tr.any
        self.tr_clamp = tr.clamp
        self.tr_where = tr.where
        self.tr_cat = tr.cat
        
        self.time_calc = 0.
        self.time_max = 0.
        self.time_judge = 0.
        self.time_judge_ = 0.
        self.time_judge__ = 0.
        
        #self.start = tr.cuda.Event(enable_timing=True)
        #self.end = tr.cuda.Event(enable_timing=True)
        
        self.time_clac = 0.
        self.time_itr = 0.
        self.time_max = 0.
        self.time_mask = 0.
        self.time_judge = 0.
        self.time_rdc = 0.
        self.time_itr_sub  = 0.
        self.time_max_sub = 0.
        self.time_mask_sub = 0.
        self.time_rdc_sub = 0.

    def absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))

    # Newton-Raphson method. Note using for loop significantly slows your code.
    def NR_search(self, ini):
        #self.start.record()
        self.p = self.tr_log( ini.V )
        self.p_ = self.p.clone()
        self.calc = self.arg + (self.B * self.tr_log(self.V0 * ini.state / ini.L_dyn) - ini.tau_ini - ini.f) / self.A
        self.ep = self.etaA * ini.V
        self.T = self.calc + self.p + self.ep
        self.dT = self.ep + self.one
        #self.end.record()
        #tr.cuda.synchronize()
        #self.time_calc += self.start.elapsed_time(self.end)
        
        for _ in range(self.max_rep):
            if _ == 0:
                
                #self.start.record()
                self.p_ -= (self.T / self.dT)
                self.p_ = self.tr_clamp(self.p_, max=7.)
                self.ep = (self.etaA * self.tr_exp(self.p_))
                self.T = (self.calc + self.p_ + self.ep)
                #self.end.record()
                #tr.cuda.synchronize()
                #self.time_itr += self.start.elapsed_time(self.end)
        
                #self.start.record()
                self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.p_, self.ep), dim=0))
                #self.end.record()
                #tr.cuda.synchronize()
                #self.time_max += self.start.elapsed_time(self.end)
                
                #self.start.record()
                self.mask = self.tr_abs(self.T) >= self.tol
                #self.end.record()
                #tr.cuda.synchronize()
                #self.time_mask += self.start.elapsed_time(self.end)
                
                #self.start.record()
                if self.tr_any(self.mask):
                    self.end.record()
                    #tr.cuda.synchronize()
                    #self.time_judge += self.start.elapsed_time(self.end)
                    
                    #self.start.record()
                    self.id_mask = self.mask.nonzero(as_tuple=True)[0]
                    self.p = self.p_.index_select(0, self.id_mask)
                    self.etaA_ = self.etaA.index_select(0, self.id_mask)
                    self.calc = self.calc.index_select(0, self.id_mask)
                    self.T = self.T.index_select(0, self.id_mask)
                    self.dT = self.ep.index_select(0, self.id_mask) + self.one
                    
                    #self.end.record()
                    #tr.cuda.synchronize()
                    #self.time_rdc += self.start.elapsed_time(self.end)
                    continue
                else:
                    return self.ep / self.etaA
            else:
                #self.start.record()
                self.p -= (self.T / self.dT)
                self.p = self.tr_clamp(self.p, max=7.)
                self.ep = self.etaA_ * self.tr_exp(self.p)
                self.T = self.calc + self.p + self.ep
                #self.end.record()
                #tr.cuda.synchronize()
                #self.time_itr_sub += self.start.elapsed_time(self.end)
                
                #self.start.record()
                self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.p, self.ep), dim=0))
                #self.end.record()
                #tr.cuda.synchronize()
                #self.time_max_sub += self.start.elapsed_time(self.end)
                
                #self.start.record()
                self.mask = self.tr_abs(self.T) >= self.tol
                #self.end.record()
                #tr.cuda.synchronize()
                #self.time_mask_sub += self.start.elapsed_time(self.end)
                
                #self.start.record()
                self.id_mask_ = (~self.mask).nonzero(as_tuple=True)[0]
                self.id_mask__ = self.id_mask.index_select(0, self.id_mask_)
                self.p_.index_copy_(0, self.id_mask__, self.p.index_select(0, self.id_mask_))
                if self.tr_any(self.mask):
                    self.id_mask_ = self.mask.nonzero(as_tuple=True)[0]
                    self.id_mask = self.id_mask.index_select(0, self.id_mask_)
                    self.p = self.p.index_select(0, self.id_mask_)
                    self.etaA_ = self.etaA_.index_select(0, self.id_mask_)
                    self.calc = self.calc.index_select(0, self.id_mask_)
                    self.T = self.T.index_select(0, self.id_mask_)
                    self.dT = self.ep.index_select(0, self.id_mask_) + self.one
                    #self.end.record()
                    #tr.cuda.synchronize()
                    #self.time_rdc_sub += self.start.elapsed_time(self.end)
                    continue
                else:
                    return self.tr_exp(self.p_)
    
    def NR_search_straight(self, ini):
        self.p = self.tr_log( ini.V )
        self.calc = self.arg + (self.B * self.tr_log(self.V0 * ini.state / ini.L_dyn) - ini.tau_ini - ini.f) / self.A
        self.ep = self.etaA * ini.V
        self.T = self.calc + self.p + self.ep
        self.dT = self.ep + self.one
        
        for _ in range(self.max_rep):
            self.p -= (self.T / self.dT)
            self.p = self.tr_clamp(self.p, max=7.)
            self.ep = (self.etaA * self.tr_exp(self.p))
            self.T = (self.calc + self.p + self.ep)
            self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.p, self.ep), dim=0))
            self.mask = self.tr_abs(self.T) >= self.tol
            if self.tr_any(self.mask):
                self.dT = self.ep + self.one
                continue
            else:
                #print(_)
                return self.ep / self.etaA
            
    def Halley(self, ini):
        #self.T, self.dT1, self.dT2, self.p, self.calc = step_1(ini.V, ini.state, ini.L_dyn, ini.tau_ini, ini.f, 
        #                                                       self.A, self.B, self.arg, self.etaA, self.V0)
        #return step_2(self.p, self.T, self.dT1, self.dT2, self.calc, self.etaA, self.epsilon)
    
        self.p = self.tr_log( ini.V )
        self.calc = self.arg + (self.B * self.tr_log(self.V0 * ini.state / ini.L_dyn) - ini.tau_ini - ini.f) / self.A
        self.ep = self.etaA * ini.V
        self.T = self.calc + self.p + self.ep
        self.dT1 = self.ep + 1.
        self.dT2 = self.ep
        
        for _ in range(self.max_rep):
            self.p -= 2. * self.T * self.dT1 / (2. * (self.dT1**2) - self.T * self.dT2)
            self.p = self.tr_clamp(self.p, max=7.)
            self.ep = (self.etaA * self.tr_exp(self.p))
            self.T = (self.calc + self.p + self.ep)
            self.tol = self.epsilon * self.absmax(self.tr_cat((self.calc, self.p, self.ep), dim=0))
            self.mask = self.tr_abs(self.T) >= self.tol
            if self.tr_any(self.mask):
                self.dT1 = self.ep + 1.
                self.dT2 = self.ep
                continue
            else:
                #print(_)
                return self.ep / self.etaA
        
    
    def AdamW_optimize(self, ini):
        self.p = self.tr_log( ini.V )#.requires_grad_()
        self.calc = (- self.log_V0 +  
                     ( - self.etaVpl + (self.Sigma - ini.tau_ini - ini.f) + 
                      self.B * self.tr_log(self.V0 * ini.state / ini.L_dyn) ) / self.A
        )
        self.max_calc = self.absmax(self.calc)
        print(self.p.is_leaf)
        self.optimizer = optimizers.AdamW([self.p], lr=0.00001, weight_decay=0.0)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100, factor=0.1)

        for _ in range(self.max_rep):
            self.ep = self.etaA * self.tr_exp(self.p)
            self.tol = (10**(-8)) * max(self.max_calc, self.tr_max(self.p), self.tr_max(self.ep))
            self.optimizer.zero_grad()
            self.loss = (self.calc + self.p + self.ep).abs()
            self.mask = self.loss > self.tol
            print(self.mask)
            if tr.any(self.mask):#tr.any(self.loss_filtered > self.tol):
                #print(self.mask)
                self.loss_filtered = self.loss[self.mask]
                self.total_loss = self.loss_filtered.sum()
                self.total_loss.backward()
                self.optimizer.step()
                self.scheduler.step(self.total_loss)
                continue
            else:
                print('here')
                return self.ep / self.etaA
            
            #print(self.tr_max(self.loss_filtered), self.tr_min(self.loss_filtered))
            
    
    # Evaluate Lambert-Omega function.
    def lambert(self, arg):
        self.log_arg = self.tr_log(arg[arg <= 500.] + 1.)
        self.W[arg <= 500.] = 0.665 * (1. + 0.0195 * self.log_arg) * self.log_arg + 0.04
        
        self.log_arg = self.tr_log(arg[arg > 500.])
        self.W[arg > 500.] = self.tr_log(arg[arg > 500.] - 4.) - (1. - 1./self.log_arg) * self.tr_log(self.log_arg)
        
        for _ in range(self.max_rep):
            self.F = self.W + self.tr_log(self.W) - self.tr_log(arg)
            self.upt = self.F / (1. + 1./self.W) / self.W
            self.W = self.W * self.tr_exp(-self.upt)
            if self.epsilon > self.absmax(self.upt):
                return self.W
            else:
                continue
    
    def Lambert_Omega(self, ini):
        self.calc = (- self.etaVpl + self.Sigma - ini.tau_ini - ini.f + self.B*self.tr_log(self.V0 * ini.state / ini.L_dyn)) / self.A
        self.p = - self.lambert(self.etaV0A * self.tr_exp(-self.calc)) - self.calc
        print(self.tr_max(self.lambert(self.etaV0A * self.tr_exp(-self.calc))))
        return self.V0 * self.tr_exp(self.p)


if version.parse(tr.__version__) >= version.parse("2.0.0") and tr.cuda.is_available():
    @tr.compile(dynamic=False, mode='max-autotune-no-cudagraphs')
    def step_1(V: tr.Tensor, state: tr.Tensor, L_dyn: tr.Tensor, tau_ini: tr.Tensor,
            f: tr.Tensor, A: tr.Tensor, B: tr.Tensor, arg: tr.Tensor,
            etaA: tr.Tensor, V0: tr.Tensor) -> Tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor]:
        p = tr.log( V )
        calc = arg + (B * tr.log(V0 * state / L_dyn) - tau_ini - f) / A
        ep = etaA * V
        T = calc + p + ep
        dT1 = ep + 1.
        dT2 = ep
        return T, dT1, dT2, p, calc


    @tr.compile(dynamic=False, mode='max-autotune-no-cudagraphs')
    def step_2(p: tr.Tensor, T: tr.Tensor, dT1: tr.Tensor, dT2: tr.Tensor,
            calc: tr.Tensor, etaA: tr.Tensor, epsilon: tr.Tensor) -> tr.Tensor:
        for _ in range(10000):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = tr.clamp(p, max=7.)
            ep = (etaA * tr.exp(p))
            T = (calc + p + ep)
            tol = epsilon * tr.max(tr.abs(tr.stack((calc, p, ep), dim=0)))
            mask = tr.abs(T) >= tol
            if tr.any(mask):
                dT1 = ep + 1.
                dT2 = ep
                continue
            else:
                break
        return tr.exp(p)
else:
    @tr.jit.script
    def step_1(V: tr.Tensor, state: tr.Tensor, L_dyn: tr.Tensor, tau_ini: tr.Tensor,
            f: tr.Tensor, A: tr.Tensor, B: tr.Tensor, arg: tr.Tensor,
            etaA: tr.Tensor, V0: tr.Tensor) -> Tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor]:
        p = tr.log( V )
        calc = arg + (B * tr.log(V0 * state / L_dyn) - tau_ini - f) / A
        ep = etaA * V
        T = calc + p + ep
        dT1 = ep + 1.
        dT2 = ep
        return T, dT1, dT2, p, calc


    @tr.jit.script
    def step_2(p: tr.Tensor, T: tr.Tensor, dT1: tr.Tensor, dT2: tr.Tensor,
            calc: tr.Tensor, etaA: tr.Tensor, epsilon: tr.Tensor) -> tr.Tensor:
        for _ in range(10000):
            p -= 2. * T * dT1 / (2. * (dT1**2) - T * dT2)
            p = tr.clamp(p, max=7.)
            ep = (etaA * tr.exp(p))
            T = (calc + p + ep)
            tol = epsilon * tr.max(tr.max(tr.max(tr.abs(calc)), tr.max(tr.abs(p))), tr.max(tr.abs(ep)))
            mask = tr.abs(T) >= tol
            if tr.any(mask):
                dT1 = ep + 1.
                dT2 = ep
                continue
            else:
                break
        return tr.exp(p)


# Class for updating physical variables.
class Update():
    def __init__(self, TW=Conditions['TW'], L=FaultParams['L'], tc=FaultParams['tc'], 
                 tau_rate=Medium['tau_rate']):
        self.decay    = 0.       # Decay rate of state evolution.
        self.state_ss = 0.       # Steady-state state variables.
        self.L        = L        # Reference characteristic weakening distance.
        self.tc       = tc       # Characteristic time for time-weakening model.
        self.TW       = TW       # Flag for time-weakening model.
        self.tau_rate = tau_rate # Stressing rate.
        
        self.epsilon = 10**(-50)
    
        # Define functions in advance.
        self.tr_exp   = tr.exp
        self.tr_expm1 = tr.expm1

    
    # First time evolution step.
    def first(self, ini, dt):
        if self.TW:
            ini.L_dyn = self.L + ini.V * self.tc
        else:
            pass
        self.state_ss = ini.L_dyn / ini.V
        self.decay = - dt / self.state_ss
        
        self.tmp = self.tr_expm1(self.decay)
        ini.state = ini.state * (self.tmp + 1.) - self.state_ss * self.tmp
        
        ini.delta += ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini
  

    # Second time evolution step.
    def second(self, ini, dt): # Note input dt is a given full step, not a half.
        if self.TW:
            ini.L_dyn = self.L + ini.keep_V * self.tc
        else:
            pass
        
        self.decay = self.decay * 0.5
        self.tmp = self.tr_expm1(self.decay)
        ini.state = ini.state_prv * (self.tmp + 1.) - self.state_ss * self.tmp
        
        self.state_ss = ini.L_dyn / ini.keep_V
        self.decay = - (dt * 0.5) / self.state_ss
        self.tmp = self.tr_expm1(self.decay)
        ini.state = ini.state * (self.tmp + 1.) - self.state_ss * self.tmp
        
        ini.delta = ini.delta_prv + ini.V * dt
        
        ini.tau_ini += self.tau_rate * dt
        return ini

    # Take average of velocity
    def ave(self, ini):
        ini.keep_V = ini.V.clone()
        ini.V = (ini.V_prv + ini.V) * 0.5
        return ini