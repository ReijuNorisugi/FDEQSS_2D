# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
"""

#This is the module for V prediction and variable update with RSF-aging law.

import torch as tr
import pickle

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
  

fname = load_pickle('fname.pkl')['fname']
Conditions = load_pickle('Output{}Conditions.pkl'.format(fname))
Medium = load_pickle('Output{}Medium.pkl'.format(fname))
FaultParams = load_pickle('Output{}FaultParams.pkl'.format(fname))


class NR():
    def __init__(self, eta=Medium['eta'], Vpl = Medium['Vpl'], f0=Medium['f0'], V0=Medium['V0'],
                 a=FaultParams['a'], b=FaultParams['b'], an=FaultParams['an'], bn=FaultParams['bn'], cn=FaultParams['cn'],
                 sigma=FaultParams['sigma'], tau_ini=FaultParams['tau_ini']):
        self.p = 0.                        # log (V/V0)
        self.calc = 0.                     # calculation unit
        self.ep = 0.                       # calculation unit
        self.T = 0.                        # objective funtion T
        self.dT = 0.                       # derivative of objective function T
        self.tol = 0.                      # torelance for NR search
    
        self.eta = eta                     # radiation dumping coefficient
        self.Vpl = Vpl                     # loading rate
        self.f0 = f0                       # reference friction
        self.V0 = V0                       # reference velocity
        self.sigma = sigma                 # normal stress
        self.Bn = bn * sigma               # shear strength coefficient
        self.An = an * sigma               # direct effect coeeficient in Nagata law
        self._cn = 1 / (1 + cn)            # stress weakening coefficient
        self.tau_ini = tau_ini             # tau0
        self.etaV0A = eta * V0 / self.An    # coefficient
        self.Theta = 0.                    # for NR search
        self.epsilon = 10**(-14)           # criteria
        self.dtype = tr.float64

        # define functions in advance
        self.tr_log = tr.log
        self.tr_exp = tr.exp
        self.tr_max = tr.max
        self.tr_abs = tr.abs
        self.tr_tensor = tr.tensor


    # get maximum absolute value which contribute error in NR search
    def get_absmax(self, tensor):
        return self.tr_max(self.tr_abs(tensor))


    # Newton-Raphson method, note using for loop significantly slows your code
    def NR_search(self, ini):
        self.Theta = self.Bn * self.tr_log(self.V0 * ini.state / ini.L_dyn)
        self.p = self.tr_log( ini.V / self.V0 )
        self.calc = ( ( ( self.f0*self.sigma + self.Theta - self.tau_ini - ini.f ) / self.An ) -
                      (self.eta * self.Vpl / self.An)
        )
        self.ep = self.etaV0A * self.tr_exp(self.p)
        self.T = self.calc + self._cn * self.p + self.ep
        self.dT = 1. + self.ep
        self.tol = self.epsilon * self.get_absmax((self.tr_tensor([self.get_absmax(self.calc), 
                                                                   self.get_absmax(self._cn * self.p), 
                                                                   self.get_absmax(self.ep)], dtype=self.dtype)))
        while self.tr_max(self.tr_abs(self.T)) >= self.tol:
            self.p -= (self.T / self.dT)
            self.T = self.calc + self._cn * self.p + self.etaV0A * self.tr_exp(self.p)
            self.dT = 1. + self.etaV0A * self.tr_exp(self.p)
        return self.V0 * self.tr_exp(self.p)
  

# Class for updating physical variables
class Update():
    def __init__(self, TW=Conditions['TW'], L=FaultParams['L'], tc=FaultParams['tc'], cn=FaultParams['cn']):
        self.decay = 0.           # decay rate for state evolution
        self.state_ss = 0.        # steady-state state variables
        self.L = L                # characteristic weakening distance
        self._cn = 1 / (1 + cn)   # stress weakening coefficient
        self.tc = tc              # characteristic time for time-weakening model
        self.TW = TW              # flag for time-weakening model
    
        # define functions in advance
        self.tr_exp = tr.exp
        self.tr_expm1 = tr.expm1

    
    # first time evolution step
    def first(self, ini, dt):
        if self.TW:
            ini.L_dyn = self.L + ini.V * self.tc
        else:
            pass
        self.state_ss = ini.L_dyn / ini.V
        self.decay = - self._cn * (1/self.state_ss) * dt

        ini.delta += ini.V * dt
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        ini.Dell += ini.d_Dell * dt
        return ini
  
  
    # second time evolution step
    def second(self, bef, ini, dt): # Note input dt is full time step, not a half.
        if self.TW:
            ini.L_dyn = self.L + ini.keep_V * self.tc
        else:
            pass
        ini.delta = bef.delta + ini.V * dt
        ini.state = bef.state * self.tr_exp(self.decay*0.5) - self.state_ss * self.tr_expm1(self.decay*0.5)

        self.state_ss = ini.L_dyn / ini.keep_V
        self.decay = - self._cn * (1/self.state_ss) * dt * 0.5
        ini.state = ini.state * self.tr_exp(self.decay) - self.state_ss * self.tr_expm1(self.decay)
        ini.Dell = bef.Dell + ini.keep_dDell * dt
        return ini

    # take average of velocity
    def ave(self, bef, ini):
        ini.keep_V = ini.V
        ini.V = (bef.V + ini.V) * 0.5
        ini.keep_dDell = (bef.d_Dell + ini.d_Dell) * 0.5
        return ini