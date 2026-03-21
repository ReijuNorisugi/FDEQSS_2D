# -*- coding: utf-8 -*-
"""
Code for dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
"""

#This is the module for adaptive time stepper Lapusta et al (2000): https://doi.org/10.1029/2000JB900250

import pickle
import torch as tr

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

fname = load_pickle('fname.pkl')['fname']
Conditions = load_pickle('Output{}Conditions.pkl'.format(fname))
Medium = load_pickle('Output{}Medium.pkl'.format(fname))
FaultParams = load_pickle('Output{}FaultParams.pkl'.format(fname))

if Conditions == 'RRF':
    FaultParams['xi'] = 0.
    FaultParams['L'] = 0.

class Time_LR_mdf():
    def __init__(self, tmax=Conditions['tmax'], dtmin=Medium['dtmin'], dt0=Medium['dt0_guess'],
                 xi=FaultParams['xi'], L=FaultParams['L'], safety=Medium['safety_LR'], Vt=Medium['Vt']):
        self.t = 0.            # Initial time
        self.dt0 = dt0         # Initial time step
        self.T_unit = 10**(9)  # Unit for decomposed time to prevent digit loss when t is very large. No need to cahnge
        self.safety = safety      # Safety factor for time step evolution. No need to change
        self.tmax = tmax       # Maximum time for computation
        self.dtmin = dtmin     # Minimum time step
        self.dtmax = 1. *365*24*60*60  #Max time step size during one time step
        self.dev = 0           # Factor to set time step as integer multiple of dtmin (see Lapusta et al., 2000)
        self.dev_ = self.dtmax // self.dtmin   # Factor for dtmax
        self.criteria = xi * L
        self.Vt = Vt

        self.t_split = self.t  # Decomposed t
        self.id_unit = 0       # If t exceeds T_unit, id_unit increase by one and t is decomposed

        # define functions in advance
        self.tr_min = tr.min   
    
    # Time evolution
    def tev(self):
        self.t_split += self.dt0

        if self.t_split >= self.T_unit:
            self.t_split -= self.T_unit
            self.id_unit += 1
        else:
            pass

        self.t = self.t_split + self.id_unit * self.T_unit

    # Time step evolution, Lapusta et al., 2000
    # .item() provides scalar. Using this make dtev faster.
    def dtev(self, V):
        self.dt0 = self.safety * self.tr_min((self.criteria/V) / (1 + (V/self.Vt)**0.3)).item()
        self.dev = int(self.dt0 // self.dtmin)
        if self.dev == 0:
            self.dt0 = self.dtmin
        else:
            self.dt0 = self.dev * self.dtmin

        if self.dt0 >= self.dtmax:
            self.dt0 = self.dev_ * self.dtmin