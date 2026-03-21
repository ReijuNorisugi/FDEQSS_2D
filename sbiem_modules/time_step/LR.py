# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260311.
"""

# This is the module for adaptive time stepper of Lapusta et al., (2000).

import torch as tr

# Class for time step evolution.
class Time_LR():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.dtmin    = Medium['dtmin']      # Minimum time step.
        self.dt0      = Medium['dt0_guess']  # Time step.
        self.safety   = Medium['safety_LR']  # Safety factor for time step evolution.
        # Criteria for detemining time step (see Lapusta et al., 2000).
        self.criteria = FaultParams['xi'] * FaultParams['L']

        self.t        = 0.    # Time
        self.t_split  = 0.    # Decomposed t.
        self.id_unit  = 0     # If t > T_unit, decompose time to prevent digit loss.
        self.T_unit   = 1.e9  # Decomposition unit.
        self.dev      = 0     # Factor to set time step as integer multiple of dtmin (see Lapusta et al., 2000).

        self.dtmax    = 1. *365*24*60*60   # Maximum time step size during one time step, 1 yr.
        self.dev_     = int(self.dtmax / self.dtmin)  # Factor for dtmax.

        # Define functions in advance.
        self.tr_min   = tr.min
        self.tr_div   = tr.div

    # Time evolution.
    def tev(self):
        self.t_split += self.dt0

        # This step prevents digit loss for a long computation.
        if self.t_split <= self.T_unit:
            pass
        else:
            self.t_split -= self.T_unit
            self.id_unit += 1

        self.t = self.t_split + self.id_unit * self.T_unit


    # Time step evolution.
    def dtev(self, V):
        self.keep_dt0 = 2 * self.dt0
        # Estimation of stable dt.
        self.dt0 = self.safety * self.tr_min(self.criteria/V).item()
        # Make dt integer-multiple of dtmin. self.dev is integer.
        self.dev = int(self.dt0 / self.dtmin)
        if self.dev == 0:
            # If dt < dtmin, make dt = dtmin.
            self.dt0 = self.dtmin
        else:
            # Otherwise.
            self.dt0 = self.dev * self.dtmin

            # If dt becomes larger than 2*dt, time step is set as 2*dt.
            if self.dt0 > self.keep_dt0:
                self.dt0 = self.keep_dt0
            else:
                pass
            # If dt > dtmax, make dt = dtmax.
            if self.dt0 <= self.dtmax:
                pass
            else:
                self.dt0 = self.dev_ * self.dtmin
