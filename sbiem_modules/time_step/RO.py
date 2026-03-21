# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260311.
"""

# This is the module for adaptive time stepper Romanet & Ozawa (2022).

import torch as tr

# Class for time step evolution.
class Time_RO():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.dt0 = Medium['dt0_guess']   # Time step.
        self.safety = 0.9                # Safety factor for time step evolution. No need to change.
        self.error0 = Medium['error0']   # Error criteria for adaptive step control.
        self.keep_dt0 = self.dt0         # If dt0 becomes twice larger than previous dt0, keep it as 2*dt0.
        self.dtmin = Medium['dtmin']     # Minimum time step.

        self.dtmax = 1. *365*24*60*60 # Max time step size during one time step. 1yr.
        self.T_unit = 1.e9            # Unit for decomposed time to prevent digit loss if t is very large.
        self.dev = 0                  # Factor to set time step as integer multiple of dtmin.
        self.dev_ = int(self.dtmax / self.dtmin)  # Factor for dtmax.

        self.t = 0.            # Initial time.
        self.t_split = self.t  # Decomposed t.
        self.id_unit = 0       # When t exceeds T_unit, id_unit increase by one and t is decomposed.

        self.tr_nan = tr.isnan

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


    # Time step evolution by Romanet & Ozawa (2022).
    # This is the local error control. But it is more stable than global error control.
    def dtev(self, error):
        if self.tr_nan(error):
            # If prediction output non, make dt0 = dt0 / 2.
            self.dt0 = self.dt0 * 0.5
            if self.dt0 <= self.dtmin:
                self.dt0 = self.dtmin
            return
        else:
            # Otherwise estimate next dt.
            pass

        self.keep_dt0 = 2. * self.dt0

        # Estimate maximum dt to satisfy the desired truncation error \epsilon_0.
        self.dt0 = (self.safety * ( ( self.error0 / error )**(1./3.) ) * self.dt0).item()
        # Make dt integer-multiple of dtmin.
        self.dev = int(self.dt0 / self.dtmin)

        if self.dev == 0:
            # If dt < dtmin, make dt = dtmin.
            self.dt0 = self.dtmin
        else:
            self.dt0 = self.dev * self.dtmin

            if self.dt0 > self.keep_dt0:
                # If dt0 > 2*dt0, make dt = 2*dt0.
                self.dt0 = self.keep_dt0
            else:
                pass

            if self.dt0 <= self.dtmax:
                pass
            else:
                # If dt0 > dtmax, make dt = dtmax.
                self.dt0 = self.dev_ * self.dtmin
