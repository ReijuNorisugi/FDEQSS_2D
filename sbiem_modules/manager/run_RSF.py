# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

# Run main loop.

import torch as tr

class Simulate():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        # Main device for computation.
        self.device = Devices['device_1']

        self.error    = tr.tensor(Medium['error0'], dtype=tr.float64, device=self.device).reshape(1)
        self.outerror = Conditions['outerror']
        self.Frict    = Conditions['Frict']
        self.tmax     = Conditions['tmax']
        self.V0       = Medium['V0']
        self.B        = FaultParams['b'] * FaultParams['sigma']

        self.tr_sum   = tr.sum
        self.tr_abs   = tr.abs
        self.tr_max   = tr.max
        self.tr_exp   = tr.exp
        self.tr_log10 = tr.log10

        self.synchronize = tr.cuda.synchronize

    # For Romanet & Ozawa (2022) scheme.
    def loop_RO(self, single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti):
        while ti.t < self.tmax:
            # Single-step prediction.
            single.guess(update, ker, cv, pr, ti, ti.dt0, ti.dt0, True, False, False)
            # Double step prediction.
            double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0*0.5, True, False, True)
            double.upt_prv(double) # Class variables for double step should be updated.
            double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0, True, False, True)

            # Compute relative error. I use logarithmic state variables.
            self.error = self.tr_max(self.tr_abs( self.tr_log10(single.state) - self.tr_log10(double.state) )).reshape(1)

            # Manage outputs and time steps.
            st.stepper_RO(single, double, self.error, wr, ti, ker, cv, cv_)
            if self.device != 'cpu':
                self.synchronize() # Synchronization just in case.

    # For Lapusta scheme. Now it is only for aging law.
    def loop_LR(self, single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti):
        if not self.outerror:
            self.error = 0.
            while ti.t < self.tmax:

                fac = ti.t_split + ti.dt0 + ti.id_unit * ti.T_unit
                if fac > self.tmax:
                    dev = int((self.tmax - ti.t) / ti.dtmin)
                    ti.dt0 = (dev + 1) * ti.dtmin

                single.guess(update, ker, cv, pr, ti, ti.dt0, ti.dt0, True, False, True)

                st.stepper_LR(single, double, self.error, wr, ti, ker, cv, cv_)
                if self.device != 'cpu':
                    self.synchronize() # Synchronization just in case.

        else:
            # Error monitoring with LR adaptive time step.
            while ti.t < self.tmax:
                single.guess(update, ker, cv, pr, ti, ti.dt0, ti.dt0, True, False, False)
                double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0*0.5, True, False, True)
                double.upt_prv(double)
                double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0, True, False, True)

                self.error = self.tr_max(self.tr_abs( self.tr_log10(single.state) - self.tr_log10(double.state) )).reshape(1)

                st.stepper_LR(single, double, self.error, wr, ti, ker, cv, cv_)
                if self.device != 'cpu':
                    self.synchronize() # Synchronization just in case.

    # For constant time step.
    def loop_CS(self, single, st, ker, cv, pr, update, wr, ti):
        while ti.t < self.tmax:
            single.guess(update, ker, cv, pr, ti, ti.dt0, ti.dt0, True, False, True)

            st.stepper_CS(single, wr, ti, ker)
            if self.device != 'cpu':
                self.synchronize() # Synchronization just in case.
##########################################################################################################################
