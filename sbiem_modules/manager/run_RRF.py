# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260309.
"""

# Run main loop.

import torch as tr

class Simulate():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):        
        # Main device for computation.
        self.device = Devices['device_1']
        
        self.error    = tr.tensor(Medium['error0'], dtype=tr.float64, device=self.device).reshape(1)
        self.tmax     = Conditions['tmax']
        
        self.tr_sum   = tr.sum
        self.tr_abs   = tr.abs
        self.tr_max   = tr.max
        self.tr_log10 = tr.log10
        
        self.synchronize = tr.cuda.synchronize
    
    # For Romanet & Ozawa method
    def loop_RO(self, single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti):
        while ti.t < self.tmax:
            single.guess(update, ker, cv, pr, ti, ti.dt0, ti.dt0, True, False, False)
            double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0*0.5, True, False, True)
            double.upt_prv(double)
            double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0, True, False, True)
            self.error = self.tr_max(self.tr_abs(self.tr_log10(single.state) - self.tr_log10(double.state))).reshape(1)
            st.stepper_RO(single, double, self.error, wr, ti, ker, cv, cv_)
            if self.device != 'cpu':
                self.synchronize() # Synchronization just in case.
    
    
    def loop_RO_cvtest(self, single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti):
        while ti.t < self.tmax:
            
            fac = ti.t_split + ti.dt0 + ti.id_unit * ti.T_unit
            if fac > self.tmax:
                dev = int((self.tmax - ti.t) / ti.dtmin)
                ti.dt0 = (dev + 1) * ti.dtmin
            
            print(tr.max(double.V), ':::', ti.t)
            
            single.guess(update, ker, cv, pr, ti, ti.dt0, ti.dt0, True, False, False)
            double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0*0.5, True, False, True)
            double.upt_prv(double)
            double.guess(update_, ker, cv_, pr_, ti, ti.dt0*0.5, ti.dt0, True, False, True)
            self.error = self.tr_max(self.tr_abs(self.tr_log10(single.state) - self.tr_log10(double.state))).reshape(1)
            st.stepper_RO(single, double, self.error, wr, ti, ker, cv, cv_)
            if self.device != 'cpu':
                self.synchronize() # Synchronization just in case.
##########################################################################################################################