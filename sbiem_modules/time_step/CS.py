# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260311.
"""

# This is the module for constant time stepper.
# This is mainly used for single dynsmic rupture simulation with slip-weakening law.

class Time_CS():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.t = 0.                # Simulation time.
        self.dt0 = Medium['dtmin'] # Constatnt time step.
        self.T_unit = 1.e9      # Unit for decomposed time to prevent digit loss when t is very large.

        self.t_split = self.t  # Decomposed t.
        self.id_unit = 0       # If t exceeds T_unit, id_unit increase by one and t is decomposed to prevent digit loss.

    # Time evolution.
    def tev(self):
        self.t_split += self.dt0

        if self.t_split >= self.T_unit:
            self.t_split -= self.T_unit
            self.id_unit += 1
        else:
            pass

        self.t = self.t_split + self.id_unit * self.T_unit
