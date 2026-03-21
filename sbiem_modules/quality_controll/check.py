# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260309.
"""

# This is the module for quality controll.
# If you choose insufficient conditions, simulation stops.

import sys

class Check():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.mirror = Conditions['mirror']
        self.qd = Conditions['qd']
        self.mode = Conditions['mode']
        self.sparse = Conditions['sparse']
        self.snap_EQ = Conditions['snap_EQ']
        self.dense = Conditions['dense']
        self.Frict = Conditions['Frict']
        self.Stepper = Conditions['Stepper']
        self.tmax = Conditions['tmax']
        self.outerror = Conditions['outerror']
        
        self.lam = Medium['lam']
        self.nperi = Medium['nperi']
        self.Tw = Medium['Tw']
        self.dtmin = Medium['dtmin']
        
        self.Devices = Devices

    def exe(self):
        if self.mirror and self.mode == 'IV':
            print(' ')
            print('###########################################################')
            print(' ')
            print('  Choose different rupture mode or compute without mirror  ')
            print(' ')
            print('###########################################################')
            print(' ')
            sys.exit()