# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20250928.
"""

# This is the module which plots the initial conditions.

import torch as tr
import numpy as np

class Vis():
    def __init__(self, fname, Devices, Conditions, Medium, FaultParams, FieldVariables, id_station):
        self.fname = fname
        self.id_station = id_station
        self.xp = tr.tensor(Medium['xp'], dtype=tr.float64)
        self.a  = FaultParams['a']
        self.b  = FaultParams['b']
        self.L  = FaultParams['L']
        self.sigma = FaultParams['sigma']
        self.A = self.a * self.sigma
        self.B = self.b * self.sigma
        self.V = FieldVariables['V']
        self.delta = FieldVariables['delta']
        self.state = FieldVariables['state']
        self.tau = FieldVariables['tau']
        self.Frict = Conditions['Frict']
        self.stations_uni = Conditions['stations_uni']
        self.stations = Conditions['stations']
        self.ncell = Medium['ncell']
        self.V0 = Medium['V0']
        self.f0 = Medium['f0']

    def figs(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axhline(0, color='dimgray', linestyle='dashed', lw=2.)
        ax.plot(self.xp*1.e-3, self.a-self.b, color='black', lw=0.5, label=r'$a-b$', marker='o', markersize=3)
        ax.plot(self.xp*1.e-3, self.a, color='black', linestyle='dotted', lw=3, label=r'$a$')
        ax.set_xlabel('Distance along fault (km)', fontsize=10)
        ax.set_ylabel(r'$a-b$, $a$', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper right', fontsize=10, framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Fault_a_b.png'.format(self.fname), dpi=600)
        plt.clf()
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axhline(0, color='dimgray', linestyle='dashed', lw=3.)
        ax.plot(self.xp*1.e-3, self.A*1.e-6, linestyle='dotted', color='black', lw=3, label=r'$A$')
        ax.plot(self.xp*1.e-3, (self.A-self.B)*1.e-6, color='black', lw=3, label=r'$A-B$')
        ax.set_xlabel('Distance along fault (km)', fontsize=10)
        ax.set_ylabel(r'$A-B$, $A$ (MPa)', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper right', fontsize=10, framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Fault_AB.png'.format(self.fname), dpi=600)
        plt.clf()
        plt.close()
            
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(self.xp*1.e-3, self.L, color='black', lw=3)
        ax.set_xlabel('Distance along fault (km)', fontsize=10)
        ax.set_ylabel(r'$L$ (m)', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Fault_L.png'.format(self.fname), dpi=600)
        plt.clf()
        plt.close()
    
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(self.xp*1.e-3, self.V, color='blue', lw=3)
        ax1.set_ylabel('Slip rate (m/s)', fontsize=10)
        ax1.set_yscale('log')
        plt.grid()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.xp*1.e-3, self.state, color='blue', lw=3)
        ax2.set_xlabel('Distance along fault (km)', fontsize=10)
        ax2.set_ylabel('State variable (sec)', fontsize=10)
        ax2.set_yscale('log')
        plt.grid()
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(self.xp*1.e-3, self.delta, color='blue', lw=3)
        ax3.set_ylabel('Slip (m)', fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Initial_condistions.png'.format(self.fname), dpi=600)
        plt.clf()
        plt.close()
            
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(self.xp*1.e-3, self.tau[:len(self.xp)]*(10**(-6)), color='black', lw=2, linestyle='solid')
        ax.set_xlabel('Dstance along fault (km)', fontsize=10)
        ax.set_ylabel('Shear stress (MPa)', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Initial_stress.png'.format(self.fname), dpi=600)
        plt.clf()
        plt.close()

        if self.stations_uni:
            ch_id = (self.ncell*np.arange(0, self.stations+1, 1)/(self.stations)).astype(np.int64)
            ch_id[1:] = ch_id[1:] - 1
        else:
            ch_id = self.stations

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.array([self.xp[0], self.xp[-1]])*1.e-3, np.zeros(2, dtype=np.float64), color='black', lw=1)
        ax.scatter(self.xp[ch_id]*1.e-3, np.zeros_like(self.xp[ch_id])*1.e-3, facecolor='None', edgecolor='black', s=20)
        ax.set_xlabel('Distance along fault (km)', fontsize=10)
        ax.set_title('Station location', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Locations.png'.format(self.fname), dpi=600)
        plt.clf()
        plt.close()
        

        stat = 3
        fig, ax = plt.subplots(figsize=(5, 5))
        V_plot = tr.logspace(-12, 0, 100)
        if self.Frict == 'reg_RSF_AG':
            tau_ss = (self.A[stat] * 
                      tr.asinh((0.5*V_plot/self.V0) * tr.exp( 
                          (self.f0 - self.b[stat]*tr.log(V_plot/self.V0))/self.a[stat] ) )
            )
        else:
            tau_ss = (self.f0 * self.sigma[stat] + 
                      (self.A[stat] - self.B[stat]) * tr.log(V_plot / self.V0)
                      )
        ax.plot(np.log10(V_plot), tau_ss*1.e-6, color='darkorange', lw=3)
        ax.set_xlabel('Slip rate (m/s)', fontsize=15)
        ax.set_ylabel('Shear stress (MPa)', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Steady_state.png'.format(self.fname), dpi=600)
        plt.close()