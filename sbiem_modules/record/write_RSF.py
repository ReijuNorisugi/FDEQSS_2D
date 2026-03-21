# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260309.
"""

# This is the module which summarizes the output functions.
# If you want additional output, edit this file to make output files.
# This class is mainly executed in manager. You should also edit this script.

class Write():
    def __init__(self, fname, Devices, Conditions, Medium, FaultParams, FieldVariables):
        self.fname = fname
        # Define file open option.
        self.mode = 'wb'

    # Open files for writing.
    def open_files(self):
        self.out_catalog       = open('Output{}out_catalog.bin'.format(self.fname), mode=self.mode)
        self.out_err_source    = open('Output{}out_err_source.bin'.format(self.fname), mode=self.mode)
        self.out_idmain        = open('Output{}out_idmain.bin'.format(self.fname), mode=self.mode)
        self.out_t_sparse      = open('Output{}out_t_sparse.bin'.format(self.fname), mode=self.mode)
        self.out_t_dense       = open('Output{}out_t_dense.bin'.format(self.fname), mode=self.mode)
        self.out_delta         = open('Output{}out_delta.bin'.format(self.fname), mode=self.mode)
        self.out_V             = open('Output{}out_V.bin'.format(self.fname), mode=self.mode)
        self.out_state         = open('Output{}out_state.bin'.format(self.fname), mode=self.mode)
        self.out_tau           = open('Output{}out_tau.bin'.format(self.fname), mode=self.mode)
        self.out_Ldyn          = open('Output{}out_Ldyn.bin'.format(self.fname), mode=self.mode)
        self.out_Pr            = open('Output{}out_Pr.bin'.format(self.fname), mode=self.mode)
        self.out_Gcd           = open('Output{}out_Gcd.bin'.format(self.fname), mode=self.mode)
        self.out_delta_station = open('Output{}out_delta_station.bin'.format(self.fname), mode=self.mode)
        self.out_V_station     = open('Output{}out_V_station.bin'.format(self.fname), mode=self.mode)
        self.out_tau_station   = open('Output{}out_tau_station.bin'.format(self.fname), mode=self.mode)
        self.out_state_station = open('Output{}out_state_station.bin'.format(self.fname), mode=self.mode)
        self.out_Ldyn_station  = open('Output{}out_Ldyn_station.bin'.format(self.fname), mode=self.mode)
        self.out_error         = open('Output{}out_error.bin'.format(self.fname), mode=self.mode)
        self.out_dt            = open('Output{}out_dt.bin'.format(self.fname), mode=self.mode)


    # Close files at the end of simulaiton.
    def close_files(self):
        self.out_catalog.close()
        self.out_err_source.close()
        self.out_idmain.close()
        self.out_t_sparse.close()
        self.out_t_dense.close()
        self.out_delta.close()
        self.out_V.close()
        self.out_state.close()
        self.out_tau.close()
        self.out_Ldyn.close()
        self.out_Pr.close()
        self.out_Gcd.close()
        self.out_delta_station.close()
        self.out_V_station.close()
        self.out_tau_station.close()
        self.out_state_station.close()
        self.out_Ldyn_station.close()
        self.out_error.close()
        self.out_dt.close()

    # EQ catalog.
    # [0]Initial time, [1]End time, [2]Hypocenter ID, [3]Seismic potency, [4]Moment-based stress drop,
    # [5]Average fracture energy, [6]Average slip, [7]Total dissipation, [8]Available energy, [9]Radiated energy.
    # [10]Energy-based stress drop.
    def write_catalog(self, catalog):
        self.out_catalog.write(catalog.tobytes())

    # Estimation ranges of source parameters due to the arbitrary definition of ruptured area.
    # Lower bounds of [0]Moment-based stress drop, [1]Seismic potency,
    #                 [2]Average fracture energy, [3]Average slip, [4]Energy-based stress drop.
    # Upper bounds of [5]Moment-based stress drop, [6]Seismic potency,
    #                 [7]Average fracture energy, [8]Average slip, [9]Energy-based stress drop.
    def write_err_source(self, err):
        self.out_err_source.write(err.tobytes())

    # Index of event timing to plot dense and sparse outputs.
    def write_idmain(self, idmain):
        self.out_idmain.write(idmain.tobytes())

    # Time record for temporaly sparse outputs.
    def write_t_sparse(self, t):
        self.out_t_sparse.write(t.tobytes())

    # Time record for temporaly dense outputs.
    def write_t_dense(self, t):
        self.out_t_dense.write(t.tobytes())

    # Temporaly sparse bulk output: slip, slip rate, state variables, & shear stress.
    def write_sparse(self, delta, V, state, tau):
        self.out_delta.write(delta.tobytes())
        self.out_V.write(V.tobytes())
        self.out_state.write(state.tobytes())
        self.out_tau.write(tau.tobytes())

    # Temporaly sparse bulk output: L dynamic if TW = True.
    def write_Ldyn(self, L_dyn):
        self.out_Ldyn.write(L_dyn.tobytes())

    # Moment rate function. Output by every time step.
    def write_Pr(self, Pr):
        self.out_Pr.write(Pr.tobytes())

    # Fracture energy density distribution.
    def write_Gcd(self, Gcd):
        self.out_Gcd.write(Gcd.tobytes())

    # Temporaly dense output at stations.
    def write_station(self, delta, V, tau, state, L_dyn):
        self.out_delta_station.write(delta.tobytes())
        self.out_V_station.write(V.tobytes())
        self.out_tau_station.write(tau.tobytes())
        self.out_state_station.write(state.tobytes())
        self.out_Ldyn_station.write(L_dyn.tobytes())

    # Error and time step output if outerror = True.
    def write_error(self, error, dt):
        self.out_error.write(error.tobytes())
        self.out_dt.write(dt.tobytes())
