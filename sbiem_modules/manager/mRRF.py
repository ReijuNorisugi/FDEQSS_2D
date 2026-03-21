# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 2020309.
"""

# This is the module which summarizes the functions for adaptive stepper.

import torch as tr
import numpy as np
import torch_dct as dct
import logging

class Management():
    def __init__(self, Devices, Conditions, Medium, FaultParams, FieldVariables, id_station):
        # Main device for computation.
        self.device     = Devices['device_1']
        self.id_station = id_station
        
        self.id_count = 0      # ID for temporally sparse output.    
        self.id_count_st = 0   # ID for temporally dense output.
        self.ndt = 0           # For counting time steps.
        self.out_ndt = 100     # Output interval during interseismic.
        self.tmp_out = 10000   # Writing interval. Too large, then memory-consuming. 
        
        self.lam    = Medium['lam']    # Domain size.
        self.ncell  = Medium['ncell']  # Number of cells for fault domain.
        self.nperi  = Medium['nperi']  # Number of replication.
        self.hcell  = Medium['hcell'] # dx.
        self.Cs     = Medium['Cs']     # S-wave velocity
        self.mu     = Medium['mu']     # Rigidity.
        self.error0 = Medium['error0'] # epsilon_0 in Romanet & Ozawa, (2022).
        self.Nele   = Medium['Nele']  # Number of elements.
        self.k_size = FaultParams['k_size']   # Number of state variables.
        self.xp     = tr.tensor(Medium['xp'], dtype=tr.float64, device=self.device) # Coordinate.
        
        self.stations   = Conditions['stations']
        self.sparse     = Conditions['sparse']
        self.snap_EQ    = Conditions['snap_EQ']
        self.dense      = Conditions['dense']
        self.outerror   = Conditions['outerror']
        self.mirror     = Conditions['mirror']
        self.num_GPU    = Conditions['num_GPU']
        self.downsample = Conditions['downsample']    # Downsampling rate for dense spatial output.
        self.out_sample = int(self.ncell // self.downsample)  # Number of cells for dense spatial output.
        
        # Do not change.
        self.stock_source = False
        self.dtype = tr.float64
        self.dtype_ = np.int64
        self.dtype__ = np.float64

        self.EQ_num     = 0             # Number of EQ. This is required to unpack output.
        self.id_EQ      = 0             # Flag, 1: coseismic,  0: not coseismic.
        self.init_EQ    = 0.            # Store EQ start time.
        self.init_id    = 0             # Store EQ start id (time stamp).
        self.hypo       = 0             # Store hypocenter ID.
        self.slip_thres = 0.1           # Slip threshold to determine ruptured area (Somerville et al., 1990).
        self.moratorium_time  = 0.      # Wait event-end determination until moratorium.
        self.slip_range = np.array([0.05, 0.3])     # Lower and upper bound of slip thresholds.
        
        # Define EQ threshold. It is based on the competition between Alog(V/V0) and damping.
        self.v_cos      = tr.min(2. * self.Cs * FaultParams['a'] * FaultParams['sigma'] / self.mu)
        self.v_ecos     = self.v_cos
        
        # Moratorium to define event ends.
        self.moratorium = int(self.lam / self.nperi) / self.Cs
        self.moratorium = tr.tensor(self.moratorium, dtype=tr.float64, device=self.device)
        self.potency    = tr.zeros(1, dtype=self.dtype, device=self.device)          # Potency.
        self.Pr         = tr.zeros(1, dtype=self.dtype, device=self.device)          # Moment rate.
        self.Gc         = tr.zeros(self.ncell, dtype=self.dtype, device=self.device) # Fracture energy.
        self.delta_Gc   = tr.zeros(self.ncell, dtype=self.dtype, device=self.device) # Coseismic displacement.
        self.pad        = tr.zeros(int(self.Nele*(self.nperi-1)/self.nperi), dtype=tr.float64, device=self.device)
        
        if self.num_GPU == 2:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
        elif self.num_GPU == 4:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.stream_3 = tr.cuda.Stream(device='cuda:2')
            self.stream_4 = tr.cuda.Stream(device='cuda:3')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
            self.event_3 = tr.cuda.Event()
            self.event_4 = tr.cuda.Event()
        
        # Substitute all the module functions in advance for efficient computation.
        self.tr_max    = tr.max
        self.tr_min    = tr.min
        self.tr_argmax = tr.argmax
        self.tr_log10  = tr.log10
        self.tr_zeros  = tr.zeros
        self.tr_sum    = tr.sum
        self.tr_mean   = tr.mean
        self.tr_sqrt   = tr.sqrt
        self.tr_where  = tr.where
        self.tr_cat    = tr.cat
        self.tr_any    = tr.any
        self.tr_all    = tr.all
        self.tr_irfft  = tr.fft.irfft
        self.tr_rfft   = tr.fft.rfft
        self.tr_dct    = dct.dct
        self.tr_idct   = dct.idct
        self.empty     = np.empty
        self.np_array  = np.array
        self.append    = np.append
        self.reshape   = np.reshape
        self.hstack    = np.hstack
        self.concatenate = np.concatenate
        
        self.cuda_stream = tr.cuda.stream
        self.synchronize = tr.cuda.synchronize
    
    # Do not change.
    def __getstate__(self):
        states = self.__dict__.copy()
        if self.num_GPU == 2:
            states['stream_1'] = None
            states['stream_2'] = None
            states['event_1'] = None
            states['event_2'] = None
        elif self.num_GPU == 4:
            states['stream_1'] = None
            states['stream_2'] = None
            states['stream_3'] = None
            states['stream_4'] = None
            states['event_1'] = None
            states['event_2'] = None
            states['event_3'] = None
            states['event_4'] = None
        return states
    
    # Do not change.
    def __setstate__(self, states):
        self.__dict__ = states
        if self.num_GPU == 2:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
        elif self.num_GPU == 4:
            self.stream_1 = tr.cuda.Stream(device='cuda:0')
            self.stream_2 = tr.cuda.Stream(device='cuda:1')
            self.stream_3 = tr.cuda.Stream(device='cuda:2')
            self.stream_4 = tr.cuda.Stream(device='cuda:3')
            self.event_1 = tr.cuda.Event()
            self.event_2 = tr.cuda.Event()
            self.event_3 = tr.cuda.Event()
            self.event_4 = tr.cuda.Event()
    
    # Execute writing output files.
    # The dimension is (time steps, quantities). They are stored as 1-D array, so should be reshaped.
    # Note that time-related quantities are np.float64, and others are torch.float64 for some reasons.
    def exe_write(self, wr):
        wr.write_idmain(self.reshape(self.idmain, (-1, 4)))
        wr.write_t_sparse(self.reshape(self.store_t_sparse, (-1, 1)))
        wr.write_t_dense(self.reshape(self.store_t_dense, (-1, 1)))
        
        self.t_EQ = self.reshape(self.t_EQ, (-1, 2))
        self.EQcatalog = self.reshape(self.to_numpy(self.EQcatalog), (-1, 8))
        wr.write_catalog(self.hstack([self.t_EQ, self.EQcatalog]))
        self.err_source = self.reshape(self.to_numpy(self.err_source), (-1, 8))
        wr.write_err_source(self.err_source)
        
        wr.write_Pr(self.reshape(self.to_numpy(self.store_Pr*self.hcell), (-1, 1)))
        if self.snap_EQ or self.sparse:
            wr.write_Gcd(self.reshape(self.to_numpy(self.store_Gcd), (-1, self.out_sample)))
            if self.k_size <= 12:
                wr.write_sparse(self.reshape(self.to_numpy(self.store_delta), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_V), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_0), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_1), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_2), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_3), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_4), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_5), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_6), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_7), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_8), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_9), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_10), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_11), (-1, self.out_sample)),
                                0, 0, 0,
                                self.reshape(self.to_numpy(self.store_tau), (-1, self.out_sample))
                                )
            elif self.k_size == 13:
                wr.write_sparse(self.reshape(self.to_numpy(self.store_delta), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_V), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_0), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_1), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_2), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_3), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_4), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_5), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_6), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_7), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_8), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_9), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_10), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_11), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_12), (-1, self.out_sample)),
                                0, 0,
                                self.reshape(self.to_numpy(self.store_tau), (-1, self.out_sample))
                                )
            elif self.k_size == 14:
                wr.write_sparse(self.reshape(self.to_numpy(self.store_delta), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_V), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_0), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_1), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_2), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_3), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_4), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_5), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_6), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_7), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_8), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_9), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_10), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_11), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_12), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_13), (-1, self.out_sample)),
                                0,
                                self.reshape(self.to_numpy(self.store_tau), (-1, self.out_sample))
                                )
            elif self.k_size == 15:
                wr.write_sparse(self.reshape(self.to_numpy(self.store_delta), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_V), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_0), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_1), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_2), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_3), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_4), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_5), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_6), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_7), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_8), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_9), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_10), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_11), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_12), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_13), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_state_14), (-1, self.out_sample)),
                                self.reshape(self.to_numpy(self.store_tau), (-1, self.out_sample))
                                )
        if self.dense:
            if self.k_size <= 12:
                wr.write_station(self.reshape(self.to_numpy(self.store_delta_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_V_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state0_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state1_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state2_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state3_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state4_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state5_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state6_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state7_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state8_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state9_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state10_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state11_st), (-1, self.stations)),
                                0, 0, 0,
                                self.reshape(self.to_numpy(self.store_tau_st), (-1, self.stations))
                                )
            elif self.k_size == 13:
                wr.write_station(self.reshape(self.to_numpy(self.store_delta_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_V_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state0_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state1_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state2_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state3_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state4_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state5_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state6_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state7_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state8_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state9_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state10_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state11_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state12_st), (-1, self.stations)),
                                0, 0,
                                self.reshape(self.to_numpy(self.store_tau_st), (-1, self.stations))
                                )
            elif self.k_size == 14:
                wr.write_station(self.reshape(self.to_numpy(self.store_delta_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_V_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state0_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state1_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state2_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state3_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state4_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state5_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state6_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state7_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state8_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state9_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state10_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state11_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state12_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state13_st), (-1, self.stations)),
                                0,
                                self.reshape(self.to_numpy(self.store_tau_st), (-1, self.stations))
                                )
            elif self.k_size == 15:
                wr.write_station(self.reshape(self.to_numpy(self.store_delta_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_V_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state0_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state1_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state2_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state3_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state4_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state5_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state6_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state7_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state8_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state9_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state10_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state11_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state12_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state13_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_state14_st), (-1, self.stations)),
                                self.reshape(self.to_numpy(self.store_tau_st), (-1, self.stations))
                                )
        if self.outerror:
            wr.write_error(self.reshape(self.to_numpy(self.store_error), (-1, 1)), self.reshape(self.store_dt, (-1, 1)))
        self.init_strage()
    
    
    # Initialize strage arrays or tensors.
    # This is done for reducing file I/O process when using GPU.
    # Note that time-related quantities are np.float64, and others are torch.float64 for some reasons.
    def init_strage(self):
        # Numpy array to store variables.
        self.idmain         = self.empty(0, dtype=self.dtype_)
        self.store_dt       = self.empty(0, dtype=self.dtype__)
        self.store_t_sparse = self.empty(0, dtype=self.dtype__)
        self.store_t_dense  = self.empty(0, dtype=self.dtype__)
        self.t_EQ           = self.empty(0, dtype=self.dtype__)
        
        # Torch tensors to store variables.
        self.EQcatalog        = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.err_source       = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_error      = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_Pr         = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_Gcd        = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_delta      = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_V          = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_0    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_1    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_2    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_3    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_4    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_5    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_6    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_7    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_8    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_9    = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_10   = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state_11   = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        if self.k_size >= 13:
            self.store_state_12 = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        if self.k_size >= 14:
            self.store_state_13 = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        if self.k_size >= 15:
            self.store_state_14 = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_tau        = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_delta_st   = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_V_st       = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state0_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state1_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state2_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state3_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state4_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state5_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state6_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state7_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state8_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state9_st  = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state10_st = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_state11_st = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        if self.k_size >= 13:
            self.store_state12_st = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        if self.k_size >= 14:
            self.store_state13_st = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        if self.k_size >= 15:
            self.store_state14_st = self.tr_zeros(0, dtype=self.dtype, device=self.device)
        self.store_tau_st     = self.tr_zeros(0, dtype=self.dtype, device=self.device)
    

    # Transform tensor to ndarray.
    def to_numpy(self, tensor):
        return tensor.cpu().detach().numpy().copy()


    # Check events. EQcatalog is established here.
    def flag_EQ(self, result, ker):
        # If slpi rate at any point exceeds threshold, then EQ starts.
        if self.id_EQ == 0:
            # Monitoring dt0 decrease the number of taking max on GPU.
            if self.dt0 >= 5.:
                return
            else:
                pass
            if self.tr_any(result.V >= self.v_cos):
                self.moratorium_time = 0.
                pass
            else:
                return
            self.id_EQ = 1
            self.out_ndt = 10 # During events, output interval becomes short.
            if self.snap_EQ: # For saving snapshot at the initial time.
                self.ndt = self.out_ndt - 1
            else:
                self.ndt = 0
            # Store quantities at event initiation.
            self.init_EQ = self.t
            self.hypo = self.tr_argmax(result.V).reshape(1)
            self.init_id_sparse = self.id_count
            self.init_id_dense = self.id_count_st
            self.ini_delta = result.delta.clone()
            self.ini_tau = result.tau.clone()
        # If slip rate at all points is lower than threshold, then EQ ends.
        # Due to the oscillation, slip rate criteria sometimes picks tiny, invalid events.
        # To prevent the mispicking, we introduce moratorium.
        # After max(V) < v_cos, wait during moratorium before determining event end.
        # Note that the event end is the time that max(V) becomes lower than v_cos.
        else:
            # Monitoring hypocenter decrease the number of taking max on GPU.
            if bool(result.V[self.hypo] >= self.v_cos):
                return
            else:
                pass
            if self.tr_all(result.V < self.v_ecos):
                self.moratorium_time += self.dt0
                
                if self.stock_source:
                    pass
                else:
                    # Snap EQ start and end times.
                    self.tmp = self.np_array([self.init_EQ, self.t])
                    self.tmp_ = self.np_array([self.init_id_sparse, self.id_count, 
                                                self.init_id_dense, self.id_count_st])
                    # Snap total coseismic slip, potency, Gc distribution, and shear stress change.
                    self.EQ_slip = result.delta - self.ini_delta
                    self.tau_change = result.tau - self.ini_tau
                    self.Gcd = self.Gc - self.delta_Gc * (result.tau + result.tau_prv) * 0.5
                    # Compute static stress drop.
                    if not self.mirror:
                        self.def_del = self.tr_rfft(-self.EQ_slip, n=self.ncell*self.nperi)
                        self.stress_drop = self.tr_irfft(ker.kernel_st * self.def_del, n=self.ncell*self.nperi)[:self.ncell]
                    else:
                        self.def_del = self.tr_dct(self.tr_cat((-self.EQ_slip, self.pad), dim=0))
                        self.stress_drop = self.tr_idct(ker.kernel_st * self.def_del)[:self.ncell]
                    
                    # Energy dissipation.
                    self.EG = self.tr_sum(self.Gcd) * self.hcell
                    # Available energy.
                    self.Ea = self.tr_sum(- self.tau_change * self.EQ_slip) * self.hcell * 0.5
                    # Radiation energy.
                    self.ER = self.Ea - self.EG
                    
                    # Capture ruptured area, > self.slip_thres % of maximum total slip.
                    self.ruptured = self.tr_where(self.EQ_slip >= self.slip_thres * self.tr_max(self.EQ_slip))
                    # Elliptical weight function for computing stress drop.
                    self.R = (self.tr_max(self.xp[self.ruptured]) - self.tr_min(self.xp[self.ruptured])) * 0.5
                    self.center = (self.tr_max(self.xp[self.ruptured]) + self.tr_min(self.xp[self.ruptured])) * 0.5
                    self.coord = self.xp[self.ruptured] - self.center
                    self.ellip = 2. * self.R * self.tr_sqrt(1. - (self.coord/self.R)**2.)
                    # Static stress drop.
                    self.sd = ( self.tr_sum( self.stress_drop[self.ruptured] * self.ellip ) 
                                        / self.tr_sum( self.ellip )
                    )
                    # Seismic potency.
                    self.potency = self.tr_sum(self.EQ_slip[self.ruptured]) * self.hcell
                    # Average fracture energy.
                    self.Gc_ave = self.tr_mean(self.Gcd[self.ruptured])
                    # Average coseismic slip.
                    self.slip_ave = self.tr_mean(self.EQ_slip[self.ruptured])
                    
                    # Error of source parameters due to the slip threshold to determine ruptured area.
                    # 8 components. Lower and upper bounds of stress drop, seismic potency, fracture energy, and average slip.
                    self.err = self.tr_zeros(0, dtype=self.dtype, device=self.device)
                    for err in range(len(self.slip_range)):
                        self.ruptured = self.tr_where(self.EQ_slip >= self.slip_range[err] * self.tr_max(self.EQ_slip))
                        self.R = (self.tr_max(self.xp[self.ruptured]) - self.tr_min(self.xp[self.ruptured])) * 0.5
                        self.center = (self.tr_max(self.xp[self.ruptured]) + self.tr_min(self.xp[self.ruptured])) * 0.5
                        self.coord = self.xp[self.ruptured] - self.center
                        self.ellip = 2. * self.R * self.tr_sqrt(1. - (self.coord/self.R)**2.)
                        
                        self.sd_ = ( self.tr_sum( self.stress_drop[self.ruptured] * self.ellip ) 
                                            / self.tr_sum( self.ellip )
                        )
                        self.pot_ = self.tr_sum(self.EQ_slip[self.ruptured]) * self.hcell
                        self.Gc_ = self.tr_mean(self.Gcd[self.ruptured])
                        self.slip_ = self.tr_mean(self.EQ_slip[self.ruptured])
                        
                        self.err = self.tr_cat((self.err,
                                                self.sd_.reshape(1), self.pot_.reshape(1),
                                                self.Gc_.reshape(1), self.slip_.reshape(1)), dim=0)
                    
                    if self.snap_EQ: # For saveing snapshot at the end time.
                        self.ndt = self.out_ndt - 1
                    else:
                        self.ndt = 0
                    
                    self.stock_source = True
                
                if self.moratorium_time < self.moratorium:
                    return
                else:
                    # If slip rate exceeds threshold again during moratorium time,
                    # the source parameters are re-evaluated a the new event end point.
                    self.stock_source = False
                    pass
                self.id_EQ = 0
                self.EQ_num += 1
                self.out_ndt = 100
                
                # Store EQcatalog, event ID, and source parameters from diffenrent slip threshold.
                self.t_EQ = self.concatenate([self.t_EQ, self.tmp])
                self.EQcatalog = self.tr_cat((self.EQcatalog,
                                            self.hypo, self.potency.reshape(1),
                                            self.sd.reshape(1), self.Gc_ave.reshape(1),
                                            self.slip_ave.reshape(1), self.EG.reshape(1),
                                            self.Ea.reshape(1), self.ER.reshape(1)), dim=0)
                self.idmain = self.concatenate([self.idmain, self.tmp_])
                self.err_source = self.tr_cat((self.err_source, self.err), dim=0)
                
                if self.snap_EQ:
                    self.store_Gcd = self.tr_cat((self.store_Gcd, self.Gcd[::self.downsample]), dim=0)
                else:
                    pass
                # Reset storage variables.
                self.Gc = self.Gc.zero_()
                self.delta_Gc = self.delta_Gc.zero_()
                self.delta_inst = self.delta_inst.zero_()
                self.log(f'EQ = {self.EQ_num} ..... t = {self.t} sec ..... prog = {self.t/self.tmax:.3f}')
            else:
                self.moratorium_time = 0.
                self.stock_source = False
                return


    # Compute source parameters here.
    def calc_source(self, result):
        # Store Moment rate at every time step.
        self.store_t_dense = self.append(self.store_t_dense, self.t)
        self.store_Pr = self.tr_cat((self.store_Pr, self.tr_sum(result.V).reshape(1)), dim=0)
        if self.id_EQ == 1:
            # Compute fracture energy Gc and coseismic displacement.
            self.delta_inst = result.delta - result.delta_prv
            self.Gc += 0.5 * (result.tau + result.tau_prv) * self.delta_inst
            self.delta_Gc += self.delta_inst
        else:
            pass


    # Store snapshot on the entire fautlt.
    def snap_sparse(self, result):
        if self.sparse:
            self.ndt += 1
        else:
            pass
        if self.ndt != self.out_ndt:
            pass
        else:
            self.ndt = 0
            self.id_count += 1
            self.store_t_sparse = self.append(self.store_t_sparse, self.t)
            self.store_delta = self.tr_cat((self.store_delta, result.delta[::self.downsample]), dim=0)
            self.store_V = self.tr_cat((self.store_V, result.V[::self.downsample]), dim=0)
            self.store_state_0  = self.tr_cat((self.store_state_0, result.state[::self.downsample, 0]), dim=0)
            self.store_state_1  = self.tr_cat((self.store_state_1, result.state[::self.downsample, 1]), dim=0)
            self.store_state_2  = self.tr_cat((self.store_state_2, result.state[::self.downsample, 2]), dim=0)
            self.store_state_3  = self.tr_cat((self.store_state_3, result.state[::self.downsample, 3]), dim=0)
            self.store_state_4  = self.tr_cat((self.store_state_4, result.state[::self.downsample, 4]), dim=0)
            self.store_state_5  = self.tr_cat((self.store_state_5, result.state[::self.downsample, 5]), dim=0)
            self.store_state_6  = self.tr_cat((self.store_state_6, result.state[::self.downsample, 6]), dim=0)
            self.store_state_7  = self.tr_cat((self.store_state_7, result.state[::self.downsample, 7]), dim=0)
            self.store_state_8  = self.tr_cat((self.store_state_8, result.state[::self.downsample, 8]), dim=0)
            self.store_state_9  = self.tr_cat((self.store_state_9, result.state[::self.downsample, 9]), dim=0)
            self.store_state_10 = self.tr_cat((self.store_state_10, result.state[::self.downsample, 10]), dim=0)
            self.store_state_11 = self.tr_cat((self.store_state_11, result.state[::self.downsample, 11]), dim=0)
            if self.k_size >= 13:
                self.store_state_12 = self.tr_cat((self.store_state_12, result.state[::self.downsample, 12]), dim=0)
            if self.k_size >= 14:
                self.store_state_13 = self.tr_cat((self.store_state_13, result.state[::self.downsample, 13]), dim=0)
            if self.k_size >= 15:
                self.store_state_14 = self.tr_cat((self.store_state_14, result.state[::self.downsample, 14]), dim=0)
            self.store_tau = self.tr_cat((self.store_tau, result.tau[::self.downsample]), dim=0)
    
    # Store dense output at stations.
    def snap_dense(self, result):
        self.store_delta_st = self.tr_cat((self.store_delta_st, result.delta[self.id_station]), dim=0)
        self.store_V_st = self.tr_cat((self.store_V_st, result.V[self.id_station]), dim=0)
        self.store_state0_st = self.tr_cat((self.store_state0_st, result.state[self.id_station, 0]), dim=0)
        self.store_state1_st = self.tr_cat((self.store_state1_st, result.state[self.id_station, 1]), dim=0)
        self.store_state2_st = self.tr_cat((self.store_state2_st, result.state[self.id_station, 2]), dim=0)
        self.store_state3_st = self.tr_cat((self.store_state3_st, result.state[self.id_station, 3]), dim=0)
        self.store_state4_st = self.tr_cat((self.store_state4_st, result.state[self.id_station, 4]), dim=0)
        self.store_state5_st = self.tr_cat((self.store_state5_st, result.state[self.id_station, 5]), dim=0)
        self.store_state6_st = self.tr_cat((self.store_state6_st, result.state[self.id_station, 6]), dim=0)
        self.store_state7_st = self.tr_cat((self.store_state7_st, result.state[self.id_station, 7]), dim=0)
        self.store_state8_st = self.tr_cat((self.store_state8_st, result.state[self.id_station, 8]), dim=0)
        self.store_state9_st = self.tr_cat((self.store_state9_st, result.state[self.id_station, 9]), dim=0)
        self.store_state10_st = self.tr_cat((self.store_state10_st, result.state[self.id_station, 10]), dim=0)
        self.store_state11_st = self.tr_cat((self.store_state11_st, result.state[self.id_station, 11]), dim=0)
        if self.k_size >= 13:
            self.store_state12_st = self.tr_cat((self.store_state12_st, result.state[self.id_station, 12]), dim=0)
        if self.k_size >= 14:
            self.store_state13_st = self.tr_cat((self.store_state13_st, result.state[self.id_station, 13]), dim=0)
        if self.k_size >= 15:
            self.store_state14_st = self.tr_cat((self.store_state14_st, result.state[self.id_station, 14]), dim=0)
        self.store_tau_st = self.tr_cat((self.store_tau_st, result.tau[self.id_station]), dim=0)
    
    
    # Store error and time step if you want.
    def snap_error(self, error):
        self.store_error = self.tr_cat((self.store_error, error), dim=0)
        self.store_dt = self.append(self.store_dt, self.dt0)

    
    # Update IDs.
    def id_ev(self):
        if self.ndt == self.out_ndt:
            self.ndt = 0
            self.id_count += 1
        else:
            pass
    
    # Time step evolution.
    def time_upt(self, ti):
        self.t = ti.t*1
        self.dt0 = ti.dt0*1
        self.id_count_st += 1
    
    # Store outputs, you should somehow change here when adding new output.
    def store_RO(self, double, error, ker):
        # Compute source parameters.
        self.calc_source(double)
        # Check criteria for EQ occurrence and make EQ catalog.
        self.flag_EQ(double, ker)
        # Sparse output on the fault.
        self.snap_sparse(double)
        if self.outerror:
            # Store error and dt if you want.
            self.snap_error(error)
        else:
            pass
        if not self.dense:
            pass
        else:
            # Dense output at stations.
            self.snap_dense(double)
            
    
    # Time stepper for Romanet & Ozawa method.
    def stepper_RO(self, single, double, error, wr, ti, ker, cv, cv_):
        if error < self.error0:
            # Adopt time.
            ti.tev()
            # Update class variables for time.
            self.time_upt(ti)
            # Align history between single and double steps.
            cv.align_hist(cv_, False)
            # Write output.
            self.store_RO(double, error, ker)
            # Update self for next time step.
            double.upt_prv(double)
            single.upt_prv(double)
            single.upt_prs(double)
            # Adopt time step.
            ti.dtev(error)
            # Transfer storage variables on GPU to CPU and write outputs.
            if not self.id_count_st % self.tmp_out == 0:
                pass
            else:
                self.exe_write(wr)
        else:
            # During EQ.
            if ti.dt0 <= ti.dtmin:
                ti.tev()
                self.time_upt(ti)
                cv.align_hist(cv_, False)
                self.store_RO(double, error, ker)
                double.upt_prv(double)
                single.upt_prv(double)
                single.upt_prs(double)
                ti.dtev(error)
                if not self.id_count_st % self.tmp_out == 0:
                    pass
                else:
                    self.exe_write(wr)
            else: # Re-evaluate the current step with modified step size.
                ti.dtev(error)
                cv_.align_hist(cv, True)
                single.rollback_tensors(single)
                double.rollback_tensors(single)