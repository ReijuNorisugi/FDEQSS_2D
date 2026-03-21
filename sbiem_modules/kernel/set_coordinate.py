# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

import torch as tr
import numpy as np

# Function to set spatial coordinate.
def set_coord(mirror, Nele, lam, mode, nperi, beta_min, Cs, Cp, ratio, Z, H, device_1):
    # Cell size (m).
    hcell = lam/Nele
    dtype = tr.float64
    if not mirror: # Without mirror.
        # Wavenumber.
        kn = 2. * np.pi * np.arange(0, (0.5*Nele)+1, 1) / lam
        if mode == 'IV': # See Lapusta D-thesis. Do not change here.
            kn = np.sqrt( (Z * kn)**2 + (1/H)**2 )
        
        # Coordinate
        xp = tr.as_tensor(np.arange(0, lam/nperi, hcell) + 0.5*hcell - 0.5*lam/nperi, 
                        dtype=dtype, device=device_1)
    else:
        # Wavenumber. Note that mirror part should be included.
        kn = np.pi * np.arange(0, Nele, 1) / lam
        # Coordinate.
        xp  = tr.as_tensor(np.arange(0, lam/nperi, hcell) + 0.5*hcell, 
                           dtype=dtype, device=device_1)
    
    ncell = len(xp)  # Cell number on the physical domain.
    nconv = Nele     # Cell number for the convolution.

    # The fastest seismic wave.
    if mode == 'I' or mode == 'II':
        C = Cp
    else:
        C = Cs
    
    # Compute minimum time step. Do not change here.
    dtmin = beta_min * hcell / C
    # Compute truncation window size.
    Tw = ratio * (lam/nperi/C)
        
    return hcell, kn, xp, ncell, nconv, dtmin, Tw

# Function to set stations' location.
def set_station(stations_uni, ncell, stations):
    if stations_uni:
        id_station = (ncell*np.arange(0, stations+1, 1)/stations).astype(np.int64)
        id_station[1:] = id_station[1:] - 1
        return id_station, stations + 1
    else:
        id_station = stations
        return id_station, len(stations)