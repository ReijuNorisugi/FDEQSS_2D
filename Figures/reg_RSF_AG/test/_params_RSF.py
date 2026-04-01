# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

# This is the script to set parameters.
# Note that this is based on the SI units.

import numpy as np
import torch as tr
import torch_dct as dct # DCT for PyTorch. See details at https://github.com/zh217/torch-dct
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(next(p for p in Path(__file__).resolve().parents if p.name == sys.argv[1])))
from sbiem_modules.utils import utils
from sbiem_modules.kernel import set_coordinate as setc


#################   Flags to change the computatinal mode and output conditions.    ###############
Initial = False  # True: You can check parameter settings without runnning the whole simulation.
act_res = True   # True: Save files for restarting.
mirror = True   # True: Hlaf infinite-space like Lapusta et al. (2000).
qd = False       # False: Fully-dynamic simulation.  True: Quasi-dynamic simulation.
mode = 'III'     # Rupture modes I(opening), II(in-plane), III(anti-plane), and IV(crustal plane) are available.
sparse = True    # Allow dense spatial snapshot at sparse time steps.         !!! Storage consuming !!!
downsample = 1   # Downsampling for spatially dense output.
snap_EQ = True   # Allow dense spatial snapshot at before and after an event. !!! Storage consuming !!!
dense = True     # Allow every time step output on sparse stations.           !!! Storage consuming !!!
outerror = False  # Output error between full and half step solutions. Ture enforces ['RO'] time-marching scheme.
Frict = 'reg_RSF_AG' # Choose fault constitutive law. RSF_AG is only available now. reg_RSF_AG is regularized version.
Stepper = 'LR'  # Choose scheme for time step evolution.
                # Romanet & Ozawa (2022) ['RO'], Lapusta et al. (2000) ['LR'], and constant step ['CS'] are implemented.
rmPB = True    # True: Remove periodic boundaries. Available for both infinite-scpace and half infinite-space solutions.

# Set filename to save output.
fname = '/' + Frict + '/test/'

# If GPU is available, PyTorch automatically uses GPU.
# num_GPU defines the number of GPU for convolution.
# 1, 2, and 4 are available now.
num_GPU = 1

# Do not change here.
os.makedirs('Figures{}Initial'.format(fname), exist_ok=True) # Plot of initial conditions are saved here.
os.makedirs('Output{}'.format(fname), exist_ok=True)  # Output files are saved here.
os.makedirs('Restart{}'.format(fname), exist_ok=True) # Restart files are saved here.
logging.basicConfig(filename='Log{}log_{}.log'.format(fname, datetime.now().strftime('%Y%m%d_%H')),
                    level=logging.INFO, format='%(message)s', filemode='w', force=True)
#####################################################################################################

######   Set tmax   ######
tmax = 1500. * 365. * 24. * 60. * 60.
##########################

###############################   Important   #################################
# When you choose ['RO'] for time stepper, you should set error0.
# If the simulation does not converge at initiation, use a smaller value.
error0 = 1.e-4  # Torlelance for time step. See Romanet & Ozawa, (2022).

# When you choose ['LR'] for time stepper, you can change safety factor.
# Usually, 0.5 is fine.
safety_LR = 0.5

# Initial guess of dt.
# Should be larger than truncation window for convolution Tw.
dt0_guess = 1.e2
###############################################################################

#######################    Set stations' location    ##########################
stations_uni = True   # Uniform or non-uniform distribution of stations.
if stations_uni:      # Uniform stations.
    stations = 2**4   # Number of stations on the fault. Multiple of 2 is nice.
else:                 # Non-uniform stations.
    stations = np.array([])  # Pick up cell number for stations.
###############################################################################

################# Set devices ##################
G2G = True # Do not change.
num_GPU, device_1 = utils.set_device(num_GPU)
################################################

################################   Medium parameters   ###############################
dtype = tr.float64 # Important, dtype for tensor. Do not change.

Cs = 3.464*1.e3 # S-wave velocity (m/s).
Cp = 5.e3     # P-wave velocity (m/s). 
rho = 2670.   # Density (kg/m^3).
mu = rho * (Cs**2) # Rigidity (Pa).
nu = 1 - ((Cp/Cs)**2) / (2 * ((Cp/Cs)**2 - 1)) # Poisson's ratio.

f0 = 0.6       # Referrence friction (non-dimentinal).
V0 = 1.e-6     # Referrence slip rate for standard RSF law (m/s).

Vpl = 1.e-9     # Loading rate (m/s).
beta_min = 0.5  # CFL. For defining minimum time step.
tau_rate = 0.   # Uniform stressing rate.

# If you choose mode IV (crustal-plane model), please set the following.
H = 5.e3              # Depth for averaging. See Lehner (1981).
H = (0.25*np.pi) * H  # Do not change.
Z = 1 / (1 - nu)      # Argument for crustal plane model. See Lehner (1981).

# Radiation dumping coefficient. Do not change here.
if mode == 'II' or mode == 'III' or mode == 'IV':
    eta = 0.5 * rho * Cs
elif mode == 'I':
    eta = 0.5 * rho * Cp

# Set coordinate.
if not mirror:        # Without mirror.
    lam = 4000.      # Size of computational regeme (m).
    Nele = 2048        # Total cell number for computation. Power of 2 is nice.

    if mode == 'IV':
        nperi = 1     # Fault replication. Do not change here.
    else:
        if rmPB:      # If you do not have special commitment, rmPB will be better.
            nperi = 2
        else:         # rmPB = False includes the effect from periodic faults. Choose larger replication.
            nperi = 2 # At least larger than 2.

    # * * * * * * Importnat * * * * * * #
    # Define truncation for dynamic kernel convolution.
    # Ratio = 1 allows that the wave propagate through a single fault length.
    # Note that wavenumber-dependent truncation is not implemented.
    if rmPB: # Without periodic boundaries.
        ratio = 2.0  # You can take longer truncation window without increasing system size.
    else:    # With periodic boundaries.
        ratio = 1.0  # Choose longer value as possible as you can.
    # * * * * * * * * * * * * * * * * * #
else:                 # With mirror.
    lam = 80000.    # Size of computational regime. Mirror part is excluded.
    Nele = 2048      # Total cell numer for computation.
    nperi = 2  # At least larger than 2.

    if rmPB:
        ratio = 2.0
    else:
        ratio = 2.0

hcell, kn, xp, ncell, nconv, dtmin, Tw = setc.set_coord(mirror, Nele, lam, mode, nperi, beta_min,
                                                        Cs, Cp, ratio, Z, H, device_1)

# Set stations. Do not change.
id_station, stations = setc.set_station(stations_uni, ncell, stations)
np.save('Output{}id_station.npy'.format(fname), id_station)
###########################################################################################

##################   Save variables by pickle.   ###################
# If you add new variables, maybe useful to add it here.
Conditions = {'Initial': Initial,
              'act_res': act_res,
              'mirror': mirror,
              'qd': qd,
              'mode': mode,
              'sparse': sparse,
              'downsample': downsample,
              'snap_EQ': snap_EQ,
              'dense': dense,
              'outerror': outerror,
              'Frict': Frict,
              'Stepper': Stepper,
              'rmPB': rmPB,
              'num_GPU': num_GPU,
              'tmax': tmax,
              'stations_uni': stations_uni,
              'stations': stations,
              'G2G': G2G,
              }

Devices = {'device_1': device_1}

Medium = {'Cs': Cs,
          'Cp': Cp,
          'mu': mu,
          'nu': nu,
          'eta': eta,
          'f0': f0,
          'V0': V0,
          'Vpl': Vpl,
          'H': H,
          'Z': Z,
          'tau_rate': tau_rate,
          'beta_min': beta_min ,
          'lam': lam,
          'Nele': Nele,
          'hcell': hcell,
          'nperi': nperi,
          'kn': kn,
          'xp': xp.cpu().detach().numpy().copy(),
          'ncell': ncell,
          'dtmin': dtmin,
          'Tw': Tw,
          'error0': error0,
          'dt0_guess': dt0_guess,
          'safety_LR': safety_LR,
          }

utils.save('Output{}Conditions.pkl'.format(fname), Conditions)
utils.save('Output{}Medium.pkl'.format(fname), Medium)
utils.save('Output{}Devices.pkl'.format(fname), Devices)
####################################################################

# Print conditions.
utils.show(fname, qd, mirror, Frict, mode, Stepper, lam, nperi,
           Tw, dtmin, sparse, dense, snap_EQ, outerror)

##########################   Set fault constitutive law's parameters here   ########################
# Useful function. Allow smooth parameter transition.
def smoothed_boxcar(w1, w2, x, param_in, param_out):
    return param_in + (param_out - param_in) * 0.5 * (1. - tr.tanh((w2/(tr.abs(x)-w1-w2))+(w2/(tr.abs(x)-w1))))

# Frictional parameters.
sigma = tr.zeros_like(xp, dtype=dtype, device=device_1)
a =     tr.zeros_like(xp, dtype=dtype, device=device_1)
b =     tr.zeros_like(xp, dtype=dtype, device=device_1)
A =     tr.zeros_like(xp, dtype=dtype, device=device_1)
B =     tr.zeros_like(xp, dtype=dtype, device=device_1)
L =     tr.zeros_like(xp, dtype=dtype, device=device_1)

if mirror: # Hlaf infinite-space.
    # This is the setting for SCEC Benchmark problem BP1-FD and BP1_QD.
    for i in range(len(xp)):
        sigma[i] = 50. * 1.e6
        b[i] = 0.015
        L[i] = 0.008
        if xp[i] < 15. * 1.e3:
            a[i] = 0.010
        elif xp[i] >= 15. * 1.e3 and xp[i] < 18. * 1.e3:
            a[i] = 0.010 + (0.025 - 0.010) * (xp[i] - 15. * 1.e3) / (3. * 1.e3)
        else:
            a[i] = 0.025
else: # Infinite space.
    sigma += 50. * 1.e6

    W1 = 500.
    W2 = 400.

    a_out = 0.012
    a_in = 0.012
    L_out = 5.e-4
    L_in = 5.e-4
    ab_out = 0.005
    ab_in = -0.004
    b_out = a_out - ab_out
    b_in = a_in - ab_in

    for i in range(len(xp)):
        if tr.abs(xp[i]) + 0. > W1 + W2:
            a[i] = a_out
            b[i] = b_out
            L[i] = L_out
        elif tr.abs(xp[i]) + 0. < W1:
            a[i] = a_in
            b[i] = b_in
            L[i] = L_in
        else:
            a[i] = smoothed_boxcar(W1, W2, xp[i], a_in, a_out)
            b[i] = smoothed_boxcar(W1, W2, xp[i], b_in, b_out)
            L[i] = smoothed_boxcar(W1, W2, xp[i], L_in, L_out)

A = a * sigma
B = b * sigma

# Check resolution.
h_star_RR = tr.zeros_like(xp, dtype=dtype)
h_star_RA = tr.zeros_like(xp, dtype=dtype)
kai       = tr.zeros_like(xp, dtype=dtype)
xi        = tr.zeros_like(xp, dtype=dtype)
lambda_0  = tr.zeros_like(xp, dtype=dtype)
pi = tr.tensor(np.pi, dtype=dtype, device=device_1)
if mode == 'III' or mode == 'IV':
    arg = mu
else:
    arg = mu / (1 - nu)

for i in range(ncell):
    if a[i] - b[i] < 0.:
        h_star_RR[i] = (0.25*pi)* arg * L[i] / (B[i] - A[i])
        h_star_RA[i] = (2./pi)* arg * b[i] * L[i] / (sigma[i]*((b[i] - a[i])**2.))
    else:
        h_star_RR[i] = 0.
        h_star_RA[i] = 0.

    kai[i] = 0.25 * ( ( 0.25*pi*mu*L[i]/hcell/A[i] - (B[i] - A[i])/A[i] )**2. ) - 0.25*pi*mu*L[i]/hcell/A[i]
    if kai[i] > 0:
        xi[i] = tr.min(tr.tensor(((A[i]/(0.25*pi*mu*L[i]/hcell - B[i] + A[i]), 0.5))))
    else:
        xi[i] = tr.min(tr.tensor((1 - (B[i] - A[i])/(0.25*pi*mu*L[i]/hcell), 0.5)))

lambda_0 = (9.*pi/32.) * arg * L / B

logging.info(f' *  h*RR / h = {tr.min(h_star_RR[h_star_RR > 0.]).cpu().detach().numpy()/hcell:.1f}')
logging.info(f' *  h*RA / h = {tr.min(h_star_RA[h_star_RA > 0.]).cpu().detach().numpy()/hcell:.1f}')
logging.info(' *  At least > 20 will be fine, > 40 will be nicely resolved')
logging.info('')
logging.info(f' *  Lambda0 / h = {tr.min(lambda_0/hcell):.1f}')
logging.info(' *   At least > 3 is required.')
logging.info(f' *  hcell = {hcell}')
logging.info('')
logging.info('*************************************************')
logging.info('')
#######################################################################################################

######################    Define initial values   #####################
# Define field variables.
V       = tr.zeros_like(xp, dtype=dtype, device=device_1) # Slip rate (m/s).
state   = tr.zeros_like(xp, dtype=dtype, device=device_1) # State variables (1/s).
delta   = tr.zeros_like(xp, dtype=dtype, device=device_1) # Slip (m).
tau     = tr.zeros_like(xp, dtype=dtype, device=device_1) # Shear stress (Pa).
f       = tr.zeros_like(xp, dtype=dtype, device=device_1) # Stress transfer functional (Pa).
tau_ini = tr.zeros_like(xp, dtype=dtype, device=device_1) # tau_0 in Lapusta et al., (2000) (Pa).
def_del = tr.zeros_like(xp, dtype=dtype, device=device_1) # Slip deficit (m).

# Set initial slip rate.
if mirror:
    V += Vpl
else:
    W1 = -25.
    V += Vpl - 0.9 * Vpl / tr.cosh((xp - W1)/1.e3)

# Assume steady-state shear stress at the initial.
if Frict == 'reg_RSF_AG':
    # This is for SCEC benchmark problem BP1-FD and BP1_QD.
    if not qd:
        tau_ini = 0.025 * sigma * tr.asinh((0.5*V/V0) * tr.exp( (f0 + b*tr.log(V0/V))/0.025 ))
        tau = tau_ini.clone()
    else:
        tau_ini = 0.025 * sigma * tr.asinh((0.5*V/V0) * tr.exp( (f0 + b*tr.log(V0/V))/0.025 )) + eta * V
        tau = tau_ini.clone()
else:
    tau = (f0 + (a - b) * tr.log(V/V0)) * sigma

# Set steady-state state variables for a given slip rate.
if Frict == 'reg_RSF_AG':
    # This is for SCEC benchmark problem BP1-FD and BP1_QD.
    if not qd:
        state += (L / V0) * tr.exp( (a/b) * tr.log( (2.*V0/V) * tr.sinh((tau_ini)/A) ) - f0/b)
    else:
        state += (L / V0) * tr.exp( (a/b) * tr.log( (2.*V0/V) * tr.sinh((tau_ini - eta * V)/A) ) - f0/b)
else:
    state += L / V

# Initial Fourier coefficients of displacement and slip rate.
if not mirror:  # Without mirror.
    Dell = tr.fft.rfft(def_del, n=nconv)
    d_Dell = tr.fft.rfft(V-Vpl, n=nconv)
else: # With mirror.
    pad = tr.zeros(int(Nele*(nperi-1)/nperi), dtype=dtype, device=device_1)
    Dell = dct.dct(tr.cat((def_del, pad), dim=0)) # torch_dct requires explicit zero-padding.
    d_Dell = dct.dct(tr.cat((V-Vpl, pad), dim=0))

FaultParams = {'a': a,
               'b': b,
               'L': L,
               'sigma': sigma,
               'xi': xi}

FieldVariables = {'V': V,
                  'state': state,
                  'delta': delta,
                  'tau': tau,
                  'Dell': Dell,
                  'd_Dell': d_Dell,
                  'f': f,
                  'tau_ini': tau_ini}

#######################   Save variables by pickle   #########################
utils.save('Output{}FaultParams.pkl'.format(fname), FaultParams)
utils.save('Output{}FieldVariables.pkl'.format(fname), FieldVariables)
##############################################################################