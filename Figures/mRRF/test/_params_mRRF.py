# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260323.
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
Initial = True  # True: You can check parameter settings without runnning the whole simulation.
act_res = True   # True: Save files for restarting.
mirror = False   # True: Hlaf infinite-space like Lapusta et al. (2000).
qd = False       # False: Fully-dynamic simulation.  True: Quasi-dynamic simulation.
mode = 'III'     # Rupture modes I(opening), II(in-plane), III(anti-plane), and IV(crustal plane) are available.
sparse = True    # Allow dense spatial snapshot at sparse time steps.         !!! Storage consuming !!!
downsample = 1   # Downsampling for spatially dense output.
snap_EQ = True   # Allow dense spatial snapshot at before and after an event. !!! Storage consuming !!!
dense = True     # Allow every time step output on sparse stations.           !!! Storage consuming !!!
outerror = False  # Output error between full and half step solutions. Ture enforces ['RO'] time-marching scheme.
Frict = 'mRRF' # Choose fault constitutive law. RSF_AG and RSF_SL, are available now.
Stepper = 'RO'   # Choose scheme for time step evolution. Romanet & Ozawa (2022) ['RO'] 
                 # and Lapusta et al. (2000) ['LR'] is implemented.
rmPB = True    # True removes periodic boundaries. Recommended if you have resources.
Cutoff = True   # Introduce cut-off slip rate for direct effect.

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
tmax = 1.e8
##########################

###############################   Important   #################################
# When you choose ['RO'] for time stepper, you should set error0.
# If the simulation does not converge at initiation, use a smaller value.
error0 = 1.e-6  # Torlelance for time step. See Romanet & Ozawa, (2022).
dt0_guess = 1.e2   # Initial guess of dt.
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

Cs = 3.e3  # S-wave velocity (m/s).
Cp = 5.e3  # P-wave velocity (m/s).
mu = 30.e9 # Rigidity (Pa).
rho = mu / (Cs**2)  # Density (kg/m^3).
nu = 1 - ((Cp/Cs)**2) / (2 * ((Cp/Cs)**2 - 1)) # Poisson's ratio.

f0 = 0.6     # Referrence friction (non-dimentinal).
V0 = 3.e-9   # Referrence velocity for standard RSF law (m/s).

Vpl = 3.e-9  # Loading rate (m/s).
beta_min = 0.5  # CFL. For defining minimum time step.
tau_rate = 0.   # Uniform stressing rate.

# If you choose mode IV (crustal-plane model), please set the following.
H = 5.e3   # Depth for averaging. See Lehner (1981).
H = (0.25*np.pi) * H  # Do not change.
Z = 1 / (1 - nu)      # Argument for crustal plane model. See Lehner (1981).

# Radiation dumping coefficient. Do not change here.
if mode == 'II' or mode == 'III' or mode == 'IV':
    eta = 0.5 * rho * Cs
elif mode == 'I':
    eta = 0.5 * rho * Cp

# Set coordinate.
if not mirror:        # Without mirror.
    lam = 2000.      # Size of computational regeme (m).
    Nele = 4096        # Total cell number for computation.
    hcell = lam/Nele  # Cell size (m).
    
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
    # And not that wavenumber-dependent truncation is not implemented.
    if rmPB: # Without periodic boundaries.
        ratio = 2.0  # You can take longer truncation window. 3 will be OK.
    else:    # With periodic boundaries.
        ratio = 1.0  # Choose longer value as possible as you can.
    # * * * * * * * * * * * * * * * * * #
else:                 # With mirror.
    lam = 48000.      # Size of computational regime. Mirror part is excluded.
    Nele = 1024       # Total cell numer for computation.
    hcell = lam/Nele  # Cell size.
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
              'Cutoff': Cutoff,
              'num_GPU': num_GPU,
              'tmax': tmax,
              'stations_uni': stations_uni,
              'stations': stations,
              'G2G': G2G
              }

Devices = {'device_1': device_1}

Medium = {'Cs': Cs,
          'Cp': Cp,''
          'mu': mu,
          'nu': nu,
          'eta': eta,
          'f0': f0,
          'V0': V0,
          'Vpl': Vpl,
          'tau_rate': tau_rate,
          'beta_min': beta_min,
          'Z': Z,
          'H': H,
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
          'dt0_guess': dt0_guess
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
    return param_in + (param_out - param_in) * 0.5 * (1 - tr.tanh((w2/(tr.abs(x)-w1-w2))+(w2/(tr.abs(x)-w1))))

def smooth_brend(x):
    return tr.where(x <= 0, 0.0, tr.where( x >= 1, 1.0, tr.exp(-1/x) / (tr.exp(-1/x) + tr.exp(-1/(1-x))) ))

# Frictional parameters, depending on what you use for constitutive law.
if Frict == 'mRRF':
    Lf = 1.e1    #10m
    D_inv = 4*(4.e-4)*(np.pi**3)/Lf  # factor 4 * 10**-4 is for Ohnaka's measurement.
    n_up = 11
    n_low = 0
    k_size = int(n_up-n_low+1)
    n_array = 2**np.linspace(n_low, n_up, k_size)
    
    k_size = len(n_array)

    k = np.zeros(k_size)
    for i in range(k_size):
        k[i] = 2*np.pi*n_array[i]/Lf
    n_array = tr.as_tensor(n_array, dtype=tr.float64).to(device_1)
    k = tr.as_tensor(k, dtype=tr.float64).to(device_1)
    ku = k[0]
    kl = k[-1]
    k2 = k**2

    dim = -3.
    Y_bar = tr.sqrt(D_inv*(k**dim))    # Y_bar^2 should be proportional to k^-3 (observation of fractured surface)

    Dc = 1.e0    # (m)

    a =          tr.zeros_like(xp, dtype=dtype, device=device_1)
    c =          tr.zeros_like(xp, dtype=dtype, device=device_1)
    f0 =         tr.zeros_like(xp, dtype=dtype, device=device_1)
    sigma =      tr.zeros_like(xp, dtype=dtype, device=device_1)
    alpha =      tr.zeros_like(xp, dtype=dtype, device=device_1)
    beta =       tr.zeros_like(xp, dtype=dtype, device=device_1)
    beta_alpha = tr.zeros_like(xp, dtype=dtype, device=device_1)
    Vc =         tr.zeros_like(xp, dtype=dtype, device=device_1)

    sigma += 50 * (10**6) # Normal stress is uniform on the fault.
    alpha += 1 / (Dc*ku)
    
    
    W1 = 250.
    W2 = 10. * hcell
    W3 = 5. * hcell
    if mode == 'IV':
        W1 = 1000
        W2 = 25
    a_out = 0.02
    a_in = 0.01
    c_out = 0.25
    c_in = 2.
    c_inright = 1.5
    f0_out = 0.7
    f0_in = 0.3
    betaalpha = 10**(-10)
    if Cutoff:
        f0_out = 1.0
        f0_in = 0.6
        c_out = 0.5
        c_in = 1.1 
        c_inright = 0.9
        Vc_out = 1.e2
        Vc_in = 20.e-6
        
    for i in range(len(xp)):
        if tr.abs(xp[i]) > W1 + W2:
            a[i] = a_out
            c[i] = c_out
            f0[i] = f0_out
            beta_alpha[i] = betaalpha
            if Cutoff:
                Vc[i] = Vc_out
        elif tr.abs(xp[i]) < W1:
            a[i] = a_in
            c[i] = c_in
            f0[i] = f0_in
            beta_alpha[i] = betaalpha
            c[i] = (c_in + c_inright) * 0.5 + 0.5 * (c_inright - c_in) * xp[i] / W1
            if Cutoff:
                Vc[i] = Vc_in
        else:
            a[i] = a_in + (a_out - a_in) * 0.5 * (1 - tr.tanh((W2/(tr.abs(xp[i])-W1-W2))+(W2/(tr.abs(xp[i])-W1))))
            f0[i] = f0_in + (f0_out - f0_in) * 0.5 * (1 - tr.tanh((W2/(tr.abs(xp[i])-W1-W2))+(W2/(tr.abs(xp[i])-W1))))
            #c[i] = c_in + (c_out - c_in) * 0.5 * (1 - tr.tanh((W2/(tr.abs(xp[i])-W1-W2))+(W2/(tr.abs(xp[i])-W1))))
            beta_alpha[i] = betaalpha
            if Cutoff:
                Vc[i] = Vc_in + (Vc_out - Vc_in) * 0.5 * (1 - tr.tanh((W2/(tr.abs(xp[i])-W1-W2))+(W2/(tr.abs(xp[i])-W1))))
            
            
            if xp[i] < 0:
                c[i] = c_in + (c_out - c_in) * 0.5 * (1 - tr.tanh((W2/(tr.abs(xp[i])-W1-W2))+(W2/(tr.abs(xp[i])-W1))))
            else:
                c[i] = c_inright + (c_out - c_inright) * 0.5 * (1 - tr.tanh((W2/(tr.abs(xp[i])-W1-W2))+(W2/(tr.abs(xp[i])-W1))))
            
            if xp[i] <= W1 + W3:
                smooth = smooth_brend((tr.abs(xp[i])-(W1)) / W3)
                c[i] = smooth * c[i] + (1 - smooth) * ((c_in + c_inright) * 0.5 + 0.5 * (c_inright - c_in) * xp[i] / W1)
            elif xp[i] > - W1 - W3:
                smooth = smooth_brend((tr.abs(xp[i])-(W1)) / W3)
                c[i] = smooth * c[i] + (1 - smooth) * ((c_in + c_inright) * 0.5 + 0.5 * (c_inright - c_in) * xp[i] / W1)
    
    beta = beta_alpha * alpha
    
    A = a*sigma
    C = c*sigma
    tauc = f0*sigma
    
    arg = tr.tensor(2., dtype=tr.float64, device=device_1)
    arg = tr.log(arg)
    
    Vww = 1.e-2
    
    Yss = beta[int(ncell/2)] * k * Y_bar / (alpha[int(ncell/2)] * Vpl + beta[int(ncell/2)] * k)
    WW = ( (alpha[int(ncell/2)] * tr.max(c) * sigma[int(ncell/2)]) * tr.sum( n_array * arg * (k**3) * (Yss**2) ) / 
    (tr.sqrt(tr.sum( n_array * arg * (k**2) * (Yss**2) )))
    )
    lambda0 = (9.*np.pi/32.) * mu / WW
    
    logging.info(f'delta x {hcell}')
    logging.info(f'lambda_0 / delta x {(lambda0 / hcell).item()}')
    logging.info(f'For required resolution {3 / (lambda0 / hcell).item()}')
#######################################################################################################



######################    Define initial values   #####################
# Define field variables.
V =       tr.zeros_like(xp, dtype=dtype, device=device_1)           # Slip velocity.
state =   tr.zeros((len(xp), k_size), dtype=dtype, device=device_1) # State variables.
delta =   tr.zeros_like(xp, dtype=dtype, device=device_1)           # Slip discontinuity.
tau =     tr.zeros_like(xp, dtype=dtype, device=device_1)           # Shear traction.
phi =     tr.zeros_like(xp, dtype=dtype, device=device_1)           # Shear strength.
f   =     tr.zeros_like(xp, dtype=dtype, device=device_1)           # Stress transfer fanctional.
tau_ini = tr.zeros_like(xp, dtype=dtype, device=device_1)           # tau_0 in Lapusta et al., (2000).
def_del = tr.zeros_like(xp, dtype=dtype, device=device_1)           # Slip-dificit.
pad =     tr.zeros(int(Nele*(nperi-1)/nperi), dtype=dtype, device=device_1)  # Zero-padding for torch_dct.

# Set initial condition.
W1 = -100.
V += Vpl - 0.95* Vpl / tr.cosh((xp - W1)/50.)

for i in range(len(xp)):
    state[i, :] += beta[i]*k*Y_bar / (alpha[i]*V[i]+beta[i]*k)
if not Cutoff:
    arg = tr.tensor(2., dtype=tr.float64, device=device_1)
    arg = tr.log(arg)
    phi = tauc + C*tr.sqrt(tr.sum(n_array*(k**2)*(state**2)*arg, axis=1))
    tau = A*tr.log(V/V0) + phi
else:
    arg = tr.tensor(2., dtype=tr.float64, device=device_1)
    arg = tr.log(arg)
    phi = tauc + C*tr.sqrt(tr.sum(n_array*(k**2)*(state**2)*arg, axis=1))
    tau = A*tr.log(V/(V+Vc)) + phi

# Initial FFT components.
if not mirror:
    Dell = tr.fft.rfft(def_del, n=nconv) # should be complex
    d_Dell = tr.fft.rfft(V-Vpl, n=nconv)
else:
    Dell = dct.dct(tr.cat((def_del, pad), dim=0))
    d_Dell = dct.dct(tr.cat((V-Vpl, pad), dim=0))

FaultParams = {'Lf': Lf, 'D_inv': D_inv, 'n_up': n_up, 'n_low': n_low, 
               'k_size': k_size, 'n_array': n_array, 'dim': dim, 'Dc': Dc, 
               'k': k, 'Ybar': Y_bar, 'alpha': alpha, 'beta': beta, 
               'a': a, 'c': c, 'f0': f0, 'sigma': sigma, 'tau_ini': tau_ini,
               'tauc': tauc, 'Vc': Vc}

FieldVariables = {'V': V, 'state': state, 'delta': delta, 'tau': tau, 'phi': phi,
                'Dell': Dell, 'd_Dell': d_Dell, 'f': f}

#######################   Save variables by pickle   #########################
utils.save('Output{}FaultParams.pkl'.format(fname), FaultParams)
utils.save('Output{}FieldVariables.pkl'.format(fname), FieldVariables)
##############################################################################