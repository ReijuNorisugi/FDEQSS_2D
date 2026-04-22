# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

# This is mainly based on the numerical technique, spectral boundary integral equation method,
# by Lapusta et al. (2000); Noda (2021); and Romanet & Ozawa (2022).
# This is writen by PyTorch for fast and efficient parallel computation on multiple GPUs and CPUs.
# The whole code is based on the SI unit.

# You can edit this file to execute the simulation with parameter file _params_{CL}.py.
# You should place the parameter file in the directory of corresponding constitutive law.
# e.g., /Figures/RSF_AG/test/_params_RSF.py

# Choose Constitutive Law (CL). The followings are available.
# 'RSF': Rate- and state-dependent friction law (Dieterich, 1978, 1979; Ruina, 1983).
#        Aging law is only available now. Standard form and regularized form are available.
# 'RRF': Rate- and roughness-dependent fault constitutive law (Norisugi & Noda, 2026).
# 'mRRF': Modified version of rate- and roughness-dependent law (Norisugi & Noda, 2026).
CL = 'RSF'
#CL = 'RRF'  # Test script.
#CL = 'mRRF' # Test script.

# Set filename where _params_{CL}.py locates.
# The default is SCEC community validation problem BP1-FD.
# https://strike.scec.org/cvws/seas/benchmark_descriptions.html
fname = '/reg_RSF_AG/test/'
#fname = '/RSF_AG/test/' # Example of standard RSF-AG form in an infinit space.
#fname = '/RRF/test/'  # Test script.
#fname = '/mRRF/test/' # Test script.

# If you want to restart the simulation,
# activate "RESTART" and set tmax for the successive simulation.
# Do not activate it at the first trial.
RESTART = False
tmax_restart = 1.e9

# Import the module for driving simulation.
import os
import logging
from pathlib import Path
from datetime import datetime
from sbiem_modules import driver
dir_name = Path(__file__).resolve().parent.name

# Keep log to check the progress.
# You can check the directory /Log/RSF_AG/test/ for the default scenario.
os.makedirs('Log{}'.format(fname), exist_ok=True)
logging.basicConfig(filename='Log{}log_{}.log'.format(fname, datetime.now().strftime('%Y%m%d_%H')),
                    level=logging.INFO, format='%(message)s', filemode='a', force=True)

# Run simulation.
dr = driver.Start(fname, CL, RESTART, tmax_restart, dir_name)
if __name__ == '__main__':
    dr.start()