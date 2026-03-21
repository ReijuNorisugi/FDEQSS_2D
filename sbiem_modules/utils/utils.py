# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
"""

# This is the module which summarizes miscellaneous function.
# I try no to make too huge utils script.

import pickle
import logging
import torch as tr

# Function to save data as pickle files.
def save(fname, params):
    with open(fname, 'wb') as f:
        pickle.dump(params, f)

# Function to load pickle files.
def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

# Function to set devices.
def set_device(num_GPU):
    if tr.cuda.is_available() and num_GPU != 0:
        return num_GPU, tr.device('cuda:0')
    else:
        num_GPU = 0
        return num_GPU, 'cpu'

# Function to transform tensor to numpy.
def ttn(tensor):
    if isinstance(tensor, tr.Tensor):
        return tensor.cpu().detach().numpy().copy()
    else:
        return tensor 

# Function to show a log.
def show(fname, qd, mirror, Frict, mode, Stepper, lam, nperi,
         Tw, dtmin, sparse, dense, snap_EQ, outerror):
    
    logging.info('')
    logging.info('********************************')
    logging.info(fname)
    logging.info('')
    logging.info(' * Computational conditions *')
    logging.info('')
    if not qd:
        logging.info(' *  Fully dynamic')
    else:
        logging.info(' *  Quasi dynamic')
    if not mirror:
        logging.info(' *  No Mirror')
    else:
        logging.info(' *  Mirror')
    logging.info(' *  Constitutive law = {}'.format(Frict))
    logging.info(' *  Rupture mode = {}'.format(mode))
    logging.info(' *  Stepper = {}'.format(Stepper))
    logging.info(' ')

    logging.info(' * Coordinate setting *')
    logging.info('')
    logging.info(' *  Domain size = {} km'.format(lam/1e3))
    logging.info(' *  Fault size = {} km'.format(lam/nperi/1e3))
    logging.info(' *  Replication = {}'.format(nperi))
    logging.info(' *  Tw = {} sec'.format(Tw))
    logging.info(' *  dtmin = {} msec'.format(dtmin*1e3))
    logging.info('')

    logging.info(' * Output conditions * ')
    logging.info('')
    if sparse:
        logging.info(' * Sparse Bulk output is available')
    if dense:
        logging.info(' * Dense station output is available')
    if snap_EQ:
        logging.info(' * EQ snapshot is available')
    if outerror:
        logging.info(' * Error output is available')
    logging.info('')