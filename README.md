# FDEQSS_2D
PyTorch-based numerical simulation code for 2D Fully Dynamic Earthquake Sequence Simulation.

![](sbiem_modules/zzz/top.png)

## Overview
This repository provides a numerical simulation framework to simulate fully dynamic earthquake sequences by coupling the elasticity and rate-and-state friction laws.  The code enalbes a long term simulation of earthquake sequence, including inter-seismic stress accumulation, spontaneous nucleation of dynamic events, dynamic rupture propagation, and post-seismic relaxation, within a unified physical framework. The numerical scheme is mainly based on the previous works Lapusta et al. (2000), Noda (2021), and Romanet & Ozawa (2022).

This code is designed for reserach use, with a focus on flexibility, reproducibility, and computational efficiency. It supports rapid prototyping of new physical models and is suitable for large-scale simulations on modern hardware.

Currently available physical assumptions are below:
### Medium
・Linearly elastic body.

### Boundary condition
・Infinite space.
・Half infinite space.

### Geometry
・Flat fault.

### Rupture mode
・Mode I, II, III, and IV (crustal plane model).

### On-fault constitutive law
・Rate- and state-dependent friction (RSF) law, regularized form, aging law.
・RSF law, standard form, aging law.


## Installation
I prepare the virtual environment for running this code by docker.
At first, please install docker to your machine.
You can find the information for docker from the link below.
link

After installing docker, please clone this library.

If you want a code which corresponds to the past publications, please see the taged versions.
git clone 

## Examples
I provide the movie to show how to run the test script.

## References
