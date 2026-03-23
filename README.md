# FDEQSS_2D
PyTorch-based numerical simulation code for 2D Fully Dynamic Earthquake Sequence Simulation.

![](sbiem_modules/zzz/top.png)

## Overview

This repository provides a numerical simulation framework for **fully dynamic earthquake sequence simulations** by coupling:

- Elastodynamics
- Rate-and-state friction laws

The code enables long-term simulations of earthquake cycles within a unified physical framework, including:

- Inter-seismic stress accumulation
- Spontaneous nucleation of dynamic events
- Dynamic rupture propagation
- Post-seismic relaxation

The numerical scheme is mainly based on previous works:

- [Lapusta et al. (2000)](https://doi-org/10.1029/2000JB900250)
- [Noda (2021)](https://doi.org/10.1186/s40623-021-01465-6)
- [Romanet & Ozawa (2022)](https://doi-org/10.1785/0120210178)

This code is designed for **research use**, with a focus on:

- Flexibility
- Reproducibility
- Computational efficiency

It supports rapid prototyping of new physical models and is suitable for large-scale simulations on modern hardware.

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
