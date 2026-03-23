# FDEQSS_2D
PyTorch-based numerical simulation code for 2D Fully Dynamic Earthquake Sequence Simulation.

![](sbiem_modules/zzz/top.png)

## Overview

This repository provides a numerical simulation framework for **fully dynamic earthquake sequence simulations** by coupling:

- Elastodynamics
- Rate-and-state friction (RSF) laws

The code enables long-term simulations of earthquake cycles within a unified physical framework, including:

- Inter-seismic stress accumulation
- Spontaneous nucleation of dynamic events
- Dynamic rupture propagation
- Post-seismic relaxation

The numerical scheme is mainly based on previous works:

- [Lapusta et al. (2000)](https://doi-org.kyoto-u.idm.oclc.org/10.1029/2000JB900250)
- [Noda (2021)](https://doi.org/10.1186/s40623-021-01465-6)
- [Romanet & Ozawa (2022)](https://doi-org.kyoto-u.idm.oclc.org/10.1785/0120210178)

This code is designed for **research use**, with a focus on:

- Flexibility
- Reproducibility
- Computational efficiency

It supports rapid prototyping of new physical models and is suitable for large-scale simulations on modern hardware.



## Currently available physics

### Medium
- Linearly elastic body

### Boundary Conditions
- Infinite space
- Half-infinite space

### Geometry
- Flat fault

### Rupture Modes
- Mode I
- Mode II
- Mode III
- Mode IV (crustal plane model)

### On-fault Constitutive Laws
- RSF law, **regularized form** with aging law
- RSF law, **standard form** with aging law


## Installation

This project uses Docker to provide a fully reproducible runtime environment.  
You do not need to manually install Python or any dependencies, and your local environment is not contaminated.

### 1. Install Docker

First, install Docker on your system:

https://www.docker.com/get-started

### 2. Pull Docker image

The simulation environment is distributed as a pre-built Docker image.
**Note**: The Docker image is approximately **20 GB** in size. Please ensure that you have sufficient disk space.

#### For arm64 machines:
```bash
docker pull reiju123/sbiem:latest
```

#### For x86_64 machines:
```bash
docker pull reiju123/sbiem_x86_64:latest
```

You can verify the installation:

```bash
docker images
```

### 3. Clone this repository

Clone the source code from GitHub:

```bash
git clone https://github.com/ReijuNorisugi/FDEQSS_2D.git
cd FDEQSS_2D
```

If you want to reproduce results from specific publications or stable versions,
please checkout a tagged release:

```bash
git checkout v0.2.0
```

The correspondance between each tag and version is summarized in tag.txt.

## Examples
I provide the movie to show how to run the test script.


## References
- Lapusta, N., Rice, J. R., Ben‐Zion, Y., & Zheng, G. (2000). Elastodynamic analysis for slow tectonic loading with spontaneous rupture episodes on faults with rate‐and state‐dependent friction. Journal of Geophysical Research: Solid Earth, 105(B10), 23765-23789.

- Noda, H. (2021). Dynamic earthquake sequence simulation with a SBIEM without periodic boundaries. Earth, Planets and Space, 73(1), 137.

- Romanet, P., & Ozawa, S. (2022). Fully dynamic earthquake cycle simulations on a nonplanar fault using the spectral boundary integral element method (sBIEM). Bulletin of the Seismological Society of America, 112(1), 78-97.