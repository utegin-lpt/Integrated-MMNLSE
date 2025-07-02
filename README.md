# Integrated-MMNLSE
This repository contains codes to calculate modes, dispersions and mode coupling coefficient, and solve the Generalized Multimode Nonlinear Schrodinger Equation (GMMNLSE) for integrated waveguides. 

# Code
Code/modesolver contains the code to solve the modes, which can be run from pymodesolver.py.

Code/simulation contains the code to solve the GMMNLSE, which can be run from sim.py. The simulation solver can be adjusted based on the available solver on the operator.py.
Please enable Jax-support for 64-bit floating number to prevent NaN value while calculating higher order dispersion.

# System requirements 
- Python 3
- scipy
- numpy
- Jax

# Related work
TBA

# License
This project is covered under the Creative Common (CC BY NC) License. The data and code are avaiable for non-commercial research purposes only with proper citation to aforementioned manuscript.
