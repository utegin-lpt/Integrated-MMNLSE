"""
Finite Difference Mode Solver

Based on modesolverpy: https://github.com/jtambasco/modesolverpy
Implements semi-vectorial and fully-vectorial finite difference mode solvers.

References:
- A. B. Fallahkhair, K. S. Li and T. E. Murphy, 
  "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides", 
  J. Lightwave Technol. 26(11), 1423-1431, (2008).
"""

# from .semi_vectorial import ModeSolverSemiVectorial  # TODO: Not yet implemented
from .fully_vectorial import FullyVectorial
from .mode import Mode

__all__ = ['FullyVectorial', 'Mode']
