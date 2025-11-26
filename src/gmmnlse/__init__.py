"""
gmmnlse

A JAX-based solver for the Generalized Multimode Nonlinear Schrödinger Equation.
"""

__version__ = "0.1.0"

from .solver import DiffraxSolver, RK4IPSolver
from .pulse import Pulse
from .sim import SimParams
from .waveguide import Waveguide
from .io import load_waveguide_data, WaveguideLoader, load_waveguide_arrays
from .solver_result import SolverResult

__all__ = ['DiffraxSolver', 'RK4IPSolver', 'Pulse', 'SimParams', 'Waveguide', 'load_waveguide_data', 'WaveguideLoader', 'load_waveguide_arrays', 'SolverResult']
