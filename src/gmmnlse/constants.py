# Physical constants
C_m_s = 299792458  # Speed of light in vacuum (m/s)
C_m_ps = C_m_s * 1e-12  # Speed of light in meters per picosecond
C_um_ps = C_m_s * 1e-6  # Speed of light in micrometers per picosecond (kept for backward compatibility)
C_um_s = C_m_s * 1e6  # Speed of light in micrometers per second

# Vacuum permittivity
EPSILON_0_F_m = 8.8541878128e-12  # Vacuum permittivity in F/m
EPSILON_0_m_ps = EPSILON_0_F_m * 1e-12  # Vacuum permittivity in F·ps²/(m·C²)

# Quantum mechanics
H_PLANCK = 6.62607015e-34  # Planck's constant (J·s)

# Numerical tolerances
TOLERANCE_ZERO = 1e-15  # Tolerance for zero comparisons
TOLERANCE_DIVISION = 1e-30  # Minimum value for safe division
TOLERANCE_WAVELENGTH = 1e-20  # Minimum wavelength to avoid division by zero

# Mathematical constants
import jax.numpy as jnp
TWO_PI = 2.0 * jnp.pi  # 2π

__all__ = [
    'C_m_s', 'C_m_ps', 'C_um_ps', 'C_um_s',
    'EPSILON_0_F_m', 'EPSILON_0_m_ps',
    'H_PLANCK',
    'TOLERANCE_ZERO', 'TOLERANCE_DIVISION', 'TOLERANCE_WAVELENGTH',
    'TWO_PI',
]
