import jax.numpy as jnp
from .constants import C_um_ps, TOLERANCE_WAVELENGTH


class SimParams:
    """Simulation parameters including time and frequency grids.
    
    This class is the reference for simulation grid parameters,
    allowing multiple Pulse objects to share the same underlying grid data
    to optimize memory usage.
    
    Attributes:
        t (jnp.ndarray): Time grid points (ps).
        dt (float): Time step (ps).
        n_points (int): Number of time points.
        center_freq_thz (float): Center frequency (THz).
        omega_0 (float): Center angular frequency (rad/ps).
        freqs_abs_thz (jnp.ndarray): Absolute frequencies (THz).
        wavelengths_um (jnp.ndarray): Wavelengths (µm).
    """
    
    def __init__(self, t: jnp.ndarray, center_freq_thz: float):
        """Initialize SimParams.

        Args:
            t (jnp.ndarray): Time grid points (ps).
            center_freq_thz (float): Center frequency (THz).
        """
        self.update(t, center_freq_thz)

    def update(self, t: jnp.ndarray, center_freq_thz: float) -> None:
        """Update simulation parameters.
        
        Args:
            t (jnp.ndarray): Time grid points (ps).
            center_freq_thz (float): Center frequency (THz).
        """
        if t.ndim != 1:
            raise ValueError(
                f"Time grid `t` must be a 1-D array, got {t.ndim}D array with shape {t.shape}."
            )
            
        self.t = t.astype(jnp.float64)
        self.dt = (self.t[1] - self.t[0]).astype(jnp.float64)
        self.n_points = self.t.size

        self.center_freq_thz = center_freq_thz
        self.omega_0 = 2 * jnp.pi * self.center_freq_thz

        # Frequency grids
        self._freqs_relative_thz = jnp.fft.fftfreq(self.t.size, self.dt)
        self.freqs_abs_thz = self.center_freq_thz + self._freqs_relative_thz
        self._freqs_relative_shifted_thz = jnp.fft.fftshift(self._freqs_relative_thz)
        self._freqs_absolute_shifted_thz = jnp.fft.fftshift(self.freqs_abs_thz)
        self.omega_abs = 2 * jnp.pi * self.freqs_abs_thz
        self._omega_relative = 2 * jnp.pi * self._freqs_relative_thz

        self.center_wavelength_um = C_um_ps / self.center_freq_thz
        # Avoid division by zero if frequency is 0 (unlikely for optical freqs but good practice)
        self.wavelengths_um = C_um_ps / jnp.where(
            self._freqs_absolute_shifted_thz == 0, 
            TOLERANCE_WAVELENGTH, 
            self._freqs_absolute_shifted_thz
        )

