"""Quantum noise utilities for GMMNLSE solver."""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from .common import _return_float_if_possible
from ..constants import H_PLANCK, TOLERANCE_ZERO, TOLERANCE_DIVISION

ScalarLike = Union[float, jnp.ndarray]


def _calculate_noise_amplitude_per_bin(
    freq_hz: jnp.ndarray,
    delta_f: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Calculate quantum noise amplitude per frequency bin.
    
    Args:
        freq_hz: Frequency in Hz (can be array).
        delta_f: Frequency resolution in Hz (can be scalar or array).
    
    Returns:
        Noise amplitude: sqrt(h * freq * delta_f) for positive frequencies, 0 otherwise.
    """
    positive_mask = freq_hz > 0.0
    return jnp.where(
        positive_mask,
        jnp.sqrt(H_PLANCK * freq_hz * delta_f),
        0.0,
    )


def quantum_noise(
    pulse=None,
    sim_params=None,
    dt_ps=None,
    n_points=None,
    center_freq_thz=None,
) -> ScalarLike:
    """Calculate quantum noise floor based on one photon per frequency bin.
    
    Unified function that accepts either a Pulse object or SimParams, or individual
    simulation parameters. The quantum noise floor is an absolute physical limit:
    one photon per frequency bin, independent of signal strength.
    
    Args:
        pulse: Pulse object (provides dt, n_points, center_freq_thz). If provided,
            other parameters are ignored.
        sim_params: SimParams object (provides dt, n_points, center_freq_thz). 
            Used if pulse is None.
        dt_ps: Time step in picoseconds. Required if pulse and sim_params are None.
        n_points: Number of time points. Required if pulse and sim_params are None.
        center_freq_thz: Center frequency in THz. Required if pulse and sim_params are None.
    
    Returns:
        float: Absolute quantum noise power in Watts (one photon per frequency bin).
    """
    # Extract parameters from pulse or sim_params
    # Use stop_gradient to prevent tracing through boundary setup calculations
    if pulse is not None:
        dt_ps_val = float(jax.lax.stop_gradient(pulse.dt))
        n_points_val = int(jax.lax.stop_gradient(pulse.n_points))
        center_freq_thz_val = float(jax.lax.stop_gradient(pulse.center_freq_thz))
    elif sim_params is not None:
        dt_ps_val = float(sim_params.dt)
        n_points_val = int(sim_params.n_points)
        center_freq_thz_val = float(sim_params.center_freq_thz)
    else:
        if dt_ps is None or n_points is None or center_freq_thz is None:
            raise ValueError(
                "Must provide either pulse, sim_params, or all of (dt_ps, n_points, center_freq_thz)"
            )
        dt_ps_val = float(dt_ps)
        n_points_val = int(n_points)
        center_freq_thz_val = float(center_freq_thz)
    
    # Calculate quantum noise at center frequency (one photon per frequency bin)
    dt_seconds = dt_ps_val * 1e-12
    time_window_seconds = dt_seconds * n_points_val
    delta_f = 1.0 / time_window_seconds  # Frequency resolution (Hz)
    center_freq_hz = center_freq_thz_val * 1e12

    noise_power_w = _calculate_noise_amplitude_per_bin(
        jnp.asarray([center_freq_hz]), delta_f
    )[0] ** 2  # Power = amplitude^2
    
    return _return_float_if_possible(noise_power_w)


def apply_noise_to_boundary(
    boundary_window: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    """Apply quantum noise floor to boundary window, only modifying zero values.
    
    This function is specifically designed for boundary windows where zero values
    should get a noise baseline, but nonzero values are preserved as-is.
    
    The noise floor uses a fixed small threshold appropriate for the normalized
    boundary window scale (0-1), independent of signal power.
    
    Args:
        boundary_window: Boundary window array (values typically 0-1).
    
    Returns:
        Boundary window with noise floor applied to zero values only.
        Same type as input (numpy or jax array).
    """
    if len(boundary_window) == 0:
        return boundary_window
    
    noise_floor_normalized = TOLERANCE_DIVISION
    
    is_jax = isinstance(boundary_window, jnp.ndarray)
    
    if is_jax:
        noise_floor_arr = jnp.asarray(noise_floor_normalized, dtype=boundary_window.dtype)
        zero_mask = jnp.abs(boundary_window) < TOLERANCE_ZERO
        return jnp.where(zero_mask, noise_floor_arr, boundary_window)
    else:
        noise_floor_arr = np.asarray(noise_floor_normalized, dtype=boundary_window.dtype)
        zero_mask = np.abs(boundary_window) < TOLERANCE_ZERO
        result = np.where(zero_mask, noise_floor_arr, boundary_window)
        return np.asarray(result)


def add_shot_noise_to_field(
    field: jnp.ndarray,
    dt_ps: float,
    n_points: int,
    center_freq_thz: float,
    noise_seed: int | None = None,
) -> jnp.ndarray:
    """Add quantum shot noise to a pulse field.
    
    Implements quantum shot noise by adding one photon per frequency bin,
    respecting the Hermitian symmetry of the frequency-domain representation.
    
    Args:
        field: Complex electric field array, shape (n_modes, n_points).
        dt_ps: Time step in picoseconds.
        n_points: Number of time points.
        center_freq_thz: Center frequency in THz.
        noise_seed: Random seed for noise generation. If None, uses seed 0.
    
    Returns:
        Field array with shot noise added, same shape as input.
    """
    # Use unified noise calculation
    dt_seconds = dt_ps * 1e-12
    time_window_seconds = dt_seconds * n_points
    delta_f = 1.0 / time_window_seconds
    
    # Relative frequencies returned by fftfreq are in units of Hz after scaling
    f = jnp.fft.fftfreq(n_points, dt_ps) * 1e12
    real_f = f + center_freq_thz * 1e12
    noise_amplitude = _calculate_noise_amplitude_per_bin(real_f, delta_f)

    def _complex_standard_normal(rng_key, size):
        key_real, key_imag = jax.random.split(rng_key)
        real_part = jax.random.normal(key_real, shape=(size,))
        imag_part = jax.random.normal(key_imag, shape=(size,))
        return (real_part + 1j * imag_part) / jnp.sqrt(2.0)

    if noise_seed is not None:
        key = jax.random.PRNGKey(noise_seed)
    else:
        key = jax.random.PRNGKey(0)

    nyquist_idx = n_points // 2
    has_nyquist = (n_points % 2 == 0)
    if has_nyquist:
        pos_indices = jnp.arange(1, nyquist_idx)
    else:
        pos_indices = jnp.arange(1, nyquist_idx + 1)
    neg_indices = (-pos_indices) % n_points

    n_modes = field.shape[0]
    field_with_noise = field.copy()
    
    for mode_idx in range(n_modes):
        noise_freq = jnp.zeros(n_points, dtype=jnp.complex128)

        key, dc_key = jax.random.split(key)
        noise_freq = noise_freq.at[0].set(
            noise_amplitude[0] * jax.random.normal(dc_key)
        )

        if has_nyquist and nyquist_idx > 0:
            key, nyq_key = jax.random.split(key)
            noise_freq = noise_freq.at[nyquist_idx].set(
                noise_amplitude[nyquist_idx] * jax.random.normal(nyq_key)
            )

        if pos_indices.size > 0:
            key, pos_key = jax.random.split(key)
            rand_complex = _complex_standard_normal(pos_key, pos_indices.size)
            scaled = noise_amplitude[pos_indices] * rand_complex
            noise_freq = noise_freq.at[pos_indices].set(scaled)
            noise_freq = noise_freq.at[neg_indices].set(jnp.conj(scaled[::-1]))

        noise_time = jnp.fft.ifft(noise_freq)
        field_with_noise = field_with_noise.at[mode_idx, :].add(noise_time)
    
    return field_with_noise

