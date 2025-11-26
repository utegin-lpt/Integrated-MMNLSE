"""Boundary window utilities for GMMNLSE solver.

This module provides functions and classes for creating and applying boundary
windows to prevent reflections and aliasing in pulse propagation simulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union
from ..constants import C_m_ps, C_um_s, TOLERANCE_ZERO
from .noise import quantum_noise, apply_noise_to_boundary

__all__ = [
    'create_super_gaussian_time_window',
    'create_super_gaussian_frequency_window',
    'BoundaryHandler',
]


def _estimate_transition_span(
    target_freq_thz: float,
    available_span_thz: float,
    df_thz: float,
    relative_guard: float = 0.02,
    min_bins: int = 8,
) -> float:
    """Estimate a practical transition span so tapering hugs the requested cutoff.

    Args:
        target_freq_thz: Distance between the cutoff frequency and the centre (THz).
        available_span_thz: Remaining span beyond the cutoff (THz).
        df_thz: Frequency resolution of the grid (THz).
        relative_guard: Fraction of the target frequency to keep as guard band.
        min_bins: Minimum number of frequency bins to include in the transition.

    Returns:
        Transition span (THz) clamped to the available span.
    """
    if available_span_thz <= TOLERANCE_ZERO:
        return available_span_thz
    guard_from_bins = max(float(min_bins) * df_thz, df_thz)
    guard_from_fraction = relative_guard * abs(target_freq_thz)
    desired_span = max(guard_from_bins, guard_from_fraction)
    return float(min(available_span_thz, desired_span))

def create_super_gaussian_time_window(
    t: Union[np.ndarray, jnp.ndarray], 
    n_points: int, 
    window_fraction: Union[float, Tuple[float, float]] = 0.5, 
    order: Union[int, Tuple[int, int]] = 20
) -> jnp.ndarray:
    """Create a super-Gaussian time window to damp pulse edges.
    
    Creates a super-Gaussian window in the time domain that damps the edges
    of the pulse to prevent reflections at the boundaries. The window is 1.0
    near the center and exponentially damps toward the edges.
    
    Supports symmetric (single value) or asymmetric (tuple [min, max]) windows.
    For asymmetric windows, min applies to the left edge and max to the right edge.
    
    Args:
        t: Time array.
        n_points: Number of time points.
        window_fraction: Fraction of time window where window is near 1.0.
            Can be a single float (symmetric) or tuple [left, right] (asymmetric).
            Defaults to 0.5.
        order: Super-Gaussian order. Higher values create sharper transitions.
            Can be a single int (symmetric) or tuple [left, right] (asymmetric).
            Defaults to 20.
    
    Returns:
        jnp.ndarray: Time window function, shape (n_points,), values between 0 and 1.
    """
    t_np = np.asarray(t, dtype=float)
    if t_np.ndim != 1 or n_points <= 0:
        return jnp.ones_like(t)

    center = 0.0 if np.any(np.isclose(t_np, 0.0)) else float(np.mean(t_np))
    
    # Normalize single values to tuples for uniform handling
    if isinstance(window_fraction, (int, float)):
        window_fraction = (float(window_fraction), float(window_fraction))
    else:
        window_fraction = (float(window_fraction[0]), float(window_fraction[1]))
    
    if isinstance(order, (int, np.integer)):
        order = (int(order), int(order))
    else:
        order = (int(order[0]), int(order[1]))
    
    window_fraction_left, window_fraction_right = window_fraction
    order_left, order_right = order
    
    offsets = t_np - center
    half_span = np.max(np.abs(offsets))
    
    # Create left side (negative offsets)
    left_mask = offsets < 0
    left_offsets = np.abs(offsets[left_mask])
    if left_offsets.size > 0:
        core_width_left = half_span * window_fraction_left
        gexpo_left = 2 * order_left
        margin_left = max(half_span - core_width_left, TOLERANCE_ZERO)
        sigma_left = margin_left / (2.0 * np.sqrt(np.log(2.0)))
        
        excess_left = np.maximum(left_offsets - core_width_left, 0.0)
        taper_left = np.exp(-((excess_left ** gexpo_left) / (2.0 * (sigma_left ** gexpo_left)))) ** order_left
        
        window_left = np.ones_like(left_offsets)
        window_left[left_offsets > core_width_left] = taper_left[left_offsets > core_width_left]
    else:
        window_left = np.array([])
    
    # Create right side (positive offsets)
    right_mask = offsets >= 0
    right_offsets = np.abs(offsets[right_mask])
    if right_offsets.size > 0:
        core_width_right = half_span * window_fraction_right
        gexpo_right = 2 * order_right
        margin_right = max(half_span - core_width_right, TOLERANCE_ZERO)
        sigma_right = margin_right / (2.0 * np.sqrt(np.log(2.0)))
        
        excess_right = np.maximum(right_offsets - core_width_right, 0.0)
        taper_right = np.exp(-((excess_right ** gexpo_right) / (2.0 * (sigma_right ** gexpo_right)))) ** order_right
        
        window_right = np.ones_like(right_offsets)
        window_right[right_offsets > core_width_right] = taper_right[right_offsets > core_width_right]
    else:
        window_right = np.array([])
    
    # Combine left and right windows
    window = np.ones_like(t_np)
    if left_mask.any():
        window[left_mask] = window_left
    if right_mask.any():
        window[right_mask] = window_right
    
    return jnp.asarray(window)


def create_super_gaussian_frequency_window(
    freqs_thz: Union[np.ndarray, jnp.ndarray],
    window_fraction: float = 0.6,
    order: int = 20,
    damp_side: str = "high",
    center_freq_thz: Optional[float] = None,
    window_fraction_low: float = 0.6,
    window_fraction_high: float = 0.6,
    order_low: Optional[int] = None,
    order_high: Optional[int] = None,
    span_low: Optional[float] = None,
    span_high: Optional[float] = None,
    transition_span_low: Optional[float] = None,
    transition_span_high: Optional[float] = None,
) -> jnp.ndarray:
    """Create an asymmetric super-Gaussian frequency window.

    Applies a super-Gaussian roll-off on the requested side(s) of the spectrum.
    The cutoff frequency is determined by the window fraction and span, set to correspond
    to the point where the window value is 10^-3 (-30dB in 10*log10 scale).
    
    Args:
        freqs_thz: Frequency grid in THz (FFT ordering).
        window_fraction: Legacy fraction used if per-side values are not provided.
        order: Base super-Gaussian order (overall exponent is 2*order). Used if order_low
            or order_high are not specified.
        damp_side: "low", "high", or "both" sides to damp relative to the center.
        center_freq_thz: Optional explicit center frequency. If None, zero/mean is used.
        window_fraction_low: Fraction for the low-frequency side 3dB cutoff.
        window_fraction_high: Fraction for the high-frequency side 3dB cutoff.
        order_low: Super-Gaussian order for the low-frequency side. If None, uses order.
        order_high: Super-Gaussian order for the high-frequency side. If None, uses order.
        span_low/span_high: Optional per-side spans used to convert the fractional cutoffs
            back into absolute frequencies.
        transition_span_low/transition_span_high: Legacy parameters, ignored in new implementation.
            The transition steepness is now determined solely by the order.
    
    Returns:
        jnp.ndarray window of shape (n_points,) with values in [0, 1].
    """ 
    freqs_np = np.asarray(freqs_thz, dtype=np.float64)
    if freqs_np.ndim != 1 or freqs_np.size == 0:
        return jnp.ones_like(freqs_thz)

    if center_freq_thz is None:
        # Assume grids are zero-centred (relative frequencies). Fall back to mean otherwise.
        center = 0.0 if np.any(np.isclose(freqs_np, 0.0)) else float(np.mean(freqs_np))
    else:
        # If explicitly set to 0.0, use 0.0 directly
        center = 0.0 if center_freq_thz == 0.0 else float(center_freq_thz)
    

    window = np.ones_like(freqs_np, dtype=np.float64)
    eps = 1e-18
    damp_side_norm = damp_side.lower()
    if damp_side_norm not in {"low", "high", "both"}:
        raise ValueError(f"damp_side must be 'low', 'high', or 'both', got '{damp_side}'")

    # Set orders for each side
    order_low_val = order_low if order_low is not None else order
    order_high_val = order_high if order_high is not None else order

    def _apply_side(
        mask: np.ndarray,
        freqs_side: np.ndarray,
        fraction: float,
        side_order: int,
        span_override: float | None = None,
    ) -> None:
        if freqs_side.size == 0:
            return
        # Use span_override if provided
        if span_override is not None:
            span = float(span_override)
        else:
            span = float(np.max(freqs_side))
        
        if span <= eps:
            return

        # cutoff is the 30dB point on the plot (amplitude = 10^-3)
        # Note: The plotter uses 10*log10(amplitude), so to get -30dB on the plot
        # we need amplitude = 10^(-30/10) = 10^-3.
        # In standard 20*log10(amplitude) terms, this is -60dB.
        cutoff = span * fraction
        
        if cutoff <= eps:
            window[mask] = 0.0
            return
            
        # Standard super-Gaussian formula:
        # W(f) = exp( -C * (f/cutoff)^(2*n) )
        # We want W(cutoff) = 10^-3
        # exp(-C) = 10^-3 => C = 3.0 * ln(10)
        
        norm_freq = freqs_side / cutoff
        exponent = 2 * side_order
        
        # Calculate argument for exp
        c_factor = 3.0 * np.log(10.0)
        arg = c_factor * (norm_freq ** exponent)
        taper = np.exp(-arg)
        window[mask] *= taper

    # High-frequency (short-wavelength) side: freqs > center
    if damp_side_norm in {"high", "both"}:
        high_mask = freqs_np > center
        freqs_high = freqs_np[high_mask] - center
        fraction_high = window_fraction_high if window_fraction_high is not None else window_fraction
        span_high_val = span_high if span_high is not None else None
        _apply_side(
            high_mask,
            freqs_high,
            fraction_high,
            order_high_val,
            span_override=span_high_val,
        )

    # Low-frequency side: freqs < center (use absolute distance)
    if damp_side_norm in {"low", "both"}:
        low_mask = freqs_np < center
        freqs_low = center - freqs_np[low_mask]
        fraction_low = window_fraction_low if window_fraction_low is not None else window_fraction
        span_low_val = span_low if span_low is not None else None
        _apply_side(
            low_mask,
            freqs_low,
            fraction_low,
            order_low_val,
            span_override=span_low_val,
        )
    
    return jnp.asarray(window)


class BoundaryHandler:
    """Handler for boundary window creation and application.
    
    Centralizes logic for creating time and frequency boundary windows,
    applying noise floors, and managing boundary state for solvers.
    
    Attributes:
        use_boundaries (bool): Whether boundaries are enabled.
        time_window (jnp.ndarray | None): Time-domain window array.
        freq_window_shifted (jnp.ndarray | None): Frequency-domain window (shifted ordering).
    """
    
    def __init__(
        self,
        use_boundaries: bool = False,
        time_window: Optional[jnp.ndarray] = None,
        freq_window_shifted: Optional[jnp.ndarray] = None,
    ):
        """Initialize BoundaryHandler.
        
        Args:
            use_boundaries: Whether boundaries are enabled.
            time_window: Pre-computed time window (optional).
            freq_window_shifted: Pre-computed frequency window in shifted ordering (optional).
        """
        self.use_boundaries = use_boundaries
        self.time_window = time_window
        self.freq_window_shifted = freq_window_shifted
    
    def setup_boundaries(
        self,
        pulse,
        include_self_steepening: bool,
        boundary_mode: str,
        wavelength_limits_um: Optional[Tuple[float, float]],
        freq_window_fraction: Optional[Tuple[float, float]],
        time_boundary_order: Union[int, Tuple[int, int]],
        freq_boundary_order: Tuple[int, int],
        time_window_fraction: Union[float, Tuple[float, float]],
    ) -> None:
        """Setup boundary windows based on solver configuration.
        
        Args:
            pulse: Pulse object providing time and frequency grids.
            include_self_steepening: Whether self-steepening is enabled.
            boundary_mode: Boundary mode ("both", "low", "high", "none").
            wavelength_limits_um: Optional wavelength limits (min, max) in micrometers.
            freq_window_fraction: Optional fractional FWHM for frequency taper [low, high].
                If None, defaults to [0.6, 0.6] or computed from wavelength_limits_um.
            time_boundary_order: Super-Gaussian order for time window.
                Can be a single int (symmetric) or tuple [left, right] (asymmetric).
            freq_boundary_order: Super-Gaussian order for frequency window [low, high].
            time_window_fraction: Fraction of time window kept near unity.
                Can be a single float (symmetric) or tuple [left, right] (asymmetric).
        """
        self.use_boundaries = bool(
            (include_self_steepening and boundary_mode != "none")
            or (wavelength_limits_um is not None)
        )
        
        if not self.use_boundaries:
            return
        
        # Normalize single values to tuples for uniform handling
        if isinstance(time_boundary_order, (int, np.integer)):
            time_boundary_order = (int(time_boundary_order), int(time_boundary_order))
        else:
            time_boundary_order = (int(time_boundary_order[0]), int(time_boundary_order[1]))
        
        if isinstance(time_window_fraction, (int, float)):
            time_window_fraction = (float(time_window_fraction), float(time_window_fraction))
        else:
            time_window_fraction = (float(time_window_fraction[0]), float(time_window_fraction[1]))
        
        # Create time window
        time_window = create_super_gaussian_time_window(
            pulse.t,
            pulse.n_points,
            window_fraction=time_window_fraction,
            order=time_boundary_order,
        )
        
        # Determine frequency window parameters
        freq_window_fraction_low, freq_window_fraction_high = (
            self._determine_freq_window_fractions(
                pulse, wavelength_limits_um, freq_window_fraction
            )
        )
        
        # Compute spans for window creation (same as in _determine_freq_window_fractions)
        freqs_rel_thz = np.asarray(pulse.sim_params._freqs_relative_thz, dtype=np.float64)
        freqs_high = freqs_rel_thz[freqs_rel_thz > 0]
        span_high = float(np.max(freqs_high)) if freqs_high.size > 0 else 0.0
        freqs_low_rel = freqs_rel_thz[freqs_rel_thz < 0]
        freqs_low_dist = -freqs_low_rel
        span_low = float(np.max(freqs_low_dist)) if freqs_low_dist.size > 0 else 0.0
        # Use Nyquist frequency for span if it's larger
        if freqs_low_dist.size > 0:
            nyquist_freq = float(np.max(np.abs(freqs_low_rel)))
            span_high = max(span_high, nyquist_freq)
            span_low = max(span_low, nyquist_freq)
        
        freq_spacing_thz = (
            float(np.abs(freqs_rel_thz[1] - freqs_rel_thz[0]))
            if freqs_rel_thz.size > 1
            else 0.0
        )
        
        transition_span_high = None
        transition_span_low = None
        freq_rel_min_thz = None
        freq_rel_max_thz = None
        if wavelength_limits_um is not None:
            wavelength_min_um, wavelength_max_um = wavelength_limits_um
            if wavelength_min_um is not None:
                freq_rel_min_thz = (C_um_s / wavelength_min_um) * 1e-12 - pulse.center_freq_thz
            if wavelength_max_um is not None:
                freq_rel_max_thz = (C_um_s / wavelength_max_um) * 1e-12 - pulse.center_freq_thz
        
        if (
            freq_rel_min_thz is not None
            and freq_rel_min_thz > 0.0
            and span_high > TOLERANCE_ZERO
        ):
            available_high = max(span_high - freq_rel_min_thz, TOLERANCE_ZERO)
            transition_span_high = _estimate_transition_span(
                freq_rel_min_thz,
                available_high,
                max(freq_spacing_thz, TOLERANCE_ZERO),
            )
        
        if (
            freq_rel_max_thz is not None
            and freq_rel_max_thz < 0.0
            and span_low > TOLERANCE_ZERO
        ):
            target_low = abs(freq_rel_max_thz)
            available_low = max(span_low - target_low, TOLERANCE_ZERO)
            transition_span_low = _estimate_transition_span(
                target_low,
                available_low,
                max(freq_spacing_thz, TOLERANCE_ZERO),
            )
        
        # Extract frequency boundary orders
        freq_order_low, freq_order_high = freq_boundary_order
        
        # Create frequency window
        # Use the same frequency array as in _determine_freq_window_fractions
        # to ensure consistent span calculations - use sim_params directly to get FFT-ordered
        freq_window = create_super_gaussian_frequency_window(
            freqs_rel_thz,
            window_fraction=freq_window_fraction_low,  # Legacy parameter
            order=freq_order_low,
            damp_side="both" if wavelength_limits_um else boundary_mode,
            center_freq_thz=0.0,  # Explicitly set to 0.0 for relative frequencies
            window_fraction_low=freq_window_fraction_low,
            window_fraction_high=freq_window_fraction_high,
            order_low=freq_order_low,
            order_high=freq_order_high,
            span_low=span_low,
            span_high=span_high,
            transition_span_low=transition_span_low,
            transition_span_high=transition_span_high,
        )
        
        # Apply quantum noise floor to boundary window (only zero values get noise baseline)
        # Noise floor is a fixed small threshold, independent of signal power
        freq_window = apply_noise_to_boundary(freq_window)
        
        # Store windows
        self.time_window = jnp.asarray(time_window, dtype=jnp.complex128)
        freq_window_shifted = jnp.fft.fftshift(freq_window)
        self.freq_window_shifted = jnp.asarray(freq_window_shifted, dtype=jnp.complex128)
    
    def _determine_freq_window_fractions(
        self,
        pulse,
        wavelength_limits_um: Optional[Tuple[float, float]],
        freq_window_fraction: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Determine frequency window fractions from wavelength limits or defaults.
        
        Args:
            pulse: Pulse object with frequency grids.
            wavelength_limits_um: Optional wavelength limits (min, max).
            freq_window_fraction: Optional frequency window fractions [low, high].
                If None, defaults to [0.6, 0.6] or computed from wavelength_limits_um.
        
        Returns:
            Tuple of (freq_window_fraction_low, freq_window_fraction_high).
        """
        # Extract low and high fractions from tuple if provided
        if freq_window_fraction is not None:
            freq_window_fraction_low, freq_window_fraction_high = freq_window_fraction
        else:
            freq_window_fraction_low = None
            freq_window_fraction_high = None
        
        if wavelength_limits_um is None:
            # Use provided fractions or defaults
            if freq_window_fraction_low is None:
                freq_window_fraction_low = 0.6
            if freq_window_fraction_high is None:
                freq_window_fraction_high = 0.6
            return freq_window_fraction_low, freq_window_fraction_high
        
        # Compute fractions from wavelength limits
        wavelength_min_um, wavelength_max_um = wavelength_limits_um
        center_freq_thz = pulse.center_freq_thz
        # Use sim_params._freqs_relative_thz directly to ensure FFT-ordered (not shifted)
        freqs_rel_thz = np.asarray(pulse.sim_params._freqs_relative_thz, dtype=np.float64)
        
        # Compute frequency spans
        # For FFT-ordered frequencies, positive frequencies are at indices [0, n/2)
        # The maximum positive frequency is at index n/2 - 1 (or the last positive value)
        # But we need to account for the full span including Nyquist frequency
        freqs_high = freqs_rel_thz[freqs_rel_thz > 0]
        if freqs_high.size > 0:
            max_positive = float(np.max(freqs_high))
            freqs_negative = freqs_rel_thz[freqs_rel_thz < 0]
            if freqs_negative.size > 0:
                nyquist_freq = float(np.max(np.abs(freqs_negative)))
                span_high = max(max_positive, nyquist_freq)
            else:
                span_high = max_positive
        else:
            span_high = 0.0
        
        freqs_low_rel = freqs_rel_thz[freqs_rel_thz < 0]
        freqs_low_dist = -freqs_low_rel
        span_low = float(np.max(freqs_low_dist)) if freqs_low_dist.size > 0 else 0.0
        
        # Compute high-frequency fraction from minimum wavelength
        # For min wavelength (short wavelength = high frequency), we want to damp
        # frequencies above the frequency corresponding to min wavelength.
        # The window fraction represents what portion of the span to keep (passband).
        # The super-Gaussian window starts damping at cutoff = span * fraction,
        # where frequencies < cutoff are fully passed (window = 1.0).
        if wavelength_min_um is not None:
            freq_min_hz = C_um_s / wavelength_min_um
            freq_min_thz = freq_min_hz * 1e-12
            freq_rel_min_thz = freq_min_thz - center_freq_thz
            if span_high > 0:
                if freq_rel_min_thz > 0:
                    if freq_rel_min_thz <= span_high:
                        # Min wavelength frequency is within the span
                        # Set cutoff exactly at min wavelength frequency
                        fraction_val = freq_rel_min_thz / span_high
                        freq_window_fraction_high = float(fraction_val)
                    else:
                        # Min wavelength frequency is beyond the span
                        # We want to damp all high frequencies since they all correspond
                        # to wavelengths shorter than min wavelength (even though they're
                        # below the min wavelength frequency, the grid doesn't extend far enough)
                        # Set fraction to clamp cutoff at span maximum, but this will still
                        # keep all frequencies. Instead, we need to think about this differently:
                        # If min wavelength is 0.5 µm but grid max is 0.83 µm, we should
                        # still try to create a transition. Use a fraction that places
                        # the cutoff at the span edge, which will keep all frequencies.
                        # But actually, if the user specifies min wavelength, they want
                        # wavelengths shorter than that damped. Since grid doesn't reach
                        # that frequency, we can't create the cutoff there. For now, keep
                        # all frequencies (fraction = 1.0) as a reasonable default.
                        freq_window_fraction_high = 1.0
                else:
                    # Min wavelength frequency is below center, so keep all high frequencies
                    freq_window_fraction_high = 1.0
            else:
                freq_window_fraction_high = 1.0
        else:
            freq_window_fraction_high = freq_window_fraction_high or 0.6
        
        # Compute low-frequency fraction from maximum wavelength
        # For max wavelength (long wavelength = low frequency), we want to damp
        # frequencies below the frequency corresponding to max wavelength.
        # The window fraction represents what portion of the span to keep (passband).
        if wavelength_max_um is not None:
            freq_max_hz = C_um_s / wavelength_max_um
            freq_max_thz = freq_max_hz * 1e-12
            freq_rel_max_thz = freq_max_thz - center_freq_thz
            if span_low > 0:
                if freq_rel_max_thz < 0:
                    # Fraction is the ratio of the max wavelength frequency distance to the span
                    # This directly sets the cutoff: cutoff = span * fraction
                    fraction_val = abs(freq_rel_max_thz) / span_low
                    freq_window_fraction_low = max(
                        0.0, min(1.0, float(fraction_val))
                    )
                else:
                    # Max wavelength frequency is above center, so keep all low frequencies
                    freq_window_fraction_low = 1.0
            else:
                freq_window_fraction_low = 1.0
        else:
            freq_window_fraction_low = freq_window_fraction_low or 0.6
        
        return freq_window_fraction_low, freq_window_fraction_high
    
    def apply_time_window(self, array: jnp.ndarray) -> jnp.ndarray:
        """Apply time-domain boundary window to array.
        
        Args:
            array: Array to apply window to, shape (..., n_points).
        
        Returns:
            Windowed array, same shape as input.
        """
        if not self.use_boundaries or self.time_window is None:
            return array
        reshape = (1,) * (array.ndim - 1) + (self.time_window.shape[0],)
        return array * self.time_window.reshape(reshape)
    
    def apply_freq_window(self, array: jnp.ndarray) -> jnp.ndarray:
        """Apply frequency-domain boundary window to array.
        
        Args:
            array: Array to apply window to, shape (..., n_points).
        
        Returns:
            Windowed array, same shape as input.
        """
        if not self.use_boundaries or self.freq_window_shifted is None:
            return array
        reshape = (1,) * (array.ndim - 1) + (self.freq_window_shifted.shape[0],)
        shifted = jnp.fft.fftshift(array, axes=-1)
        tapered = shifted * self.freq_window_shifted.reshape(reshape)
        return jnp.fft.ifftshift(tapered, axes=-1)
