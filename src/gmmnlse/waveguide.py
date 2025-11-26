import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import factorial
import numpy as np
from .constants import C_m_ps
from typing import Optional

class Waveguide:
    """
    A class representing a multimode waveguide, containing its dispersion and
    nonlinear properties.
    """
    def __init__(
        self, 
        betas: jnp.ndarray, 
        s_tensor: jnp.ndarray, 
        n2: float, 
        center_freq_thz: float,
        s_tensor_freq: Optional[jnp.ndarray] = None,
        freqs_hz: Optional[jnp.ndarray] = None,
        num_modes: Optional[int] = None
    ):
        """
        Initializes the Waveguide object.

        Args:
            betas (jnp.ndarray): A 2D array of shape `(num_orders, num_modes)`
                containing the Taylor expansion coefficients of the propagation
                constant (β) for each mode. Units are ps^n/m.
            s_tensor (jnp.ndarray): A 4D array representing the nonlinear
                coupling coefficients at center frequency, shape (num_modes, num_modes, num_modes, num_modes).
                If s_tensor_freq is provided, this is used as fallback or for non-frequency-dependent calculations.
            n2 (float): The nonlinear refractive index of the waveguide material.
            center_freq_thz (float): The center frequency of the simulation in THz.
            s_tensor_freq (jnp.ndarray, optional): Frequency-dependent s_tensor (Q_plmn),
                shape (n_freqs, num_modes, num_modes, num_modes, num_modes). If provided,
                enables proper shock term calculation using derivative of ln[Q_plmn(ω)].
            freqs_hz (jnp.ndarray, optional): Frequency grid in Hz corresponding to s_tensor_freq.
                Required if s_tensor_freq is provided.
            num_modes (int, optional): Number of modes to use. If None, uses all available modes.
                If specified, slices betas and s_tensor to use only the first num_modes modes.
        """
        # Slice to requested number of modes if specified
        if num_modes is not None:
            if num_modes <= 0:
                raise ValueError(f"num_modes must be positive, got {num_modes}")
            if num_modes > betas.shape[1]:
                raise ValueError(
                    f"num_modes ({num_modes}) exceeds available modes ({betas.shape[1]})"
                )
            betas = betas[:, :num_modes]
            s_tensor = s_tensor[:num_modes, :num_modes, :num_modes, :num_modes]
            if s_tensor_freq is not None:
                s_tensor_freq = s_tensor_freq[:, :num_modes, :num_modes, :num_modes, :num_modes]
        
        self.betas = betas
        self.s_tensor = s_tensor
        self.n2 = n2
        self.center_freq_thz = center_freq_thz
        self.num_modes = betas.shape[1]
        
        # Initialize frequency-dependent s_tensor if provided
        if s_tensor_freq is not None:
            if freqs_hz is None:
                raise ValueError(
                    "freqs_hz must be provided when s_tensor_freq is provided. "
                    f"s_tensor_freq shape: {s_tensor_freq.shape}, "
                    f"freqs_hz should have shape ({s_tensor_freq.shape[0]},) to match first dimension."
                )
            self._init_frequency_dependent_tensor(s_tensor_freq, freqs_hz, center_freq_thz)
        else:
            self.s_tensor_freq = None
            self.freqs_hz = None
            self.dlnQ_domega = None
            self._interp_s_tensor_fn = None

    def get_dispersion_operator(
        self,
        omega_relative: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculates the dispersion operator for each mode in the frequency domain.
        
        Uses relative frequencies (ω - ω₀) because beta coefficients are Taylor-expanded
        around the center frequency: β(ω) = β₀ + β₁(ω-ω₀) + (β₂/2)(ω-ω₀)² + ...
        
        The dispersion operator for mode p is given by:
        
        D_p(ω) = i[β₀_p - Re(β₀₀)] + i[β₁_p - Re(β₁₀)](ω - ω₀) 
                 + i Σ_{k=2}^N (β_k_p / k!) (ω - ω₀)^k
        
        where:
        - β_k_p is the k-th order Taylor expansion coefficient for mode p
        - β₀₀ and β₁₀ are reference values from the first mode
        - ω₀ is the center angular frequency

        Args:
            freqs_thz (jnp.ndarray, optional): A 1D array of relative frequency
                values in THz. This should match the FFT frequency ordering. Provide this
                when you do not already have the angular-frequency grid available.
            omega_relative (jnp.ndarray, optional): Relative angular frequency array
                in rad/ps (ω - ω₀). If supplied, this takes priority over freqs_thz and
                avoids regenerating the same values.

        Returns:
            jnp.ndarray: A 2D array of shape `(num_modes, num_freq_points)`
                representing the dispersion operator for each mode.
        """

        D = jnp.zeros((self.num_modes, omega_relative.size), dtype=jnp.complex128)

        for mode_i in range(self.num_modes):
            D = D.at[mode_i, :].set(
                1j * (self.betas[0, mode_i] - self.betas[0, 0]) +
                1j * (self.betas[1, mode_i] - self.betas[1, 0]) * omega_relative
            )
        
        for mode_i in range(self.num_modes):
            for order in range(2, self.betas.shape[0]):
                D = D.at[mode_i, :].add(
                    1j * (self.betas[order, mode_i] / factorial(order)) * (omega_relative ** order)
                )
        return D

    def _init_frequency_dependent_tensor(
        self,
        s_tensor_freq: jnp.ndarray,
        freqs_hz: jnp.ndarray,
        center_freq_thz: float,
    ) -> None:
        """Initialize frequency-dependent s_tensor and setup interpolation.
        
        Args:
            s_tensor_freq: Frequency-dependent s_tensor array, shape
                (n_freqs, num_modes, num_modes, num_modes, num_modes).
            freqs_hz: Frequency grid in Hz corresponding to s_tensor_freq.
            center_freq_thz: Center frequency in THz.
        
        Raises:
            ValueError: If freqs_hz is None or shapes don't match.
        """
        if freqs_hz is None:
            raise ValueError("freqs_hz must be provided when s_tensor_freq is provided")
        
        freqs_hz_arr = np.asarray(freqs_hz)
        if freqs_hz_arr.ndim != 1:
            raise ValueError("freqs_hz must be a 1D array")
        
        s_tensor_freq_arr = np.asarray(s_tensor_freq)
        if len(freqs_hz_arr) != s_tensor_freq_arr.shape[0]:
            raise ValueError(
                f"freqs_hz length ({len(freqs_hz_arr)}) must match first dimension of "
                f"s_tensor_freq ({s_tensor_freq_arr.shape[0]})"
            )
        
        # Sort frequencies for interpolation (np.interp requires sorted x values)
        sort_indices = np.argsort(freqs_hz_arr)
        freqs_hz_sorted = freqs_hz_arr[sort_indices]
        s_tensor_freq_sorted = s_tensor_freq_arr[sort_indices]
        
        # Store both original and sorted arrays
        self.freqs_hz = jnp.array(freqs_hz_arr)
        self.s_tensor_freq = jnp.array(s_tensor_freq_arr)
        self._freqs_hz_sorted = freqs_hz_sorted
        self._s_tensor_freq_sorted = s_tensor_freq_sorted
        
        # Convert center frequency to Hz
        self.center_freq_hz = center_freq_thz * 1e12
        center_freq_hz_val = float(self.center_freq_hz)
        
        # Setup interpolation function
        interp_fn = self._setup_interpolation(freqs_hz_sorted, s_tensor_freq_sorted)
        self._interp_s_tensor_fn = interp_fn
        
        # Replace stored s_tensor with interpolated value at center frequency
        self.s_tensor = jnp.array(interp_fn(center_freq_hz_val))
        
        # Compute shock term derivative
        self.dlnQ_domega = self._compute_shock_derivative(
            freqs_hz_sorted, center_freq_hz_val, interp_fn
        )

    def _setup_interpolation(
        self,
        freqs_hz_sorted: np.ndarray,
        s_tensor_freq_sorted: np.ndarray,
    ):
        """Setup interpolation function for frequency-dependent s_tensor.
        
        Args:
            freqs_hz_sorted: Sorted frequency array in Hz.
            s_tensor_freq_sorted: Sorted s_tensor array matching frequencies.
        
        Returns:
            Interpolation function that takes frequency in Hz and returns s_tensor.
        """
        # Prepare flattened arrays for interpolation
        flat_shape = (s_tensor_freq_sorted.shape[0], -1)
        s_tensor_flat = s_tensor_freq_sorted.reshape(flat_shape)
        s_tensor_flat_real = np.real(s_tensor_flat)
        s_tensor_flat_imag = np.imag(s_tensor_flat)

        def _interp(freq_hz: float) -> np.ndarray:
            """Interpolate complex s-tensor at given absolute frequency (Hz)."""
            real_interp = np.empty(s_tensor_flat_real.shape[1], dtype=np.float64)
            imag_interp = np.empty(s_tensor_flat_imag.shape[1], dtype=np.float64)
            
            for col_idx in range(s_tensor_flat_real.shape[1]):
                real_column = s_tensor_flat_real[:, col_idx]
                imag_column = s_tensor_flat_imag[:, col_idx]
                real_interp[col_idx] = np.interp(
                    freq_hz,
                    freqs_hz_sorted,
                    real_column,
                    left=real_column[0],
                    right=real_column[-1],
                )
                imag_interp[col_idx] = np.interp(
                    freq_hz,
                    freqs_hz_sorted,
                    imag_column,
                    left=imag_column[0],
                    right=imag_column[-1],
                )
            
            tensor = real_interp + 1j * imag_interp
            return tensor.reshape(s_tensor_freq_sorted.shape[1:])
        
        return _interp

    def _compute_shock_derivative(
        self,
        freqs_hz_sorted: np.ndarray,
        center_freq_hz_val: float,
        interp_fn,
    ) -> float:
        """Compute derivative d/dω ln[Q(ω)] at center frequency for shock term.
        
        Uses finite-difference approximation with frequency spacing from the
        provided frequency grid. This derivative is needed for self-steepening
        calculations.
        
        Args:
            freqs_hz_sorted: Sorted frequency array in Hz.
            center_freq_hz_val: Center frequency in Hz.
            interp_fn: Interpolation function for s_tensor.
        
        Returns:
            Derivative value dlnQ_domega in units of ps/rad.
        """
        # Helper function for mean magnitude
        def mean_abs(freq_hz: float) -> float:
            return float(np.abs(np.mean(interp_fn(freq_hz))))

        # Determine frequency spacing for finite-difference derivative
        if len(freqs_hz_sorted) > 1:
            spacing = np.diff(freqs_hz_sorted)
            positive_spacing = np.abs(spacing[spacing != 0.0])
            min_spacing = float(np.min(positive_spacing)) if positive_spacing.size else 0.0
        else:
            min_spacing = 0.0

        if min_spacing <= 0.0:
            return 0.0

        # Compute derivative using finite differences
        freq_prev = max(freqs_hz_sorted[0], center_freq_hz_val - min_spacing)
        freq_next = min(freqs_hz_sorted[-1], center_freq_hz_val + min_spacing)

        omega_prev = 2 * np.pi * freq_prev
        omega_next = 2 * np.pi * freq_next
        omega_center = 2 * np.pi * center_freq_hz_val

        Q_prev = mean_abs(freq_prev)
        Q_center = mean_abs(center_freq_hz_val)
        Q_next = mean_abs(freq_next)

        # Small epsilon to avoid log(0)
        eps = 1e-20

        # Compute derivative based on position relative to frequency grid
        if freq_prev < center_freq_hz_val < freq_next:
            domega = omega_next - omega_prev
            dlnQ_domega = (np.log(Q_next + eps) - np.log(Q_prev + eps)) / domega
        elif center_freq_hz_val <= freqs_hz_sorted[0]:
            domega = omega_next - omega_center
            dlnQ_domega = (np.log(Q_next + eps) - np.log(Q_center + eps)) / domega
        else:
            domega = omega_center - omega_prev
            dlnQ_domega = (np.log(Q_center + eps) - np.log(Q_prev + eps)) / domega

        return float(dlnQ_domega)

    def get_nonlinear_operator(
        self,
        pulse_field: jnp.ndarray,
        include_self_steepening: bool = False,
        pulse=None,
        dt: float | None = None,
        omega_abs: jnp.ndarray | None = None,
        omega_relative: jnp.ndarray | None = None,
        omega_0: float | None = None,
    ) -> jnp.ndarray:
        """Calculate the nonlinear operator for the GMMNLSE.
        
        Computes the nonlinear term:
        
        η_p(t) = γ(ω) Σ_{l,m,n} S_{plmn} A_l(t) A_m(t) A_n*(t)
        
        where:
        - γ(ω) = i n₂ ω / c is the nonlinear coefficient (frequency-dependent)
        - S_{plmn} is the nonlinear coupling tensor
        - A_p(t) is the electric field envelope for mode p
        - ω is the absolute angular frequency
        
        With self-steepening included:
        
        η_p(t) = F^{-1}[F[γ(ω) Σ_{l,m,n} S_{plmn} A_l(t) A_m(t) A_n*(t)] * shock_factor(ω)]
        
        where F and F^{-1} are forward and inverse Fourier transforms.

        Args:
            pulse_field: Complex electric field in time domain,
                shape `(num_modes, num_points)`.
            include_self_steepening: Include self-steepening effect. Defaults to False.
            pulse: Pulse object supplying frequency grids. If provided, frequency
                grids are extracted from pulse. Mutually exclusive with explicit
                frequency grid parameters.
            dt: Time step (ps). Required if pulse is None and frequency grids
                are not explicitly provided.
            omega_abs: Absolute angular frequency array (rad/ps). If provided,
                overrides pulse/dt computation.
            omega_relative: Relative angular frequency array (rad/ps), 
                ω - ω₀. If provided, overrides pulse/dt computation.
            omega_0: Center angular frequency (rad/ps). Required if omega_abs
                or omega_relative are provided.

        Returns:
            Nonlinear operator array, shape `(num_modes, num_points)`.

        Raises:
            ValueError: If insufficient parameters provided to compute frequency grids.
        """
        # Compute frequency grids from provided parameters
        omega_abs, omega_relative, omega_0 = self._get_frequency_grids(
            pulse_field, pulse, dt, omega_abs, omega_relative, omega_0
        )
        
        # Compute nonlinear term using Einstein summation for faster computation
        nonlinear_term = jnp.einsum(
            'l...,m...,n...,plmn->p...',
            pulse_field,
            pulse_field,
            jnp.conjugate(pulse_field),
            self.s_tensor
        )

        # Apply frequency-dependent gamma in frequency domain
        # γ(ω) = i n₂ ω / c
        NL_freq = jnp.fft.fft(nonlinear_term, axis=-1)
        gamma = 1j * self.n2 * omega_abs / C_m_ps
        NL_freq = NL_freq * gamma[jnp.newaxis, :]
        nonlinear_term = jnp.fft.ifft(NL_freq, axis=-1)
        
        # Apply self-steepening if requested
        if include_self_steepening:
            shock_factor = self._compute_shock_factor(omega_relative, omega_0)
            NL_freq = jnp.fft.fft(nonlinear_term, axis=-1)
            NL_freq = NL_freq * shock_factor[jnp.newaxis, :]
            nonlinear_term = jnp.fft.ifft(NL_freq, axis=-1)

        return nonlinear_term

    def _get_frequency_grids(
        self,
        pulse_field: jnp.ndarray,
        pulse=None,
        dt: float | None = None,
        omega_abs: jnp.ndarray | None = None,
        omega_relative: jnp.ndarray | None = None,
        omega_0: float | None = None,
    ):
        """Extract or compute frequency grids from various input options.
        
        Args:
            pulse_field: Field array to determine n_points.
            pulse: Pulse object (optional).
            dt: Time step in ps (optional).
            omega_abs: Precomputed absolute frequency (optional).
            omega_relative: Precomputed relative frequency (optional).
            omega_0: Center frequency in rad/ps (optional).
        
        Returns:
            Tuple of (omega_abs, omega_relative, omega_0).
        
        Raises:
            ValueError: If insufficient parameters provided.
        """
        # If explicit frequency grids provided, use them
        if omega_abs is not None or omega_relative is not None:
            if omega_abs is None or omega_relative is None or omega_0 is None:
                raise ValueError(
                    "If providing explicit frequency grids, all of "
                    "(omega_abs, omega_relative, omega_0) must be provided"
                )
            return omega_abs, omega_relative, omega_0
        
        # Extract from pulse if available
        if pulse is not None:
            return pulse.omega_abs, pulse._omega_relative, pulse.omega_0
        
        # Compute from dt if provided
        if dt is not None:
            n_points = pulse_field.shape[-1]
            freqs_rel = jnp.fft.fftfreq(n_points, dt)  # THz
            omega_relative = 2 * jnp.pi * freqs_rel  # rad/ps
            omega_0 = 2 * jnp.pi * self.center_freq_thz  # rad/ps
            omega_abs = omega_relative + omega_0
            return omega_abs, omega_relative, omega_0
        
        raise ValueError(
            "Must provide either 'pulse', 'dt', or explicit frequency grids "
            "(omega_abs, omega_relative, omega_0)"
        )
    
    def _compute_shock_factor(self, omega_relative: jnp.ndarray, omega_0: float) -> jnp.ndarray:
        """
        Compute shock factor using proper formula from equation (8):
        τ_plmn^(1,2) = (1 / ω₀) + {∂/∂ω ln[Q_plmn^(1,2)(ω)]}_ω₀
        
        The shock factor applied in frequency domain is:
        shock_factor(ω) = ω * τ_plmn(ω) / ω₀
        
        For real-valued mode functions: Q_plmn^(1) = Q_plmn^(2), so τ_plmn^(1) = τ_plmn^(2)
        
        Args:
            omega_relative: Relative angular frequency (ω - ω₀) in rad/ps
            omega_0: Center angular frequency in rad/ps
            
        Returns:
            Shock factor array, shape (n_freq_points,)
        """
        # If frequency-dependent s_tensor is available, compute proper derivative
        if self.s_tensor_freq is not None and self.freqs_hz is not None and self.dlnQ_domega is not None:
            tau = (1.0 / omega_0) + self.dlnQ_domega
            
            # Shock factor: shock_factor(ω) = 1 + ω * τ
            shock_factor = 1.0 + omega_relative * tau
        else:
            # Fallback to approximation: 1 + ω_rel/ω₀
            shock_factor = 1.0 + (omega_relative / omega_0)
        
        return shock_factor
    
    def _interpolate_s_tensor_freq(self, freqs_hz: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate frequency-dependent s_tensor to given frequencies.
        
        Args:
            freqs_hz: Frequencies in Hz to interpolate to, shape (n_freqs,)
            
        Returns:
            Interpolated s_tensor, shape (n_freqs, num_modes, num_modes, num_modes, num_modes)
        """
        if self.s_tensor_freq is None or self.freqs_hz is None or self._interp_s_tensor_fn is None:
            raise ValueError("s_tensor_freq and freqs_hz must be set to use this method")
        
        freqs = np.asarray(freqs_hz)
        out_shape = (
            freqs.shape[0],
            self.num_modes,
            self.num_modes,
            self.num_modes,
            self.num_modes,
        )
        interpolated = np.zeros(out_shape, dtype=np.complex128)
        for idx, freq in enumerate(freqs):
            interpolated[idx] = self._interp_s_tensor_fn(float(freq))
        return jnp.array(interpolated)

