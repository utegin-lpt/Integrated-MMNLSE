from typing import Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import jax.scipy as jsp

from .constants import C_um_ps
from .sim import SimParams
from .utils.noise import add_shot_noise_to_field
from .utils.common import _return_float_if_possible


FloatLike = Union[float, jnp.ndarray]

jax.config.update("jax_enable_x64", True)

class Pulse:
    """Multimode optical pulse in time domain.

    Attributes:
        sim_params (SimParams): Simulation parameters (grid, frequencies).
        field (jnp.ndarray): Complex electric field, shape (n_modes, n_points).
        n_modes (int): Number of spatial modes.
    """
    def __init__(self, sim_params: SimParams, field_t: jnp.ndarray):
        """Initialize Pulse object.

        Args:
            sim_params (SimParams): Simulation parameters object.
            field_t (jnp.ndarray): Complex electric field, shape (n_modes, n_points).
        """
        self.sim_params = sim_params

        if field_t.shape[1] != self.sim_params.n_points:
            raise ValueError(
                f"Field size {field_t.shape[1]} does not match SimParams size {self.sim_params.n_points}. "
                f"Field shape: {field_t.shape}, expected second dimension: {self.sim_params.n_points}."
            )

        self.field = field_t.astype(jnp.complex128, copy=False)
        self.n_modes = self.field.shape[0]

    def __getattr__(self, name: str):
        """Delegate attribute access to sim_params for grid-related attributes.
        
        This allows Pulse objects to transparently access all SimParams attributes
        (t, dt, n_points, center_freq_thz, frequencies, wavelengths, etc.) without
        explicitly defining properties for each one.
        
        Args:
            name: Attribute name to look up.
        
        Returns:
            Attribute value from sim_params.
        
        Raises:
            AttributeError: If attribute is not found in sim_params.
        """
        if hasattr(self.sim_params, name):
            return getattr(self.sim_params, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Available attributes: {', '.join(dir(self))}"
        )

    @classmethod
    def gaussian(
        cls,
        peak_power_w: float,
        fwhm_ps: float,
        time_window_ps: float,
        n_modes: int,
        n_time_points: int,
        center_wavelength_um: float,
        modal_coefficients: jnp.ndarray | None = None,
        center_ps: float = 0.0,
        normalize_modal_coefficients: bool = True,
        supergaussian_order: int = 1,
        chirp: float = 0.0,
        include_shot_noise: bool = False,
        noise_seed: int | None = None,
        sim_params: SimParams | None = None,
    ) -> "Pulse":
        """Create Gaussian or super-Gaussian pulse.
        
        The pulse envelope is given by:
        
        A(t) = √P₀ exp(-[(t - t₀)/σ]^(2n)) exp(i·C·(t - t₀)²/2)
        
        where:
        - P₀ is the peak power
        - σ = FWHM / (2 [ln(2)]^(1/(2n))) is the width parameter
        - n is the super-Gaussian order (n=1 for Gaussian)
        - C is the chirp parameter
        - t₀ is the center time

        Args:
            peak_power_w (float): Peak power (W).
            fwhm_ps (float): Full width at half maximum (ps).
            time_window_ps (float): Time window duration (ps). Ignored if sim_params is provided.
            n_modes (int): Number of spatial modes.
            n_time_points (int): Number of time grid points. Ignored if sim_params is provided.
            center_wavelength_um (float): Center wavelength (µm).
            modal_coefficients (jnp.ndarray | None): Mode coefficients. If None, single-mode.
            center_ps (float): Center time (ps). Defaults to 0.0.
            normalize_modal_coefficients (bool): If True (default), rescale the supplied
                modal coefficients so that their L2 norm equals 1, ensuring the peak
                power matches `peak_power_w` when the modes are combined.
            supergaussian_order (int): Super-Gaussian order (1=Gaussian). Defaults to 1.
            chirp (float): Linear chirp parameter (THz²). Defaults to 0.0.
            include_shot_noise (bool): Include quantum shot noise. Defaults to False.
            noise_seed (int | None): Random seed for noise. Defaults to None.
            sim_params (SimParams | None): Optional pre-calculated SimParams. If provided, 
                time_window_ps and n_time_points are ignored.

        Returns:
            Pulse: Pulse object with specified properties.
        """
        # Determine simulation parameters
        if sim_params is not None:
            sim = sim_params
            times = sim.t
            n_points_actual = sim.n_points
            dt = float(sim.dt)
        else:
            dt = time_window_ps / n_time_points
            times = (jnp.arange(-n_time_points // 2, n_time_points // 2) * dt).astype(jnp.float64)
            center_freq_thz = C_um_ps / center_wavelength_um
            sim = SimParams(times, center_freq_thz)
            n_points_actual = n_time_points

        if modal_coefficients is None:
            modal_coefficients = jnp.zeros(n_modes, dtype=jnp.complex128).at[0].set(1.0)
        if len(modal_coefficients) != n_modes:
            raise ValueError("Length of `modal_coefficients` must match `n_modes`.")

        coeffs = jnp.asarray(modal_coefficients, dtype=jnp.complex128)
        if normalize_modal_coefficients:
            norm = jnp.linalg.norm(coeffs)
            if float(norm) == 0.0:
                raise ValueError("`modal_coefficients` cannot all be zero when normalization is requested.")
            coeffs = coeffs / norm

        sigma = fwhm_ps / (2 * (jnp.log(2))**(1 / (2 * supergaussian_order)))
        
        exponent = 2 * supergaussian_order
        envelope = jnp.sqrt(peak_power_w) * jnp.exp(
            -((times - center_ps) ** exponent) / (2 * sigma ** exponent)
        )

        if chirp != 0.0:
            chirp_phase = jnp.exp(1j * chirp * (times - center_ps) ** 2 / 2.0)
            envelope *= chirp_phase

        field = jnp.outer(coeffs, envelope)
        
        if include_shot_noise:
            field = add_shot_noise_to_field(
                field=field,
                dt_ps=dt,
                n_points=n_points_actual,
                center_freq_thz=sim.center_freq_thz,
                noise_seed=noise_seed,
            )
        
        return cls(sim, field)

    @classmethod
    def secant(
        cls,
        peak_power_w: float,
        fwhm_ps: float,
        time_window_ps: float,
        n_modes: int,
        n_time_points: int,
        center_wavelength_um: float,
        modal_coefficients: jnp.ndarray | None = None,
        normalize_modal_coefficients: bool = True,
        center_ps: float = 0.0,
        sim_params: SimParams | None = None,
    ) -> "Pulse":
        """Create hyperbolic secant (sech) pulse.
        
        The pulse envelope is given by:
        
        A(t) = √P₀ sech((t - t₀)/T₀)
        
        where:
        - P₀ is the peak power
        - T₀ = FWHM / (2 arccosh(√2)) is the characteristic width
        - t₀ is the center time

        Args:
            peak_power_w (float): Peak power (W).
            fwhm_ps (float): Full width at half maximum (ps).
            time_window_ps (float): Time window duration (ps). Ignored if sim_params is provided.
            n_modes (int): Number of spatial modes.
            n_time_points (int): Number of time grid points. Ignored if sim_params is provided.
            center_wavelength_um (float): Center wavelength (µm).
            modal_coefficients (jnp.ndarray | None): Mode coefficients. If None, single-mode.
            center_ps (float): Center time (ps). Defaults to 0.0.
            sim_params (SimParams | None): Optional pre-calculated SimParams. If provided, 
                time_window_ps and n_time_points are ignored.

        Returns:
            Pulse: Pulse object with sech profile.
        """
        # Determine simulation parameters
        if sim_params is not None:
            sim = sim_params
            times = sim.t
        else:
            dt = time_window_ps / n_time_points
            times = (jnp.arange(-n_time_points // 2, n_time_points // 2) * dt).astype(jnp.float64)
            center_freq_thz = C_um_ps / center_wavelength_um
            sim = SimParams(times, center_freq_thz)

        if modal_coefficients is None:
            modal_coefficients = jnp.zeros(n_modes, dtype=jnp.complex128).at[0].set(1.0)
        if len(modal_coefficients) != n_modes:
            raise ValueError("Length of `modal_coefficients` must match `n_modes`.")
        
        coeffs = jnp.asarray(modal_coefficients, dtype=jnp.complex128)
        if normalize_modal_coefficients:
            norm = jnp.linalg.norm(coeffs)
            if float(norm) == 0.0:
                raise ValueError("`modal_coefficients` cannot all be zero when normalization is requested.")
            coeffs = coeffs / norm
        
        T0 = fwhm_ps / (2 * jnp.arccosh(jnp.sqrt(2)))
        envelope = jnp.sqrt(peak_power_w) * jnp.cosh((times - center_ps) / T0)**-1

        field = jnp.outer(coeffs, envelope)
        return cls(sim, field)

    def __getitem__(self, idx):
        """Extract pulse for specific mode(s).

        Example:
            single_mode_pulse = my_pulse[0]
        """
        sub_field = jnp.take(self.field, idx, axis=0)
        if sub_field.ndim == 1:
            sub_field = sub_field[jnp.newaxis, :]
        return self.__class__(self.sim_params, sub_field)

    def pad_with_zeros(self, pad_points: int) -> "Pulse":
        """Return a new pulse with symmetric zero padding in time.

        Args:
            pad_points (int): Number of additional samples to append to each end.
                Must be non-negative. A value of zero returns the pulse unchanged.

        Returns:
            Pulse: New pulse instance with extended time window and zero-padded field.
        """
        if pad_points is None or pad_points <= 0:
            return Pulse(self.sim_params, self.field)

        pad_points = int(pad_points)
        new_count = self.n_points + 2 * pad_points
        dt = float(self.dt)
        times = (
            jnp.arange(-new_count // 2, new_count // 2, dtype=jnp.float64) * dt
        )

        padded_field = jnp.zeros(
            (self.n_modes, new_count), dtype=self.field.dtype
        )
        start = pad_points
        end = start + self.n_points
        padded_field = padded_field.at[:, start:end].set(self.field)

        # Create new SimParams for the padded grid
        new_sim = SimParams(times, self.center_freq_thz)
        return Pulse(new_sim, padded_field)

    def encode_phase_information(self, phase_info: jnp.ndarray, fwhm_mask: jnp.ndarray, num_points_in_mask: int) -> "Pulse":
        """Encode phase array onto pulse spectrum within FWHM.

        Args:
            phase_info (jnp.ndarray): Phase values (rad) to encode.
            fwhm_mask (jnp.ndarray): Boolean mask for FWHM region (shifted frequency ordering).
            num_points_in_mask (int): Number of True values in fwhm_mask.

        Returns:
            Pulse: New pulse with encoded phase.
        """
        F = jnp.fft.fftshift(jnp.fft.fft(self.field, axis=-1), axes=-1)
        
        if len(phase_info) == 0:
            return Pulse(self.sim_params, self.field)
            
        repeats_per_element = num_points_in_mask // len(phase_info)
        
        total_length = repeats_per_element * len(phase_info)
        repeated_phase = jnp.repeat(phase_info, repeats_per_element, total_repeat_length=total_length)
        
        num_padding = num_points_in_mask - total_length
        padding = jnp.zeros(num_padding)
        
        full_phase_modulation_flat = jnp.concatenate([repeated_phase, padding])

        phase_modulation = jnp.zeros(self.n_points)
        phase_modulation = phase_modulation.at[fwhm_mask].set(full_phase_modulation_flat)

        phase_modulator = jnp.exp(1j * phase_modulation)
        F_modulated = F * phase_modulator[jnp.newaxis, :]

        field_t_modulated = jnp.fft.ifft(jnp.fft.ifftshift(F_modulated, axes=-1), axis=-1)
        return Pulse(self.sim_params, field_t_modulated)

    def apply_taylor_spectral_phase(self, phase_coeffs: jnp.ndarray) -> "Pulse":
        """Apply spectral phase from Taylor series coefficients.

        Applies a spectral phase given by the Taylor expansion:
        
        φ(ω) = Σ_{k=2}^{N+1} (c_{k-2} / k!) (ω - ω₀)^k
        
        where:
        - c_k are the phase coefficients (c₀ = GDD in ps², c₁ = TOD in ps³, etc.)
        - ω₀ is the center angular frequency
        - The phase is applied as: A(ω) → A(ω) exp(i φ(ω))

        Args:
            phase_coeffs (jnp.ndarray): Taylor coefficients starting from GDD (ps²).
                coeffs[0]: GDD (ps²), coeffs[1]: TOD (ps³), etc.

        Returns:
            Pulse: New pulse with applied spectral phase.
        """
        F = jnp.fft.fft(self.field, axis=-1)
        omega_relative = self._omega_relative

        total_phase = jnp.zeros_like(omega_relative)
        for k, coeff in enumerate(phase_coeffs):
            order = k + 2
            total_phase += (coeff / jsp.special.factorial(order)) * omega_relative**order
        
        phase_modulator = jnp.exp(1j * total_phase)
        F_modulated = F * phase_modulator[jnp.newaxis, :]
        field_t_modulated = jnp.fft.ifft(F_modulated, axis=-1)
        return Pulse(self.sim_params, field_t_modulated)

    def get_fwhm_info(self, sum_type: str = 'coherent') -> dict:
        """Get FWHM information for pulse spectrum.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            dict: Dictionary with 'start_index', 'end_index', 'count', 'mask'.
        """
        spectrum = self.get_total_spectrum(sum_type=sum_type)
        half_max = jnp.max(spectrum) / 2.0
        
        is_above = spectrum >= half_max
        start_index = jnp.argmax(is_above)
        end_index = self.n_points - 1 - jnp.argmax(is_above[::-1])
        count = end_index - start_index + 1
        
        mask = jnp.arange(self.n_points) >= start_index
        mask &= jnp.arange(self.n_points) <= end_index
        
        return {
            'start_index': start_index, 'end_index': end_index,
            'count': count, 'mask': mask
        }

    def get_temporal_fwhm(self, sum_type: str = 'coherent') -> float:
        """Get temporal FWHM.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            float: Temporal FWHM (ps).
        """
        intensity = self.get_total_intensity(sum_type=sum_type)
        half_max = jnp.max(intensity) / 2.0
        
        is_above = intensity >= half_max
        start_index = jnp.argmax(is_above)
        end_index = self.n_points - 1 - jnp.argmax(is_above[::-1])
        
        fwhm_time_low = self.t[start_index]
        fwhm_time_high = self.t[end_index]
        
        return float(fwhm_time_high - fwhm_time_low)

    def get_fwhm_bandwidth_um(self, sum_type: str = 'coherent') -> float:
        """Get spectral FWHM bandwidth.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            float: Spectral FWHM bandwidth (µm).
        """
        freqs_relative_shifted_thz = self._freqs_relative_shifted_thz
        fwhm_info = self.get_fwhm_info(sum_type=sum_type)
        
        fwhm_freq_low_thz = freqs_relative_shifted_thz[fwhm_info['start_index']]
        fwhm_freq_high_thz = freqs_relative_shifted_thz[fwhm_info['end_index']]
        bw_thz = fwhm_freq_high_thz - fwhm_freq_low_thz
        
        center_lambda_um = C_um_ps / self.center_freq_thz
        bw_um = (center_lambda_um ** 2 / C_um_ps) * bw_thz
        return float(bw_um)

    def get_intensity(self) -> jnp.ndarray:
        """Get intensity of each mode.

        Returns:
            jnp.ndarray: Intensity array, shape (n_modes, n_points).
        """
        return jnp.abs(self.field) ** 2

    def get_total_intensity(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get total intensity across all modes.

        For coherent sum:
        
        I(t) = |Σ_p A_p(t)|²
        
        For incoherent sum:
        
        I(t) = Σ_p |A_p(t)|²
        
        where A_p(t) is the electric field envelope for mode p.

        Args:
            sum_type (str): 'coherent' (I = |E₁ + E₂ + ...|²) or 'incoherent'
                (I = |E₁|² + |E₂|² + ...). Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Total intensity array, shape (n_points,).
        """
        if sum_type == 'coherent':
            total_field = jnp.sum(self.field, axis=0)
            return jnp.abs(total_field) ** 2
        elif sum_type == 'incoherent':
            return jnp.sum(self.get_intensity(), axis=0)
        else:
            raise ValueError("`sum_type` must be 'coherent' or 'incoherent'.")

    def get_total_log_intensity(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get log-normalized total intensity in dB.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Log-normalized intensity in dB.
        """
        I = self.get_total_intensity(sum_type=sum_type)
        return 10 * jnp.log10(I / jnp.max(I) + 1e-20)
    
    def get_total_normalized_intensity(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get normalized total intensity.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Normalized total intensity, shape (n_points,).
        """
        I = self.get_total_intensity(sum_type=sum_type)
        return I / jnp.max(I)

    def get_spectrum(self) -> jnp.ndarray:
        """Get power spectral density of each mode.

        Returns:
            jnp.ndarray: Spectrum array, shape (n_modes, n_points).
        """
        F = jnp.fft.fftshift(jnp.fft.fft(self.field, axis=-1), axes=-1)
        return jnp.abs(F) ** 2

    def get_log_spectrum(self) -> jnp.ndarray:
        """Get log-scaled spectrum of each mode in dB.

        Returns:
            jnp.ndarray: Log-scaled spectrum, shape (n_modes, n_points).
        """
        S = self.get_spectrum()
        return 10 * jnp.log10(S + 1e-20)

    def get_total_spectrum(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get total spectrum across all modes.

        For coherent sum:
        
        S(ω) = |F[Σ_p A_p(t)]|² = |Σ_p A_p(ω)|²
        
        For incoherent sum:
        
        S(ω) = Σ_p |A_p(ω)|²
        
        where A_p(ω) = F[A_p(t)] is the Fourier transform of the field envelope.

        Args:
            sum_type (str): 'coherent' (sum fields before FFT) or 'incoherent'
                (sum spectra). Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Total spectrum, shape (n_points,).
        """
        if sum_type == 'coherent':
            total_field = jnp.sum(self.field, axis=0)
            F_total = jnp.fft.fftshift(jnp.fft.fft(total_field, axis=-1))
            return jnp.abs(F_total) ** 2
        elif sum_type == 'incoherent':
            return jnp.sum(self.get_spectrum(), axis=0)
        else:
            raise ValueError("`sum_type` must be 'coherent' or 'incoherent'.")

    def get_log_normalized_spectrum(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get log-normalized spectrum of each mode in dB.

        Each mode is normalized to the peak of the total spectrum.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Log-normalized spectrum, shape (n_modes, n_points).
        """
        S = self.get_spectrum()
        max_S_total = jnp.max(self.get_total_spectrum(sum_type=sum_type))
        return 10*jnp.log10(S / max_S_total + 1e-20)

    def get_total_log_normalized_spectrum(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get log-normalized total spectrum in dB.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Log-normalized total spectrum, shape (n_points,).
        """
        S = self.get_total_spectrum(sum_type=sum_type)
        return 10*jnp.log10(S / jnp.max(S) + 1e-20)

    def get_total_phase(self, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get total phase across all modes.

        Args:
            sum_type (str): 'coherent' (φ = arg(E₁ + E₂ + ...)) or 'incoherent'
                (φ = Σᵢ(Iᵢ*φᵢ)/ΣᵢIᵢ). Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Total phase array (rad), shape (n_points,).
        """
        if sum_type == 'coherent':
            total_field = jnp.sum(self.field, axis=0)
            return jnp.angle(total_field)
        elif sum_type == 'incoherent':
            phases = jnp.angle(self.field)
            intensities = self.get_intensity()
            weighted_phase_sum = jnp.sum(intensities * phases, axis=0)
            total_intensity = jnp.sum(intensities, axis=0)
            return jnp.where(
                total_intensity > 1e-20,
                weighted_phase_sum / total_intensity,
                0.0
            )
        else:
            raise ValueError("`sum_type` must be 'coherent' or 'incoherent'.")

    def get_total_power(self, sum_type: str = 'incoherent') -> FloatLike:
        """Get total power.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'incoherent'.

        Returns:
            float: Total power (W).
        """
        intensity = self.get_total_intensity(sum_type=sum_type)
        total = jnp.sum(intensity)
        return _return_float_if_possible(total)

    def get_modal_power_distribution(self, sum_type: str = 'incoherent') -> jnp.ndarray:
        """Get power distribution across modes as percentages.

        Args:
            sum_type (str): How to calculate total power for normalization.
                'coherent' or 'incoherent'. Defaults to 'incoherent'.

        Returns:
            jnp.ndarray: Power percentages for each mode, shape (n_modes,). Sums to 100.0.
        """
        mode_powers = jnp.sum(self.get_intensity(), axis=-1)
        if sum_type == 'coherent':
            total_power = self.get_total_power(sum_type='coherent')
        else:
            total_power = jnp.sum(mode_powers)
        return jnp.where(
            total_power > 1e-20,
            (mode_powers / total_power) * 100.0,
            jnp.zeros_like(mode_powers)
        )

    def get_peak_power(self, sum_type: str = 'coherent') -> FloatLike:
        """Get peak power.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.

        Returns:
            float: Peak power (W).
        """
        intensity = self.get_total_intensity(sum_type=sum_type)
        peak = jnp.max(intensity)
        return _return_float_if_possible(peak)

    def get_energy(self, sum_type: str = 'incoherent') -> FloatLike:
        """Get total energy.

        Args:
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'incoherent'.

        Returns:
            float: Total energy (pJ).
        """
        intensity = self.get_total_intensity(sum_type=sum_type)
        energy = jnp.sum(intensity) * self.dt
        return _return_float_if_possible(energy)

    def get_spectral_phase(self, mode_idx: int | None = None, sum_type: str = 'coherent') -> jnp.ndarray:
        """Get phase in frequency domain.

        Args:
            mode_idx (int | None): Mode index. If None, returns total phase using sum_type.
            sum_type (str): 'coherent' or 'incoherent'. Used when mode_idx=None. Defaults to 'coherent'.

        Returns:
            jnp.ndarray: Phase array (rad), shape (n_points,). Frequencies in shifted ordering.
        """
        if mode_idx is not None:
            field_freq = jnp.fft.fftshift(jnp.fft.fft(self.field[mode_idx, :]))
            return jnp.angle(field_freq)
        else:
            if sum_type == 'coherent':
                total_field = jnp.sum(self.field, axis=0)
                field_freq = jnp.fft.fftshift(jnp.fft.fft(total_field))
                return jnp.angle(field_freq)
            else:
                field_freq = jnp.fft.fft(self.field, axis=-1)
                field_freq_shifted = jnp.fft.fftshift(field_freq, axes=-1)
                phases = jnp.angle(field_freq_shifted)
                intensities = self.get_spectrum()
                weighted_phase_sum = jnp.sum(intensities * phases, axis=0)
                total_intensity = jnp.sum(intensities, axis=0)
                return jnp.where(
                    total_intensity > 1e-20,
                    weighted_phase_sum / total_intensity,
                    jnp.zeros(self.n_points)
                )

    def plot_pulse(self, n_modes_to_plot: int = 10, sum_type: str = 'coherent', title: str = "Pulse in Time Domain", xlim: tuple | None = None):
        """Plot pulse intensity in time domain.

        Args:
            n_modes_to_plot (int): Maximum number of modes to plot. Defaults to 10.
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.
            title (str): Plot title.
            xlim (tuple | None): X-axis limits (min, max).
        """
        plt.figure(figsize=(10, 6))
        for i in range(min(self.n_modes, n_modes_to_plot)):
            plt.plot(self.t, self.get_intensity()[i, :], label=f"Mode {i}")
        if self.n_modes > 1:
                plt.plot(self.t, self.get_total_intensity(sum_type=sum_type), 'k--', label=f"Total ({sum_type})")
        plt.title(title)
        plt.xlabel("Time (ps)")
        plt.ylabel("Intensity (W)")
        if xlim:
            plt.xlim(xlim)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_spectrum(self, n_modes_to_plot: int = 10, sum_type: str = 'coherent', title: str = "Pulse Spectrum", xlim: tuple | None = (0, 4)):
        """Plot pulse power spectrum versus wavelength.

        Args:
            n_modes_to_plot (int): Maximum number of modes to plot. Defaults to 10.
            sum_type (str): 'coherent' or 'incoherent'. Defaults to 'coherent'.
            title (str): Plot title.
            xlim (tuple | None): X-axis limits (min, max). Defaults to (0, 4).
        """
        plt.figure(figsize=(10, 6))
        positive_freq_mask = self.freqs_abs_thz > 0
        wavelengths_um_sorted = self.wavelengths_um[positive_freq_mask]
        
        log_norm_spec = self.get_log_normalized_spectrum(sum_type=sum_type)[:, positive_freq_mask]
        sum_log_norm_spec = self.get_total_log_normalized_spectrum(sum_type=sum_type)[positive_freq_mask]

        for i in range(min(self.n_modes, n_modes_to_plot)):
            plt.plot(wavelengths_um_sorted, log_norm_spec[i, :], label=f"Mode {i}")
        if self.n_modes > 1:
                plt.plot(wavelengths_um_sorted, sum_log_norm_spec, 'k--', label=f"Total ({sum_type})")
        
        plt.title(title)
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Power (dB)")
        
        if xlim:
            plt.xlim(xlim)

        plt.ylim(-100, 5)
        plt.legend()
        plt.grid(True)
        plt.show()