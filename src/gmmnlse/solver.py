import jax.numpy as jnp
import numpy as np
import diffrax
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple
from .pulse import Pulse
from .solver_result import SolverResult
from .utils.noise import quantum_noise, apply_noise_to_boundary
from .utils.boundary import BoundaryHandler
from .constants import C_um_s, TOLERANCE_DIVISION
try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None


def _normalize_wavelength_window(
    wavelength_window_um: Sequence[float] | np.ndarray | jnp.ndarray | None,
) -> Tuple[float, float] | None:
    """Validate and convert a wavelength window into (min, max) tuple."""
    if wavelength_window_um is None:
        return None

    if isinstance(wavelength_window_um, (np.ndarray, jnp.ndarray)):
        values = np.asarray(wavelength_window_um).flatten().tolist()
    elif isinstance(wavelength_window_um, (list, tuple)):
        values = list(wavelength_window_um)
    else:
        raise TypeError(
            "wavelength_window_um must be provided as a list, tuple, or array [wl_min, wl_max]."
        )

    if len(values) != 2:
        raise ValueError(
            f"wavelength_window_um must contain exactly two values [wl_min, wl_max], got {len(values)}."
        )

    wl_min, wl_max = float(values[0]), float(values[1])
    if wl_min <= 0.0 or wl_max <= 0.0:
        raise ValueError("wavelength_window_um values must be positive.")
    if wl_min >= wl_max:
        raise ValueError("wavelength_window_um must satisfy wl_min < wl_max.")

    return wl_min, wl_max


class SolverBase:
    """Base class for GMMNLSE solvers with common boundary window methods.
    
    Provides shared functionality for applying time and frequency boundary windows
    to prevent reflections and aliasing in pulse propagation simulations.
    """
    
    def _apply_time_window(self, array: jnp.ndarray) -> jnp.ndarray:
        """Apply time-domain boundary window to array.
        
        Args:
            array: Array to apply window to, shape (..., n_points).
        
        Returns:
            Windowed array, same shape as input.
        """
        boundary_handler = getattr(self, '_boundary_handler', None)
        if boundary_handler is not None:
            return boundary_handler.apply_time_window(array)
        # Fallback for backward compatibility
        use_boundaries = getattr(self, '_use_boundaries', False)
        if not use_boundaries:
            return array
        time_window = getattr(self, '_time_window', None)
        if time_window is None:
            return array
        reshape = (1,) * (array.ndim - 1) + (time_window.shape[0],)
        return array * time_window.reshape(reshape)

    def _apply_freq_window(self, array: jnp.ndarray) -> jnp.ndarray:
        """Apply frequency-domain boundary window to array.
        
        Args:
            array: Array to apply window to, shape (..., n_points).
        
        Returns:
            Windowed array, same shape as input.
        """
        boundary_handler = getattr(self, '_boundary_handler', None)
        if boundary_handler is not None:
            return boundary_handler.apply_freq_window(array)
        # Fallback for backward compatibility
        use_boundaries = getattr(self, '_use_boundaries', False)
        if not use_boundaries:
            return array
        freq_window_shifted = getattr(self, '_freq_window_shifted', None)
        if freq_window_shifted is None:
            return array
        reshape = (1,) * (array.ndim - 1) + (freq_window_shifted.shape[0],)
        shifted = jnp.fft.fftshift(array, axes=-1)
        tapered = shifted * freq_window_shifted.reshape(reshape)
        return jnp.fft.ifftshift(tapered, axes=-1)


class DiffraxSolver(SolverBase):
    """GMMNLSE solver using interaction picture with Diffrax ODE solvers.

    Solves the Generalized Multimode Nonlinear Schrödinger Equation (GMMNLSE):
    
    ∂A_p(ω, z)/∂z = [D_p(ω) + N_p(ω, z)] A_p(ω, z)
    
    where:
    - D_p(ω) is the linear dispersion operator
    - N_p(ω, z) is the nonlinear operator (Fourier transform of η_p(t, z))
    
    Uses interaction picture: Ã_p(ω, z) = A_p(ω, z) exp(-D_p(ω) z), integrates nonlinear
    part, then transforms back. This allows efficient numerical integration.

    Supported solvers:
    - Tsit5: 5th order Runge-Kutta (default)
    - Dopri5: Dormand-Prince 5th order
    - Dopri8: Dormand-Prince 8th order
    - Kvaerno3: 3rd order SDIRK implicit
    - Kvaerno5: 5th order SDIRK implicit
    """
    def __init__(
        self,
        waveguide,
        pulse,
        solver_method='Tsit5',
        include_self_steepening=False,
        boundary_mode: str = "both",
        time_window_fraction: float | tuple[float, float] | list[float] = 0.5,
        wavelength_window_um: Sequence[float] | np.ndarray | jnp.ndarray | None = None,
        freq_window_fraction: float | tuple[float, float] | list[float] | None = None,
        time_boundary_order: int | tuple[int, int] | list[int] | None = None,
        freq_boundary_order: int | tuple[int, int] | list[int] | None = None,
        boundary_order: int = 20,
    ):
        """Initialize GMMNLSE solver.

        Args:
            waveguide (Waveguide): Waveguide object.
            pulse (Pulse): Initial pulse.
            solver_method (str): Diffrax solver ('Tsit5', 'Dopri5', 'Dopri8',
                'Kvaerno3', 'Kvaerno5'). Defaults to 'Tsit5'.
            include_self_steepening (bool): Include self-steepening. Defaults to False.
            boundary_mode (str): Super-Gaussian boundary mode when self-steepening is
                active. Options: "both", "low", "high", or "none". Defaults to "both".
            time_window_fraction (float | tuple[float, float] | list[float]): Fraction of time 
                window where the time-domain boundary window is near 1.0. Can be a single 
                float (symmetric) or tuple/list [left, right] (asymmetric). Defaults to 0.5.
            wavelength_window_um (sequence | array | None): Optional [wl_min, wl_max] (µm)
                specifying the passband to preserve in the spectrum. If provided, overrides
                freq_window_fraction.
            freq_window_fraction (float | tuple[float, float] | list[float] | None): Fractional 
                FWHM for the frequency-domain boundary window. Can be:
                - A single float: same fraction for both low and high frequency sides
                - A tuple/list [low, high]: different fractions for low and high sides
                - None: defaults to [0.6, 0.6] or computed from wavelength_window_um if provided
            time_boundary_order (int | tuple[int, int] | list[int] | None): Super-Gaussian order 
                for the time-domain boundary window. Can be a single int (symmetric) or 
                tuple/list [left, right] (asymmetric). If None, uses boundary_order.
            freq_boundary_order (int | tuple[int, int] | list[int] | None): Super-Gaussian order 
                for the frequency-domain boundary window. Can be:
                - A single int: same order for both low and high frequency sides
                - A tuple/list [low, high]: different orders for low and high sides
                - None: uses boundary_order for both sides
            boundary_order (int): Default super-Gaussian order for both boundaries when 
                time_boundary_order or freq_boundary_order are not specified. Defaults to 20.
        """
        self.waveguide = waveguide
        self.pulse = pulse
        self.solver_method = solver_method
        self.include_self_steepening = include_self_steepening
        boundary_mode_norm = boundary_mode.lower()
        if boundary_mode_norm not in {"both", "low", "high", "none"}:
            raise ValueError(
                f"boundary_mode must be 'both', 'low', 'high', or 'none', got '{boundary_mode}'"
            )
        self.boundary_mode = boundary_mode_norm
        self.L_op = self.waveguide.get_dispersion_operator(self.pulse._freqs_relative_thz)
        self._wavelength_limits_um = _normalize_wavelength_window(wavelength_window_um)
        self.boundary_order = boundary_order
        self._boundary_handler: Optional[BoundaryHandler] = None
        self._use_boundaries = False
        default_window = jnp.ones(self.pulse.n_points, dtype=jnp.complex128)
        self._time_window = default_window
        self._freq_window_shifted = default_window
        
        # Normalize time_boundary_order: single value -> tuple, None -> boundary_order
        if time_boundary_order is None:
            time_boundary_order = boundary_order
        if isinstance(time_boundary_order, (int, np.integer)):
            time_boundary_order = (int(time_boundary_order), int(time_boundary_order))
        elif isinstance(time_boundary_order, (tuple, list)) and len(time_boundary_order) == 2:
            time_boundary_order = (int(time_boundary_order[0]), int(time_boundary_order[1]))
        else:
            raise ValueError(
                "time_boundary_order must be an int, tuple/list [left, right], or None, "
                f"got {type(time_boundary_order).__name__}: {time_boundary_order}"
            )
        self.time_boundary_order = time_boundary_order
        
        # Normalize freq_boundary_order: single value -> tuple, None -> boundary_order
        if freq_boundary_order is None:
            freq_order_low = boundary_order
            freq_order_high = boundary_order
        elif isinstance(freq_boundary_order, (int, np.integer)):
            freq_order_low = int(freq_boundary_order)
            freq_order_high = int(freq_boundary_order)
        elif isinstance(freq_boundary_order, (tuple, list)) and len(freq_boundary_order) == 2:
            freq_order_low = int(freq_boundary_order[0])
            freq_order_high = int(freq_boundary_order[1])
        else:
            raise ValueError(
                "freq_boundary_order must be an int, tuple/list [low, high], or None, "
                f"got {type(freq_boundary_order).__name__}: {freq_boundary_order}"
            )
        # Store freq_boundary_order as tuple for consistency
        self.freq_boundary_order = (freq_order_low, freq_order_high)
        
        # Normalize time_window_fraction: single value -> tuple
        if isinstance(time_window_fraction, (int, float)):
            time_window_fraction = (float(time_window_fraction), float(time_window_fraction))
        elif isinstance(time_window_fraction, (tuple, list)) and len(time_window_fraction) == 2:
            time_window_fraction = (float(time_window_fraction[0]), float(time_window_fraction[1]))
        else:
            raise ValueError(
                "time_window_fraction must be a float or tuple/list [left, right], "
                f"got {type(time_window_fraction).__name__}: {time_window_fraction}"
            )
        self.time_window_fraction = time_window_fraction
        
        # Normalize freq_window_fraction: single value -> tuple, None -> None
        if freq_window_fraction is None:
            freq_window_fraction_normalized = None
        elif isinstance(freq_window_fraction, (int, float)):
            freq_window_fraction_normalized = (float(freq_window_fraction), float(freq_window_fraction))
        elif isinstance(freq_window_fraction, (tuple, list)) and len(freq_window_fraction) == 2:
            freq_window_fraction_normalized = (float(freq_window_fraction[0]), float(freq_window_fraction[1]))
        else:
            raise ValueError(
                "freq_window_fraction must be a float, tuple/list [low, high], or None, "
                f"got {type(freq_window_fraction).__name__}: {freq_window_fraction}"
            )
        self.freq_window_fraction = freq_window_fraction_normalized

        self._boundary_handler = BoundaryHandler()
        self._boundary_handler.setup_boundaries(
            pulse=self.pulse,
            include_self_steepening=self.include_self_steepening,
            boundary_mode=self.boundary_mode,
            wavelength_limits_um=self._wavelength_limits_um,
            freq_window_fraction=freq_window_fraction_normalized,
            time_boundary_order=self.time_boundary_order,
            freq_boundary_order=self.freq_boundary_order,
            time_window_fraction=self.time_window_fraction,
        )
        self._use_boundaries = self._boundary_handler.use_boundaries
        if self._boundary_handler.time_window is not None:
            self._time_window = self._boundary_handler.time_window
        if self._boundary_handler.freq_window_shifted is not None:
            self._freq_window_shifted = self._boundary_handler.freq_window_shifted


    def _phase_factor(self, z: float):
        """Return exp(L_op * z) broadcasting over modes/frequencies."""
        return jnp.exp(self.L_op * jnp.asarray(z))

    def _to_physical_freq(self, A_w_tilde, z: float):
        """Transform interaction-picture field to physical frequency domain."""
        return self._phase_factor(z) * A_w_tilde

    def _to_interaction_freq(self, A_w_phys, z: float):
        """Transform physical frequency-domain field back to interaction picture."""
        return jnp.exp(-self.L_op * jnp.asarray(z)) * A_w_phys

    def solve(self, z_points, rtol=1e-5, atol=1e-8, max_steps=16**4, progress_bar=False, adjoint=None,
              use_fixed_step=False, dz=None):
        """Solve GMMNLSE using interaction picture with Diffrax.
        
        The GMMNLSE in the interaction picture is given by:
        
        ∂Ã_p(ω, z)/∂z = Ñ_p(ω, z) exp(-D_p(ω) z)
        
        where:
        - Ã_p(ω, z) = A_p(ω, z) exp(-D_p(ω) z) is the field in interaction picture
        - D_p(ω) is the dispersion operator
        - Ñ_p(ω, z) = F[η_p(t, z)] is the Fourier transform of the nonlinear operator
        
        The field is transformed back to the original picture after integration:
        
        A_p(ω, z) = Ã_p(ω, z) exp(D_p(ω) z)

        Args:
            z_points (jnp.ndarray): Distances to save solution (m). Must start at 0.
            rtol (float): Relative tolerance. Defaults to 1e-5. Only used with adaptive step size.
            atol (float): Absolute tolerance. Defaults to 1e-8. Only used with adaptive step size.
            max_steps (int): Maximum solver steps. Defaults to 16**4.
            progress_bar (bool): If True, show progress bar using tqdm. Defaults to False.
            adjoint (diffrax.AbstractAdjoint | None): Adjoint method. If None, omitted for speed.
            use_fixed_step (bool): Use fixed step size instead of adaptive. Defaults to False.
            dz (float | None): Fixed step size (m). If None and use_fixed_step=True, defaults to
                1e-6 m (1 micron).

        Returns:
            dict: Dictionary with 'ts', 'ys', 'stats', 'pulses', 'final_pulse'.
        """
        z0 = float(z_points[0])
        if abs(z0) > 1e-12:
            raise ValueError("z_points must start at 0 for the interaction picture formulation.")
        
        A_w_initial = self._apply_freq_window(
            jnp.fft.fft(self.pulse.field.copy(), axis=-1).astype(jnp.complex128)
        )
        
        y0 = A_w_initial
        
        rhs_args = (
            self.L_op,
            self._time_window,
            self._freq_window_shifted,
            self._use_boundaries,
            self.include_self_steepening,
            self.waveguide,
            self.pulse,
        )

        def rhs(z, A_w, args):
            """Right-hand side of ODE in interaction picture.
            
            Computes:
            
            dÃ_p(ω, z)/dz = Ñ_p(ω, z) exp(-D_p(ω) z)
            
            where Ñ_p is the Fourier transform of the nonlinear operator η_p.
            """
            (
                L_op,
                time_window,
                freq_window_shifted,
                use_boundaries,
                include_self_steepening,
                waveguide,
                pulse,
            ) = args

            # Transform to physical picture before evaluating the nonlinear operator
            phase_forward = jnp.exp(L_op * jnp.asarray(z))
            A_w_phys = phase_forward * A_w
            A_t = jnp.fft.ifft(A_w_phys, axis=-1)
            if use_boundaries:
                reshape_time = (1,) * (A_t.ndim - 1) + (time_window.shape[0],)
                A_t = A_t * time_window.reshape(reshape_time)
            
            N_t = waveguide.get_nonlinear_operator(
                A_t,
                include_self_steepening=include_self_steepening,
                pulse=pulse
            )
            if use_boundaries:
                reshape_time = (1,) * (N_t.ndim - 1) + (time_window.shape[0],)
                N_t = N_t * time_window.reshape(reshape_time)
            
            N_w = jnp.fft.fft(N_t, axis=-1)
            if use_boundaries:
                reshape_freq = (1,) * (N_w.ndim - 1) + (freq_window_shifted.shape[0],)
                shifted = jnp.fft.fftshift(N_w, axes=-1)
                tapered = shifted * freq_window_shifted.reshape(reshape_freq)
                N_w = jnp.fft.ifftshift(tapered, axes=-1)
            
            # Convert nonlinear term back into interaction picture
            phase_backward = jnp.exp(-L_op * jnp.asarray(z))
            return phase_backward * N_w
        
        term = diffrax.ODETerm(rhs)
        solver_map = {
            'Tsit5': diffrax.Tsit5,
            'Dopri5': diffrax.Dopri5,
            'Dopri8': diffrax.Dopri8,
            'Kvaerno3': diffrax.Kvaerno3,
            'Kvaerno5': diffrax.Kvaerno5,
        }
        
        if self.solver_method not in solver_map:
            raise ValueError(f"Unknown solver method '{self.solver_method}'. "
                           f"Available options: {list(solver_map.keys())}")
        
        solver = solver_map[self.solver_method]()
        saveat = diffrax.SaveAt(ts=z_points)
        
        if use_fixed_step:
            if dz is None:
                # Default to 1 micron (1e-6 m) for fixed step size
                dz = 1e-6
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = dz
        else:
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
            dt0 = None  # Let solver choose initial step
        
        if progress_bar:
            if hasattr(diffrax, "TqdmProgressMeter"):
                progress_meter = diffrax.TqdmProgressMeter()
            else:
                progress_meter = diffrax.NoProgressMeter()
        else:
            progress_meter = diffrax.NoProgressMeter()
        
        solve_kwargs = {
            'terms': term,
            'solver': solver,
            't0': z_points[0],
            't1': z_points[-1],
            'dt0': dt0,
            'y0': y0,
            'saveat': saveat,
            'stepsize_controller': stepsize_controller,
            'max_steps': max_steps,
            'progress_meter': progress_meter,
        }
        
        solve_kwargs['args'] = rhs_args
        if adjoint is not None:
            solve_kwargs['adjoint'] = adjoint
        
        solution = diffrax.diffeqsolve(**solve_kwargs)
        
        if solution.ys is None:
            raise RuntimeError("Diffrax solver returned None for solution.ys")
        A_w_all = solution.ys
        
        L_op = self.L_op
        exp_L_z = jnp.exp(L_op[jnp.newaxis, :, :] * z_points[:, jnp.newaxis, jnp.newaxis])
        
        A_w_all = A_w_all * exp_L_z
        A_t_all = jnp.fft.ifft(self._apply_freq_window(A_w_all), axis=-1)
        A_t_all = self._apply_time_window(A_t_all)
        
        if jnp.abs(z_points[0]) < 1e-12:
            A_t_all = A_t_all.at[0].set(self.pulse.field)
        
        pulses = []
        for i in range(len(z_points)):
            pulse_result = Pulse(
                sim_params=self.pulse.sim_params,
                field_t=A_t_all[i],
            )
            pulses.append(pulse_result)
        
        return SolverResult(
            pulses=pulses,
            z_points=z_points,
            stats=solution.stats
        )


class RK4IPSolver(SolverBase):
    """GMMNLSE solver using Runge-Kutta 4th order in Interaction Picture (RK4IP).

    Solves the Generalized Multimode Nonlinear Schrödinger Equation (GMMNLSE):
    
    ∂A_p(ω, z)/∂z = [D_p(ω) + N_p(ω, z)] A_p(ω, z)
    
    Uses interaction picture: Ã_p(ω, z) = A_p(ω, z) exp(-D_p(ω) z), integrates nonlinear part
    with 4th-order Runge-Kutta, then transforms back. This provides a good balance between
    accuracy and computational efficiency.
    """
    
    def __init__(
        self,
        waveguide,
        pulse,
        dz=1.0,
        include_self_steepening=False,
        boundary_mode: str = "both",
        time_window_fraction: float | tuple[float, float] | list[float] = 0.5,
        wavelength_window_um: Sequence[float] | np.ndarray | jnp.ndarray | None = None,
        freq_window_fraction: float | tuple[float, float] | list[float] | None = None,
        time_boundary_order: int | tuple[int, int] | list[int] | None = None,
        freq_boundary_order: int | tuple[int, int] | list[int] | None = None,
        boundary_order: int = 20,
    ):
        """Initialize RK4IP GMMNLSE solver.

        Args:
            waveguide (Waveguide): Waveguide object.
            pulse (Pulse): Initial pulse.
            dz (float): Step size (m). Defaults to 1.0.
            include_self_steepening (bool): Include self-steepening. Defaults to False.
            boundary_mode (str): Super-Gaussian boundary mode when self-steepening is
                active. Options: "both", "low", "high", or "none". Defaults to "both".
            time_window_fraction (float | tuple[float, float] | list[float]): Fraction of time 
                window where the time-domain boundary window is near 1.0. Can be a single 
                float (symmetric) or tuple/list [left, right] (asymmetric). Defaults to 0.5.
            wavelength_window_um (sequence | array | None): Optional [wl_min, wl_max] (µm)
                specifying the spectral passband to preserve. If provided, overrides
                freq_window_fraction.
            freq_window_fraction (float | tuple[float, float] | list[float] | None): Fractional 
                FWHM for the frequency-domain boundary window. Can be:
                - A single float: same fraction for both low and high frequency sides
                - A tuple/list [low, high]: different fractions for low and high sides
                - None: defaults to [0.6, 0.6] or computed from wavelength_window_um if provided
            time_boundary_order (int | tuple[int, int] | list[int] | None): Super-Gaussian order 
                for the time-domain boundary window. Can be a single int (symmetric) or 
                tuple/list [left, right] (asymmetric). If None, uses boundary_order.
            freq_boundary_order (int | tuple[int, int] | list[int] | None): Super-Gaussian order 
                for the frequency-domain boundary window. Can be:
                - A single int: same order for both low and high frequency sides
                - A tuple/list [low, high]: different orders for low and high sides
                - None: uses boundary_order for both sides
            boundary_order (int): Default super-Gaussian order for both boundaries when 
                time_boundary_order or freq_boundary_order are not specified. Defaults to 20.
        """
        self.waveguide = waveguide
        self.pulse = pulse
        self.dz = dz
        self.include_self_steepening = include_self_steepening
        boundary_mode_norm = boundary_mode.lower()
        if boundary_mode_norm not in {"both", "low", "high", "none"}:
            raise ValueError(
                f"boundary_mode must be 'both', 'low', 'high', or 'none', got '{boundary_mode}'"
            )
        self.boundary_mode = boundary_mode_norm
        self.L_op = self.waveguide.get_dispersion_operator(self.pulse._freqs_relative_thz)
        
        self._wavelength_limits_um = _normalize_wavelength_window(wavelength_window_um)
        self.boundary_order = boundary_order
        self._boundary_handler: Optional[BoundaryHandler] = None
        self._use_boundaries = False
        default_window = jnp.ones(self.pulse.n_points, dtype=jnp.complex128)
        self._time_window = default_window
        self._freq_window_shifted = default_window
        self.time_boundary_order = time_boundary_order if time_boundary_order is not None else boundary_order
        if freq_boundary_order is None:
            freq_order_low = boundary_order
            freq_order_high = boundary_order
        elif isinstance(freq_boundary_order, (int, np.integer)):
            freq_order_low = int(freq_boundary_order)
            freq_order_high = int(freq_boundary_order)
        elif isinstance(freq_boundary_order, (tuple, list)) and len(freq_boundary_order) == 2:
            freq_order_low = int(freq_boundary_order[0])
            freq_order_high = int(freq_boundary_order[1])
        else:
            raise ValueError(
                f"freq_boundary_order must be an int, tuple/list [low, high], or None, "
                f"got {type(freq_boundary_order).__name__}: {freq_boundary_order}"
            )
        
        # Store freq_boundary_order as tuple for consistency
        self.freq_boundary_order = (freq_order_low, freq_order_high)
        
        # Normalize time_window_fraction: single value -> tuple
        if isinstance(time_window_fraction, (int, float)):
            time_window_fraction = (float(time_window_fraction), float(time_window_fraction))
        elif isinstance(time_window_fraction, (tuple, list)) and len(time_window_fraction) == 2:
            time_window_fraction = (float(time_window_fraction[0]), float(time_window_fraction[1]))
        else:
            raise ValueError(
                "time_window_fraction must be a float or tuple/list [left, right], "
                f"got {type(time_window_fraction).__name__}: {time_window_fraction}"
            )
        self.time_window_fraction = time_window_fraction
        
        # Normalize time_boundary_order: single value -> tuple, None -> boundary_order
        if time_boundary_order is None:
            time_boundary_order = boundary_order
        if isinstance(time_boundary_order, (int, np.integer)):
            time_boundary_order = (int(time_boundary_order), int(time_boundary_order))
        elif isinstance(time_boundary_order, (tuple, list)) and len(time_boundary_order) == 2:
            time_boundary_order = (int(time_boundary_order[0]), int(time_boundary_order[1]))
        else:
            raise ValueError(
                "time_boundary_order must be an int, tuple/list [left, right], or None, "
                f"got {type(time_boundary_order).__name__}: {time_boundary_order}"
            )
        self.time_boundary_order = time_boundary_order
        
        # Normalize freq_window_fraction: single value -> tuple, None -> None
        if freq_window_fraction is None:
            freq_window_fraction_normalized = None
        elif isinstance(freq_window_fraction, (int, float)):
            freq_window_fraction_normalized = (float(freq_window_fraction), float(freq_window_fraction))
        elif isinstance(freq_window_fraction, (tuple, list)) and len(freq_window_fraction) == 2:
            freq_window_fraction_normalized = (float(freq_window_fraction[0]), float(freq_window_fraction[1]))
        else:
            raise ValueError(
                "freq_window_fraction must be a float, tuple/list [low, high], or None, "
                f"got {type(freq_window_fraction).__name__}: {freq_window_fraction}"
            )
        self.freq_window_fraction = freq_window_fraction_normalized
        
        self._boundary_handler = BoundaryHandler()
        self._boundary_handler.setup_boundaries(
            pulse=self.pulse,
            include_self_steepening=self.include_self_steepening,
            boundary_mode=self.boundary_mode,
            wavelength_limits_um=self._wavelength_limits_um,
            freq_window_fraction=freq_window_fraction_normalized,
            time_boundary_order=self.time_boundary_order,
            freq_boundary_order=self.freq_boundary_order,
            time_window_fraction=self.time_window_fraction,
        )
        self._use_boundaries = self._boundary_handler.use_boundaries
        if self._boundary_handler.time_window is not None:
            self._time_window = self._boundary_handler.time_window
        if self._boundary_handler.freq_window_shifted is not None:
            self._freq_window_shifted = self._boundary_handler.freq_window_shifted
    
    def _get_L_op(self, z):
        """Get dispersion operator at position z."""
        return self.L_op
    
    def plot_boundary(self, ax=None, xlim_min=None, xlim_max=None, **plot_kwargs):
        """Plot the frequency boundary window as a function of wavelength.
        
        Args:
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            xlim_min: Minimum x-axis limit (wavelength in µm). If None, uses default.
            xlim_max: Maximum x-axis limit (wavelength in µm). If None, uses default.
            **plot_kwargs: Additional keyword arguments passed to plt.plot().
        
        Returns:
            matplotlib.axes.Axes: The axes object used for plotting.
        """
        if not self._use_boundaries:
            raise ValueError("Boundaries are not enabled. Cannot plot boundary window.")
        
        freq_window = np.asarray(self._freq_window_shifted)
        wavelengths_um = np.asarray(self.pulse.wavelengths_um)
        
        # Filter out invalid wavelengths (negative, zero, or infinite) to match _wavelength_axis behavior
        # This ensures the minimum wavelength matches what's used in other plots
        mask = np.isfinite(wavelengths_um) & (wavelengths_um > 0.0)
        
        wavelengths_um = wavelengths_um[mask]
        freq_window = freq_window[mask]
        
        sort_idx = np.argsort(wavelengths_um)
        wavelengths_sorted = wavelengths_um[sort_idx]
        boundary_values = np.abs(freq_window[sort_idx])
        
        # Normalize boundary to peak and apply fixed noise floor for plotting
        boundary_max = np.max(boundary_values) if len(boundary_values) > 0 else 1.0
        if boundary_max > 0:
            # Normalize to peak (same as spectral plots do)
            boundary_normalized = boundary_values / boundary_max
            # Apply fixed noise floor (independent of signal power)
            noise_floor_normalized = TOLERANCE_DIVISION
            boundary_clipped = np.maximum(boundary_normalized, noise_floor_normalized)
            # Convert to dB
            boundary_db = 10 * np.log10(np.maximum(boundary_clipped, 1e-30))
        else:
            boundary_db = np.full_like(boundary_values, -np.inf)
        
        created_new_figure = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(wavelengths_sorted, boundary_db, **plot_kwargs)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Boundary Window Value (dB)")
        ax.set_title("Frequency Boundary Window vs Wavelength")
        ax.grid(True, alpha=0.3)
        
        if self._wavelength_limits_um is not None:
            wl_min, wl_max = self._wavelength_limits_um
            ax.axvline(wl_min, color='r', linestyle='--', alpha=0.5, 
                      label=f'Min wavelength: {wl_min:.2f} µm')
            ax.axvline(wl_max, color='b', linestyle='--', alpha=0.5,
                      label=f'Max wavelength: {wl_max:.2f} µm')
            ax.legend()
        
        if xlim_min is not None or xlim_max is not None:
            current_xlim = ax.get_xlim()
            xlim_left = xlim_min if xlim_min is not None else current_xlim[0]
            xlim_right = xlim_max if xlim_max is not None else current_xlim[1]
            ax.set_xlim(left=xlim_left, right=xlim_right)
        
        if created_new_figure:
            plt.tight_layout()
            plt.show()
        
        return ax
    

    def _rk4_step(self, A_w, z, dz, a5_prev=None):
        """Perform one RK4IP step based on equations (9)-(11) and (12a)-(12f).
        
        Implements the Fourth-Order Runge-Kutta in the Interaction Picture Method (RK4IP)
        as described in the paper. The algorithm advances A(z, T) to A(z + h, T) in the
        normal picture using a step size h.
        
        The RK4IP method transforms the NLSE into an interaction picture to separate
        dispersion (D̂) from nonlinear effects (N̂). The transformation is defined by:
        
        (9) A_I = exp(-(z - z') * D̂) * A
        (10) ∂A_I/∂z = N_I_hat * A_I
        (11) N_I_hat = exp(-(z - z') * D̂) * N̂ * exp((z - z') * D̂)
        
        where z' is the separation distance between the interaction and normal pictures.
        
        The RK4IP algorithm uses the specific choice z' = z + h/2, which eliminates
        the dispersion exponentials for the midpoint trajectories k_2 and k_3, reducing
        the computational cost from 16 FFTs to 8 FFTs per step.
        
        The RK4IP step equations:
        
        (12a) A_I = exp((h/2) * D̂) * A(z, T)
        (12b) k_1 = exp((h/2) * D̂) * [h * N̂(A(z, T))] * A(z, T)
        (12c) k_2 = h * N̂(A_I + k_1/2) * [A_I + k_1/2]
        (12d) k_3 = h * N̂(A_I + k_2/2) * [A_I + k_2/2]
        (12e) k_4 = h * N̂(exp((h/2) * D̂) * (A_I + k_3)) * exp((h/2) * D̂) * [A_I + k_3]
        (12f) A(z + h, T) = exp((h/2) * D̂) * [A_I + k_1/6 + k_2/3 + k_3/3] + k_4/6
        
        where:
        - D̂ is the dispersion operator
        - N̂ is the nonlinear operator
        - h is the step size (dz)
        - A(z, T) is the field in the normal picture
        - A_I is the field in the interaction picture at the midpoint z' = z + h/2
        
        The separation distance z' = z + h/2 eliminates the dispersion exponentials
        for k_2 and k_3 (the midpoint trajectories), providing computational efficiency.
        
        Args:
            A_w_tilde: Field in interaction picture at z (frequency domain)
            z: Current position
            dz: Step size h (m)
            a5_prev: Previous step's term for reuse (optional, for future optimization)
        
        Returns:
            A_w_tilde_new: Field in interaction picture at z+dz (frequency domain)
        """
        L_op_z = self._get_L_op(z)
        h = dz
        
        # Pre-compute dispersion operator for half-step (h/2)
        D_half = L_op_z * (h / 2)
        expD_half = jnp.exp(D_half)
        
        # -------------------------------------------------------------------
        # Equation (12a): A_I = exp((h/2) * D_hat) * A(z, T) [cite: 107]
        # -------------------------------------------------------------------
        # A_I_w is the field transformed into the interaction picture at z' = z + h/2
        A_I_w = expD_half * A_w
        
        # -------------------------------------------------------------------
        # Equation (12b): k_1 = exp((h/2) * D_hat) * [h * N_hat(A(z, T))] * A(z, T)
        # -------------------------------------------------------------------
        # 1. Get physical field at z (start) to evaluate nonlinearity
        A_at_z_t = jnp.fft.ifft(A_w, axis=-1)
        N_at_z_t = self._get_nonlinear_operator_time(A_at_z_t)
        N_at_z_w = jnp.fft.fft(N_at_z_t, axis=-1)
        
        # 2. Propagate the slope from z to the midpoint (z + h/2)
        k_1_w = expD_half * (h * N_at_z_w)
        
        # -------------------------------------------------------------------
        # Equation (12c): k_2 = h * N_hat(A_I + k_1/2) * [A_I + k_1/2]
        # -------------------------------------------------------------------
        # Estimate field at midpoint.
        A_mid_guess_1_w = A_I_w + k_1_w / 2
        A_mid_guess_1_t = jnp.fft.ifft(A_mid_guess_1_w, axis=-1)
        
        N_mid_1_t = self._get_nonlinear_operator_time(A_mid_guess_1_t)
        N_mid_1_w = jnp.fft.fft(N_mid_1_t, axis=-1)
        
        k_2_w = h * N_mid_1_w
        
        # -------------------------------------------------------------------
        # Equation (12d): k_3 = h * N_hat(A_I + k_2/2) * [A_I + k_2/2]
        # -------------------------------------------------------------------
        # Refined estimate at midpoint.
        A_mid_guess_2_w = A_I_w + k_2_w / 2
        A_mid_guess_2_t = jnp.fft.ifft(A_mid_guess_2_w, axis=-1)
        
        N_mid_2_t = self._get_nonlinear_operator_time(A_mid_guess_2_t)
        N_mid_2_w = jnp.fft.fft(N_mid_2_t, axis=-1)
        
        k_3_w = h * N_mid_2_w

        # -------------------------------------------------------------------
        # Equation (12e): 
        # k_4 = h * N_hat( exp((h/2)D) * (A_I + k_3) )
        # -------------------------------------------------------------------
        # 1. Estimate field at the midpoint (A_I + k_3)
        A_mid_final_guess_w = A_I_w + k_3_w
        
        # 2. Propagate this guess from Midpoint to End (z + h)
        # This requires applying the half-step dispersion operator
        A_end_phys_w = expD_half * A_mid_final_guess_w
        A_end_phys_t = jnp.fft.ifft(A_end_phys_w, axis=-1)
        
        # 3. Evaluate nonlinearity at the End
        N_end_t = self._get_nonlinear_operator_time(A_end_phys_t)
        N_end_w = jnp.fft.fft(N_end_t, axis=-1)
        
        k_4_w = h * N_end_w 
        
        # -------------------------------------------------------------------
        # Equation (12f): 
        # A(z+h) = exp((h/2)D) * [A_I + k_1/6 + k_2/3 + k_3/3] + k_4/6
        # -------------------------------------------------------------------
        
        A_w_new = expD_half * (A_I_w + k_1_w/6 + k_2_w/3 + k_3_w/3) + k_4_w/6
        
        # Apply boundary window
        A_w_new = self._apply_freq_window(A_w_new)
        
        return A_w_new
    
    def _get_nonlinear_operator_time(self, A_t):
        """Get nonlinear operator in time domain (helper for RK4IP).
        """
        nonlinear_term = self.waveguide.get_nonlinear_operator(
            A_t,
            include_self_steepening=self.include_self_steepening,
            pulse=self.pulse
        )
        return nonlinear_term
    
    def solve(self, z_points, progress_bar=False):
        """Solve GMMNLSE using RK4IP method.
        
        Solves the GMMNLSE in interaction picture using 4th-order Runge-Kutta:
        
        ∂Ã_p(ω, z)/∂z = Ñ_p(ω, z) exp(-D_p(ω) z)
        
        The solution is transformed back to the original picture at each save point:
        
        A_p(ω, z) = Ã_p(ω, z) exp(D_p(ω) z)

        Args:
            z_points (jnp.ndarray): Distances to save solution (m). Must start at 0.
            progress_bar (bool): If True, show progress bar using tqdm. Defaults to False.

        Returns:
            dict: Dictionary with 'ts', 'ys', 'stats', 'pulses', 'final_pulse'.
        """
        z0 = float(z_points[0])
        if abs(z0) > 1e-12:
            raise ValueError("z_points must start at 0 for the interaction picture formulation.")
        
        A_w = self._apply_freq_window(
            jnp.fft.fft(self.pulse.field, axis=-1)
        ).astype(jnp.complex128)
        
        # Transform initial condition to interaction picture at z = 0
        A_w_tilde = A_w
        
        n_save = len(z_points)
        output = jnp.zeros((n_save, self.pulse.n_modes, self.pulse.n_points), dtype=jnp.complex128)
        output = output.at[0].set(self.pulse.field)
        
        current_z = z_points[0]
        save_idx = 1
        num_steps = 0

        tqdm_progress_bar = None
        if progress_bar:
            if tqdm is not None:
                total_distance_mm = float(z_points[-1] - z_points[0]) * 1000.0
                tqdm_progress_bar = tqdm(
                    total=total_distance_mm,
                    unit="mm",
                    desc="RK4IP Progress",
                    bar_format='{l_bar}{bar}| {n:.3f}/{total:.3f} {unit} [{elapsed}<{remaining}, {rate_fmt}]'
                )
        
        while save_idx < n_save:
            target_z = z_points[save_idx]
            
            while current_z < target_z - self.dz/2:
                A_w_tilde_next = self._rk4_step(A_w_tilde, current_z, self.dz)
                if np.isnan(np.asarray(A_w_tilde_next)).any():
                    palpha = jnp.max(jnp.abs(jnp.fft.ifft(self._apply_freq_window(A_w_tilde), axis=-1)))
                    peak_amp = float(palpha) if jnp.isfinite(palpha) else float("nan")
                    palpha_freq = jnp.max(jnp.abs(A_w_tilde))
                    peak_amp_freq = float(palpha_freq) if jnp.isfinite(palpha_freq) else float("nan")
                    raise FloatingPointError(
                        f"NaN encountered at z={current_z} after step size {self.dz}. "
                        f"Peak |A| prior to failure: {peak_amp:.3e}, "
                        f"peak |A_w|: {peak_amp_freq:.3e}"
                    )
                A_w_tilde = A_w_tilde_next
                current_z += self.dz
                num_steps += 1
                
                if tqdm_progress_bar is not None:
                    if isinstance(self.dz, (jnp.ndarray, np.ndarray)):
                        dz_float = float(self.dz.item())
                    else:
                        dz_float = float(self.dz)
                    tqdm_progress_bar.update(dz_float * 1000.0)
            
            if abs(current_z - target_z) > 1e-10:
                remaining_dz = target_z - current_z
                A_w_tilde_next = self._rk4_step(A_w_tilde, current_z, remaining_dz)
                if np.isnan(np.asarray(A_w_tilde_next)).any():
                    palpha = jnp.max(jnp.abs(jnp.fft.ifft(self._apply_freq_window(A_w_tilde), axis=-1)))
                    peak_amp = float(palpha) if jnp.isfinite(palpha) else float("nan")
                    palpha_freq = jnp.max(jnp.abs(A_w_tilde))
                    peak_amp_freq = float(palpha_freq) if jnp.isfinite(palpha_freq) else float("nan")
                    raise FloatingPointError(
                        f"NaN encountered while finalizing step to z={target_z}. "
                        f"Peak |A| prior to failure: {peak_amp:.3e}, "
                        f"peak |A_w|: {peak_amp_freq:.3e}"
                    )
                A_w_tilde = A_w_tilde_next
                current_z = target_z
                num_steps += 1
                
                if tqdm_progress_bar is not None:
                    if isinstance(remaining_dz, (jnp.ndarray, np.ndarray)):
                        remaining_dz_float = float(remaining_dz.item())
                    else:
                        remaining_dz_float = float(remaining_dz)
                    tqdm_progress_bar.update(remaining_dz_float * 1000.0)
            
            A_w = A_w_tilde
            A_w = self._apply_freq_window(A_w)
           
            A_t = jnp.fft.ifft(A_w, axis=-1)
            output = output.at[save_idx].set(A_t)
            save_idx += 1
        
        if tqdm_progress_bar is not None:
            tqdm_progress_bar.close()
        
        stats = {'num_steps': num_steps}
        pulses = []
        for i in range(len(z_points)):
            pulse = Pulse(
                sim_params=self.pulse.sim_params,
                field_t=output[i]
            )
            pulses.append(pulse)
        
        return SolverResult(
            pulses=pulses,
            z_points=z_points,
            stats=stats
        )


class RK4IPPMLSolver(RK4IPSolver):
    """RK4IP solver with temporal perfectly matched layer (PML) absorption."""

    def __init__(
        self,
        waveguide,
        pulse,
        dz=1.0,
        include_self_steepening=False,
        boundary_mode: str = "both",
        time_window_fraction: float | tuple[float, float] | list[float] = 0.5,
        wavelength_window_um: Sequence[float] | np.ndarray | jnp.ndarray | None = None,
        freq_window_fraction: float | tuple[float, float] | list[float] | None = None,
        time_boundary_order: int | tuple[int, int] | list[int] | None = None,
        freq_boundary_order: int | tuple[int, int] | list[int] | None = None,
        boundary_order: int = 20,
        pml_width_ps: float = 1.0,
        pml_order: int = 3,
        pml_attenuation_db: float = 60.0,
    ):
        """
        Initialize RK4IPPMLSolver.

        Args:
            waveguide: Waveguide instance.
            pulse: Pulse instance.
            dz (float): Step size in meters.
            include_self_steepening (bool): Whether to include self-steepening.
            boundary_mode (str): Boundary damping mode ('both', 'low', 'high', 'none').
            pml_width_ps (float): Width of the absorbing layer in picoseconds.
            pml_order (int): Polynomial order of the absorption profile.
            pml_attenuation_db (float): Target attenuation (dB) across the PML.
        """
        super().__init__(
            waveguide=waveguide,
            pulse=pulse,
            dz=dz,
            include_self_steepening=include_self_steepening,
            boundary_mode=boundary_mode,
            time_window_fraction=time_window_fraction,
            wavelength_window_um=wavelength_window_um,
            freq_window_fraction=freq_window_fraction,
            time_boundary_order=time_boundary_order,
            freq_boundary_order=freq_boundary_order,
            boundary_order=boundary_order,
        )
        self._time_window = jnp.ones_like(self._time_window)
        self._freq_window_shifted = jnp.ones_like(self._freq_window_shifted)
        self._use_boundaries = True
        if pml_width_ps <= 0.0:
            raise ValueError("pml_width_ps must be positive.")
        if pml_order < 1:
            raise ValueError("pml_order must be at least 1.")
        if pml_attenuation_db <= 0.0:
            raise ValueError("pml_attenuation_db must be positive.")

        times = np.asarray(self.pulse.t)
        t_max = float(np.max(np.abs(times)))
        t_physical = t_max - float(pml_width_ps)
        if t_physical <= 0.0:
            raise ValueError(
                "pml_width_ps is too large for the pulse time window. "
                "Increase time_window_ps or reduce pml_width_ps."
            )

        eta = np.clip((np.abs(times) - t_physical) / float(pml_width_ps), 0.0, 1.0)
        self._pml_profile = eta ** pml_order
        self._pml_order = pml_order
        self._pml_attenuation_db = float(pml_attenuation_db)
        self._pml_sigma = None

    def _configure_pml(self, z_points: jnp.ndarray) -> None:
        total_length = float(z_points[-1] - z_points[0])
        if total_length <= 0.0:
            self._pml_sigma = None
            return
        ratio = 10.0 ** (-self._pml_attenuation_db / 20.0)
        sigma_max = (self._pml_order + 1) * (-np.log(ratio)) / total_length
        self._pml_sigma = jnp.asarray(sigma_max * self._pml_profile, dtype=jnp.float64)

    def _apply_time_window(self, array):
        """Override base windowing; PML handles absorption without static damping."""
        return array

    def _get_nonlinear_operator_time(self, A_t):
        A_t_windowed = A_t
        nonlinear_term = self.waveguide.get_nonlinear_operator(
            A_t_windowed,
            include_self_steepening=self.include_self_steepening,
            pulse=self.pulse,
        )
        if self._pml_sigma is None:
            return nonlinear_term
        reshape = (1,) * (nonlinear_term.ndim - 1) + (self._pml_sigma.shape[0],)
        sigma = self._pml_sigma.reshape(reshape)
        return nonlinear_term - sigma * A_t_windowed

    def solve(self, z_points, progress_bar=False):
        """Solve GMMNLSE using RK4IP method with PML boundaries.
        
        Args:
            z_points (jnp.ndarray): Distances to save solution (m). Must start at 0.
            progress_bar (bool): If True, show progress bar using tqdm. Defaults to False.
        
        Returns:
            dict: Dictionary with 'ts', 'ys', 'stats', 'pulses', 'final_pulse'.
        """
        self._configure_pml(z_points)
        result = super().solve(z_points, progress_bar=progress_bar)
        self._pml_sigma = None
        return result

