import scipy.io as sio
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Union, Optional, Literal, Tuple
from .constants import C_m_ps, C_m_s
from scipy.interpolate import CubicSpline


def fit_dispersion_from_neff(
    freqs: np.ndarray,
    freq_disp: float,
    n_eff: np.ndarray,
    orders: int,
    in_si_units: bool = False
) -> np.ndarray:
    """
    Fit dispersion coefficients from effective index data.
    
    Fits a polynomial to beta(ω) = n_eff * ω / C_m_ps and extracts
    Taylor expansion coefficients around the dispersion frequency.
    
    Units:
    - If in_si_units=False (default): freqs in THz
    - If in_si_units=True: freqs in Hz (will be converted to THz)
    - freq_disp: dispersion frequency in same units as freqs
    - n_eff: effective index (dimensionless)
    - Returns: betas in ps^n/m units, shape (orders, num_modes)
    
    Args:
        freqs: Array of frequencies, shape (n_freqs,)
            If in_si_units=False: frequencies in THz
            If in_si_units=True: frequencies in Hz
        freq_disp: Center frequency for Taylor expansion (same units as freqs)
        n_eff: Effective index array, shape (n_freqs, n_modes) or (n_freqs,) for single mode
        orders: Number of dispersion orders to compute (including beta_0)
        in_si_units: If True, frequencies are in Hz (SI units) and will be converted to THz.
            If False, frequencies are assumed to be in THz (default: False)
    
    Returns:
        betas: Taylor expansion coefficients, shape (orders, n_modes)
            Units: ps^n/m
    """
    # Convert frequencies to SI (Hz) if provided in THz
    if in_si_units:
        freqs_hz = np.asarray(freqs, dtype=np.float64)
        freq_disp_hz = float(freq_disp)
    else:
        freqs_hz = np.asarray(freqs, dtype=np.float64) * 1e12
        freq_disp_hz = float(freq_disp) * 1e12

    # Ensure n_eff is 2D
    if n_eff.ndim == 1:
        n_eff = n_eff[:, np.newaxis]

    n_modes = n_eff.shape[1]
    num_disp_orders = orders - 1

    # Angular frequencies in rad/s (SI)
    omega = 2.0 * np.pi * freqs_hz  # rad/s
    omega_disp = 2.0 * np.pi * freq_disp_hz  # rad/s

    # Shift angular frequency for numerical stability
    omega_shifted = omega - omega_disp

    # Container for beta coefficients, shape (orders, n_modes)
    b_coefficients = np.zeros((orders, n_modes), dtype=np.float64)

    # Calculate beta(omega) = n_eff * omega / C_m_s for each mode (rad/m)
    beta_calc = np.real(n_eff) * (omega[:, np.newaxis] / C_m_s)

    for mode_idx in range(n_modes):
        beta_mode = beta_calc[:, mode_idx]

        # Fit polynomial around the expansion point using shifted omega
        beta_fit = np.polyfit(omega_shifted, beta_mode, orders - 1)

        for disp_order in range(orders):
            deriv_coeff = np.polyder(beta_fit, m=disp_order)
            beta_deriv_si = np.polyval(deriv_coeff, 0.0)  # evaluate at omega = omega_disp

            # Convert from s^n/m to ps^n/m (for disp_order >= 1). disp_order = 0 unchanged.
            scale = (1e12) ** disp_order
            b_coefficients[disp_order, mode_idx] = beta_deriv_si * scale

    return b_coefficients


def _finite_difference_weights(x: np.ndarray, x0: float, order: int) -> np.ndarray:
    """
    Compute Fornberg finite-difference weights for derivative approximation.

    Parameters
    ----------
    x : array-like
        Sample locations.
    x0 : float
        Evaluation point.
    order : int
        Derivative order.

    Returns
    -------
    np.ndarray
        Finite-difference weights for the requested derivative order.
    """
    x = np.asarray(x, dtype=float)
    n_points = x.size
    if order >= n_points:
        raise ValueError("order must be less than the number of stencil points")
    coeff = np.zeros((n_points, order + 1), dtype=float)
    coeff[0, 0] = 1.0
    c1 = 1.0
    c4 = x[0] - x0
    for i in range(1, n_points):
        mn = min(i, order)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            if c3 == 0.0:
                raise ValueError("Points in stencil must be distinct")
            c2 *= c3
            for k in range(mn, 0, -1):
                coeff[i, k] = (
                    c1 * (k * coeff[i - 1, k - 1] - c5 * coeff[i - 1, k])
                ) / c2
            coeff[i, 0] = (-c1 * c5 * coeff[i - 1, 0]) / c2
            for k in range(mn, 0, -1):
                coeff[j, k] = (c4 * coeff[j, k] - k * coeff[j, k - 1]) / c3
            coeff[j, 0] = (c4 * coeff[j, 0]) / c3
        c1 = c2
    return coeff[:, order]


def _compute_betas_from_neff(
    freqs: np.ndarray,
    n_eff: np.ndarray,
    freq_disp: float,
    orders: int,
    method: Literal["polynomial", "spline", "finite_difference"] = "polynomial",
    in_si_units: bool = False,
    fd_window: int = 5,
) -> np.ndarray:
    """
    Compute dispersion coefficients using a selected differentiation method.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if in_si_units:
        freqs_hz = freqs
        freq_disp_hz = float(freq_disp)
    else:
        freqs_hz = freqs * 1e12
        freq_disp_hz = float(freq_disp) * 1e12

    omega = 2.0 * np.pi * freqs_hz
    omega_disp = 2.0 * np.pi * freq_disp_hz

    n_eff = np.asarray(n_eff)
    if n_eff.ndim == 1:
        n_eff = n_eff[:, np.newaxis]

    n_modes = n_eff.shape[1]
    betas = np.zeros((orders, n_modes), dtype=np.float64)

    if method == "polynomial":
        return fit_dispersion_from_neff(freqs, freq_disp, n_eff, orders, in_si_units=in_si_units)

    for mode_idx in range(n_modes):
        n_mode = np.real(n_eff[:, mode_idx])
        beta_mode = n_mode * (omega / C_m_s)

        if method == "spline":
            cs = CubicSpline(omega, n_mode, bc_type="natural")
            n_derivs = [cs(omega_disp, nu) for nu in range(orders)]

            betas[0, mode_idx] = beta_mode[np.argmin(np.abs(omega - omega_disp))]
            for order in range(1, orders):
                term = omega_disp * n_derivs[order] + order * n_derivs[order - 1]
                betas[order, mode_idx] = term / C_m_s

        elif method == "finite_difference":
            idx_center = int(np.argmin(np.abs(freqs_hz - freq_disp_hz)))
            betas[0, mode_idx] = beta_mode[idx_center]
            for order in range(1, orders):
                half_window = max(int(fd_window), (order + 1) // 2)
                start = max(0, int(idx_center - half_window))
                end = min(len(omega), int(idx_center + half_window + 1))
                if end - start <= order:
                    raise ValueError(
                        f"Not enough points to compute order {order} finite difference derivative."
                    )
                omega_window = omega[start:end]
                beta_window = beta_mode[start:end]
                weights = _finite_difference_weights(omega_window, omega_disp, order)
                betas[order, mode_idx] = np.dot(weights, beta_window)
        else:
            raise ValueError(f"Unknown beta computation method '{method}'")

    # Convert higher-order derivatives from s^n/m to ps^n/m
    for order in range(1, orders):
        betas[order, :] *= (1e12) ** order

    return betas


def load_waveguide_arrays(
    betas_path: Union[str, Path],
    s_tensor_path: Union[str, Path],
    s_tensor_freq_path: Optional[Union[str, Path]] = None,
    width_index: int = 0,
    betas_in_si_units: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load waveguide arrays from .npy files.
    
    Utility function to load betas and s_tensor arrays from files.
    File I/O is handled outside the library; this is a convenience function.
    
    Args:
        betas_path: Path to betas file (.npy).
        s_tensor_path: Path to s_tensor file (.npy).
        s_tensor_freq_path: Optional path to frequency-dependent s_tensor file (.npy).
        width_index: Index to extract if arrays have width dimension. Defaults to 0.
        betas_in_si_units: If True, betas are in SI units (s^n/m) and will be converted
            to ps^n/m. Defaults to False.
    
    Returns:
        Tuple of (betas, s_tensor, s_tensor_freq, freqs_hz).
        s_tensor_freq and freqs_hz are None if s_tensor_freq_path is not provided.
        freqs_hz is None if s_tensor_freq is None.
    
    Raises:
        FileNotFoundError: If required files don't exist.
        ValueError: If array shapes are invalid.
    """
    betas_path = Path(betas_path)
    s_tensor_path = Path(s_tensor_path)
    
    if not betas_path.exists():
        raise FileNotFoundError(f"Betas file not found: {betas_path}")
    if not s_tensor_path.exists():
        raise FileNotFoundError(f"S-tensor file not found: {s_tensor_path}")
    
    # Load betas
    betas_full = np.load(betas_path)
    if betas_full.ndim == 2:
        betas = betas_full
    elif betas_full.ndim == 3:
        if width_index < 0 or width_index >= betas_full.shape[0]:
            raise ValueError(
                f"width_index {width_index} out of range [0, {betas_full.shape[0]})"
            )
        betas = betas_full[width_index]
    else:
        raise ValueError(
            f"betas must have 2 or 3 dimensions, got {betas_full.ndim}. "
            f"Expected shape: (betas, modes) or (widths, betas, modes)"
        )
    
    if betas.shape[0] == 0 or betas.shape[1] == 0:
        raise ValueError("betas must have at least one coefficient and one mode")
    
    # Convert from SI units if needed
    if betas_in_si_units:
        betas_converted = np.zeros_like(betas)
        for i in range(betas.shape[0]):
            betas_converted[i, :] = betas[i, :] * (1e12)**i
        betas = betas_converted
    
    # Load s_tensor
    s_tensor_full = np.load(s_tensor_path)
    if s_tensor_full.ndim == 4:
        s_tensor = s_tensor_full
    elif s_tensor_full.ndim == 5:
        if width_index < 0 or width_index >= s_tensor_full.shape[0]:
            raise ValueError(
                f"width_index {width_index} out of range [0, {s_tensor_full.shape[0]})"
            )
        s_tensor = s_tensor_full[width_index]
    else:
        raise ValueError(
            f"s_tensor must have 4 or 5 dimensions, got {s_tensor_full.ndim}. "
            f"Expected shape: (num_modes, num_modes, num_modes, num_modes) or "
            f"(widths, num_modes, num_modes, num_modes, num_modes), "
            f"got {s_tensor_full.shape}"
        )
    
    if not all(s_tensor.shape[0] == s_tensor.shape[i] for i in range(1, 4)):
        raise ValueError(
            f"s_tensor must have equal mode dimensions. "
            f"Expected all dimensions to be equal, got shape: {s_tensor.shape}"
        )
    
    if betas.shape[1] != s_tensor.shape[0]:
        raise ValueError(
            f"Number of modes must match: betas has {betas.shape[1]}, "
            f"s_tensor has {s_tensor.shape[0]}"
        )
    
    # Load frequency-dependent s_tensor if provided
    s_tensor_freq = None
    freqs_hz = None
    if s_tensor_freq_path is not None:
        s_tensor_freq_path = Path(s_tensor_freq_path)
        if s_tensor_freq_path.exists():
            s_tensor_freq_full = np.load(s_tensor_freq_path)
            if s_tensor_freq_full.ndim == 5:
                s_tensor_freq = s_tensor_freq_full
            elif s_tensor_freq_full.ndim == 6:
                if width_index < 0 or width_index >= s_tensor_freq_full.shape[0]:
                    raise ValueError(
                        f"width_index {width_index} out of range "
                        f"[0, {s_tensor_freq_full.shape[0]})"
                    )
                s_tensor_freq = s_tensor_freq_full[width_index]
            else:
                raise ValueError(
                    f"s_tensor_freq must have 5 or 6 dimensions, got {s_tensor_freq_full.ndim}. "
                    f"Expected shape: (n_freqs, num_modes, num_modes, num_modes, num_modes) or "
                    f"(widths, n_freqs, num_modes, num_modes, num_modes, num_modes), "
                    f"got {s_tensor_freq_full.shape}"
                )
            
            if s_tensor_freq.shape[1] != s_tensor.shape[0]:
                raise ValueError(
                    f"Number of modes must match: s_tensor has {s_tensor.shape[0]}, "
                    f"s_tensor_freq has {s_tensor_freq.shape[1]}"
                )
    
    return betas, s_tensor, s_tensor_freq, freqs_hz


class WaveguideLoader:
    """Convenience class for preparing waveguide data arrays.
    
    Handles validation, unit conversion, and mode slicing for waveguide arrays.
    Does not handle file I/O - arrays should be loaded externally using
    `load_waveguide_arrays()` or numpy directly.
    
    Output shapes:
    - betas: (num_orders, num_modes) - for Waveguide class
    - s_tensor: (num_modes, num_modes, num_modes, num_modes) - for Waveguide class
    """
    
    def __init__(
        self,
        betas: Union[np.ndarray, jnp.ndarray],
        s_tensor: Union[np.ndarray, jnp.ndarray],
        s_tensor_freq: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        freqs_hz: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        convert_to_jax: bool = True,
        num_modes: Optional[int] = None,
    ):
        """Initialize WaveguideLoader from arrays.
        
        Args:
            betas: Betas array, shape (num_orders, num_modes) or (widths, num_orders, num_modes).
                If 3D, first width is used.
            s_tensor: S_tensor array, shape (num_modes, num_modes, num_modes, num_modes) or
                (widths, num_modes, num_modes, num_modes, num_modes). If 5D, first width is used.
            s_tensor_freq: Optional frequency-dependent s_tensor array,
                shape (n_freqs, num_modes, num_modes, num_modes, num_modes) or
                (widths, n_freqs, num_modes, num_modes, num_modes, num_modes).
                If 6D, first width is used.
            freqs_hz: Optional frequency grid in Hz corresponding to s_tensor_freq,
                shape (n_freqs,). Required if s_tensor_freq is provided.
            convert_to_jax: If True, convert arrays to JAX arrays. Defaults to True.
            num_modes: Number of modes to use. If None, uses all available modes.
                If specified, slices arrays to use only the first num_modes modes.
        
        Raises:
            ValueError: If array shapes are invalid or don't match.
        """
        # Extract single width if arrays have width dimension
        betas = np.asarray(betas)
        if betas.ndim == 3:
            betas = betas[0]  # Use first width
        elif betas.ndim != 2:
            raise ValueError(
                f"betas must have 2 or 3 dimensions, got {betas.ndim}. "
                f"Expected shape: (num_orders, num_modes) or (widths, num_orders, num_modes)"
            )
        
        s_tensor = np.asarray(s_tensor)
        if s_tensor.ndim == 5:
            s_tensor = s_tensor[0]  # Use first width
        elif s_tensor.ndim != 4:
            raise ValueError(
                f"s_tensor must have 4 or 5 dimensions, got {s_tensor.ndim}. "
                f"Expected shape: (num_modes, num_modes, num_modes, num_modes) or "
                f"(widths, num_modes, num_modes, num_modes, num_modes)"
            )
        
        if s_tensor_freq is not None:
            s_tensor_freq = np.asarray(s_tensor_freq)
            if s_tensor_freq.ndim == 6:
                s_tensor_freq = s_tensor_freq[0]  # Use first width
            elif s_tensor_freq.ndim != 5:
                raise ValueError(
                    f"s_tensor_freq must have 5 or 6 dimensions, got {s_tensor_freq.ndim}"
                )
            
            if freqs_hz is None:
                raise ValueError("freqs_hz must be provided when s_tensor_freq is provided")
            freqs_hz = np.asarray(freqs_hz)
            # s_tensor_freq is guaranteed to be not None here due to outer if check
            assert s_tensor_freq is not None
            if len(freqs_hz) != s_tensor_freq.shape[0]:
                raise ValueError(
                    f"freqs_hz length ({len(freqs_hz)}) must match first dimension "
                    f"of s_tensor_freq ({s_tensor_freq.shape[0]})"
                )
        
        # Validate shapes
        if betas.shape[0] == 0 or betas.shape[1] == 0:
            raise ValueError("betas must have at least one coefficient and one mode")
        
        if not all(s_tensor.shape[0] == s_tensor.shape[i] for i in range(1, 4)):
            raise ValueError("s_tensor must have equal mode dimensions")
        
        if betas.shape[1] != s_tensor.shape[0]:
            raise ValueError(
                f"Number of modes must match: betas has {betas.shape[1]}, "
                f"s_tensor has {s_tensor.shape[0]}"
            )
        
        if s_tensor_freq is not None and s_tensor.shape[0] != s_tensor_freq.shape[1]:
            raise ValueError(
                f"Number of modes must match: s_tensor has {s_tensor.shape[0]}, "
                f"s_tensor_freq has {s_tensor_freq.shape[1]}"
            )
        
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
        
        # Store arrays
        self._store_arrays(convert_to_jax, betas, s_tensor, s_tensor_freq, freqs_hz)
        
        # Store metadata
        self.num_orders = betas.shape[0]
        self.num_modes = betas.shape[1]
        self.betas_shape = betas.shape
        self.s_tensor_shape = s_tensor.shape


    def _store_arrays(
        self,
        convert_to_jax: bool,
        betas: Union[np.ndarray, jnp.ndarray],
        s_tensor: Union[np.ndarray, jnp.ndarray],
        s_tensor_freq: Optional[Union[np.ndarray, jnp.ndarray]],
        freqs_hz: Optional[Union[np.ndarray, jnp.ndarray]],
    ) -> None:
        """Store arrays as instance attributes, converting to JAX if requested.
        
        Args:
            convert_to_jax: If True, convert arrays to JAX arrays.
            betas: Betas array.
            s_tensor: S_tensor array.
            s_tensor_freq: Frequency-dependent s_tensor array (optional).
            freqs_hz: Frequency grid in Hz (optional).
        """
        if convert_to_jax:
            self.betas = jnp.array(betas)
            self.s_tensor = jnp.array(s_tensor)
            if s_tensor_freq is not None:
                self.s_tensor_freq = jnp.array(s_tensor_freq)
                self.freqs_hz = jnp.array(freqs_hz)
            else:
                self.s_tensor_freq = None
                self.freqs_hz = None
        else:
            self.betas = betas
            self.s_tensor = s_tensor
            self.s_tensor_freq = s_tensor_freq
            self.freqs_hz = freqs_hz
    
    @classmethod
    def from_neff(
        cls,
        freqs: np.ndarray,
        n_eff: np.ndarray,
        freq_disp: float,
        orders: int,
        s_tensor: Union[np.ndarray, jnp.ndarray],
        s_tensor_freq: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        convert_to_jax: bool = True,
        in_si_units: bool = False,
        num_modes: Optional[int] = None,
        beta_method: Literal["polynomial", "spline", "finite_difference"] = "polynomial",
        beta_fd_window: int = 5,
    ):
        """
        Create WaveguideLoader from effective index data.
        
        Args:
            freqs: Array of frequencies, shape (n_freqs,)
                If in_si_units=False: frequencies in THz
                If in_si_units=True: frequencies in Hz
            n_eff: Effective index array, shape (n_freqs, n_modes) or (n_freqs,)
            freq_disp: Center frequency for Taylor expansion (same units as freqs)
            orders: Number of dispersion orders to compute (including beta_0)
            s_tensor: S-tensor array, shape (num_modes, num_modes, num_modes, num_modes)
            s_tensor_freq: Frequency-dependent s_tensor array (optional),
                shape (n_freqs, num_modes, num_modes, num_modes, num_modes)
            convert_to_jax: If True, convert arrays to JAX arrays (default: True)
            in_si_units: If True, frequencies are in Hz (SI units) and will be converted to THz.
                If False, frequencies are assumed to be in THz (default: False)
            num_modes: Number of modes to use. If None, uses all available modes.
                If specified, slices n_eff, betas, and s_tensor to use only the first num_modes modes.
            beta_method: Method used to compute dispersion coefficients from n_eff.
                Supported values: "polynomial", "spline", "finite_difference".
            beta_fd_window: Half window size (in samples) used by the finite-difference method.
        
        Returns:
            WaveguideLoader instance
        """
        # Ensure n_eff is 2D
        freqs = np.asarray(freqs, dtype=np.float64)
        n_eff = np.asarray(n_eff)
        if n_eff.ndim == 1:
            n_eff = n_eff[:, np.newaxis]
        
        # Slice n_eff to requested number of modes if specified
        if num_modes is not None:
            if num_modes <= 0:
                raise ValueError(f"num_modes must be positive, got {num_modes}")
            if num_modes > n_eff.shape[1]:
                raise ValueError(
                    f"num_modes ({num_modes}) exceeds available modes ({n_eff.shape[1]})"
                )
            n_eff = n_eff[:, :num_modes]
        
        # Fit betas from n_eff
        betas = _compute_betas_from_neff(
            freqs,
            n_eff,
            freq_disp,
            orders,
            method=beta_method,
            in_si_units=in_si_units,
            fd_window=beta_fd_window,
        )
        
        # Slice s_tensor to match number of modes
        s_tensor = np.asarray(s_tensor)
        if num_modes is not None:
            if num_modes > s_tensor.shape[0]:
                raise ValueError(
                    f"num_modes ({num_modes}) exceeds s_tensor modes ({s_tensor.shape[0]})"
                )
            s_tensor = s_tensor[:num_modes, :num_modes, :num_modes, :num_modes]
        elif betas.shape[1] != s_tensor.shape[0]:
            # If num_modes not specified, check that betas and s_tensor match
            raise ValueError(
                f"Number of modes must match between betas and s_tensor. "
                f"betas modes: {betas.shape[1]}, s_tensor modes: {s_tensor.shape[0]}"
            )
        
        # Handle frequency-dependent s_tensor
        freq_grid_hz = freqs if in_si_units else freqs * 1e12

        if s_tensor_freq is not None:
            s_tensor_freq = np.asarray(s_tensor_freq)
            if s_tensor_freq.shape[0] != len(freqs):
                raise ValueError(
                    f"Frequency array length ({len(freqs)}) must match first dimension of s_tensor_freq ({s_tensor_freq.shape[0]})"
                )
            if num_modes is not None:
                s_tensor_freq = s_tensor_freq[:, :num_modes, :num_modes, :num_modes, :num_modes]
        
        # Create instance
        loader = cls.__new__(cls)
        if convert_to_jax:
            loader.betas = jnp.array(betas)
            loader.s_tensor = jnp.array(s_tensor) if not isinstance(s_tensor, jnp.ndarray) else s_tensor
            if s_tensor_freq is not None:
                loader.s_tensor_freq = jnp.array(s_tensor_freq) if not isinstance(s_tensor_freq, jnp.ndarray) else s_tensor_freq
            else:
                loader.s_tensor_freq = None
            loader.freqs_hz = jnp.array(freq_grid_hz)
        else:
            loader.betas = betas
            loader.s_tensor = s_tensor
            loader.s_tensor_freq = s_tensor_freq
            loader.freqs_hz = freq_grid_hz
        
        loader.num_orders = betas.shape[0]
        loader.num_modes = betas.shape[1]
        loader.betas_shape = betas.shape
        loader.s_tensor_shape = s_tensor.shape
        
        return loader


def load_waveguide_data(filepath: str, num_modes: int, num_betas: int):
    """
    Loads waveguide dispersion and nonlinear coefficient data from .mat files.

    Args:
        filepath (str): Path to the directory containing the data files.
        num_modes (int): The number of modes to load.
        num_betas (int): The number of beta coefficients to load.

    Returns:
        tuple: A tuple containing:
            - betas (jnp.ndarray): Taylor expansion coefficients of the propagation constant.
            - s_tensor (jnp.ndarray): Nonlinear coupling coefficients.
    """
    # Load beta coefficients
    betas_si = sio.loadmat(f"{filepath}/betas.mat")['betas'][:num_betas, :num_modes]
    
    # Convert betas from SI units (s^n/m) to ps^n/m
    # For order n: β[ps^n/m] = β[s^n/m] * (10^12)^n
    # Since 1 s = 10^12 ps, then 1 s^n = (10^12)^n ps^n
    betas = np.zeros_like(betas_si)
    for i in range(betas_si.shape[0]):
        betas[i, :] = betas_si[i, :] * (1e12)**i
    
    # Load nonlinear coupling tensor
    s_tensor_si = sio.loadmat(f"{filepath}/s_tensor_{num_modes}modes.mat")['s_tensor_all']
    
    # S-tensor is already in 1/m^2, no conversion needed
    s_tensor = s_tensor_si
    
    return jnp.array(betas), jnp.array(s_tensor)
