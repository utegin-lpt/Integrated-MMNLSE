import numpy as np
import matplotlib.pyplot as plt

class Pulse:
    """Discrete time-domain representation of a multimode optical pulse."""

    def __init__(self, t: np.ndarray, field_t: np.ndarray):
        if t.ndim != 1:
            raise ValueError("Time grid must be 1-D array")
        if field_t.shape[1] != t.size:
            raise ValueError("Field and time grid size mismatch")
        
        self.t = t.astype(np.float64)
        self.dt = float(self.t[1] - self.t[0])  # seconds
        self.field = field_t.astype(np.complex128, copy=False)
        self.n_points = self.t.size
        self.n_modes = self.field.shape[0]
        self.peak_power = np.max(np.abs(self.field) ** 2)

    @classmethod
    def gaussian(
        cls,
        peak_power_w: float,
        fwhm: float,
        time_window: float,
        n_modes: int,
        n_time_points: int,
        modal_coefficients: np.ndarray | None = None,
        center: float = 0.0,
        supergaussian_order: int = 1,
    ) -> "Pulse":
        if modal_coefficients is None:
            modal_coefficients = np.ones(n_modes)
        if len(modal_coefficients) != n_modes:
            raise ValueError("Length of modal_coefficients must match n_modes")
        
        modal_coefficients = np.asarray(modal_coefficients, dtype=np.complex128)
        modal_coefficients /= np.linalg.norm(modal_coefficients)

        dt = time_window / n_time_points
        times = (np.arange(-n_time_points // 2, n_time_points // 2) * dt).astype(np.float64)

        sigma = fwhm / 2.355  # FWHM ➜ σ for Gaussian intensity
        exponent = 2 * supergaussian_order
        envelope = np.sqrt(peak_power_w) * np.exp(-np.abs(times - center) ** exponent / (2 * sigma ** exponent))
        
        field = np.outer(modal_coefficients, envelope)

        return cls(times, field)
    
    @classmethod
    def secant(
        cls,
        peak_power_w: float,
        fwhm: float,
        time_window: float,
        n_modes: int,
        n_time_points: int,
        modal_coefficients: np.ndarray | None = None,
        center: float = 0.0,
        omega0: float = 2 * np.pi * 193.1e12,  # ~1550 nm in rad/s
    ) -> "Pulse":
        """Build a secant pulse in the time domain."""
        if modal_coefficients is None:
            modal_coefficients = np.ones(n_modes)
        if len(modal_coefficients) != n_modes:
            raise ValueError("Length of modal_coefficients must match n_modes")
        
        modal_coefficients = np.asarray(modal_coefficients, dtype=np.complex128)
        modal_coefficients /= np.linalg.norm(modal_coefficients)

        dt = time_window / n_time_points
        times = (np.arange(-n_time_points // 2, n_time_points // 2) * dt).astype(np.float64)

        envelope = np.sqrt(peak_power_w) * np.abs(np.tanh((times - center) / (fwhm / 2)))

        field = np.outer(modal_coefficients, envelope)

        return cls(times, field)

    def plot_pulse(self):
        """Plot the pulse in the time domain."""
        plt.figure(figsize=(10, 6))
        for i in range(self.n_modes):
            plt.plot(self.t, np.abs(self.field[i, :])**2, label=f'Mode {i}')
        plt.title('Pulse in Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity (W)')
        plt.legend()
        plt.grid()
        plt.show()
