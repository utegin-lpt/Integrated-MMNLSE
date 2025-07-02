import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax
# Set jax to use 64-bit floating point numbers
jax.config.update("jax_enable_x64", True)

class Pulse:
    """Discrete time-domain representation of a multimode optical pulse."""
    def __init__(self, t: jnp.ndarray, field_t: jnp.ndarray):
        if t.ndim != 1:
            raise ValueError("Time grid must be 1-D array")
        if field_t.shape[1] != t.size:
            raise ValueError("Field and time grid size mismatch")

        self.t = t.astype(jnp.float64)
        self.dt = (self.t[1] - self.t[0])  # seconds
        self.field = field_t.astype(jnp.complex128, copy=False)
        self.n_points = self.t.size
        self.n_modes = self.field.shape[0]
        self.peak_power = jnp.max(jnp.abs(self.field) ** 2)

    @classmethod
    def gaussian(
        cls,
        peak_power_w: float,
        fwhm: float,
        time_window: float,
        n_modes: int,
        n_time_points: int,
        modal_coefficients: jnp.ndarray | None = None,
        center: float = 0.0,
        supergaussian_order: int = 1,
    ) -> "Pulse":
        if modal_coefficients is None:
            modal_coefficients = jnp.ones(n_modes)
        if len(modal_coefficients) != n_modes:
            raise ValueError("Length of modal_coefficients must match n_modes")

        modal_coefficients = jnp.asarray(modal_coefficients, dtype=jnp.complex128)
        modal_coefficients /= jnp.linalg.norm(modal_coefficients)

        dt = time_window / n_time_points
        times = (jnp.arange(-n_time_points // 2, n_time_points // 2) * dt).astype(jnp.float64)

        sigma = fwhm / 2.355  # FWHM → σ for Gaussian intensity
        exponent = 2 * supergaussian_order
        envelope = jnp.sqrt(peak_power_w) * jnp.exp(
            -(times - center) ** exponent / (2 * sigma ** exponent)
        )

        field = jnp.outer(modal_coefficients, envelope)
        return cls(times, field)

    @classmethod
    def secant(
        cls,
        peak_power_w: float,
        fwhm: float,
        time_window: float,
        n_modes: int,
        n_time_points: int,
        modal_coefficients: jnp.ndarray | None = None,
        center: float = 0.0,
        omega0: float = 2 * jnp.pi * 193.1e12,  # ~1550 nm in rad/s
    ) -> "Pulse":
        """Build a secant pulse in the time domain."""
        if modal_coefficients is None:
            modal_coefficients = jnp.ones(n_modes)
        if len(modal_coefficients) != n_modes:
            raise ValueError("Length of modal_coefficients must match n_modes")

        modal_coefficients = jnp.asarray(modal_coefficients, dtype=jnp.complex128)
        modal_coefficients /= jnp.linalg.norm(modal_coefficients)

        dt = time_window / n_time_points
        times = (jnp.arange(-n_time_points // 2, n_time_points // 2) * dt).astype(jnp.float64)

        envelope = jnp.sqrt(peak_power_w) / (jnp.cosh((times - center) / (fwhm / 2)) ** 2)

        field = jnp.outer(modal_coefficients, envelope)
        return cls(times, field)

    def __getitem__(self, idx):
        """Allow indexing by mode: pulse[idx] → Pulse of that mode (or modes if slice/list)."""
        sub_field = jnp.take(self.field, idx, axis=0)
        if sub_field.ndim == 1:
            sub_field = sub_field[jnp.newaxis, :]
        return self.__class__(self.t, sub_field)

    def plot_pulse(self):
        """Plot the pulse in the time domain."""
        plt.figure(figsize=(10, 6))
        for i in range(self.n_modes):
            plt.plot(self.t, jnp.abs(self.field[i, :])**2, label=f"Mode {i}")
        plt.title("Pulse in Time Domain")
        plt.xlabel("Time (s)")
        plt.ylabel("Intensity (W)")
        plt.legend()
        plt.grid()
        plt.show()

    def get_intensity(self):
        """Compute the intensity of the pulse."""
        I = jnp.abs(self.field) ** 2
        if I.shape[0] == 1:
            return I[0, :]
        return I

    def get_normalized_intensity(self):
        """Compute the normalized intensity of the pulse."""
        I = self.get_intensity()
        return I / jnp.max(I)

    def get_log_intensity(self):
        """Compute the log intensity of the pulse."""
        return jnp.log10(self.get_intensity())

    def get_log_normalized_intensity(self):
        """Compute the log-normalized intensity of the pulse."""
        I = self.get_intensity()
        return jnp.log10(I / jnp.max(I))

    def get_sum_intensity(self):
        """Compute the sum of the intensity across all modes."""
        return jnp.sum(self.get_intensity(), axis=0)

    def get_sum_log_intensity(self):
        """Compute the log intensity of the sum of the pulse."""
        return jnp.log10(self.get_sum_intensity())

    def get_spectrum(self):
        """Compute the spectrum of the pulse."""
        F = jnp.fft.fftshift(jnp.fft.fft(self.field, axis=-1), axes=-1)
        S = jnp.abs(F) ** 2
        if S.shape[0] == 1:
            return S[0, :]
        return S

    def get_normalized_spectrum(self):
        """Compute the normalized spectrum of the pulse."""
        S = self.get_spectrum()
        return S / jnp.max(S)

    def get_log_spectrum(self):
        """Compute the log spectrum of the pulse."""
        return 10*jnp.log10(self.get_spectrum())

    def get_log_normalized_spectrum(self):
        """Compute the log-normalized spectrum of the pulse."""
        S = self.get_spectrum()
        return 10*jnp.log10(S / jnp.max(S))

    def get_sum_spectrum(self):
        """Compute the sum of the spectrum across all modes."""
        return jnp.sum(self.get_spectrum(), axis=0)

    def get_sum_log_spectrum(self):
        """Compute the log spectrum of the sum of the pulse."""
        return 10*jnp.log10(self.get_sum_spectrum())

    def get_sum_log_normalized_spectrum(self):
        """Compute the log-normalized spectrum of the sum of the pulse."""
        S = self.get_sum_spectrum()
        return 10*jnp.log10(S / jnp.max(S))

