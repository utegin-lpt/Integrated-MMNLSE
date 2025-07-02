import jax.numpy as jnp
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from pulse     import Pulse
from waveguide import Waveguide
from constants import *
from propagator import *

import jax
from jax._src.ad_checkpoint import remat
# Set jax to use 64-bit floating point numbers
jax.config.update("jax_enable_x64", True)

class Simulation:
    def __init__(self, waveguide: Waveguide, pulse: Pulse, central_wavelength: float):
        self.waveguide = waveguide
        self.pulse = pulse
        self.num_points = pulse.n_points
        self.num_modes = pulse.n_modes
        self.dt = pulse.dt
        self.t = pulse.t
        self.omega0 = 2 * jnp.pi * C / central_wavelength
        self.freq = C / central_wavelength + jnp.fft.fftshift(jnp.fft.fftfreq(self.num_points, d=self.dt))  # Frequency array (Hz)
        self.wavelength = C / self.freq
        self.omegas = 2 * jnp.pi * self.freq - self.omega0  # Angular frequency array (rad/s)
        self.lambda_cut = 1e-9
        self.lpf_mask = (self.wavelength >= self.lambda_cut)
        self.time_boundary = super_gauss(self.t, p=60)
        self.nonzero_idx_plmn, self.nonzero_idx_mn, self.SR = self.waveguide.get_etas()
        self.DFR = self.waveguide.get_dispersion(self.omegas)
        self.gamma = self.waveguide.n2 * self.omega0 / C  
        self.nonlinear_coeff = self.gamma
        self.self_steepening = 1 + self.omegas / self.omega0

    def run(self, length: float, dz: float, num_steps: int | None, scan: bool = False):
        """
        Run the simulation for a given length and step size.
        :param length: Length of the waveguide to simulate (in meters).
        :param dz: Step size for the simulation (in meters).
        :param scan: If True, use lax.scan for the simulation. 
                    lax.scan is more efficient for large number of steps, especially in the order if tens of thousands, 
                    but for small number of steps, it is better to use a for loop.
        :return: The propagated pulse.
        """

        if num_steps is None:
            # Calculate the number of steps based on the length and dz
            num_steps = int(math.ceil(length / dz))
            self.num_steps = num_steps
        else:
            self.num_steps = num_steps
        if scan:
            # use lax.scan to propagate in a single JIT’d pass
            def step(carry, _):
                new_field = RK4IP(
                    carry,
                    self.DFR,
                    self.nonlinear_coeff,
                    self.nonzero_idx_plmn,
                    self.SR,
                    dz,
                    freq_mask=self.lpf_mask,
                    temporal_mask=self.time_boundary,
                )
                return new_field, None

            # scan over the number of steps
            initial_carry = self.pulse.field
            self.pulse.field, _ = jax.lax.scan(
                step,
                initial_carry,
                None,
                length=self.num_steps,
            )
        else:
            for i in range(self.num_steps):
                self.pulse.field = RK4IP(self.pulse.field, self.DFR, self.nonlinear_coeff, self.nonzero_idx_plmn, self.SR, dz, freq_mask=self.lpf_mask, temporal_mask=self.time_boundary)#, self_steepening=self.self_steepening)
        return self.pulse
    
if __name__ == "__main__":
    import scipy.io as sio
    # Load the waveguide parameters
    material = "SiN"
    order = 5  # Taylor series order
    betas = jnp.array(sio.loadmat("betas.mat")["betas"])
    n = jnp.arange(0, order + 1, 1).reshape(-1, 1)
    unit_conversion = 1e-15 ** n  * 1e3
    betas = betas[:order+1, ] * unit_conversion # convert to sᵏ m⁻¹
    etas = jnp.array(sio.loadmat("etas.mat")["SR"])
    waveguide = Waveguide(material, betas, etas)

    # Create a pulse
    peak_power_w = 25e3  # 25 kW
    fwhm = 250e-15  # 250 fs
    time_window = 20e-12  # 20 ps
    n_modes = 6
    n_time_points = 2**13
    modal_coefficients = jnp.array([1.0, 0.1, 0.5, 0.40, 0.7, 0.2])  # Example coefficients for the modes
    modal_coefficients = jax.nn.softplus(modal_coefficients)  # Ensure coefficients are positive
    modal_coefficients = modal_coefficients / jnp.linalg.norm(modal_coefficients)  # Normalize to sum to 1
    print(f"Modal coefficients: {modal_coefficients}")
    # Generate the pulse
    pulse = Pulse.gaussian(peak_power_w=peak_power_w, fwhm=fwhm, time_window=time_window, n_modes=n_modes, n_time_points=n_time_points, modal_coefficients=modal_coefficients)

    # Initialize the simulation
    central_wavelength = 1550e-9  # 1550 nm
    sim = Simulation(waveguide, pulse, central_wavelength)
    # Run the simulation
    length = 3e-3  # 3 mm
    dz = 1e-6  # 1 um
    num_steps = int(jax.lax.ceil(length / dz))  # number of steps in the simulation
    pulse           = sim.run(length, dz, num_steps=num_steps, scan=True)
    intensities     = pulse.get_intensity()
    print(f"Energy per mode: {jnp.sum(intensities, axis=-1)} W")
    # Plot the energy per mode as a histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_modes + 1), jnp.sum(intensities, axis=-1), color='blue', alpha=0.7)
    plt.title('Energy per Mode at Output')
    plt.xlabel('Mode Number')
    plt.ylabel('Energy (W)')
    plt.xticks(range(1, n_modes + 1))
    # plt.grid()
    plt.show()
    spectra         = pulse.get_log_normalized_spectrum()

    # sort wavelength and spectra together
    sort_idx        = jnp.argsort(sim.wavelength)         # indices that sort λ ascending
    wl_sorted       = sim.wavelength[sort_idx]            # sorted wavelengths
    spectra_sorted  = spectra[:, sort_idx]                # same re‐ordering on spectra

    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # for i, intensity in enumerate(intensities):
    #     plt.plot(sim.t * 1e12, intensity, label=f'Mode {i+1}')
    # plt.title('Pulse Intensity in Time Domain')
    # plt.xlabel('Time (ps)')
    # plt.ylabel('Intensity (W)')
    # plt.ylim(bottom=-50)
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.figure(figsize=(10, 6))

    # for i, spectrum in enumerate(spectra):
    #     plt.plot(sim.wavelength * 1e9, spectrum, label=f'Mode {i+1}')
    # plt.title('Pulse Log Normalized Spectrum')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Log Normalized Spectrum')
    # plt.xlim(800, 2500)
    # plt.ylim(bottom=-100)
    # plt.legend()
    # plt.grid()
    # plt.show()