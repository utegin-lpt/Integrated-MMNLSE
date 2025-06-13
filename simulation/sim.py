import numpy as np
import os
from constants import *
from utils import *
from propagator import *
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from .plot      import Plotter
from .pulse     import Pulse
from .waveguide import Waveguide

class Simulation:
    def __init__(self, waveguide: Waveguide, pulse: Pulse, central_wavelength: float):
        self.waveguide = waveguide
        self.pulse = pulse
        self.num_points = pulse.n_points
        self.num_modes = pulse.n_modes

        self.dt = pulse.dt
        self.t = pulse.t
        self.fs = 1 / (pulse.n_points * self.dt)  
        self.omega0 = 2 * np.pi * C / central_wavelength
        self.freq = C / central_wavelength + self.fs * np.linspace(-self.num_points/2, self.num_points/2, num=self.num_points)  # Frequency array (Hz)
        self.wavelength = C / self.freq 
        self.omegas = 2 * np.pi * self.freq - self.omega0  # Angular frequency array (rad/s)
        print(f"Min wavelength: {self.wavelength.min() * 1e9:.2f} nm")
        print(f"Max wavelength: {self.wavelength.max() * 1e9:.2f} nm")
        self.lambda_cut = 1e-9
        self.lpf_mask = (self.wavelength >= self.lambda_cut)

        self.time_boundary = super_gauss(self.t, p=60)

    def run(self, field: np.ndarray, length: float, dz: float, save_path = None, num_plots: int = 30, plot_propagation: int = 0):

        self.nonzero_idx_plmn, self.nonzero_idx_mn, self.SR = self.waveguide.get_etas()
        self.DFR = self.waveguide.get_dispersion(self.omegas)
        self.gamma = self.waveguide.n2 * self.omega0 / C  # W^-1 m
        self.nonlinear_coeff = self.gamma
        self.self_steepening = 1 + self.omegas/ self.omega0
        
        self.num_steps = int(np.ceil(length / dz))
        print("Running simulation...")
        plotter = Plotter(self.t, self.wavelength)
        if plot_propagation > 0:
            time_propagation = []
            spectral_propagation = []
        # Check if the save_path exists, if not create it
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        fig_path = os.path.join(save_path, "figures")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        print(f"Figures will be saved in {fig_path}")
        no_lim_fig_path = os.path.join(save_path, "figures/no_lim")
        if not os.path.exists(no_lim_fig_path):
            os.makedirs(no_lim_fig_path)
        tqdm_bar = tqdm(range(self.num_steps), desc="Simulation Progress", unit="steps")
        
        # boundary = square_window(self.pulse.t, 8)
        # Check energy conservation
        input_energy = np.sum(np.abs(field) ** 2) * self.dt
        propagated_length = 0
        saved_length_list = []
        
        plotter.plot_pulses(field, os.path.join(fig_path, f"pulse_{int(propagated_length)}_um"), show=False, log_scale=True)
        plotter.plot_spectra(field, os.path.join(fig_path, f"spectrum_{int(propagated_length)}_um"), show=False)
        plotter.plot_spectra_no_lim(field, os.path.join(no_lim_fig_path, f"spectrum_{int(propagated_length)}_um"), show=False)
        plotter.plot_individual_pulses(field, os.path.join(fig_path, f"individual_pulse_{int(propagated_length)}_um"), show=False)
        plotter.plot_individual_spectra(field, os.path.join(fig_path, f"individual_spectrum_{int(propagated_length)}_um"), show=False)
        plotter.plot_individual_spectra_no_lim(field, os.path.join(no_lim_fig_path, f"individual_spectrum_{int(propagated_length)}_um"), show=False)
        plotter.plot_total_pulse(field, os.path.join(fig_path, f"total_pulse_{int(propagated_length)}_um"), show=False)
        plotter.plot_total_spectrum(field, os.path.join(fig_path, f"total_spectrum_{int(propagated_length)}_um"), show=False)
        plotter.plot_total_spectrum_no_lim(field, os.path.join(no_lim_fig_path, f"total_spectrum_{int(propagated_length)}_um"), show=False)
        
        for i in tqdm_bar:
            field = RK4IP(field, self.DFR, self.nonlinear_coeff, self.nonzero_idx_plmn, self.SR, dz, freq_mask=self.lpf_mask, temporal_mask=self.time_boundary)#, self_steepening=self.self_steepening)
            propagated_length = (i + 1) * dz * 1e6  # Convert to micrometers
            tqdm_bar.set_description(f"Simulation Progress: {propagated_length:.2f} um/{length * 1e6:.2f} um")
            
            if save_path:
                
                if i == 0 or (i + 1) % (self.num_steps // num_plots) == 0:
                    plotter.plot_pulses(field, os.path.join(fig_path, f"pulse_{int(propagated_length)}_um"), show=False, log_scale=True)
                    plotter.plot_spectra(field, os.path.join(fig_path, f"spectrum_{int(propagated_length)}_um"), show=False)
                    plotter.plot_spectra_no_lim(field, os.path.join(no_lim_fig_path, f"spectrum_{int(propagated_length)}_um"), show=False)
                    plotter.plot_individual_pulses(field, os.path.join(fig_path, f"individual_pulse_{int(propagated_length)}_um"), show=False)
                    plotter.plot_individual_spectra(field, os.path.join(fig_path, f"individual_spectrum_{int(propagated_length)}_um"), show=False)
                    plotter.plot_individual_spectra_no_lim(field, os.path.join(no_lim_fig_path, f"individual_spectrum_{int(propagated_length)}_um"), show=False)
                    plotter.plot_total_pulse(field, os.path.join(fig_path, f"total_pulse_{int(propagated_length)}_um"), show=False)
                    plotter.plot_total_spectrum(field, os.path.join(fig_path, f"total_spectrum_{int(propagated_length)}_um"), show=False)
                    plotter.plot_total_spectrum_no_lim(field, os.path.join(no_lim_fig_path, f"total_spectrum_{int(propagated_length)}_um"), show=False)
                
                if plot_propagation > 0 and (i + 1) % (self.num_steps // plot_propagation) == 0:
                    time_propagation.append((np.abs(field)**2).sum(axis=0))
                    spectral_propagation.append((np.abs(np.fft.fftshift(np.fft.fft(field, axis=-1), axes=-1))**2).sum(axis=0))
                    saved_length_list.append(propagated_length*1e-3)  # Convert to mm for saving
        
        if plot_propagation > 0:
            time_propagation = np.array(time_propagation)
            spectral_propagation = np.array(spectral_propagation)
            # save the propagation fields in time and frequency domain
            plotter.plot_propagation(time_propagation, spectral_propagation, os.path.join(save_path, "propagation"), saved_length_list, show=False)
        output_energy = np.sum(np.abs(field) ** 2) * self.dt
        # Check energy conservation
        print(f"Relative energy error: {np.abs((output_energy - input_energy) / input_energy) * 100:.8f}%")
        np.save(os.path.join(save_path, "output_field.npy"), field)
        print("Simulation completed.")

        return field
    
