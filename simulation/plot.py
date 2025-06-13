import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, t: np.ndarray, wavelengths: np.ndarray):
        self.t = t * 1e12
        self.wavelengths = wavelengths * 1e9
        self.min_wavelength = 250
        self.min_time = -7
        self.max_time = 5
        self.max_wavelength = 4000
        # self.start_wavelength_idx = np.argmin(np.abs(self.wavelengths - self.max_wavelength))
        # self.end_wavelength_idx = np.argmin(np.abs(self.wavelengths - self.min_wavelength))
        # self.wavelengths = self.wavelengths[self.start_wavelength_idx:self.end_wavelength_idx][::-1]
        # print(self.wavelengths)
        # print(self.start_wavelength_idx, self.end_wavelength_idx)
        
    def plot_pulses(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = False):
        """Plot the pulse in the time domain."""
        num_modes = field.shape[0]
        pulses = []
        plt.figure(figsize=(10, 6))
        for i in range(num_modes):
            if log_scale:
                normalized_field = np.abs(field[i, :])**2 #/ np.max(np.abs(field[i, :])**2)
                plt.plot(self.t, 10 * np.log10(normalized_field), label=f'Mode {i}')
            else:
                plt.plot(self.t, np.abs(field[i, :])**2, label=f'Mode {i}')
            pulses.append(np.abs(field[i, :])**2)
        plt.title('Pulse in Time Domain')
        # plt.xlim(self.min_time, self.max_time)
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.grid()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
        np.savetxt(save_path + ".csv", np.array(pulses).T, delimiter=",")
    
    def plot_spectra(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = True):
        """Plot the spectrum of the pulse."""
        num_modes = field.shape[0]
        plt.figure(figsize=(10, 6))
        for i in range(num_modes):
            spectrum = np.fft.fftshift(np.fft.fft(field[i, :]))
            spectrum = np.abs(spectrum)**2
            if log_scale:
                spectrum = 10 * np.log10(spectrum / np.max(spectrum))
            plt.plot(self.wavelengths, spectrum, label=f'Mode {i}')
        
        plt.title('Spectrum of Pulse')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (dB)')
        plt.xlim(self.min_wavelength, self.max_wavelength)
        plt.legend()
        plt.grid()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
        np.savetxt(save_path + ".csv", spectrum, delimiter=",")

    def plot_spectra_no_lim(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = True):
        """Plot the spectrum of the pulse."""
        num_modes = field.shape[0]
        plt.figure(figsize=(10, 6))
        for i in range(num_modes):
            spectrum = np.fft.fftshift(np.fft.fft(field[i, :]))
            spectrum = np.abs(spectrum)**2
            if log_scale:
                spectrum = 10 * np.log10(spectrum / np.max(spectrum))
            plt.plot(self.wavelengths, spectrum, label=f'Mode {i}')

        plt.title('Spectrum of Pulse')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (dB)')
        # plt.xlim(self.min_wavelength, self.max_wavelength)
        plt.legend()
        plt.grid()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
    
    # Plot individual pulses in subfigures
    def plot_individual_pulses(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = False):
        """Plot the pulse in the time domain."""
        num_modes = field.shape[0]
        fig, axs = plt.subplots(num_modes, 1, figsize=(10, 3 * num_modes))
        pulses = []
        for i in range(num_modes):
            pulses.append(np.abs(field[i, :])**2)
            if log_scale:
                axs[i].plot(self.t, 10 * np.log10(np.abs(field[i, :])**2), label=f'Mode {i}')
            else:
                axs[i].plot(self.t, np.abs(field[i, :])**2, label=f'Mode {i}')
            axs[i].set_title(f'Mode {i}')
            axs[i].set_xlim(self.min_time, self.max_time)
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Intensity (a.u.)')
            axs[i].grid()
        plt.suptitle('Individual Pulses in Time Domain', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
        np.savetxt(save_path + ".csv", np.array(pulses).T, delimiter=",")
    
    # Plot individual spectrums in subfigures
    def plot_individual_spectra(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = True):
        """Plot the spectrum of the pulse."""
        num_modes = field.shape[0]
        fig, axs = plt.subplots(num_modes, 1, figsize=(5, 3 * num_modes))
        spectra = []
        for i in range(num_modes):
            spectrum = np.fft.fftshift(np.fft.fft(field[i, :]))
            spectrum = np.abs(spectrum)**2
            spectra.append(spectrum)
            if log_scale:
                spectrum = 10 * np.log10(spectrum / np.max(spectrum))
            axs[i].plot(self.wavelengths, spectrum)
            axs[i].set_title(f'Mode {i}')
            axs[i].set_xlabel('Wavelength (nm)')
            axs[i].set_ylabel('Intensity (dB)')
            axs[i].grid()
            axs[i].set_xlim(self.min_wavelength, self.max_wavelength)
        plt.tight_layout()
        plt.suptitle('Individual Spectra of Pulse', fontsize=16, y=1.1)
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
        np.savetxt(save_path+'.csv', np.array(spectra).T, delimiter=",")
    
    def plot_individual_spectra_no_lim(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = True):
        """Plot the spectrum of the pulse."""
        num_modes = field.shape[0]
        fig, axs = plt.subplots(num_modes, 1, figsize=(5, 3 * num_modes))
        for i in range(num_modes):
            spectrum = np.fft.fftshift(np.fft.fft(field[i, :]))
            spectrum = np.abs(spectrum)**2
            if log_scale:
                spectrum = 10 * np.log10(spectrum / np.max(spectrum))
            axs[i].plot(self.wavelengths, spectrum)
            axs[i].set_title(f'Mode {i}')
            axs[i].set_xlabel('Wavelength (nm)')
            axs[i].set_ylabel('Intensity (dB)')
            axs[i].grid()
            # axs[i].set_xlim(self.min_wavelength, self.max_wavelength)
        plt.tight_layout()
        plt.suptitle('Individual Spectra of Pulse', fontsize=16, y=1.1)
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()


    def plot_total_pulse(self, field: np.ndarray, save_path, show: bool = False):
        """Plot the total pulse in the time domain."""
        total_field = np.abs(field)**2
        total_field = np.sum(total_field, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(self.t, total_field)
        plt.xlim(self.min_time, self.max_time)
        plt.title('Total Pulse in Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity (a.u.)')
        plt.grid()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
        np.savetxt(save_path + ".csv", np.abs(total_field)**2, delimiter=",")
    
    def plot_total_spectrum(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = True):
        """Plot the total spectrum of the pulse."""
        total_spectrum = np.fft.fftshift(np.fft.fft(field, axis=-1), axes=-1)
        total_spectrum = np.abs(total_spectrum)**2
        total_spectrum = np.sum(total_spectrum, axis=0)
        if log_scale:
            total_spectrum = 10 * np.log10(total_spectrum / np.max(total_spectrum))
        else:
            total_spectrum = total_spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(self.wavelengths, total_spectrum)
        plt.title('Total Spectrum of Pulse')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (dB)')
        plt.xlim(self.min_wavelength, self.max_wavelength)
        plt.grid()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()
        np.savetxt(save_path + ".csv", total_spectrum, delimiter=",")

    def plot_total_spectrum_no_lim(self, field: np.ndarray, save_path, show: bool = False, log_scale: bool = True):
        """Plot the total spectrum of the pulse."""
        total_field = np.sum(field, axis=0)
        total_spectrum = np.fft.fftshift(np.fft.fft(total_field))
        total_spectrum = np.abs(total_spectrum)**2
        if log_scale:
            total_spectrum = 10 * np.log10(total_spectrum / np.max(total_spectrum))
        else:
            total_spectrum = total_spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(self.wavelengths, total_spectrum)
        plt.title('Total Spectrum of Pulse')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (dB)')
        plt.grid()
        plt.savefig(save_path + ".png")
        if show:
            plt.show()
        plt.close()

    def plot_propagation(self, time_fields: np.ndarray, spectral_fields: np.ndarray, save_path, saved_length_list, show: bool = False):
        """ Plot the propagation of the pulse in time and frequency domain using pcolormesh 
        Parameters:
        -----------
        fields : np.ndarray
            A 2D array of shape (propagation_steps, n_points) containing the field values at each propagation step.
        save_path : str
            Path to save the figure.
        saved_length_list : list
            List of lengths at which the fields were saved.
        show : bool
            Whether to show the figure or not.
        """ 

        min_time_idx = np.argmin(np.abs(self.t - self.min_time))
        max_time_idx = np.argmin(np.abs(self.t - self.max_time))
        min_wavelength_idx = np.argmin(np.abs(self.wavelengths - self.min_wavelength))
        max_wavelength_idx = np.argmin(np.abs(self.wavelengths - self.max_wavelength))
        order = np.argsort(self.wavelengths)     # indices that make λ monotone ↑
        lambda_axis_sorted = self.wavelengths[order]  
        
        field_plot = time_fields / np.max(time_fields)
        field_plot = 10 * np.log10(field_plot)
        field_plot = field_plot[:, min_time_idx:max_time_idx]
        fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        time_fig = ax[0].pcolormesh(self.t[min_time_idx:max_time_idx], saved_length_list, field_plot, shading='auto', vmin=-50, cmap = 'inferno')
        fig.colorbar(time_fig, ax=ax[0], label='Normalized Intensity (dB)', pad=0.025)

        # ax[0].set_xlim(self.min_time, self.max_time)
        np.savetxt((save_path+ "_time.csv"), field_plot, delimiter=",")
        np.savetxt((save_path+ "_time_full_t.csv"), self.t, delimiter=",")
        np.savetxt((save_path+ "_time_t.csv"), self.t[min_time_idx:max_time_idx], delimiter=",")
        ax[0].set_title('Propagation in Time Domain')
        ax[0].set_ylabel('Propagation Distance (mm)')
        ax[0].set_xlabel('Time (ps)')

        spectrum = spectral_fields
        spectrum = 10 * np.log10(spectrum / np.max(spectrum))
        spectrum   = spectrum[:, order]       # reorder columns to match
        np.savetxt(save_path+"_spectrum.csv", spectrum, delimiter=",")
        np.savetxt(save_path+"_spectrum_full_wavelength.csv", self.wavelengths, delimiter=",")
        np.savetxt(save_path+"_spectrum_wavelength.csv", lambda_axis_sorted, delimiter=",")

        spect_fig = ax[1].pcolormesh(lambda_axis_sorted, saved_length_list, spectrum, shading='auto', vmin=-50, vmax=0, cmap = 'inferno')
        fig.colorbar(spect_fig, ax=ax[1], label='Intensity (dB)', pad=0.025)
        ax[1].set_title('Propagation in Frequency Domain')
        ax[1].set_ylabel('Propagation Distance (mm)')
        ax[1].set_xlabel('Wavelength (nm)')
        ax[1].set_xlim(self.min_wavelength, self.max_wavelength)
        plt.tight_layout()
        plt.savefig(save_path + ".png")
        print(f"Propagation plot saved to {save_path}.png")
        if show:
            plt.show()
        plt.close()
