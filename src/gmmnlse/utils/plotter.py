"""
Plotting utilities for GMMNLSE solver results.

Provides a Plotter class for visualizing propagation results, including
waterfall plots, input/output comparisons, and spectral evolution.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import jax.numpy as jnp
from typing import Dict, Optional, List, Tuple, Union, Any, cast
from ..pulse import Pulse
from ..constants import C_um_ps


class Plotter:
    """Plotting utilities for GMMNLSE solver results.
    
    Provides configurable plotting with consistent styling for common
    visualization tasks in optical pulse propagation simulations.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6)):
        """Initialize Plotter with style configuration.
        
        Args:
            style: Matplotlib style ('default', 'publication', 'seaborn', etc.)
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style based on selected style."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 2,
                'grid.alpha': 0.3,
            })
        elif self.style == 'seaborn':
            try:
                plt.style.use('seaborn-v0_8')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    pass  # Fall back to default if seaborn not available
    
    def _get_colorbar_label(self, plot_type: str) -> str:
        """Get colorbar label based on plot type."""
        labels = {
            'intensity': 'Intensity (W)',
            'phase': 'Phase (rad)',
            'field_real': 'Real Field (√W)',
            'field_imag': 'Imaginary Field (√W)',
        }
        return labels.get(plot_type, '')
    
    def plot_propagation(self, solution: Dict, mode_idx: int = 0,
                        plot_type: str = 'intensity',
                        figsize: Optional[Tuple[int, int]] = None,
                        ax: Optional[Axes] = None,
                        cmap: str = 'viridis',
                        **kwargs) -> Figure:
        """Waterfall plot showing pulse evolution along z.
        
        Args:
            solution: Solution dict from solver.solve() with keys:
                'ts': z-positions (m), 'ys': field array, 'pulses': list of Pulse objects
            mode_idx: Which mode to plot (0-indexed)
            plot_type: 'intensity', 'phase', 'field_real', 'field_imag'
            figsize: Figure size (width, height). If None, uses default.
            ax: Optional axes to plot on. If None, creates new figure.
            cmap: Colormap for waterfall plot
            **kwargs: Additional arguments passed to pcolormesh
            
        Returns:
            matplotlib Figure object
        """
        pulses = solution['pulses']
        if mode_idx >= pulses[0].n_modes:
            raise ValueError(f"mode_idx={mode_idx} is out of bounds for n_modes={pulses[0].n_modes}")
        
        if plot_type not in ['intensity', 'phase', 'field_real', 'field_imag']:
            raise ValueError(f"plot_type must be 'intensity', 'phase', 'field_real', or 'field_imag'")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        z_points = solution['ts']
        
        # Get time axis from first pulse
        t_ps = np.array(pulses[0].t)
        
        # Extract data based on plot_type
        data_list = []
        for pulse in pulses:
            if plot_type == 'intensity':
                data_list.append(np.array(pulse.get_intensity()[mode_idx, :]))
            elif plot_type == 'phase':
                field = pulse.field[mode_idx, :]
                data_list.append(np.array(jnp.angle(field)))
            elif plot_type == 'field_real':
                data_list.append(np.array(jnp.real(pulse.field[mode_idx, :])))
            elif plot_type == 'field_imag':
                data_list.append(np.array(jnp.imag(pulse.field[mode_idx, :])))
        
        data = np.array(data_list)  # Shape: (n_z, n_points)
        
        # Create meshgrid for pcolormesh
        T, Z = np.meshgrid(t_ps, z_points)
        
        # Convert z to micrometers for display
        Z_um = Z * 1e6
        
        # Create waterfall plot
        im = ax.pcolormesh(T, Z_um, data, cmap=cmap, shading='gouraud', **kwargs)
        
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Propagation Distance (μm)')
        ax.set_title(f'Mode {mode_idx} - {plot_type.title()} Evolution')
        plt.colorbar(im, ax=ax, label=self._get_colorbar_label(plot_type))
        
        return fig
    
    def plot_total_input_output(self, solution: Dict, 
                                sum_type: str = 'coherent',
                                figsize: Optional[Tuple[int, int]] = None,
                                ax: Optional[Axes] = None,
                                log_scale: bool = False,
                                **kwargs) -> Figure:
        """Plot total input vs total output intensity.
        
        Args:
            solution: Solution dict from solver.solve()
            sum_type: 'coherent' or 'incoherent' for mode summation
            figsize: Figure size. If None, uses default.
            ax: Optional axes to plot on
            log_scale: If True, plot in dB (log scale). If False, linear scale.
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib Figure object
        """
        if sum_type not in ['coherent', 'incoherent']:
            raise ValueError("sum_type must be 'coherent' or 'incoherent'")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        pulses = solution['pulses']
        input_pulse = pulses[0]
        output_pulse = pulses[-1]
        
        t_ps = np.array(input_pulse.t)
        
        if log_scale:
            input_intensity = np.array(input_pulse.get_total_log_intensity(sum_type=sum_type))
            output_intensity = np.array(output_pulse.get_total_log_intensity(sum_type=sum_type))
            ylabel = 'Intensity (dB)'
        else:
            input_intensity = np.array(input_pulse.get_total_intensity(sum_type=sum_type))
            output_intensity = np.array(output_pulse.get_total_intensity(sum_type=sum_type))
            ylabel = 'Intensity (W)'
        
        ax.plot(t_ps, input_intensity, 'r--', label='Input', linewidth=2, **kwargs)
        ax.plot(t_ps, output_intensity, 'b-', label='Output', linewidth=2, **kwargs)
        
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Total Input vs Output ({sum_type.title()} Sum)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_input_output_per_mode(self, solution: Dict,
                                   n_modes: Optional[int] = None,
                                   figsize: Optional[Tuple[int, int]] = None,
                                   **kwargs) -> Figure:
        """Plot input vs output intensity for each mode.
        
        Args:
            solution: Solution dict from solver.solve()
            n_modes: Number of modes to plot. If None, plots all modes.
            figsize: Figure size. If None, uses default.
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib Figure object
        """
        pulses = solution['pulses']
        input_pulse = pulses[0]
        output_pulse = pulses[-1]
        
        n_modes_to_plot = n_modes or input_pulse.n_modes
        n_modes_to_plot = min(n_modes_to_plot, input_pulse.n_modes)
        
        # Create subplots grid
        n_cols = min(3, n_modes_to_plot)
        n_rows = (n_modes_to_plot + n_cols - 1) // n_cols
        
        figsize_actual = figsize or (15, 4*n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_actual)
        if n_modes_to_plot == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_modes_to_plot > 1 else [axes]
        else:
            axes = axes.flatten()
        
        t_ps = np.array(input_pulse.t)
        
        for i in range(n_modes_to_plot):
            ax = axes[i]
            
            input_intensity = np.array(input_pulse.get_intensity()[i, :])
            output_intensity = np.array(output_pulse.get_intensity()[i, :])
            
            ax.plot(t_ps, input_intensity, 'r--', label='Input', linewidth=2, **kwargs)
            ax.plot(t_ps, output_intensity, 'b-', label='Output', linewidth=2, **kwargs)
            
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Intensity (W)')
            ax.set_title(f'Mode {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_modes_to_plot, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle('Input vs Output per Mode', fontsize=14, y=0.995)
        plt.tight_layout()
        
        return fig
    
    def plot_total_spectrum_input_output(self, solution: Dict,
                                        sum_type: str = 'coherent',
                                figsize: Optional[Tuple[int, int]] = None,
                                ax: Optional[Axes] = None,
                                log_scale: bool = True,
                                xlim: Optional[Tuple[float, float]] = None,
                                **kwargs) -> Figure:
        """Plot total input vs output spectrum.
        
        Args:
            solution: Solution dict from solver.solve()
            sum_type: 'coherent' or 'incoherent' for mode summation
            figsize: Figure size. If None, uses default.
            ax: Optional axes to plot on
            log_scale: If True, plot in dB (log scale). If False, linear scale.
            xlim: Optional x-axis limits (wavelength in μm)
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib Figure object
        """
        if sum_type not in ['coherent', 'incoherent']:
            raise ValueError("sum_type must be 'coherent' or 'incoherent'")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        pulses = solution['pulses']
        input_pulse = pulses[0]
        output_pulse = pulses[-1]
        
        # Get wavelength axis
        freqs_thz = jnp.fft.fftshift(jnp.fft.fftfreq(input_pulse.n_points, input_pulse.dt))
        abs_freqs_thz = input_pulse.center_freq_thz + freqs_thz
        
        # Only plot positive frequencies
        positive_freq_mask = abs_freqs_thz > 0
        abs_freqs_thz_pos = abs_freqs_thz[positive_freq_mask]
        wavelengths_um = C_um_ps / abs_freqs_thz_pos
        
        # Sort by wavelength
        sort_indices = np.argsort(wavelengths_um)
        wavelengths_sorted = np.array(wavelengths_um)[sort_indices]
        
        # Get spectra
        input_spectrum = input_pulse.get_total_spectrum(sum_type=sum_type)
        output_spectrum = output_pulse.get_total_spectrum(sum_type=sum_type)
        
        input_spectrum_pos = np.array(input_spectrum)[positive_freq_mask][sort_indices]
        output_spectrum_pos = np.array(output_spectrum)[positive_freq_mask][sort_indices]
        
        if log_scale:
            # Convert to dB
            input_spectrum_db = 10 * np.log10(input_spectrum_pos / np.max(input_spectrum_pos) + 1e-20)
            output_spectrum_db = 10 * np.log10(output_spectrum_pos / np.max(output_spectrum_pos) + 1e-20)
            
            ax.plot(wavelengths_sorted, input_spectrum_db, 'r--', label='Input', linewidth=2, **kwargs)
            ax.plot(wavelengths_sorted, output_spectrum_db, 'b-', label='Output', linewidth=2, **kwargs)
            ax.set_ylabel('Power (dB)')
            ax.set_ylim((-100, 5))
        else:
            ax.plot(wavelengths_sorted, input_spectrum_pos, 'r--', label='Input', linewidth=2, **kwargs)
            ax.plot(wavelengths_sorted, output_spectrum_pos, 'b-', label='Output', linewidth=2, **kwargs)
            ax.set_ylabel('Power (W)')
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_title(f'Total Input vs Output Spectrum ({sum_type.title()} Sum)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set default x-axis limits: 0.75 * lambda_center to 1.5 * lambda_center
        center_wavelength_um = input_pulse.center_wavelength_um
        default_xlim = (0.75 * center_wavelength_um, 1.5 * center_wavelength_um)
        
        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(default_xlim)
        
        return fig
    
    def plot_mode_coupling(self, solution: Dict,
                          figsize: Optional[Tuple[int, int]] = None,
                                ax: Optional[Axes] = None,
                                **kwargs) -> Figure:
        """Plot power distribution across modes vs. z.
        
        Args:
            solution: Solution dict from solver.solve()
            figsize: Figure size. If None, uses default.
            ax: Optional axes to plot on
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        z_points = solution['ts']
        pulses = solution['pulses']
        
        # Get power in each mode at each z position
        n_modes = pulses[0].n_modes
        z_um = np.array(z_points) * 1e6
        
        for mode_idx in range(n_modes):
            powers = []
            for pulse in pulses:
                power = np.sum(np.array(pulse.get_intensity()[mode_idx, :])) * pulse.dt
                powers.append(power)
            ax.plot(z_um, powers, label=f'Mode {mode_idx}', linewidth=2, **kwargs)
        
        ax.set_xlabel('Propagation Distance (μm)')
        ax.set_ylabel('Mode Power (W)')
        ax.set_title('Mode Coupling - Power Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_power_conservation(self, solution: Dict,
                               sum_type: str = 'incoherent',
                               figsize: Optional[Tuple[int, int]] = None,
                                ax: Optional[Axes] = None,
                                **kwargs) -> Figure:
        """Plot total power vs. z to check power conservation.
        
        Args:
            solution: Solution dict from solver.solve()
            sum_type: 'coherent' or 'incoherent' for power calculation
            figsize: Figure size. If None, uses default.
            ax: Optional axes to plot on
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib Figure object
        """
        if sum_type not in ['coherent', 'incoherent']:
            raise ValueError("sum_type must be 'coherent' or 'incoherent'")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        z_points = solution['ts']
        pulses = solution['pulses']
        
        z_um = np.array(z_points) * 1e6
        powers = []
        
        for pulse in pulses:
            power = pulse.get_total_power(sum_type=sum_type)
            powers.append(power)
        
        powers = np.array(powers)
        initial_power = powers[0]
        
        # Plot in dB relative to initial
        power_db = 10 * np.log10(powers / initial_power + 1e-20)
        
        ax.plot(z_um, power_db, 'b-', linewidth=2, **kwargs)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0 dB (100%)')
        ax.set_xlabel('Propagation Distance (μm)')
        ax.set_ylabel('Power Conservation (dB)')
        ax.set_title(f'Power Conservation ({sum_type.title()} Sum)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        if len(power_db) > 0:
            y_range = max(5.0, np.max(np.abs(power_db)) * 1.2)
            ax.set_ylim((-y_range, y_range))
        
        return fig
    
    def plot_spectral_evolution(self, solution: Dict, mode_idx: int = 0,
                                figsize: Optional[Tuple[int, int]] = None,
                                ax: Optional[Axes] = None,
                                n_z_points: Optional[int] = None,
                                **kwargs) -> Figure:
        """Plot spectrum evolution along propagation (waterfall).
        
        Args:
            solution: Solution dict from solver.solve()
            mode_idx: Which mode to plot
            figsize: Figure size. If None, uses default.
            ax: Optional axes to plot on
            n_z_points: Number of z points to plot. If None, plots all.
            **kwargs: Additional arguments passed to pcolormesh
            
        Returns:
            matplotlib Figure object
        """
        pulses = solution['pulses']
        if mode_idx >= pulses[0].n_modes:
            raise ValueError(f"mode_idx={mode_idx} is out of bounds for n_modes={pulses[0].n_modes}")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        z_points = solution['ts']
        
        # Get wavelength axis
        first_pulse = pulses[0]
        freqs_thz = jnp.fft.fftshift(jnp.fft.fftfreq(first_pulse.n_points, first_pulse.dt))
        abs_freqs_thz = first_pulse.center_freq_thz + freqs_thz
        
        positive_freq_mask = abs_freqs_thz > 0
        abs_freqs_thz_pos = abs_freqs_thz[positive_freq_mask]
        wavelengths_um = C_um_ps / abs_freqs_thz_pos
        
        sort_indices = np.argsort(wavelengths_um)
        wavelengths_sorted = np.array(wavelengths_um)[sort_indices]
        
        # Select z points to plot
        if n_z_points is None:
            indices_to_plot = range(len(pulses))
        else:
            indices_to_plot = np.linspace(0, len(pulses)-1, n_z_points, dtype=int)
        
        # Get spectra
        spectra_list = []
        z_to_plot = []
        for idx in indices_to_plot:
            spectrum = pulses[idx].get_spectrum()[mode_idx, :]
            spectrum_pos = np.array(spectrum)[positive_freq_mask][sort_indices]
            spectra_list.append(spectrum_pos)
            z_to_plot.append(z_points[idx])
        
        data = np.array(spectra_list)  # Shape: (n_z, n_wavelengths)
        
        # Convert to dB
        data_db = 10 * np.log10(data / np.max(data) + 1e-20)
        
        # Create meshgrid
        W, Z = np.meshgrid(wavelengths_sorted, z_to_plot)
        Z_um = Z * 1e6
        
        # Create waterfall plot
        im = ax.pcolormesh(W, Z_um, data_db, cmap='viridis', shading='gouraud', **kwargs)
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Propagation Distance (μm)')
        ax.set_title(f'Mode {mode_idx} - Spectral Evolution')
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Set default x-axis limits: 0.75 * lambda_center to 1.5 * lambda_center
        center_wavelength_um = first_pulse.center_wavelength_um
        default_xlim = (0.75 * center_wavelength_um, 1.5 * center_wavelength_um)
        ax.set_xlim(default_xlim)
        
        return fig
    
    def plot_spectrogram(self, solution: Dict, mode_idx: int = 0,
                        z_idx: Optional[int] = None,
                        windowed: bool = True,
                        window_size: Optional[int] = None,
                        overlap: float = 0.5,
                        freq_axis: str = 'frequency',
                        figsize: Optional[Tuple[int, int]] = None,
                                ax: Optional[Axes] = None,
                                **kwargs) -> Figure:
        """Plot spectrogram showing time-frequency representation.
        
        Args:
            solution: Solution dict from solver.solve()
            mode_idx: Which mode to plot
            z_idx: Which z position to plot. If None, uses final pulse.
            windowed: If True, use STFT. If False, use full FFT at each time point.
            window_size: Size of time window for STFT. If None, auto-based on FWHM.
            overlap: Overlap fraction for windows (0 to 1)
            freq_axis: 'frequency' (THz) or 'wavelength' (μm)
            figsize: Figure size. If None, uses default.
            ax: Optional axes to plot on
            **kwargs: Additional arguments passed to pcolormesh
            
        Returns:
            matplotlib Figure object
        """
        pulses = solution['pulses']
        if mode_idx >= pulses[0].n_modes:
            raise ValueError(f"mode_idx={mode_idx} is out of bounds for n_modes={pulses[0].n_modes}")
        
        if freq_axis not in ['frequency', 'wavelength']:
            raise ValueError("freq_axis must be 'frequency' or 'wavelength'")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            fig = cast(Figure, fig)
        else:
            fig = cast(Figure, ax.figure)
        
        # Select pulse
        if z_idx is None:
            pulse = pulses[-1]
        else:
            if z_idx >= len(pulses):
                raise ValueError(f"z_idx={z_idx} is out of bounds for {len(pulses)} pulses")
            pulse = pulses[z_idx]
        
        t_ps = np.array(pulse.t)
        field = np.array(pulse.field[mode_idx, :])
        n_points = pulse.n_points
        dt = pulse.dt
        
        # Get frequency axis
        freqs_thz = jnp.fft.fftshift(jnp.fft.fftfreq(n_points, dt))
        abs_freqs_thz = pulse.center_freq_thz + freqs_thz
        
        if windowed:
            # Short-Time Fourier Transform (STFT)
            if window_size is None:
                # Auto-determine window size based on FWHM
                fwhm_info = pulse.get_fwhm_info()
                fwhm_freq_range = fwhm_info['count']
                # Use window size that captures ~10% of signal
                window_size = max(32, min(n_points // 10, 256))
            
            # Calculate step size based on overlap
            step_size = int(window_size * (1 - overlap))
            step_size = max(1, step_size)
            
            # Create windows
            time_windows = []
            freq_data = []
            
            for i in range(0, n_points - window_size + 1, step_size):
                window = field[i:i+window_size]
                # Apply window function (Hanning)
                window_hanning = window * np.hanning(len(window))
                # FFT
                fft_window = np.fft.fftshift(np.fft.fft(window_hanning))
                freq_data.append(np.abs(fft_window)**2)
                time_windows.append(t_ps[i + window_size // 2])  # Center time of window
            
            if len(time_windows) == 0:
                # Fallback: use single window
                window_hanning = field * np.hanning(n_points)
                fft_window = np.fft.fftshift(np.fft.fft(window_hanning))
                freq_data = [np.abs(fft_window)**2]
                time_windows = [t_ps[n_points // 2]]
            
            spectrogram_data = np.array(freq_data).T  # Shape: (n_freq, n_time)
            time_axis = np.array(time_windows)
            
        else:
            # Full FFT at each time point (simpler but less informative)
            # For each time point, compute FFT of entire signal
            # This is essentially the same as the spectrum, repeated for each time
            fft_full = np.fft.fftshift(np.fft.fft(field))
            spectrogram_data = np.tile(np.abs(fft_full)**2, (n_points, 1)).T
            time_axis = t_ps
        
        # Prepare frequency/wavelength axis
        if freq_axis == 'frequency':
            y_axis = abs_freqs_thz
            y_label = 'Frequency (THz)'
            y_data = spectrogram_data
        else:  # wavelength
            positive_freq_mask = abs_freqs_thz > 0
            abs_freqs_thz_pos = abs_freqs_thz[positive_freq_mask]
            wavelengths_um = C_um_ps / abs_freqs_thz_pos
            sort_indices = np.argsort(wavelengths_um)
            wavelengths_sorted = np.array(wavelengths_um)[sort_indices]
            y_axis = wavelengths_sorted
            y_label = 'Wavelength (μm)'
            y_data = spectrogram_data[positive_freq_mask, :][sort_indices, :]
        
        # Convert to dB
        y_data_db = 10 * np.log10(y_data / np.max(y_data) + 1e-20)
        
        # Create meshgrid
        T, Y = np.meshgrid(time_axis, y_axis)
        
        # Create spectrogram plot
        im = ax.pcolormesh(T, Y, y_data_db, cmap='viridis', shading='gouraud', **kwargs)
        
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel(y_label)
        ax.set_title(f'Mode {mode_idx} - Spectrogram')
        if z_idx is not None:
            z_pos = solution['ts'][z_idx] * 1e6
            ax.set_title(f'Mode {mode_idx} - Spectrogram (z={z_pos:.1f} μm)')
        plt.colorbar(im, ax=ax, label='Power Spectral Density (dB)')
        
        # Set default y-axis limits for wavelength axis: 0.75 * lambda_center to 1.5 * lambda_center
        if freq_axis == 'wavelength':
            center_wavelength_um = pulse.center_wavelength_um
            default_ylim = (0.75 * center_wavelength_um, 1.5 * center_wavelength_um)
            ax.set_ylim(default_ylim)
        
        return fig
    
    def plot_comparison_grid(self, solution: Dict, mode_idx: int = 0,
                            figsize: Optional[Tuple[int, int]] = None,
                            **kwargs) -> Figure:
        """Create a comprehensive comparison grid with multiple plots.
        
        Creates a 2×2 grid with:
        - Top left: Temporal evolution (waterfall)
        - Top right: Spectral evolution (waterfall)
        - Bottom left: Input vs Output temporal
        - Bottom right: Input vs Output spectrum
        
        Args:
            solution: Solution dict from solver.solve()
            mode_idx: Which mode to plot
            figsize: Figure size. If None, uses (16, 12).
            **kwargs: Additional arguments passed to subplot methods
            
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = (16, 12)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Top left: Temporal evolution
        self.plot_propagation(solution, mode_idx=mode_idx, plot_type='intensity', ax=axes[0, 0])
        
        # Top right: Spectral evolution
        self.plot_spectral_evolution(solution, mode_idx=mode_idx, ax=axes[0, 1])
        
        # Bottom left: Input vs Output temporal
        self.plot_total_input_output(solution, ax=axes[1, 0])
        
        # Bottom right: Input vs Output spectrum
        self.plot_total_spectrum_input_output(solution, ax=axes[1, 1])
        
        plt.tight_layout()
        
        return fig

