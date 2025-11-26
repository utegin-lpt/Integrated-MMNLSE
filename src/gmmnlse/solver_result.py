"""Solver result container with convenient helper methods.

This module provides the SolverResult class which wraps solver outputs and provides
convenient methods for accessing pulses, plotting propagation maps, and extracting
data at specific positions.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Optional, Tuple, Union, Sequence
from .pulse import Pulse
from .constants import C_um_ps


class SolverResult:
    """Container for solver results with convenient helper methods.
    
    Stores propagation results and provides methods for accessing pulses,
    plotting propagation maps, and extracting data at specific positions.
    
    Attributes:
        pulses (List[Pulse]): List of Pulse objects at each save point.
        z_points (jnp.ndarray): Propagation distances (m).
        stats (dict): Solver statistics (e.g., num_steps).
    """
    
    def __init__(
        self,
        pulses: List[Pulse],
        z_points: jnp.ndarray,
        stats: dict,
    ):
        """Initialize SolverResult.
        
        Args:
            pulses: List of Pulse objects, one for each save point.
            z_points: Propagation distances in meters, shape (n_save,).
            stats: Dictionary with solver statistics.
        """
        if len(pulses) != len(z_points):
            raise ValueError(
                f"Number of pulses ({len(pulses)}) must match number of z_points ({len(z_points)})"
            )
        
        self.pulses = pulses
        self.z_points = jnp.asarray(z_points)
        self.stats = stats
    
    @property
    def n_save(self) -> int:
        """Number of save points."""
        return len(self.pulses)
    
    @property
    def propagation_grid(self) -> jnp.ndarray:
        """Get propagation grid (z_points)."""
        return self.z_points
    
    def get_input_pulse(self) -> Pulse:
        """Get the input pulse (at z=0)."""
        return self.pulses[0]
    
    def get_final_pulse(self) -> Pulse:
        """Get the final pulse (at maximum z)."""
        return self.pulses[-1]
    
    def get_pulse_at(
        self,
        z: float,
        method: str = "nearest"
    ) -> Pulse:
        """Get pulse at specific propagation distance.
        
        Args:
            z: Propagation distance in meters.
            method: Interpolation method. Options:
                - "nearest": Return pulse at nearest save point (default)
                - "interpolate": Interpolate between two nearest pulses
        
        Returns:
            Pulse object at requested position.
        """
        z_arr = np.asarray(self.z_points)
        
        if method == "nearest":
            idx = np.argmin(np.abs(z_arr - z))
            return self.pulses[idx]
        
        elif method == "interpolate":
            # Find surrounding points
            idx_below = np.searchsorted(z_arr, z, side='right') - 1
            idx_above = idx_below + 1
            
            # Handle boundaries
            if idx_below < 0:
                return self.pulses[0]
            if idx_above >= len(self.pulses):
                return self.pulses[-1]
            
            z_below = float(z_arr[idx_below])
            z_above = float(z_arr[idx_above])
            
            if abs(z_above - z_below) < 1e-12:
                return self.pulses[idx_below]
            
            # Linear interpolation factor
            alpha = (z - z_below) / (z_above - z_below)
            
            # Interpolate field
            field_below = self.pulses[idx_below].field
            field_above = self.pulses[idx_above].field
            field_interp = (1 - alpha) * field_below + alpha * field_above
            
            # Create new pulse with interpolated field
            return Pulse(
                sim_params=self.pulses[0].sim_params,
                field_t=field_interp
            )
        
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'nearest' or 'interpolate'.")
    
    def get_field_array(self) -> jnp.ndarray:
        """Get raw field array, shape (n_save, n_modes, n_points)."""
        return jnp.array([pulse.field for pulse in self.pulses])
    
    def get_intensity_array(self, sum_type: str = "coherent") -> jnp.ndarray:
        """Get intensity array, shape (n_save, n_points).
        
        Args:
            sum_type: 'coherent' or 'incoherent'. Defaults to 'coherent'.
        
        Returns:
            Intensity array.
        """
        fields = self.get_field_array()
        
        if sum_type == "coherent":
            field_sum = jnp.sum(fields, axis=1)  # Sum over modes
            return jnp.abs(field_sum) ** 2
        elif sum_type == "incoherent":
            return jnp.sum(jnp.abs(fields) ** 2, axis=1)
        else:
            raise ValueError(f"sum_type must be 'coherent' or 'incoherent', got '{sum_type}'")
    
    def get_spectrum_array(self, sum_type: str = "coherent") -> jnp.ndarray:
        """Get spectral power array, shape (n_save, n_points).
        
        Args:
            sum_type: 'coherent' or 'incoherent'. Defaults to 'coherent'.
        
        Returns:
            Spectral power array.
        """
        fields = self.get_field_array()
        fields_freq = jnp.fft.fftshift(jnp.fft.fft(fields, axis=-1), axes=-1)
        
        if sum_type == "coherent":
            field_sum = jnp.sum(fields_freq, axis=1)
            return jnp.abs(field_sum) ** 2
        elif sum_type == "incoherent":
            return jnp.sum(jnp.abs(fields_freq) ** 2, axis=1)
        else:
            raise ValueError(f"sum_type must be 'coherent' or 'incoherent', got '{sum_type}'")
    
    def plot_temporal_propagation(
        self,
        sum_type: str = "coherent",
        z_scale: str = "mm",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Axes:
        """Plot temporal propagation map (intensity vs time vs z).
        
        Args:
            sum_type: 'coherent' or 'incoherent'. Defaults to 'coherent'.
            z_scale: Scale for z-axis. Options: 'm', 'mm', 'um'. Defaults to 'mm'.
            xlim: X-axis limits (time, ps). If None, uses full range.
            ylim: Y-axis limits (z). If None, uses full range.
            ax: Matplotlib axes. If None, creates new figure.
            **kwargs: Additional arguments passed to plt.imshow().
        
        Returns:
            Matplotlib axes object.
        """
        intensity = np.asarray(self.get_intensity_array(sum_type=sum_type))
        time_ps = np.asarray(self.pulses[0].t)
        
        # Normalize intensity
        intensity_norm = intensity / np.max(intensity)
        intensity_db = 10.0 * np.log10(np.clip(intensity_norm, 1e-30, None))
        
        # Convert z to requested scale
        z_converted = self._convert_z_scale(self.z_points, z_scale)
        
        # Create plot
        created_new_figure = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        extent = (
            float(time_ps[0]),
            float(time_ps[-1]),
            float(z_converted[0]),
            float(z_converted[-1])
        )
        
        im = ax.imshow(
            intensity_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=kwargs.pop('cmap', 'viridis'),
            **kwargs
        )
        
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(f"Propagation Distance ({z_scale})")
        ax.set_title(f"Temporal Propagation ({sum_type})")
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        plt.colorbar(im, ax=ax, label="Intensity (dB)")
        
        if created_new_figure:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def plot_spectral_propagation(
        self,
        sum_type: str = "coherent",
        z_scale: str = "mm",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Axes:
        """Plot spectral propagation map (power vs wavelength vs z).
        
        Args:
            sum_type: 'coherent' or 'incoherent'. Defaults to 'coherent'.
            z_scale: Scale for z-axis. Options: 'm', 'mm', 'um'. Defaults to 'mm'.
            xlim: X-axis limits (wavelength, µm). If None, uses valid range.
            ylim: Y-axis limits (z). If None, uses full range.
            ax: Matplotlib axes. If None, creates new figure.
            **kwargs: Additional arguments passed to plt.imshow().
        
        Returns:
            Matplotlib axes object.
        """
        spectrum = np.asarray(self.get_spectrum_array(sum_type=sum_type))
        
        # Get wavelength axis from first pulse
        base_pulse = self.pulses[0]
        wavelengths_um, mask = self._get_wavelength_axis(base_pulse)
        valid_indices = np.where(mask)[0]
        if valid_indices.size == 0:
            raise ValueError("No positive-frequency wavelengths available for plotting.")
        wavelengths_valid = wavelengths_um[valid_indices]
        sort_idx = np.argsort(wavelengths_valid)
        wavelengths_sorted = wavelengths_valid[sort_idx]
        spectrum_masked = spectrum[:, valid_indices][:, sort_idx]
        
        # Normalize spectrum
        spectrum_norm = spectrum_masked / np.max(spectrum_masked)
        spectrum_db = 10.0 * np.log10(np.clip(spectrum_norm, 1e-30, None))
        
        # Convert z to requested scale
        z_converted = self._convert_z_scale(self.z_points, z_scale)
        
        # Create plot
        created_new_figure = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        extent = (
            float(wavelengths_sorted[0]),
            float(wavelengths_sorted[-1]),
            float(z_converted[0]),
            float(z_converted[-1])
        )
        
        im = ax.imshow(
            spectrum_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=kwargs.pop('cmap', 'viridis'),
            **kwargs
        )
        
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel(f"Propagation Distance ({z_scale})")
        ax.set_title(f"Spectral Propagation ({sum_type})")
        
        default_xlim = self._default_wavelength_xlim(wavelengths_sorted)
        ax.set_xlim(xlim if xlim else default_xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        plt.colorbar(im, ax=ax, label="Power (dB)")
        
        if created_new_figure:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def plot_propagation(
        self,
        ax_temporal: Axes,
        ax_spectral: Axes,
        temporal_sum_type: str = "coherent",
        spectral_sum_type: str = "coherent",
        z_scale: str = "mm",
        layout: str = "horizontal",
        time_ax_lim: Optional[Tuple[float, float]] = None,
        spect_ax_x_lim: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> None:
        """Plot temporal and spectral propagation maps on provided axes.
        
        Plots temporal and spectral propagation on the provided axes. The figure
        and axes should be created by the caller (e.g., in a notebook).
        
        Args:
            ax_temporal: Matplotlib axes for temporal plot.
            ax_spectral: Matplotlib axes for spectral plot.
            temporal_sum_type: 'coherent' or 'incoherent' for temporal intensity.
                Defaults to 'coherent'.
            spectral_sum_type: 'coherent' or 'incoherent' for spectral intensity.
                Defaults to 'coherent'.
            z_scale: Scale for z-axis. Options: 'm', 'mm', 'um'. Defaults to 'mm'.
            layout: 'horizontal' (side-by-side) or 'vertical' (stacked).
                Used for colorbar placement. Defaults to 'horizontal'.
            time_ax_lim: X-axis limits for temporal plot (time, ps).
                If None, uses full range.
            spect_ax_x_lim: X-axis limits for spectral plot (wavelength, µm).
                If None, uses valid wavelength range.
            **kwargs: Additional arguments passed to pcolormesh (e.g., cmap, vmin, vmax).
        """
        if layout not in {"horizontal", "vertical"}:
            raise ValueError(f"layout must be 'horizontal' or 'vertical', got '{layout}'")
        
        # Get data
        intensity = np.asarray(self.get_intensity_array(sum_type=temporal_sum_type))
        spectrum = np.asarray(self.get_spectrum_array(sum_type=spectral_sum_type))
        
        time_ps = np.asarray(self.pulses[0].t)
        base_pulse = self.pulses[0]
        wavelengths_um, mask = self._get_wavelength_axis(base_pulse)
        valid_indices = np.where(mask)[0]
        if valid_indices.size == 0:
            raise ValueError("No positive-frequency wavelengths available for plotting.")
        wavelengths_valid = wavelengths_um[valid_indices]
        sort_idx = np.argsort(wavelengths_valid)
        wavelengths_masked = wavelengths_valid[sort_idx]
        spectrum_masked = spectrum[:, valid_indices][:, sort_idx]
        
        # Normalize
        intensity_norm = intensity / np.max(intensity)
        intensity_db = 10.0 * np.log10(np.clip(intensity_norm, 1e-30, None))
        
        spectrum_norm = spectrum_masked / np.max(spectrum_masked)
        spectrum_db = 10.0 * np.log10(np.clip(spectrum_norm, 1e-30, None))
        
        # Convert z to requested scale
        z_converted = self._convert_z_scale(self.z_points, z_scale)
        
        # Temporal plot
        if time_ax_lim is None:
            time_ax_lim = (float(time_ps[0]), float(time_ps[-1]))
        if spect_ax_x_lim is None:
            spect_ax_x_lim = self._default_wavelength_xlim(wavelengths_masked)
        
        TT, ZZ = np.meshgrid(time_ps, z_converted)
        vmin = kwargs.pop('vmin', -80)
        vmax = kwargs.pop('vmax', 0)
        cmap = kwargs.pop('cmap', 'magma')
        
        pcm_temporal = ax_temporal.pcolormesh(
            TT, ZZ, intensity_db,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs
        )
        ax_temporal.set_title(f"Temporal evolution ({temporal_sum_type})")
        ax_temporal.set_xlabel("Time (ps)")
        ax_temporal.set_xlim(time_ax_lim[0], time_ax_lim[1])
        
        # Add colorbar for temporal plot
        fig = ax_temporal.figure
        if layout == "horizontal":
            ax_temporal.tick_params(left=False, labelleft=False, right=True, labelright=True)
            cbar_temporal = fig.colorbar(
                pcm_temporal, ax=ax_temporal, label="Power (dB)",
                location="left", pad=0.01
            )
            cbar_temporal.ax.yaxis.set_ticks_position("left")
            cbar_temporal.ax.yaxis.set_label_position("left")
        else:  # vertical
            ax_temporal.set_ylabel(f"Distance ({z_scale})")
            fig.colorbar(pcm_temporal, ax=ax_temporal, label="Power (dB)", pad=0.01)
        
        # Spectral plot
        WW, ZZ2 = np.meshgrid(wavelengths_masked, z_converted)
        pcm_spectral = ax_spectral.pcolormesh(
            WW, ZZ2, spectrum_db,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs
        )
        ax_spectral.set_title(f"Spectral evolution ({spectral_sum_type})")
        ax_spectral.set_xlabel("Wavelength (µm)")
        ax_spectral.set_xlim(spect_ax_x_lim[0], spect_ax_x_lim[1])
        ax_spectral.set_ylabel(f"Distance ({z_scale})")
        
        # Add colorbar for spectral plot
        fig.colorbar(pcm_spectral, ax=ax_spectral, label="Power (dB)", pad=0.01)
    
    def plot_snapshot(
        self,
        axes: np.ndarray,
        snapshot_positions: Union[List[float], Sequence[float], np.ndarray],
        snapshot_scale: str = "mm",
        sum_type: str = "coherent",
        mode_labels: Optional[List[str]] = None,
        time_ax_x_lim: Optional[Union[Tuple[float, float], List[float], Sequence[float]]] = None,
        time_ax_y_lim: Optional[Union[Tuple[float, float], List[float], Sequence[float]]] = None,
        spect_ax_x_lim: Optional[Union[Tuple[float, float], List[float], Sequence[float]]] = None,
        spect_ax_y_lim: Optional[Union[Tuple[float, float], List[float], Sequence[float]]] = None,
    ) -> None:
        """Plot snapshots at specific propagation distances on provided axes.
        
        Plots temporal and spectral views at multiple propagation distances on the
        provided axes array. The figure and axes should be created by the caller
        (e.g., in a notebook). Axes should be a 2D array with shape (2, n_cols) where
        axes[0, col] is temporal and axes[1, col] is spectral for each column.
        
        Args:
            axes: 2D array of matplotlib axes with shape (2, n_cols).
                axes[0, col] for temporal plots, axes[1, col] for spectral plots.
            snapshot_positions: List of propagation distances for snapshots.
                Units determined by snapshot_scale.
            snapshot_scale: Scale for snapshot positions. Options: 'm', 'mm', 'um'.
                Defaults to 'mm'.
            sum_type: 'coherent' or 'incoherent' for total intensity calculation.
                Defaults to 'coherent'.
            mode_labels: Optional list of mode labels. If None, uses 'Mode 0', 'Mode 1', etc.
            time_ax_x_lim: X-axis limits for temporal plots (time, ps).
                If None, uses full range.
            time_ax_y_lim: Y-axis limits for temporal plots (dB).
                If None, uses [-200, 0].
            spect_ax_x_lim: X-axis limits for spectral plots (wavelength, µm).
                If None, uses valid wavelength range.
            spect_ax_y_lim: Y-axis limits for spectral plots (dB).
                If None, uses [-80, 0].
        """
        if sum_type not in {"coherent", "incoherent"}:
            raise ValueError(f"sum_type must be 'coherent' or 'incoherent', got '{sum_type}'")
        
        # Validate axes shape
        if axes.ndim != 2 or axes.shape[0] != 2:
            raise ValueError(f"axes must be 2D array with shape (2, n_cols), got shape {axes.shape}")
        
        n_cols = axes.shape[1]
        
        # Convert snapshot positions to meters
        snapshot_positions_arr = np.asarray(snapshot_positions)
        if len(snapshot_positions_arr) != n_cols:
            raise ValueError(
                f"Number of snapshot positions ({len(snapshot_positions_arr)}) must match "
                f"number of columns in axes ({n_cols})"
            )
        
        if snapshot_scale == "m":
            snapshot_positions_m = snapshot_positions_arr
        elif snapshot_scale == "mm":
            snapshot_positions_m = snapshot_positions_arr * 1e-3
        elif snapshot_scale == "um":
            snapshot_positions_m = snapshot_positions_arr * 1e-6
        else:
            raise ValueError(f"Unknown scale '{snapshot_scale}'. Use 'm', 'mm', or 'um'.")
        
        # Convert z_points to same scale for finding indices
        z_m = np.asarray(self.z_points)
        
        # Get base pulse for axes
        base_pulse = self.pulses[0]
        time_axis = np.asarray(base_pulse.t)
        wavelengths_um, mask = self._get_wavelength_axis(base_pulse)
        valid_indices = np.where(mask)[0]
        if valid_indices.size == 0:
            raise ValueError("No positive-frequency wavelengths available for plotting.")
        wavelengths_valid = wavelengths_um[valid_indices]
        sort_idx = np.argsort(wavelengths_valid)
        wavelengths_sorted = wavelengths_valid[sort_idx]
        spectral_indices = valid_indices[sort_idx]
        
        # Set default mode labels
        if mode_labels is None:
            n_modes = base_pulse.n_modes
            mode_labels = [f"Mode {i}" for i in range(n_modes)]
        
        # Set default axis limits and convert lists to tuples
        if time_ax_x_lim is None:
            time_ax_x_lim = (float(time_axis[0]), float(time_axis[-1]))
        else:
            time_ax_x_lim = tuple(float(x) for x in time_ax_x_lim)
        if time_ax_y_lim is None:
            time_ax_y_lim = (-200, 0)
        else:
            time_ax_y_lim = tuple(float(x) for x in time_ax_y_lim)
        if spect_ax_x_lim is None:
            spect_ax_x_lim = self._default_wavelength_xlim(wavelengths_sorted)
        else:
            spect_ax_x_lim = tuple(float(x) for x in spect_ax_x_lim)
        if spect_ax_y_lim is None:
            spect_ax_y_lim = (-80, 0)
        else:
            spect_ax_y_lim = tuple(float(x) for x in spect_ax_y_lim)
        
        # Get field array for efficient access
        fields_array = self.get_field_array()  # (n_save, n_modes, n_points)
        
        for col, snapshot_z_m in enumerate(snapshot_positions_m):
            # Find nearest z index
            z_idx = int(np.argmin(np.abs(z_m - snapshot_z_m)))
            fields = np.asarray(fields_array[z_idx])  # (n_modes, n_points)
            fields_freq = np.fft.fftshift(np.fft.fft(fields, axis=-1), axes=-1)
            
            # Calculate total intensity
            if sum_type == 'coherent':
                # Coherent sum: I(t) = |Σ_p A_p(t)|²
                total_field = np.sum(fields, axis=0)
                total_field_intensity = np.abs(total_field) ** 2
                # Coherent sum: S(ω) = |Σ_p A_p(ω)|²
                total_spectrum = np.sum(fields_freq, axis=0)
                total_spectrum_intensity = np.abs(total_spectrum) ** 2
            else:  # incoherent
                total_field_intensity = np.sum(np.abs(fields) ** 2, axis=0)
                total_spectrum_intensity = np.sum(np.abs(fields_freq) ** 2, axis=0)
            
            # Normalize to peak
            total_time_peak = np.max(total_field_intensity)
            total_spec_peak = np.max(total_spectrum_intensity[spectral_indices])
            
            total_time_db = 10 * np.log10(np.maximum(total_field_intensity, 1e-30) / total_time_peak)
            total_spec_db = 10 * np.log10(
                np.maximum(total_spectrum_intensity[spectral_indices], 1e-30) / total_spec_peak
            )
            
            # Plot total (black line)
            axes[0, col].plot(time_axis, total_time_db, color="black", linewidth=2, label="Total")
            axes[1, col].plot(wavelengths_sorted, total_spec_db, color="black", linewidth=2, label="Total")
            
            # Plot individual modes
            for mode_idx in range(fields.shape[0]):
                label = mode_labels[mode_idx] if mode_idx < len(mode_labels) else f"Mode {mode_idx}"
                mode_field = fields[mode_idx]
                
                # Temporal intensity
                mode_time_intensity = np.abs(mode_field) ** 2
                mode_time_db = 10 * np.log10(np.maximum(mode_time_intensity, 1e-30) / total_time_peak)
                axes[0, col].plot(time_axis, mode_time_db, alpha=0.6, label=label)
                
                # Spectral intensity
                spectrum = np.fft.fftshift(np.fft.fft(mode_field))
                mode_spec_intensity = np.abs(spectrum) ** 2
                mode_spec_db = 10 * np.log10(
                    np.maximum(mode_spec_intensity[spectral_indices], 1e-30) / total_spec_peak
                )
                axes[1, col].plot(wavelengths_sorted, mode_spec_db, alpha=0.6, label=label)
            
            # Set title and labels
            snapshot_value = snapshot_positions_arr[col]
            axes[0, col].set_title(f"z = {snapshot_value:.1f} {snapshot_scale}")
            
            if col == 0:
                axes[0, col].set_ylabel("|A(t)| (dB)")
                axes[1, col].set_ylabel("|A(λ)| (dB)")
            else:
                axes[0, col].set_ylabel("")
                axes[1, col].set_ylabel("")
            
            # Set axis limits
            axes[0, col].set_xlim(time_ax_x_lim[0], time_ax_x_lim[1])
            axes[0, col].set_ylim(time_ax_y_lim[0], time_ax_y_lim[1])
            axes[1, col].set_xlim(spect_ax_x_lim[0], spect_ax_x_lim[1])
            axes[1, col].set_ylim(spect_ax_y_lim[0], spect_ax_y_lim[1])
            
            axes[0, col].set_xlabel("Time (ps)")
            axes[1, col].set_xlabel("Wavelength (µm)")
            
            # Add legend on last column
            if col == n_cols - 1:
                axes[0, col].legend(loc="upper right", fontsize="small")
                axes[1, col].legend(loc="upper right", fontsize="small")
    
    def _get_wavelength_axis(self, pulse: Pulse) -> Tuple[np.ndarray, np.ndarray]:
        """Get wavelength axis and mask for valid wavelengths.
        
        Helper method similar to _wavelength_axis in the notebook.
        
        Args:
            pulse: Pulse object to extract wavelength axis from.
        
        Returns:
            Tuple of (wavelengths_um, mask) where mask indicates valid wavelengths.
        """
        freqs_shifted_thz = np.asarray(pulse.sim_params._freqs_absolute_shifted_thz)
        positive_mask = np.isfinite(freqs_shifted_thz) & (freqs_shifted_thz > 0.0)
        wavelengths_um = C_um_ps / np.where(freqs_shifted_thz > 0.0, freqs_shifted_thz, np.nan)
        mask = positive_mask & np.isfinite(wavelengths_um)
        return wavelengths_um, mask
    
    def _convert_z_scale(self, z_m: jnp.ndarray, scale: str) -> jnp.ndarray:
        """Convert z from meters to requested scale.
        
        Args:
            z_m: Z values in meters.
            scale: Target scale. Options: 'm', 'mm', 'um'.
        
        Returns:
            Z values in requested scale.
        """
        if scale == "m":
            return z_m
        elif scale == "mm":
            return z_m * 1e3
        elif scale == "um":
            return z_m * 1e6
        else:
            raise ValueError(f"Unknown scale '{scale}'. Use 'm', 'mm', or 'um'.")

    @staticmethod
    def _default_wavelength_xlim(
        wavelengths_um: np.ndarray,
        preferred_range: Tuple[float, float] = (0.5, 4.0),
    ) -> Tuple[float, float]:
        """Compute a sensible default wavelength window for plots."""
        if wavelengths_um.size == 0:
            return preferred_range
        wl_min = float(np.min(wavelengths_um))
        wl_max = float(np.max(wavelengths_um))
        lower_pref, upper_pref = preferred_range
        lower = max(lower_pref, wl_min)
        upper = min(upper_pref, wl_max)
        if lower >= upper:
            return (min(wl_min, wl_max), max(wl_min, wl_max))
        return (lower, upper)
    
    def __getitem__(self, key: str):
        """Dict-like access for backward compatibility.
        
        Supports:
        - result["ys"] → get_field_array()
        - result["pulses"] → self.pulses
        - result["final_pulse"] → get_final_pulse()
        - result["ts"] → self.z_points
        - result["stats"] → self.stats
        
        Args:
            key: Dictionary key.
        
        Returns:
            Corresponding value.
        
        Raises:
            KeyError: If key is not recognized.
        """
        if key == "ys":
            return self.get_field_array()
        elif key == "pulses":
            return self.pulses
        elif key == "final_pulse":
            return self.get_final_pulse()
        elif key == "ts":
            return self.z_points
        elif key == "stats":
            return self.stats
        else:
            raise KeyError(f"Unknown key '{key}'. Available keys: 'ys', 'pulses', 'final_pulse', 'ts', 'stats'")
    
    def __len__(self) -> int:
        """Return number of save points."""
        return len(self.pulses)
    
    def __repr__(self) -> str:
        """String representation of SolverResult."""
        return f"SolverResult(n_save={self.n_save}, z_range=[{float(self.z_points[0]):.6e}, {float(self.z_points[-1]):.6e}] m)"

