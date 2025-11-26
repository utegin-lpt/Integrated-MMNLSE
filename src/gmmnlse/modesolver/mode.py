"""
Mode class to represent electromagnetic modes.
"""

import numpy as np
from scipy import integrate


def centered2d(x):
    """
    Convert node-centered 2D array to cell-centered by averaging over 2x2 grid.
    
    Args:
        x: 2D array of shape (nx, ny) on node grid
        
    Returns:
        2D array of shape (nx-1, ny-1) on cell centers
    """
    if x.ndim != 2:
        raise ValueError("centered2d expects a 2D array")
    return 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])


class Mode:
    """
    Represents a waveguide mode with electric and magnetic field components.
    
    Attributes:
        wavelength (float): Wavelength in micrometers
        x (ndarray): X-coordinates in micrometers
        y (ndarray): Y-coordinates in micrometers
        n_eff (complex): Effective refractive index
        Ex, Ey, Ez (ndarray): Electric field components
        Hx, Hy, Hz (ndarray): Magnetic field components
    """
    
    def __init__(self, wavelength, frequency, x, y, n_eff, n_complex, Ex, Ey, Ez, Hx, Hy, Hz):
        """
        Initialize a Mode object.
        
        Args:
            wavelength (float): Wavelength in micrometers
            x (ndarray): X-coordinates
            y (ndarray): Y-coordinates  
            n_eff (complex): Effective refractive index
            Ex, Ey, Ez (ndarray): Electric field components
            Hx, Hy, Hz (ndarray): Magnetic field components
        """
        self.wavelength = wavelength # in μm
        self.frequency = frequency  # in THz
        self.x = x
        self.y = y
        self.n_eff = n_eff
        self.n_complex = n_complex
        
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Hx = Hx
        self.Hy = Hy
        self.Hz = Hz
        
        self.fields = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
            'Hx': Hx, 'Hy': Hy, 'Hz': Hz
        }
    
    def intensity(self):
        """
        Calculate the time-averaged Poynting vector (intensity).
        
        The intensity is given by the magnitude of the Poynting vector:
        
        I = |S| = |Re(E × H*)| / 2
        
        where:
        - S = 0.5 * Re(E × H*) is the time-averaged Poynting vector
        - E = (Ex, Ey, Ez) and H = (Hx, Hy, Hz) are the electric and magnetic fields
        
        Note: E fields are on cell centers, H fields are on node grid.
        We need to center H fields to match E field shapes.
        
        Returns:
            ndarray: Intensity distribution |E x H*|
        """
        # Center H fields to match E field shapes (cell centers)
        # E fields are (nx-1, ny-1), H fields are (nx, ny)
        Hx_centered = centered2d(self.Hx)
        Hy_centered = centered2d(self.Hy)
        Hz_centered = centered2d(self.Hz)
        
        # S = 0.5 * Re(E x H*)
        # |S| for intensity
        Sx = 0.5 * np.real(self.Ey * np.conj(Hz_centered) - self.Ez * np.conj(Hy_centered))
        Sy = 0.5 * np.real(self.Ez * np.conj(Hx_centered) - self.Ex * np.conj(Hz_centered))
        Sz = 0.5 * np.real(self.Ex * np.conj(Hy_centered) - self.Ey * np.conj(Hx_centered))
        
        return np.sqrt(Sx**2 + Sy**2 + Sz**2)
    
    def intensity_TE_TM(self):
        """
        Calculate TE and TM intensity components.
        
        Returns:
            tuple: (TE_intensity, TM_intensity)
        """
        # TE: Ey, Hx, Hz dominant
        # TM: Hy, Ex, Ez dominant
        TE_intensity = np.abs(self.Ey)**2
        TM_intensity = np.abs(self.Ex)**2 + np.abs(self.Ez)**2
        
        return TE_intensity, TM_intensity
    
    def norm(self):
        """
        Calculate the norm of the mode (total power).
        
        Returns:
            float: Square root of integrated intensity
        """
        x_centered = 0.5 * (self.x[:-1] + self.x[1:])
        y_centered = 0.5 * (self.y[:-1] + self.y[1:])
        
        intensity = self.intensity()
        
        # Double integration
        result = integrate.trapezoid([integrate.trapezoid(intensity[i, :], y_centered) for i in range(len(x_centered))], x_centered)
        
        return np.sqrt(np.abs(result))
    
    def normalize(self):
        """
        Normalize the mode so that integrated intensity equals 1.
        
        Returns:
            Mode: Self (for chaining)
        """
        norm = self.norm()
        
        if norm > 0:
            self.Ex /= norm
            self.Ey /= norm
            self.Ez /= norm
            self.Hx /= norm
            self.Hy /= norm
            self.Hz /= norm
            
            # Update fields dictionary
            self.fields = {
                'Ex': self.Ex, 'Ey': self.Ey, 'Ez': self.Ez,
                'Hx': self.Hx, 'Hy': self.Hy, 'Hz': self.Hz
            }
        
        return self
    
    def __repr__(self):
        return f"Mode(wavelength={self.wavelength:.4f} μm, n_eff={self.n_eff:.6f})"
