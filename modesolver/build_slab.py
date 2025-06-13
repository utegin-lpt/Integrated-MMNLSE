import numpy as np
import matplotlib.pyplot as plt

def build_slab(width, height, Nx, Ny, dx, dy, nclad, ncore):
    """
    Create a squared permittivity distribution for a slab waveguide.
    
    Parameters:
      width          : Core width (microns)
      height         : Core height (microns)
      Nx             : Number of grid points (assumed square grid)
      spatial_window : Size of the spatial window (microns)
      nclad          : Cladding refractive index
      ncore          : Core refractive index
      
    Returns:
      epsilon : The squared permittivity distribution (Nx x Nx array)
      x       : Spatial coordinates (1D array)
      dx      : Spatial step size (microns)
    """
    spatial_window_x = Nx * dx
    spatial_window_y = Ny * dy
    # Create the spatial grid
    x = np.linspace(-spatial_window_x / 2, spatial_window_x / 2, Nx)  # spatial coordinates
    y = np.linspace(-spatial_window_y / 2, spatial_window_y / 2, Ny)  # y-coordinates (not used)
    # Create the cladding matrix (background refractive index)
    clad = nclad * np.ones((Ny, Nx))

    # Size of the core in grid points
    core_width = int(round(width / dx))
    core_height = int(round(height / dy))
    # Create the core refractive index difference distribution
    core = (ncore - nclad) * np.ones((core_height, core_width))

    # Padding calculations
    pad_y1 = int(np.floor((Ny - core_height) / 2))
    pad_y2 = Ny - core_height - pad_y1
    pad_x1 = int(np.floor((Nx - core_width) / 2))
    pad_x2 = Nx - core_width - pad_x1
    # Create the padded core by embedding the core in a zero array.
    core_padded = np.zeros((Ny, Nx))
    core_padded[pad_y1:pad_y1 + core_height, pad_x1:pad_x1 + core_width] = core

    # Add the core to the cladding matrix
    epsilon = clad + core_padded

    # Final squared permittivity distribution
    epsilon = epsilon ** 2

    return epsilon, x, dx, spatial_window_x, y, dy, spatial_window_y
