import numpy as np
import matplotlib.pyplot as plt

def build_curved_slab(width, height, Nx, Ny, dx, dy, nclad, ncore, R):
    """
    Build a curved slab waveguide index profile with arbitrary dx and dy grid spacing.
    
    Parameters:
        width (float): Width of the waveguide core in microns
        height (float): Height of the waveguide core in microns
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        dx (float): Grid spacing in x-direction (microns)
        dy (float): Grid spacing in y-direction (microns)
        nclad (float): Refractive index of the cladding
        ncore (float): Refractive index of the core
        R (float): Corner radius in microns
        
    Returns:
        epsilon (ndarray): Squared permittivity distribution (Ny x Nx)
        x (ndarray): Spatial coordinates in x-direction
        dx (float): Grid spacing in x-direction
        spatial_window_x (float): Total window size in x-direction
        y (ndarray): Spatial coordinates in y-direction
        dy (float): Grid spacing in y-direction
        spatial_window_y (float): Total window size in y-direction
    """
    spatial_window_x = Nx * dx
    spatial_window_y = Ny * dy
    
    # Create the spatial grid
    x = np.linspace(-spatial_window_x / 2, spatial_window_x / 2, Nx)  # x-coordinates
    y = np.linspace(-spatial_window_y / 2, spatial_window_y / 2, Ny)  # y-coordinates
    
    # Create the cladding matrix (background refractive index)
    clad = nclad * np.ones((Ny, Nx))
    
    # Size of the core in grid points
    core_width = round(width / dx)
    core_height = round(height / dy)
    
    # Create the core refractive index distribution
    core = (ncore - nclad) * np.ones((core_height, core_width))
    
    # Padding calculations
    pad_y1 = int(np.floor((Ny - core_height) / 2))
    pad_y2 = Ny - core_height - pad_y1
    pad_x1 = int(np.floor((Nx - core_width) / 2))
    pad_x2 = Nx - core_width - pad_x1

    # Create the padded core by embedding the core in a zero array
    core_padded = np.zeros((Ny, Nx))
    core_padded[pad_y1:pad_y1 + core_height, pad_x1:pad_x1 + core_width] = core

    # Create coordinate meshgrid in pixel coordinates
    yy, xx = np.meshgrid(np.arange(Nx), np.arange(Ny))
    
    # Convert radius to pixels for x and y directions separately
    Rx_pixels = R / dx  # Radius in x-pixels
    Ry_pixels = R / dy  # Radius in y-pixels
    
    # Bottom-left rounded corner
    # Calculate distance in a stretched coordinate system
    dist_bottom_left = np.sqrt(
        ((xx - (pad_y1 + core_height - Ry_pixels)) / Ry_pixels)**2 + 
        ((yy - pad_x1 - Rx_pixels) / Rx_pixels)**2
    )
    # Apply the mask: points outside the circle of radius 1 in stretched coordinates
    mask_bottom_left = (dist_bottom_left > 1) & (xx > pad_y1 + core_height - Ry_pixels) & (yy < pad_x1 + Rx_pixels)
    core_padded[mask_bottom_left] = 0
    
    # Bottom-right rounded corner
    dist_bottom_right = np.sqrt(
        ((xx - (pad_y1 + core_height - Ry_pixels)) / Ry_pixels)**2 + 
        ((yy - (pad_x1 + core_width - Rx_pixels)) / Rx_pixels)**2
    )
    mask_bottom_right = (dist_bottom_right > 1) & (xx > pad_y1 + core_height - Ry_pixels) & (yy > pad_x1 + core_width - Rx_pixels)
    core_padded[mask_bottom_right] = 0
    
    # Add the core to the cladding matrix
    epsilon = clad + core_padded
    
    # Final squared permittivity distribution
    epsilon = epsilon ** 2
    
    return epsilon, x, dx, spatial_window_x, y, dy, spatial_window_y
