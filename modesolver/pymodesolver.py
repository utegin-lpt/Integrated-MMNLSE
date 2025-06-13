import numpy as np
import os 

from solve_modes import solve_modes
from calc_dispersion import calculate_dispersion
from calc_SR_SK import calculate_SR_SK
from delete_mat import delete_mat_files
from tqdm import tqdm

width = 6              
height = 0.8             
dx = 0.01  
dy = 0.01  
Nx = int((width + 1) / dx)
Ny = int((height + 1) / dy)            
spatial_window_x = Nx * dx
spatial_window_y = Ny * dy

lambda0 = 1550e-9               # center wavelength used for mode calculation (m)
lambda0_disp = 1550e-9          # center wavelength used for dispersion calculation (m)
lrange = 1000e-9                # wavelength range in m; must be same as used in mode calc.
Nf = 100                        # number of frequency/wavelength grid points
num_modes = 6                   # number of modes to compute
mode_list = np.arange(0, num_modes)  # list of modes to compute

mode = 'TE'                     # mode type: 'TE', 'TM', or 'scalar'
material = 'SiN'                # material type: 'SiN', or 'Si'

# Define the folder where the output will be stored
main_folder = '06. Waveguides'  # main folder where the output will be stored

# Do not modify the following folder naming convention
folder_name = f'{main_folder}/test_{material}_waveguide/paper_slab_{mode}_h_{int(height*1000)}_nm_w_{int(width*1000)}_nm_wl_{int(lambda0*1e9)}_nm'  # folder where the output will be stored

# Create the folder if it does not exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Solve the modes
solve_modes(material, folder_name, dx, dy, Nx, Ny,width, height, lambda0, lrange, Nf, num_modes, mode)

polynomial_fit_order = 6  # order of polynomial fit for the effective index
num_disp_orders = 6     # number of dispersion orders (0th through 2nd order)

# calculate the dispersion
betas = calculate_dispersion(folder_name, width, height, lambda0, lambda0_disp, lrange, Nf, mode_list, 
                polynomial_fit_order, num_disp_orders, mode)

# calculate the overlap integrals
calculate_SR_SK(folder_name, Nx, Ny, width, height, lambda0, 0, num_modes, mode)

# delete the .mat files to save space since we don't need it for propagation
delete_mat_files(folder_name)

