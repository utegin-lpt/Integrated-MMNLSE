import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import factorial
from constants import C
from pathlib import Path
from tqdm import tqdm

from utils import *
import os
np.seterr(divide = 'ignore') 

import sys, os
# prepend the project root so that 'simlibrary' can be imported
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from simlibrary.propagator import *
from simlibrary.waveguide import Waveguide
from simlibrary.sim import Simulation
from simlibrary.pulse import Pulse
from simlibrary.utils import *

# Load the betas and coupling tensors from the mode solver output
base = Path("06. Waveguides/SiN_waveguide/slab_TE_h_800_nm_w_6000_nm_wl_1550_nm")
betas = loadmat(base / "betas_1550.mat")["betas"]

# --- Define simulation parameters -----------------------------------
order = 5   # Dispersion order  
sim_modes = [0, 1, 2, 3, 4, 5]  # Simulation modes to use
num_modes = len(sim_modes)  
n = np.arange(0, order + 1, 1).reshape(-1, 1)
unit_conversion = 1e-15 ** n  * 1e3
betas = betas[:order+1, sim_modes] * unit_conversion # convert to sᵏ m⁻¹
print("Betas shape:", betas.shape)  

# Load the coupling tensor
SR_tensor = loadmat(base / "S_tensors_12modes.mat")["SR"]
SR_tensor = np.array(SR_tensor, dtype=np.complex128)
SR_tensor = SR_tensor[:num_modes, :num_modes, :num_modes, :num_modes]

C = 299792458
# Center wavelength and simulation parameters
lambda_0 = 1550e-9
N = 2**16
time_window = 50e-12
print("Time resolution:", time_window / N * 1e15, "fs")

# Material properties
# Define the material for the waveguide
material = "SiN"

# --- Define parameters ------------------------------------------
Ppeak = 25e3
wg = Waveguide(material, betas, SR_tensor)
fwhm = 250e-15
length = 3000e-6

# --- Define modal coefficients -----------------------------------
modal_coefficients = np.zeros((wg.num_modes))
modal_coefficients[5] = 1
modal_coefficients[4] = 1
modal_coefficients[3] = 1
modal_coefficients[2] = 1
modal_coefficients[1] = 1
modal_coefficients[0] = 1

# --- Build pulse ---------------------------------------------------------
pulse = Pulse.gaussian(
    peak_power_w=Ppeak,          # Peak power in Watts
    fwhm=fwhm,                    # FWHM
    time_window=time_window,      # total window 
    n_modes=wg.num_modes,
    n_time_points=N,
    modal_coefficients=modal_coefficients,
)

# --- Run simulation ------------------------------------------------------
sim = Simulation(wg, pulse, lambda_0)
sim_folder = os.path.join(base, "paper", f"simulations_{length*1e6}_um_{int(Ppeak)}_W_{int(fwhm * 1e15)}_fs_coeff_{np.array2string(modal_coefficients, separator='_')}")

output_field = sim.run(pulse.field, save_path = sim_folder, length=length, dz=2e-6, plot_propagation=600)
