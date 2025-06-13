import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from build_slab import build_slab
from build_curved_slab import build_curved_slab
from wgmodes import wgmodes
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment
def field_centroid(field, x_grid, y_grid):
    """
    Return power-weighted (cx, cy) centroid of a complex field.
    field : 2-D complex ndarray  (Ny × Nx)
    x_grid, y_grid : 1-D arrays of length Nx and Ny (same ones you already have)
    """
    power = np.abs(field)**2
    cx = (power * x_grid).sum() / power.sum()
    cy = (power * y_grid[:, None]).sum() / power.sum()
    return np.array([cx, cy])

def get_SiN_refractive_index(wavelength):
    """
    Calculate the refractive index of SiN at a given wavelength using the Sellmeier equation.
    
    Parameters:
        wavelength (float): Wavelength in micrometers.
        
    Returns:
        n (float): Refractive index of SiN.
    """
    # Sellmeier coefficients for SiN
    B1 = 3.0249
    B2 = 40314
    C1 = 0.1353406**2
    C2 = 1239.842**2
    # Calculate the refractive index using the Sellmeier equation
    n = np.sqrt(1 + (B1 * wavelength**2) / (wavelength**2 - C1) + (B2 * wavelength**2) / (wavelength**2 - C2))
    
    return n
def get_SiO2_refractive_index(wavelength):
    """
    Calculate the refractive index of SiO2 at a given wavelength using the Sellmeier equation.
    
    Parameters:
        wavelength (float): Wavelength in micrometers.
        
    Returns:
        n (float): Refractive index of SiO2.
    """
    # Sellmeier coefficients for SiO2
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2
    # Calculate the refractive index using the Sellmeier equation
    n = np.sqrt(1 + (B1 * wavelength**2) / (wavelength**2 - C1) + (B2 * wavelength**2) / (wavelength**2 - C2) + (B3 * wavelength**2) / (wavelength**2 - C3))
    
    return n

def get_Si_refractive_index(wavelength):
    """
    Calculate the refractive index of Si at a given wavelength using the Sellmeier equation.
    Parameters:
        wavelength (float): Wavelength in micrometers.
    Returns:
        n (float): Refractive index of Si.
    """

    B1 = 0.939816
    B2 = 0.003043475
    C1 = 0.00810475
    C2 = 100

    # Calculate the refractive index using the Sellmeier equation
    n = np.sqrt(11.6858 + (B1 * wavelength**2) / (wavelength**2 - C1) + (B2 * wavelength**2) / (wavelength**2 - C2) )
    return n

# ─── tracking parameters ─────────────────────────────────────────────
STEP_INT        = 5          # store one frame every 5 wavelengths
MAX_HISTORY_STEPS = 20       # total λ-span of the history bank
HIST_SIZE       = MAX_HISTORY_STEPS // STEP_INT   # == 4 frames

def solve_modes(material, folder_name, 
                dx, dy,
                Nx, Ny,
                width, height, 
                lambda0, lrange, Nf, 
                num_modes, mode,
                R = None):
    """
    Solve the modes of a slab waveguide using the semivectorial finite difference method.
    
    Parameters:
        width (float): Width of the waveguide core in um.
        height (float): Height of the waveguide core in um.
        ncore (float): Refractive index of the core.
        nclad (float): Refractive index of the cladding.
        lambda0 (float): Center wavelength in m.
        lrange (float): Wavelength range in m. If 0, only the center wavelength will be used.
        Nf (int): Number of frequency points at which the modes will be calculated.
        num_modes (int): Number of modes to compute.
        mode (str): Mode type: 'TE', 'TM', or 'scalar'.
        """
    
    # Create output folder if it does not exist
    os.makedirs(os.path.join(folder_name, 'modes'), exist_ok=True)

    # Set frequency range in wavelength domain (in um)
    c = 2.99792458e-4  # speed of light in m/ps
    if lrange == 0:
        # Only use the center wavelength (convert m -> um)
        wavelengths = np.array([lambda0 * 1e6])
    else:
        f0 = c / lambda0  # center frequency in THz (ps^-1)
        frange = c / lambda0**2 * lrange
        df = frange / Nf
        # MATLAB: f = f0 + (-Nf/2 : Nf/2 -1)*df; Here, use integer indices.
        f = f0 + np.arange(-Nf//2, Nf//2) * df
        wavelengths = c / f * 1e6  # wavelengths in um

        print(f"Largest wavelength: {np.max(wavelengths):.3f} um")
        print(f"Smallest wavelength: {np.min(wavelengths):.3f} um")
        print(f"Number of wavelengths: {len(wavelengths)}")

    all_neff_values = np.zeros((num_modes, len(wavelengths)))

    # Loop over each wavelength
    tqdm_bar = tqdm(enumerate(np.flip(wavelengths)), desc="Calculating modes", total=len(wavelengths))
    savemat(os.path.join(folder_name, 'wavelengths.mat'), {'wavelengths': wavelengths})
    
    field_hist = []      # list of Ny×Nx×nm arrays
    neff_hist  = []      # list of length-nm vectors

    def should_store(step_idx):
        """True if this wavelength index should be kept in history."""
        return step_idx % STEP_INT == 0
    
    for kk, lambda_um in tqdm_bar:
        # Calculate the refractive index of SiN at the current wavelength
        if material == 'SiN':
            ncore = get_SiN_refractive_index(lambda_um)
        elif material == 'Si':
            ncore = get_Si_refractive_index(lambda_um)
        else:
            raise ValueError("Invalid material. Choose 'SiN' or 'Si'.")
        nclad = get_SiO2_refractive_index(lambda_um)

        # Build the slab waveguide
        if R is not None:
            epsilon, x, dx, spatial_window_x, y, dy, spatial_window_y = build_curved_slab(width, height, Nx, Ny, dx, dy, nclad, ncore, R)
        else:
            epsilon, x, dx, spatial_window_x, y, dy, spatial_window_y = build_slab(width, height, Nx, Ny, dx, dy, nclad, ncore)
        
        # Use the center of the domain as guess:
        guess = np.sqrt(epsilon[Ny//2, Nx//2])
        
        # Print the index profile
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        cax = axs[0].pcolormesh(x, y, np.sqrt(epsilon), cmap=plt.cm.gray_r)
        axs[0].set_aspect('equal')
        axs[0].set_title('Index Profile')
        fig.colorbar(cax, ax=axs[0])
        axs[1].plot(y, np.sqrt(epsilon[:, Nx//2]))
        axs[1].set_title('Vertical cross-section')
        plt.tight_layout()
        profile_fname = os.path.join(folder_name, 'waveguide_profile.png')
        plt.savefig(profile_fname, dpi=150)
        plt.close(fig)
        
        # Call svmodes to calculate the modes.
        # Note: here lambda_um is in um.
        if mode == 'scalar':
            field = 'scalar'
            field_name = r'$Scalar$'
            boundary = '0000'
        elif mode == 'TE':
            field = 'ex'
            field_name = r'$E_x$'
            boundary = '000S'
        elif mode == 'TM':
            field = 'ey'
            field_name = r'$E_y$'
            boundary = '000S'
        else:
            raise ValueError("Invalid mode type. Choose 'TE', 'TM', or 'scalar'.")
        
        # Call the solver
        t_start = time.perf_counter()
        phi1, neff1 = wgmodes(lambda_um, guess, num_modes, dx, dy, epsilon, boundary, field)
        t_elapsed = time.perf_counter() - t_start
        
        # ─── mode-tracking over a sparse history bank ────────────────
        if lrange > 0:
            if kk == 0:
                # first wavelength: nothing to compare with
                if should_store(kk):
                    field_hist.append(phi1.copy())
                    neff_hist.append(neff1.copy())
            else:
                nm = phi1.shape[2]

                # ---- build cost aggregated over history frames --------------
                cost = np.zeros((nm, nm))

                # pre-compute centroids and norms for current λ
                cent_curr = np.array([field_centroid(phi1[:, :, m], x, y) for m in range(nm)])
                norm_curr = np.array([np.linalg.norm(phi1[:, :, m])       for m in range(nm)])

                for frame_idx, (f_prev, n_prev) in enumerate(zip(field_hist, neff_hist)):
                    cent_prev = np.array([field_centroid(f_prev[:, :, m], x, y) for m in range(nm)])
                    norm_prev = np.array([np.linalg.norm(f_prev[:, :, m])       for m in range(nm)])
                    for i in range(nm):
                        for j in range(nm):
                            # 1) overlap error
                            denom = norm_prev[i] * norm_curr[j]
                            if denom < 1e-12:
                                ov_err = 1.0  # treat as max error
                            else:
                                ov = abs((f_prev[:, :, i].conj() * phi1[:, :, j]).sum()) / denom
                                ov_err = 1.0 - ov
                            
                            # 2) neff gap
                            neff_err = abs(n_prev[i] - neff1[j])
                            print(f"neff_err: {neff_err}")
                            # 3) centroid gap
                            cent_err = np.linalg.norm(cent_prev[i] - cent_curr[j])

                            cost[i, j] += ov_err + 5.0 * neff_err + 2.0 * cent_err
                # print(cost)
                cost /= len(field_hist)                # mean over frames
                rows, cols = linear_sum_assignment(cost)
                phi1  = phi1[:, :, cols]
                neff1 = neff1[cols]

                f_last = field_hist[-1]                # newest frame in history
                for m in range(nm):
                    if np.real((f_last[:, :, m].conj() * phi1[:, :, m]).sum()) < 0:
                        phi1[:, :, m] *= -1

                if should_store(kk):
                    field_hist.append(phi1.copy())
                    neff_hist.append(neff1.copy())
                    # trim to HIST_SIZE (==4) frames
                    if len(field_hist) > HIST_SIZE:
                        field_hist.pop(0)
                        neff_hist.pop(0)

        tqdm_bar.set_description(f"Solving width: {width} um, height: {height} um, wavelength: {lambda_um:.3f} um, solved in {t_elapsed:.3e} seconds")
        fig_width = 10
        aspect_ratio = (y.max() - y.min()) / (x.max() - x.min())
        fig_height = fig_width * aspect_ratio

        # Save each mode in a separate file
        for ii in range(num_modes):
            fig = plt.figure(figsize=(fig_width, fig_height))
            phi = phi1[:, :, ii]
            neff = neff1[ii]
            all_neff_values[ii, kk] = neff
            
            mesh = plt.pcolormesh(x, y, np.real(phi), cmap=plt.cm.coolwarm, shading='auto', vmin=-1, vmax=1)
            plt.title(f"{field_name} Field {mode}-Mode {ii}, n_eff = {neff:.4f}")
            plt.axis('auto')
            plt.colorbar(mesh, label='Magnitude')
            plt.xlabel('x (um)')
            plt.ylabel('y (um)')
            plt.tight_layout()
            
            # Build filename; for slab, use width and height scaled by 10.
            fname = os.path.join(folder_name, 'modes', f"w{int(width*1000)}h{int(height*1000)}boundary{boundary}field{field}mode{ii}wavelength{round(lambda_um*1000)}.png")
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            
            # Save corresponding .mat file containing x, phi, epsilon, neff.
            mat_fname = os.path.join(folder_name, 'modes', f"w{int(width*1000)}h{int(height*1000)}boundary{boundary}field{field}mode{ii}wavelength{round(lambda_um*1000)}.mat")
            savemat(mat_fname, {'x': x, 'y': y, 'phi': phi, 'epsilon': epsilon, 'neff': neff})
            np.savetxt(mat_fname.replace('.mat', '.csv'), np.real(phi), delimiter=',')
        # if lambda_um == lambda0, plot and save the mode profile in a single image:
        if lambda_um == lambda0:
            fig, axes = plt.subplots(num_modes//2, 2, figsize=(10, num_modes // 2 * 3))
            axes = axes.flatten()
            for ii in range(num_modes):
                phi = phi1[:, :, ii]
                neff = neff1[ii]
                
                mesh = axes[ii].pcolormesh(x, y, np.real(phi), cmap=plt.cm.coolwarm, shading='auto', vmin=-1, vmax=1)
                axes[ii].set_title(f"{field_name} Field {mode}-Mode {ii}, n_eff = {neff:.4f}")
                axes[ii].axis('auto')
                fig.colorbar(mesh, ax=axes[ii], label='Magnitude')
                axes[ii].set_xlabel('x (um)')
                axes[ii].set_ylabel('y (um)')
            plt.tight_layout()
            # Save the figure
            fig_fname = os.path.join(folder_name, f"w{int(width*1000)}h{int(height*1000)}boundary{boundary}field{field}mode_all_wavelengths_{lambda_um}.png")
            plt.savefig(fig_fname, dpi=150)
            plt.close(fig)

    # Save all neff values for all modes into a single .mat file
    all_neff_filename = os.path.join(folder_name, 'all_neff_values.mat')
    savemat(all_neff_filename, {'all_neff_values': all_neff_values})     

    # Find pair of modes, that has similar neff values within a tolerance
    tol = 1e-4  # Tolerance for neff matching
    n_modes, n_wl = all_neff_values.shape

    # Find any pair of modes at the same wavelength that differ by < tol
    matches = []
    for j in range(n_wl):
        for m1 in range(n_modes):
            for m2 in range(m1 + 1, n_modes):
                if abs(all_neff_values[m1, j] - all_neff_values[m2, j]) < tol:
                    matches.append((m1, m2, wavelengths[j], all_neff_values[m1, j], all_neff_values[m2, j]))

    if len(matches) != 0:
        # Write results to a text file
        with open('degenerate_neff_pairs.txt', 'w') as f:
            f.write('mode1\tmode2\twavelength_um\tneff_mode1\tneff_mode2\n')
            for m1, m2, wl, ne1, ne2 in matches:
                f.write(f'{m1}\t{m2}\t{wl:.8f}\t{ne1:.6f}\t{ne2:.6f}\n')

