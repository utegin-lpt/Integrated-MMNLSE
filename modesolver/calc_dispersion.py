import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
from scipy.io import loadmat, savemat
from tqdm import tqdm
from pathlib import Path

def calculate_dispersion(folder_name, width, height, 
                         lambda0_build, lambda0_disp, 
                         lrange, Nf, 
                         modes_list, 
                         polynomial_fit_order, num_disp_orders,
                         mode_type='TM'):
    """
    Calculate the dispersion of a slab waveguide.
    
    Parameters:
        width (float): Width of the waveguide core in um.
        height (float): Height of the waveguide core in um.
        lambda0_build (float): Center wavelength used for mode calculation (m).
        lmambda0_disp (float): Center wavelength used for dispersion calculation (m).
        lrange (float): Wavelength range in m. If 0, only the center wavelength will be used.
        Nf (int): Number of frequency points at which the modes will be calculated.
        mode_min (int): Minimum mode index to calculate.
        mode_max (int): Maximum mode index to calculate.
        polynomial_fit_order (int): Order of polynomial fit for the effective index.
        num_disp_orders (int): Number of dispersion orders to calculate.
        mode_type (str): Type of mode to calculate ('TE', 'TM', or 'scalar').

    """
    # Speed of light (m/ps)
    c = 2.99792458e-4

    # Set frequency/wavelength grid in proper units:
    if lrange == 0:
        raise ValueError("Cannot calculate dispersion with only one frequency point")
    else:
        f0 = c / lambda0_build         # center frequency in THz (1/ps)
        frange = c / lambda0_build**2 * lrange
        df = frange / Nf
        # Create frequency array (column vector) equivalent to MATLAB: f = (f0 + (-Nf/2:Nf/2-1)*df)'
        f = f0 + np.arange(-Nf//2, Nf//2) * df
        f = f.reshape(-1)  # ensure 1D
        # Wavelengths in um: lambda = c ./ f * 1e6
        wavelengths = c / f * 1e6
        savemat(os.path.join(folder_name, 'wavelengths.mat'), {'wavelengths': wavelengths})
    num_modes = int(len(modes_list))
    if mode_type == 'scalar':
        field = 'scalar'
        boundary = '0000'
    elif mode_type == 'TE':
        field = 'ex'
        field_name = r'$E_x$'
        boundary = '000S'
    elif mode_type == 'TM':
        field = 'ey'
        field_name = r'$E_y$'
        boundary = '000S'
    else:
        raise ValueError("Invalid mode type. Choose 'scalar', 'TE', or 'TM'.")
    # Convert lambda0_build to um for later use
    lambda0_build_um = lambda0_build * 1e6

    # Load the calculated effective index values
    # We'll build an array n_calc (Nf x num_modes) by loading each .mat file.
    n_calc = np.zeros((Nf, num_modes))

    tqdm_bar = tqdm(range(Nf), desc="Loading wavelength", unit="wavelength")
    for kk in tqdm_bar:
        lam = wavelengths[kk]  # in um
        for ii in range(num_modes):
            fname = os.path.join(
                folder_name, 'modes',
                f"w{int(width*1000)}h{int(height*1000)}boundary{boundary}field{field}mode{ii}wavelength{round(lam*1000)}"
            )
            # load the .mat file; assume it contains a variable 'neff'
            matdata = loadmat(fname + '.mat')
            # Some MATLAB files add extra fields; extract 'neff'
            neff = matdata['neff'].squeeze()
            n_calc[kk, ii] = neff
        tqdm_bar.set_description(f"Loading wavelength = {round(lam*1000)} nm")

    # Calculate the propagation constants
    beta_calc = np.zeros((Nf, num_modes))
    w = 2 * np.pi * f  # angular frequency in 1/ps
    for midx in range(num_modes):
        # beta in 1/m; note: MATLAB divides by c (m/ps), so units should be consistent.
        beta_calc[:, midx] = n_calc[:, midx] * w / c

    # Fit the propagation constants to a polynomial and compute derivatives
    dw = 2 * np.pi * df
    w_disp = 2 * np.pi * c / lambda0_disp  # angular frequency (1/ps) at which dispersion is calculated

    # b_coefficients: each row is a mode, columns: [beta0, beta1, ...]
    b_coefficients = np.zeros((num_modes, num_disp_orders+1))

    for midx in range(num_modes):
        beta_calc_i = beta_calc[:, midx]
        # Fit polynomial of order polynomial_fit_order to (w, beta_calc)
        # np.polyfit returns coefficients in descending powers.
        beta_fit = np.polyfit(w, beta_calc_i, polynomial_fit_order)
        # Evaluate fitted polynomial at w_disp; beta0 in 1/m converted to 1/mm via /1000:
        beta0 = np.polyval(beta_fit, w_disp) / 1000
        b_coefficients[midx, 0] = beta0
        # For each dispersion order, use np.polyder to compute derivative coefficients.
        for disp_order in range(1, num_disp_orders+1):
            # Compute disp_order-th derivative
            deriv_coeff = np.polyder(beta_fit, m=disp_order)
            # Evaluate derivative at w_disp.
            # Multiply by (10^3)^disp_order/1000 as in MATLAB (scale factors)
            beta_deriv = np.polyval(deriv_coeff, w_disp) * (10**3) ** disp_order / 1000
            b_coefficients[midx, disp_order] = beta_deriv

    # Make beta0 and beta1 relative to the fundamental mode (mode 0)
    b_coefficients[:, 0] = b_coefficients[:, 0] - b_coefficients[0, 0]
    b_coefficients[:, 1] = b_coefficients[:, 1] - b_coefficients[0, 1]

    # Transpose to match MATLAB output (each row is coefficient order)
    betas = b_coefficients.T

    # Save betas to a .mat file
    betas_filename = os.path.join(folder_name, f'betas_{ int(lambda0_disp * 1e9) }.mat')
    savemat(betas_filename, {'betas': betas})

    # Build cell arrays (lists) of w_vectors and l_vectors.
    w_vectors = []
    l_vectors = []

    w_vectors.append(w)  # order 0
    l_vectors.append(2 * np.pi * c / w * 1e6)  # in um

    # For each higher dispersion order, each vector has one less point.
    for disp_order in range(1, num_disp_orders+1):
        w_prev = w_vectors[disp_order - 1]
        # New omega vector: dw/2 + first (len(w_prev)-1) elements of w_prev
        new_w = dw/2 + w_prev[:-1]
        w_vectors.append(new_w)
        # Corresponding lambda vector in um
        new_l = 2 * np.pi * c / new_w * 1e6
        l_vectors.append(new_l)

    # Now compute beta_full: a list of arrays with dispersion orders.
    beta_full = []
    beta_full.append(beta_calc / 1000)  # order 0: 1/mm

    # For each dispersion order, calculate finite differences (using np.diff)
    for disp_order in range(1, num_disp_orders+1):
        # For each mode, take the difference of the previous order along axis 0.
        delta_beta = np.diff(beta_full[disp_order - 1], axis=0) / dw * 1000
        beta_full.append(delta_beta)

    # Plotting all orders for each mode
    coo = plt.cm.hsv(np.linspace(0, 1, num_modes))
    ylabels = ['1/mm', 'fs/mm']
    for disp_order in range(2, num_disp_orders+1):
        ylabels.append(rf'fs^{{{disp_order}}}/mm')

    # Plot dispersion for each mode separately
    for midx in range(num_modes):
        # Zero-dispersion-wavelength (ZDW) finder
        # --- 6. Dispersion parameter D(λ) & Zero-dispersion wavelengths
        lambda_vec   = l_vectors[2]                 # (N,)  µm

        lambda_col   = lambda_vec[:, None]               # ← NEW (N,1) for broadcasting
        D_full  = (-2*np.pi*c / (lambda_col*1e-6)**2   # ps/(nm·km)
                * beta_full[2] * 1e-6)         # shapes now (N,1)*(N,num_modes)

        zdw_path = Path(folder_name) / "zero_dispersion_wavelengths.txt"
        with zdw_path.open("w", encoding="utf-8") as f:
            f.write("# mode_index  ZDW_1[µm]  ZDW_2[µm] ...\n")
            for midx in range(num_modes):
                Dm  = D_full[:, midx]
                idx = np.where(np.diff(np.signbit(Dm)))[0]     # sign changes
                zdws = []
                for i in idx:                                  # linear interp
                    lam1, lam2 = lambda_vec[i], lambda_vec[i+1]
                    D1,   D2   = Dm[i],        Dm[i+1]
                    zdw = lam1 - D1 * (lam2 - lam1) / (D2 - D1)
                    zdws.append(zdw)
                line = f"{midx} " + " ".join(f"{z:.6f}" for z in zdws) + "\n"
                f.write(line)

        print(f"★ Zero-dispersion wavelengths saved to: {zdw_path}")
        # ────────────────────────────────────────────────────────────────
    for midx in range(num_modes):
        fig, axs = plt.subplots(num_disp_orders+1, 1, figsize=(8, 3 * (num_disp_orders + 1)))
        for order in range(num_disp_orders+1):
            ax = axs[order] if num_disp_orders+1 > 1 else axs
            # Only plot data for the current mode
            ax.plot(l_vectors[order], beta_full[order][:, midx], color=coo[midx])
            ax.set_ylabel(ylabels[order])
            ax.set_xlabel('µm')
            ax.set_title(rf'$\beta_{{{order}}}$ for {mode_type}-Mode {midx}')
            ax.autoscale(enable=True, tight=True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, f'betas_plot_{mode_type}_mode_{midx}.png'), dpi=150)
        plt.close(fig)

    # plot D for each mode in a single figure
    fig, axs = plt.subplots(num_modes, 1, figsize=(8, 3 * num_modes))
    for midx in range(num_modes):
        ax = axs[midx] if num_modes > 1 else axs
        # Plot D (2nd order dispersion)
        D = -2 * np.pi * c / (l_vectors[2] * 1e-6)**2 * beta_full[2][:, midx] * 1e-9 # in ps/nm/km
        ax.plot(l_vectors[2], D, color=coo[midx])
        ax.set_ylabel(r'$ps^2$/(nm km)')
        ax.set_xlabel('µm')
        ax.set_title(rf'$D$ for {mode_type}-Mode {midx}')
        ax.autoscale(enable=True, tight=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, f'GVD_plot_{mode_type}.png'), dpi=150)
    plt.close(fig)

    # Save all the dispersion data (l_vectors, beta_full) to a csv file )
    print(np.array(beta_full[2]).shape)
    print(np.expand_dims(np.array(l_vectors[2]), -1).shape)
    dispersion_data = np.concatenate([np.expand_dims(np.array(l_vectors[2]), -1), np.array(beta_full[2])], axis=1)
    print(dispersion_data.shape)
    dispersion_filename = os.path.join(folder_name, f'dispersion_data_{int(lambda0_disp * 1e9)}.csv')
    np.savetxt(dispersion_filename, dispersion_data, delimiter=',', header='lambda, beta', comments='')
    print(f"Dispersion data saved to: {dispersion_filename}")
    return b_coefficients