import numpy as np
import os
from scipy.io import loadmat, savemat
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_SR_SK(folder_name, 
                    Nx, Ny,
                    width, height,
                    lambda0, 
                    min_mode, max_mode,
                    mode_type='TM',):
    
    # Set parameters
    modes_list = np.arange(min_mode, max_mode) 
    num_modes = len(modes_list)
    linear_yes = True   # True = linear polarization, False = circular
    gpu_yes = False     # True = run on GPU, False = run on CPU
    single_yes = True   # True = single precision, False = double

    # File name parameters
    if mode_type == 'scalar':
        boundary = '0000'
        field = 'scalar'
    elif mode_type == 'TE':
        boundary = '000S'
        field = 'ex'
    elif mode_type == 'TM':
        boundary = '000S'
        field = 'ey'
    # dir_prefix = f'02. pysvmodes/SiN_waveguide/slab_{mode_type}_h_{int(height*1000)}_nm_w_{int(width*1000)}_nm_wl_{int(lambda0*1e9)}_nm'  # folder where the output will be stored
    dir_prefix = folder_name

    # Load the modes
    fields = np.zeros((Ny, Nx, num_modes), dtype=np.complex64 if single_yes else np.complex128)
    norms = np.zeros(num_modes, dtype=np.complex64 if single_yes else np.complex128)

    for ii, mode in enumerate(modes_list):
        # Construct file name
        name = os.path.join(
            dir_prefix, 'modes',
            f"w{int(width*1000)}h{int(height*1000)}boundary{boundary}field{field}mode{mode}wavelength{int(lambda0*1e9)}.mat"
        )
        mat = loadmat(name)
        phi = mat['phi']
        fields[:, :, ii] = phi.astype(fields.dtype)
        norms[ii] = np.sqrt(np.sum(np.sum(np.abs(phi)**2, axis=0), axis=0))# normalization factor

    # Load spatial information for dx
    x = mat['x'].squeeze()
    dx = float(x[1] - x[0]) * 1e-6  # spatial step in m
    y = mat['y'].squeeze()
    dy = float(y[1] - y[0]) * 1e-6  # spatial step in m
    # Calculate the overlap integrals
    SR = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=fields.dtype)

    for m1 in tqdm(range(num_modes), desc="Calculating overlap integrals", unit="mode"):
        # print(f"Starting m1 = {m1+1}")
        for m2 in range(num_modes):
            # print(f"Starting m2 = {m2+1}")
            for m3 in range(num_modes):
                for m4 in range(num_modes):
                    prod = (fields[:, :, m1] * fields[:, :, m2] *
                            fields[:, :, m3] * fields[:, :, m4])
                    SR[m1, m2, m3, m4] = np.sum(np.sum(prod, axis = 0), axis = 0) / (
                        norms[m1] * norms[m2] * norms[m3] * norms[m4]
                    )

    SR = SR / (dx * dy)

    # Eliminate zero elements
    thresholdzero = SR.flat[0] / 100000
    cnt = 0
    SR_flat = SR.flatten()
    for idx in range(SR_flat.size):
        if abs(SR_flat[idx]) < thresholdzero:
            SR_flat[idx] = 0
        else:
            cnt += 1
    print(f"Calculated {cnt} nonzero entries in the S_R tensor")
    SR = SR_flat.reshape((num_modes, num_modes, num_modes, num_modes))

    # For linear polarization SK=SR, for circular polarization SK=2/3*SR
    mult_factor = 1 if linear_yes else 2/3
    SK = mult_factor * SR
    Aeff = 1 / SR[0, 0, 0, 0]

    # Save to disk
    print(f'Largest value in SK tensor: {np.max(SK)}')
    save_name = os.path.join(dir_prefix, f"S_tensors_{num_modes}modes.mat")
    savemat(save_name, {'SK': SK, 'SR': SR, 'Aeff': Aeff})

    # Plot flattened SK tensor
    flat = SK.ravel()                  # same as T.reshape(-1) – C‐order flatten
    flat = np.abs(flat / np.max(flat))  # normalize to max value

    # coords[k] is an array with the k‑th coordinate at every position
    coords = np.indices(SK.shape)              # shape (4, P, L, M, N)
    p_idx, l_idx, m_idx, n_idx = (c.ravel() for c in coords)

    # pattern masks
    mask_pppp = (p_idx == l_idx) & (l_idx == m_idx) & (m_idx == n_idx)
    mask_pqpq = (p_idx == m_idx) & (l_idx == n_idx) & (p_idx != l_idx)
    mask_ppqq = (p_idx == l_idx) & (m_idx == n_idx) & (p_idx != m_idx)
    mask_qppp = (l_idx == m_idx) & (m_idx == n_idx) & (p_idx != l_idx)

    other = ~(mask_pppp | mask_pqpq | mask_ppqq | mask_qppp)

    # sort by value (descending)
    order         = np.argsort(-flat)         # permutation for descending sort
    flat_sorted   = flat[order]

    # reorder masks (and, if you like, the coordinates) identically
    mask_pppp_s, mask_pqpq_s, mask_ppqq_s, mask_qppp_s, other_s = (
        arr[order] for arr in (mask_pppp, mask_pqpq, mask_ppqq, mask_qppp, other)
    )

    # quick map “flat index  ➜  rank after sort”
    rank_from_flat = np.empty_like(order)
    rank_from_flat[order] = np.arange(order.size)

    nonzero = flat_sorted != 0               # keep only non‑zero values

    # apply the nonzero filter to every mask
    mask_pppp_s &= nonzero
    mask_pqpq_s &= nonzero
    mask_ppqq_s &= nonzero
    mask_qppp_s &= nonzero
    other_s     &= nonzero

    fig, ax = plt.subplots(figsize=(11, 4))

    x = np.arange(flat_sorted.size)           # rank after sorting

    ax.scatter(x[mask_pppp_s], flat_sorted[mask_pppp_s], s=22, c='blue',    label='Intramodal Nonlinearity [p,p,p,p]')
    ax.scatter(x[mask_pqpq_s], flat_sorted[mask_pqpq_s], s=22, c='red',   label='Intermodal XPM [p,q,p,q]')
    ax.scatter(x[mask_ppqq_s], flat_sorted[mask_ppqq_s], s=22, c='red',  label='Intermodal XPM [p,p,q,q]')
    ax.scatter(x[mask_qppp_s], flat_sorted[mask_qppp_s], s=22, c='green', label='Intermodal FWM [q,p,p,p]')
    ax.scatter(x[other_s],     flat_sorted[other_s],     s=6,  c='lightgray', alpha=0.3, label='Intermodal FWM (other)')

    ax.set_xlabel('Overlap Integral Number')
    ax.set_ylabel(r'Q{_plmn}/max(Q{_plmn})')
    ax.set_title('Flattened S_R tensor')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(dir_prefix, f"SK_tensor_flattened_{num_modes}modes.png"), dpi=300)

    positive_qppp = mask_qppp & (flat > 0)    # before sorting: pattern & value>0

    flat_indices_pos   = np.nonzero(positive_qppp)[0]                 # linear
    coords_pos         = np.vstack((p_idx, l_idx, m_idx, n_idx)).T[positive_qppp]
    ranks_pos          = rank_from_flat[flat_indices_pos]             # after sort

    output_file = os.path.join(dir_prefix, f"qppp_entries_{num_modes}modes.txt")
    with open(output_file, 'w') as f:
        f.write("\nEntries with pattern [q,p,p,p] and value > 0\n")
        f.write("---------------------------------------------------------------\n")
        f.write("   flat_idx   rank_after_sort    (p,l,m,n)        value\n")
        f.write("----------------------------- d----------------------------------\n")
        
        # Create a list of tuples (index, rank, coordinates, value) for sorting
        entries = []
        for idx, rank, coord in zip(flat_indices_pos, ranks_pos, coords_pos):
            clean_coord = tuple(int(x) for x in coord)
            entries.append((idx, rank, clean_coord, flat[idx]))
        
        # Sort by value in descending order
        entries.sort(key=lambda x: x[3], reverse=True)
        
        # Write sorted entries
        for idx, rank, coord, value in entries:
            f.write(f"{idx:10d}   {rank:14d}    {coord}   {value: .6f}\n")
        
        f.write("---------------------------------------------------------------\n")
        f.write(f"Total count: {flat_indices_pos.size}\n")

    print(f"Results saved to {output_file}")
