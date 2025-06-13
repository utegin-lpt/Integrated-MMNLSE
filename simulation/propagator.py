import numpy as np
import matplotlib.pyplot as plt
def hann_boundary(time_grid) -> np.ndarray:
    window = 0.5 * (1 + np.cos(2 * np.pi * time_grid/time_grid[-1]))
    window[np.abs(time_grid) > (time_grid[-1]/2)] = 0
    plt.plot(time_grid, window)
    plt.title('Hann Window')
    plt.xlabel('Time (s)')
    plt.show()
    return window

def super_gauss(t: np.ndarray, p=60):

    """define normalized super Gaussian pulse with FWHM w and exponent p"""
    gaussian_boundary = np.exp(-2*((np.sqrt(t**2)/((t.max()-1e-12)))**p))
    plt.plot(t, gaussian_boundary)
    plt.title('Super Gaussian Boundary')
    plt.xlabel('Time (s)')
    plt.show()

    return gaussian_boundary

def square_window(time_grid, n_boundary):
    window = np.ones_like(time_grid)
    window[:n_boundary] = np.random.random(n_boundary)
    window[-n_boundary:] = np.random.random(n_boundary)
    return window

def RK4IP(field: np.ndarray, DFR: np.ndarray, nonlinear_coeff, nonzero_idx_plmn, SR, dz: float,freq_mask=None, temporal_mask=None, self_steepening = None) -> np.ndarray:
    field = D_operator(field, DFR, dz/2, freq_mask)
    
    k1 = NL_operator(field, nonlinear_coeff, nonzero_idx_plmn, SR, dz, self_steepening)
    k2 = NL_operator(field + k1/2, nonlinear_coeff, nonzero_idx_plmn, SR, dz, self_steepening)
    k3 = NL_operator(field + k2/2, nonlinear_coeff, nonzero_idx_plmn, SR, dz, self_steepening)
    k4 = NL_operator(field + k3, nonlinear_coeff, nonzero_idx_plmn, SR, dz, self_steepening)

    field = field + (k1 + 2*k2 + 2*k3 + k4)/ 6

    field = D_operator(field, DFR, dz/2, freq_mask)
    if temporal_mask is not None:
        field = field * temporal_mask
    return field

def RK2(field: np.ndarray, DFR, nonlinear_coeff, nonzero_idx_plmn, SR, dz: float) -> np.ndarray:
    field = D_operator(field, DFR, dz/2)
    
    k1 = NL_operator(field, nonlinear_coeff, nonzero_idx_plmn, SR, dz)
    k2 = NL_operator(field + k1, nonlinear_coeff, nonzero_idx_plmn, SR, dz)

    field = field + (k1 + k2) / 2

    field = D_operator(field, DFR, dz/2)

    return field

def D_operator(field: np.ndarray, DFR, dz: float, mask=None) -> np.ndarray:
    field_D = np.fft.fftshift(np.fft.fft(field), axes= -1)
    field_D = field_D * np.exp(DFR * dz) 
    if mask is not None:
        field_D = field_D * mask
    field_D = np.fft.ifft(np.fft.ifftshift(field_D, axes = -1))
    return field_D

def NL_operator(field: np.ndarray, nonlinear_coeff, nonzero_idx_plmn, SR, dz: float, self_steepening=None) -> np.ndarray:
    # eta_mn containes the nonlinear terms for interaction between mn, with shape (num_modes, num_modes, num_time_points)
    # eta_mn = np.zeros((field.shape[-2], field.shape[-2], field.shape[-1]), dtype=complex)
    # eta_p containes the nonlinear terms for each mode, with shape (num_modes, num_time_points)
    eta_p = np.zeros((field.shape[-2], field.shape[-1]), dtype=complex)
    # eta_pl = np.zeros((field.shape[-2], field.shape[-2], field.shape[-1]), dtype=complex)

    # for idx in range(nonzero_idx_mn.shape[1]):
    #     m = nonzero_idx_mn[0, idx]
    #     n = nonzero_idx_mn[1, idx]
    #     eta_mn[m, n, :] = field[m, :] * np.conj(field[n, :])

    for idx in range(nonzero_idx_plmn.shape[1]):
        p = nonzero_idx_plmn[0, idx]
        l = nonzero_idx_plmn[1, idx]
        m = nonzero_idx_plmn[2, idx]
        n = nonzero_idx_plmn[3, idx]

        eta_p[p, :] += SR[idx] * \
                        field[l, :] * field[m, :] * np.conj(field[n, :])
        
        # eta_pl[p, l, :] += SR[idx] * eta_mn[:, m, n] 
    
    field_NL = 1j * nonlinear_coeff * eta_p
    if self_steepening is not None:
        field_NL_freq = np.fft.fftshift(np.fft.fft(field_NL), axes=-1)
        field_NL_freq = field_NL_freq * self_steepening
        field_NL = np.fft.ifft(np.fft.ifftshift(field_NL_freq, axes=-1))

    return field_NL  * dz
