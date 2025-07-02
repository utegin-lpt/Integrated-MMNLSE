import jax.numpy as jnp
import jax

# Set jax to use 64-bit floating point numbers
jax.config.update("jax_enable_x64", True)

def super_gauss(t: jnp.ndarray, p=60):
    """define normalized super Gaussian pulse with FWHM w and exponent p"""
    gaussian_boundary = jnp.exp(-2*((jnp.sqrt(t**2)/((t.max()-1e-12)))**p))

    return gaussian_boundary

# @jax.jit
def RK4IP(field: jnp.ndarray, DFR: jnp.ndarray, nonlinear_coeff, nonzero_idx_plmn, SR, dz: float,freq_mask=None, temporal_mask=None, self_steepening = None) -> jnp.ndarray:
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

@jax.jit
def RK2(field: jnp.ndarray, DFR, nonlinear_coeff, nonzero_idx_plmn, SR, dz: float) -> jnp.ndarray:
    field = D_operator(field, DFR, dz/2)
    
    k1 = NL_operator(field, nonlinear_coeff, nonzero_idx_plmn, SR, dz)
    k2 = NL_operator(field + k1, nonlinear_coeff, nonzero_idx_plmn, SR, dz)

    field = field + (k1 + k2) / 2

    field = D_operator(field, DFR, dz/2)

    return field

@jax.jit
def SSFM(field: jnp.ndarray, DFR, nonlinear_coeff, nonzero_idx_plmn, SR, dz: float) -> jnp.ndarray:
    field = D_operator(field, DFR, dz/2)

    field = NL_operator(field, nonlinear_coeff, nonzero_idx_plmn, SR, dz)

    field = D_operator(field, DFR, dz/2)

    return field

@jax.jit
def D_operator(field: jnp.ndarray, DFR, dz: float, mask=None) -> jnp.ndarray:
    field_D = jnp.fft.fftshift(jnp.fft.fft(field), axes= -1)
    field_D = field_D * jnp.exp(DFR * dz) 
    if mask is not None:
        field_D = field_D * mask
    field_D = jnp.fft.ifft(jnp.fft.ifftshift(field_D, axes = -1))
    return field_D


@jax.jit
def NL_operator(field: jnp.ndarray,
                nonlinear_coeff,
                nonzero_idx_plmn,
                SR,
                dz: float,
                self_steepening=None) -> jnp.ndarray:
    eta_p = jnp.zeros((field.shape[-2], field.shape[-1]), dtype=jnp.complex128)
    # unpack indices
    p_idx, l_idx, m_idx, n_idx = nonzero_idx_plmn
    # vectorized contribs
    contribs = SR[:, None] * field[l_idx] * field[m_idx] * jnp.conj(field[n_idx])
    # scatter once
    eta_p = eta_p.at[p_idx].add(contribs)

    field_NL = 1j * nonlinear_coeff * eta_p
    if self_steepening is not None:
        F = jnp.fft.fftshift(jnp.fft.fft(field_NL), axes=-1)
        F = F * self_steepening
        field_NL = jnp.fft.ifft(jnp.fft.ifftshift(F, axes=-1))

    return field_NL * dz
