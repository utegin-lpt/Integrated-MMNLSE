import jax.numpy as jnp
from math import factorial
import numpy as np
import jax
# Set jax to use 64-bit floating point numbers
jax.config.update("jax_enable_x64", True)
class Waveguide:
    def __init__(self, material, betas: jnp.ndarray, etas: jnp.ndarray):
        self.material = material
        self.betas = betas
        self.etas = etas
        self.num_modes = betas.shape[1]
        if material == "SiN":
            self.n2 = 2.4e-19
        elif material == "Si":
            self.n2 = 5.6e-18
        else:
            raise ValueError("Material not supported. Choose 'SiN' or 'Si'.")
        
        p_idx, l_idx, m_idx, n_idx = np.nonzero(np.array(self.etas))
        sr_vals = self.etas[p_idx, l_idx, m_idx, n_idx]

        self.nonzero_idx_plmn = jnp.stack([p_idx, l_idx, m_idx, n_idx], axis=0).astype(jnp.uint32)
        self.nonzero_idx_mn = jnp.unique(np.stack([p_idx, l_idx], axis=0).T, axis=0).T.astype(np.uint32)
        self.SR = sr_vals.astype(jnp.complex128)


        # number of Taylor orders
        N_ord = self.betas.shape[0]
        # build an array [0,1,2,...,N_ord-1]
        self.orders = jnp.arange(N_ord)
        # factorials as a JAX array
        self.fact = jnp.array([factorial(o) for o in self.orders])

    def get_dispersion(self, omegas: jnp.ndarray):
        # subtract the reference β₀₀ and β₁₀ from the 0th and 1st order
        betas0 = self.betas[0] - jnp.real(self.betas[0, 0])
        betas1 = self.betas[1] - jnp.real(self.betas[1, 0])
        betas_adj = self.betas.at[0].set(betas0).at[1].set(betas1)
        # divide each order by its factorial
        coeffs = betas_adj / self.fact[:, None]              # shape (orders, modes)
        # build omegas**order for each order
        omega_p = omegas[None, :] ** self.orders[:, None]         # shape (orders, Nω)
        # sum over orders: result is (modes, Nω)
        DFR = 1j * jnp.einsum('om,on->mn', coeffs, omega_p)
        return DFR
    
    def get_etas(self):
        return self.nonzero_idx_plmn, self.nonzero_idx_mn, self.SR
