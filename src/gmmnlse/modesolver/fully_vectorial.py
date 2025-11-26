"""
Fully Vectorial Finite Difference Mode Solver

Based on the vectorial finite difference method for anisotropic waveguides.
Reference: Fallahkhair et al., J. Lightwave Technol. 26(11), 1423-1431, (2008).
"""

import jax.numpy as jnp
import jax
import numpy as np
from jax.experimental.sparse import COO, BCOO, CSR, BCSR
from scipy.sparse.linalg import eigs
try:
    import h5py
except ImportError:
    h5py = None  # Optional dependency
import sys
from pathlib import Path

# Support both direct execution and module import
try:
    from .mode import Mode
    from ..constants import C_um_ps
except ImportError:
    # When running directly, add parent to path
    _parent_dir = Path(__file__).parent.parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from gmmnlse.modesolver.mode import Mode
    from gmmnlse.constants import C_um_ps

def centered2d(x):
    return (x[1:, 1:] + x[1:, :-1] + x[:-1, 1:] + x[:-1, :-1]) / 4.

class FullyVectorial:
    """
    Fully-vectorial finite difference mode solver.
    
    Supports anisotropic materials and provides all field components.
    
    Args:
        wavelength (float): Wavelength in micrometers
        x (ndarray): X-coordinates in micrometers
        y (ndarray): Y-coordinates in micrometers
        eps_func (callable): Function eps_func(x, y) returning:
            - Single value: isotropic epsilon
            - Tuple: (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz) for anisotropic
        boundary (str): Boundary conditions '0000' (default)
            Order: North, South, East, West
    """
    
    def __init__(self, wavelength, x, y, eps_func, boundary='0000'):
        """Initialize the fully-vectorial mode solver."""
        self.wavelength = wavelength
        self.frequency = C_um_ps / wavelength  # in THz
        # Swap x and y to match NumPy implementation convention
        # NumPy version does: self.x = structure.y, self.y = structure.x
        self.x = y
        self.y = x
        self.eps_func = eps_func
        self.boundary = boundary
        
        self.n_complexs = None
        self.n_effs = None
        self.modes = None
    
    def _build_matrix(self):
        """
        Build the finite difference matrix for the fully vectorial eigenvalue problem.
        
        Solves for Hx and Hy components.
        
        Returns:
            scipy.sparse matrix: The system matrix
        """
        wl = self.wavelength
        
        x = self.x
        y = self.y
        eps_func = self.eps_func
        boundary = self.boundary
        
        # Calculate grid spacing
        dx = jnp.diff(x)
        dy = jnp.diff(y)

        dx = jnp.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
        dy = jnp.r_[dy[0], dy, dy[-1]].reshape(1, -1)

        # Cell centers
        xc = (x[:-1] + x[1:]) / 2
        yc = (y[:-1] + y[1:]) / 2
        
        # Get permittivity - check if anisotropic
        tmp = eps_func(yc, xc)
        if isinstance(tmp, tuple):
            # Anisotropic: (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)
            tmp = [jnp.c_[t[:, 0:1], t, t[:, -1:]] for t in tmp]
            tmp = [jnp.r_[t[0:1, :], t, t[-1:, :]] for t in tmp]
            epsyy, epsyx, epsxy, epsxx, epszz = tmp
        else:
            # Isotropic
            tmp = jnp.c_[tmp[:, 0:1], tmp, tmp[:, -1:]]
            tmp = jnp.r_[tmp[0:1, :], tmp, tmp[-1:, :]]
            epsxx = epsyy = epszz = tmp
            epsxy = epsyx = jnp.zeros_like(epsxx)

        nx = len(x)
        ny = len(y)
        
        self.nx = nx
        self.ny = ny
        
        k = 2 * jnp.pi / wl
        
        ones_nx = jnp.ones((nx, 1))
        ones_ny = jnp.ones((1, ny))
        
        # Grid spacing
        n = jnp.dot(ones_nx, dy[:, 1:]).flatten()
        s = jnp.dot(ones_nx, dy[:, :-1]).flatten()
        e = jnp.dot(dx[1:, :], ones_ny).flatten()
        w = jnp.dot(dx[:-1, :], ones_ny).flatten()
        
        # Permittivity at staggered grid points
        exx1 = epsxx[:-1, 1:].flatten()
        exx2 = epsxx[:-1, :-1].flatten()
        exx3 = epsxx[1:, :-1].flatten()
        exx4 = epsxx[1:, 1:].flatten()
        
        eyy1 = epsyy[:-1, 1:].flatten()
        eyy2 = epsyy[:-1, :-1].flatten()
        eyy3 = epsyy[1:, :-1].flatten() 
        eyy4 = epsyy[1:, 1:].flatten()
        
        exy1 = epsxy[:-1, 1:].flatten()
        exy2 = epsxy[:-1, :-1].flatten()
        exy3 = epsxy[1:, :-1].flatten()
        exy4 = epsxy[1:, 1:].flatten()
        
        eyx1 = epsyx[:-1, 1:].flatten()
        eyx2 = epsyx[:-1, :-1].flatten()
        eyx3 = epsyx[1:, :-1].flatten()
        eyx4 = epsyx[1:, 1:].flatten()
        
        ezz1 = epszz[:-1, 1:].flatten()
        ezz2 = epszz[:-1, :-1].flatten()
        ezz3 = epszz[1:, :-1].flatten()
        ezz4 = epszz[1:, 1:].flatten()
        
        # Averaged permittivity combinations
        ns21 = n * eyy2 + s * eyy1
        ns34 = n * eyy3 + s * eyy4
        ew14 = e * exx1 + w * exx4
        ew23 = e * exx2 + w * exx3
        
        axxn = ((2 * eyy4 * e - eyx4 * n) * (eyy3 / ezz4) / ns34 +
                (2 * eyy1 * w + eyx1 * n) * (eyy2 / ezz1) / ns21) / (n * (e + w))
        axxs = ((2 * eyy3 * e + eyx3 * s) * (eyy4 / ezz3) / ns34 +
                (2 * eyy2 * w - eyx2 * s) * (eyy1 / ezz2) / ns21) / (s * (e + w))
        ayye = (2 * n * exx4 - e * exy4) * exx1 / ezz4 / e / ew14 / \
            (n + s) + (2 * s * exx3 + e * exy3) * \
            exx2 / ezz3 / e / ew23 / (n + s)
        ayyw = (2 * exx1 * n + exy1 * w) * exx4 / ezz1 / w / ew14 / \
            (n + s) + (2 * exx2 * s - exy2 * w) * \
            exx3 / ezz2 / w / ew23 / (n + s)
        axxe = 2 / (e * (e + w)) + \
            (eyy4 * eyx3 / ezz3 - eyy3 * eyx4 / ezz4) / (e + w) / ns34
        axxw = 2 / (w * (e + w)) + \
            (eyy2 * eyx1 / ezz1 - eyy1 * eyx2 / ezz2) / (e + w) / ns21
        ayyn = 2 / (n * (n + s)) + \
            (exx4 * exy1 / ezz1 - exx1 * exy4 / ezz4) / (n + s) / ew14
        ayys = 2 / (s * (n + s)) + \
            (exx2 * exy3 / ezz3 - exx3 * exy2 / ezz2) / (n + s) / ew23

        axxne = +eyx4 * eyy3 / ezz4 / (e + w) / ns34
        axxse = -eyx3 * eyy4 / ezz3 / (e + w) / ns34
        axxnw = -eyx1 * eyy2 / ezz1 / (e + w) / ns21
        axxsw = +eyx2 * eyy1 / ezz2 / (e + w) / ns21

        ayyne = +exy4 * exx1 / ezz4 / (n + s) / ew14
        ayyse = -exy3 * exx2 / ezz3 / (n + s) / ew23
        ayynw = -exy1 * exx4 / ezz1 / (n + s) / ew14
        ayysw = +exy2 * exx3 / ezz2 / (n + s) / ew23

        axxp = -axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw + k ** 2 * \
            (n + s) * \
            (eyy4 * eyy3 * e / ns34 + eyy1 * eyy2 * w / ns21) / (e + w)
        ayyp = -ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw + k ** 2 * \
            (e + w) * \
            (exx1 * exx4 * n / ew14 + exx2 * exx3 * s / ew23) / (n + s)
        axyn = (eyy3 * eyy4 / ezz4 / ns34 - eyy2 * eyy1 / ezz1 /
                ns21 + s * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
        axys = (eyy1 * eyy2 / ezz2 / ns21 - eyy4 * eyy3 / ezz3 /
                ns34 + n * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
        ayxe = (exx1 * exx4 / ezz4 / ew14 - exx2 * exx3 / ezz3 /
                ew23 + w * (exx2 * exx4 - exx1 * exx3) / ew23 / ew14) / (n + s)
        ayxw = (exx3 * exx2 / ezz2 / ew23 - exx4 * exx1 / ezz1 /
                ew14 + e * (exx4 * exx2 - exx1 * exx3) / ew23 / ew14) / (n + s)

        axye = (eyy4 * (1 + eyy3 / ezz4) - eyy3 * (1 + eyy4 / ezz4)) / ns34 / (e + w) - \
               (2 * eyx1 * eyy2 / ezz1 * n * w / ns21 +
                2 * eyx2 * eyy1 / ezz2 * s * w / ns21 +
                2 * eyx4 * eyy3 / ezz4 * n * e / ns34 +
                2 * eyx3 * eyy4 / ezz3 * s * e / ns34 +
                2 * eyy1 * eyy2 * (1. / ezz1 - 1. / ezz2) * w ** 2 / ns21) / e / (e + w) ** 2

        axyw = (eyy2 * (1 + eyy1 / ezz2) - eyy1 * (1 + eyy2 / ezz2)) / ns21 / (e + w) - \
               (2 * eyx1 * eyy2 / ezz1 * n * e / ns21 +
                2 * eyx2 * eyy1 / ezz2 * s * e / ns21 +
                2 * eyx4 * eyy3 / ezz4 * n * w / ns34 +
                2 * eyx3 * eyy4 / ezz3 * s * w / ns34 +
                2 * eyy3 * eyy4 * (1. / ezz3 - 1. / ezz4) * e ** 2 / ns34) / w / (e + w) ** 2

        ayxn = (exx4 * (1 + exx1 / ezz4) - exx1 * (1 + exx4 / ezz4)) / ew14 / (n + s) - \
               (2 * exy3 * exx2 / ezz3 * e * s / ew23 +
                2 * exy2 * exx3 / ezz2 * w * n / ew23 +
                2 * exy4 * exx1 / ezz4 * e * s / ew14 +
                2 * exy1 * exx4 / ezz1 * w * n / ew14 +
                2 * exx3 * exx2 * (1. / ezz3 - 1. / ezz2) * s ** 2 / ew23) / n / (n + s) ** 2

        ayxs = (exx2 * (1 + exx3 / ezz2) - exx3 * (1 + exx2 / ezz2)) / ew23 / (n + s) - \
               (2 * exy3 * exx2 / ezz3 * e * n / ew23 +
                2 * exy2 * exx3 / ezz2 * w * n / ew23 +
                2 * exy4 * exx1 / ezz4 * e * s / ew14 +
                2 * exy1 * exx4 / ezz1 * w * s / ew14 +
                2 * exx1 * exx4 * (1. / ezz1 - 1. / ezz4) * n ** 2 / ew14) / s / (n + s) ** 2

        axyne = +eyy3 * (1 - eyy4 / ezz4) / (e + w) / ns34
        axyse = -eyy4 * (1 - eyy3 / ezz3) / (e + w) / ns34
        axynw = -eyy2 * (1 - eyy1 / ezz1) / (e + w) / ns21
        axysw = +eyy1 * (1 - eyy2 / ezz2) / (e + w) / ns21

        ayxne = +exx1 * (1 - exx4 / ezz4) / (n + s) / ew14
        ayxse = -exx2 * (1 - exx3 / ezz3) / (n + s) / ew23
        ayxnw = -exx4 * (1 - exx1 / ezz1) / (n + s) / ew14
        ayxsw = +exx3 * (1 - exx2 / ezz2) / (n + s) / ew23

        axyp = -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw) - k ** 2 * (w * (n * eyx1 *
                                                                                             eyy2 + s * eyx2 * eyy1) / ns21 + e * (s * eyx3 * eyy4 + n * eyx4 * eyy3) / ns34) / (e + w)
        ayxp = -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw) - k ** 2 * (n * (w * exy1 *
                                                                                             exx4 + e * exy4 * exx1) / ew14 + s * (w * exy2 * exx3 + e * exy3 * exx2) / ew23) / (n + s)
        ii = jnp.arange(nx * ny).reshape(nx, ny)
        
        # Build sparse matrix for [Hx; Hy] system
        # Matrix is 2*(nx*ny) x 2*(nx*ny)
        # NORTH boundary

        ib = ii[:, -1]

        if boundary[0] == 'S':
            sign = 1
        elif boundary[0] == 'A':
            sign = -1
        elif boundary[0] == '0':
            sign = 0
        else:
            raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

        axxs = axxs.at[ib].add(sign * axxn[ib])
        axxse = axxse.at[ib].add(sign * axxne[ib])
        axxsw = axxsw.at[ib].add(sign * axxnw[ib])
        ayxs = ayxs.at[ib].add(sign * ayxn[ib])
        ayxse = ayxse.at[ib].add(sign * ayxne[ib])
        ayxsw = ayxsw.at[ib].add(sign * ayxnw[ib])
        ayys = ayys.at[ib].add(-sign * ayyn[ib])
        ayyse = ayyse.at[ib].add(-sign * ayyne[ib])
        ayysw = ayysw.at[ib].add(-sign * ayynw[ib])
        axys = axys.at[ib].add(-sign * axyn[ib])
        axyse = axyse.at[ib].add(-sign * axyne[ib])
        axysw = axysw.at[ib].add(-sign * axynw[ib])

        # SOUTH boundary

        ib = ii[:, 0]

        if boundary[1] == 'S':
            sign = 1
        elif boundary[1] == 'A':
            sign = -1
        elif boundary[1] == '0':
            sign = 0
        else:
            raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

        axxn = axxn.at[ib].add(sign * axxs[ib])
        axxne = axxne.at[ib].add(sign * axxse[ib])
        axxnw = axxnw.at[ib].add(sign * axxsw[ib])
        ayxn = ayxn.at[ib].add(sign * ayxs[ib])
        ayxne = ayxne.at[ib].add(sign * ayxse[ib])
        ayxnw = ayxnw.at[ib].add(sign * ayxsw[ib])
        ayyn = ayyn.at[ib].add(-sign * ayys[ib])
        ayyne = ayyne.at[ib].add(-sign * ayyse[ib])
        ayynw = ayynw.at[ib].add(-sign * ayysw[ib])
        axyn = axyn.at[ib].add(-sign * axys[ib])
        axyne = axyne.at[ib].add(-sign * axyse[ib])
        axynw = axynw.at[ib].add(-sign * axysw[ib])

        # EAST boundary

        ib = ii[-1, :]

        if boundary[2] == 'S':
            sign = 1
        elif boundary[2] == 'A':
            sign = -1
        elif boundary[2] == '0':
            sign = 0
        else:
            raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

        axxw = axxw.at[ib].add(sign * axxe[ib])
        axxnw = axxnw.at[ib].add(sign * axxne[ib])
        axxsw = axxsw.at[ib].add(sign * axxse[ib])
        ayxw = ayxw.at[ib].add(sign * ayxe[ib])
        ayxnw = ayxnw.at[ib].add(sign * ayxne[ib])
        ayxsw = ayxsw.at[ib].add(sign * ayxse[ib])
        ayyw = ayyw.at[ib].add(-sign * ayye[ib])
        ayynw = ayynw.at[ib].add(-sign * ayyne[ib])
        ayysw = ayysw.at[ib].add(-sign * ayyse[ib])
        axyw = axyw.at[ib].add(-sign * axye[ib])
        axynw = axynw.at[ib].add(-sign * axyne[ib])
        axysw = axysw.at[ib].add(-sign * axyse[ib])

        # WEST boundary

        ib = ii[0, :]

        if boundary[3] == 'S':
            sign = 1
        elif boundary[3] == 'A':
            sign = -1
        elif boundary[3] == '0':
            sign = 0
        else:
            raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

        axxe = axxe.at[ib].add(sign * axxw[ib])
        axxne = axxne.at[ib].add(sign * axxnw[ib])
        axxse = axxse.at[ib].add(sign * axxsw[ib])
        ayxe = ayxe.at[ib].add(sign * ayxw[ib])
        ayxne = ayxne.at[ib].add(sign * ayxnw[ib])
        ayxse = ayxse.at[ib].add(sign * ayxsw[ib])
        ayye = ayye.at[ib].add(-sign * ayyw[ib])
        ayyne = ayyne.at[ib].add(-sign * ayynw[ib])
        ayyse = ayyse.at[ib].add(-sign * ayysw[ib])
        axye = axye.at[ib].add(-sign * axyw[ib])
        axyne = axyne.at[ib].add(-sign * axynw[ib])
        axyse = axyse.at[ib].add(-sign * axysw[ib])

        # Assemble sparse matrix

        iall = ii.flatten()
        i_s = ii[:, :-1].flatten()
        i_n = ii[:, 1:].flatten()
        i_e = ii[1:, :].flatten()
        i_w = ii[:-1, :].flatten()
        i_ne = ii[1:, 1:].flatten()
        i_se = ii[1:, :-1].flatten()
        i_sw = ii[:-1, :-1].flatten()
        i_nw = ii[:-1, 1:].flatten()

        Ixx = jnp.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
        Jxx = jnp.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
        Vxx = jnp.r_[axxp[iall], axxe[i_w], axxw[i_e], axxn[i_s], axxs[
            i_n], axxsw[i_ne], axxnw[i_se], axxne[i_sw], axxse[i_nw]]

        Ixy = jnp.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
        Jxy = jnp.r_[
            iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
        Vxy = jnp.r_[axyp[iall], axye[i_w], axyw[i_e], axyn[i_s], axys[
            i_n], axysw[i_ne], axynw[i_se], axyne[i_sw], axyse[i_nw]]

        Iyx = jnp.r_[
            iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw] + nx * ny
        Jyx = jnp.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
        Vyx = jnp.r_[ayxp[iall], ayxe[i_w], ayxw[i_e], ayxn[i_s], ayxs[
            i_n], ayxsw[i_ne], ayxnw[i_se], ayxne[i_sw], ayxse[i_nw]]

        Iyy = jnp.r_[
            iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw] + nx * ny
        Jyy = jnp.r_[
            iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
        Vyy = jnp.r_[ayyp[iall], ayye[i_w], ayyw[i_e], ayyn[i_s], ayys[
            i_n], ayysw[i_ne], ayynw[i_se], ayyne[i_sw], ayyse[i_nw]]

        I = jnp.r_[Ixx, Ixy, Iyx, Iyy]
        J = jnp.r_[Jxx, Jxy, Jyx, Jyy]
        V = jnp.r_[Vxx, Vxy, Vyx, Vyy]
        A = COO((V, I, J), shape=(2 * nx * ny, 2 * nx * ny))

        return A

    def compute_other_fields(self, neffs, Hxs, Hys):

        from scipy.sparse import coo_matrix

        wl = self.wavelength
        x = self.x
        y = self.y
        epsfunc = self.eps_func
        boundary = self.boundary

        Hzs = []
        Exs = []
        Eys = []
        Ezs = []
        for neff, Hx, Hy in zip(neffs, Hxs, Hys):

            dx = jnp.diff(x)
            dy = jnp.diff(y)

            dx = jnp.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
            dy = jnp.r_[dy[0], dy, dy[-1]].reshape(1, -1)

            xc = (x[:-1] + x[1:]) / 2
            yc = (y[:-1] + y[1:]) / 2

            tmp = epsfunc(yc, xc)
            if isinstance(tmp, tuple):
                tmp = [jnp.c_[t[:, 0:1], t, t[:, -1:]] for t in tmp]
                tmp = [jnp.r_[t[0:1, :], t, t[-1:, :]] for t in tmp]
                epsxx, epsxy, epsyx, epsyy, epszz = tmp
            else:
                tmp = jnp.c_[tmp[:, 0:1], tmp, tmp[:, -1:]]
                tmp = jnp.r_[tmp[0:1, :], tmp, tmp[-1:, :]]
                epsxx = epsyy = epszz = tmp
                epsxy = epsyx = jnp.zeros_like(epsxx)

            nx = len(x)
            ny = len(y)

            k = 2 * jnp.pi / wl

            ones_nx = jnp.ones((nx, 1))
            ones_ny = jnp.ones((1, ny))

            n = jnp.dot(ones_nx, dy[:, 1:]).flatten()
            s = jnp.dot(ones_nx, dy[:, :-1]).flatten()
            e = jnp.dot(dx[1:, :], ones_ny).flatten()
            w = jnp.dot(dx[:-1, :], ones_ny).flatten()

            exx1 = epsxx[:-1, 1:].flatten()
            exx2 = epsxx[:-1, :-1].flatten()
            exx3 = epsxx[1:, :-1].flatten()
            exx4 = epsxx[1:, 1:].flatten()

            eyy1 = epsyy[:-1, 1:].flatten()
            eyy2 = epsyy[:-1, :-1].flatten()
            eyy3 = epsyy[1:, :-1].flatten()
            eyy4 = epsyy[1:, 1:].flatten()

            exy1 = epsxy[:-1, 1:].flatten()
            exy2 = epsxy[:-1, :-1].flatten()
            exy3 = epsxy[1:, :-1].flatten()
            exy4 = epsxy[1:, 1:].flatten()

            eyx1 = epsyx[:-1, 1:].flatten()
            eyx2 = epsyx[:-1, :-1].flatten()
            eyx3 = epsyx[1:, :-1].flatten()
            eyx4 = epsyx[1:, 1:].flatten()

            ezz1 = epszz[:-1, 1:].flatten()
            ezz2 = epszz[:-1, :-1].flatten()
            ezz3 = epszz[1:, :-1].flatten()
            ezz4 = epszz[1:, 1:].flatten()

            b = neff * k

            bzxne = (0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * eyx4 / ezz4 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy3 * eyy1 * w * eyy2 +
                     0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (1 - exx4 / ezz4) / ezz3 / ezz2 / (w * exx3 + e * exx2) / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx1 * s) / b

            bzxse = (-0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * eyx3 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy1 * w * eyy2 +
                     0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (1 - exx3 / ezz3) / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * n * exx1 * exx4) / b

            bzxnw = (-0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * eyx1 / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy2 * e -
                     0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (1 - exx1 / ezz1) / ezz3 / ezz2 / (w * exx3 + e * exx2) / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx4 * s) / b

            bzxsw = (0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * eyx2 / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * e -
                     0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (1 - exx2 / ezz2) / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx3 * n * exx1 * exx4) / b

            bzxn = ((0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * n * ezz1 * ezz2 / eyy1 * (2 * eyy1 / ezz1 / n ** 2 + eyx1 / ezz1 / n / w) + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * n * ezz4 * ezz3 / eyy4 * (2 * eyy4 / ezz4 / n ** 2 - eyx4 / ezz4 / n / e)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * ((1 - exx1 / ezz1) / n / w - exy1 / ezz1 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (2. / n ** 2 - 2 / n ** 2 * s / (n + s))) / exx1 * ezz1 * w + (ezz4 - ezz1) * s / n / (n + s) + 0.5 * ezz1 * (-(1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (2. / n ** 2 - 2 / n ** 2 * s / (n + s))) / exx4 * ezz4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (-ezz3 * exy2 / n / (n + s) / exx2 * w + (ezz3 - ezz2) * s / n / (n + s) - ezz2 * exy3 / n / (n + s) / exx3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxs = ((0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * s * ezz2 * ezz1 / eyy2 * (2 * eyy2 / ezz2 / s ** 2 - eyx2 / ezz2 / s / w) + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * s * ezz3 * ezz4 / eyy3 * (2 * eyy3 / ezz3 / s ** 2 + eyx3 / ezz3 / s / e)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (-ezz4 * exy1 / s / (n + s) / exx1 * w - (ezz4 - ezz1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   * n / s / (n + s) - ezz1 * exy4 / s / (n + s) / exx4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (-(1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (2. / s ** 2 - 2 / s ** 2 * n / (n + s))) / exx2 * ezz2 * w - (ezz3 - ezz2) * n / s / (n + s) + 0.5 * ezz2 * ((1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (2. / s ** 2 - 2 / s ** 2 * n / (n + s))) / exx3 * ezz3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxe = ((n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (2. / e ** 2 - eyx4 / ezz4 / n / e) + 0.5 * s * ezz3 * ezz4 / eyy3 * (2. / e ** 2 + eyx3 / ezz3 / s / e)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
                    (-0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz1 * (1 - exx4 / ezz4) / n / exx4 * ezz4 - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz2 * (1 - exx3 / ezz3) / s / exx3 * ezz3) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxw = ((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * (2. / w ** 2 + eyx1 / ezz1 / n / w) + 0.5 * s * ezz2 * ezz1 / eyy2 * (2. / w ** 2 - eyx2 / ezz2 / s / w)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
                    (0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz4 * (1 - exx1 / ezz1) / n / exx1 * ezz1 + 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz3 * (1 - exx2 / ezz2) / s / exx2 * ezz2) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxp = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * (-2. / w ** 2 - 2 * eyy1 / ezz1 / n ** 2 + k ** 2 * eyy1 - eyx1 / ezz1 / n / w) + 0.5 * s * ezz2 * ezz1 / eyy2 * (-2. / w ** 2 - 2 * eyy2 / ezz2 / s ** 2 + k ** 2 * eyy2 + eyx2 / ezz2 / s / w)) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (-2. / e ** 2 - 2 * eyy4 / ezz4 / n ** 2 + k ** 2 * eyy4 + eyx4 / ezz4 / n / e) + 0.5 * s * ezz3 * ezz4 / eyy3 * (-2. / e ** 2 - 2 * eyy3 / ezz3 / s ** 2 + k ** 2 * eyy3 - eyx3 / ezz3 / s / e))) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * (-k **
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     2 * exy1 - (1 - exx1 / ezz1) / n / w - exy1 / ezz1 * (-2. / n ** 2 - 2 / n ** 2 * (n - s) / s)) / exx1 * ezz1 * w + (ezz4 - ezz1) * (n - s) / n / s + 0.5 * ezz1 * (-k ** 2 * exy4 + (1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (-2. / n ** 2 - 2 / n ** 2 * (n - s) / s)) / exx4 * ezz4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (-k ** 2 * exy2 + (1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (-2. / s ** 2 + 2 / s ** 2 * (n - s) / n)) / exx2 * ezz2 * w + (ezz3 - ezz2) * (n - s) / n / s + 0.5 * ezz2 * (-k ** 2 * exy3 - (1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (-2. / s ** 2 + 2 / s ** 2 * (n - s) / n)) / exx3 * ezz3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzyne = (0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (1 - eyy4 / ezz4) / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy3 * eyy1 * w *
                     eyy2 + 0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * exy4 / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx1 * s) / b

            bzyse = (-0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (1 - eyy3 / ezz3) / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy1 * w *
                     eyy2 + 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * exy3 / ezz3 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * n * exx1 * exx4) / b

            bzynw = (-0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (1 - eyy1 / ezz1) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 *
                     eyy2 * e - 0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * exy1 / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx4 * s) / b

            bzysw = (0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (1 - eyy2 / ezz2) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 *
                     e - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * exy2 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx3 * n * exx1 * exx4) / b

            bzyn = ((0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * ezz1 * ezz2 / eyy1 * (1 - eyy1 / ezz1) / w - 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * ezz4 * ezz3 / eyy4 * (1 - eyy4 / ezz4) / e) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w *
                    eyy2 * e + (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * (2. / n ** 2 + exy1 / ezz1 / n / w) / exx1 * ezz1 * w + 0.5 * ezz1 * (2. / n ** 2 - exy4 / ezz4 / n / e) / exx4 * ezz4 * e) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzys = ((-0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * ezz2 * ezz1 / eyy2 * (1 - eyy2 / ezz2) / w + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * ezz3 * ezz4 / eyy3 * (1 - eyy3 / ezz3) / e) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w *
                    eyy2 * e - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (2. / s ** 2 - exy2 / ezz2 / s / w) / exx2 * ezz2 * w + 0.5 * ezz2 * (2. / s ** 2 + exy3 / ezz3 / s / e) / exx3 * ezz3 * e) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzye = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (-n * ezz2 / eyy1 * eyx1 / e / (e + w) + (ezz1 - ezz2) * w / e / (e + w) - s * ezz1 / eyy2 * eyx2 / e / (e + w)) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (-(1 - eyy4 / ezz4) / n / e - eyx4 / ezz4 * (2. / e ** 2 - 2 / e ** 2 * w / (e + w))) + 0.5 * s * ezz3 * ezz4 / eyy3 * ((1 - eyy3 / ezz3) / s / e - eyx3 / ezz3 * (2. / e ** 2 - 2 / e ** 2 * w / (e + w))) + (ezz4 - ezz3) * w / e / (e + w))) / ezz4 /
                    ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + (0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz1 * (2 * exx4 / ezz4 / e ** 2 - exy4 / ezz4 / n / e) / exx4 * ezz4 * e - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz2 * (2 * exx3 / ezz3 / e ** 2 + exy3 / ezz3 / s / e) / exx3 * ezz3 * e) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzyw = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * ((1 - eyy1 / ezz1) / n / w - eyx1 / ezz1 * (2. / w ** 2 - 2 / w ** 2 * e / (e + w))) - (ezz1 - ezz2) * e / w / (e + w) + 0.5 * s * ezz2 * ezz1 / eyy2 * (-(1 - eyy2 / ezz2) / s / w - eyx2 / ezz2 * (2. / w ** 2 - 2 / w ** 2 * e / (e + w)))) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (-n * ezz3 / eyy4 * eyx4 / w / (e + w) - s * ezz4 / eyy3 * eyx3 / w / (e + w) - (ezz4 - ezz3) * e / w / (e + w))) / ezz4 /
                    ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + (0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz4 * (2 * exx1 / ezz1 / w ** 2 + exy1 / ezz1 / n / w) / exx1 * ezz1 * w - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz3 * (2 * exx2 / ezz2 / w ** 2 - exy2 / ezz2 / s / w) / exx2 * ezz2 * w) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzyp = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * (-k ** 2 * eyx1 - (1 - eyy1 / ezz1) / n / w - eyx1 / ezz1 * (-2. / w ** 2 + 2 / w ** 2 * (e - w) / e)) + (ezz1 - ezz2) * (e - w) / e / w + 0.5 * s * ezz2 * ezz1 / eyy2 * (-k ** 2 * eyx2 + (1 - eyy2 / ezz2) / s / w - eyx2 / ezz2 * (-2. / w ** 2 + 2 / w ** 2 * (e - w) / e))) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (-k ** 2 * eyx4 + (1 - eyy4 / ezz4) / n / e - eyx4 / ezz4 * (-2. / e ** 2 - 2 / e ** 2 * (e - w) / w)) + 0.5 * s * ezz3 * ezz4 / eyy3 * (-k ** 2 * eyx3 - (1 - eyy3 / ezz3) / s / e - eyx3 / ezz3 * (-2. / e ** 2 - 2 / e ** 2 * (e - w) / w)) + (ezz4 - ezz3) * (e - w) / e / w)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) /
                    ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * (-2. / n ** 2 - 2 * exx1 / ezz1 / w ** 2 + k ** 2 * exx1 - exy1 / ezz1 / n / w) / exx1 * ezz1 * w + 0.5 * ezz1 * (-2. / n ** 2 - 2 * exx4 / ezz4 / e ** 2 + k ** 2 * exx4 + exy4 / ezz4 / n / e) / exx4 * ezz4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (-2. / s ** 2 - 2 * exx2 / ezz2 / w ** 2 + k ** 2 * exx2 + exy2 / ezz2 / s / w) / exx2 * ezz2 * w + 0.5 * ezz2 * (-2. / s ** 2 - 2 * exx3 / ezz3 / e ** 2 + k ** 2 * exx3 - exy3 / ezz3 / s / e) / exx3 * ezz3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            ii = jnp.arange(nx * ny).reshape(nx, ny)

            # NORTH boundary

            ib = ii[:, -1]

            if boundary[0] == 'S':
                sign = 1
            elif boundary[0] == 'A':
                sign = -1
            elif boundary[0] == '0':
                sign = 0
            else:
                raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

            bzxs = bzxs.at[ib].add(sign * bzxn[ib])
            bzxse = bzxse.at[ib].add(sign * bzxne[ib])
            bzxsw = bzxsw.at[ib].add(sign * bzxnw[ib])
            bzys = bzys.at[ib].add(-sign * bzyn[ib])
            bzyse = bzyse.at[ib].add(-sign * bzyne[ib])
            bzysw = bzysw.at[ib].add(-sign * bzynw[ib])

            # SOUTH boundary

            ib = ii[:, 0]

            if boundary[1] == 'S':
                sign = 1
            elif boundary[1] == 'A':
                sign = -1
            elif boundary[1] == '0':
                sign = 0
            else:
                raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

            bzxn = bzxn.at[ib].add(sign * bzxs[ib])
            bzxne = bzxne.at[ib].add(sign * bzxse[ib])
            bzxnw = bzxnw.at[ib].add(sign * bzxsw[ib])
            bzyn = bzyn.at[ib].add(-sign * bzys[ib])
            bzyne = bzyne.at[ib].add(-sign * bzyse[ib])
            bzynw = bzynw.at[ib].add(-sign * bzysw[ib])

            # EAST boundary

            ib = ii[-1, :]

            if boundary[2] == 'S':
                sign = 1
            elif boundary[2] == 'A':
                sign = -1
            elif boundary[2] == '0':
                sign = 0
            else:
                raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

            bzxw = bzxw.at[ib].add(sign * bzxe[ib])
            bzxnw = bzxnw.at[ib].add(sign * bzxne[ib])
            bzxsw = bzxsw.at[ib].add(sign * bzxse[ib])
            bzyw = bzyw.at[ib].add(-sign * bzye[ib])
            bzynw = bzynw.at[ib].add(-sign * bzyne[ib])
            bzysw = bzysw.at[ib].add(-sign * bzyse[ib])

            # WEST boundary

            ib = ii[0, :]

            if boundary[3] == 'S':
                sign = 1
            elif boundary[3] == 'A':
                sign = -1
            elif boundary[3] == '0':
                sign = 0
            else:
                raise ValueError('Invalid boundary condition. Available options are S, A, 0.')

            bzxe = bzxe.at[ib].add(sign * bzxw[ib])
            bzxne = bzxne.at[ib].add(sign * bzxnw[ib])
            bzxse = bzxse.at[ib].add(sign * bzxsw[ib])
            bzye = bzye.at[ib].add(-sign * bzyw[ib])
            bzyne = bzyne.at[ib].add(-sign * bzynw[ib])
            bzyse = bzyse.at[ib].add(-sign * bzysw[ib])

            # Assemble sparse matrix

            iall = ii.flatten()
            i_s = ii[:, :-1].flatten()
            i_n = ii[:, 1:].flatten()
            i_e = ii[1:, :].flatten()
            i_w = ii[:-1, :].flatten()
            i_ne = ii[1:, 1:].flatten()
            i_se = ii[1:, :-1].flatten()
            i_sw = ii[:-1, :-1].flatten()
            i_nw = ii[:-1, 1:].flatten()

            Izx = jnp.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
            Jzx = jnp.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
            Vzx = jnp.r_[bzxp[iall], bzxe[i_w], bzxw[i_e], bzxn[i_s], bzxs[
                i_n], bzxsw[i_ne], bzxnw[i_se], bzxne[i_sw], bzxse[i_nw]]

            Izy = jnp.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
            Jzy = jnp.r_[
                iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
            Vzy = jnp.r_[bzyp[iall], bzye[i_w], bzyw[i_e], bzyn[i_s], bzys[
                i_n], bzysw[i_ne], bzynw[i_se], bzyne[i_sw], bzyse[i_nw]]

            I = jnp.r_[Izx, Izy]
            J = jnp.r_[Jzx, Jzy]
            V = jnp.r_[Vzx, Vzy]
            B = COO((V, I, J), shape=(2 * nx * ny, 2 * nx * ny))

            HxHy = jnp.r_[Hx, Hy]
            # JAX COO doesn't support * operator, use @ (matmul) instead
            # B @ HxHy returns [Hz_from_Hx; Hz_from_Hy], but we only need Hz
            Hz_full = (B @ HxHy.ravel()) / 1j
            # Take first nx*ny elements and reshape to (nx, ny)
            Hz = Hz_full[:nx * ny].reshape(nx, ny)

            # in xc e yc
            # Permittivity arrays are (ny, nx) after padding and slicing
            # but field arrays are (nx-1, ny-1), so we need to transpose
            exx = epsxx[1:-1, 1:-1].T
            exy = epsxy[1:-1, 1:-1].T
            eyx = epsyx[1:-1, 1:-1].T
            eyy = epsyy[1:-1, 1:-1].T
            ezz = epszz[1:-1, 1:-1].T
            edet = (exx * eyy - exy * eyx)

            h = e.reshape(nx, ny)[:-1, :-1]
            v = n.reshape(nx, ny)[:-1, :-1]

            # in xc e yc
            Dx = neff * centered2d(Hy) + (
                Hz[:-1, 1:] + Hz[1:, 1:] - Hz[:-1, :-1] - Hz[1:, :-1]) / (2j * k * v)
            Dy = -neff * centered2d(Hx) - (
                Hz[1:, :-1] + Hz[1:, 1:] - Hz[:-1, 1:] - Hz[:-1, :-1]) / (2j * k * h)
            Dz = ((Hy[1:, :-1] + Hy[1:, 1:] - Hy[:-1, 1:] - Hy[:-1, :-1]) / (2 * h) -
                  (Hx[:-1, 1:] + Hx[1:, 1:] - Hx[:-1, :-1] - Hx[1:, :-1]) / (2 * v)) / (1j * k)

            Ex = (eyy * Dx - exy * Dy) / edet
            Ey = (exx * Dy - eyx * Dx) / edet
            Ez = Dz / ezz

            # Store Hz directly (not centered) to match NumPy implementation
            Hzs.append(Hz)
            Exs.append(Ex)
            Eys.append(Ey)
            Ezs.append(Ez)

        return (Hzs, Exs, Eys, Ezs)

    def solve(self, n_modes=4, tol=0, guess=None, mode_profiles=True, initial_mode_guess=None, file_name=None):
        """
        This function finds the eigenmodes.

        Parameters
        ----------
        n_modes : int
            number of eigenmodes to find
        tol : float
            Relative accuracy for eigenvalues. The default value of 0 implies machine precision.
        guess : float
            a guess for the refractive index. Only finds eigenvectors with an effective refractive index
            higher than this value.

        Returns
        -------
        self : 
        """
        from ..utils import eigs 

        self.nmodes = n_modes
        self.tol = tol

        A = self._build_matrix() # This is your (N, N) sparse matrix

        if guess is not None:
            # calculate shift for eigs function
            k = 2 * jnp.pi / self.wavelength
            shift = (guess * k) ** 2
            
            # This is the inverse operator Op(v) = (A - sigma*I)^-1 @ v
            # Use JAX's built-in GMRES solver
            # This handles JAX tracing correctly and avoids closure issues
            from jax.scipy.sparse.linalg import gmres as jax_gmres
            
            # Create shifted operator
            # Pass A and shift explicitly to avoid closure issues
            # This matches NumPy implementation which uses SciPy's eigs with sigma parameter
            def A_shifted(v):
                # A and shift are captured from outer scope
                # This is consistent with NumPy implementation
                return A @ v - shift * v
            
            # This is the inverse operator Op(v) = (A - sigma*I)^-1 @ v
            # JAX's GMRES handles JIT compilation internally
            # Use optimized parameters for accuracy while maintaining reasonable speed
            # This matches NumPy implementation: SciPy's eigs with sigma uses shift-invert internally
            def Op_inverse(v, *args):
                # args is not used here, but kept for compatibility with eigs interface
                b = v
                x0 = jnp.zeros_like(v)
                
                # Use JAX's built-in GMRES with optimized parameters
                # The solver is called many times, so we need good convergence
                # Use tighter tolerance and more iterations for accuracy
                # This is the inner solver, but accuracy is critical for correct eigenvalues
                x, _ = jax_gmres(
                    A=A_shifted,
                    b=b,
                    x0=x0,
                    tol=1e-6,                   # Tighter tolerance for accuracy
                    atol=1e-8,                  # Absolute tolerance
                    restart=30,                 # Larger restart for better convergence
                    maxiter=10,                 # More restarts for accuracy (GMRES is inner solver)
                    solve_method='batched'      # Use batched method for better GPU performance
                )
                return x
            
            # --- This is the new logic ---
            op_to_use = Op_inverse
            op_args = [A, shift]  # args kept for compatibility, but not used in Op_inverse
            which = 'LM' # Find largest magnitude eigenvalues of the *inverse*
            # Note: For shift-invert, we want eigenvalues of (A - sigma*I)^-1
            # The largest magnitude eigenvalues correspond to eigenvalues of A closest to sigma
        
        else:
            # --- Original logic (no shift) ---
            # eigs.eigs expects a callable, so we wrap the matrix
            def matvec(v, *args):
                return args[0] @ v
                
            op_to_use = matvec
            op_args = [A]
            which = 'LR' # Find largest eigenvalues (default)
            shift = None # No shift-invert

        # Get shape and dtype for the eigs solver
        N = 2 * self.nx * self.ny # Based on your eigvecs slicing
        op_shape = (N,)
        op_dtype = A.dtype

        [inverse_eigvals, eigvecs_list] = eigs.eigs(
            A=op_to_use,
            args=op_args,
            shape=op_shape,
            dtype=op_dtype,
            numeig=n_modes,
            tol=tol,
            which=which,
            maxiter=25
        )
        
        # Convert eigvecs list to array ---
        eigvecs = jnp.stack(eigvecs_list, axis=1)

        # Transform eigenvalues back ---
        if shift is not None:
            # We found eigenvalues w of (A - sigma*I)^-1
            # If w is an eigenvalue of (A - sigma*I)^-1, then:
            #   w = 1 / (lambda - sigma)  where lambda is an eigenvalue of A
            # So: lambda = 1 / w + sigma
            # This matches NumPy implementation which uses SciPy's eigs with sigma parameter
            eigvals = (1.0 / inverse_eigvals) + shift
        else:
            eigvals = inverse_eigvals

        # Convert eigenvalues to effective refractive indices
        # This matches NumPy: neffs = self.wl * numpy.sqrt(eigvals) / (2 * numpy.pi)
        n_complex = self.wavelength * jnp.sqrt(eigvals) / (2 * jnp.pi)
        n_effs = n_complex.real

        if mode_profiles:
            Hxs = []
            Hys = []
            nx = self.nx
            ny = self.ny
            for ieig in range(n_modes):
                Hxs.append(eigvecs[:nx * ny, ieig].reshape(nx, ny))
                Hys.append(eigvecs[nx * ny:, ieig].reshape(nx, ny))

        # sort the modes
        idx = jnp.flipud(jnp.argsort(n_effs))
        self.n_effs = n_effs[idx]
        self.n_complexs = n_complex[idx]

        if mode_profiles:
            tmpx = []
            tmpy = []
            for i in idx:
                tmpx.append(Hxs[i])
                tmpy.append(Hys[i])
            Hxs = tmpx
            Hys = tmpy

            [Hzs, Exs, Eys, Ezs] = self.compute_other_fields(n_effs, Hxs, Hys)

            # Save to mode object
            # JAX doesn't support object arrays, use Python list instead
            self.modes = []

            for i in range(n_modes):
                # Convert JAX arrays to NumPy arrays for Mode class (which uses in-place operations)
                # Store Hx, Hy, Hz directly (not centered) to match NumPy implementation
                mode = Mode(self.wavelength, self.frequency, 
                           np.array(self.x), np.array(self.y),  # Convert coordinates
                           float(self.n_effs[i]), complex(self.n_complexs[i]),  # Convert scalars
                           np.array(Exs[i]), np.array(Eys[i]), np.array(Ezs[i]),  # Convert E fields
                           np.array(Hxs[i]), np.array(Hys[i]), np.array(Hzs[i]))  # Convert H fields
                # Normalize to match NumPy implementation
                mode.normalize()
                self.modes.append(mode)
        else:
            # No mode profiles requested
            self.modes = []
        
        return self.modes
    
    def __repr__(self):
        return f"ModeSolverFullyVectorial(wavelength={self.wavelength:.4f} μm)"

