import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import time
def wgmodes(lambda_, guess, nmodes, dx, dy, eps, boundary, field):
    """
    Calculates the modes of a dielectric waveguide using the semivectorial finite difference method.
    Adopted from MATLAB code by T.E Murphy, et. al. Original code can be found at:
    https://photonics.umd.edu/software/wgmodes/
    The original paper can be found at:
    A. B. Fallahkhair, K. S. Li and T. E. Murphy, “Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides“, J. Lightwave Technol. 26(11), 1423-1431, (2008).
    
    Parameters:
      lambda_  : optical wavelength.
      guess    : scalar shift to apply when calculating the eigenvalues.
      nmodes   : number of modes to calculate.
      dx       : horizontal grid spacing (scalar or 1D array).
      dy       : vertical grid spacing (scalar or 1D array).
      eps      : index mesh (n^2(x,y)), a 2D NumPy array.
      boundary : 4-letter string specifying boundary conditions in order [North, South, East, West].
      field    : 'ex', 'ey', or 'scalar'
      
    Returns:
      phi  : 3D array (nx x ny x nmodes) containing each computed mode (normalized).
      neff : vector of modal effective indices.
    """
    # Enforce uppercase for boundary
    boundary = boundary.upper()
    nx, ny = eps.shape

    # Pad eps on all sides by one grid point (edge replication does the same as MATLAB padding)
    eps = np.pad(eps, pad_width=1, mode='edge')
    
    # Free-space wavevector
    k = 2 * np.pi / lambda_
    
    # Process dx: if scalar, make an array; if array, pad with first and last elements.
    if np.isscalar(dx):
        dx = np.full(nx+2, dx)
    else:
        dx = np.asarray(dx).flatten()
        dx = np.r_[dx[0], dx, dx[-1]]
        
    # Process dy similarly.
    if np.isscalar(dy):
        dy = np.full(ny+2, dy)
    else:
        dy = np.asarray(dy).flatten()
        dy = np.r_[dy[0], dy, dy[-1]]
    
    # Construct grid-dependent variables.
    # For the vertical directions: note padded dy has length ny+2.
    n_val = (dy[2:ny+2] + dy[1:ny+1]) / 2
    s_val = (dy[0:ny] + dy[1:ny+1]) / 2
    n_mat = np.tile(n_val, (nx, 1))
    s_mat = np.tile(s_val, (nx, 1))
    
    # For horizontal directions: padded dx has length nx+2.
    e_val = (dx[2:nx+2] + dx[1:nx+1]) / 2
    w_val = (dx[0:nx] + dx[1:nx+1]) / 2
    e_mat = np.tile(e_val.reshape(nx, 1), (1, ny))
    w_mat = np.tile(w_val.reshape(nx, 1), (1, ny))
    
    # p and q on the interior
    p_val = dx[1:nx+1]
    p_mat = np.tile(p_val.reshape(nx, 1), (1, ny))
    q_val = dy[1:ny+1]
    q_mat = np.tile(q_val, (nx, 1))
    
    # Extract eps components from padded eps.
    en = eps[1:nx+1, 2:ny+2]
    es = eps[1:nx+1, 0:ny]
    ee = eps[2:nx+2, 1:ny+1]
    ew = eps[0:nx, 1:ny+1]
    ep = eps[1:nx+1, 1:ny+1]
    
    field = field.lower()
    if field == 'ex':
        an = 2 / (n_mat * (n_mat + s_mat))
        _as = 2 / (s_mat * (n_mat + s_mat))
        ae = (8 * (p_mat*(ep-ew) + 2*w_mat*ew) * ee /
              ((p_mat*(ep-ee) + 2*e_mat*ee) * (p_mat**2*(ep-ew) + 4*w_mat**2*ew) +
               (p_mat*(ep-ew) + 2*w_mat*ew) * (p_mat**2*(ep-ee) + 4*e_mat**2*ee)))
        aw = (8 * (p_mat*(ep-ee) + 2*e_mat*ee) * ew /
              ((p_mat*(ep-ee) + 2*e_mat*ee) * (p_mat**2*(ep-ew) + 4*w_mat**2*ew) +
               (p_mat*(ep-ew) + 2*w_mat*ew) * (p_mat**2*(ep-ee) + 4*e_mat**2*ee)))
        ap = ep * k**2 - an - _as - ae * ep/ee - aw * ep/ew
    elif field == 'ey':
        an = (8 * (q_mat*(ep-es) + 2*s_mat*es) * en /
              ((q_mat*(ep-en) + 2*n_mat*en) * (q_mat**2*(ep-es) + 4*s_mat**2*es) +
               (q_mat*(ep-es) + 2*s_mat*es) * (q_mat**2*(ep-en) + 4*n_mat**2*en)))
        _as = (8 * (q_mat*(ep-en) + 2*n_mat*en) * es /
              ((q_mat*(ep-en) + 2*n_mat*en) * (q_mat**2*(ep-es) + 4*s_mat**2*es) +
               (q_mat*(ep-es) + 2*s_mat*es) * (q_mat**2*(ep-en) + 4*n_mat**2*en)))
        ae = 2 / (e_mat * (e_mat + w_mat))
        aw = 2 / (w_mat * (e_mat + w_mat))
        ap = ep * k**2 - an * ep/en - _as * ep/es - ae - aw
    elif field == 'scalar':
        an = 2 / (n_mat * (n_mat + s_mat))
        _as = 2 / (s_mat * (n_mat + s_mat))
        ae = 2 / (e_mat * (e_mat + w_mat))
        aw = 2 / (w_mat * (e_mat + w_mat))
        ap = ep * k**2 - an - _as - ae - aw
    else:
        raise ValueError("Unsupported field type. Use 'ex', 'ey', or 'scalar'.")
    
    # Modify matrix elements to account for boundary conditions.
    # North boundary (last column).
    if boundary[0] == 'S':
        ap[:, -1] += an[:, -1]
    elif boundary[0] == 'A':
        ap[:, -1] -= an[:, -1]
        
    # South boundary (first column).
    if boundary[1] == 'S':
        ap[:, 0] += _as[:, 0]
    elif boundary[1] == 'A':
        ap[:, 0] -= _as[:, 0]
        
    # East boundary (last row).
    if boundary[2] == 'S':
        ap[-1, :] += ae[-1, :]
    elif boundary[2] == 'A':
        ap[-1, :] -= ae[-1, :]
        
    # West boundary (first row).
    if boundary[3] == 'S':
        ap[0, :] += aw[0, :]
    elif boundary[3] == 'A':
        ap[0, :] -= aw[0, :]
    
    # Create index mapping following MATLAB's column-major (Fortran) order.
    ii = np.arange(nx * ny).reshape((nx, ny), order='F')
    iall = ii.flatten('F')
    in_idx = ii[:, 1:ny].flatten('F')      # north interior (columns 2:ny)
    is_idx = ii[:, 0:ny-1].flatten('F')    # south interior (columns 1:ny-1)
    ie_idx = ii[1:nx, :].flatten('F')      # east interior (rows 2:nx)
    iw_idx = ii[0:nx-1, :].flatten('F')    # west interior (rows 1:nx-1)
    
    # Build sparse matrix A.
    # Main diagonal from ap.
    data_all = ap.flatten('F')
    # Off-diagonals:
    data_iw = ae[:-1, :].flatten('F')
    data_ie = aw[1:, :].flatten('F')
    data_is = an[:, :ny-1].flatten('F')
    data_in = _as[:, 1:ny].flatten('F')
    
    rows = np.concatenate([iall, iw_idx, ie_idx, is_idx, in_idx])
    cols = np.concatenate([iall, ie_idx, iw_idx, in_idx, is_idx])
    data = np.concatenate([data_all, data_iw, data_ie, data_is, data_in])


    A = sp.coo_matrix((data, (rows, cols)), shape=(nx*ny, nx*ny)).tocsr()
    
    # Set up the eigenvalue problem and solve.
    shift = (2*np.pi*guess/lambda_)**2
    
    # Solve eigenvalue problem
    eigenvals, eigenvecs = eigs(A, k=nmodes, sigma=shift, tol=1e-12)

    # Calculate effective indices.
    neff = lambda_ * np.sqrt(np.real(eigenvals)) / (2 * np.pi)
    
    # Normalize eigenvectors and reshape into (nx,ny,nmodes)
    phi = np.zeros((nx, ny, nmodes), dtype=complex)
    for k in range(nmodes):
        vec = eigenvecs[:, k]
        vec = vec / np.max(np.abs(vec))
        phi[:, :, k] = np.reshape(vec, (nx, ny), order='F')
    
    return phi, neff
