import numpy as np
from numpy.linalg import inv
from utils_maths import w_norm, rho_fn, logdet_part

def OLS(phis, ys, tol=1e-4):
    """
        Compute the OLS estimate of theta given data (Ms, ys) and regularization P.
            Phis is of shape (T, nphi)
            ys is of shape (T, ny)
            P is of shape (ntheta, ntheta)
    """
    assert (len(phis.shape) == 2) and (len(ys.shape) == 2) and (len(ys)==len(phis)), f"wrong shapes for Phis, or ys, you got {phis.shape} and {ys.shape}"
    tmax, ny = ys.shape
    _, nphi = phis.shape
    PHIs = np.zeros((tmax, nphi, nphi))
    THETAs = np.ones((tmax, ny, nphi)) * np.nan
    grad = np.zeros((ny, nphi))
    psd_flags = np.zeros(tmax, dtype=bool)
    for t in range(tmax):
        if not psd_flags[t]:
            if np.all( np.linalg.eigvalsh(PHIs[t]) > tol):
                psd_flags[t:] = True
        if psd_flags[t]:
            THETAs[t] = grad @ inv(PHIs[t])
        if t < tmax - 1:
            PHIs[t+1] = PHIs[t] + np.outer(phis[t], phis[t])
            grad = grad + np.outer(ys[t], phis[t])
    return THETAs, PHIs, psd_flags

def beta_LTI(PHIbar, PHI, nx, cw, delta, method="operator"):
    assert np.linalg.det(PHI) > 0. # det should be positive 

    logdet_part_ = logdet_part(PHIbar, PHI)
    rho_part_ = (1.0 + rho_fn(inv(PHI) @ PHIbar))

    if method == "operator":
        # beta_op = 2* cw * sqrt( (1 + rho(Phi^{-1} Phibar)) ( log|I + Phibar^{-1} Phi| + 2 nx log(5) + 2 log(1/delta) ) )
        return 2 * cw* np.sqrt(  rho_part_ * (logdet_part_ + 2*nx*np.log(5) - 2*np.log(delta)) )
    elif method == "Frobenius":
        # beta_fr^2 = cw^2 (1 + rho(Phi^{-1} Phibar)) ( nx log|I + Phibar^{-1} Phi| + 2 log(1/delta) )
        return cw * np.sqrt(  rho_part_ * (nx * logdet_part_ - 2*np.log(delta)) )
    else:
        raise ValueError(f"Unknown method {method}")

def OLS_output_bounds(phis, ys, phi_grid, PHIbar, cw, delta, method="operator"):
    THETAs, PHIs, psd_flags = OLS(phis, ys)
    hat_ys_grid = np.einsum("typ, gp -> gyt", THETAs, phi_grid) # (ngrid, ny, tmax)

    tmax, ny = ys.shape
    ngrid = len(phi_grid)
    bs_grid = np.zeros((ngrid, tmax), dtype=np.float64)
    for t in range(tmax):
        if psd_flags[t]:
            invG = inv(PHIs[t])
            beta = beta_LTI(PHIbar, PHIs[t], nx=ny, cw=cw, delta=delta, method=method)
            for g in range(ngrid):
                bs_grid[g, t] = beta * w_norm(phi_grid[g], invG)
        else:
            bs_grid[:, t] = np.inf
    return hat_ys_grid, bs_grid