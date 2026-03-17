import numpy as np
from numpy.linalg import inv
from utils_maths import *

def RLS(Ms, ys, P, mu=None, tol=1e-4):
    """
        Compute the RLS estimate of theta given data (Ms, ys) and regularization P.
            Ms is of shape (T, ny, ntheta)
            ys is of shape (T, ny)
            P is of shape (ntheta, ntheta)
    """
    assert (len(Ms.shape) == 3) and (len(ys.shape) == 2), f"wrong shapes for Ms, or ys, you got {Ms.shape} and {ys.shape}"
    tmax, ny, ntheta = Ms.shape
    if mu is None:
        mu = np.zeros(ntheta)
    Vs = np.zeros((tmax, ntheta, ntheta))
    thetas = np.zeros((tmax, ntheta))
    psd_flags = np.zeros(tmax, dtype=bool)

    grad =  P @ mu
    for t in range(tmax):
        if not psd_flags[t]:
            if np.all( np.linalg.eigvalsh(P + Vs[t]) > tol):
                psd_flags[t:] = True
        if psd_flags[t]:
            thetas[t] = inv(P + Vs[t]) @ grad
        if t < tmax - 1:
            Vs[t+1] = Vs[t] + Ms[t].T @ Ms[t]
            grad = grad + Ms[t].T @ ys[t]
    return thetas, Vs, psd_flags

def beta_fn(Vbar: np.ndarray, V: np.ndarray, ctheta: float, cw: float, delta: float, method="ours", tol=1e-4) -> float:
    log_part = logdet_part(Vbar, V) - 2*np.log(delta) # =  log|I + P^{-1} V| + 2 log(1/delta)
    if method == "ours": # Corollary 1 of the paper (P ≻ 0)
        return np.sqrt(ctheta**2 + cw**2 * log_part)
    elif method == "triangular": # Abbasi-Yadkori et al. bound
        return ctheta + cw*np.sqrt(log_part) 
    elif method == "noprior": # Corollary 2 of the paper (P>=0), here P corresponds to \bar{V} of the paper
        if np.all( np.linalg.eigvalsh(V) > tol):
            return np.sqrt( (1.0 + rho_fn(inv(V) @ Vbar)) * (ctheta**2 + cw**2 * log_part) )
        else:
            return np.inf
    else:
        raise ValueError(f"Unknown method {method}")

def is_bound_violated(theta_star, theta_hats, P, Vs, ctheta, cw, delta, method="ours"):
    # check if the time-uniform bound is satisfied
    for t in range(len(theta_hats)):
        beta = beta_fn(P, Vs[t], ctheta=ctheta, cw=cw, delta=delta, method=method)
        lhs = w_norm(theta_star - theta_hats[t], P + Vs[t])
        if lhs > beta:
            return True
    return False

def RLS_output_bounds(Ms, ys, M_grid, Vbar, ctheta, cw, delta, method="ours"):
    """
        Ms is of shape (T, ny, ntheta)
        ys is of shape (T, ny)
        M_grid is of shape (Ngrid, ny, ntheta)

        outputs:
            hat_ys_grid is of shape (T, ny, Ngrid)
            bs_grid is of shape (T, Ngrid)
    """
    assert (len(Ms.shape) == 3) and (len(ys.shape) == 2) and (len(M_grid.shape) == 3), f"wrong shapes for Ms, ys or M_grid, you got {Ms.shape}, {ys.shape}, {M_grid.shape}"
    if method == "noprior":
        P = 0. * Vbar
    else:
        P = Vbar
    tmax = len(Ms)
    ngrid = len(M_grid)
    thetas, Vs, psd_flags = RLS(Ms, ys, P)
    hat_ys_grid = np.einsum("gyp, tp -> gyt",  M_grid, thetas, out=None, casting='safe') # (ngrid, ny, tmax)
    hat_ys_grid[..., ~psd_flags] = np.inf # set the predictions to inf when the bound is not defined
    bs_grid = np.zeros((ngrid, tmax))
    for t in range(tmax):
        if psd_flags[t]:
            beta = beta_fn(Vbar, Vs[t], ctheta=ctheta, cw=cw, delta=delta, method=method)
            invG = inv(P + Vs[t])
            for g in range(ngrid):
                bs_grid[g, t] = beta * op_norm(M_grid[g], invG)
        else:
            bs_grid[:, t] = np.inf
    return hat_ys_grid, bs_grid