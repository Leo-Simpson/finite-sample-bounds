
"""
    Multi dimensional example (heat transfer problem).
    Utility functions
"""
import numpy as np

def simulate_heat_system(alpha, beta, temp_init, us, ws):
    """
        Simulate the heat transfer chain system.
    """
    tmax, nx = ws.shape
    all_temps = np.zeros((tmax + 1, nx+1))
    all_temps[0, :] = temp_init
    all_temps[:-1, 0] = 1 * us[:,0]
    for t in range(tmax):
        all_temps[t+1, 1:] = all_temps[t, 1:] + alpha * (all_temps[t, :-1] - all_temps[t, 1:]) + beta * (us[t, 1] - all_temps[t, 1:]) + ws[t]
    xs = all_temps[:, 1:]
    return xs

def build_LTI_regressors(us, xs):
    """
        Build phis such that g_{t+1} := x_{t+1} - x_t  = THETA phi_t,
    """
    tmax, nx = xs[:-1].shape
    phis = np.zeros((tmax, nx+2))
    phis[:, :2] = 1 * us
    phis[:, 2:] = xs[:-1]
    return phis

def build_parametric_regressors(us, xs):
    """
        Build ys, Ms such that g_{t+1} := x_{t+1} - x_t = M_t theta = M1_t alpha + M2_t beta, where M1_t and M2_t are the first and second slices of M_t along the last axis.
    """
    tmax, nx = xs[:-1].shape
    M1s = np.zeros((tmax, nx))
    M1s[:, 0] = us[:,0] - xs[:-1, 0]
    M1s[:, 1:] = xs[:-1, :-1] - xs[:-1, 1:]

    M2s = us[:,1:] - xs[:-1]
    Ms = np.stack((M1s, M2s), axis=-1)
    return Ms

