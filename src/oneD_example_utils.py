
"""
    Illustrative example (scalar output, polynomial g(u;theta), closed-loop inputs, bounded noise).
    Utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
from utils_maths import w_norm

# Parameters of the problem
params = {
    "theta_star": np.array([0, 1., 0, -1.]),
    "umax": 1.,
    "xmax": 2,
    "tmax": 20,
    "K":1.,
    "omega_ref":0.1,
    "delta":0.05,
    "lambda_P":1.,
    "ctheta": "truth" # use the true P-norm as ctheta
}



# Problem-specific utility functions

exponents = np.arange(4)  # [0,1,2,3]

def build_basis(u) -> np.ndarray:
    """
        Compute [1, u, u^2, u^3]
    """
    if np.isscalar(u):
        return np.power(u, exponents)[:, None]
    else:
        return np.power(u[:, None], exponents)[:, None, :]

def l2_regularizer_matrix(u_max:float, n_grid: int = 100) -> np.ndarray:
    # R = ∫_{-umax}^umax phi(u) phi(u)^T du  (numerical quadrature)
    ug = np.linspace(-u_max, u_max, n_grid)
    Phi = build_basis(ug).squeeze()  # (N, 4)
    return (2 * u_max / n_grid ) * (Phi.T @ Phi) 

if params["ctheta"] == "truth":
    params["ctheta"] = w_norm(params["theta_star"], params["lambda_P"] * l2_regularizer_matrix(params["umax"]))


def simulate_once(w, xr, theta_star, u_max=1., K=1.):
    T = len(w)
    u = np.zeros(T)
    y = np.zeros(T)
    x = np.zeros(T+1) # x0 is 0.
    for t, w_t in enumerate(w):
        u_uncstr = K*(xr[t] - x[t]) # u_uncstr approx g^-1(xr - x[t])
        u[t] = np.clip(u_uncstr, -u_max, u_max) 
        g_t = build_basis(u[t]).squeeze() @ theta_star
        x[t+1] = x[t] + g_t + w_t
    y = x[1:] - x[:-1]  # y_{t+1} = x_{t+1}-x_t
    y = y[:, None]  # make it (T,1) for consistency with the bounds code
    return u, y, x

def generate_noise(rng, size, type="gaussian"):
    wmax =  0.3
    cw = wmax / np.sqrt(3) # subgaussian proxy constant
    if type == "uniform":
        ws = wmax * rng.uniform(-1, 1, size=size)
    elif type == "gaussian":
        sigma =  0.2
        ws = sigma * rng.normal(0, 1, size=size)
        cw = sigma # subgaussian proxy constant
    return ws, cw

def simulate(rng,  tmax, noise_distribution="gaussian"):
    # simulate a trajectory
    phi_ref = rng.uniform(0, 2*np.pi) # random phase for the reference trajectory
    xr = params["xmax"] * np.sin(params["omega_ref"]*np.arange(tmax)+phi_ref)
    ws, cw = generate_noise(rng, tmax, type=noise_distribution)
    us, ys, _ = simulate_once(ws, xr, params["theta_star"], u_max=params["umax"], K=params["K"])
    return us, ys, cw

# Utilities functions for plotting
color_star = "C0"
color_hat = "C1"

def plot_learning_on_ax(ax, us, ys, u_grid, g_star, g_hat, b, umax=None):
    """
         Plot the learned function and the confidence bounds
    """
    ax.set_xlabel(r"$u_k$")
    ax.set_xlim(u_grid.min(), u_grid.max())
  
    ax.plot(u_grid, g_star, linewidth=2, color=color_star, linestyle="-.", label=r"$g(u_k; \theta^\star)$")
    ax.plot(us, ys, marker="+", markersize=6, linestyle='None', alpha=0.9, color="green", label=r"$y_{k+1}$")

    ax.plot(u_grid, g_hat, linewidth=2, linestyle="--", color=color_hat, label=r"$g(u_k; \hat{\theta}_t)$")


    ax.plot(u_grid, g_hat+b, linewidth=1, linestyle="-", color=color_hat)
    ax.plot(u_grid, g_hat-b, linewidth=1, linestyle="-", color=color_hat)
    ax.fill_between(u_grid, g_hat-b, g_hat+b, alpha=0.25, color=color_hat)
    

def plot_many_learning_on_ax(ax, u_grid, g_star, lower_bounds, upper_bounds):
    """
            Plot many learned functions and confidence bounds overlayed (e.g. for multiple runs)
    """
    ax.set_xlabel(r"$u_k$")
    ax.set_xlim(u_grid.min(), u_grid.max())

    for i in range(len(lower_bounds)):
        ax.plot(u_grid, lower_bounds[i], alpha=0.3, color=color_hat)
        ax.plot(u_grid, upper_bounds[i], alpha=0.3, color=color_hat)
        ax.fill_between(u_grid, lower_bounds[i], upper_bounds[i], alpha=0.01, color=color_hat)
    ax.plot(u_grid, g_star, color=color_star, linestyle="-.", linewidth=2) # plot true once
    

def plot_trajectory(xs, xr, us, u_max):
    fig, axs = plt.subplots(2, figsize=(8,6))
    ts = np.arange(len(xs)-1)
    axs[0].plot(ts, xs[:-1], label=r"$x_t$")
    axs[0].plot(ts, xr, label=r"$\bar{x}_t$")
    axs[1].plot(ts, us, label=r"$u_t$")
    axs[1].axhline(u_max, color="black", label=r"$u_{\max}$")
    axs[1].axhline(-u_max, color="black")
    for ax in axs:
        ax.grid()
        ax.set_xlabel(r"$t$")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig