import numpy as np
import matplotlib.pyplot as plt

from bounds_vectors import w_norm, RLS_output_bounds
from oneD_example_utils import build_basis, l2_regularizer_matrix, simulate_once, plot_learning_on_ax, plot_trajectory, generate_noise, params
from utils_maths import latexify
latexify()

theta_star = params["theta_star"]
umax = params["umax"]
delta = params["delta"]

tmax = 500

# Grid for plotting
ngrid = 200
u_grid = np.linspace(-umax-0.2, umax+0.2, ngrid)
M_grid = build_basis(u_grid)
g_star = M_grid @ theta_star

# --- simulation
rng = np.random.default_rng(1)
phi_ref = np.random.uniform(0, 2*np.pi) # random phase for the reference trajectory
xr = params["xmax"] * np.sin(params["omega_ref"]*np.arange(tmax)+phi_ref)

ws, cw = generate_noise(rng, tmax)
us, ys, xs = simulate_once(ws, xr, theta_star, u_max=umax, K=params["K"])

# --- Plot a trajectory of the system (optional)
fig = plot_trajectory(xs, xr, us, umax)
fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# learning parameters
P = params["lambda_P"] * l2_regularizer_matrix(umax) # regularizer
ctheta = params["ctheta"]

Ms = build_basis(us)
g_hats, bs = RLS_output_bounds(Ms, ys, M_grid, P, ctheta, cw, delta, method="ours")
g_hat, b = g_hats[...,-1].squeeze(), bs[:,-1] # only the estimate at time t is of interest here.

# --- Plot the learned function and the confidence bounds
fig, ax = plt.subplots(figsize=(8,3))
plot_learning_on_ax(ax, us, ys, u_grid, g_star, g_hat, b, umax=umax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
plt.show(block=False)
plt.pause(100)