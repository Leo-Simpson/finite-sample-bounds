import numpy as np
import pickle
import sys
from os.path import join, dirname

main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)

from multi_example_utils import (
    simulate_heat_system,
    build_LTI_regressors,
    build_parametric_regressors,
)

from bounds_vectors import RLS_output_bounds
from bounds_matrix import OLS_output_bounds


# system parameters
nx = 5
alpha = 0.5
beta = 0.1
ntheta = 2

temp_ext = 25
temp_init = 20
temp_max = 100

tmax = 2000
delta = 0.05
cw = 1  / np.sqrt(3) # subgaussian proxy constant for uniform noise in [-temp_max, temp_max]

rng = np.random.default_rng(0)

# generate identification trajectory
temp0 = temp_max * (1 + np.sin(0.1 * np.arange(tmax)))
us = np.array([temp0, temp_ext * np.ones(tmax)]).T

ws = rng.uniform(-1, 1, size=(tmax, nx))
xs = simulate_heat_system(alpha, beta, temp_init, us, ws)
ys = xs[1:] - xs[:-1]
phis = build_LTI_regressors(us, xs) # Create the data for LTI identification
Ms = build_parametric_regressors(us, xs) # Create the data for parametric identification


# Create test trajectory
n_grid = 1000 # for clean: 1000, for fast: 20
temp_grid = rng.uniform(0, temp_max, size=(n_grid, nx+2))
us_grid = temp_grid[:-1, :2]
xs_grid = temp_grid[:, 2:]

phis_grid = build_LTI_regressors(us_grid, xs_grid) # Create the data for LTI test
Ms_grid = build_parametric_regressors(us_grid, xs_grid) # Create the data for parametric test
g_stars = alpha * Ms_grid[..., 0] + beta * Ms_grid[..., 1]


# Make parametric estimates and bounds
Vbar = np.eye(ntheta) * temp_max**2
ctheta = 0. # OLS
g_hats_p, bs_p = RLS_output_bounds(Ms, ys, Ms_grid, Vbar, ctheta, cw, delta, method="noprior")
print("Parametric bounds compute!")
bs_p_truth = np.sqrt( np.sum((g_hats_p - g_stars[..., None])**2, axis=1) )
max_b_p = bs_p.max(axis=0) # worst-case bound accross test trajectory
max_b_p_truth = bs_p_truth.max(axis=0) # worst-case error accross test trajectory


# Make LTI estimates and bounds
PHIbar = np.eye(nx+2) * temp_max**2
g_hats_LTI, bs_LTI = OLS_output_bounds(phis, ys, phis_grid, PHIbar, cw, delta, method="Frobenius")
print("Non-parametric bounds compute!")

bs_LTI_truth = np.sqrt( np.sum((g_hats_LTI - g_stars[..., None])**2, axis=1) )
max_b_LTI = bs_LTI.max(axis=0) # worst-case bound accross test trajectory
max_b_LTI_truth = bs_LTI_truth.max(axis=0) # worst-case error accross test trajectory

# Make the same but for the operator norm bound
_, bs_LTI_op = OLS_output_bounds(phis, ys, phis_grid, PHIbar, cw, delta, method="operator")
max_b_LTI_op = bs_LTI_op.max(axis=0) # worst-case bound accross test trajectory


outdir = join(main_dir, "pickles")
dict_to_save = {
    "max_b_p": max_b_p,
    "max_b_LTI": max_b_LTI,
    "max_b_LTI_op": max_b_LTI_op,
    "max_b_p_truth": max_b_p_truth,
    "max_b_LTI_truth": max_b_LTI_truth,
}

with open(join(outdir, "example2.pkl"), 'wb') as f:
    pickle.dump( dict_to_save, f)
