import numpy as np
import matplotlib.pyplot as plt

from multi_example_utils import (
    simulate_heat_system,
    build_LTI_regressors,
    build_parametric_regressors
)

from bounds_vectors import RLS_output_bounds
from bounds_matrix import OLS_output_bounds
from utils_maths import latexify, save_and_show
latexify()


# system parameters
nx = 5
alpha = 0.5
beta = 0.05

temp_ext = 25
temp_init = 20
temp_max = 100

tmax = 2000
delta = 0.05
cw = 1  / np.sqrt(3) # subgaussian proxy constant for uniform noise in [-temp_max, temp_max]

rng = np.random.default_rng(0)

# generate identification trajectory
us = temp_max * (1 + np.sin(0.1 * np.arange(tmax)))
ws = rng.uniform(-1, 1, size=(tmax, nx))
_, xs = simulate_heat_system(alpha, beta, temp_ext, temp_init, us, ws)
ys = xs[1:] - xs[:-1]
phis = build_LTI_regressors(us, xs, temp_max) # Create the data for LTI identification
Ms = build_parametric_regressors(us, xs, temp_max) # Create the data for parametric identification

ntheta = Ms.shape[-1]

# Create PHI and V
PHIs = np.zeros((tmax, nx+2, nx+2))
Vs = np.zeros((tmax, ntheta, ntheta))
for t in range(tmax-1):
    PHIs[t+1] = PHIs[t] + np.outer( phis[t], phis[t])
    Vs[t+1] = Vs[t] + Ms[t].T @ Ms[t]

PHIs = PHIs / (temp_max)*2
Vs = Vs / (temp_max)**2

min_eig_PHI = np.zeros(tmax)
min_eig_V = np.zeros(tmax)
max_eig_PHI = np.zeros(tmax)
max_eig_V = np.zeros(tmax)
for t in range(tmax):
    eigs_PHI = np.linalg.eigvalsh(PHIs[t])
    eigs_V = np.linalg.eigvalsh(Vs[t])

    min_eig_PHI[t] = eigs_PHI.min()
    max_eig_PHI[t] = eigs_PHI.max()

    min_eig_V[t] = eigs_V.min()
    max_eig_V[t] = eigs_V.max()

ts = np.arange(tmax)
fig, ax = plt.subplots()
ax.plot(ts, min_eig_PHI, label=r"$\lambda_{\min}(\bar{\Phi}_t)$", color="C0")
# ax.plot(ts, max_eig_PHI, label=r"$\lambda_{\max}(\bar{\Phi}_t)$", color="C0", linestyle="--")

ax.plot(ts, min_eig_V, label=r"$\lambda_{\min}(\bar{V}_t)$", color="C1")
# ax.plot(ts, max_eig_V, label=r"$\lambda_{\max}(\bar{V}_t)$", color="C1", linestyle="--")

ax.set_xlabel("Time")
ax.legend()
plt.show()




    
