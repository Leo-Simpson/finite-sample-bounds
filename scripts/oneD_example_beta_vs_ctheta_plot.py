
"""
    Illustrative example (scalar output, polynomial g(u;theta), closed-loop inputs, bounded noise).
    Generates the figures
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import join, dirname

main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)
from bounds_vectors import w_norm, beta_fn
from oneD_example_utils import build_basis, l2_regularizer_matrix, simulate, params
from utils_plotting import latexify, save_and_show, plot_many, MyPatch
latexify()

theta_star = params["theta_star"]
umax = params["umax"]
delta = params["delta"]
tmax = params["tmax"]

# learning parameters
P = params["lambda_P"] * l2_regularizer_matrix(umax) # regularizer
ctheta_min = w_norm(theta_star, P) # true P-norm as ctheta

# --- Max error bound vs time (shrinking) : ours vs Abbasi
N_real = 20
N_ctheta = 10
ctheta_min_plot = 0.3
cthetas = np.linspace(ctheta_min, 1., N_ctheta) # test a range of ctheta values around the true P-norm

n_ctheta = 10
rng = np.random.default_rng(0)
betas_truth = np.zeros((N_real, N_ctheta))
betas_ours = np.zeros((N_real, N_ctheta))
betas_tri = np.zeros((N_real, N_ctheta))

for i in range(N_real):
    us, ys, cw = simulate(rng, tmax)
    ys = ys.squeeze() # (T,)
    Ms = build_basis(us).squeeze() # (T,4)
    V = Ms.T @ Ms
    theta_hat = np.linalg.inv(P + V) @ (Ms.T @ ys)
    betas_truth[i, :] = w_norm(theta_star - theta_hat, P + V) # this is constant across ctheta but we repeat it for ease of plotting.
    for j, ctheta in enumerate(cthetas):
        betas_ours[i, j] = beta_fn(P, V, ctheta, cw, delta, method="ours")
        betas_tri[i, j] = beta_fn(P, V, ctheta, cw, delta, method="triangular")

# Plot beta 
fig, ax = plt.subplots(figsize=(4.75, 1.2))
ax.grid()
ax.set_xlabel(r"prior bound $c_{\theta}$")
# ax.set_ylabel(r"$\beta_t$")
ax.set_ylim(0, 2)
ax.set_xlim(ctheta_min_plot, cthetas[-1])
ax.axvline(ctheta_min, linewidth=1, color="red", alpha=0.8, linestyle="-")
ax.axvspan(ctheta_min_plot, ctheta_min, color="red", alpha=0.2, linestyle="-") # forbidden region
plot_many(ax, betas_truth, r"$\max\limits_{k \leq t} \| \hat{\theta}_k - \theta^\star \|_{P+V_k}$", "C0", linestyle="-.", xs=cthetas, with_mean=False)
plot_many(ax, betas_ours, r"$\beta_t$ (novel bound)", "C1", xs=cthetas, with_mean=False, linestyle="-")
plot_many(ax, betas_tri, r"$\tilde{\beta_t}$ (existing bound)", "purple", xs=cthetas, with_mean=False, linestyle=":")


fig.subplots_adjust(right=0.63, top=1) # make space for the legend
handles, labels = ax.get_legend_handles_labels()

handles.append(MyPatch(type_of_region="forbidden_region"))
labels.append(r"$c_{\theta} < \| \theta^\star \|_{P}$")


fig.legend(handles, labels, loc="upper right")

save_and_show(fig, "example1_beta_vs_ctheta.pdf")
plt.pause(0.1)


plt.pause(100)