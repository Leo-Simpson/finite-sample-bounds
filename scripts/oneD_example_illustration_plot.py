
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
from oneD_example_utils import build_basis, l2_regularizer_matrix, simulate, plot_learning_on_ax, plot_many_learning_on_ax, params
from bounds_vectors import RLS_output_bounds
from utils_plotting import latexify, save_and_show
latexify()


version = 2

theta_star = params["theta_star"]
umax = params["umax"]
delta = params["delta"]
tmax = params["tmax"]

# Grid for plotting
ngrid = 200
u_grid = np.linspace(-umax-0.2, umax+0.2, ngrid)
M_grid = build_basis(u_grid)
g_star = M_grid @ theta_star

# learning parameters
P = params["lambda_P"] * l2_regularizer_matrix(umax) # regularizer
ctheta = params["ctheta"]

# --- Two illustrative realizations
infos = []
for seed in [1, 2]:
    rng = np.random.default_rng(seed)
    us, ys, cw = simulate(rng, tmax)
    Ms = build_basis(us)
    g_hats, bs_ours = RLS_output_bounds(Ms, ys, M_grid, P, ctheta, cw, delta, method="ours")
    g_hat, b_ours = g_hats[...,-1].squeeze(), bs_ours[:,-1] # only the estimate at time t is of interest here.
    info = {"us": us, "ys": ys, "g_hat": g_hat, "b_ours": b_ours, "b_others": None}

    if version == 2:
        _, bs_others = RLS_output_bounds(Ms, ys, M_grid, P, ctheta, cw, delta, method="triangular")
        b_others = bs_others[:,-1]
        info["b_others"] = b_others

        print(f"Illustrative example {seed}: b_ours = {b_ours.max():.3f}, b_others = {b_others.max():.3f}")
        print(f"      factor = {b_others.max()/b_ours.max():.3f}")

    infos.append(info)



# --- Many confidence intervals overlay (20 runs)
rng = np.random.default_rng(0)
N_real = 20
g_hats_exp = np.zeros((N_real, len(M_grid)))
b_exp_ours = np.zeros((N_real, len(M_grid)))
b_exp_others = np.zeros((N_real, len(M_grid)))
for i in range(N_real):
    us, ys, cw = simulate(rng, tmax) # simulate a trajectory
    Ms = build_basis(us)
    g_hats, bs_ours = RLS_output_bounds(Ms, ys, M_grid, P, ctheta, cw, delta, method="ours")
    g_hats_exp[i], b_exp_ours[i] = g_hats[...,-1].squeeze(), bs_ours[:,-1] # only the estimate at time t is of interest here.

    if version == 2:
        _, bs_others = RLS_output_bounds(Ms, ys, M_grid, P, ctheta, cw, delta, method="triangular")
        b_exp_others[i] = bs_others[:,-1]
    
lb_ours = g_hats_exp - b_exp_ours
ub_ours = g_hats_exp + b_exp_ours

if version == 2:
    lb_others = g_hats_exp - b_exp_others
    ub_others = g_hats_exp + b_exp_others
else:
    lb_others, ub_others = None, None


fig, axs = plt.subplots(2, 2, figsize=(4.6, 3.5), constrained_layout=True) # gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
    ax.axvline(umax, color="black", alpha=0.4)
    ax.axvline(-umax, color="black", alpha=0.4)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
for i in range(2):
    plot_learning_on_ax(axs[0, i], infos[i]["us"], infos[i]["ys"], u_grid, g_star, infos[i]["g_hat"], infos[i]["b_ours"], b_others=infos[i]["b_others"])
plot_many_learning_on_ax(axs[1, 0], u_grid, g_star, lb_ours, ub_ours, lb_others=lb_others, ub_others=ub_others)

# Make the legend
ax4legend = axs[1, 1]
ax4legend.axis("off") # dummy axis for legend
handles, labels = axs[0, 0].get_legend_handles_labels()


ax4legend.legend(handles, labels, loc="center", ncol=1)

save_and_show(fig, f"example1_illustrative_runs_v{version}.pdf")
plt.pause(30)