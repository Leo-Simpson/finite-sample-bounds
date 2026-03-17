
"""
    Illustrative example (scalar output, polynomial g(u;theta), closed-loop inputs, bounded noise).
    Generates the figures
"""
import numpy as np
import pickle
import sys
from os.path import join, dirname

main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)
from bounds_vectors import RLS, is_bound_violated
from oneD_example_utils import build_basis, l2_regularizer_matrix, params, simulate

theta_star = params["theta_star"]
umax = params["umax"]

# learning parameters
P = params["lambda_P"] * l2_regularizer_matrix(umax) # regularizer 
ctheta = params["ctheta"]

# --- Empirical violation probability vs delta (coverage)
deltas = np.geomspace(0.5, 0.005, 10)  # confidence levels to test, DECREASING order is important here
rng = np.random.default_rng(0)

n_mc, tmax = 10000, 200 # for clean: 10000, 200 for fast: 1000, 50
n_violations = {"ours":np.zeros(len(deltas), dtype=int), "triangular":np.zeros(len(deltas), dtype=int)}
for i in range(n_mc):
    print(f"MC iteration {i+1}/{n_mc}", end="\r")
    us, ys, cw = simulate(rng, tmax)
    Ms = build_basis(us)  # (T,1,4)
    thetas, Vs, _ = RLS(Ms, ys, P)
    for method in ["ours", "triangular"]:
        for i, delta in enumerate(deltas):
            if is_bound_violated(theta_star, thetas, P, Vs, ctheta, cw, delta, method=method):
                n_violations[method][i] += 1
            else:
                break # since deltas are decreasing, the bounds get looser, so if one is not violated, the rest won't be either

freq_ours = n_violations["ours"] / n_mc
freq_tri = n_violations["triangular"]  / n_mc

print(f"For delta={deltas[0]}, got violation frequencies: our bound {freq_ours[0]:.3f}, existing bound {freq_tri[0]:.3f}")

outdir = join(main_dir, "pickles")
with open(join(outdir, "oneD_example_violations.pkl"), 'wb') as f:
    pickle.dump( {"deltas": deltas, "freq_ours": freq_ours, "freq_tri": freq_tri}, f)

