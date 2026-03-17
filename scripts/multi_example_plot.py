import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from os.path import join, dirname

main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)

from utils_plotting import latexify, save_and_show
latexify()

file_saving = join(main_dir, "pickles", "example2.pkl")
with open(file_saving, 'rb') as f:
    loaded = pickle.load(f)

max_b_p = loaded["max_b_p"]
max_b_LTI = loaded["max_b_LTI"]
max_b_LTI_op = loaded["max_b_LTI_op"]
max_b_p_truth = loaded["max_b_p_truth"]
max_b_LTI_truth = loaded["max_b_LTI_truth"]


tmax = len(max_b_p)
t_axis = np.arange(tmax)

fig, ax = plt.subplots(figsize=(4.75, 1.5))
ax.grid()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"Number of data points $t$")
# ax.set_ylabel(r"Prediction error")

ax.plot(t_axis, max_b_p_truth, "-.", label="l.h.s. in param. id.", linewidth=2, color="C1")
ax.plot(t_axis, max_b_p, label="r.h.s. in param. id.", linewidth=2, color="C1")

ax.plot(t_axis, max_b_LTI_truth, "--", label="l.h.s. in LTI id.", linewidth=2, color="red")
ax.plot(t_axis, max_b_LTI, label="r.h.s. in LTI id. (Fr.)", linewidth=1, color="purple")
ax.plot(t_axis, max_b_LTI_op, ":", label="r.h.s. in LTI id. (op.)", linewidth=2, color="purple")


ax.set_ylim(top=5000)
# ax.set_xlim(0, tmax)

fig.subplots_adjust(right=0.61, top=1) # make space for the legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=1)

save_and_show(fig, "example2_shrink.pdf")
plt.pause(100)

