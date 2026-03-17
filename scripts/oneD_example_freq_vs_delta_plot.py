
"""
    Illustrative example (scalar output, polynomial g(u;theta), closed-loop inputs, bounded noise).
    Generates the figures
"""
import pickle
import matplotlib.pyplot as plt
import sys
from os.path import join, dirname

main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)
from utils_plotting import latexify, save_and_show
latexify()

file_saving = join(main_dir, "pickles", "oneD_example_violations.pkl")
with open(file_saving, 'rb') as f:
    loaded = pickle.load(f)

deltas = loaded["deltas"]
freq_ours = loaded["freq_ours"]
freq_tri = loaded["freq_tri"]

fig, ax = plt.subplots(figsize=(4.75, 1.3))
ax.grid()
ax.set_xlabel(r"confidence level $\delta$")
ax.set_ylabel(r"violation frequency")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(deltas, deltas, linestyle="--", label=r"$\delta$", color="C0")
ax.plot(deltas, freq_ours, marker=".", linestyle="-", label= "Novel bound", color="C1")
ax.plot(deltas, freq_tri, marker=".", linestyle=":", label="Existing bound", color="purple")


fig.subplots_adjust(right=0.69, top=1) # make space for the legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=1)
save_and_show(fig, "example1_violation_vs_delta.pdf")

plt.pause(100)