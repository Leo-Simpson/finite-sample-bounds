import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os.path import join, dirname
import os

def latexify():
    matplotlib.rcParams.update( {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': True,
            'font.family': 'serif'
        } )

def plot_many(ax,Ys, label, color, linestyle="-", xs=None, with_mean=True):
    """
        On some ax, plot many trajectories of ys as a function of xs, together with its mean
    """
    if xs is None:
        xs = np.arange(Ys.shape[1])
    for ys in Ys:
        ax.plot(xs, ys, linewidth=1, alpha=0.1, color=color, linestyle=linestyle)
    if with_mean:
        ax.plot(xs, Ys.mean(axis=0), linewidth=2, label=label, color=color, linestyle=linestyle)
    else:
        ax.plot([], [], linewidth=2, alpha=0.5, label=label, color=color, linestyle=linestyle)

def save_and_show(fig, filename):
    outdir = join(dirname(dirname(__file__)), "images")
    fig.savefig(join(outdir, filename), bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(0.1)



"""
    This part is for creating specific matplotlib handles for legends of shaded regions.
"""

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
import matplotlib.colors as mcolors

class MyPatch(Artist):
    def __init__(self, type_of_region="ouptut_constraints"):
        super().__init__()
        self.type_of_region = type_of_region

class MyHandler(HandlerBase):
    def create_artists( self, legend, patch, x_left, y_bottom, width, height, fontsize, trans):
        x_right = x_left + width
        y_top = y_bottom + height
        symbols: list[Artist] = []
        if patch.type_of_region == "ouptut_constraints":
            facecolor = mcolors.to_rgba("C1", 0.25)
            edgecolor = mcolors.to_rgba("C1", 1)
            symbols.append(
                Line2D( [x_left, x_right], [y_top, y_top], color=edgecolor, linewidth=1, transform=trans, solid_capstyle="butt")
            )
            symbols.append(
                Line2D( [x_left, x_right], [y_bottom, y_bottom], color=edgecolor, linewidth=1, transform=trans, solid_capstyle="butt")
            )
        elif patch.type_of_region == "forbidden_region":
            facecolor = mcolors.to_rgba("red", 0.2)
            edgecolor = mcolors.to_rgba("red", 0.8)
            symbols.append(
                Line2D( [x_right, x_right], [y_bottom, y_top], color=edgecolor, linewidth=1, transform=trans, solid_capstyle="butt")
            )

        symbols.append(
            Rectangle( (x_left, y_bottom), width, height, facecolor=facecolor, edgecolor="none", transform=trans)
        )
        return symbols
Legend.update_default_handler_map({MyPatch: MyHandler()})
