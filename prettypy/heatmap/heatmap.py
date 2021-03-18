# -*- coding: utf-8 -*-
"""
@created: 03/18/21
@modified: 03/18/21
@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France


Defines internal classes user-level functions for plotting simple heatmaps.

"""

import copy
from   dataclasses import dataclass, field
import numpy        as    np
import pandas       as    pd
import os
from   typing      import Dict, List, Tuple, Union

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# type aliases
DataFrame  = pd.core.frame.DataFrame

def default_field(obj):
    return field(default_factory=lambda: obj)

@dataclass
class HeatmapConfig:
    figure: Dict[str, Union[str,tuple]] = default_field({
        "figsize": (8,8),
        "n_grid": 10,
        "dpi": 300,
    })
    heatmap: Dict[str, Union[int, str, bool, float]] = default_field({
        "xticklabels"          : True,
        "yticklabels"          : True,
        "ticks_labelsize"      : 8,
        "xticks_labelrotation" : 90,
        "yticks_labelrotation" : 0,
        "linecolor"            : "white",
        "linewidths"           : 0.5,
        "square"               : True,
    })
    legend: Dict[str, Union[int, float, str]] = default_field({
        'edgecolor': 'k',
        'fancybox': False,
        'facecolor': 'w',
        'fontsize': 10,
        'framealpha': 1,
        'frameon': False,
        'handle_length': 1,
        'handle_height': 1.125,
        'title_fontsize': 12,
    })
    cbar: Dict[str, Union[int, float, str, bool]] = default_field({
        'boundaries'      : [1,5,10,15,20,50,200,500],
        'cmap'            : sns.color_palette("Blues", n_colors=7, as_cmap=True),
        'fraction'        : 0.25,
        'aspect'          : None,
        'reverse'         : True,
        'xy'              : (0.5, 0.1),
        'title'           : "Counts",
        'title_fontsize'  : 12,
        'title_pad'       : 6,
        'ticks_rotation'  : 0,
        'ticks_length'    : 5,
        'ticks_labelsize' : 8,
        'ticks_pad'       : 4,
    })


class _HeatmapPlot(object):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self._automatic_config()

    def _automatic_config(self):
        if self.config.cbar["aspect"] is None:
            self.config.cbar["aspect"] = len(self.config.cbar["boundaries"])-1

        # Color for na in ratios 
        cmap = copy.copy(self.config.cbar["cmap"])
        cmap.set_bad("#F2F2F2")
        self.config.cbar["cmap"] = cmap

    def _build_figure_layout(self):
        fig = plt.figure(figsize=self.config.figure["figsize"], dpi=self.config.figure["dpi"])
        gs = fig.add_gridspec(self.config.figure["n_grid"], self.config.figure["n_grid"])
        gridspecs = {}
        gridspecs["cbar"] = gs[:, -1]
        gridspecs["heatmap"] = gs[:, :-1]
        return fig, gridspecs

    def _colorbar_ax(self, fig, gs_parent, x=0, y=0.5, orientation="horizontal", fraction=0.2, aspect=10):
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("%s is not a valid value for orientation; supported values are 'vertical', 'horizontal'" %
                             orientation)

        if orientation == "horizontal":
            if x > (1-fraction):
                raise ValueError("Set the x anchor position to value less than 1-fraction")
            else:
                if x==0:
                    width_ratios = [fraction, 1-fraction]
                    gs = gs_parent.subgridspec(1,2,wspace=0,width_ratios=width_ratios)
                    ss_cbar = gs[0]
                elif x==1:
                    width_ratios = [1-fraction, fraction]
                    gs = gs_parent.subgridspec(1,2,wspace=0,width_ratios=width_ratios)
                    ss_cbar = gs[1]
                else:
                    width_ratios = [x, fraction, 1-x-fraction]
                    gs = gs_parent.subgridspec(1,3,wspace=0,width_ratios=width_ratios)
                    ss_cbar = gs[1]

            cax = fig.add_subplot(ss_cbar)
            cax.set_aspect(1/aspect, anchor=(1,y), adjustable="box")
        else:
            if y > fraction:
                raise ValueError("Set the y anchor position to value less than fraction")
            else:
                if y==0:
                    height_ratios = [fraction, 1-fraction]
                    gs = gs_parent.subgridspec(2,1,hspace=0,height_ratios=height_ratios)
                    ss_cbar = gs[0]
                elif y==1:
                    height_ratios = [1-fraction, fraction]
                    gs = gs_parent.subgridspec(2,1,hspace=0,height_ratios=height_ratios)
                    ss_cbar = gs[1]
                else:
                    height_ratios = [y, fraction, 1-y-fraction]
                    gs = gs_parent.subgridspec(3,1,hspace=0,height_ratios=height_ratios)
                    ss_cbar = gs[1]

            cax = fig.add_subplot(ss_cbar)
            cax.set_aspect(aspect, anchor=(x,1), adjustable="box")

        return cax

    def _plot_colorbar(self, cax, cmap, labels, reverse, orientation, title, title_fontsize, title_pad,
                       ticks_rotation, ticks_length, ticks_labelsize, ticks_pad):

        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("%s is not a valid value for orientation; supported values are 'vertical', 'horizontal'" %
                             orientation)

        colors = [cmap(i/len(labels)) for i in range(1, len(labels))]

        if reverse:
            cols = colors[::-1]
            labs = labels[::-1]
        else:
            cols = colors
            labs = labels

        cax.margins(x=0,y=0)
        positions = [i/(len(colors)-1) for i in range(len(colors))]

        if orientation=="horizontal":
            cax.set_xlim([0, 1])
            cax.set_ylim([0, 1])
            cax.bar(x=positions,
                    height=1,
                    color=cols,
                    align="edge",
                    width=1)

            cax.spines["left"].set_visible(False)
            cax.spines["bottom"].set_visible(False)
            cax.spines["top"].set_visible(True)
            cax.spines["right"].set_visible(False)

            # ticks
            cax.xaxis.tick_top()
            cax.set_xticks(positions+[len(colors)/(len(colors)-1)])
            cax.set_xticklabels(labs, va="center")
            cax.tick_params(axis='x',
                            which='major',
                            labelrotation=ticks_rotation,
                            labelsize=ticks_labelsize,
                            length=ticks_length,
                            pad=ticks_pad)
            cax.set_yticks([])

            # title
            cax.xaxis.set_label_position("top")
            cax.set_xlabel(title, fontsize=title_fontsize, pad=title_pad)
        else:
            cax.set_xlim([0, 1])
            cax.set_ylim([0, 1])
            cax.barh(y=positions,
                     width=1,
                     color=cols,
                     align="edge",
                     height=1)

            cax.spines["left"].set_visible(False)
            cax.spines["bottom"].set_visible(False)
            cax.spines["top"].set_visible(False)
            cax.spines["right"].set_visible(True)

            # ticks
            cax.yaxis.tick_right()
            cax.set_yticks(positions+[len(colors)/(len(colors)-1)])
            cax.set_yticklabels(labs, ha="left")
            cax.tick_params(axis='y',
                            which='major',
                            labelrotation=ticks_rotation,
                            labelsize=ticks_labelsize,
                            length=ticks_length,
                            pad=ticks_pad)
            cax.set_xticks([])

            # title
            cax.set_title(title, fontsize=title_fontsize, pad=title_pad)

    def _plot_heatmap(self, ax):
        sns.heatmap(
            self.df,
            linecolor   = self.config.heatmap["linecolor"],
            linewidths  = self.config.heatmap["linewidths"],
            cmap        = self.config.cbar["cmap"],
            square      = self.config.heatmap["square"],
            xticklabels = self.config.heatmap["xticklabels"],
            yticklabels = self.config.heatmap["yticklabels"],
            ax          = ax,
            norm        = cm.colors.BoundaryNorm(boundaries=self.config.cbar["boundaries"], ncolors = 256),
            cbar_ax     = None,
            cbar        = False,
        )

        # Labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis           = 'x',
            which          = 'major',
            labelrotation  = self.config.heatmap["xticks_labelrotation"],
            labelsize      = self.config.heatmap["ticks_labelsize"],
            length         = 0,
        )
        ax.tick_params(
            axis           = 'y',
            which          = 'major',
            labelrotation  = self.config.heatmap["yticks_labelrotation"],
            labelsize      = self.config.heatmap["ticks_labelsize"],
            length         = 0,
        )

    def plot_heatmap(self):
        # layout
        fig, gridspecs = self._build_figure_layout()

        # matplotlib axes
        axes = {}

        # colorbar
        config = self.config.cbar
        axes["cbar"] = self._colorbar_ax(fig=fig, gs_parent=gridspecs["cbar"],
                                         x=config["xy"][0], y=config["xy"][1],
                                         fraction=config["fraction"],
                                         aspect=config["aspect"],
                                         orientation="vertical")

        self._plot_colorbar(cax=axes["cbar"], cmap=config["cmap"], labels=config["boundaries"],
                            reverse=config["reverse"], orientation="vertical",
                            title=config["title"], title_fontsize=config["title_fontsize"],
                            title_pad=config["title_pad"],
                            ticks_rotation=config["ticks_rotation"], ticks_length=config["ticks_length"],
                            ticks_labelsize=config["ticks_labelsize"], ticks_pad=config["ticks_pad"])

        # heatmap
        axes["heatmap"] = fig.add_subplot(gridspecs["heatmap"])
        self._plot_heatmap(ax=axes["heatmap"])

        return fig, axes


def plot_heatmap(df: DataFrame, config: HeatmapConfig):
    """
    Plots a simple heatmap.

    Many parameters are customizable, see :func:`~prettypy.heatmap.HeatmapConfig.`

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
    config: HeatmapConfig
       Graphical parameters.

    Returns
    -------
    fig, axes: matplotlib.pyplot.Figure, dict
    """
    plotter = _HeatmapPlot(df=df,
                           config=config)
    return plotter.plot_heatmap()
