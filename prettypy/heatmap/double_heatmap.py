# -*- coding: utf-8 -*-
"""
@created: 01/29/21
@modified: 01/29/21
@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France


Defines internal classes user-level functions for building and plotting double heatmaps.
"""

import copy
from   dataclasses import dataclass, field
import numpy        as    np
import pandas       as    pd
import os
from   typing      import Dict, List, Tuple, Union
from   tqdm        import tqdm
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.sandbox.stats.multicomp  as  mp

from scipy.stats import fisher_exact

# type aliases
DataFrame  = pd.core.frame.DataFrame
Vector = Union[List[float], np.ndarray, pd.core.series.Series]
Table  = Union[List[List[float]], np.ndarray, pd.core.frame.DataFrame]

def _cont_table(A: Vector, B: Vector) -> Table:
    """
    Compute 2 x 2 contignecy table from A and B binary vectors.

    Parameters
    ----------
    A: array-like
        A vector of binary entries
    B: array-like
        A vector of binary entries

    Returns
    -------
    tab: array-like
        A 2x2 contigency table.
    """
    tab = np.zeros((2, 2))
    A_anti = np.where(A==1, 0, 1)
    B_anti = np.where(B==1, 0, 1)
    tab[0,0] = np.sum(A*B)
    tab[0,1] = np.sum(A*B_anti)
    tab[1,0] = np.sum(A_anti*B)
    tab[1,1] = np.sum(A_anti*B_anti)
    return tab

def _odds_ratio(tab: Table) -> float:
    """
    Computes the odds ratio of a contigency table
    -------------------
        a        b
        c        d
    -------------------
    as (a/b)/(c/d) or ad/bc

    Parameters
    ----------
    tab: array-like
        The table.

    Returns
    -------
    _odds_ratio: float
    """

    if tab[0,1] == 0 or tab[1,0] == 0:
        _odds_ratio = tab[0,0] * tab[1,1] / max(tab[1,0], tab[0,1], 1)
    else:
        _odds_ratio = tab[0,0] * tab[1,1] / (tab[1,0] * tab[0,1])

    return _odds_ratio

class _DoubleHeatmapBuild(object):
    def __init__(self, pair_count="cooccurrence", pair_ratio="odds", pair_test="fisher_exact"):
        """
        Parameters
        ----------
        pair_count: str, default="cooccurrence"
            Either a string or a callable taking as input two iterables of the same size (lists or arrays) and that
            returns a float. For each pair of variables, this will be plotted in one half of the heatmap.
        pair_ratio: str, default="odds"
            Either a string, a dataframe or a callable taking as input two iterables of the same size (lists or arrays)
            and that returns a float. For each pair of variables, this will be plotted in one half of the heatmap.
        pair_test: str, default="fisher_exact"
            Either a string None or a callable taking as input two iterables of the same size (lists or arrays) and that
            returns a p-value. Pairs that have a significant test will have a star above their cell.
        """
        self.pair_count = pair_count
        self.pair_ratio = pair_ratio
        self.pair_test = pair_test

    def _pair_count(self, A, B):
        if isinstance(self.pair_ratio, str) and self.pair_count == "cooccurrence":
            assert set(A).issubset(set([0,1]))
            assert set(B).issubset(set([0,1]))
            return sum((A==1) & (B==1))
        elif isinstance(self.pair_count, Callable):
            return self.pair_count(A,B)
        else:
            raise ValueError("Invalid value for parameter 'pair_count'. Specify a Callable or one of 'cooccurrence'")

    def _pair_ratio(self, A, B):
        if isinstance(self.pair_ratio, str) and self.pair_ratio == "odds":
            c_tab = _cont_table(A, B)
            ratio = _odds_ratio(c_tab)
            return ratio
        elif isinstance(self.pair_ratio, Callable):
            return self.pair_ratio(A,B)
        else:
            raise ValueError("Invalid value for parameter 'pair_ratio'. Specify a Callable or one of 'cooccurrence'")

    def _pair_test(self, A, B):
        if self.pair_test is None:
            return None

        if type(self.pair_test) == str and self.pair_test == "fisher_exact":
            c_tab = _cont_table(A, B)
            _, pval = fisher_exact(c_tab)
            return pval
        else:
            return self.pair_test(A,B)

    def _build_half_matrix(self, df, pair, use_diagonal=True):
        """
        Builds a half matrix of size (n_var, n_var) from a matrix of size (n_obs, n_var).

        Parameters
        ----------
        df: array-like, (n_obs, n_var)
            It defines the values used to build the half matrix
        pair:
            A callable function taking as input two iterables of the same size (lists or arrays) and that returns a
            float. For each pair of variables, the float will be fill the half-matrix.

        Returns
        -------
        half_df:
            Half-filled matrix
        """
        vars = df.columns.tolist()
        n_vars = len(vars)
        m_half = []

        if use_diagonal:
            for i in tqdm(range(n_vars)):
                l_half = [np.nan for _ in range(n_vars)]
                for j in range(0, i + 1):
                    l_half[j] = pair(df[vars[i]], df[vars[j]])
                m_half.append(l_half)
        else:
            m_half.append([np.nan for _ in range(n_vars)])
            for i in tqdm(range(1, n_vars)):
                l_half = [np.nan for _ in range(n_vars)]
                for j in range(0, i):
                    l_half[j] = pair(df[vars[i]], df[vars[j]])
                m_half.append(l_half)

        df_half = pd.DataFrame(m_half, vars)
        df_half.columns = df.columns
        return df_half

    def build_half_matrices(self, df_values, df_active=None):
        """
        Builds one, two or three half-matrices from a matrix of activation and a matrix of values of size (n_obs, n_var).
        Each half-matrix is a square matrix of size (n_var, n_var).

        Parameters
        ----------
        df_values: array-like, (n_obs, n_var)
            It defines the values used to build the half matrices of ratios and tests in observations x variables
            format.
        df_active: array-like, (n_obs, n_var) default=None
            If None, df_active=df_values. It defines the binary activation indicator of variables in sample used to
            build the half matrix of counts in observations x variables format.

        Returns
        -------
        dfs: dict of dataframe
            Dict containing the half-matrices of "count", "ratio" and "test"
        """
        if df_active is None:
            df_active = df_values

        if self.pair_count is None:
            df_count = None
        else:
            df_count = self._build_half_matrix(df_active, self._pair_count)

        if self.pair_ratio is None:
            df_ratio = None
        else:
            df_ratio = self._build_half_matrix(df_values, self._pair_ratio, use_diagonal=False)

        if self.pair_test is None:
            df_test = None
        else:
            df_test = self._build_half_matrix(df_values, self._pair_test, use_diagonal=False)

        return {"count": df_count, "ratio": df_ratio, "test": df_test}


def build_double_heatmap(df_values, df_active=None, pair_count="cooccurrence", pair_ratio="odds",
                         pair_test="fisher_exact"):
        """
        Builds one, two or three half-matrices from a matrix of activation and a matrix of values of size (n_obs, n_var).
        Each half-matrix is a square matrix of size (n_var, n_var).

        Parameters
        ----------
        pair_count: str, default="cooccurrence"
            Either a string or a callable taking as input two iterables of the same size (lists or arrays) and that
            returns a float. For each pair of variables, this will be plotted in one half of the heatmap.
        pair_ratio: str, default="odds"
            Either a string, a dataframe or a callable taking as input two iterables of the same size (lists or arrays)
            and that returns a float. For each pair of variables, this will be plotted in one half of the heatmap.
        pair_test: str, default="fisher_exact"
            Either a string None or a callable taking as input two iterables of the same size (lists or arrays) and that
            returns a p-value. Pairs that have a significant test will have a star above their cell.

        Parameters
        ----------
        df_values: array-like, (n_obs, n_var)
            It defines the values used to build the half matrices of ratios and tests in observations x variables
            format.
        df_active: array-like, (n_obs, n_var) default=None
            If None, df_active=df_values. It defines the binary activation indicator of variables in sample used to
            build the half matrix of counts in observations x variables format.

        Returns
        -------
        dfs: dict of dataframe
            Dict containing the half-matrices of "count", "ratio" and "test"
        """
        builder = _DoubleHeatmapBuild(pair_count, pair_ratio, pair_test)
        return builder.build_half_matrices(df_values, df_active)


def default_field(obj):
    return field(default_factory=lambda: obj)

@dataclass
class DoubleHeatmapConfig:
    figure: Dict[str, Union[str,tuple]] = default_field({
        "figsize": (8,8),
        "dpi": 300,
        "n_grid": 10,
    })
    heatmap: Dict[str, Union[int, str, bool, float]] = default_field({
        "orientation"          : "antidiagonal",
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
    count: Dict[str, Union[int, float, str, bool]] = default_field({
        'boundaries'          : [1,5,10,15,20,50,200,500],
        'auto_boundaries'     : {"n": 7, "decimals": 0, "middle": None, "regular": True},
        'cmap'                : sns.color_palette("Blues", n_colors=7, as_cmap=True),
        'cbar_fraction'       : 0.25,
        'cbar_aspect'         : None,
        'cbar_reverse'        : True,
        'cbar_xy'             :  (0, 0.5),
        'cbar_title'          : "Counts",
        'cbar_title_fontsize' : 12,
        'cbar_title_pad'      : 6,
        'cbar_ticks_rotation' : 0,
        'cbar_ticks_length'   : 5,
        'cbar_ticks_labelsize': 8,
        'cbar_ticks_pad'      : 4,
    })
    ratio: Dict[str, Union[int, float, str]] = default_field({
        'boundaries'          : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'auto_boundaries'     : {"n": 7, "decimals": 0, "middle": None, "regular": True},
        'cmap'                : sns.diverging_palette(50, 200, s=90, l=50, sep=1, as_cmap=True),
        'cbar_fraction'       : 0.25,
        'cbar_aspect'         : None,
        'cbar_reverse'        : False,
        'cbar_xy'             : (0.5, 0.1),
        'cbar_title'          : "Ratios",
        'cbar_title_pad'      : 6,
        'cbar_title_fontsize' : 12,
        'cbar_ticks_rotation' : 0,
        'cbar_ticks_length'   : 5,
        'cbar_ticks_labelsize': 8,
        'cbar_ticks_pad'      : 4,
    })
    test: Dict[str, Union[int, float, str]] = default_field({
        'pval_level': 0.05,
        'fwer_level': 0.05,
        'fdr_level': 0.1,
        'fwer_size': 10,
        'fwer_marker': '*',
        'fwer_color': 'black',
        'fdr_size':  1,
        'fdr_marker':  's',
        'fdr_color': 'black',
    })


class _DoubleHeatmapPlot(object):
    def __init__(self, df_count: DataFrame, df_ratio: DataFrame, df_test: DataFrame,  config: DoubleHeatmapConfig):
        """
        Plots double heatmap.

        Parameters
        ----------
        df_count: pandas.core.frame.DataFrame
            Pandas half-filled dataframe of counts.
        df_ratio: pandas.core.frame.DataFrame
            Pandas half-filled dataframe of ratios.
        df_test: pandas.core.frame.DataFrame
            Pandas half-filled dataframe of p-values.
        config: DoubleHeatmapConfig
           Graphical parameters.
        """
        self.df_count = df_count.copy()
        self.df_ratio = df_ratio.copy()
        self.df_test = df_test.copy()
        self.n_var = self.df_count.shape[0]

        self.config = config
        self._check_config(config)
        self._automatic_config()

    def _check_config(self, config):
        for cmap in [self.config.ratio["cmap"], self.config.count["cmap"]]:
            if not isinstance(config.ratio["cmap"], cm.colors.LinearSegmentedColormap):
                raise ValueError("""Please specify color maps of that are instances of LinearSegmentedColormap
                                 as produced by the sns.color_palette with cmap=True function for instance""")

        if self.config.heatmap["orientation"] not in ["diagonal", "antidiagonal"]:
            raise ValueError("%s is invalid for heatmap orientation. Choose 'diagonal' or 'antidiagonal'" %
                             self.config.heatmap["orientation"] == "antidiagonal")

    def _automatic_boundaries(self, df, use_diagonal=True, n=9, middle=None, decimals=1):
        if use_diagonal:
            vals = np.array([self.df_ratio.iloc[i,j] for i in range(self.n_var) for j in range(i)])
        else:
            vals = np.array([self.df_ratio.iloc[i,j] for i in range(1,self.n_var) for j in range(i-1)])

        min_val = np.round(min(vals), decimals=decimals)
        max_val = np.round(max(vals), decimals=decimals)

        if middle is not None:
            below_middle = pd.qcut(vals[vals < middle], q=(n-1)//2).categories.mid.values
            below_middle = np.round(below_middle, decimals=decimals)
            above_middle = pd.qcut(vals[vals > middle], q=(n-1)//2).categories.mid.values
            above_middle = np.round(above_middle, decimals=decimals)
            boundaries = [min_val] + below_middle + [middle] + above_middle + [min_val]
        else:
            inbetween = pd.qcut(vals, q=n-1).categories.mid.values
            inbetween = np.round(inbetween, decimals=decimals)
            boundaries = [min(vals)] + inbetween + [max(vals)]

        boundaries = sort(list(set(boundaries)))
        return boundaries

    def _automatic_config(self):
        if self.config.count["boundaries"] is None:
             boundaries = self._automatic_boundaries(df=self.df_count, use_diagonal=True,
                                                     n=self.config.count["auto_boundaries"]["n"],
                                                     middle=self.config.count["auto_boundaries"]["middle"],
                                                     decimals=self.config.count["auto_boundaries"]["decimals"])
             self.config.count["boundaries"] = boundaries

        if self.config.ratio["boundaries"] is None:
             boundaries = self._automatic_boundaries(df=self.df_ratio, use_diagonal=False,
                                                     n=self.config.ratio["auto_boundaries"]["n"],
                                                     middle=self.config.ratio["auto_boundaries"]["middle"],
                                                     decimals=self.config.ratio["auto_boundaries"]["decimals"])
             self.config.ratio["boundaries"] = boundaries

        if self.config.count["cbar_aspect"] is None:
            self.config.count["cbar_aspect"] = len(self.config.count["boundaries"])-1

        if self.config.ratio["cbar_aspect"] is None:
            self.config.ratio["cbar_aspect"] = len(self.config.ratio["boundaries"])-1

        # Color for na in ratios 
        cmap = copy.copy(self.config.ratio["cmap"])
        cmap.set_bad("#F2F2F2")
        self.config.ratio["cmap"] = cmap

    def _hide_non_significant(self):
        for i in range(self.n_var):
            for j in range(0, i):
                if self.df_test.iloc[i, j] > self.config.test["pval_level"]:
                    self.df_ratio.iloc[i, j] = np.nan

    def _build_figure_layout(self):
        fig = plt.figure(figsize=self.config.figure["figsize"], dpi=self.config.figure["dpi"])
        gs = fig.add_gridspec(self.config.figure["n_grid"], self.config.figure["n_grid"])
        gridspecs = {}
        gridspecs["count_cbar"] = gs[0, :-1]
        gridspecs["ratio_cbar"] = gs[1:, -1]
        gridspecs["heatmap"] = gs[1:, :-1]
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

    def plot_double_heatmap(self):
        # layout
        fig, gridspecs = self._build_figure_layout()

        # matplotlib axes
        axes = {}

        # colorbars
        config = self.config.count
        axes["count_cbar"] = self._colorbar_ax(fig=fig, gs_parent=gridspecs["count_cbar"],
                                               x=config["cbar_xy"][0], y=config["cbar_xy"][1],
                                               fraction=config["cbar_fraction"],
                                               aspect=config["cbar_aspect"],
                                               orientation="horizontal")

        self._plot_colorbar(cax=axes["count_cbar"], cmap=config["cmap"], labels=config["boundaries"],
                            reverse=config["cbar_reverse"], orientation="horizontal",
                            title=config["cbar_title"], title_fontsize=config["cbar_title_fontsize"],
                            title_pad=config["cbar_title_pad"],
                            ticks_rotation=config["cbar_ticks_rotation"], ticks_length=config["cbar_ticks_length"],
                            ticks_labelsize=config["cbar_ticks_labelsize"], ticks_pad=config["cbar_ticks_pad"])


        config = self.config.ratio
        axes["ratio_cbar"] = self._colorbar_ax(fig=fig, gs_parent=gridspecs["ratio_cbar"],
                                               x=config["cbar_xy"][0], y=config["cbar_xy"][1],
                                               fraction=config["cbar_fraction"],
                                               aspect=config["cbar_aspect"],
                                               orientation="vertical")

        self._plot_colorbar(cax=axes["ratio_cbar"], cmap=config["cmap"], labels=config["boundaries"],
                            reverse=config["cbar_reverse"], orientation="vertical",
                            title=config["cbar_title"], title_fontsize=config["cbar_ticks_labelsize"],
                            title_pad=config["cbar_title_pad"],
                            ticks_rotation=config["cbar_ticks_rotation"], ticks_length=config["cbar_ticks_length"],
                            ticks_labelsize=config["cbar_ticks_labelsize"], ticks_pad=config["cbar_ticks_pad"])

        # heatmap
        axes["heatmap"] = fig.add_subplot(gridspecs["heatmap"])

        self._hide_non_significant()
        self._plot_heatmap(ax=axes["heatmap"])
        self._plot_significant(ax=axes["heatmap"])

        return fig, axes


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
            cax.set_xlabel(title, fontsize=title_fontsize)
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

    def _plot_significant(self, ax):
        pvals = []
        for i in range(self.n_var):
            for j in range(0, i):
                pvals.append(self.df_test.iloc[i, j])

        mask_fdr = mp.multipletests(pvals, method="fdr_bh", alpha=self.config.test["fdr_level"])[0]
        mask_fwer = mp.multipletests(pvals, method="bonferroni", alpha=self.config.test["fwer_level"])[0]
        k = 0
        x_fwer, y_fwer, x_fdr, y_fdr = [], [], [], []
        for i in range(self.n_var):
            for j in range(0, i):
                if not np.isnan(self.df_ratio.iloc[i, j]):
                    if mask_fwer[k]:
                        x_fwer.append(i + 0.5)

                        if self.config.heatmap["orientation"] == "antidiagonal":
                            y_fwer.append(self.n_var - 1 - j + 0.5)
                        else:
                            y_fwer.append(j + 0.5)
                    elif mask_fdr[k]:
                        x_fdr.append(i + 0.5)

                        if self.config.heatmap["orientation"] == "antidiagonal":
                            y_fdr.append(self.n_var - 1 - j + 0.5)
                        else:
                            y_fdr.append(j + 0.5)
                k += 1

        #### scatter plots fdr and fwer
        self._scatter_fdr = ax.scatter(
            x      = x_fdr,
            y      = y_fdr,
            s      = self.config.test['fdr_size'],
            marker = self.config.test["fdr_marker"],
            c      = self.config.test["fdr_color"],
            label  = "FDR < %.1g" % (self.config.test["fdr_level"])
        )
        self._scatter_fdr = ax.scatter(
            x      = x_fwer,
            y      = y_fwer,
            s      = self.config.test['fwer_size'],
            marker = self.config.test["fwer_marker"],
            c      = self.config.test["fwer_color"],
            label  = "FWER < %.1g" % (self.config.test["fwer_level"])
        )

    def _plot_heatmap(self, ax):
        if self.config.heatmap["orientation"] == "antidiagonal":
            dfs = [self._reverse_half_matrix(self.df_ratio[::-1]), self.df_count[::-1]]
        else:
            dfs = [self._reverse_half_matrix(self.df_ratio), self.df_count]

        for config, df in zip([self.config.ratio, self.config.count], dfs):
            sns.heatmap(
                df,
                linecolor   = self.config.heatmap["linecolor"],
                linewidths  = self.config.heatmap["linewidths"],
                cmap        = config["cmap"],
                square      = self.config.heatmap["square"],
                xticklabels = self.config.heatmap["xticklabels"],
                yticklabels = self.config.heatmap["yticklabels"],
                ax          = ax,
                norm        = cm.colors.BoundaryNorm(boundaries=config["boundaries"], ncolors=256, extend="both"),
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

    def _reverse_half_matrix(self, df):
        row_order = df.index
        col_order = df.columns
        df_T = df.T.loc[row_order, col_order]
        return df_T


def plot_double_heatmap(df_count: DataFrame, df_ratio: DataFrame, df_test: DataFrame,  config: DoubleHeatmapConfig):
    """
    Plots a heatmap of two half-filled matrices.

    The first half-filled heatmap is drawn from the values of df_count while the second is drawn from the values of
    df_ratio. The third dataframe, df_test, allows to optionally add markers on the ratio cells that are significant
    according to some statistical test.

    The orientation of the heatmap may be either 'antidiagonal' or 'diagonal'. This parameter may be set in the config
    object. Many other parameters are customizable, see :func:`~prettypy.heatmap.DoubleHeatmapConfig.`

    Parameters
    ----------
    df_count: pandas.core.frame.DataFrame
        Pandas half-filled dataframe of counts.
    df_ratio: pandas.core.frame.DataFrame
        Pandas half-filled dataframe of ratios.
    df_test: pandas.core.frame.DataFrame
        Pandas half-filled dataframe of p-values.
    config: DoubleHeatmapConfig
       Graphical parameters.

    Returns
    -------
    fig, axes: matplotlib.pyplot.Figure, dict
    """
    plotter = _DoubleHeatmapPlot(df_count=df_count,
                                 df_ratio=df_ratio,
                                 df_test=df_test,
                                 config=config)
    return plotter.plot_double_heatmap()
