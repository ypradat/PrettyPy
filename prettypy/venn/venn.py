# -*- coding: utf-8 -*-
"""
@created: 12/15/20
@modified: 12/15/20
@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France


Defines internal classes user-level functions for building and plotting venn plots.
"""

from   dataclasses import dataclass, field
import itertools
import numpy        as    np
import pandas       as    pd
import os
from   typing      import Dict, List, Tuple, Union

import matplotlib.pyplot as plt

# type aliases
DataFrame  = pd.core.frame.DataFrame

@dataclass
class VennConfig:
    figsize: tuple = (8,8)
    colors: List = field(default_factory=lambda: ["#59CD90","#3FA7D6"])
    alpha: float=1 # transparency level of colors
    enhance_linewidth: float=0.5
    enhance_linestyle: str="dashed"
    offset_label: List = field(default_factory=lambda: [0.1, 0])
    arrow_r: float = 0.5 # arrow tip positioned at r * xy_label + (1-r) * xy_center
    arrow_color: str = "black"
    arrow_connection_arc: int = 3
    arrow_connection_rad: float = 0.25

class _VennPlot(object):
    """
    Takes a dataframe along some information about columns in the dataframe and makes a venn plot
    """
    def __init__(self, config: VennConfig):
        self.config = config

    def _get_venn_params(self, table, col_label, col_identifier):
        set_labels = sorted(table[col_label].unique().tolist())
        n_labels = len(set_labels)
        set_values = {label: table.loc[table[col_label]==label, col_identifier].unique() for label in set_labels}

        set_indices = list(itertools.product([0, 1], repeat=n_labels))
        set_indices.remove((0,)*n_labels)

        subsets = {}
        for set_ind in set_indices:
            values_1 = []
            values_0 = []
            for set_i, set_l in zip(set_ind, set_labels):
                if set_i:
                    values_1.append(set_values[set_l])
                else:
                    values_0.append(set_values[set_l])

            set_ind_name = "".join([str(x) for x in set_ind])

            if len(values_1) == 0:
                values_1 = set()
            else:
                values_1 = set.intersection(*map(set,values_1))

            if len(values_0) == 0:
                values_0 = set()
            else:
                values_0 = set.union(*map(set,values_0))

            size = len(values_1.difference(values_0))
            subsets[set_ind_name] = size

        return subsets, set_labels

    def _get_venn_funcs(self, n):
        if n==2:
            from  matplotlib_venn  import venn2, venn2_circles
            return venn2, venn2_circles
        elif n==3:
            from  matplotlib_venn  import venn3, venn3_circles
            return venn3, venn3_circles
        elif n==4:
            from  matplotlib_venn  import venn4, venn4_circles
            return venn4, venn4_circles
        elif n==5:
            from  matplotlib_venn  import venn5, venn5_circles
            return venn5, venn5_circles
        else:
            raise ValueError("Only venn plots of 2,3,4 or 5 sets are supported")

    def _make_venn_labels(self, v, subsets, set_labels, ax):
        for sub_in in subsets.keys():
            sum_sub_in =  sum([int(s) for s in sub_in])
            if sum_sub_in == 1:
                for j, ind in enumerate(sub_in):
                    if ind=='1':
                        break

                label_id = chr(ord('A')+j)
                xy_label = v.get_label_by_id(label_id).get_position()
                xy_center = v.get_circle_center(id=j)

                xy = (self.config.arrow_r * xy_label[0] + (1-self.config.arrow_r) * xy_center[0],
                      self.config.arrow_r * xy_label[1] + (1-self.config.arrow_r) * xy_center[1])

                v.get_label_by_id(label_id).set_text("")

                if j % 2 == 0:
                    xytext = xy_label - np.array(self.config.offset_label)
                    connectionstyle = 'arc%d,rad=-%f' % (self.config.arrow_connection_arc,
                                                         self.config.arrow_connection_rad)
                else:
                    xytext = xy_label + np.array(self.config.offset_label)
                    connectionstyle = 'arc%d,rad=%f' % (self.config.arrow_connection_arc,
                                                        self.config.arrow_connection_rad)

                ax.annotate(set_labels[j], xy=xy, xytext=xytext, ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=5,
                                            connectionstyle=connectionstyle,
                                            color=self.config.arrow_color))

    def plot_venn(self, df, col_set, col_identifier, ax=None):
        """
        Main function of the class
        """
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(figsize=self.config.figsize)
        else:
            ax_was_none = False

        subsets, set_labels = self._get_venn_params(df, col_label=col_set, col_identifier=col_identifier)
        venn_func, venn_circles_func = self._get_venn_funcs(n=len(set_labels))

        v = venn_func(subsets=subsets, set_labels=set_labels, set_colors=self.config.colors,
                      alpha=self.config.alpha, normalize_to=1, ax=ax)
        c = venn_circles_func(subsets=subsets, linestyle=self.config.enhance_linestyle, normalize_to=1,
                              linewidth=self.config.enhance_linewidth)
        self._make_venn_labels(v, subsets, set_labels, ax)

        if ax_was_none:
            return fig, ax


def plot_venn(df, col_set, col_identifier, config, ax=None):
    """
    Make a venn plot from a dataframe. The sets are identified by the column `field_set` and `col_identifier`.

    Parameters
    ----------
    df: DataFrame
    field_set: str
    col_identifier: str
    ax: matplolib axes object

    Returns
    -------
    None or a tuple (fig,ax) if input ax was None
    """
    plotter = _VennPlot(config=config)
    return plotter.plot_venn(df=df, col_set=col_set, col_identifier=col_identifier, ax=ax)
