# -*- coding: utf-8 -*-
"""
@created: 02/25/21
@modified: 02/25/21
@author: Yoann Pradat
@reference: https://github.com/fomightez/donut_plots_with_subgroups

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Function for making a donut plot. The code was largely taken from the github repo fomightez/donut_plots_with_subgroups.
"""

import pandas as pd
import numpy as np

import copy
from   dataclasses import dataclass, field
import numpy        as    np
import pandas       as    pd
import os
from   typing      import Dict, List, Tuple, Union
from   tqdm        import tqdm
import sys

import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class DonutConfig:
    figsize: tuple = (8,8)
    title: str = ""
    title_fontsize: int = 24 # font size for title above plot
    plot_fontsize: int = 14  # font size for text in the plot
    show_group_sizes: bool = True # should group sizes be indicated in labels?
    show_subgroup_sizes: bool = True # should group sizes be indicated in labels?
    sizes_fmt: str = "%s\n(%d)" # should group sizes be indicated below the labels?
    prefix_subgroup_names: bool = False # if True, subgroup names are prefixed by the group name
    group_labeldistance: float = 1.2
    subgroup_labeldistance: float = 0.8

    colors: List = field(default_factory=lambda: ["Blues","Reds", "Greens","Oranges", "Purples"])

    outer_ring_radius: float = 1.3 # radius of the outer ring of the donut plot
    inner_ring_radius: float = 1 # radius of the inner ring of donut
    outer_ring_width: float = 0.3
    inner_ring_width: float = 0.4

    sort_on_subgroup_name: bool = True # if False, sort on subgroup values
    advance_color_increments: int = 0 # set to more than 0 to skip some colors
    hilolist: List = field(default_factory=lambda: []) # High to low list of intensity values
    light_color_for_last_in_subgroup: bool = True # Set this to False to reverse the order of the subgroup coloring.


class _DonutPlot(object):
    """
    Takes a dataframe along some information about columns in the dataframe and makes a donut plot.  The plot is a
    breakdown of the main groups to subgroups with the main groups in an outer ring of the dount plot and the subgroups
    on the inner ring. The style sought is available at https://python-graph-gallery.com/163-donut-plot-with-subgroups/.
    """
    def __init__(self, config: DonutConfig):
        self.config = config

    def _sequential_color_maps_generator(self):
        """
        Generator to yield a never-ending supply of sequential color palettes/ color maps.
        """
        for color in self.config.colors:
            # `plt.get_cmap` use based on https://matplotlib.org/tutorials/colors/colormaps.html
            try:
                yield plt.get_cmap(color)
            except ValueError:
                try:
                    yield sns.light_palette(color, as_cmap=True)
                except ValueError:
                    try:
                        yield sns.light_palette(color, as_cmap=True,input="xkcd")
                    except:
                        yield sns.light_palette(rgb, input="rgb", as_cmap=True)

    def _is_number(self, s):
        """
        Check if a string can be cast to a float or numeric (integer).
        Fixed from https://www.pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/
        later noted similar code is at https://code-maven.com/slides/python-programming/is-number
        """
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def _cast_to_number(self, s):
        """
        Cast a string to a float or integer. Based on fixed code from
        https://www.pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/
        """
        try:
            number = float(s)
            try:
                number = int(s)
                return number
            except ValueError:
                pass
            return number
        except ValueError:
            pass
        try:
            import unicodedata
            num = unicodedata.numeric(s)
            return num
        except (TypeError, ValueError):
            pass
        return False

    def _remove_dup_keep_order(self, seq):
        """
        Remove duplicates from a list whilst preserving order. From https://stackoverflow.com/a/480227/8508004
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def plot_donut(self, df, col_groups, col_subgroups):
        """
        Main function of the class
        """
        # Prepare derivatives of the dataframe that may be needed for delineating
        # the plotting data
        tc = df[col_subgroups].value_counts()
        total_state_names = tc.index.tolist()
        total_state_size = tc.tolist()
        grouped = df.groupby(col_groups)

        # use `value_counts()` on each group to get the count and name of each state
        list_o_subgroup_names_l = []
        list_o_subgroup_size_l = []
        subgroups_per_group_l = []
        for name, group in grouped:
            dfc = group[col_subgroups].value_counts()
            if self.config.sort_on_subgroup_name:
                dfc = group[col_subgroups].value_counts().sort_index()
            # to make the subgroup names like in the example, incorporate
            # group name to each as well
            list_o_subgroup_names_l.append([(name,x) for x in dfc.index.tolist()])
            list_o_subgroup_size_l.append(dfc.tolist())

        # Delineate data for the plot:
        group_names= grouped.size().index.tolist()
        group_size= grouped.size().tolist() #len of each groupby grouping

        # flatten each list of lists made above to get the list needed
        if self.config.prefix_subgroup_names:
            subgroup_names=["%s.%s" % i for sublt in list_o_subgroup_names_l for i in sublt]
        else:
            subgroup_names=[i[1] for sublt in list_o_subgroup_names_l for i in sublt]
        subgroup_size=[i for sublt in list_o_subgroup_size_l for i in sublt]
        assert len(subgroup_size) == len(subgroup_names)

        # Create colors generator and colors
        colormp = self._sequential_color_maps_generator()
        [next(colormp) for g in range(self.config.advance_color_increments)]
        colorm_per_grp=[next(colormp) for g in group_names]

        fig, ax = plt.subplots(figsize=self.config.figsize)

        ### First Ring (outside)
        ### This will be the main groups
        if self.config.show_group_sizes:
            labels = [self.config.sizes_fmt % (x,y) for x, y in zip(group_names, group_size)]
        else:
            labels = ["%s" % x for x in group_names]
        mypie, _ = plt.pie(
            group_size, radius=self.config.outer_ring_radius, labels=labels, startangle=90,
            textprops={'fontsize': self.config.plot_fontsize, 'ha': 'center', 'va': 'center'},
            labeldistance=self.config.group_labeldistance, colors=[colormp(0.63) for colormp in colorm_per_grp] )
        plt.setp(mypie, width=self.config.outer_ring_width, edgecolor='white')

        ### Second Ring (Inside)
        ### This will be the subgroup counting for each group
        list_sub_grp_colors_l  = []
        subgroups_represented = self._remove_dup_keep_order(df[col_subgroups].tolist())

        if len(self.config.hilolist) > 0:
            assert len(self.config.hilolist) == len(subgroups_represented), "The list provided "
            "to specify the intensity degree must include all subgroups. Subgroups "
            "are: '{}'.format(subgroups_represented)"
            subgroups_represented = self.config.hilolist
        else:
            # Provide feedback on what is being used as high to low intensity list
            # so user can adjust; using `if __name__ == "__main__"` to customize
            # note depending if script called from command line.
            sys.stderr.write("Note: No list to specify high to low intensity coloring "
                "provided, and so using '{}',\nwhere leftmost identifer corresponds "
                "to most intense and rightmost is least.\n".format(
                ",".join(str(i) for i in subgroups_represented))) # because subgroups
            # could be integers as in example from
            # https://python-graph-gallery.com/163-donut-plot-with-subgroups/, best
            # to have conversion to string,
            if __name__ == "__main__":
                sys.stderr.write("Look into adding use of the `--hilolist` option "
                    "to specify the order.\n\n")
            else:
                sys.stderr.write("Provide a Python list as `hilolist` when calling "
                    "the function to specify the order.\n\n")

        # assign intensity degree settings for each subgroup so consistent among
        # other groups
        int_degree = np.linspace(0.6, 0.2, num=len(subgroups_represented))
        if not self.config.light_color_for_last_in_subgroup:
            int_degree = int_degree[::-1]

        # determine colors for each subgroup before `plt.pie` step
        for idx,subgroups_l in enumerate(list_o_subgroup_names_l):
            cm = colorm_per_grp[idx]
            grp_colors = [cm(int_degree[subgroups_represented.index(sgrp[1])]) for sgrp in subgroups_l]
            # `int(sgrp.split(".")[1])` is
            # to get the subgroup name back to an integer, e.g., `C.3` becomes `3`.
            list_sub_grp_colors_l.append(grp_colors)

        # flatten that list
        if self.config.show_subgroup_sizes:
            labels = [self.config.sizes_fmt % (x,y) for x, y in zip(subgroup_names, subgroup_size)]
        else:
            labels = ["%s" % x for x in subgroup_names]
        sub_grp_colors = [i for sublt in list_sub_grp_colors_l for i in sublt]

        mypie2, _ = plt.pie(
            subgroup_size, radius=self.config.inner_ring_radius, labels=labels, startangle=90,
            textprops={'fontsize': self.config.plot_fontsize, 'ha': 'center', 'va': 'center'},
            labeldistance=self.config.subgroup_labeldistance, colors=sub_grp_colors)
        plt.setp(mypie2, width=self.config.inner_ring_width, edgecolor='white')
        plt.margins(0,0)

        # title
        plt.suptitle(self.config.title, fontsize=self.config.title_fontsize, fontweight="bold")

        return fig, ax


def plot_donut(df, col_groups, col_subgroups, config: DonutConfig):
    """
    Takes a dataframe along some information about columns in the dataframe and makes a donut plot.  The plot is a
    breakdown of the main groups to subgroups with the main groups in an outer ring of the dount plot and the subgroups
    on the inner ring. The style sought is available at https://python-graph-gallery.com/163-donut-plot-with-subgroups/.
    """
    plotter = _DonutPlot(config=config)
    return plotter.plot_donut(df=df, col_groups=col_groups, col_subgroups=col_subgroups)
