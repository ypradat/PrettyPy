# -*- coding: utf-8 -*-
"""
@modified: Dec 02 2020
@created: Dec 02 2020
@author: Yoann Pradat
@reference: https://github.com/ImSoErgodic/py-upset

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Update and fixes of the code from https://github.com/ImSoErgodic/py-upset.
"""

from   itertools          import chain, combinations
from   functools          import partial
from   matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot   as    plt
from   matplotlib         import gridspec
import numpy               as    np
import pandas              as    pd


class DrawPyUpsetPlot(object):
    def __init__(self, field_set: str, df: pd.core.frame.DataFrame, color_bar: list=[245/255, 170/255, 50/255, 1],
                 color_que: list=[199/255, 30/255, 30/255, 1]):
        self.field_set = field_set
        self.df = df
        self.color_bar = color_bar
        self.color_que = color_que

    def _get_hue_name(self, hue_row: pd.core.series.Series, dt_names: dict):
        hue_ll = []

        for k,v in sorted(hue_row.to_dict().items()):
            if dt_names is None or k not in dt_names.keys():
                hue_ll += [k,v]
            else:
                if dt_names[k]["key"]:
                    hue_ll += [k]
                if v in dt_names[k].keys():
                    hue_ll += [dt_names[k][v]]
                else:
                    hue_ll += [v]

        hue_name = "_".join(hue_ll)
        return hue_name

    def _get_data_dict(self, hue_vars, fields2vals_keep=None, fields2vals_drop=None, dt_names=None) -> dict:
        dt_data = {}

        mask_keep = pd.Series(True, index=self.df.index)
        if fields2vals_keep is not None:
            for (field, vals) in fields2vals_keep.items():
                if not type(vals)==list:
                    vals = [vals]
                mask_keep = mask_keep & self.df[field].isin(vals)

        if fields2vals_drop is not None:
            for (field, vals) in fields2vals_drop.items():
                if not type(vals)==list:
                    vals = [vals]
                mask_keep = mask_keep & ~self.df[field].isin(vals)

        df_mask = self.df.loc[mask_keep]
        df_hue_unique = df_mask[hue_vars].drop_duplicates()

        for i, hue_row in df_hue_unique.iterrows():

            hue_mask = pd.Series(True, index=df_mask.index)
            for hue_var in hue_vars:
                hue_mask =  hue_mask & (df_mask[hue_var] == hue_row[hue_var])

            hue_name = self._get_hue_name(hue_row, dt_names)
            dt_data[hue_name] = pd.DataFrame({self.field_set: df_mask.loc[hue_mask, self.field_set].unique()})

        return dt_data

    def draw(self, hue_vars, fields2vals_keep=None, fields2vals_drop=None, dt_names=None, height_ratio=4,
             width_setsize=1, width_names=2, names_fontsize=8, circle_size=75, figsize: tuple=(16, 9),
             inters_min=1) -> dict:

        data_dict = self._get_data_dict(hue_vars, fields2vals_keep, fields2vals_drop, dt_names)

        #### draw
        dt_fig = plot_pyupset(
            data_dict          = data_dict,
            figsize            = figsize,
            unique_keys        = [self.field_set],
            inters_size_bounds = (inters_min, np.inf),
            query              = [tuple(data_dict.keys())],
            colors_query       = [self.color_que],
            color_vbar         = self.color_bar,
            color_hbar         = self.color_bar,
            vbar_rot           = 45,
            vbar_fmt           = "d",
            height_ratio       = height_ratio,
            width_setsize      = width_setsize,
            width_names        = width_names,
            names_fontsize     = names_fontsize,
            wspace             = 0,
            circle_size        = 75
        )

        return dt_fig

    def save(self, filename: str):
        current_wd = setwd_to_results()

def plot_pyupset(data_dict,
                 figsize,
                 unique_keys=None,
                 sort_by='size',
                 inters_size_bounds=(0, np.inf),
                 inters_degree_bounds=(1, np.inf),
                 additional_plots=None,
                 names_fontsize=14,
                 query=None,
                 colors_query=None,
                 color_vbar=None,
                 color_hbar=None,
                 color_matr=None,
                 vbar_rot=90,
                 vbar_fmt=".2g",
                 circle_size=300,
                 height_ratio=4,
                 width_setsize=3,
                 width_names=2,
                 hspace=0.2,
                 wspace=0.1,
                 grid_barplot=False,
                 invert_barplot=False):
    """
    Plots a main set of graph showing intersection size, intersection matrix and the size of base sets. If given,
    additional plots are placed below the main graph.

    :param data_dict: dictionary like {data_frame_name: data_frame}

    :param figsize: tuple, size of the figure.

    :param unique_keys: list. Specifies the names of the columns that, together, can uniquely identify a row. If left
    empty, pyUpSet will try to use all common columns in the data frames and may possibly raise an exception (no
    common columns) or produce unexpected results (columns in different data frames with same name but different
    meanings/data).

    :param sort_by: 'size' or 'degree'. The order in which to sort the intersection bar chart and matrix in the main
    graph

    :param inters_size_bounds: tuple. Specifies the size limits of the intersections that will be displayed.
    Intersections (and relative data) whose size is outside the interval will not be plotted. Defaults to (0, np.inf).

    :param inters_degree_bounds: tuple. Specified the degree limits of the intersections that will be displayed.
    Intersections (and relative data) whose degree is outside the interval will not be plotted. Defaults to (0, np.inf).

    :param additional_plots: list of dictionaries. See below for details.

    :param names_fontsize: float giving the fontsize of names.

    :param query: list of tuples. See below for details.

    :param colors_query: list of colors.

    :param color_vbar: 4-length array. If None, a gray is used.

    :param color_hbar: 4-length array or matplotlib color name. If None, a gray is used.

    :param color_matr: 4-length array. If None, a gray is used.

    :param vbar_rot: float, otation of vertical bars annotations.

    :param vbar_fmt: str, print format of vertical bars annotations.

    :param circle_size: int, size of the circles in the matrix plot.

    :param height_ratio: int, ratio of the intersection barplot plot height to the matrix plot height.

    :param width_setsize: int or float, width of the set size plot

    :param width_names: int or float, width of the names plot

    :param hspace: float, hspace in GridSpec

    :param wspace: float, wspace in GridSpec

    :param invert_barplot: bool, set to True to invert orientation of intersection barplot plot.

    :param grid_barplot: bool, set to True to draw grid on barplot plot

    :return: dictionary of matplotlib objects, namely the figure and the axes.

    :raise ValueError: if no unique_keys are specified and the data frames have no common column names.

    The syntax to specify additional plots follows the signature of the corresponding matplotlib method in an Axes
    class. For each additional plot one specifies a dictionary with the kind of plot, the columns name to retrieve
    relevant data and the kwargs to pass to the plot function, as in `{'kind':'scatter', 'data':{'x':'col_1',
    'y':'col_2'}, 'kwargs':{'s':50}}`.

    Currently supported additional plots: scatter.

    It is also possible to highlight intersections. This is done through the `query` argument, where the
    intersections to highligh must be specified with the names used as keys in the data_dict.

    """
    query = [] if query is None else query
    ap = [] if additional_plots is None else additional_plots
    all_columns = unique_keys if unique_keys is not None else __get_all_common_columns(data_dict)
    all_columns = list(all_columns)

    plot_data = DataExtractor(data_dict, all_columns)
    ordered_inters_sizes, ordered_in_sets, ordered_out_sets = \
        plot_data.get_filtered_intersections(sort_by,inters_size_bounds,inters_degree_bounds)
    ordered_dfs, ordered_df_names = plot_data.ordered_dfs, plot_data.ordered_df_names

    upset = UpSetPlot(
        figsize         = figsize,
        rows             = len(ordered_dfs),
        cols             = len(ordered_in_sets),
        additional_plots = additional_plots,
        names_fontsize   = names_fontsize,
        query            = query,
        colors_query     = colors_query,
        color_vbar       = color_vbar,
        color_hbar       = color_hbar,
        color_matr       = color_matr,
        vbar_rot         = vbar_rot,
        vbar_fmt         = vbar_fmt,
        circle_size      = circle_size,
        height_ratio     = height_ratio,
        width_setsize    = width_setsize,
        width_names      = width_names,
        invert_barplot   = invert_barplot,
        grid_barplot     = grid_barplot,
        hspace           = hspace,
        wspace           = wspace,
    )
    fig_dict = upset.main_plot(
        ordered_dfs,
        ordered_df_names,
        ordered_in_sets,
        ordered_out_sets,
        ordered_inters_sizes
    )
    fig_dict['additional'] = []

    # ap = [{kind:'', data:{x:'', y:''}, s:'', ..., kwargs:''}]
    for i, graph_settings in enumerate(ap):
        plot_kind = graph_settings.pop('kind')
        data_vars = graph_settings.pop('data_quantities')
        graph_properties = graph_settings.get('graph_properties', {})
        data_values = plot_data.extract_data_for(data_vars, query)
        ax = upset.additional_plot(i, plot_kind, data_values, graph_properties, labels=data_vars)
        fig_dict['additional'].append(ax)

    return fig_dict


def __get_all_common_columns(data_dict):
    """
    Computes an array of (unique) common columns to the data frames in data_dict
    :param data_dict: Dictionary of data frames
    :return: array.
    """
    common_columns = []
    for i, k in enumerate(data_dict.keys()):
        if i == 0:
            common_columns = data_dict[k].columns
        else:
            common_columns = common_columns.intersection(data_dict[k].columns)
    if len(common_columns.values) == 0:
        raise ValueError('Data frames should have homogeneous columns with the same name to use for computing '
                         'intersections')
    return common_columns.unique()

class UpSetPlot():
    def __init__(self, figsize, rows, cols, additional_plots, names_fontsize, query, colors_query, color_vbar, color_hbar,
                 color_matr, vbar_rot, vbar_fmt, circle_size, height_ratio, width_setsize, width_names, invert_barplot,
                 grid_barplot, hspace, wspace):
        """
        Generates figures and axes.

        :param figsize: Size of the figure

        :param rows: The number of rows of the intersection matrix

        :param cols: The number of columns of the intersection matrix

        :param additional_plots: list of dictionaries as specified in plot_pyupset()

        :param names_fontsize: float as specified in plot_pyupset()

        :param query: list of tuples as specified in plot_pyupset()

        :param colors_query: list of colors as specified in plot_pyupset()

        :param color_vbar: 4-length array or color name

        :param color_hbar: 4-length array

        :param color_matr: 4-length array

        :param vbar_rot: float

        :param vbar_fmt: str

        :param circle_size: float

        :param height_ratio: float

        :param width_setsize: int or float

        :param width_names: int

        :param invert_barplot: bool

        :param grid_barplot: bool

        :param hspace: float

        :param wspace: float
        """

        self.figsize        = figsize
        self.vbar_rot       = vbar_rot
        self.vbar_fmt       = vbar_fmt
        self.names_fontsize = names_fontsize
        self.circle_size    = circle_size
        self.height_ratio   = height_ratio
        self.width_names    = width_names
        self.width_setsize  = width_setsize
        self.invert_barplot = invert_barplot
        self.grid_barplot   = grid_barplot
        self.hspace         = hspace
        self.wspace         = wspace

        # set standard colors
        self.greys = plt.cm.Greys([.22, .8])

        if color_hbar is None:
            self.color_hbar = self.greys[1]
        else:
            self.color_hbar = color_hbar

        if color_vbar is None:
            self.color_vbar = self.greys[1]
        else:
            self.color_vbar = color_vbar

        if color_matr is None:
            self.color_matr = self.greys[1]
        else:
            self.color_matr = color_matr

        # map queries to graphic properties
        self.query = query

        if colors_query is None:
            colors_query = plt.cm.rainbow(np.linspace(.01, .99, len(self.query)))

        self.query2color = dict(zip([frozenset(q) for q in self.query], colors_query))
        self.query2zorder = dict(zip([frozenset(q) for q in self.query], np.arange(len(self.query)) + 1))

        # set figure properties
        self.rows = rows
        self.cols = cols
        self.x_values, self.y_values = self._create_coordinates(rows, cols)
        self.fig, self.ax_intbars, self.ax_intmatrix, \
        self.ax_setsize, self.ax_tablenames, self.additional_plots_axes = self._prepare_figure(additional_plots)

        self.standard_graph_settings = {
            'scatter': {
                'alpha': .3,
                'edgecolor': None
            },
            'hist': {
                'histtype': 'stepfilled',
                'alpha': .3,
                'lw': 0
            }
        }

        # single dictionary may be fragile - I leave it here as a future option
        # self.query2kwargs = dict(zip([frozenset(q) for q in self.query],
        # [dict(zip(['color', 'zorder'],
        # [col, 1])) for col in qu_col]))

    def _create_coordinates(self, rows, cols):
        """
        Creates the x, y coordinates shared by the main plots.

        :param rows: number of rows of intersection matrix
        :param cols: number of columns of intersection matrix
        :return: arrays with x and y coordinates
        """
        x_values = (np.arange(cols) + 1)
        y_values = (np.arange(rows) + 1)
        return x_values, y_values

    def _prepare_figure(self, additional_plots):
        """
        Prepares the figure, axes (and their grid) taking into account the additional plots.

        :param additional_plots: list of dictionaries as specified in plot_pyupset()
        :return: references to the newly created figure and axes
        """
        fig = plt.figure(figsize=self.figsize)
        if additional_plots:
            main_gs = gridspec.GridSpec(3, 1, hspace=.4)
            topgs = main_gs[:2, 0]
            botgs = main_gs[2, 0]
        else:
            topgs = gridspec.GridSpec(1, 1)[0, 0]

        if type(self.width_setsize) == float:
            self.width_setsize = np.int(np.rint(self.width_setsize*self.cols))
        if type(self.width_names)   == float:
            self.width_names = np.int(np.rint(self.width_names*self.cols))

        top_ncols = self.cols + self.width_names + self.width_setsize
        top_nrows = self.rows + self.rows * self.height_ratio

        if (top_ncols - self.cols) <= 1:
            raise ValueError("Reduce width ratio so that 2 plots can fit on the left (set size plot and names).")

        gs_top = gridspec.GridSpecFromSubplotSpec(
            nrows        = top_nrows,
            ncols        = top_ncols,
            subplot_spec = topgs,
            wspace       = self.wspace,
            hspace       = self.hspace
        )
        print("top_nrows: %d, top_ncols: %d" % (top_nrows, top_ncols))

        setsize_w   , setsize_h   = self.width_setsize , self.rows
        tablesize_w , tablesize_h = self.width_names   , self.rows
        intmatrix_w , intmatrix_h = self.cols          , self.rows
        intbars_w   , intbars_h   = self.cols          , top_nrows - self.rows

        if self.invert_barplot:
            ax_setsize    = plt.subplot(gs_top[:setsize_h, 0:setsize_w])
            ax_tablenames = plt.subplot(gs_top[:tablesize_h, setsize_w:(setsize_w+tablesize_w)])
            ax_intmatrix  = plt.subplot(gs_top[:intmatrix_h, (setsize_w+tablesize_w):-1])
            ax_intbars    = plt.subplot(gs_top[-intbars_h:-1, (setsize_w+tablesize_w):-1])
        else:
            ax_setsize    = plt.subplot(gs_top[-setsize_h:-1, 0:setsize_w])
            ax_tablenames = plt.subplot(gs_top[-tablesize_h:-1, setsize_w:(setsize_w+tablesize_w)])
            print("intmatrix_h: %d, intmatrix_w: %d" % (self.rows, self.cols))
            ax_intmatrix  = plt.subplot(gs_top[-intmatrix_h:-1, (setsize_w+tablesize_w):-1])
            ax_intbars    = plt.subplot(gs_top[:intbars_h, (setsize_w+tablesize_w):-1])

        add_ax = []
        if additional_plots:
            num_plots = len(additional_plots)
            num_bot_rows, num_bot_cols = int(np.ceil(num_plots / 2)), 2
            gs_bottom = gridspec.GridSpecFromSubplotSpec(
                nrows        = num_bot_rows,
                ncols        = num_bot_cols,
                subplot_spec = botgs,
                wspace       = self.wspace,
                hspace       = self.hspace
            )
            from itertools import product

            for r, c in product(range(num_bot_rows), range(num_bot_cols)):
                if r+c+1>num_plots: break
                new_plotL = plt.subplot(gs_bottom[r, c])
                add_ax.append(new_plotL)

        return fig, ax_intbars, ax_intmatrix, ax_setsize, ax_tablenames, tuple(add_ax)

    def _color_for_query(self, query, mode):
        """
        Helper function that returns the standard dark grey for non-queried intersections, and the color assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :param mode: str.
        :return: color as length 4 array.
        """
        # query_color = self.query2color.get(query, self.greys[1])
        if mode == "matr":
            query_color = self.query2color.get(query, self.color_matr)
        elif mode == "vbar":
            query_color = self.query2color.get(query, self.color_vbar)
        return query_color

    def _zorder_for_query(self, query):
        """
        Helper function that returns 0 for non-queried intersections, and the zorder assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: zorder as int.
        """
        query_zorder = self.query2zorder.get(query, 0)
        return query_zorder

    def main_plot(self, ordered_dfs, ordered_df_names, ordered_in_sets, ordered_out_sets, ordered_inters_sizes):
        """
        Creates the main graph comprising bar plot of base set sizes, bar plot of intersection sizes and intersection
        matrix.

        :param ordered_dfs: array of input data frames, sorted w.r.t. the sorting parameters provided by the user (if
        any)

        :param ordered_df_names: array of names of input data frames, sorted (as above)

        :param ordered_in_sets: list of tuples. Each tuple represents an intersection. The list must be sorted as the
        other parameters.

        :param ordered_out_sets: list of tuples. Each tuple represents the sets excluded from the corresponding
        intersection described by ordered_in_sets.

        :param ordered_inters_sizes: array of ints. Contains the intersection sizes, sorted as the other arguments.

        :return: dictionary containing figure and axes references.
        """
        ylim = self._base_sets_plot(ordered_dfs, ordered_df_names)
        self._table_names_plot(ordered_df_names, ylim)
        xlim = self._inters_sizes_plot(ordered_in_sets, ordered_inters_sizes)
        set_row_map = dict(zip(ordered_df_names, self.y_values))
        self._inters_matrix(ordered_in_sets, ordered_out_sets, xlim, ylim, set_row_map)
        return {'figure': self.fig,
                'intersection_bars': self.ax_intbars,
                'intersection_matrix': self.ax_intmatrix,
                'base_setsize': self.ax_setsize,
                'tablenames': self.ax_tablenames}

    def _table_names_plot(self, sorted_set_names, ylim):
        ax = self.ax_tablenames
        ax.set_ylim(ylim)
        xlim = ax.get_xlim()
        tr = ax.transData.transform
        for i, name in enumerate(sorted_set_names):
            ax.text(
                x         = (xlim[1]-xlim[0])/2,
                y         = self.y_values[i],
                s         = name,
                fontsize  = self.names_fontsize,
                clip_on   = True,
                va        = 'center',
                ha        = 'center',
                transform = ax.transData,
                family    = 'monospace'
            )

        if len(self.x_values) > 1:
            row_width = self.x_values[1] - self.x_values[0]
        else:
            row_width = self.x_values[0]

        background = plt.cm.Greys([.09])[0]

        for r, y in enumerate(self.y_values):
            if r % 2 == 0:
                ax.add_patch(
                    Rectangle((xlim[0], y - row_width / 2),
                              height=row_width,
                              width=xlim[1],
                              color=background, zorder=0)
                )
        ax.axis('off')


    def _base_sets_plot(self, sorted_sets, sorted_set_names):
        """
        Plots horizontal bar plot for base set sizes.

        :param sorted_sets: list of data frames, sorted according to user's directives.
        :param sorted_set_names: list of names for the data frames.
        :return: Axes.
        """
        ax = self.ax_setsize
        ax.invert_xaxis()
        height = .7
        bar_center = self.y_values

        ax.barh(
            y      = bar_center,
            width  = [len(x) for x in sorted_sets],
            height = height,
            color  = self.color_hbar,
            align  = "center"
        )

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
        self._strip_axes(ax, keep_spines=['bottom'], keep_ticklabels=['bottom'])
        ax.set_ylim((height / 2, ax.get_ylim()[1] + height / 2))
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] + 0.04 * (xlim[1]-xlim[0]))
        ax.spines['bottom'].set_bounds(xlim[0], xlim[1] + 0.04 * (xlim[1]-xlim[0]))

        ax.set_xlabel("Set size", fontweight='bold', fontsize=13)

        return ax.get_ylim()

    def _strip_axes(self, ax, keep_spines=None, keep_ticklabels=None):
        """
        Removes spines and tick labels from ax, except those specified by the user.

        :param ax: Axes on which to operate.
        :param keep_spines: Names of spines to keep.
        :param keep_ticklabels: Names of tick labels to keep.

        Possible names are 'left'|'right'|'top'|'bottom'.
        """
        tick_params_dict = {
            'which'       : 'both',
            'bottom'      : False,
            'top'         : False,
            'left'        : False,
            'right'       : False,
            'labelbottom' : False,
            'labeltop'    : False,
            'labelleft'   : False,
            'labelright'  : False
        }

        if keep_ticklabels is None:
            keep_ticklabels = []
        if keep_spines is None:
            keep_spines = []
        lab_keys = [(k, "".join(["label", k])) for k in keep_ticklabels]
        for k in lab_keys:
            tick_params_dict[k[0]] = True
            tick_params_dict[k[1]] = True
        ax.tick_params(**tick_params_dict)
        for sname, spine in ax.spines.items():
            if sname not in keep_spines:
                spine.set_visible(False)

    def _inters_sizes_plot(self, ordered_in_sets, inters_sizes):
        """
        Plots bar plot for intersection sizes.
        :param ordered_in_sets: array of tuples. Each tuple represents an intersection. The array is sorted according
        to the user's directives

        :param inters_sizes: array of ints. Sorted, likewise.

        :return: Axes
        """
        ax = self.ax_intbars
        width = .7

        bar_center = self.x_values
        bar_colors = [self._color_for_query(frozenset(inter), mode="vbar") for inter in ordered_in_sets]

        ax.bar(
            x         = bar_center,
            height    = inters_sizes,
            width     = width,
            color     = bar_colors,
            linewidth = 0,
            align     = "center"
        )

        ylim = ax.get_ylim()
        hgap = (ylim[1] - ylim[0]) / 60

        if self.invert_barplot:
            for x, y in zip(self.x_values, inters_sizes):
                ax.text(
                    x,
                    y + 3 * hgap,
                    ("{:%s}" % self.vbar_fmt).format(y),
                    rotation = self.vbar_rot,
                    ha       = 'center',
                    va       = 'bottom'
                )

            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

            self._strip_axes(ax, keep_spines=['right'], keep_ticklabels=['right'])
        else:
            for x, y in zip(self.x_values, inters_sizes):
                ax.text(
                    x,
                    y + hgap,
                    ("{:%s}" % self.vbar_fmt).format(y),
                    rotation = self.vbar_rot,
                    ha       = 'center',
                    va       = 'bottom'
                )

            self._strip_axes(ax, keep_spines=['left'], keep_ticklabels=['left'])

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))
        ylim = ax.get_ylim()

        if self.invert_barplot:
            ax.spines['right'].set_bounds(ylim[1], ylim[0] + 0.04 * (ylim[0]-ylim[1]))
        else:
            ax.spines['left'].set_bounds(ylim[0], ylim[1] + 0.04 * (ylim[1]-ylim[0]))

        if self.grid_barplot:
            ax.yaxis.grid(True, lw=.25, color='grey', ls=':')
            ax.set_axisbelow(True)

        ax.set_ylabel("Intersection size", labelpad=6, fontweight='bold', fontsize=13)

        return ax.get_xlim()

    def _inters_matrix(self, ordered_in_sets, ordered_out_sets, xlims, ylims, set_row_map):
        """
        Plots intersection matrix.

        :param ordered_in_sets: Array of tuples representing sets included in an intersection. Sorted according to
        the user's directives.

        :param ordered_out_sets: Array of tuples representing sets excluded from an intersection. Sorted likewise.

        :param xlims: tuple. x limits for the intersection matrix plot.

        :param ylims: tuple. y limits for the intersection matrix plot.

        :param set_row_map: dict. Maps data frames (base sets) names to a row of the intersection matrix

        :return: Axes
        """
        ax = self.ax_intmatrix
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        if len(self.x_values) > 1:
            row_width = self.x_values[1] - self.x_values[0]
        else:
            row_width = self.x_values[0]

        self._strip_axes(ax)

        background = plt.cm.Greys([.09])[0]

        for r, y in enumerate(self.y_values):
            if r % 2 == 0:
                if self.invert_barplot:
                    ax.add_patch(
                        Rectangle(
                            (xlims[1], y - row_width / 2),
                            height = row_width,
                            width  = xlims[0],
                            color  = background,
                            zorder = 0
                        )
                    )

                else:
                    ax.add_patch(
                        Rectangle(
                            (xlims[0], y - row_width / 2),
                            height = row_width,
                            width  = xlims[1],
                            color  = background,
                            zorder = 0
                        )
                    )

        for col_num, (in_sets, out_sets) in enumerate(zip(ordered_in_sets, ordered_out_sets)):
            in_y = [set_row_map[s] for s in in_sets]
            out_y = [set_row_map[s] for s in out_sets]
            # in_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[1]) for y in in_y]
            # out_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[0]) for y in out_y]
            # for c in chain.from_iterable([in_circles, out_circles]):
            # ax.add_patch(c)
            ax.scatter(
                np.repeat(self.x_values[col_num], len(in_y)),
                in_y,
                color = np.tile(self._color_for_query(frozenset(in_sets), mode = "matr"), (len(in_y), 1)),
                s     = self.circle_size,
            )
            ax.scatter(
                np.repeat(self.x_values[col_num], len(out_y)),
                out_y,
                color = self.greys[0],
                s     = self.circle_size
            )
            ax.vlines(
                self.x_values[col_num],
                min(in_y), max(in_y),
                lw = 3.5,
                color = self._color_for_query(frozenset(in_sets), mode="matr")
            )

    def additional_plot(self, ax_index, kind, data_values, graph_args, *, labels=None):
        """
        Scatter plot (for additional plots).

        :param ax_index: int. Index for the relevant axes (additional plots' axes are stored in a list)

        :param data_values: list of dictionary. Each dictionary is like {'x':data_for_x, 'y':data_for_y,
        'in_sets':tuple}, where the tuple represents the intersection the data for x and y belongs to.

        :param plot_kwargs: kwargs accepted by matplotlib scatter

        :param labels: dictionary. {'x':'x_label', 'y':'y_label'}

        :return: Axes
        """
        ax = self.additional_plots_axes[ax_index]

        plot_method = getattr(ax, kind)

        for k, v in self.standard_graph_settings.get(kind, {}).items():
            graph_args.setdefault(k, v)

        plot_method = partial(plot_method, **graph_args)

        # data_values = [{query:{relevant data}}]
        ylim, xlim = [np.inf, -np.inf], [np.inf, -np.inf]
        for query, data_item in data_values.items():
            clr = self._color_for_query(frozenset(query))
            plot_method(color=self._color_for_query(frozenset(query)),
                        zorder=self._zorder_for_query(frozenset(query)),
                        **data_item
                        )
            new_xlim, new_ylim = ax.get_xlim(), ax.get_ylim()
            for old, new in zip([xlim, ylim], [new_xlim, new_ylim]):
                old[0] = new[0] if old[0] > new[0] else old[0]
                old[1] = new[1] if old[1] < new[1] else old[1]

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

        self._strip_axes(ax, keep_spines=['bottom', 'left'], keep_ticklabels=['bottom', 'left'])
        # ylim, xlim = ax.get_ylim(), ax.get_xlim()
        gap_y, gap_x = max(ylim) / 500.0 * 20, max(xlim) / 500.0 * 20
        ax.set_ylim(ylim[0] - gap_y, ylim[1] + gap_y)
        ax.set_xlim(xlim[0] - gap_x, xlim[1] + gap_x)
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        ax.spines['left'].set_bounds(ylim[0], ylim[1])
        ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

        for l, text in labels.items():
            getattr(ax, 'set_%slabel' % l)(text, labelpad=3,
                                           fontweight='bold', fontsize=13) if l in ['x', 'y'] else None
        return ax


class DataExtractor:
    def __init__(self, data_dict, unique_keys):
        """
        Packages the data in a way that can be consumed by the plot methods in UpSetPlot.

        :param data_dict: dict. {'name': pandas DataFrame}

        :param unique_keys: list of names of columns that uniquely identify a row in the data frames.
        """
        self.unique_keys = unique_keys if len(unique_keys) > 1 else unique_keys[0]
        self.ordered_dfs, self.ordered_df_names, self.df_dict = self.extract_base_sets_data(data_dict,
                                                                                            unique_keys)
        self.in_sets_list, self.inters_degrees, \
        self.out_sets_list, self.inters_df_dict = self.extract_intersection_data()


    def extract_base_sets_data(self, data_dict, unique_keys):
        """
        Extracts data for the bar graph of the base sets sizes.

        :param data_dict: dict. {'name': data frame}

        :param unique_keys: list of column names to uniquely identify rows.

        :return: list of data frames sorted by shape[0], list of names sorted accordingly, dictionary zipping the two.
        """
        dfs = []
        df_names = []
        # extract interesting columns from dfs
        for name, df in data_dict.items():
            df_names.append(name)
            dfs.append(df[unique_keys])
        df_names = np.array(df_names)
        # order dfs
        base_sets_order = np.argsort([x.shape[0] for x in dfs])[::-1]
        ordered_base_set_names = df_names[base_sets_order]
        ordered_base_sets = [data_dict[name] for name in ordered_base_set_names]
        set_dict = dict(zip(ordered_base_set_names, ordered_base_sets))

        return ordered_base_sets, ordered_base_set_names, set_dict

    def extract_intersection_data(self):
        """
        Extract data to use in intersection bar plot and matrix.

        :return: list of tuples (sets included in intersections), list of integers (corresponding degrees of
        intersections), list of tuples (sets excluded from intersections), dict {tuple:data frame}, where each data
        frame contains only the rows corresponding to the intersection described by the tuple-key.
        """
        in_sets_list = []
        out_sets_list = []
        inters_dict = {}
        inters_degrees = []
        for col_num, in_sets in enumerate(chain.from_iterable(
                combinations(self.ordered_df_names, i) for i in np.arange(1, len(self.ordered_dfs) + 1))):

            in_sets = frozenset(in_sets)

            inters_degrees.append(len(in_sets))
            in_sets_list.append(in_sets)
            out_sets = set(self.ordered_df_names).difference(set(in_sets))
            in_sets_l = list(in_sets)
            out_sets_list.append(set(out_sets))

            seed = in_sets_l.pop()
            exclusive_intersection = pd.Index(self.df_dict[seed][self.unique_keys])
            for s in in_sets_l:
                exclusive_intersection = exclusive_intersection.intersection(pd.Index(self.df_dict[s][
                    self.unique_keys]))
            for s in out_sets:
                exclusive_intersection = exclusive_intersection.difference(pd.Index(self.df_dict[s][
                    self.unique_keys]))
            final_df = self.df_dict[seed].set_index(pd.Index(self.df_dict[seed][self.unique_keys])).loc[
                exclusive_intersection].reset_index(drop=True)
            inters_dict[in_sets] = final_df

        return in_sets_list, inters_degrees, out_sets_list, inters_dict

    def get_filtered_intersections(self, sort_by, inters_size_bounds, inters_degree_bounds):
        """
        Filter the intersection data according to the user's directives and return it.

        :param sort_by: 'degree'|'size'. Whether to sort intersections by degree or size.
        :param inters_size_bounds: tuple. Specifies the size interval of the intersections that will be plotted.
        :param inters_degree_bounds: tuple. Specifies the degree interval of the intersections that will be plotted.
        :return: Array of int (sizes), array of tuples (sets included in intersection), array of tuples (sets
        excluded from intersection), all filtered and sorted.
        """
        inters_sizes = np.array([self.inters_df_dict[x].shape[0] for x in self.in_sets_list])
        inters_degrees = np.array(self.inters_degrees)

        size_clip = (inters_sizes <= inters_size_bounds[1]) & (inters_sizes >= inters_size_bounds[0]) & (
            inters_degrees >= inters_degree_bounds[0]) & (inters_degrees <= inters_degree_bounds[1])

        in_sets_list = np.array(self.in_sets_list)[size_clip]
        out_sets_list = np.array(self.out_sets_list)[size_clip]
        inters_sizes = inters_sizes[size_clip]
        inters_degrees = inters_degrees[size_clip]

        # sort as requested
        if sort_by == 'size':
            order = np.argsort(inters_sizes)[::-1]
        elif sort_by == 'degree':
            order = np.argsort(inters_degrees)

        # store ordered data
        self.filtered_inters_sizes = inters_sizes[order]
        self.filtered_in_sets = in_sets_list[order]
        self.filtered_out_sets = out_sets_list[order]

        return self.filtered_inters_sizes, self.filtered_in_sets, self.filtered_out_sets

    def extract_data_for(self, var_dict, queries):
        """
        Extract data from named columns (values) and place in named variables (keys).

        :return: list of dict. [{query:{x:, y:, ...}}]
        """
        data_values = {}
        poss_queries = [q for q in queries if frozenset(q) in self.filtered_in_sets]
        for q in poss_queries:
            data_values[q] = dict(zip(var_dict.keys(),
                [self.inters_df_dict[frozenset(q)][v].values for k, v in var_dict.items()]))
        data_values['others'] = dict(zip(var_dict.keys(),
            [chain(*[self.inters_df_dict[frozenset(q)][v].values for q in self.filtered_in_sets if q not in poss_queries])
             for k, v in var_dict.items()]))
        for k, vals in data_values['others'].items():
            data_values['others'][k] = [x for x in vals]

        return data_values
