"""
The :mod:`prettypy.heatmap` module defines functions for drawing beautiful heatmaps.
"""

from .double_heatmap import build_double_heatmap, plot_double_heatmap, DoubleHeatmapConfig

__all__ = [
    'build_double_heatmap',
    'plot_double_heatmap',
    'DoubleHeatmapConfig'
]
