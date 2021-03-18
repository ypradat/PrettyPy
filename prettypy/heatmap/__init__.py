"""
The :mod:`prettypy.heatmap` module defines functions for drawing beautiful heatmaps.
"""

from .heatmap import plot_heatmap, HeatmapConfig
from .double_heatmap import build_double_heatmap, plot_double_heatmap, DoubleHeatmapConfig

__all__ = [
    'plot_heatmap',
    'HeatmapConfig',
    'build_double_heatmap',
    'plot_double_heatmap',
    'DoubleHeatmapConfig'
]
