"""
The :mod:`prettypy.upset` module defines functions for drawing beautiful upset plots in Python.
"""

from .upset_plot import upset_plot, prepare_data_dict_upset_plot

__all__ = [
    'upset_plot',
    'prepare_data_dict_upset_plot',
]
