# -*- coding: utf-8 -*-
"""
@modified: Dec 02 2020
@created: Dec 02 2020
@author: Yoann Pradat

Tests for _pyupset.py module.
"""

import matplotlib.pyplot as plt
import pandas as pd
import random
from coolpyplots import DrawPyUpsetPlot

def test_plot_pyupset():
    df = pd.DataFrame({"Group": random.choices(["A","B","C","D"], k=100), "Index": random.choices(range(20), k=100)})
    pyupset_drawer = DrawPyUpsetPlot(field_set="Index", df=df)
    dt_fig = pyupset_drawer.draw(hue_vars=["Group"],
                                 fields2vals_keep=None,
                                 fields2vals_drop=None,
                                 dt_names={"Group": {'key': False}},
                                 width_setsize=3,
                                 width_names=3,
                                 names_fontsize=8,
                                 figsize=(8,6))

    plt.savefig("./coolpyplots/tests/test_pyupset_all.pdf")
    plt.close()
