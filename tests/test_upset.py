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
from prettypy.upset import prepare_data_dict_upset_plot, plot_upset

def make_mock_df(nrows=100):
    df = pd.DataFrame({"Group": random.choices(["A","B","C","D"], k=nrows),
                       "Index": random.choices(range(20), k=nrows)})

    return df

def test_plot_pyupset():
    df = make_mock_df()
    data_dict = prepare_data_dict_upset_plot(df, field_for_upset="Index", fields_for_sets=["Group"])
    fig_dict = plot_upset(data_dict=data_dict, unique_keys=["Index"],
                          colors_query=[[199/255, 30/255, 30/255, 1]],
                          color_vbar=[245/255, 170/255, 50/255, 1],
                          color_hbar=[245/255, 170/255, 50/255, 1],
                          query=[tuple(data_dict.keys())],
                          width_setsize=3, width_names=3, names_fontsize=8, circle_size=50, figsize=(8,6))

    plt.savefig("./img/test_pyupset_all.png", dpi=300)
    plt.close()
