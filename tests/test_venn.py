# -*- coding: utf-8 -*-
"""
@created: 12/15/20
@modified: 12/15/20
@author: Yoann Pradat

Tests for venn/venn.py module.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import random
from prettypy.venn import plot_venn, VennConfig

def make_mock_df(nrows=100, range_group=["A", "B", "C"], range_idx=30):
    random.seed(123)
    df = pd.DataFrame({"Group": random.choices(range_group, k=nrows),
                       "Index": random.choices(range(range_idx), k=nrows)})
    df = df.drop_duplicates()
    return df

def test_venn_plots():
    df = make_mock_df(100)

    config = VennConfig(figsize=(8,8), alpha=1, arrow_r=0.8, arrow_color="black",
                        colors=["#FFA3AF","#007CBE", "#00AF54"])
    fig, ax = plot_venn(df=df, col_set="Group", col_identifier="Index", config=config)
    plt.tight_layout()
    plt.savefig("./img/test_venn.png", dpi=300)
