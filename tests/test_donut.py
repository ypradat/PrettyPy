# -*- coding: utf-8 -*-
"""
@modified: Feb 25 2021
@created: Feb 25 2021
@author: Yoann Pradat

Tests for prettypy.donut module.
"""

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from prettypy.donut import plot_donut, DonutConfig

def make_mock_df_1():
    obs = [
       ('A', 1, "frizzled"),
       ('A', 1, "lethargic"),
       ('A', 1, "polythene"),
       ('A', 1, "epic"),
       ('A', 2, "frizzled"),
       ('A', 2, "lethargic"),
       ('A', 2, "epic"),
       ('A', 3, "frizzled"),
       ('A', 3, "lethargic"),
       ('A', 3, "polythene"),
       ('A', 3, "epic"),
       ('A', 3, "bedraggled"),
       ('B', 1, "frizzled"),
       ('B', 1, "lethargic"),
       ('B', 1, "polythene"),
       ('B', 1, "epic"),
       ('B', 1, "bedraggled"),
       ('B', 1, "moombahcored"),
       ('B', 2, "frizzled"),
       ('B', 2, "lethargic"),
       ('B', 2, "polythene"),
       ('B', 2, "epic"),
       ('B', 2, "bedraggled"),
       ('C', 1, "frizzled"),
       ('C', 1, "lethargic"),
       ('C', 1, "polythene"),
       ('C', 1, "epic"),
       ('C', 1, "bedraggled"),
       ('C', 1, "moombahcored"),
       ('C', 1, "zoned"),
       ('C', 1, "erstaz"),
       ('C', 1, "mined"),
       ('C', 1, "liberated"),
       ('C', 2, "frizzled"),
       ('C', 2, "lethargic"),
       ('C', 2, "polythene"),
       ('C', 2, "epic"),
       ('C', 2, "bedraggled"),
       ('C', 3, "frizzled"),
       ('C', 3, "lethargic"),
       ('C', 3, "polythene"),
       ('C', 3, "epic"),
       ('C', 3, "bedraggled"),
       ('C', 4, "bedraggled"),
       ('C', 4, "frizzled"),
       ('C', 4, "lethargic"),
       ('C', 4, "polythene"),
       ('C', 4, "epic"),
       ('C', 5, "frizzled"),
       ('C', 5, "lethargic"),
       ('C', 5, "polythene"),
       ('C', 5, "epic"),
       ('C', 5, "bedraggled"),
       ('C', 5, "moombahcored")]
    labels = ['group', 'subgroup', 'sub-subgroup']
    return pd.DataFrame.from_records(obs, columns=labels)

def test_plot_donut_1():
    df = make_mock_df_1()
    config = DonutConfig(figsize=(8,8), sizes_fmt="%s (%d)", group_labeldistance=1.1,
                         colors=["#59CD90","#3FA7D6","#FFC759"])

    fig, ax = plot_donut(df=df,
                         col_groups="group",
                         col_subgroups="subgroup",
                         config=config)

    # title
    ax.set_title("Test", fontsize=24, fontweight="medium", pad=25)
    plt.savefig("./img/test_donut_1.png", dpi=300)
    plt.close()

def make_mock_df_2():
    obs = [
        ("Positive", "Negative"),
        ("Positive", "Negative"),
        ("Positive", "Negative"),
        ("Positive", "Negative"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Positive", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Positive"),
        ("Negative", "Negative"),
        ("Negative", "Negative")]
    labels = ['group', 'subgroup']
    return pd.DataFrame.from_records(obs, columns=labels)

def test_plot_donut_2():
    df = make_mock_df_2()
    config = DonutConfig(plot_fontsize=12, colors=["#59CD90","#3FA7D6"],
                         light_color_for_last_in_subgroup=False)

    fig, ax = plt.subplots(1,1,figsize=(10,8))

    plot_donut(df=df,
               col_groups="group",
               col_subgroups="subgroup",
               config=config,
               ax=ax)

    # title
    ax.set_title("Example", fontsize=24, fontweight="medium", pad=25)
    plt.tight_layout()
    plt.savefig("./img/test_donut_2.png", dpi=300)
    plt.close()
