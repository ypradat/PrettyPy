# -*- coding: utf-8 -*-
"""
@modified: Feb 01 2020
@created: Feb 01 2020
@author: Yoann Pradat

Tests for prettypy.double_heatmap module.
"""

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import stats

from prettypy.heatmap import build_double_heatmap, plot_double_heatmap, DoubleHeatmapConfig

def make_correlated_bernoulli(n=100, p=10, cov_mean=1, cov_sd=0.5, probs=0.05):
    if isinstance(probs, float):
        probs = [probs for _ in range(p)]

    # generate covariance matrix
    A = np.array([cov_sd*np.random.randn(p) + cov_mean for i in range(p)])
    A = A.dot(np.transpose(A))
    D_half = np.diag(np.diag(A)**(-0.5))
    cov = (D_half).dot(A).dot(D_half)

    # get bernoulli by thresholding multivariate-normal variables
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    for j in range(p):
        X[:,j] = np.where(X[:,j] >= stats.norm.ppf(1-probs[j]), 1, 0)

    return X

def make_mock_df_values(n_obs=1000, n_var=52, n_groups=7, cov_mean=1, cov_sd=0.5, seed=123):
    np.random.seed(seed)
    X = np.zeros((n_obs, n_var))

    # variable names
    n_letters = np.int(np.ceil(np.log(n_var)/np.log(26)))
    var_names = ["".join([chr(ord('A') + (i//26**k) % 26) for k in range(n_letters)][::-1]) for i in range(n_var)]

    # group lims
    group_lims = [0] + list(sorted(np.random.randint(1, n_var, size=n_groups-1))) + [n_var]

    for i_group in range(n_groups):
        group_indices = np.arange(group_lims[i_group], group_lims[i_group+1])
        X_group = make_correlated_bernoulli(n=n_obs,p=len(group_indices), cov_mean=cov_mean, cov_sd=cov_sd,
                                            probs=0.10-0.06*i_group/(n_groups-1))
        X[:, group_indices] = X_group

    df = pd.DataFrame(X, columns=var_names)
    return df


def test_plot_double_heatmap():
    df = make_mock_df_values(n_obs=1000, n_var=52, n_groups=7, cov_mean=0.4)
    dfs = build_double_heatmap(df_values=df)

    brown_to_green_colors = ["#8c5322", "#c2842a", "#debf7c", "#f5ebc4", "#c8e7e5", "#7fcdc2", "#339a92",  "#11675f"]
    brown_to_green_cmap = cm.colors.LinearSegmentedColormap.from_list("BrownToGreen", colors=brown_to_green_colors)

    config = DoubleHeatmapConfig()
    config.heatmap["orientation"] = "antidiagonal"
    config.ratio["boundaries"] = [0, 0.001, 0.01, 0.1, 1, 5, 20, 100, 1000]
    config.ratio["cmap"] = brown_to_green_cmap

    fig, axes = plot_double_heatmap(df_count=dfs["count"],
                                    df_ratio=dfs["ratio"],
                                    df_test=dfs["test"],
                                    config=config)
    plt.savefig("./img/test_double_heatmap.png", dpi=300)
    plt.close()
