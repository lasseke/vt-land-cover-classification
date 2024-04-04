'''
Create plots related to variable correlation.
'''

import json
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster import hierarchy
from src import helpers as hlp

# Constants
SAVE_PATH = "../results/plots/var_correlation/"
PLT_STYLE_PATH = hlp.get_plt_style_config_path()

if not Path(SAVE_PATH).is_dir():
    Path.mkdir(Path(SAVE_PATH), parents=True)

with open('../data/dict/predictors.json', encoding='utf-8') as json_file:
    predictor_dict = json.load(json_file)


def plot_dendrogram(
    feat_matrix: pd.DataFrame, feat_corr_linkage: np.ndarray,
    figsize: Tuple[int, int], save_as: Optional[str] = None
) -> plt.Figure:
    '''
    Plot dendrogram.

    feature_matrix: Pandas Dataframe.
    '''

    # Define style to use
    plt.style.use(PLT_STYLE_PATH)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    hierarchy.dendrogram(
        feat_corr_linkage,
        labels=[
            predictor_dict[x]['long_name'] for x
            in feat_matrix.columns.tolist()
            ],
        ax=ax,
        leaf_rotation=90
    )

    ax.set_ylabel("Sum of squares index\n(Ward's method)")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        size=8
    )

    fig.tight_layout()

    if save_as is not None:
        fig.savefig(SAVE_PATH + save_as)

    return fig


def plot_spearman_heatmap(
    feature_matrix_df: pd.DataFrame, figsize: tuple,
    save_as: Optional[str] = None
) -> plt.Figure:
    '''
    Plot heatmap to visualize spearman rank correlation.
    '''

    # Define style to use
    plt.style.use(PLT_STYLE_PATH)

    # Create a mask to hide redundant values (upper half of matrix)
    heatmap_mask = np.triu(np.ones_like(feature_matrix_df, dtype=bool), k=0)

    # Create a custom diverging color palette
    cmap_heatmap = sns.diverging_palette(
        250, 15,
        s=75,
        l=40,
        n=12,
        center="light",
        as_cmap=True
    )

    long_labels = [
        predictor_dict[x]['long_name']
        for x in feature_matrix_df.columns.tolist()
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(
        feature_matrix_df,
        mask=heatmap_mask,
        center=0,
        annot=False,
        square=True,
        cmap=cmap_heatmap,
        cbar_kws={
            'label': r"Spearman's $\rho$",
            "shrink": 0.3,
            "location": "top",
            "pad": 0.01,
            "anchor": (0, 0),
            },
        vmin=-1,
        vmax=1,
        yticklabels=long_labels,
        xticklabels=long_labels
    )

    # Set additional tick parameters
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.collections[0].colorbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    if save_as is not None:
        fig.savefig(SAVE_PATH + save_as)

    return fig


def plot_cramer_heatmap(
    cramer_v_mat: pd.DataFrame, figsize: tuple,
    save_as: Optional[str] = None
) -> plt.Figure:
    '''
    Plot heatmap to visualize Cramer's V categorical variable association
    '''

    # Define style to use
    plt.style.use(PLT_STYLE_PATH)

    # Create a mask to hide redundant values (upper half of matrix)
    heatmap_mask = np.triu(np.ones_like(cramer_v_mat, dtype=bool), k=0)

    # Create a custom diverging color palette
    cmap_heatmap = sns.diverging_palette(
        250, 15,
        s=75,
        l=40,
        n=12,
        center="light",
        as_cmap=True
    )

    long_labels = [
        predictor_dict[x]['long_name']
        for x in cramer_v_mat.columns.tolist()
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(
        cramer_v_mat,
        mask=heatmap_mask,
        center=0,
        annot=True,
        square=True,
        cmap=cmap_heatmap,
        cbar_kws={
            'label': r"Corrected Cramér's $V$",
            "shrink": 0.4,
            },
        vmin=0,
        vmax=1,
        yticklabels=long_labels,
        xticklabels=long_labels
    )

    # Set additional tick parameters
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    fig.tight_layout()

    if save_as is not None:
        fig.savefig(SAVE_PATH + save_as)

    return fig
