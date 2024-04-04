"""Visualize Vegetation Type dataset"""

import pandas as pd
import matplotlib.pyplot as plt
from src import helpers as hlp
from pathlib import Path
from typing import Optional, Tuple

# Constants
SAVE_PATH = "../results/plots/vt_data/"
PLT_STYLE_PATH = hlp.get_plt_style_config_path()

if not Path(SAVE_PATH).is_dir():
    Path.mkdir(Path(SAVE_PATH), parents=True)


def barplot_class_frequency(
    vt_freq_series: pd.Series, figsize: Tuple[int, int],
    color: str = '#50C878', save_as: Optional[str] = None
) -> plt.Figure:
    """Create horizontal bar plot for VT class frequencies"""

    import matplotlib.pyplot as plt
    import json

    # Define style to use
    plt.style.use(PLT_STYLE_PATH)

    # Load dict with long vt names
    with open('../data/dict/vt_classes.json') as json_file:
        vt_dict = json.load(json_file)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    bar_plot = ax.barh(
        y=vt_freq_series.index,
        width=vt_freq_series.values,
        align='center',
        color=color,
        tick_label=[
            vt_dict[str(x)]['long_name'] + f" ({x})"
            for x in vt_freq_series.index
        ],
        zorder=2
    )

    def set_bar_labels(bar_rectangles, labels):
        """Function to place bar labels"""

        for rect, label in zip(bar_rectangles, labels):

            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

            ax.text(
                x=x_range*0.2,
                y=rect.get_y()+0.8,
                s=label,
                style='italic',
                ha='right',
                va='bottom'
            )

    # Calculate percentages, cast to string
    total_n = sum(vt_freq_series.values)
    percent_labels = [
        f"{str(round(cur_val/total_n*100, 1))}%"
        for cur_val in vt_freq_series.values
    ]

    set_bar_labels(bar_plot.patches, percent_labels)

    # Plot layout
    ax.invert_yaxis()
    ax.grid(axis='x', color='gray', linestyle='dashed', zorder=0)
    ax.set_xlabel("Absolute frequency")

    fig.tight_layout()

    if save_as is not None:
        fig.savefig(SAVE_PATH + save_as)

    return fig


def plot_on_norway(X, Y, colors, alphas, title=None, bbox=None,
                   file_path="./Data/Spatial/",
                   file_name="Norway_border_EPSG32633.shp"):
    '''
    Plot spatial data on a Norway background map.
    '''

    import geopandas as gpd
    import pandas as pd

    if len(X) != len(Y):
        raise ValueError("'X' and 'Y' must have same length!")

    if bbox is not None:
        if len(bbox) != 4 or not isinstance(bbox, list):
            raise ValueError(
                "'bbox' must be a list and contain: [xmin, ymin, xmax, ymax]."
            )
        else:
            xmin, ymin, xmax, ymax = bbox

    # Create background shape (Norway shapefile or bounding box)
    if bbox is None:
        df_norshp = gpd.read_file(file_path+file_name, bbox=None)
        ax = df_norshp.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

    else:
        # Only plot bbox
        from shapely.geometry import Polygon

        df_norshp = gpd.GeoDataFrame()

        poly = Polygon(
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ]
        )
        df_norshp.loc[0, 'geometry'] = poly
        df_norshp.set_crs(epsg=32633)

        # Plot box
        ax = df_norshp.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

    # Create geodataframes from input data frame
    gdfs = []

    for i in range(len(X)):

        # Get current point coordinates
        cur_df = pd.DataFrame([X[i], Y[i]]).T

        gdfs.append(
            gpd.GeoDataFrame(
                cur_df,
                geometry=gpd.points_from_xy(cur_df["x"], cur_df["y"])
            )
        )

        if bbox is not None:
            gdfs[i] = gdfs[i].cx[xmin:xmax, ymin:ymax]

        gdfs[i].plot(ax=ax, color=colors[i])

    ax.set_title(title)

    return ax
