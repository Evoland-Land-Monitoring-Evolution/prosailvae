import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib


def get_tile_dates(dir):
    tile_dates = pd.DataFrame(columns=["tile", "date"])
    image_files = os.listdir(dir)
    for image_file in image_files:
        filename_components = image_file.split("_")
        tile = filename_components[2]
        date = datetime.strptime(filename_components[1], "%Y%m%d")
        tile_dates = pd.concat(
            (tile_dates, pd.DataFrame({"tile": tile, "date": date}, index=[0])),
            ignore_index=True,
        )
    return tile_dates


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def tile_and_dates_plot(tile_dates):
    tiles = pd.unique(tile_dates["tile"])

    fig, ax = plt.subplots()
    scatterplots = []
    min_x = np.inf
    max_x = -np.inf

    for i, tile in enumerate(tiles):
        dates = tile_dates[tile_dates["tile"] == tile]["date"].values
        ax.axhline(i, color="k", zorder=0, lw=1)
        s = ax.scatter(dates, [i for _ in range(len(dates))])
        s = s.get_offsets().data
        max_x = max(s[:, 0].max(), max_x)
        min_x = min(s[:, 0].min(), min_x)
        scatterplots.append(s)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    xtickslabels = [
        ax.get_xticklabels()[j].get_text() for j in range(len(ax.get_xticklabels()))
    ]
    xtickspos = ax.get_xticks()
    plt.close("all")
    fig, ax = plt.subplots()
    for i, tile in enumerate(tiles):
        ax.axhline(i, color="k", zorder=0, lw=1)
        ax.scatter(
            (scatterplots[i][:, 0] - min_x) / (max_x - min_x), scatterplots[i][:, 1]
        )

    ax.set_xticks((xtickspos - min_x) / (max_x - min_x), xtickslabels)
    xtickslabels = [
        ax.get_xticklabels()[j].get_text() for j in range(len(ax.get_xticklabels()))
    ]
    ax.set_yticklabels(tiles)
    ax.set_yticks([i for i in range(len(tiles))])
    ax.set_ylabel("MGRS Tile")
    ax.set_xlabel("Acquisition Date")
    return fig, ax


def main():
    dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/all_rois"
    tile_dates = get_tile_dates(dir)
    fig, ax = tile_and_dates_plot(tile_dates)
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(
        "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/tile_dates.tex"
    )
    fig.savefig(
        "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/tile_acquisition_dates.svg"
    )
    pass


if __name__ == "__main__":
    main()
