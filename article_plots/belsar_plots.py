import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import os 
from validation.belsar_validation import load_belsar_validation_data, get_sites_geometry
import pandas as pd
import seaborn as sns
import tikzplotlib

@dataclass
class MeasurementDates:
    # wheat_names = ["W1", "W2", "W3", "W4", "W5"]
    # wheat_names = ["W", "W", "W", "W", "W"]
    wheat_names = [r"$W$", r"$W$", r"$W$", r"$W$", r"$W$"]
    wheat_dates = ["2018-05-17", "2018-05-18", "2018-06-05", "2018-06-21", "2018-07-19"]
    # maize_names = ["M1", "M2", "M3", "M4", "M5", "M6"]
    # maize_names = ["M", "M", "M", "M", "M", "M"]
    maize_names = [r"$M$", r"$M$", r"$M$", r"$M$", r"$M$", r"$M$"]
    maize_dates = ["2018-05-31", "2018-06-01", "2018-06-22", "2018-06-21", "2018-08-02", "2018-08-29"]

def plot_sampling_dates(s2_dates=None):

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained", dpi=150)
    ax.set(title="Image dates in data-set")
    if s2_dates is not None:
        s2_dates = [datetime.strptime(d, "%Y-%m-%d") for d in s2_dates]
        s2_levels = np.tile([ 2, 2, 2, 2, 2, 2],
                int(np.ceil(len(s2_dates)/6)))[:len(s2_dates)]
        ax.vlines(s2_dates, 0, s2_levels, color="tab:green")  # The vertical stems.
        ax.plot(s2_dates, np.zeros_like(s2_dates), "-o",
                color="k", markerfacecolor="w")  # Baseline and markers on it.

    # format x-axis with 1-week intervals
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y "))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y-axis and spines
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    return fig, ax


def plot_measurements_and_s2_dates(s2_dates=None, s2_names=None):
    meas_dates = MeasurementDates()

    wheat_dates = [datetime.strptime(d, "%Y-%m-%d") for d in meas_dates.wheat_dates]
    # Convert date strings (e.g. 2014-10-18) to datetime
    wheat_levels = np.tile([1, 1, 1, 1, 1],
                int(np.ceil(len(wheat_dates)/5)))[:len(wheat_dates)]
    maize_dates = [datetime.strptime(d, "%Y-%m-%d") for d in meas_dates.maize_dates]
    # Convert date strings (e.g. 2014-10-18) to datetime
    maize_levels = np.tile([-1, -1, -1, -1, -1, -1],
                int(np.ceil(len(maize_dates)/6)))[:len(maize_dates)]
    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 2), layout="constrained", dpi=150)
    # ax.set(title="Measurement dates in BelSAR campaign")

    # ax.vlines(wheat_dates, 1, wheat_levels, color="tab:red")  # The vertical stems.
    ax.scatter(wheat_dates, 1 * np.ones_like(wheat_dates), marker="o",
                color="k", facecolor="w", zorder=40, label="Wheat")  # Baseline and markers on it.
    ax.axhline(1, color="k",zorder=0, lw=1)
    ax.axhline(-1, color="k",zorder=0, lw=1)
    ax.axhline(-1, color="k",zorder=0, lw=1)
    # annotate lines
    # wheat_d_offset = [-3,3,0,0,0]
    # for i, (d, l, r) in enumerate(zip(wheat_dates, wheat_levels, meas_dates.wheat_names)):
    #     ax.annotate(r, xy=(d, l),
    #                 xytext=(0 + wheat_d_offset[i], np.sign(l) * 0.5), textcoords="offset points",
    #                 horizontalalignment="center",
    #                 verticalalignment="bottom" if l > 0 else "top")
    # maize_d_offset = [-3,3,3,-3,0,0]    
    # ax.vlines(maize_dates, 0, maize_levels, color="tab:blue")  # The vertical stems.
    ax.scatter(maize_dates, 0*np.ones_like(maize_dates), marker="o",
            color="k", facecolor="w", zorder=50, label="Maize")  # Baseline and markers on it.

    # annotate lines
    # for i, (d, l, r) in enumerate(zip(maize_dates, maize_levels, meas_dates.maize_names)):
    #     ax.annotate(r, xy=(d, l),
    #                 xytext=(maize_d_offset[i], np.sign(l)*2), textcoords="offset points",
    #                 horizontalalignment="center",
    #                 verticalalignment="bottom" if l > 0 else "top")
    # s2_d_offset = [0,0,0,0,0,0,0,0,0,0,0,0,10]
    if s2_dates is not None and s2_names is not None:
        s2_dates = [datetime.strptime(d, "%Y-%m-%d") for d in s2_dates]
        # s2_levels = np.tile([ 2, 2, 2, 2, 2, 2],
        #                     int(np.ceil(len(s2_dates)/6)))[:len(s2_dates)]
        # s2_levels = np.tile([ 0, 0, 0, 0, 0, 0],
        #                     int(np.ceil(len(s2_dates)/6)))[:len(s2_dates)]
        # ax.vlines(s2_dates, 0, s2_levels, color="tab:green")  # The vertical stems.
        ax.scatter(s2_dates, -1 * np.ones_like(s2_dates), marker="*", s=100,
                    color="k", facecolor="g", zorder=100, label="S2")  # Baseline and markers on it.

        # annotate lines
        # for i, (d, l, r) in enumerate(zip(s2_dates, s2_levels, s2_names)):
        #     ax.annotate(r, xy=(d, l),
        #                 xytext=(-3 + s2_d_offset[i], -6), textcoords="offset points",
        #                 horizontalalignment="right",
        #                 verticalalignment="bottom" if l > 0 else "top")
    # format x-axis with 1-week intervals
    ax.set_ylim(-1.5,1.5)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y "))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y-axis and spines
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    return fig, anext


# months_to_get = ["2016-02-01",
#                     "2016-06-01",
#                     "2016-10-01",
#                     "2018-04-01",
#                     "2018-08-01",
#                     "2018-10-01",
#                     "2017-03-01",
#                     "2017-07-01",
#                     "2017-11-01",
#                     "2019-01-01",
#                     "2019-05-01",
#                     "2019-09-01"]

# fig, ax = plot_sampling_dates(months_to_get)

def get_belsar_sites_time_series(metrics, belsar_data_dir, site="W1", fig=None, ax=None, label="", use_ref_metrics=False):
    validation_df, _, _, _, _, _, _ = load_belsar_validation_data(belsar_data_dir, "2A_20180508_both_BelSAR_agriculture_database") 
    measurement_dates = []
    lai = []
    lai_std = []
    fields = []
    for date in pd.unique(validation_df["date"]):
        sub_val_df = validation_df[validation_df["date"]==date]
        # fields_ids = pd.unique(sub_val_df["Field ID"])
        # for field_id in fields_ids:
        if not len(sub_val_df[sub_val_df["Field ID"]==site])==0:
            measurement_dates.append(date)
            lai.append(np.mean(sub_val_df[sub_val_df["Field ID"]==site]["lai"].values))
            lai_std.append(np.std(sub_val_df[sub_val_df["Field ID"]==site]["lai"].values))
            fields.append(site)
    site_metrics = metrics[metrics["name"]==site]
    all_metrics = site_metrics[["lai_mean", "lai_sigma_mean", "name"]].copy()
    all_metrics.insert(0,'type', [label for _ in range(len(all_metrics))])
    all_metrics = all_metrics.rename(columns={'lai_mean': 'LAI'})
    all_metrics = all_metrics.rename(columns={'lai_sigma_mean': 'lai_std'})
    dates_pred = site_metrics["date"].values
    dates_pred = [datetime.strptime(d, "%Y-%m-%d") for d in dates_pred]
    all_metrics.insert(0, "Date", dates_pred)
        # all_metrics = pd.concat((ref_metrics, all_metrics))
    all_metrics.reset_index(inplace=True, drop=True)
    # lai_pred = site_metrics["lai_mean"].values
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=150)
    # sns.scatterplot(data=all_metrics, x="Date", y="LAI",ax=ax, hue="type")
    ax.scatter(all_metrics["Date"].values, all_metrics['LAI'].values, label=label)
    ax.errorbar(all_metrics['Date'].values, all_metrics['LAI'].values, yerr=all_metrics['lai_std'],
                ecolor='k', capthick=1, fmt='o', linestyle='', markersize=0.1,
                elinewidth=0.5, zorder=0)
    if use_ref_metrics:
        ref_metrics = pd.DataFrame({"Date":[datetime.strptime(d, "%Y-%m-%d") for d in measurement_dates],
                                    "LAI":lai,
                                    "lai_std":lai_std,
                                    "name":fields,
                                    "type":["Measurement" for d in measurement_dates]})
        ax.scatter(ref_metrics["Date"].values, ref_metrics['LAI'].values, label="Measurement")
        ax.errorbar(ref_metrics['Date'].values, ref_metrics['LAI'].values, yerr=ref_metrics['lai_std'],
                ecolor='k', capthick=1, fmt='o', linestyle='', markersize=0.1,
                elinewidth=0.5, zorder=0)
    # ax.scatter(dates_pred, lai_pred)
    ax.set_ylabel("LAI")
    # ax.set_ylabel("Date")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y "))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig, ax

def get_belsar_lai_vs_hspot(metrics, belsar_data_dir, sites=["W1"], fig=None, ax=None, label=""):
    validation_df, _, _, _, _, _, _ = load_belsar_validation_data(belsar_data_dir, "2A_20180508_both_BelSAR_agriculture_database") 
    measurement_dates = []
    lai = []
    lai_std = []
    fields = []
    for site in sites:
        for date in pd.unique(validation_df["date"]):
            sub_val_df = validation_df[validation_df["date"]==date]
            # fields_ids = pd.unique(sub_val_df["Field ID"])
            # for field_id in fields_ids:
            if not len(sub_val_df[sub_val_df["Field ID"]==site])==0:
                measurement_dates.append(date)
                lai.append(np.mean(sub_val_df[sub_val_df["Field ID"]==site]["lai"].values))
                lai_std.append(np.std(sub_val_df[sub_val_df["Field ID"]==site]["lai"].values))
                fields.append(site)
    site_metrics = metrics[metrics["name"]==site]
    all_metrics = site_metrics[["lai_mean", "lai_sigma_mean", "hspot_mean", "hspot_sigma_mean", "name"]].copy()
    all_metrics['type'] = [label for _ in range(len(all_metrics))]
    all_metrics = all_metrics.rename(columns={'lai_mean': 'LAI'})
    all_metrics = all_metrics.rename(columns={'lai_sigma_mean': 'lai_std'})
    all_metrics = all_metrics.rename(columns={'hspot_mean': 'hspot'})
    all_metrics = all_metrics.rename(columns={'hspot_sigma_mean': 'hspot_std'})
    dates_pred = site_metrics["date"].values
    dates_pred = [datetime.strptime(d, "%Y-%m-%d") for d in dates_pred]
    all_metrics["Date"] = dates_pred
        # all_metrics = pd.concat((ref_metrics, all_metrics))
    all_metrics.reset_index(inplace=True, drop=True)
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=150)
    # sns.scatterplot(data=all_metrics, x="Date", y="LAI",ax=ax, hue="type")
    ax.scatter(all_metrics["LAI"].values, all_metrics['hspot'].values, label=label)
    ax.errorbar(all_metrics['LAI'].values, all_metrics['hspot'].values, 
                xerr=all_metrics['lai_std'], yerr=all_metrics['hspot_std'],
                ecolor='k', capthick=1, fmt='o', linestyle='', markersize=0.1,
                elinewidth=0.5, zorder=0)
    ax.set_xlabel("LAI")
    ax.set_ylabel("hspot")
    return fig, ax

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

if __name__=="__main__":
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        'mathtext.fontset' : 'custom',
        'mathtext.rm': 'Bitstream Vera Sans',
        'mathtext.it': 'Bitstream Vera Sans:italic',
        'mathtext.bf': 'Bitstream Vera Sans:bold'
    })
    s2_dates = ["2018-05-08", 
                "2018-05-18", 
                "2018-05-26", 
                "2018-05-28", 
                "2018-06-20", 
                "2018-06-27",
                "2018-06-30",
                "2018-07-15", 
                "2018-07-22",
                "2018-07-25", 
                "2018-07-27", 
                "2018-08-04"]
    fig, ax = plot_measurements_and_s2_dates(s2_dates=s2_dates, 
                                            #  s2_names=["2A","2A", "2A", "2A", "2A", "2B", "2B", "2B", "2A", "2B"]
                                            s2_names=[r"$S$" for i in range(len(s2_dates))]
                                                )
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation/belsar_dates.tex")
    fig.savefig("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation/dates.svg")

    list_output_filenames = ["2A_20180518_both_BelSAR_agriculture_database",
                                "2A_20180528_both_BelSAR_agriculture_database",
                                "2A_20180620_both_BelSAR_agriculture_database",
                                "2A_20180627_both_BelSAR_agriculture_database",
                                "2B_20180715_both_BelSAR_agriculture_database",
                                "2B_20180722_both_BelSAR_agriculture_database",
                                "2A_20180727_both_BelSAR_agriculture_database",
                                "2B_20180801_both_BelSAR_agriculture_database",
                                "2B_20180804_both_BelSAR_agriculture_database",
                                "2A_20180508_both_BelSAR_agriculture_database"]

    data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation/"
    # for output_filename in list_output_filenames:
        # plot_belsar_site(data_dir, output_filename)

