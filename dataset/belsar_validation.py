import os
import pandas as pd
import geopandas as gpd
from sensorsio import sentinel2, utils
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import socket
import argparse
from utils.image_utils import tensor_to_raster
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import zipfile
import shutil
from rasterio.mask import mask

@dataclass
class MeasurementDates:
    # wheat_names = ["W1", "W2", "W3", "W4", "W5"]
    wheat_names = ["W", "W", "W", "W", "W"]
    wheat_dates = ["2018-05-17", "2018-05-18", "2018-06-05", "2018-06-21", "2018-07-19"]
    # maize_names = ["M1", "M2", "M3", "M4", "M5", "M6"]
    maize_names = ["M", "M", "M", "M", "M", "M"]
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
    ax.set(title="Measurement dates in BelSAR campaign")

    ax.vlines(wheat_dates, 0, wheat_levels, color="tab:red")  # The vertical stems.
    ax.scatter(wheat_dates, np.zeros_like(wheat_dates), marker="o",
                color="k", facecolor="w")  # Baseline and markers on it.
    ax.axhline(0, color="k",zorder=0)
    # annotate lines
    wheat_d_offset = [-3,3,0,0,0]
    for i, (d, l, r) in enumerate(zip(wheat_dates, wheat_levels, meas_dates.wheat_names)):
        ax.annotate(r, xy=(d, l),
                    xytext=(0 + wheat_d_offset[i], np.sign(l)*0.5), textcoords="offset points",
                    horizontalalignment="center",
                    verticalalignment="bottom" if l > 0 else "top")
    maize_d_offset = [-3,3,3,-3,0,0]    
    ax.vlines(maize_dates, 0, maize_levels, color="tab:blue")  # The vertical stems.
    ax.scatter(maize_dates, np.zeros_like(maize_dates), marker="o",
            color="k", facecolor="w")  # Baseline and markers on it.

    # annotate lines
    for i, (d, l, r) in enumerate(zip(maize_dates, maize_levels, meas_dates.maize_names)):
        ax.annotate(r, xy=(d, l),
                    xytext=(maize_d_offset[i], np.sign(l)*2), textcoords="offset points",
                    horizontalalignment="center",
                    verticalalignment="bottom" if l > 0 else "top")
    s2_d_offset = [0,0,0,0,0,0,0,0,10]
    if s2_dates is not None and s2_names is not None:
        s2_dates = [datetime.strptime(d, "%Y-%m-%d") for d in s2_dates]
        # s2_levels = np.tile([ 2, 2, 2, 2, 2, 2],
        #                     int(np.ceil(len(s2_dates)/6)))[:len(s2_dates)]
        s2_levels = np.tile([ 0, 0, 0, 0, 0, 0],
                            int(np.ceil(len(s2_dates)/6)))[:len(s2_dates)]
        # ax.vlines(s2_dates, 0, s2_levels, color="tab:green")  # The vertical stems.
        ax.scatter(s2_dates, np.zeros_like(s2_dates), marker="*", s=100,
                    color="k", facecolor="g")  # Baseline and markers on it.

        # annotate lines
        for i, (d, l, r) in enumerate(zip(s2_dates, s2_levels, s2_names)):
            ax.annotate(r, xy=(d, l),
                        xytext=(-3 + s2_d_offset[i], -2), textcoords="offset points",
                        horizontalalignment="right",
                        verticalalignment="bottom" if l > 0 else "top")
    # format x-axis with 1-week intervals
    ax.set_ylim(-1.2,1.2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y "))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y-axis and spines
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    return fig, anext

def get_data_point_bb(gdf, dataset, margin=100, res=10):
    left, right, bottom, top = (dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top)
    x_data_point = np.round(gdf["geometry"].x.values / res) * res
    y_data_point = np.round(gdf["geometry"].y.values / res) * res
    assert all(x_data_point > left) and all(x_data_point < right)
    assert all(y_data_point > bottom) and all(x_data_point < top)
    left = float(int(min(x_data_point) - margin))
    bottom = float(int(min(y_data_point) - margin))
    right = float(int(max(x_data_point) + margin))
    top = float(int(max(y_data_point) + margin))
    return rio.coords.BoundingBox(left, bottom, right, top)

def get_data_idx_in_image(gdf, xmin_image_bb, ymax_image_bb, col_offset, row_offset, res=10):
    x_data_point = (np.round(gdf["geometry"].x.values / 10) * 10 - xmin_image_bb) / res
    y_data_point = (ymax_image_bb - np.round(gdf["geometry"].y.values / 10) * 10) / res
    gdf["x_idx"] = x_data_point - col_offset
    print(gdf["x_idx"])
    gdf["y_idx"] = y_data_point - row_offset
    print(gdf["y_idx"])


def get_bb_array_index(bb, image_bb, res=10):
    xmin = (bb[0] - image_bb[0]) / res
    ymin = ( - (bb[3] - image_bb[3])) / res
    xmax = xmin + (bb[2] - bb[0]) / res
    ymax = ymin + (bb[3] - bb[1]) / res
    return int(xmin), int(ymin), int(xmax), int(ymax)

def get_prosailvae_train_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-f", dest="data_filename",
                        help="name of data files (without extension)",
                        type=str, default="FRM_Veg_Barrax_20180605")

    parser.add_argument("-d", dest="data_dir",
                        help="name of data files (without extension)",
                        type=str, default="/home/uz/zerahy/scratch/prosailvae/data/silvia_validation/")
    
    parser.add_argument("-p", dest="product_name",
                        help="Theia product name",
                        type=str, default="")
    return parser

def get_variable_column_names(variable="lai"):
    if variable == "lai":
        return "LAI", "Uncertainty.1"
    if variable == "lai_eff":
        return "LAIeff", "Uncertainty"
    if variable == "ccc":
        return "CCC (g m-2)", "Uncertainty (g m-2).2"
    if variable == "ccc_eff":
        return "CCCeff (g m-2)", "Uncertainty (g m-2).1"
    else:
        raise NotImplementedError

def compute_belsar_validation_data(data_dir, filename, s2_product_name, crop ="maize", margin=100):

    output_file_name = s2_product_name[8:19] + f"_{crop}_" + filename
    data_file = filename + ".xlsx"
    if crop != "both":
        data_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_{crop}", skiprows=[0])
        data_df.rename(columns={"Date": "date", "PAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
    else:
        wheat_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_wheat", skiprows=[0])
        wheat_df.rename(columns={"Date": "date", "PAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        maize_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_maize", skiprows=[0])
        maize_df.rename(columns={"Date": "date", "GAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        data_df = pd.concat((wheat_df, maize_df))
    data_df = data_df.drop(columns=['Flight N°', 'BBCH', 'Line 1-1',
       'Line 1-2', 'Line 1-3', 'Line 2-1', 'Line 2-2', 'Line 2-3', 'Line 3-1',
       'Line 3-2', 'Line 3-3', 'Mean','FCOVER',
       'Total Fresh weight (g)', 'Sample fresh wieght (g)', 'Dry matter content (%)',
       'Interline distance mean (cm)', 'Interplant distance (cm)',
       'Note/comment']).dropna()
    if crop == "maize":
        left, bottom, right, top = (526023, 6552201, 535123, 6558589)
    elif crop== "wheat":
        left, bottom, right, top = (522857, 6544318, 531075, 6553949)
    elif crop== "both":
        left, bottom, right, top = (522857, 6544318, 535123, 6558589)
        
    else:
        raise ValueError
    src_data_epsg = "epsg:3857"
    
    # data_gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df['Easting Coord. '], data_df['Northing Coord. ']))
    # data_gdf = data_gdf.set_crs('epsg:4326')

    # df = df.drop(columns=['Comments']).dropna()

    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    left, bottom, right, top = rio.warp.transform_bounds(src_data_epsg, dataset.crs.to_epsg(), left, bottom, right, top)
    bb = rio.coords.BoundingBox(left - margin, bottom - margin, right + margin, top + margin)
    # margin = 100
    # bb = get_data_point_bb(data_gdf, dataset, margin=margin)
    bands = [sentinel2.Sentinel2.B2,
             sentinel2.Sentinel2.B3,
             sentinel2.Sentinel2.B4,
             sentinel2.Sentinel2.B5,
             sentinel2.Sentinel2.B6,
             sentinel2.Sentinel2.B7,
             sentinel2.Sentinel2.B8,
             sentinel2.Sentinel2.B8A,
             sentinel2.Sentinel2.B11,
             sentinel2.Sentinel2.B12]

    xmin, ymin, xmax, ymax = get_bb_array_index(bb, dataset.bounds, res=10)

    if socket.gethostname()=='CELL200973':
        even_zen, odd_zen = dataset.read_zenith_angles_as_numpy()
        even_zen = even_zen[ymin:ymax, xmin:xmax]
        odd_zen = odd_zen[ymin:ymax, xmin:xmax]
        joint_zen = np.array(even_zen)
        joint_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
        del even_zen
        del odd_zen

        even_az, odd_az = dataset.read_azimuth_angles_as_numpy()
        even_az = even_az[ymin:ymax, xmin:xmax]
        odd_az = odd_az[ymin:ymax, xmin:xmax]
        joint_az = np.array(even_az)
        joint_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]
        del even_az
        del odd_az

    else:
        even_zen, odd_zen, even_az, odd_az = dataset.read_incidence_angles_as_numpy()
        joint_zen = np.array(even_zen)
        joint_zen[np.isnan(even_zen)]=odd_zen[np.isnan(even_zen)]
        del even_zen
        del odd_zen
        joint_az = np.array(even_az)
        joint_az[np.isnan(even_az)]=odd_az[np.isnan(even_az)]
        del even_az
        del odd_az
    sun_zen, sun_az = dataset.read_solar_angles_as_numpy()
    sun_zen = sun_zen[ymin:ymax, xmin:xmax]
    sun_az = sun_az[ymin:ymax, xmin:xmax]
    s2_a = np.stack((sun_zen, joint_zen, sun_az - joint_az), 0).data
    print(s2_a.shape)
    np.save(os.path.join(data_dir, output_file_name + "_angles.npy"), s2_a)
    s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(bands, bounds=bb,
                                                                    crs=dataset.crs,
                                                                    band_type=dataset.SRE)
    validity_mask = np.sum(masks.data, axis=0, keepdims=True).astype(bool).astype(int).astype(float)
    s2_r = s2_r.data
    print(s2_r.shape)
    np.save(os.path.join(data_dir, output_file_name + "_refl.npy"), s2_r)
    np.save(os.path.join(data_dir, output_file_name + "_xcoords.npy"), xcoords)
    np.save(os.path.join(data_dir, output_file_name + "_ycoords.npy"), ycoords)
    np.save(os.path.join(data_dir, output_file_name + "_mask.npy"), validity_mask)
    arr_rgb, dmin, dmax = utils.rgb_render(s2_r, bands=[2,1,0],
                                            dmin=np.array([0., 0., 0.]),
                                            dmax=np.array([0.2,0.2,0.2]))
    # fig, ax = plt.subplots()
    # ax.imshow(arr_rgb, extent = [bb[0], bb[2], bb[1], bb[3]])
    # plt.show()
    # fig, ax = plt.subplots()
    # ax.imshow(arr_rgb)
    # plt.show()
    
    data_df.to_csv(os.path.join(data_dir, output_file_name + ".csv"))
    return output_file_name


def load_belsar_validation_data(data_dir, filename):
    df = pd.read_csv(os.path.join(data_dir, filename + ".csv"))
    s2_r = np.load(os.path.join(data_dir, filename + "_refl.npy"))
    s2_a = np.load(os.path.join(data_dir, filename + "_angles.npy"))
    mask = np.load(os.path.join(data_dir, filename + "_mask.npy"))
    xcoords = np.load(os.path.join(data_dir, filename + "_xcoords.npy"))
    ycoords = np.load(os.path.join(data_dir, filename + "_ycoords.npy"))
    return df, s2_r, s2_a, mask, xcoords, ycoords, rio.CRS.from_epsg(32631)

def get_sites_geometry(data_dir, crs, crop:str|None=None):
    import fiona
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    sites_geometry = gpd.GeoDataFrame()
    for layer in fiona.listlayers(os.path.join(data_dir,"BelSAR"+'.kml')):
        s = gpd.read_file(os.path.join(data_dir,"BelSAR"+'.kml'), driver='KML', layer=layer)
        sites_geometry = pd.concat([sites_geometry, s], ignore_index=True)
    sites_geometry.to_crs(crs, inplace=True)
    sites_geometry = sites_geometry[sites_geometry["Name"].apply(lambda x: len(x)>0)]
    sites_geometry = sites_geometry[sites_geometry['geometry'].apply(lambda x : x.geom_type=='Polygon' )]
    sites_geometry.reset_index(inplace=True, drop=True)
    if crop is None:
        return sites_geometry
    if crop == "maize":
        return sites_geometry[sites_geometry["Name"].apply(lambda x: x[0]=="M")]    
    if crop == "wheat":
        return sites_geometry[sites_geometry["Name"].apply(lambda x: x[0]=="W")]
    else:
        raise ValueError

def plot_belsar_site(data_dir, filename):
    df, s2_r, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(data_dir, filename)

    # fig, ax = plt.subplots()
    # visu, _, _ = utils.rgb_render(s2_r)
    # ax.imshow(visu, extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]])
    # wheat_sites.plot(ax=ax,  color = 'red')
    # maize_sites.plot(ax=ax,  color = 'blue')
    maize_sites = get_sites_geometry(data_dir, crs, crop="maize")
    wheat_sites = get_sites_geometry(data_dir, crs, crop="wheat")
    mask[mask==0.] = np.nan
    fig, ax = plt.subplots(dpi=200)
    visu, _, _ = utils.rgb_render(s2_r)
    ax.imshow(visu, extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]])
    ax.imshow(mask.squeeze(), extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]], cmap='YlOrRd')
    for i in range(len(maize_sites)):
        contour = maize_sites["geometry"].iloc[i].exterior.xy
        ax.plot(contour[0], contour[1], "blue", linewidth=0.5)
    # maize_sites.plot(ax=ax,  color = 'blue')
    # wheat_sites.plot(ax=ax,  color = 'green')
    for i in range(len(maize_sites)):
        contour = wheat_sites["geometry"].iloc[i].exterior.xy
        ax.plot(contour[0], contour[1], "red", linewidth=0.5)
    for xi, yi, text in zip(wheat_sites.centroid.x, wheat_sites.centroid.y, wheat_sites["Name"]):
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points',
                color='red')
    for xi, yi, text in zip(maize_sites.centroid.x, maize_sites.centroid.y, maize_sites["Name"]):
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points',
                color='blue')
    
    fig.savefig(os.path.join(data_dir, filename + "_mask.png"))

def main():
    if socket.gethostname()=='CELL200973':
        args=["-f", "BelSAR_agriculture_database",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation/",
            #   "-p", "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"]
              "-p", "SENTINEL2A_20180518-104024-461_L2A_T31UFS_C_V2-2"]
        # "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()
    # gdf, s2_r, s2_a = load_validation_data(parser.data_dir, parser.filename)


    months_to_get = ["2016-02-01",
                     "2016-06-01",
                     "2016-10-01",
                     "2018-04-01",
                     "2018-08-01",
                     "2018-10-01",
                     "2017-03-01",
                     "2017-07-01",
                     "2017-11-01",
                     "2019-01-01",
                     "2019-05-01",
                     "2019-09-01"]
    fig, ax = plot_sampling_dates(months_to_get)

    fig, ax = plot_measurements_and_s2_dates(s2_dates=["2018-05-08", "2018-05-18", "2018-05-28", "2018-06-20", "2018-06-27",
                                                       "2018-07-15", "2018-07-22", "2018-07-27", "2018-08-04"], 
                                            #  s2_names=["2A","2A", "2A", "2A", "2A", "2B", "2B", "2B", "2A", "2B"]
                                            s2_names=["S","S", "S", "S", "S", "S", "S", "S", "S"]
                                             )
    fig.savefig("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation/dates.svg")
    s2_product_name = parser.product_name


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
    for output_filename in list_output_filenames:
        plot_belsar_site(parser.data_dir, output_filename)


    list_s2_products = ["SENTINEL2A_20180508-104025-460_L2A_T31UFS_D_V2-2.zip",
                        'SENTINEL2A_20180528-104613-414_L2A_T31UFS_D_V2-2.zip',
                        'SENTINEL2A_20180620-105211-086_L2A_T31UFS_D_V2-2.zip',
                        'SENTINEL2B_20180801-104018-457_L2A_T31UFS_D_V2-2.zip',
                        'SENTINEL2A_20180518-104024-461_L2A_T31UFS_D_V2-2.zip',
                        'SENTINEL2B_20180804-105022-459_L2A_T31UFS_D_V2-2.zip',
                        "SENTINEL2B_20180722-104020-458_L2A_T31UFS_D_V2-2.zip",
                        "SENTINEL2A_20180627-104023-457_L2A_T31UFS_D_V2-2.zip",
                        "SENTINEL2B_20180715-105300-591_L2A_T31UFS_D_V1-8.zip",
                        "SENTINEL2A_20180727-104023-458_L2A_T31UFS_D_V2-2.zip"]
    for s2_product in list_s2_products:
        if s2_product[-4:]=='.zip':
            with zipfile.ZipFile(os.path.join(parser.data_dir, s2_product), 'r') as zip_ref:
                zip_ref.extractall(parser.data_dir)
                s2_product = os.path.normpath(str(list(zipfile.Path(os.path.join(parser.data_dir, s2_product)).iterdir())[0])).split(os.sep)[-1]
            output_file_names = compute_belsar_validation_data(parser.data_dir, parser.data_filename, s2_product, crop="both")
            shutil.rmtree(os.path.join(parser.data_dir, s2_product))
        else:
            output_file_names = compute_belsar_validation_data(parser.data_dir, parser.data_filename, s2_product, crop="both")
        print(output_file_names)
if __name__ == "__main__":
    main()