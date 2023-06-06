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

def compute_validation_data(data_dir, filename, s2_product_name):
    output_file_name = s2_product_name[8:19] + "_" + filename
    data_file = filename + ".xlsx"
    data_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name="GroundData", skiprows=[0])
    data_df = data_df.drop(columns=['Comments'])
    data_gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df['Easting Coord. '], data_df['Northing Coord. ']))
    data_gdf = data_gdf.set_crs('epsg:4326')

    # df = df.drop(columns=['Comments']).dropna()

    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    data_gdf = data_gdf.to_crs(dataset.crs.to_epsg())
    margin = 100
    bb = get_data_point_bb(data_gdf, dataset, margin=margin)
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
    s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(bands, bounds=bb,
                                                                    crs=dataset.crs,
                                                                    band_type=dataset.SRE)

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
    s2_r = s2_r.data
    print(s2_r.shape)
    np.save(os.path.join(data_dir, output_file_name + "_refl.npy"), s2_r)
    np.save(os.path.join(data_dir, output_file_name + "_xcoords.npy"), xcoords)
    np.save(os.path.join(data_dir, output_file_name + "_ycoords.npy"), ycoords)
    arr_rgb, dmin, dmax = utils.rgb_render(s2_r, bands=[2,1,0],
                                            dmin=np.array([0., 0., 0.]),
                                            dmax=np.array([0.2,0.2,0.2]))
    fig, ax = plt.subplots()
    ax.imshow(arr_rgb, extent = [bb[0], bb[2], bb[1], bb[3]])
    data_gdf.plot(ax=ax)
    plt.show()
    get_data_idx_in_image(data_gdf, dataset.bounds[0], dataset.bounds[3], xmin, ymin, res=10)
    fig, ax = plt.subplots()
    ax.imshow(arr_rgb)
    ax.scatter(data_gdf["x_idx"], data_gdf['y_idx'])
    plt.show()
    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        variable_col, uncertainty_col = get_variable_column_names(variable=variable)
        gdf = data_gdf[[variable_col, uncertainty_col, "Land Cover", "x_idx", "y_idx",
                        "geometry", "Date (dd/mm/yyyy)"]].dropna()
        gdf.rename(columns = {variable_col:variable,
                              uncertainty_col:"uncertainty",
                              "Land Cover":"land cover", 
                              "Date (dd/mm/yyyy)": "date"}, inplace = True)
        if variable in ["ccc", "ccc_eff"]:
            gdf[variable] = gdf[variable] * 100
            gdf["uncertainty"] = gdf["uncertainty"] * 100
        gdf.to_file(os.path.join(data_dir, output_file_name + f"_{variable}.geojson"), driver="GeoJSON")
    return output_file_name

def load_validation_data(data_dir, filename, variable="lai"):
    gdf = gpd.read_file(os.path.join(data_dir, filename + f"_{variable}.geojson"),
                        driver="GeoJSON")
    s2_r = np.load(os.path.join(data_dir, filename + "_refl.npy"))
    s2_a = np.load(os.path.join(data_dir, filename + "_angles.npy"))
    arr_rgb, dmin, dmax = utils.rgb_render(s2_r, bands=[2, 1, 0],
                                    dmin=np.array([0., 0., 0.]),
                                    dmax=np.array([0.2,0.2,0.2]))
    xcoords = np.load(os.path.join(data_dir, filename + "_xcoords.npy"))
    ycoords = np.load(os.path.join(data_dir, filename + "_ycoords.npy"))
    return gdf, s2_r, s2_a, xcoords, ycoords

def main():
    if socket.gethostname()=='CELL200973':
        args=["-f", "FRM_Veg_Barrax_20180605",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/silvia_validation/",
            #   "-p", "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"]
              "-p", "SENTINEL2A_20180613-110957-425_L2A_T30SWJ_D_V1-8"]
        # "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()
    # gdf, s2_r, s2_a = load_validation_data(parser.data_dir, parser.filename)
    s2_product_name = parser.product_name
    output_file_names = compute_validation_data(parser.data_dir, parser.data_filename, s2_product_name)
    print(output_file_names)
if __name__ == "__main__":
    main()