import os
import pandas as pd
import geopandas as gpd
from sensorsio import sentinel2, utils
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import socket
import argparse
from datetime import datetime
import torch


from tqdm import tqdm
if __name__ == "__main__":
    from validation_utils import var_of_product, simple_interpolate
else:
    from validation.validation_utils import var_of_product, simple_interpolate
from utils.image_utils import get_encoded_image_from_batch
from dataset.weiss_utils import get_weiss_biophyiscal_from_batch
from prosailvae.ProsailSimus import BANDS

BARRAX_FILENAMES = ["2B_20180516_FRM_Veg_Barrax_20180605_V2", "2A_20180613_FRM_Veg_Barrax_20180605_V2"]
BARRAX_2021_FILENAME = "2B_20210722_FRM_Veg_Barrax_20210719"
WYTHAM_FILENAMES = ["2A_20180629_FRM_Veg_Wytham_20180703_V2", "2A_20180706_FRM_Veg_Wytham_20180703_V2"]

# from utils.image_utils import tensor_to_raster

# def get_silvia_validation_metrics(PROSAIL_VAE, data_dir, filename,  mode='lat_mode'):
#     gdf, s2_r, s2_a, xcoords, ycoords = load_frm4veg_data(data_dir, filename)
#     x_idx = torch.from_numpy(gdf["x_idx"].values) - PROSAIL_VAE.encoder.nb_enc_cropped_hw
#     y_idx = torch.from_numpy(gdf["y_idx"].values) - PROSAIL_VAE.encoder.nb_enc_cropped_hw
#     s2_r = torch.from_numpy(s2_r).unsqueeze(0).float()
#     s2_a = torch.from_numpy(s2_a).unsqueeze(0).float()
#     (rec_image, sim_image, cropped_s2_r, cropped_s2_a,
#      sigma_image) = get_encoded_image_from_batch((s2_r, s2_a), PROSAIL_VAE, patch_size=32,
#                                                  bands=torch.arange(10),
#                                                  mode=mode)
#     lai_pred = sim_image[:,6, y_idx, x_idx]
#     cab_pred = sim_image[:,1, y_idx, x_idx]
#     ccc_pred = cab_pred * lai_pred

#     lai = torch.from_numpy(gdf["LAI"].values)
#     lai_uncert = torch.from_numpy(gdf["Uncertainty.1"].values)
#     lai_eff = torch.from_numpy(gdf["LAIeff"].values)
#     lai_eff_uncert = torch.from_numpy(gdf["Uncertainty"].values)
#     ccc = torch.from_numpy(gdf["CCC"].values)
#     ccc_uncert = torch.from_numpy(gdf["Uncertainty (g m-2).2"].values)
#     ccc_eff = torch.from_numpy(gdf["CCCeff"].values)
#     ccc_eff_uncert = torch.from_numpy(gdf["Uncertainty (g m-2).1"].values)


#     return


def get_data_point_bb(gdf, dataset, margin=100, res=10):
    left, right, bottom, top = (dataset.bounds.left, dataset.bounds.right, 
                                dataset.bounds.bottom, dataset.bounds.top)
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
                        type=str, default="/home/uz/zerahy/scratch/prosailvae/data/frm4veg_validation/")
    
    parser.add_argument("-p", dest="product_name",
                        help="Theia product name",
                        type=str, default="")
    return parser

def get_variable_column_names(variable="lai", wytham=False):
    if variable == "lai":
        if wytham:
            return "LAI", "Uncertainty.5"
        return "LAI", "Uncertainty.1"
    if variable == "lai_eff":
        if wytham:
            return "LAIeff", "Uncertainty.2"
        return "LAIeff", "Uncertainty"
    if variable == "ccc":
        return "CCC (g m-2)", "Uncertainty (g m-2).2"
    if variable == "ccc_eff":
        return "CCCeff (g m-2)", "Uncertainty (g m-2).1"
    else:
        raise NotImplementedError

def get_bb_equivalent_polygon(bb, in_crs, out_crs):
    from shapely import Polygon
    coords = ((bb.left, bb.bottom), (bb.left, bb.top), (bb.right, bb.top), (bb.right, bb.bottom), (bb.left, bb.bottom))
    polygon = Polygon(coords)
    return gpd.GeoDataFrame(data={"geometry":[polygon]}).set_crs(in_crs).to_crs(out_crs)

def compute_frm4veg_data(data_dir, filename, s2_product_name, no_angle_data=False, date="2018-06-30", method="DHP"):
    output_file_name = s2_product_name[8:19] + "_" + filename
    data_file = filename + ".xlsx"
    data_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name="GroundData", skiprows=[0])
    data_df = data_df.drop(columns=['Comments'])
    data_df = data_df[data_df["Method"]==method]
    data_gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df['Easting Coord. '], data_df['Northing Coord. ']))
    data_gdf = data_gdf.set_crs('epsg:4326')

    # df = df.drop(columns=['Comments']).dropna()

    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    data_gdf = data_gdf.to_crs(dataset.crs.to_epsg())
    margin = 100
    bb = get_data_point_bb(data_gdf, dataset, margin=margin)
    gdf = get_bb_equivalent_polygon(bb, dataset.crs.to_epsg(), 'epsg:4326')
    gdf.to_file("rois_to_download.geojson", driver="GeoJSON")
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
    if no_angle_data:
        cn_path = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/iris_data/openEO_0_clip.nc"
        _, s2_a = load_iris_data(cn_path, date=date)
    else:
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
    if masks.sum()>0:
        raise ValueError
    s2_r = s2_r.data
    print(s2_r.shape)
    np.save(os.path.join(data_dir, output_file_name + "_refl.npy"), s2_r)
    np.save(os.path.join(data_dir, output_file_name + "_xcoords.npy"), xcoords)
    np.save(os.path.join(data_dir, output_file_name + "_ycoords.npy"), ycoords)
    get_data_idx_in_image(data_gdf, dataset.bounds[0], dataset.bounds[3], xmin, ymin, res=10)

    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        variable_col, uncertainty_col = get_variable_column_names(variable=variable, wytham=no_angle_data)
        gdf = data_gdf[[variable_col, uncertainty_col, "Land Cover", "x_idx", "y_idx",
                        "geometry", "Date (dd/mm/yyyy)"]].dropna().reset_index(drop=True)
        gdf.rename(columns = {variable_col:variable,
                              uncertainty_col:"uncertainty",
                              "Land Cover":"land cover", 
                              "Date (dd/mm/yyyy)": "date"}, inplace = True)
        if variable in ["ccc", "ccc_eff"]:
            gdf[variable] = gdf[variable] * 100
            gdf["uncertainty"] = gdf["uncertainty"] * 100
        gdf.to_file(os.path.join(data_dir, output_file_name + f"_{variable}.geojson"), driver="GeoJSON")
    return output_file_name

def load_frm4veg_data(data_dir, filename, variable="lai"):
    gdf = gpd.read_file(os.path.join(data_dir, filename + f"_{variable}.geojson"),
                        driver="GeoJSON")
    s2_r = np.load(os.path.join(data_dir, filename + "_refl.npy"))
    s2_a = np.load(os.path.join(data_dir, filename + "_angles.npy"))
    xcoords = np.load(os.path.join(data_dir, filename + "_xcoords.npy"))
    ycoords = np.load(os.path.join(data_dir, filename + "_ycoords.npy"))
    return gdf, s2_r, s2_a, xcoords, ycoords

def load_iris_data(file_path, date = "2018-06-29"):
    import xarray
    import datetime
    import torch
    ds = xarray.open_dataset(file_path)
    d = datetime.datetime.strptime(date, "%Y-%m-%d")
    t = ds.t.values
    idx = 0
    for i in range(len(t)):
        if pd.Timestamp(d) == pd.Timestamp(t[i]):
            idx=i
            break
    s2_r = torch.from_numpy(np.stack([ds['B02'].values, ds['B03'].values, ds['B04'].values, ds['B05'].values,
                     ds['B06'].values, ds['B07'].values, ds['B08'].values, ds['B8A'].values,
                     ds['B11'].values, ds['B12'].values], 0)[:,idx,...])
    angles = torch.from_numpy(np.stack([ds['sunZenithAngles'].values, ds['viewZenithMean'].values, 
                                        ds['sunAzimuthAngles'].values - ds['viewAzimuthMean'].values], 0)[:,idx,...])
    return s2_r, angles

def get_frm4veg_material(frm4veg_data_dir, frm4veg_filename):
    """
    s2_r, s2_a, site_idx_dict, ref_dict = get_frm4veg_material(frm4veg_data_dir, frm4veg_filename)
    """
    site_idx_dict = {}
    ref_dict = {}
    for variable in ['lai', 'lai_eff', 'ccc', 'ccc_eff']:
        gdf, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_filename, variable=variable)
        # gdf = gdf.iloc[:51]
        ref_dict[variable] = gdf[variable].values.reshape(-1)
        ref_dict[variable+"_std"] = gdf["uncertainty"].values.reshape(-1)
        site_idx_dict[variable] = {"x_idx" : torch.from_numpy(gdf["x_idx"].values).int(),
                                   "y_idx" : torch.from_numpy(gdf["y_idx"].values).int()}
    _, s2_r, s2_a, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_filename, variable="lai")
    s2_r = torch.from_numpy(s2_r).float().unsqueeze(0)
    s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
    return s2_r, s2_a, site_idx_dict, ref_dict

def get_model_frm4veg_results(model, s2_r, s2_a, site_idx_dict, ref_dict, mode="lat_mode", 
                              get_reconstruction=False):
    with torch.no_grad():
        (rec, sim_image, cropped_s2_r, cropped_s2_a,
            sigma_image) = get_encoded_image_from_batch((s2_r, s2_a), model,
                                            patch_size=32, bands=model.encoder.bands,
                                            mode=mode, padding=True, no_rec=not get_reconstruction)
        cropped_s2_r = cropped_s2_r[:,model.encoder.bands.to(cropped_s2_r.device),...]
        rec_err = (rec - cropped_s2_r.squeeze(0)).abs().mean(0, keepdim=True)
        band_rec_err = (rec - cropped_s2_r.squeeze(0)).abs()
    model_pred = {"s2_r":cropped_s2_r, "s2_a":cropped_s2_a}

    for lai_variable in ['lai', 'lai_eff']: # 'ccc', 'ccc_eff']:
        model_pred[lai_variable] = sim_image[6, site_idx_dict[lai_variable]['y_idx'], 
                                                site_idx_dict[lai_variable]['x_idx']].numpy()
        model_pred[f"{lai_variable}_std"] = sigma_image[6, site_idx_dict[lai_variable]['y_idx'], 
                                                           site_idx_dict[lai_variable]['x_idx']].numpy()
        model_pred[f"ref_{lai_variable}"] = ref_dict[lai_variable]
        model_pred[f"ref_{lai_variable}_std"] = ref_dict[f"{lai_variable}_std"]

        model_pred[f"{lai_variable}_rec_err"] = rec_err[..., site_idx_dict[lai_variable]['y_idx'], 
                                                             site_idx_dict[lai_variable]['x_idx']].numpy()
        for i, band in enumerate(np.array(BANDS)[model.encoder.bands.detach().cpu()].tolist()):
            model_pred[f"{lai_variable}_{band}_rec_err"] = band_rec_err[i, site_idx_dict[lai_variable]['y_idx'], 
                                                                             site_idx_dict[lai_variable]['x_idx']].numpy()
    for ccc_variable in ['ccc', 'ccc_eff']:
        model_pred[ccc_variable] = (sim_image[1, site_idx_dict[ccc_variable]['y_idx'], 
                                                 site_idx_dict[ccc_variable]['x_idx']] 
                                    * sim_image[6, site_idx_dict[ccc_variable]['y_idx'], 
                                                   site_idx_dict[ccc_variable]['x_idx']]).numpy()
        m_1 = sim_image[1, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']]
        m_2 = sim_image[6, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']]
        v_1 = sigma_image[1, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']].pow(2)
        v_2 = sigma_image[6, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']].pow(2)
        model_pred[f"{ccc_variable}_std"] = var_of_product(v_1, v_2, m_1, m_2).sqrt().numpy()
        model_pred[f"ref_{ccc_variable}"] = ref_dict[ccc_variable]
        model_pred[f"ref_{ccc_variable}_std"] = ref_dict[f"{ccc_variable}_std"]
        model_pred[f"{ccc_variable}_rec_err"] = rec_err[..., site_idx_dict[ccc_variable]['y_idx'], 
                                                             site_idx_dict[ccc_variable]['x_idx']].numpy()
        for i, band in enumerate(np.array(BANDS)[model.encoder.bands.detach().cpu(].tolist()):
            model_pred[f"{ccc_variable}_{band}_rec_err"] = band_rec_err[i, site_idx_dict[ccc_variable]['y_idx'], 
                                                                        site_idx_dict[ccc_variable]['x_idx']].numpy()
                                                             
    return model_pred #, rec, cropped_s2_r


def get_snap_frm4veg_results(s2_r, s2_a, site_idx_dict, ref_dict, sensor="2A"):
    (snap_lai, snap_ccc,  _) = get_weiss_biophyiscal_from_batch((s2_r, s2_a), patch_size=32, sensor=sensor)
    snap_results = {}
    for variable in ['lai', 'lai_eff']:
        snap_results[variable] = snap_lai[..., site_idx_dict[variable]['y_idx'], 
                                                site_idx_dict[variable]['x_idx']].numpy()
        snap_results[f"{variable}_std"] = np.zeros_like(snap_results[variable])
        snap_results[f"{variable}_rec_err"] = np.zeros_like(snap_results[variable])
        snap_results[f"ref_{variable}"] = ref_dict[variable]
        snap_results[f"ref_{variable}_std"] = ref_dict[f"{variable}_std"]
        for i, band in enumerate(BANDS):
            snap_results[f"{variable}_{band}_rec_err"] = np.zeros_like(snap_results[variable])
    for variable in ['ccc', 'ccc_eff']:
        snap_results[variable] = snap_ccc[..., site_idx_dict[variable]['y_idx'], 
                                                site_idx_dict[variable]['x_idx']].numpy()
        snap_results[f"{variable}_std"] = np.zeros_like(snap_results[variable])
        snap_results[f"{variable}_rec_err"] = np.zeros_like(snap_results[variable])
        snap_results[f"ref_{variable}"] = ref_dict[variable]
        snap_results[f"ref_{variable}_std"] = ref_dict[f"{variable}_std"]
        for i, band in enumerate(BANDS):
            snap_results[f"{variable}_{band}_rec_err"] = np.zeros_like(snap_results[variable])
    return snap_results

def get_frm4veg_results_at_date(model, frm4veg_data_dir, filename, 
                                is_SNAP=False, mode="sim_tg_mean", 
                                get_reconstruction=True):
    sensor = filename.split("_")[0]
    (s2_r, s2_a, site_idx_dict, ref_dict) = get_frm4veg_material(frm4veg_data_dir, filename)
    if not is_SNAP:
        validation_results = get_model_frm4veg_results(model, s2_r, s2_a, site_idx_dict, 
                                                        ref_dict, mode=mode, 
                                                        get_reconstruction=get_reconstruction)
    else:
        validation_results = get_snap_frm4veg_results(s2_r, s2_a, site_idx_dict, 
                                                             ref_dict, sensor=sensor)
    d = datetime.strptime(filename.split("_")[1], '%Y%m%d').date()
    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        gdf, _, _ , _, _ = load_frm4veg_data(frm4veg_data_dir, filename, variable=variable)
        validation_results[f"{variable}_land_cover"] = gdf["land cover"].values
        validation_results[f"{variable}_date"] = gdf["date"].apply(lambda x: (x.date() - d).days).values
    return validation_results

def interpolate_frm4veg_pred(model, frm4veg_data_dir, filename_before, filename_after=None, 
                             method="simple_interpolate", is_SNAP=False, mode="sim_tg_mean",
                             get_reconstruction=True, bands_idx=torch.arange(10)):
    validation_results_before = get_frm4veg_results_at_date(model, frm4veg_data_dir, filename_before, 
                                                            is_SNAP=is_SNAP, mode=mode,
                                                            get_reconstruction=get_reconstruction)
    d_before = datetime.strptime(filename_before.split("_")[1], '%Y%m%d').date()
    validation_results_after = get_frm4veg_results_at_date(model, frm4veg_data_dir, filename_after, 
                                                            is_SNAP=is_SNAP, mode=mode,
                                                            get_reconstruction=get_reconstruction)
    d_after = datetime.strptime(filename_after.split("_")[1], '%Y%m%d').date()
    
    model_results = {}

    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        model_results[f'ref_{variable}'] = validation_results_before[f'ref_{variable}']
        model_results[f'ref_{variable}_std'] = validation_results_before[f'ref_{variable}_std']
        gdf, _, _ , _, _ = load_frm4veg_data(frm4veg_data_dir, filename_before, variable=variable)
        # gdf = gdf.iloc[:51]
        model_results[f"{variable}_land_cover"] = gdf["land cover"].values
        dt_before = gdf["date"].apply(lambda x: (x.date() - d_before).days).values
        dt_after = gdf["date"].apply(lambda x: (x.date() - d_after).days).values
        if method=="simple_interpolate":
            model_results[variable] = simple_interpolate(validation_results_after[variable].squeeze(),
                                                         validation_results_before[variable].squeeze(),
                                                         dt_after, dt_before).squeeze()
            
            model_results[f"{variable}_rec_err"] = simple_interpolate(validation_results_after[f"{variable}_rec_err"].squeeze(),
                                                                      validation_results_before[f"{variable}_rec_err"].squeeze(),
                                                    dt_after, dt_before).squeeze()
            for band in np.array(BANDS)[bands_idx].tolist():
                model_results[f"{variable}_{band}_rec_err"] = simple_interpolate(validation_results_after[f"{variable}_{band}_rec_err"].squeeze(),
                                                                                 validation_results_before[f"{variable}_{band}_rec_err"].squeeze(),
                                                                                 dt_after, dt_before).squeeze()
            model_results[f"{variable}_std"] = simple_interpolate(validation_results_after[f"{variable}_std"].squeeze(),
                                                                    validation_results_before[f"{variable}_std"].squeeze(),
                                                                    dt_after, dt_before, is_std=True).squeeze()
            model_results[f"{variable}_date"] = (abs(dt_before) + abs(dt_after)) / 2
        elif method == "best":
            ref = validation_results_before[f"ref_{variable}"]
            err_1 = np.abs(validation_results_before[f"{variable}"] - ref)
            err_2 = np.abs(validation_results_after[f"{variable}"] - ref)
            date = np.zeros_like(ref)
            results = np.zeros_like(ref)
            results_std = np.zeros_like(ref)
            results_rec_err = np.zeros_like(ref)
            err_1_le_err_2 = (err_1 <= err_2).reshape(-1)
            results[err_1_le_err_2] = validation_results_before[f"{variable}"].reshape(-1)[err_1_le_err_2]
            results[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[variable] = results

            date[err_1_le_err_2] = abs(dt_before[err_1_le_err_2])
            date[np.logical_not(err_1_le_err_2)] = abs(dt_after[np.logical_not(err_1_le_err_2)])
            model_results[f"{variable}_date"] = date

            results_std[err_1_le_err_2] = validation_results_before[f"{variable}_std"].reshape(-1)[err_1_le_err_2]
            results_std[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}_std"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_std"] = results_std

            results_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_rec_err"].reshape(-1)[err_1_le_err_2]
            results_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_rec_err"] = results_rec_err
            
            for band in np.array(BANDS)[bands_idx].tolist():
                results_band_rec_err = np.zeros_like(ref)
                results_band_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_{band}_rec_err"].reshape(-1)[err_1_le_err_2]
                results_band_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}_{band}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
                model_results[f"{variable}_{band}_rec_err"] = results_band_rec_err
        elif method == "worst":
            ref = validation_results_before[f"ref_{variable}"]
            err_1 = np.abs(validation_results_before[f"{variable}"] - ref)
            err_2 = np.abs(validation_results_after[f"{variable}"] - ref)
            results = np.zeros_like(ref)
            results_std = np.zeros_like(ref)
            results_rec_err = np.zeros_like(ref)
            err_1_le_err_2 = (err_1 <= err_2).reshape(-1)
            date = np.zeros_like(ref)
            date[err_1_le_err_2] = abs(dt_after[err_1_le_err_2])
            date[np.logical_not(err_1_le_err_2)] = abs(dt_before[np.logical_not(err_1_le_err_2)])
            model_results[f"{variable}_date"] = date

            results[err_1_le_err_2] = validation_results_after[f"{variable}"].reshape(-1)[err_1_le_err_2]
            results[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[variable] = results

            results_std[err_1_le_err_2] = validation_results_after[f"{variable}_std"].reshape(-1)[err_1_le_err_2]
            results_std[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}_std"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_std"] = results_std

            results_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_rec_err"].reshape(-1)[err_1_le_err_2]
            results_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_rec_err"] = results_rec_err
            
            for band in np.array(BANDS[bands_idx]).tolist():
                results_band_rec_err = np.zeros_like(ref)
                results_band_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_{band}_rec_err"].reshape(-1)[err_1_le_err_2]
                results_band_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}_{band}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
                model_results[f"{variable}_{band}_rec_err"] = results_band_rec_err
        elif method == "dist_interpolate":
            raise NotImplementedError
        else:
            raise ValueError(method)
    return model_results

def main():
    if socket.gethostname()=='CELL200973':
        # args=["-f", "FRM_Veg_Wytham_20180703_V2",
        # args=["-f", "FRM_Veg_Barrax_20180605_V2",
        args=["-f", "FRM_Veg_Barrax_20210719",
            #   "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation/",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_2021_validation/",
            #   "-p", "SENTINEL2A_20180703-105938-887_L2A_T30SWJ_D_V1-8"]
            #   "-p", "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"]
            #   "-p", "SENTINEL2A_20180706-110918-241_L2A_T30UXC_C_V1-0"]
        # "-p", "SENTINEL2A_20180629-112537-824_L2A_T30UXC_C_V1-0"]
        # "-p", "SENTINEL2A_20180613-110957-425_L2A_T30SWJ_D_V1-8"]
        "-p", "SENTINEL2B_20210722-111020-007_L2A_T30SWJ_C_V3-0"]
        
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()
    # gdf, s2_r, s2_a = load_validation_data(parser.data_dir, parser.filename)
    s2_product_name = parser.product_name

    output_file_names = compute_frm4veg_data(parser.data_dir, parser.data_filename, s2_product_name,
                                             date='2018-07-06', no_angle_data=False)
    print(output_file_names)
if __name__ == "__main__":
    main()