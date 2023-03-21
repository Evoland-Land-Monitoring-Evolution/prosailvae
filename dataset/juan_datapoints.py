import pandas as pd
import numpy as np
import torch
import prosailvae
import os

def LAI_columns(site):
    if site =="france":
        LAI_columns = ['PAIeff – CE']#, 'PAIeff CE V5.1', 'PAIeff – P57']
                    #    , 'PAIeffMiller', 'PAIeff–LAI20003rings',
                    #     'PAIeff–LAI20004rings', 'PAIeff–LAI20005rings', 'PAItrue–CE',
                    #     'PAItrue–CEV5.1', 'PAItrue–P57', 'PAItrue–Miller']
    elif site in ["spain1", "spain2"]:
        LAI_columns = [ 'LicorLAI'] # [ 'LicorLAI', 'Pocket_LAI']
    elif site in ["italy1", "italy2"]:
        LAI_columns = ['LicorLAI']
    else:
        raise NotImplementedError
    return LAI_columns

def get_S2_bands(df_validation_data):
    data = torch.from_numpy(df_validation_data[['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']].values)
    return data

def convert_angles(angles):
    #TODO: convert 6 S2 "angles" into sun zenith, S2 zenith and Sun/S2 relative Azimuth (degrees)
    sun_zen = angles[:,0].unsqueeze(1)
    sun_azi = angles[:,1].unsqueeze(1)
    obs_zen = angles[:,2].unsqueeze(1)
    obs_azi = angles[:,3].unsqueeze(1)
    rel_azi = (sun_azi - obs_azi) % 360
    return torch.concat((sun_zen, obs_zen, rel_azi), axis=1)

def get_angles(df_validation_data):
    angles = torch.from_numpy(df_validation_data[['sun_zen', 'sun_az', 'join_zen', 'join_az']].values)
    angles = convert_angles(angles)
    return angles

def get_all_possible_LAIs(df_validation_data, site):
    data = torch.from_numpy(df_validation_data[LAI_columns(site)].values).mean(1).unsqueeze(1)
    return data


def clean_validation_data(df_validation_data, site, lai_min=None):
    LAIs = get_all_possible_LAIs(df_validation_data, site)
    nan_rows = torch.where(LAIs.isnan().any(1))[0].numpy().tolist()
    if len(nan_rows) > 0:
        df_validation_data.drop(nan_rows,inplace=True)
        df_validation_data.reset_index(inplace=True)
    if lai_min is not None:
        LAIs = get_all_possible_LAIs(df_validation_data, site)
        low_lai_rows = torch.where((LAIs<lai_min).any(1))[0].numpy().tolist()
        if len(low_lai_rows) > 0:
            df_validation_data.drop(low_lai_rows,inplace=True)
            df_validation_data.reset_index(inplace=True)
    s2_r = get_S2_bands(df_validation_data)
    zero_bands = torch.where(s2_r.sum(1)==0)[0].numpy().tolist()
    if len(zero_bands) > 0:
        df_validation_data.drop(zero_bands,inplace=True)
        df_validation_data.reset_index(inplace=True)
    pass

def get_time_delta(df_validation_data):
    return torch.from_numpy(df_validation_data["time_delta"].apply(lambda x : int(x.replace(" days", ""))).values)

def get_filename(site, relative_s2_time):
    if site =="france":
        tile = "T31TCJ"
        filename = f"{relative_s2_time}_france_{tile}_gai.csv"
    elif site =="spain1":
        tile = "T30TUM"
        filename = f"{relative_s2_time}_spain_{tile}_2017.csv"
    elif site =="spain2":
        tile = "T30TUM"
        filename = f"{relative_s2_time}_spain_{tile}_2018.csv"
    elif site =="italy1":
        tile = "T33TWF"
        filename = f"{relative_s2_time}_italy_{tile}_.csv"
    elif site =="italy2":
        tile = "T33TWG"
        filename = f"{relative_s2_time}_italy_{tile}_.csv"
    else:
        raise NotImplementedError
    return filename


def get_interpolated_validation_data(site, path_to_data_dir, lai_min=0, dt_max = 15, method="closest"):
    filename_b = get_filename(site, "before")
    filename_a = get_filename(site, "after")
    path_to_file_b = path_to_data_dir + filename_b
    path_to_file_a = path_to_data_dir + filename_a
    assert os.path.isfile(path_to_file_b)
    assert os.path.isfile(path_to_file_a)
    df_validation_data_b = pd.read_csv(path_to_file_b)
    df_validation_data_a = pd.read_csv(path_to_file_a)
    clean_validation_data(df_validation_data_a, site, lai_min=lai_min)
    clean_validation_data(df_validation_data_b, site, lai_min=lai_min)
    if len(df_validation_data_b)==0 or len(df_validation_data_a)==0:
        return None, None, None, None
    dt_b = get_time_delta(df_validation_data_b).unsqueeze(1)
    dt_a = get_time_delta(df_validation_data_a).unsqueeze(1)
    out_of_date_range_samples = np.logical_not(np.logical_or(np.abs(dt_b.numpy()) <= dt_max, np.abs(dt_a.numpy()) <= dt_max).astype(bool))
    df_validation_data_b.drop(np.where(out_of_date_range_samples.reshape(-1))[0],inplace=True)
    df_validation_data_b.reset_index(inplace=True)
    df_validation_data_a.drop(np.where(out_of_date_range_samples.reshape(-1))[0],inplace=True)
    df_validation_data_a.reset_index(inplace=True)
    dt_b = get_time_delta(df_validation_data_b).unsqueeze(1)
    dt_a = get_time_delta(df_validation_data_a).unsqueeze(1)
    lais = get_all_possible_LAIs(df_validation_data_b, site=site).float()
    s2_r_b = get_S2_bands(df_validation_data_b).float()
    s2_a_b = get_angles(df_validation_data_b).float()
    s2_r_a = get_S2_bands(df_validation_data_a).float()
    s2_a_a = get_angles(df_validation_data_a).float()

    s2_r = torch.zeros_like(s2_r_a)
    s2_a = torch.zeros_like(s2_a_a)
    dt = torch.zeros_like(dt_a)
    # If dt > dt_max, using the other measurement
    dt_a_ge_max = (dt_a.abs() > dt_max).squeeze()
    dt_b_ge_max = (dt_b.abs() > dt_max).squeeze()
    s2_r[dt_b_ge_max,:] = s2_r_a[dt_b_ge_max]
    s2_r[dt_a_ge_max,:] = s2_r_b[dt_a_ge_max]
    s2_a[dt_b_ge_max,:] = s2_a_a[dt_b_ge_max]
    s2_a[dt_a_ge_max,:] = s2_a_b[dt_a_ge_max]
    dt[dt_b_ge_max] = dt_a[dt_b_ge_max]
    dt[dt_a_ge_max] = dt_b[dt_a_ge_max]
    dt_a[dt_a_ge_max] = dt_b[dt_a_ge_max]
    dt_b[dt_b_ge_max] = dt_a[dt_b_ge_max]

    if method =="closest":
        # Using reflectances and angles of the measurement with smallest absolute dt
        dt_a_le_b = (dt_b.abs() <= dt_a.abs()).squeeze()
        dt_b_le_a = (dt_a.abs() <= dt_b.abs()).squeeze()
        s2_r[dt_b_le_a,:] = s2_r_b[dt_b_le_a]
        s2_r[dt_a_le_b,:] = s2_r_a[dt_a_le_b]
        s2_a[dt_b_le_a,:] = s2_a_b[dt_b_le_a]
        s2_a[dt_a_le_b,:] = s2_a_a[dt_a_le_b]
        dt[dt_b_le_a] = dt_b[dt_b_le_a]
        dt[dt_a_le_b] = dt_a[dt_a_le_b]

    elif method == "linear":

        # Interpolating remaining reflectances and angles
        idx_dt_le_max = torch.logical_and(dt_b.abs() <= dt_max, dt_b.abs() <= dt_max).squeeze()
        s2_r[idx_dt_le_max,:] = s2_r_a[idx_dt_le_max] - (s2_r_a[idx_dt_le_max] - s2_r_b[idx_dt_le_max]) / (dt_a[idx_dt_le_max] - dt_b[idx_dt_le_max]) * dt_a[idx_dt_le_max]
        s2_a[idx_dt_le_max,:] = s2_a_a[idx_dt_le_max] - (s2_a_a[idx_dt_le_max] - s2_a_b[idx_dt_le_max]) / (dt_a[idx_dt_le_max] - dt_b[idx_dt_le_max]) * dt_a[idx_dt_le_max]
        dt[dt_b.abs() <= dt_a.abs()] = dt_b[dt_b.abs() <= dt_a.abs()]
        dt[dt_a.abs() <= dt_b.abs()] = dt_a[dt_a.abs() <= dt_b.abs()]

    else: 
        raise NotImplementedError


    return s2_r, s2_a, lais, dt


def main():
    lai_min=0
    PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/processed/"
    site = "france"
    dt_max=10
    s2_r, s2_a, lais, dt = get_interpolated_validation_data(site, PATH_TO_DATA_DIR, lai_min=lai_min, dt_max=dt_max, method="closest")
    return

if __name__=="__main__":
    main()
