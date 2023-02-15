import pandas as pd
import numpy as np
import torch
import prosailvae
if __name__ == "__main__":
    from loaders import convert_angles
else:
    from dataset.loaders import convert_angles
import os 

PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/processed/"

def get_S2_bands(df_validation_data):
    data = torch.from_numpy(df_validation_data[['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']].values)
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

def get_all_possible_LAIs(df_validation_data):
    data = torch.from_numpy(df_validation_data[['PAIeff–CE', 'PAIeffCEV5.1',
       'PAIeff–P57', 'PAIeffMiller', 'PAIeff–LAI20003rings',
       'PAIeff–LAI20004rings', 'PAIeff–LAI20005rings', 'PAItrue–CE',
       'PAItrue–CEV5.1', 'PAItrue–P57', 'PAItrue–Miller']].values)
    return data

def clean_validation_data(df_validation_data):
    LAIs = get_all_possible_LAIs(df_validation_data)
    nan_rows = torch.where(LAIs.isnan().any(1))[0].numpy().tolist()
    df_validation_data.drop(nan_rows,inplace=True)
    df_validation_data.reset_index(inplace=True)
    pass


def get_small_validation_data(relative_s2_time ='before', site = "france"):
    # relative_s2_time ='before' # "after"
    # site = "france" # "spain", "italy"
    if site =="france":
        tile = "T31TCJ"
    else:
        raise NotImplementedError
        if site =="spain":
            tile = "T30TUM"
        elif site =="italy":
            tile = "T33TWF"
        else:
            raise NotImplementedError
    filename = f"{relative_s2_time}_{site}_{tile}.csv"
    path_to_file = PATH_TO_DATA_DIR + filename
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file)
    clean_validation_data(df_validation_data)
    s2_r = get_S2_bands(df_validation_data).float()
    s2_a = get_angles(df_validation_data).float()
    lais = get_all_possible_LAIs(df_validation_data).float()

    return s2_r, s2_a, lais


