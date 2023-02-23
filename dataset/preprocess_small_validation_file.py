import pandas as pd
import numpy as np
import torch
import prosailvae
if __name__ == "__main__":
    from loaders import convert_angles
else:
    from dataset.loaders import convert_angles
import os 

PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/processed_2/"
def LAI_columns(site):
    if site =="france":
        LAI_columns = ['PAIeff–CE', 'PAIeffCEV5.1','PAIeff–P57']
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

def get_position(df_validation_data, site):
    if site == 'france':
        return torch.from_numpy(df_validation_data[['X-WGS84', 'Y-WGS84']].values)
    elif site in ["italy1", "italy2"]:
        return torch.from_numpy(df_validation_data[['Latitude', 'Longitude']].values)
    elif site in ["spain1", "spain2"]:
        return torch.from_numpy(df_validation_data[['X_UTM', 'Y_UTM']].values)
    else:
        raise NotImplementedError
def get_date(df_validation_data):
    return torch.from_numpy(df_validation_data["field_date"].str.replace("-","").values.astype(int))

def count_unique_measures(positions, dates, lais):
    apositions = positions[1]
    adates = dates[1]
    bpositions = positions[0]
    bdates = dates[0]
    blais = lais[0]
    alais = lais[1] 
    number_of_unique_measures_before = len(torch.unique(bpositions.sum(1)+bdates.squeeze()+blais.sum(1).squeeze()))
    print(f"unique measures before : {number_of_unique_measures_before}")
    number_of_unique_measures_after = len(torch.unique(apositions.sum(1)+adates.squeeze()+alais.sum(1).squeeze()))
    print(f"unique measures after: {number_of_unique_measures_after}")
    # b_txy = torch.hstack((bpositions,bdates))
    # a_txy = torch.hstack((apositions,adates))
    # common_b_txy = ((b_txy[:, None, :] == b_txy[None, ...]).all(dim=2)).nonzero()
    # common_a_txy = ((a_txy[:, None, :] == a_txy[None, ...]).all(dim=2)).nonzero()
    # common_txy = ((b_txy[:, None, :] == a_txy[None, ...]).all(dim=2)).nonzero()
    return

def filter_positions(s2_r,s2_a,lais,time_delta,positions, dates):
    bs2_r = s2_r[0]
    bs2_a = s2_a[0]
    blais = lais[0]
    btime_delta = time_delta[0]
    bpositions = positions[0]
    bdates = dates[0]

    as2_r = s2_r[1]
    as2_a = s2_a[1]
    alais = lais[1]
    atime_delta = time_delta[1]
    apositions = positions[1]
    adates = dates[1]
    b_txy = torch.hstack((bpositions,bdates))
    a_txy = torch.hstack((apositions,adates))
    common_txy = ((b_txy[:, None, :] == a_txy[None, ...]).all(dim=2)).nonzero()
    unique_bidxs = torch.unique(common_txy[:,0])
    corrected_s2_r = [] 
    corrected_s2_a = [] 
    corrected_lais = [] 
    corrected_time_delta = [] 
    corrected_positions = [] 
    corrected_dates = [] 
    corrected_s2_r = [] 
    corrected_s2_a = [] 
    corrected_lais = [] 
    corrected_time_delta = [] 
    corrected_positions = [] 
    corrected_dates = [] 
    for i in range(len(unique_bidxs)):
        idx_i = unique_bidxs[i]
        common_site_idxs = torch.where(common_txy[:,0]==idx_i)[0]
        closest_site_idx = common_site_idxs[0]
        smallest_time_delta = btime_delta[common_txy[closest_site_idx,0]].abs()
        if len(common_site_idxs)>1:
            for j in range(1,len(common_site_idxs)):
                if btime_delta[common_txy[common_site_idxs[j],0]].abs() < smallest_time_delta:
                    closest_site_idx = common_site_idxs[j]
                    smallest_time_delta = btime_delta[common_txy[closest_site_idx,0]]

        corrected_s2_r.append(bs2_r[common_txy[closest_site_idx,0],:])
        corrected_s2_r.append(as2_r[common_txy[closest_site_idx,1],:])
        corrected_s2_a.append(bs2_a[common_txy[closest_site_idx,0],:])
        corrected_s2_a.append(as2_a[common_txy[closest_site_idx,1],:])
        corrected_lais.append(blais[common_txy[closest_site_idx,0],:])
        corrected_lais.append(alais[common_txy[closest_site_idx,1],:])
        corrected_time_delta.append(btime_delta[common_txy[closest_site_idx,0],:])
        corrected_time_delta.append(atime_delta[common_txy[closest_site_idx,1],:])
        corrected_positions.append(bpositions[common_txy[closest_site_idx,0],:])
        corrected_positions.append(apositions[common_txy[closest_site_idx,1],:])
        corrected_dates.append(bdates[common_txy[closest_site_idx,0],:])
        corrected_dates.append(adates[common_txy[closest_site_idx,1],:])

    return s2_r,s2_a,lais,time_delta,positions

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

def get_small_validation_data(relative_s2_time='before', site="france", filter_if_available_positions=False, lai_min=1.5):
    # relative_s2_time ='before' # "after"
    # site = "france" # "spain", "italy"
    # if site =="france":
    #     tile = "T31TCJ"
    # elif site =="spain":
    #     tile = "T30TUM"
    # elif site =="italy":
    #     tile = "T33TWF"
    # else:
    #     raise NotImplementedError
    
    if relative_s2_time != "both":
        # filename = f"{relative_s2_time}_{site}_{tile}.csv"
        filename = get_filename(site, relative_s2_time)
        path_to_file = PATH_TO_DATA_DIR + filename
        assert os.path.isfile(path_to_file)
        df_validation_data = pd.read_csv(path_to_file)
        clean_validation_data(df_validation_data, site)
        s2_r = get_S2_bands(df_validation_data).float()
        s2_a = get_angles(df_validation_data).float()
        lais = get_all_possible_LAIs(df_validation_data, site=site).float()
        time_delta = get_time_delta(df_validation_data)
    else:
        s2_r = []
        s2_a = []
        lais = []
        time_delta = []
        positions = []
        dates = []
        for relative_s2_time in ["before", "after"]:
            filename = get_filename(site, relative_s2_time)
            path_to_file = PATH_TO_DATA_DIR + filename
            assert os.path.isfile(path_to_file)
            df_validation_data = pd.read_csv(path_to_file)
            clean_validation_data(df_validation_data, site, lai_min=lai_min)
            s2_r.append(get_S2_bands(df_validation_data).float())
            s2_a.append(get_angles(df_validation_data).float())
            lais.append(get_all_possible_LAIs(df_validation_data, site=site).float())
            time_delta.append(get_time_delta(df_validation_data).unsqueeze(1))
            positions.append(get_position(df_validation_data, site))
            dates.append(get_date(df_validation_data).unsqueeze(1))
        count_unique_measures(positions, dates, lais)
        # if filter_if_available_positions:
        #     s2_r,s2_a,lais,time_delta,positions = filter_positions(s2_r,s2_a,lais,time_delta,positions, dates)
        s2_r = torch.vstack(s2_r)
        s2_a = torch.vstack(s2_a)
        lais = torch.vstack(lais)
        time_delta = torch.vstack(time_delta)
        positions = torch.vstack(positions)
    return s2_r, s2_a, lais, time_delta 


