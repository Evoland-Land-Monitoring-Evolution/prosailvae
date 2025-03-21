import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import lognorm

import prosailvae

if __name__ == "__main__":
    from loaders import convert_angles
else:
    from dataset.loaders import convert_angles

import os

PATH_TO_DATA_DIR = (
    os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/processed/"
)


def LAI_columns(site):
    if site == "france":
        LAI_columns = ["PAIeff – CE", "PAIeff CE V5.1", "PAIeff – P57"]
        #    , 'PAIeffMiller', 'PAIeff–LAI20003rings',
        #     'PAIeff–LAI20004rings', 'PAIeff–LAI20005rings', 'PAItrue–CE',
        #     'PAItrue–CEV5.1', 'PAItrue–P57', 'PAItrue–Miller']
    elif site in ["spain1", "spain2"]:
        LAI_columns = ["LicorLAI"]  # [ 'LicorLAI', 'Pocket_LAI']
    elif site in ["italy1", "italy2"]:
        LAI_columns = ["LicorLAI"]
    else:
        raise NotImplementedError
    return LAI_columns


def get_S2_bands(df_validation_data):
    data = torch.from_numpy(
        df_validation_data[
            ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        ].values
    )
    return data


def convert_angles(angles):
    # TODO: convert 6 S2 "angles" into sun zenith, S2 zenith and Sun/S2 relative Azimuth (degrees)
    sun_zen = angles[:, 0].unsqueeze(1)
    sun_azi = angles[:, 1].unsqueeze(1)
    obs_zen = angles[:, 2].unsqueeze(1)
    obs_azi = angles[:, 3].unsqueeze(1)
    rel_azi = (sun_azi - obs_azi) % 360

    return torch.concat((sun_zen, obs_zen, rel_azi), axis=1)


def get_angles(df_validation_data):
    angles = torch.from_numpy(
        df_validation_data[["sun_zen", "sun_az", "join_zen", "join_az"]].values
    )
    angles = convert_angles(angles)
    return angles


def get_all_possible_LAIs(df_validation_data, site):
    data = (
        torch.from_numpy(df_validation_data[LAI_columns(site)].values)
        .mean(1)
        .unsqueeze(1)
    )
    return data


def clean_validation_data(df_validation_data, site, lai_min=None):
    LAIs = get_all_possible_LAIs(df_validation_data, site)
    nan_rows = torch.where(LAIs.isnan().any(1))[0].numpy().tolist()
    if len(nan_rows) > 0:
        df_validation_data.drop(nan_rows, inplace=True)
        df_validation_data.reset_index(inplace=True)
    if lai_min is not None:
        LAIs = get_all_possible_LAIs(df_validation_data, site)
        low_lai_rows = torch.where((LAIs < lai_min).any(1))[0].numpy().tolist()
        if len(low_lai_rows) > 0:
            df_validation_data.drop(low_lai_rows, inplace=True)
            df_validation_data.reset_index(inplace=True)
    s2_r = get_S2_bands(df_validation_data)
    zero_bands = torch.where(s2_r.sum(1) == 0)[0].numpy().tolist()
    if len(zero_bands) > 0:
        df_validation_data.drop(zero_bands, inplace=True)
        df_validation_data.reset_index(inplace=True)
    pass


def get_time_delta(df_validation_data):
    return torch.from_numpy(
        df_validation_data["time_delta"]
        .apply(lambda x: int(x.replace(" days", "")))
        .values
    )


def get_position(df_validation_data, site):
    if site == "france":
        return torch.from_numpy(df_validation_data[["X-WGS84", "Y-WGS84"]].values)
    elif site in ["italy1", "italy2"]:
        return torch.from_numpy(df_validation_data[["Latitude", "Longitude"]].values)
    elif site in ["spain1", "spain2"]:
        return torch.from_numpy(df_validation_data[["X_UTM", "Y_UTM"]].values)
    else:
        raise NotImplementedError


def get_date(df_validation_data):
    return torch.from_numpy(
        df_validation_data["field_date"].str.replace("-", "").values.astype(int)
    )


def count_unique_measures(positions, dates, lais):
    apositions = positions[1]
    adates = dates[1]
    bpositions = positions[0]
    bdates = dates[0]
    blais = lais[0]
    alais = lais[1]
    number_of_unique_measures_before = len(
        torch.unique(bpositions.sum(1) + bdates.squeeze() + blais.sum(1).squeeze())
    )
    print(f"unique measures before : {number_of_unique_measures_before}")
    number_of_unique_measures_after = len(
        torch.unique(apositions.sum(1) + adates.squeeze() + alais.sum(1).squeeze())
    )
    print(f"unique measures after: {number_of_unique_measures_after}")
    # b_txy = torch.hstack((bpositions,bdates))
    # a_txy = torch.hstack((apositions,adates))
    # common_b_txy = ((b_txy[:, None, :] == b_txy[None, ...]).all(dim=2)).nonzero()
    # common_a_txy = ((a_txy[:, None, :] == a_txy[None, ...]).all(dim=2)).nonzero()
    # common_txy = ((b_txy[:, None, :] == a_txy[None, ...]).all(dim=2)).nonzero()
    return


def filter_positions(s2_r, s2_a, lais, time_delta, positions, dates):
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
    b_txy = torch.hstack((bpositions, bdates))
    a_txy = torch.hstack((apositions, adates))
    common_txy = ((b_txy[:, None, :] == a_txy[None, ...]).all(dim=2)).nonzero()
    unique_bidxs = torch.unique(common_txy[:, 0])
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
        common_site_idxs = torch.where(common_txy[:, 0] == idx_i)[0]
        closest_site_idx = common_site_idxs[0]
        smallest_time_delta = btime_delta[common_txy[closest_site_idx, 0]].abs()
        if len(common_site_idxs) > 1:
            for j in range(1, len(common_site_idxs)):
                if (
                    btime_delta[common_txy[common_site_idxs[j], 0]].abs()
                    < smallest_time_delta
                ):
                    closest_site_idx = common_site_idxs[j]
                    smallest_time_delta = btime_delta[common_txy[closest_site_idx, 0]]

        corrected_s2_r.append(bs2_r[common_txy[closest_site_idx, 0], :])
        corrected_s2_r.append(as2_r[common_txy[closest_site_idx, 1], :])
        corrected_s2_a.append(bs2_a[common_txy[closest_site_idx, 0], :])
        corrected_s2_a.append(as2_a[common_txy[closest_site_idx, 1], :])
        corrected_lais.append(blais[common_txy[closest_site_idx, 0], :])
        corrected_lais.append(alais[common_txy[closest_site_idx, 1], :])
        corrected_time_delta.append(btime_delta[common_txy[closest_site_idx, 0], :])
        corrected_time_delta.append(atime_delta[common_txy[closest_site_idx, 1], :])
        corrected_positions.append(bpositions[common_txy[closest_site_idx, 0], :])
        corrected_positions.append(apositions[common_txy[closest_site_idx, 1], :])
        corrected_dates.append(bdates[common_txy[closest_site_idx, 0], :])
        corrected_dates.append(adates[common_txy[closest_site_idx, 1], :])

    return s2_r, s2_a, lais, time_delta, positions


def get_filename(site, relative_s2_time):
    if site == "france":
        tile = "T31TCJ"
        filename = f"{relative_s2_time}_france_{tile}_gai.csv"
    elif site == "spain1":
        tile = "T30TUM"
        filename = f"{relative_s2_time}_spain_{tile}_2017.csv"
    elif site == "spain2":
        tile = "T30TUM"
        filename = f"{relative_s2_time}_spain_{tile}_2018.csv"
    elif site == "italy1":
        tile = "T33TWF"
        filename = f"{relative_s2_time}_italy_{tile}_.csv"
    elif site == "italy2":
        tile = "T33TWG"
        filename = f"{relative_s2_time}_italy_{tile}_.csv"
    else:
        raise NotImplementedError
    return filename


def get_all_validation_data(lai_min=1.5):
    all_s2_r = []
    all_s2_a = []
    all_lais = []
    all_dt = []

    for site in ["france", "italy1", "italy2", "spain1", "spain2"]:
        s2_r, s2_a, lais, time_delta = get_small_validation_data(
            relative_s2_time="both",
            site=site,
            filter_if_available_positions=True,
            lai_min=lai_min,
        )
        if len(s2_r) > 0:
            all_s2_r.append(s2_r)
            all_s2_a.append(s2_a)
            all_lais.append(lais)
            all_dt.append(time_delta)
    all_s2_r = torch.vstack(all_s2_r)
    all_s2_a = torch.vstack(all_s2_a)
    all_lais = torch.vstack(all_lais)
    all_dt = torch.vstack(all_dt)
    return all_s2_r, all_s2_a, all_lais, all_dt


def get_interpolated_validation_data(
    site, path_to_data_dir, lai_min=0, dt_max=15, method="closest"
):
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
    dt_b = get_time_delta(df_validation_data_b).unsqueeze(1)
    dt_a = get_time_delta(df_validation_data_a).unsqueeze(1)
    out_of_date_range_samples = np.logical_not(
        np.logical_or(
            np.abs(dt_b.numpy()) <= dt_max, np.abs(dt_a.numpy()) <= dt_max
        ).astype(bool)
    )
    df_validation_data_b.drop(
        np.where(out_of_date_range_samples.reshape(-1))[0], inplace=True
    )
    df_validation_data_b.reset_index(inplace=True)
    df_validation_data_a.drop(
        np.where(out_of_date_range_samples.reshape(-1))[0], inplace=True
    )
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
    s2_r[dt_b_ge_max, :] = s2_r_a[dt_b_ge_max]
    s2_r[dt_a_ge_max, :] = s2_r_b[dt_a_ge_max]
    s2_a[dt_b_ge_max, :] = s2_a_a[dt_b_ge_max]
    s2_a[dt_a_ge_max, :] = s2_a_b[dt_a_ge_max]
    dt[dt_b_ge_max] = dt_a[dt_b_ge_max]
    dt[dt_a_ge_max] = dt_b[dt_a_ge_max]
    dt_a[dt_a_ge_max] = dt_b[dt_a_ge_max]
    dt_b[dt_b_ge_max] = dt_a[dt_b_ge_max]

    if method == "closest":
        # Using reflectances and angles of the measurement with smallest absolute dt
        dt_a_le_b = (dt_b.abs() <= dt_a.abs()).squeeze()
        dt_b_le_a = (dt_a.abs() <= dt_b.abs()).squeeze()
        s2_r[dt_b_le_a, :] = s2_r_b[dt_b_le_a]
        s2_r[dt_a_le_b, :] = s2_r_a[dt_a_le_b]
        s2_a[dt_b_le_a, :] = s2_a_b[dt_b_le_a]
        s2_a[dt_a_le_b, :] = s2_a_a[dt_a_le_b]
        dt[dt_b_le_a] = dt_b[dt_b_le_a]
        dt[dt_a_le_b] = dt_a[dt_a_le_b]

    elif method == "linear":
        # Interpolating remaining reflectances and angles
        idx_dt_le_max = torch.logical_and(
            dt_b.abs() <= dt_max, dt_b.abs() <= dt_max
        ).squeeze()
        s2_r[idx_dt_le_max, :] = (
            s2_r_a[idx_dt_le_max]
            - (s2_r_a[idx_dt_le_max] - s2_r_b[idx_dt_le_max])
            / (dt_a[idx_dt_le_max] - dt_b[idx_dt_le_max])
            * dt_a[idx_dt_le_max]
        )
        s2_a[idx_dt_le_max, :] = (
            s2_a_a[idx_dt_le_max]
            - (s2_a_a[idx_dt_le_max] - s2_a_b[idx_dt_le_max])
            / (dt_a[idx_dt_le_max] - dt_b[idx_dt_le_max])
            * dt_a[idx_dt_le_max]
        )
        dt[dt_b.abs() <= dt_a.abs()] = dt_b[dt_b.abs() <= dt_a.abs()]
        dt[dt_a.abs() <= dt_b.abs()] = dt_a[dt_a.abs() <= dt_b.abs()]

    else:
        raise NotImplementedError

    return s2_r, s2_a, lais, dt


def get_small_validation_data(
    relative_s2_time="before",
    site="france",
    filter_if_available_positions=False,
    lai_min=1.5,
):
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
    s2_r = []
    s2_a = []
    lais = []
    time_delta = []
    positions = []
    # dates = []
    if site == "weiss":
        path_to_file = (
            os.path.join(prosailvae.__path__[0], os.pardir)
            + "/field_data/lai/InputNoNoise_2.csv"
        )
        assert os.path.isfile(path_to_file)
        df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
        n_obs = len(df_validation_data)
        s2_r = torch.full((n_obs, 10), 0.01)
        s2_r[:, 1:6] = torch.as_tensor(
            df_validation_data[["B3", "B4", "B5", "B6", "B7"]].values
        )
        s2_r[:, 7:] = torch.as_tensor(df_validation_data[["B8A", "B11", "B12"]].values)
        s2_r[:, 6] = torch.as_tensor(df_validation_data["B8A"].values)
        s2_a = torch.zeros((n_obs, 3))
        s2_a[:, 0] = torch.as_tensor(
            np.rad2deg(np.arccos(df_validation_data["cos(thetas)"].values))
        )
        s2_a[:, 1] = torch.as_tensor(
            np.rad2deg(np.arccos(df_validation_data["cos(thetav)"].values))
        )
        s2_a[:, 2] = torch.as_tensor(
            np.rad2deg(np.arccos(df_validation_data["cos(phiv-phis)"].values))
        )
        lais = torch.as_tensor(df_validation_data["lai_true"].values.reshape(-1, 1))
        lai_bv_net = torch.as_tensor(
            df_validation_data["lai_bvnet"].values.reshape(-1, 1)
        )
        time_delta = torch.zeros((n_obs, 1))
        import matplotlib.pyplot as plt

        plt.figure(dpi=200)
        plt.scatter(lai_bv_net, lais, s=2)
        plt.xlabel("LAI BVNET")
        plt.ylabel("LAI True")
        plt.axis("equal")
        plt.plot([0, 15], [0, 15], "k--")
        return s2_r, s2_a, lais, time_delta

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
        for relative_s2_time in ["before", "after"]:
            filename = get_filename(site, relative_s2_time)
            path_to_file = PATH_TO_DATA_DIR + filename
            assert os.path.isfile(path_to_file)
            df_validation_data = pd.read_csv(path_to_file)
            clean_validation_data(df_validation_data, site, lai_min=lai_min)
            try:
                time_delta.append(get_time_delta(df_validation_data).unsqueeze(1))
            except:
                print(
                    f"Warning, with site {site}, time {relative_s2_time}, and min lai {lai_min}, no data was available."
                )
                continue
            s2_r.append(get_S2_bands(df_validation_data).float())
            s2_a.append(get_angles(df_validation_data).float())
            lais.append(get_all_possible_LAIs(df_validation_data, site=site).float())
            # positions.append(get_position(df_validation_data, site))
            # dates.append(get_date(df_validation_data).unsqueeze(1))
        # count_unique_measures(positions, dates, lais)
        # if filter_if_available_positions:
        #     s2_r,s2_a,lais,time_delta,positions = filter_positions(s2_r,s2_a,lais,time_delta,positions, dates)
        if len(s2_r) > 0:
            s2_r = torch.vstack(s2_r)
            s2_a = torch.vstack(s2_a)
            lais = torch.vstack(lais)
            time_delta = torch.vstack(time_delta)
        # positions = torch.vstack(positions)
    return s2_r, s2_a, lais, time_delta


def lognorm(x, mu=0, sigma=1):
    pdf = np.exp(-np.square(np.log(x - mu)) / (2 * sigma**2)) / (
        x * sigma * np.sqrt(2 * np.pi)
    )
    return pdf


def main():
    lai_min = 1.5
    all_s2_r, all_s2_a, all_lais, all_dt = get_all_validation_data(lai_min=lai_min)
    max_dt = 3

    fig, ax = plt.subplots(dpi=150)
    ax.hist(all_lais.squeeze(), bins=30)

    lai_filtered = all_lais.squeeze()[all_dt.squeeze() < max_dt]
    sigma2 = (torch.log(1 + lai_filtered.var() / lai_filtered.mean().square())).numpy()
    mu = torch.log(lai_filtered.mean()).numpy() - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    mu = 1.3
    sigma = 0.7
    x = np.arange(lai_min, 10, 0.01)
    pdf = lognorm(x, mu=mu, sigma=sigma)
    fig, ax = plt.subplots(dpi=150)
    # weights = np.ones_like(lai_filtered.numpy()) / len(lai_filtered)
    _, _, h = ax.hist(lai_filtered.numpy(), bins=20, density=True, label="In-situ LAI")

    ax.set_xlabel(f"LAI (max dt = {max_dt} days)")
    ax.plot(x, pdf, label="Proposed prior distribution")
    ax.legend()
    fig.savefig(
        "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/validation/all_lai_dist.png"
    )
    return


if __name__ == "__main__":
    main()
