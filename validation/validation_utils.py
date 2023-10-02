import numpy as np
import rasterio as rio
from sensorsio import sentinel2
import socket

def var_of_product(var_1, var_2, mean_1, mean_2):
    return (var_1 + mean_1.pow(2)) * (var_2 + mean_2.pow(2)) - (mean_1 * mean_2).pow(2)


def std_interpolate(x0, std0, x1, std1, x):
    assert (x <= x1).all() and (x >= x0).all()
    u0 = (x1-x) / (x1-x0)
    u1 = (x-x0) / (x1-x0)
    return np.sqrt((u0 * std0)**2 + (u1*std1)**2)

def interpolate(x0, y0, x1, y1, x):
    assert (x <= x1).all() and (x >= x0).all()
    u0 = (x1-x) / (x1-x0)
    u1 = (x-x0) / (x1-x0)
    return u0 * y0 + u1 * y1 

def simple_interpolate(y_after, y_before, dt_after, dt_before, is_std=False):
    res = np.zeros_like(y_after).astype(float)
    res[dt_before==0] = y_before[dt_before==0]
    res[dt_after==0] = y_after[dt_after==0]
    idx = np.logical_and(dt_after!=0, dt_before!=0)
    dt = np.abs(dt_after[idx]) + np.abs(dt_before[idx])
    v = np.abs(dt_after[idx]) / dt
    u = np.abs(dt_before[idx])  / dt
    # if is_std:
    #     res[idx] = std_interpolate(-dt_before[idx], y_before[idx], -dt_after[idx], y_after[idx], np.zeros_like(dt_before[idx]))
    # else:    
    res[idx] = interpolate(-dt_before[idx], y_before[idx], -dt_after[idx], y_after[idx], np.zeros_like(dt_before[idx]))
    return res

def get_bb_array_index(bb, image_bb, res=10):
    xmin = (bb[0] - image_bb[0]) / res
    ymin = ( - (bb[3] - image_bb[3])) / res
    xmax = xmin + (bb[2] - bb[0]) / res
    ymax = ymin + (bb[3] - bb[1]) / res
    return int(xmin), int(ymin), int(xmax), int(ymax)

def read_data_from_theia(left, bottom, right, top, src_epsg, path_to_theia_product, margin=100):
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    left, bottom, right, top = rio.warp.transform_bounds(src_epsg, dataset.crs.to_epsg(), left, bottom, right, top)
    bb = rio.coords.BoundingBox(left - margin, bottom - margin, right + margin, top + margin)

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
    s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(bands, bounds=bb,
                                                                    crs=dataset.crs,
                                                                    band_type=dataset.SRE,
                                                                    read_atmos=True)
    validity_mask = np.sum(masks.data, axis=0, keepdims=True).astype(bool).astype(int).astype(float)
    s2_r = s2_r.data
    return s2_r, s2_a, validity_mask, xcoords, ycoords, crs