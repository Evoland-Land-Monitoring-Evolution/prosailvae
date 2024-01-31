import argparse
import os

import geopandas as gpd
import numpy as np
import torch
from rasterio.coords import BoundingBox
from sensorsio import sentinel2
from sensorsio.utils import rgb_render
from shapely import Point


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description="Parser for data generation")

    parser.add_argument(
        "-t", dest="tile", help="tile to retrieve images from", type=str, default=""
    )
    parser.add_argument(
        "-o", dest="output_dir", help="directory to put files in", type=str, default=""
    )
    parser.add_argument(
        "-m",
        dest="mask_max_percentage",
        help="maximum invalid pixels in retrieved images",
        type=float,
        default=0.05,
    )
    return parser


MONTHS_TO_RETRIEVE = [
    "2016-02-01",
    "2016-06-01",
    "2016-10-01",
    "2017-03-01",
    "2017-07-01",
    "2017-11-01",
    "2018-04-01",
    "2018-08-01",
    "2018-10-01",
    "2019-01-01",
    "2019-05-01",
    "2019-09-01",
]

TILES_BB = {
    "32ULV": {"bb_left_top": [[717249, 6273008]], "crs": "epsg:32632"},
    "31UFS": {"bb_left_top": [[527850, 6586729]], "crs": "epsg:32631"},
    "31UDP": {"bb_left_top": [[337026, 6198482]], "crs": "epsg:32631"},
    "30UWU": {"bb_left_top": [[-244922, 6127419]], "crs": "epsg:32630"},
    "30TXQ": {"bb_left_top": [[-116496, 5627644]], "crs": "epsg:32630"},  # EPSG3857
    "31TFJ": {"bb_left_top": [[503783, 5435881]], "crs": "epsg:32631"},
    "33TWF": {"bb_left_top": [[1736870, 4978840]], "crs": "epsg:32633"},
    "32TPQ": {"bb_left_top": [[1245580, 5625487]], "crs": "epsg:32632"},
    "30TUM": {"bb_left_top": [[-537544, 5184355]], "crs": "epsg:32630"},
    "30SVJ": {"bb_left_top": [[-382509, 4740869]], "crs": "epsg:32630"},
    "30SVG": {"bb_left_top": [[-427431, 4456797]], "crs": "epsg:32630"},
    "30STE": {"bb_left_top": [[-660596, 4232187]], "crs": "epsg:32630"},
    "33SVB": {"bb_left_top": [[1576370, 4500670]], "crs": "epsg:32633"},
    "31UCS": {"bb_left_top": [[74351, 6662326]], "crs": "epsg:32631"},
}


def get_bb_from_left_top(left, top, size=5120):
    bottom = top - size
    right = left + size
    return BoundingBox(left, bottom, right, top)


def get_polygon_from_left_top(left, top, size=5120):
    bottom = top - size
    right = left + size
    y_list = [top, bottom, bottom, top, top]
    x_list = [left, left, right, right, left]
    return Polygon(zip(x_list, y_list))


def get_sites_polygons(tiles_bb, tiles=None, in_crs="epsg:3857", size=5120):
    polygons_gpd_list = []
    if tiles is None:
        tiles = tiles_bb.keys()
    for tile in tiles:
        tiles.append(tile)
        left_top = gpd.GeoDataFrame(
            data={
                "name": [tile],
                "geometry": [
                    Point(
                        tiles_bb[tile]["bb_left_top"][0][0],
                        tiles_bb[tile]["bb_left_top"][0][1],
                    )
                ],
            },
            crs=in_crs,
            geometry="geometry",
        ).to_crs(tiles_bb[tile]["crs"])
        polygons = get_polygon_from_left_top(
            left_top.geometry.values.x, left_top.geometry.values.y, size=size
        )
        polygons_gpd_list.append(
            gpd.GeoDataFrame(
                data={"name": [tile], "geometry": [polygons]},
                crs=tiles_bb[tile]["crs"],
                geometry="geometry",
            )
        )
    return tiles, polygons_gpd_list


def get_sites_bb(tiles_bb, tiles=None, in_crs="epsg:3857", size=5120):
    bb_list = []
    tiles_list = []
    if tiles is None:
        tiles = tiles_bb.keys()
    for tile in tiles:
        tiles_list.append(tile)
        left_top = gpd.GeoDataFrame(
            data={
                "name": [tile],
                "geometry": [
                    Point(
                        tiles_bb[tile]["bb_left_top"][0][0],
                        tiles_bb[tile]["bb_left_top"][0][1],
                    )
                ],
            },
            crs=in_crs,
            geometry="geometry",
        ).to_crs(tiles_bb[tile]["crs"])
        bb = get_bb_from_left_top(
            left_top.geometry.values.x, left_top.geometry.values.y, size=size
        )
        bb_list.append(bb)
    return tiles_list, bb_list


def write_s3_id_file(s3_id_file_path, s3_id):
    df = pd.DataFrame(data={"s3_id": [s3_id]})
    df.to_csv(s3_id_file_path, mode="a", index=False, header=False)


def get_checked_s3(s3_id_file_path):
    if os.path.isfile(s3_id_file_path):
        df = pd.read_csv(s3_id_file_path, header=None)
        return df.values.reshape(-1).tolist()
    return []


def get_s3_id(
    tile,
    bb: BoundingBox,
    date,
    max_date=None,
    orbit=None,
    max_percentage=0.05,
    max_trials=5,
    delay=1,
    invalid_s3_id_file_path="",
    valid_s3_id_file_path="",
    unchecked_s3_id_file_path="",
):
    # if os.path.isfile(os.path.join(ROOT, ".s3_auth")):
    #    os.remove(os.path.join(ROOT, ".s3_auth"))
    s3utils.s3_enroll()
    sentinel2_index = s3utils.load_database(
        s3utils.SupportedMuscateCollections.SENTINEL2
    )
    s3_resource = s3utils.get_s3_resource()
    s3_context = storage.S3Context(resource=s3_resource, bucket="muscate")
    element = datetime.datetime.strptime(date, "%Y-%m-%d")
    tile_df = sentinel2_index[sentinel2_index["mgrs_tile"] == tile]
    df_tile_at_date = tile_df[pd.to_datetime(tile_df["acquisition_date"]) >= element]
    if max_date is not None:
        element = datetime.datetime.strptime(max_date, "%Y-%m-%d")
        df_tile_at_date = df_tile_at_date[
            pd.to_datetime(df_tile_at_date["acquisition_date"]) < element
        ]
    print(pd.unique(df_tile_at_date["acquisition_date"]))
    for s3_id in df_tile_at_date["s3_id"].values:
        if len(invalid_s3_id_file_path):
            invalid_s3 = get_checked_s3(invalid_s3_id_file_path)
            if s3_id in invalid_s3:
                print(
                    f"s3_id has already been checked as invalid, skipping it: {s3_id}"
                )
                continue
        print(f"Attempting to open zip : {s3_id}")
        trials = 0
        while trials < max_trials:
            try:
                s3_resource = s3utils.get_s3_resource()
                # Build s3 context for sensorsio
                s3_context = storage.S3Context(resource=s3_resource, bucket="muscate")
                ds = sentinel2.Sentinel2(s3_id, s3_context=s3_context)
                ALL_BANDS = [
                    ds.B2,
                    ds.B3,
                    ds.B4,
                    ds.B5,
                    ds.B6,
                    ds.B7,
                    ds.B8,
                    ds.B8A,
                    ds.B11,
                    ds.B12,
                ]
                (
                    np_arr,
                    np_arr_msk,
                    np_arr_atm,
                    xcoords,
                    ycoords,
                    out_crs,
                ) = ds.read_as_numpy(bands=ALL_BANDS, bounds=bb, band_type=ds.SRE)
                if check_mask(np_arr_msk, max_percentage=max_percentage):
                    if len(valid_s3_id_file_path):
                        write_s3_id_file(valid_s3_id_file_path, s3_id)
                    return s3_id
                else:
                    if len(invalid_s3_id_file_path):
                        write_s3_id_file(invalid_s3_id_file_path, s3_id)
                break
            except Exception as exc:
                if os.path.isfile(os.path.join(ROOT, ".s3_auth")):
                    os.remove(os.path.join(ROOT, ".s3_auth"))
                    s3utils.s3_enroll()
                print(trials, exc)
                trials += 1
                time.sleep(delay)
                if trials == max_trials - 1:
                    print(exc)
                    if len(unchecked_s3_id_file_path):
                        write_s3_id_file(unchecked_s3_id_file_path, s3_id)
                    # raise RuntimeError
    return None


def check_mask(mask, max_percentage=0.05):
    mask_sum = mask.sum(0).astype(bool).astype(int)
    max_unvalid_pixels = max_percentage * mask.shape[1] * mask.shape[2]
    print(
        f"Unvalid_pixels on ROI : {mask_sum.sum()} / {mask.shape[1] * mask.shape[2]} ({mask_sum.sum() / (mask.shape[1] * mask.shape[2]) * 100} %)"
    )
    if mask_sum.sum() >= max_unvalid_pixels:
        return False
    return True


def download_s3_id(s3_id, output_dir, max_trials=10, delay=1):
    trials = 0
    while trials < max_trials:
        try:
            s3utils.s3_enroll()
            client = s3utils.get_s3_client()
            # Retrieved from csv
            s3utils.s3_download(
                s3_client=client,
                s3_bucket="muscate",
                local_folder=output_dir,
                s3_object_id=s3_id,
                unzip=True,
                show_progress=True,
            )
            print(f"Downloaded: {s3_id}")
            break
        except Exception as exc:
            time.sleep(1)
            if os.path.isfile(os.path.join(ROOT, ".s3_auth")):
                os.remove(os.path.join(ROOT, ".s3_auth"))
            s3utils.s3_enroll()
            print(trials, exc)
            trials += 1
            time.sleep(delay)
            if trials == max_trials - 1:
                print(exc)

                print(f"Not Downloaded: {s3_id}")


def fix_product_name(path, product_name):
    all_dir = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]
    if product_name in all_dir:
        return product_name
    if product_name[-6] == "C":
        if product_name.replace("_C_", "_D_") in all_dir:
            return product_name.replace("_C_", "_D_")
        else:
            raise ValueError
    if product_name[-6] == "D":
        if product_name.replace("_D_", "_C_") in all_dir:
            return product_name.replace("_D_", "_C_")
        else:
            raise ValueError
    raise ValueError


def theia_product_to_tensor(
    data_dir,
    s2_product_name,
    part_loading=1,
    top_left=None,
    n_pixels=None,
    top_left_crs="epsg:3857",
):
    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    print(path_to_theia_product)
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    bands = [
        sentinel2.Sentinel2.B2,
        sentinel2.Sentinel2.B3,
        sentinel2.Sentinel2.B4,
        sentinel2.Sentinel2.B8,
        sentinel2.Sentinel2.B5,
        sentinel2.Sentinel2.B6,
        sentinel2.Sentinel2.B7,
        sentinel2.Sentinel2.B8A,
        sentinel2.Sentinel2.B11,
        sentinel2.Sentinel2.B12,
    ]
    even_zen, odd_zen, even_az, odd_az = dataset.read_incidence_angles_as_numpy()
    joint_zen = np.array(even_zen)
    joint_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
    del even_zen
    del odd_zen
    joint_az = np.array(even_az)
    joint_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]
    del even_az
    del odd_az
    sun_zen, sun_az = dataset.read_solar_angles_as_numpy()
    # s2_a = np.stack((sun_zen, joint_zen, sun_az - joint_az), 0)
    # print(s2_a.shape)
    if top_left is not None:
        left_top_df = gpd.GeoDataFrame(
            data={"geometry": [Point(top_left[0], top_left[1])]},
            crs=top_left_crs,
            geometry="geometry",
        ).to_crs(dataset.crs)
        res = 10
        top = left_top_df.geometry.values.y // res * res
        left = left_top_df.geometry.values.x // res * res

        left_idx = int((left - dataset.bounds.left) // res)
        top_idx = int((dataset.bounds.top - top) // res)
        # print(top_idx, n_pixels, left_idx)
        # print( s2_a[:, top_idx:(top_idx+n_pixels),:].shape)
        # print( s2_a[:, :, left_idx:(left_idx+n_pixels)].shape)
        joint_zen = joint_zen[
            ..., top_idx : (top_idx + n_pixels), left_idx : (left_idx + n_pixels)
        ]
        joint_az = joint_az[
            ..., top_idx : (top_idx + n_pixels), left_idx : (left_idx + n_pixels)
        ]
        sun_zen = sun_zen[
            ..., top_idx : (top_idx + n_pixels), left_idx : (left_idx + n_pixels)
        ]
        sun_az = sun_az[
            ..., top_idx : (top_idx + n_pixels), left_idx : (left_idx + n_pixels)
        ]

        bounding_box = BoundingBox(
            left, top - n_pixels * res, left + n_pixels * res, top
        )
        assert (
            bounding_box.left < dataset.bounds.right
            and bounding_box.left > dataset.bounds.left
        )

        s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(
            bands, bounds=bounding_box, crs=dataset.crs, band_type=dataset.SRE
        )
        s2_r = s2_r.data
        masks = masks.data
    else:
        bb = dataset.bounds
        if part_loading > 1:
            s2_r_list = []
            masks_list = []
            top_bottom_range = (
                dataset.bounds.top - dataset.bounds.bottom
            ) // part_loading
            for i in range(part_loading - 1):
                bb = BoundingBox(
                    dataset.bounds.left,
                    dataset.bounds.bottom + i * top_bottom_range,
                    dataset.bounds.right,
                    dataset.bounds.bottom + (i + 1) * top_bottom_range,
                )
                try:
                    s2_r, masks, _, _, _, _ = dataset.read_as_numpy(
                        bands, bounds=bb, crs=dataset.crs, band_type=dataset.SRE
                    )
                    print(i)
                except Exception as exc:
                    print(i, bb, top_bottom_range)
                s2_r_list.append(s2_r.data)
                masks_list.append(masks.data)
            bb = BoundingBox(
                dataset.bounds.left,
                dataset.bounds.bottom + (part_loading - 1) * top_bottom_range,
                dataset.bounds.right,
                dataset.bounds.top,
            )
            s2_r, masks, _, _, _, _ = dataset.read_as_numpy(
                bands, bounds=bb, crs=dataset.crs, band_type=dataset.SRE
            )
            s2_r_list.append(s2_r.data)
            masks_list.append(masks.data)
            s2_r = np.concatenate(s2_r_list, 1)
            masks = np.concatenate(masks_list, 1)
        else:
            s2_r, masks, _, _, _, _ = dataset.read_as_numpy(
                bands, bounds=bb, crs=dataset.crs, band_type=dataset.SRE
            )
            s2_r = s2_r.data
            masks = masks.data
    w = s2_r.shape[1]
    h = s2_r.shape[2]
    validity_mask = (
        np.sum(masks, axis=0, keepdims=True).astype(bool).astype(int).astype(float)
    )
    tile_tensor = np.concatenate(
        (
            s2_r,
            validity_mask,
            sun_zen.reshape((1, w, h)),
            sun_az.reshape((1, w, h)),
            joint_zen.reshape((1, w, h)),
            joint_az.reshape((1, w, h)),
        )
    )
    print("Tile Tensor completed")
    return torch.from_numpy(tile_tensor)


def main():
    top_left = [1576370, 4500670]
    data_dir = "/home/yoel/Téléchargements"
    s2_product_name = "SENTINEL2A_20190915-100018-897_L2A_T33SVB_C_V2-2"
    tile_array = theia_product_to_tensor(
        data_dir, s2_product_name, part_loading=1, top_left=top_left, n_pixels=512
    )
    pass


if __name__ == "__main__":
    main()
