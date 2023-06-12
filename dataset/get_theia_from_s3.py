from datalakeutils import s3utils
from sensorsio import sentinel2, storage
# from sensorsio import sentinel2
import os
import geopandas as gpd
from shapely.geometry import Polygon, Point
from rasterio.coords import BoundingBox
ROOT = "/home/uz/zerahy/"
import argparse
import socket
import pandas as pd
import datetime

def get_prosailvae_train_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-t", dest="tile",
                        help="tile to retrieve images from",
                        type=str, default="")
    parser.add_argument("-o", dest="output_dir",
                        help="directory to put files in",
                        type=str, default="")
    parser.add_argument("-m", dest="mask_max_percentage",
                        help="maximum invalid pixels in retrieved images",
                        type=float, default=0.05)
    return parser

MONTHS_TO_RETRIEVE = ["2016-02-01",
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

TILES_BB = {"32ULV": {'bb_left_top':[[ 717249, 6273008]], 'crs':"epsg:32632"},
            "31UFS": {'bb_left_top':[[ 527850, 6586729]], 'crs':"epsg:32631"},
            "31UDP": {'bb_left_top':[[ 337026, 6198482]], 'crs':"epsg:32631"},
            "30UWU": {'bb_left_top':[[-244922, 6127419]], 'crs':"epsg:32630"},
            "30TXQ": {'bb_left_top':[[-116496, 5627644]], 'crs':"epsg:32630"}, #EPSG3857
            "31TFJ": {'bb_left_top':[[ 503783, 5435881]], 'crs':"epsg:32631"},
            "33TWF": {'bb_left_top':[[1736870, 4978840]], 'crs':"epsg:32633"},
            "32TPQ": {'bb_left_top':[[1245580, 5625487]], 'crs':"epsg:32632"},
            "30TUM": {'bb_left_top':[[-537544, 5184355]], 'crs':"epsg:32630"},
            "30SVJ": {'bb_left_top':[[-382509, 4740869]], 'crs':"epsg:32630"},
            "30SVG": {'bb_left_top':[[-427431, 4456797]], 'crs':"epsg:32630"},
            "30STE": {'bb_left_top':[[-660596, 4232187]], 'crs':"epsg:32630"},
            "33SVB": {'bb_left_top':[[1576370, 4500670]], 'crs':"epsg:32633"},
            "31UCS": {'bb_left_top':[[  74351, 6662326]], 'crs':"epsg:32631"}}

def get_bb_from_left_top(left, top, size = 5120):
    bottom = top - size
    right = left + size
    return BoundingBox(left, bottom, right, top)

def get_polygon_from_left_top(left, top, size = 5120):
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
        left_top = gpd.GeoDataFrame(data={"name":[tile],
                                          "geometry":[Point(tiles_bb[tile]['bb_left_top'][0][0], 
                                                            tiles_bb[tile]['bb_left_top'][0][1])]}, 
                                          crs=in_crs, geometry="geometry").to_crs(tiles_bb[tile]["crs"])
        polygons = get_polygon_from_left_top(left_top.geometry.values.x, left_top.geometry.values.y, size=size)
        polygons_gpd_list.append(gpd.GeoDataFrame(data={"name":[tile], "geometry":[polygons]},
                                                  crs=tiles_bb[tile]["crs"], geometry="geometry"))
    return tiles, polygons_gpd_list


def get_sites_bb(tiles_bb, tiles=None, in_crs="epsg:3857", size=5120):
    bb_list = []
    tiles_list = []
    if tiles is None:
        tiles = tiles_bb.keys()
    for tile in tiles:
        tiles_list.append(tile)
        left_top = gpd.GeoDataFrame(data={"name":[tile],
                                          "geometry":[Point(tiles_bb[tile]['bb_left_top'][0][0], 
                                                            tiles_bb[tile]['bb_left_top'][0][1])]}, 
                                          crs=in_crs, geometry="geometry").to_crs(tiles_bb[tile]["crs"])
        bb = get_bb_from_left_top(left_top.geometry.values.x, left_top.geometry.values.y, size=size)
        bb_list.append(bb)
    return tiles_list, bb_list


def get_s3_id(tile, bb:BoundingBox, date, max_date=None, orbit=None, max_percentage=0.05):
    if os.path.isfile(os.path.join(ROOT, ".s3_auth")):
        os.remove(os.path.join(ROOT, ".s3_auth"))
    s3utils.s3_enroll()
    sentinel2_index = s3utils.load_database(s3utils.SupportedMuscateCollections.SENTINEL2)
    # tile = "30TUM"
    # s3_id = 'SENTINEL2/2015/11/29/SENTINEL2A_20151129-112140-218_L2A_T30TUM_D_V1-4/SENTINEL2A_20151129-112140-218_L2A_T30TUM_D_V1-4.zip'
    s3_resource = s3utils.get_s3_resource()
    # Build s3 context for sensorsio
    s3_context = storage.S3Context(resource = s3_resource,
                                    bucket = 'muscate')
    element = datetime.datetime.strptime(date,"%Y-%m-%d")
    tile_df = sentinel2_index[sentinel2_index["mgrs_tile"]==tile]
    df_tile_at_date = tile_df[pd.to_datetime(tile_df['acquisition_date']) >= element]
    if max_date is not None:
        element = datetime.datetime.strptime(max_date,"%Y-%m-%d")
        df_tile_at_date = df_tile_at_date[pd.to_datetime(df_tile_at_date['acquisition_date']) < element]
    for s3_id in df_tile_at_date["s3_id"].values:
        print(f"Attempting to open zip : {s3_id}")
        ds = sentinel2.Sentinel2(s3_id, s3_context=s3_context)
        ALL_BANDS = [ds.B2, ds.B3,ds.B4, ds.B5, ds.B6,
                    ds.B7, ds.B8, ds.B8A, ds.B11, ds.B12,]
        np_arr, np_arr_msk, np_arr_atm, xcoords, ycoords, out_crs = ds.read_as_numpy(bands = ALL_BANDS, bounds=bb)
        if check_mask(np_arr_msk, max_percentage=max_percentage):
            return s3_id
    return None
    

def check_mask(mask, max_percentage=0.05):
    mask_sum = mask.sum(0).astype(bool).astype(int)
    max_unvalid_pixels = max_percentage * mask.shape[1] * mask.shape[2]
    if mask_sum.sum() >= max_unvalid_pixels:
        return False
    return True

def download_s3_id(s3_id, output_dir):
    s3utils.s3_enroll()
    client = s3utils.get_s3_client()
    # Retrieved from csv
    s3utils.s3_download(s3_client=client,
                        s3_bucket='muscate',
                        local_folder = output_dir,
                        s3_object_id = s3_id,
                        unzip=False,
                        show_progress=True)

    print(f"Downloaded: {s3_id}")

def main():
    if socket.gethostname()=='CELL200973':
        args=["-t", "30TUM",
              "-m", "0.01"]
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()
    # tiles, polygons_gpd_list = get_sites_polygons(TILES_BB, in_crs="epsg:3857", size=5120)
    # for i, tile in enumerate(TILES_BB.keys()):
    #     polygons_gpd_list[i].to_file(filename=f'/home/yoel/Documents/Dev/PROSAIL-VAE/used_theia_tiles_samples/sites/{tile}_sites.geojson',
    #                         driver='GeoJSON')
    tiles, bb_list = get_sites_bb(TILES_BB, tiles=[parser.tile], in_crs="epsg:3857", size=5120)
    tile = tiles[0]
    bb = bb_list[0]
    list_s3_id = []
    for i, date in enumerate(MONTHS_TO_RETRIEVE):
        max_date = None
        if i < len(MONTHS_TO_RETRIEVE) - 1:
            max_date = MONTHS_TO_RETRIEVE[i+1]
        s3_id = get_s3_id(tile, bb, date, max_date, max_percentage=parser.mask_max_percentage)
        if s3_id is None:
            print(f"Warning: No sample found for tile {tile} between {date} and {max_date}.")
        else:
            list_s3_id.append(s3_id)
            print(s3_id)
    for s3_id in list_s3_id:
        download_s3_id(s3_id, parser.output_dir)


if __name__ == "__main__":
    main()
