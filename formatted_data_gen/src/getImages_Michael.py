# Extracting satellite images: https://github.com/thewati/ExtractSatelliteImagesFromCSV
# alternatives https://github.com/Jimut123/jimutmap

# Watipaso Mulwafu
# https://github.com/thewati
# https://medium.com/@watipasomulwafu
# geckodriver must be in venv/bin and firefox must be installed
import io
from datetime import datetime

import folium
from PIL import Image
import numpy as np
import cv2

from getData_Michael import get_BS_LOCATIONS_DICT, getData, writeData_formatted


R_EARTH = 6378137
CIRCUMFERENCE = 2 * np.pi * R_EARTH  # circumference of earth
MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiemwzMTAiLCJhIjoiY2xocGplMnBrMjV5djNtcW9jbHZ3Y2IzbyJ9.eAHPD0MS19DVMGMxHV7eAQ'


def new_row_col(lat_Rx, long_Rx, lat_Tx, long_Tx, pad_frac, zoom_level, orig_img_shape):
    """
    Gives the row and col based on (lat_Tx, long_Tx) and (lat_Rx, long_Rx) (the receiver should be at the centre of
    the image). The distance between Rx and Tx is calculated, and we know meter per array index based on the
    Zoom level.
    :return: row, col in padded image of Tx
    """
    # print(CIRCUMFERENCE, lat_Rx, zoom_level)
    orig_row_len, orig_col_len = orig_img_shape[0], orig_img_shape[1]
    meters_per_idx = CIRCUMFERENCE * np.cos(float(lat_Rx) * np.pi / 180) / (2 ** zoom_level) / 256
    dx = (long_Tx - long_Rx) * R_EARTH / (180 / np.pi) / np.cos(lat_Rx * np.pi / 180)
    dy = (lat_Tx - lat_Rx) * R_EARTH / (180 / np.pi)
    r = int(orig_row_len + 1 * orig_row_len * pad_frac -
            (orig_row_len / 2 + dy / meters_per_idx))
    c = int(orig_col_len / 2 + dx / meters_per_idx + orig_col_len * pad_frac)
    return r, c


def get_rotation_angle(lat_Rx, long_Rx, lat_Tx, long_Tx):
    # bs_locations_dict = get_BS_LOCATIONS_DICT()
    # bs_lat, bs_long = bs_locations_dict[data_formatted['PCI'][idx]][0:2]
    dx, dy = long_Rx - long_Tx, lat_Rx - lat_Tx
    ang = np.angle(dx + 1j * dy)
    if np.pi / 2 <= ang <= np.pi:  # quadrant II
        return np.pi / 2 - np.abs(ang)  # this is correct (provisionally)
    elif -np.pi <= ang <= -np.pi / 2:  # quadrant III
        return np.abs(ang) - 3 * np.pi / 2  # this is correct (provisionally)
    elif 0 <= ang <= np.pi / 2:  # quadrant I
        return np.pi / 2 - np.abs(ang)  # this is correct (provisionally)
    elif -np.pi / 2 <= ang <= 0:  # quadrant IV
        return np.pi / 2 + np.abs(ang)
    print("ERROR: get_rotation_angle did not return angle")
    return


def rotate_img_about_antenna(img_arr, lat_Rx, long_Rx, lat_Tx, long_Tx, pad_frac, zoom_level, orig_img_shape,
                             scale=1.0):
    """Rotates the image about the antenna."""
    # img_arr should be padded; long_centre, lat_centre are the longitude and latitude of the receiver. Rotate image such that the line connecting the user and transmitter is vertical.
    rot_matrix = get_rotation_mat(lat_Rx, long_Rx, lat_Tx, long_Tx, scale=scale, zoom_level=zoom_level,
                                  orig_img_shape=orig_img_shape, pad_frac=pad_frac)
    (h, w) = img_arr.shape[:2]
    rotated = cv2.warpAffine(img_arr, rot_matrix, (w, h))
    return rotated


def rotate_point_about_antenna(lat_Rx, long_Rx, lat_Tx, long_Tx, pad_frac, zoom_level, orig_img_shape,
                               scale=1.0, rot_matrix=None):
    # orig_img_shape=5000 using im_arr.shape
    """NOTE: rotates the point (x,y) about the antenna, not (row,col). But, the rotation is flipped in the return line, so the output is (row, col)."""
    if rot_matrix is None:
        rot_matrix = get_rotation_mat(lat_Rx, long_Rx, lat_Tx, long_Tx, scale=scale, zoom_level=zoom_level,
                                      orig_img_shape=orig_img_shape, pad_frac=pad_frac)
    point = np.ones(3)
    (point[0], point[1]) = (orig_img_shape[1] / 2 + orig_img_shape[1] * pad_frac,
                            orig_img_shape[0] / 2 + orig_img_shape[0] * pad_frac)
    # swapping to (x,y) for rotation
    # transforming the shifted (x,y) location using the rotation matrix, i.e. provides (col, row) indexing into padded img
    return np.matmul(rot_matrix, point.T)[::-1]


def get_rotation_mat(lat_Rx, long_Rx, lat_Tx, long_Tx, pad_frac, zoom_level, orig_img_shape, scale=1.0):
    r_Tx, c_Tx = new_row_col(lat_Rx, long_Rx, lat_Tx, long_Tx, pad_frac=pad_frac, zoom_level=zoom_level,
                             orig_img_shape=orig_img_shape)
    # long_centre, lat_centre: longitude and latitude of the receiver
    rot_angle = (get_rotation_angle(lat_Rx, long_Rx, lat_Tx, long_Tx)) * 180 / np.pi
    # radian to degree
    # centre is the antenna, no padding
    rot_matrix = cv2.getRotationMatrix2D(center=(c_Tx, r_Tx), angle=rot_angle, scale=scale)
    return rot_matrix


def get_Img_all(data_formatted, directory, mapbox_tile_set_ID, cropped_size, pad_frac, zoom_level):
    """
    This saves all satellite images whose centres are specified by (latitude, longitude) pairs in data_formatted
    :param data_formatted: formatted data, output of
    :param directory: in which to store the satellite image chips
    :param mapbox_tile_set_ID: for folium, should be 'mapbox.satellite'
    :param cropped_size: should be 256. Size of the square satellite image chips
    :param pad_frac: padding of the screenshot, normally 0.2 but is variable
    :param zoom_level: how much folium is zooming in, should be 17
    :return:
    """
    latitude, longitude, PCI = data_formatted['Latitude'], data_formatted['Longitude'], data_formatted['PCI']
    start = datetime.now()
    for i in range(len(PCI)):  # row = [index, longitude, latitude...]
        key = (str(i), float(latitude[i]), float(longitude[i]))  # key = [index, latitude, longitude]
        lat_Rx, long_Rx = float(latitude[i]), float(longitude[i])
        bs_locations_dict = get_BS_LOCATIONS_DICT()
        lat_Tx, long_Tx = bs_locations_dict[data_formatted['PCI'][i]][0:2]
        print(key)
        m = folium.Map(
            location=[lat_Rx, long_Rx],
            zoom_start=17,
            tiles='https://api.tiles.mapbox.com/v4/' + mapbox_tile_set_ID + '/{z}/{x}/{y}.png?access_token=' + MAPBOX_ACCESS_TOKEN,
            attr='mapbox.com')
        img_data = m._to_png(4)
        img = Image.open(io.BytesIO(img_data))
        img = np.asarray(img)
        orig_row_len, orig_col_len, _ = img.shape
        img_padded = np.pad(array=img, pad_width=((int(orig_row_len * pad_frac), int(orig_row_len * pad_frac)),
                                                  (int(orig_col_len * pad_frac), int(orig_col_len * pad_frac)),
                                                  (0, 0)), mode='edge')
        img_rotated_padded = \
            rotate_img_about_antenna(img_arr=img_padded, lat_Rx=lat_Rx, long_Rx=long_Rx,
                                     lat_Tx=lat_Tx, long_Tx=long_Tx, pad_frac=pad_frac,
                                     orig_img_shape=(orig_row_len, orig_col_len), zoom_level=zoom_level)
        row_c, col_c = rotate_point_about_antenna(lat_Rx=lat_Rx, long_Rx=long_Rx, lat_Tx=lat_Tx,
                                                  long_Tx=long_Tx, pad_frac=pad_frac,
                                                  orig_img_shape=(orig_row_len, orig_col_len), zoom_level=zoom_level)
        half_cropped_size = int(cropped_size / 2)
        chip = img_rotated_padded[int(row_c) - half_cropped_size:int(row_c) + half_cropped_size,
                                  int(col_c) - half_cropped_size:int(col_c) + half_cropped_size]
        chip_img = Image.fromarray(chip)
        chip_img.save(directory + '/' + key[0] + '.png')
        print('Time Elapsed: ' + str((datetime.now() - start).seconds) + ' seconds.\n')
    return


def main():
    direc = '../data/data_05_08_23'  # the raw data's directory
    mapbox_tile_ID = 'mapbox.satellite'
    pad_fraction_ = 0.2
    zoom_level_ = 17
    dat = getData(direc)
    cropped_chip_size = 256
    features, _ = writeData_formatted(dat, dest_dir=None, date=None, write='n')
    get_Img_all(data_formatted=features, directory='../formatted_data/map_api', mapbox_tile_set_ID=mapbox_tile_ID,
                cropped_size=cropped_chip_size, pad_frac=pad_fraction_, zoom_level=zoom_level_)


if __name__ == '__main__':
    main()
