import numpy as np
import pandas as pd

from PIL import Image
from pyproj import Transformer

import cv2

"""
Written by Michael (Zeyu) Li


Link for the data: https://dataforsyningen.dk/data/928
Download from DENMARK'S HEIGHT MODEL - SURFACE (10 km blocks)
Must have raw_data/feature_matrix.csv, height_maps/, DSM_6188_720_2x2/ (with TL, TR, BL, BR added as suffixes 
to the tiff files). 
"""

def long_lat_to_arr_idx(long, lat, from_EPSG=4326, to_EPSG=25832):
    transform_obj = Transformer.from_crs(from_EPSG, to_EPSG)
    dest_x, dest_y = transform_obj.transform(long, lat)
    # converting from (x, y) to (row, col)
    row_arr, col_arr = (TL_25832_coord_x - dest_y) / idx_per_meter, (dest_x - TL_25832_coord_y) / idx_per_meter
    return int(np.round(row_arr)), int(np.round(col_arr))


TL_25832_coord_x, TL_25832_coord_y = 6189000, 720000  # global var, top left of the 2kmx2km block, (x, y) coord
idx_per_meter = 0.4  # global var, badly named - should be meter per index
Tx_4326_x, Tx_4326_y = 55.784663, 12.523303  # long, lat of transmitter
# row, col of transmitter according to the python array convention
Tx_25832_coord_row, Tx_25832_coord_col = long_lat_to_arr_idx(Tx_4326_x, Tx_4326_y, from_EPSG=4326, to_EPSG=25832)
pad_fraction = 0.2  # padding added to original image, used in pad_width (np.padding)
chip_dimension = 256 * 1.25  # dimension of each chip (height map)


def get_rotation_angle(long_centre, lat_centre):
    """Returns the angle difference between the y-axis (with axes centred at Tx antenna) and the receiver. The sign is the rotational direction (positive == rotate counterclockwise, negative == rotate clockwise)"""
    # print('get_rotation_angle printing... ...')
    # print('receiver longitude  {:f}'.format(long_centre), '  receiver latitude  {:f}'.format(lat_centre))
    transform_obj = Transformer.from_crs(4326, 25832)
    Tx_x, Tx_y = transform_obj.transform(Tx_4326_x, Tx_4326_y)  # outputs are in terms of EPSG 25832
    Rx_x, Rx_y = transform_obj.transform(long_centre, lat_centre)  # outputs are in terms of EPSG 25832
    dx, dy = Rx_x - Tx_x, Rx_y - Tx_y
    better_angle = np.angle(dx + 1j * dy)
    # print('angle using EPSG 25832:  {:f}'.format(better_angle))
    r, c = long_lat_to_arr_idx(long=long_centre, lat=lat_centre, from_EPSG=4326, to_EPSG=25832)
    r_diff, c_diff = Tx_25832_coord_row - r, c - Tx_25832_coord_col
    # inverted the r_diff coordinate to convert from (row, col) (row 0 starts top left) to (x, y) (which starts bottom left)
    worse_ang = np.angle(c_diff + 1j * r_diff)
    # print('angle using array idx:  {:f}'.format(worse_ang))
    # print('... ...get_rotation_angle finished printing\n')
    """this is the better angle, calculated using 25832 coordinates"""
    ang = better_angle
    if np.pi / 2 <= ang <= np.pi:  # quadrant II
        return np.pi / 2 - np.abs(ang)  # this is correct (provisionally)
    elif -np.pi <= ang <= -np.pi / 2:  # quadrant III
        return np.abs(ang) - 3 * np.pi / 2  # this is correct (provisionally)
    elif 0 <= ang <= np.pi / 2:  # quadrant I
        return np.pi / 2 - np.abs(ang)  # this is correct (provisionally)
    elif -np.pi / 2 <= ang <= 0:  # quadrant IV
        return np.pi / 2 + np.abs(ang)
    print("ERROR: get_rotation_angle did not return angle")


"""Functions to rotate the image and rotate the receiver's location"""
def rotate_img_about_antenna(img_arr, long_centre, lat_centre, scale=1.0, pad_frac=pad_fraction):
    """Rotates the image about the antenna."""
    # img_arr should be padded; long_centre, lat_centre are the longitude and latitude of the receiver. Rotate image such that the line connecting the user and transmitter is vertical.
    rot_matrix = get_rotation_mat(long_centre, lat_centre, scale=scale, pad_frac=pad_frac)
    (h, w) = img_arr.shape[:2]
    rotated = cv2.warpAffine(img_arr, rot_matrix, (w, h))
    return rotated

def rotate_point_about_antenna(long_centre, lat_centre, scale=1.0, pad_frac=pad_fraction, orig_img_shape=5000, rot_matrix=None):
    # orig_img_shape=5000 using im_arr.shape
    """NOTE: rotates the point (x,y) about the antenna, not the (row,col). Need to do plt.scatter(point[1], point[0]) (i.e., flip the points). DOES NOT RETURN int array, therefore must int(output) when treating output as arr idx."""
    if rot_matrix is None:
        rot_matrix = get_rotation_mat(long_centre, lat_centre, scale=scale, pad_frac=pad_frac)
    point = np.ones(3)
    # long_lat_to_arr_idx returns the array index, hence we need to transform the indices
    (point[0], point[1]) = long_lat_to_arr_idx(long=long_centre, lat=lat_centre, from_EPSG=4326, to_EPSG=25832)
    # adding the padding (i.e., shifting) and swapping to (x,y) for rotation
    (point[0], point[1]) = (point[1] + orig_img_shape * pad_frac, point[0] + orig_img_shape * pad_frac)
    # transforming the shifted (x,y) location using the rotation matrix, i.e. provides (col, row) indexing into padded img
    return np.matmul(rot_matrix, point.T)

def get_rotation_mat(long_centre, lat_centre, scale=1.0, pad_frac=pad_fraction, orig_img_shape=5000):
    # long_centre, lat_centre: longitude and latitude of the receiver
    rot_angle = (get_rotation_angle(long_centre, lat_centre)) * 180 / np.pi # radian to degree
    # perhaps subtract 0.025 rad from rot_angle to increase accuracy
    # centre is the antenna, accounting for padding
    rot_matrix = cv2.getRotationMatrix2D(center=(Tx_25832_coord_col + orig_img_shape * pad_frac, Tx_25832_coord_row + orig_img_shape * pad_frac), angle=rot_angle, scale=scale)
    return rot_matrix


def get_chip(padded_img, test_lon, test_lat, arr_dimension=chip_dimension):
    # rotated image:
    rot_img = rotate_img_about_antenna(img_arr=padded_img, long_centre=test_lon, lat_centre=test_lat, scale=1.0, pad_frac=pad_fraction)
    # rotated coordinates of the receiver, using the padded image's col/row system:
    col_rot, row_rot = rotate_point_about_antenna(long_centre=test_lon, lat_centre=test_lat, scale=1.0, pad_frac=pad_fraction, orig_img_shape=5000, rot_matrix=None)
    return rot_img[int(row_rot-arr_dimension/2):int(row_rot+arr_dimension/2),
                   int(col_rot-arr_dimension/2):int(col_rot+arr_dimension/2)]


"""This code combines the four height maps (tiff files) into a single array according to geographical information
and pads the combined image to enable rotation without cutting off edges."""
directory = 'DSM_6188_720_2x2/'
name = 'DSM_1km_'
im1_TL = Image.open(directory + name + '6188_720_TL.tif')
im2_TR = Image.open(directory + name + '6188_721_TR.tif')
im3_BL = Image.open(directory + name + '6187_720_BL.tif')
im4_BR = Image.open(directory + name + '6187_721_BR.tif')
(row_len, col_len) = np.asarray(im1_TL).shape
im_arr = np.zeros((row_len * 2, col_len * 2))
im_arr[0:row_len, 0:col_len] = np.asarray(im1_TL)
im_arr[0:row_len, col_len:col_len*2] = np.asarray(im2_TR)

im_arr[row_len:2*row_len, 0:col_len] = np.asarray(im3_BL)
im_arr[row_len:2*row_len, col_len:col_len*2] = np.asarray(im4_BR)

rrr, ccc = im_arr.shape
pad_fraction = 0.2
im_arr_padded = np.pad(array=im_arr, pad_width=(int(rrr * pad_fraction), int(ccc * pad_fraction)), mode='edge')


# get_chip(padded_img=im_arr_padded, test_lon=0, test_lat=0, arr_dimension=chip_dimension)

"""NOTE: this code (mistakenly) flipped the names of all longitude and latitude pairs. 
I thought (longitude, latitude) = (x,y), but the correct version is (latitude, longitude) = (x,y). """

df_feature = pd.read_csv('raw_data/feature_matrix.csv')
for r_df in range(len(df_feature)):
    temp_long = df_feature['Latitude'][r_df]
    temp_lati = df_feature['Longitude'][r_df]
    temp_chip = get_chip(padded_img=im_arr_padded, test_lon=temp_long, test_lat=temp_lati, arr_dimension=chip_dimension)
    im = Image.fromarray(temp_chip).convert("L")
    im.save("height_maps/{:d}.png".format(r_df))
    print(r_df)
