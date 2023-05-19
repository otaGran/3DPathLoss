import numpy as np
import pandas as pd

from geopy import distance
from pyproj import Transformer

"""
    This code generates a grid pattern of latitude and longitude values. The resulting
    features file is fed into the model to produce path loss predictions at each of the 
    (latitude, longitude) pairs. 
"""

PCI_KEYS = [64, 65, 302]
Tx_LAT, Tx_LONG = 55.784663, 12.523303
# 55.784663, 12.516743
Tx_HEIGHT, Rx_HEIGHT = 30, 1.5  # in meters

# the following coordinates draw the corners of DTU
# top left:
coord_25832_TL_x, coord_25832_TL_y = 720781, 6188840
# top right:
coord_25832_TR_x, coord_25832_TR_y = 721226, 6188814
# bottom right:
coord_25832_BR_x, coord_25832_BR_y = 720997, 6187226
# bottom left:
coord_25832_BL_x, coord_25832_BL_y = 720167, 6187286


def write_feature_csv_grid(which_pci, x_len, y_len):
    from_EPSG = 25832
    to_EPSG = 4326
    transform_obj = Transformer.from_crs(from_EPSG, to_EPSG)  # transform from DTU coordinates to latitude, longitude

    x_arr = np.linspace(coord_25832_TL_x, coord_25832_TR_x, num=x_len)
    y_arr = np.linspace(coord_25832_TL_y, coord_25832_BL_y, num=y_len)


    latitude_arr = np.zeros(x_len * y_len)
    longitude_arr = np.zeros(x_len * y_len)
    Distance = np.zeros(x_len * y_len)
    Distance_x = np.zeros(x_len * y_len)
    Distance_y = np.zeros(x_len * y_len)
    PCI = np.zeros(x_len * y_len)

    outer_idx = 0

    for iii, x in enumerate(x_arr):
        for y in y_arr:
            latitude_arr[outer_idx], longitude_arr[outer_idx] = transform_obj.transform(xx=x, yy=y)
            two_D_dist = distance.distance((latitude_arr[outer_idx], longitude_arr[outer_idx]),
                                           (Tx_LAT, Tx_LONG)).km
            Distance[outer_idx] = np.sqrt(two_D_dist ** 2 + ((Tx_HEIGHT - Rx_HEIGHT) / 1000) ** 2)

            # Distance_x is Rx_lat - Tx_lat, Distance_y is Rx_long - Tx_long: Thrane did it very weirdly
            Distance_x[outer_idx], Distance_y[outer_idx] = latitude_arr[outer_idx] - Tx_LAT, longitude_arr[outer_idx] - Tx_LONG
            PCI[outer_idx] = int(which_pci)

            outer_idx += 1
    length_df = len(longitude_arr)
    PCI_name_list = ['PCI_{:d}'.format(k) for k in PCI_KEYS]
    col_names = ['Longitude', 'Latitude', 'Speed', 'Distance', 'Distance_x', 'Distance_y', 'PCI']
    col_names = col_names + PCI_name_list
    df = pd.DataFrame(list(zip(longitude_arr, latitude_arr, np.zeros(length_df), Distance, Distance_x, Distance_y, PCI, np.zeros(length_df), np.zeros(length_df), np.zeros(length_df))), columns=col_names)
    df['PCI_' + str(which_pci)] = np.ones(length_df)

    csv_feature = open('grid_data' + '/feature_matrix_grid' + '.csv', 'w')
    csv_feature.write(',Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI,' + ','.join(PCI_name_list) + '\n')
    for i in range(len(df['PCI'])):
        # print(type(feature_subset['PCI_' + str(3)]))
        which_PCI = [str(int(df['PCI_' + str(int(key))][i])) for key in PCI_KEYS]
        # print(which_PCI)
        csv_feature.write('{x},{a},{b},{c},{d},{e},{f},{g},{lst}\n'.
                          format(x=i, a=df['Longitude'][i], b=df['Latitude'][i],
                                 c=df['Speed'][i], d=df['Distance'][i],
                                 e=round(df['Distance_x'][i], 7),
                                 f=round(df['Distance_y'][i], 7),
                                 g=df['PCI'][i], lst=','.join(which_PCI)))
    return


if __name__ == '__main__':
    write_feature_csv_grid(64, x_len=101, y_len=301)
