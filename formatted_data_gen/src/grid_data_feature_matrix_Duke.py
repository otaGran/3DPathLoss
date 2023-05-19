# ,Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI,PCI_3,PCI_4,PCI_6,PCI_7,PCI_20,PCI_40,PCI_184,PCI_237

from getData_Michael import get_BS_LOCATIONS_DICT
# note: BS_LOCATIONS_DICT = {key: [lat, long, elevation], ...}

import numpy as np
import pandas as pd

from geopy import distance


BS_LOCATIONS_DICT = get_BS_LOCATIONS_DICT()
# defining box in which to create the gridded (lat, long) pairs
lat_BL, long_BL = 35.997700, -78.941049
lat_TL, long_TL = 36.003836, -78.941049
lat_TR, long_TR = 36.003836, -78.936245
lat_BR, long_BR = 35.997700, -78.936245

Rx_HEIGHT = 100


def write_feature_csv_grid(directory, which_pci, lat_len, long_len):
    assert which_pci in BS_LOCATIONS_DICT.keys(), print("which_pci is not a valid PCI for Duke's Data")
    lat_arr = np.linspace(lat_TL, lat_BL, num=lat_len)
    long_arr = np.linspace(long_TL, long_TR, num=long_len)
    (lat_len * long_len)
    latitude_arr = np.zeros(lat_len * long_len)
    longitude_arr = np.zeros(lat_len * long_len)
    Distance = np.zeros(lat_len * long_len)
    Distance_x = np.zeros(lat_len * long_len)
    Distance_y = np.zeros(lat_len * long_len)
    PCI = np.zeros(lat_len * long_len)

    outer_idx = 0

    for iii, x in enumerate(long_arr):
        for y in lat_arr:
            latitude_arr[outer_idx], longitude_arr[outer_idx] = y, x
            two_D_dist = distance.distance((latitude_arr[outer_idx], longitude_arr[outer_idx]),
                                           (BS_LOCATIONS_DICT[which_pci][0], BS_LOCATIONS_DICT[which_pci][1])).km
            # BS_LOCATIONS_DICT[which_pci][0], BS_LOCATIONS_DICT[which_pci][1] gives the lat, long location of which_pci
            Distance[outer_idx] = np.sqrt(two_D_dist ** 2 + ((BS_LOCATIONS_DICT[which_pci][2] - Rx_HEIGHT) / 1000) ** 2)

            # Distance_x is Rx_lat - Tx_lat, Distance_y is Rx_long - Tx_long: Thrane did it very weirdly
            Distance_x[outer_idx] = latitude_arr[outer_idx] - BS_LOCATIONS_DICT[which_pci][0]
            Distance_y[outer_idx] = longitude_arr[outer_idx] - BS_LOCATIONS_DICT[which_pci][1]
            PCI[outer_idx] = int(which_pci)

            outer_idx += 1
    length_df = len(longitude_arr)
    PCI_name_list = ['PCI_{:d}'.format(k) for k in BS_LOCATIONS_DICT.keys()]
    col_names = ['Longitude', 'Latitude', 'Speed', 'Distance', 'Distance_x', 'Distance_y', 'PCI']
    col_names = col_names + PCI_name_list
    PCI_i_list = [np.zeros(length_df) for i in range(len(PCI_name_list))]
    df = pd.DataFrame(list(zip(longitude_arr, latitude_arr, np.zeros(length_df), Distance, Distance_x, Distance_y, PCI, *PCI_i_list)), columns=col_names)
    df['PCI_' + str(which_pci)] = np.ones(length_df)

    csv_feature = open(directory + 'feature_matrix_grid' + '.csv', 'w')
    csv_feature.write(',Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI,' + ','.join(PCI_name_list) + '\n')
    for i in range(len(df['PCI'])):
        which_PCI = [str(int(df['PCI_' + str(int(key))][i])) for key in BS_LOCATIONS_DICT.keys()]
        csv_feature.write('{x},{a},{b},{c},{d},{e},{f},{g},{lst}\n'.
                          format(x=i, a=df['Longitude'][i], b=df['Latitude'][i],
                                 c=df['Speed'][i], d=round(df['Distance'][i], 10),
                                 e=round(df['Distance_x'][i], 7),
                                 f=round(df['Distance_y'][i], 7),
                                 g=df['PCI'][i], lst=','.join(which_PCI)))
    return


if __name__ == '__main__':
    write_feature_csv_grid(directory='../grid_data/', which_pci=3, lat_len=251, long_len=151)
