import pandas as pd
import numpy as np
from geopy import distance

import json
import os
import itertools

"""
    This code is based on SigCap_colormap.ipynb. Written by Michael Li. 
    
    This code formats SigCap data into the same form of csv file as provided by Thrane's raw data. 
"""

__all__ = ['getData', 'getLocationDF', 'getSortedKeys', 'getConnectedIndices', 'timeSub', 'rescaleDB',
           'getDukeNodeData', 'get_BS_LOCATIONS_DICT', 'getData', 'writeData_formatted', 'dB_to_pow',
           'pow_to_dB', 'BS_ELEVATION']

OUT_OF_BOUNDS = -300

"""
    These are not the actual locations/heights of base stations. [latitude, longitude]. These are PLACEHOLDER values. 
    # When adding/removing keys from BS_LOCATIONS_DICT, must 
"""
BS_ELEVATION = 300
BS_LOCATIONS_DICT = {3: [36.002535, -78.937431, BS_ELEVATION], 4: [35.999245, -78.939770, BS_ELEVATION],
                     6: [36.004938, -78.932040, BS_ELEVATION],
                     7: [36.002919, -78.935555, BS_ELEVATION], 20: [36.000000, -78.937000, BS_ELEVATION],
                     40: [36.002009, -78.940000, BS_ELEVATION],
                     184: [35.998999, -78.940770, BS_ELEVATION], 237: [36.003000, -78.940023, BS_ELEVATION]}


def get_BS_LOCATIONS_DICT():
    return BS_LOCATIONS_DICT


'''
    Gets data collected only from our CBRS node (pci: 40 or 20). Returns data structure of same format
'''

# def getDukeNodeData(data):
#     duke_data = {'id': [], 'carrierName': [], 'opName': [], 'location' : {'latitude': [], 'longitude': []}, 'cell_info': {'ss': [], 'band': [], 'freq': []}, 'time_stamp': [], 'date' : []}
#     for i in range(len(data['id'])):
#         x = []
#         if 40 in data['cell_info']['pci'][i]:
#             x += [data['cell_info']['ss'][i][data['cell_info']['pci'][i].index(40)]]
#         if 20 in data['cell_info']['pci'][i]:
#             x += [data['cell_info']['ss'][i][data['cell_info']['pci'][i].index(20)]]

#         if len(x) > 0:
#             duke_data['cell_info']['ss'] += [np.mean(x)]
#         else:
#             duke_data['cell_info']['ss'] += [OUT_OF_BOUNDS]

#     duke_data['id'] = data['id']
#     duke_data['location'] = data['location']
#     duke_data['time_stamp'] = data['time_stamp']
#     duke_data['date'] = data['date']
#     duke_data['cell_info']['band'] = data['cell_info']['band']
#     duke_data['cell_info']['freq'] = data['cell_info']['freq']
#     duke_data['carrierName'] = data['carrierName']
#     duke_data['opName'] = data['opName']


#     return duke_data

'''
    Converts location element in data structure to pandas data frame to be
    inputted into Google Maps API
'''


def getLocationDF(data):
    return pd.DataFrame(data['location'])


'''
    Custom function to coerce dBm to something acceptable by Google Maps API
'''


def rescaleDB(data, function):
    return [function(x) for x in data['cell_info']['ss']]


'''
    Return keys in sorted order by the time stamp, can use this to grab other data
    in chronological order.
'''


def getSortedKeys(data):
    return sorted(data['id'], key=lambda x: data['time_stamp'][x])


'''
    Return indices of measurements where a connection to CBRS node is observed
'''


def getConnectedIndices(data):
    return [i for i in data['id'] if len(data['cell_info'][i]) > 0]


'''
    Translate timestamp to readable format
'''


def timeFormat(x):
    return x[0:2] + ':' + x[2:4] + ':' + x[4:6] + ':' + x[6:8]


'''
    Translate date to readable format
'''


def dateFormat(x):
    return x[0:4] + ',' + x[4:6] + ',' + x[6::]


'''
    Find time difference between time_stamp x and time_stamp y
'''


def timeSub(x, y):
    if y > x:
        temp = '' + x
        x = '' + y
        y = '' + temp
    x_comp = x.split(':')
    x = int(x_comp[0]) * 60 * 60 + int(x_comp[1]) * 60 + int(x_comp[2]) + int(x_comp[3]) / 100
    y_comp = y.split(':')
    y = int(y_comp[0]) * 60 * 60 + int(y_comp[1]) * 60 + int(y_comp[2]) + int(y_comp[3]) / 100

    return round(x - y, 4)


"""
    Find a set of IDs of cell towers present during measurement.
"""


def unique_pci(data):
    b = set()
    for x in data['cell_info']['pci']:
        for k in x:
            b.add(k)
    return sorted(list(b))


"""
    Convert dB to power (assuming P1 == 1W, using 1 dB = 10*log10(P2/P1))
"""


def dB_to_pow(db):
    return 10 ** (db / 10)


"""
    Convert power to dB (assuming P1 == 1W, using 1 dB = 10*log10(P2/P1))
"""


def pow_to_dB(p):
    return 10 * np.log10(p)


'''
    Loads SigCap data files from directory or list of directories to get important 
    information for preprocessing.
    Returns data structure with format:
        data = {
        'id': [measurement keys],
        'location' : {'latitude': [measurement latitudes], 'longitude': [measurement longitudes]},
        'location_wrt_BS': {'d_latitude': [measurement latitudes - BS latitude], 'd_longitude': [measurement longitudes - BS longitude]}
        'distances_to_BS': [straight-line 3D distance between measurement location and BS location]
        'cell_info': [list of signal strength info],
        'time_stamp': [list of time stamps],
        'date' : [list of dates]}
    NOTE: data[location_wrt_BS][d_latitude] and data[location_wrt_BS][d_longitude] and distances_to_BS has length same as 
    data[cell_info][pci]. Other fields may have the same length as data[cell_info][pci] as well. 
'''


def getData(directory):
    data = {'id': [], 'carrierName': [], 'opName': [],
            'location': {'latitude': [], 'longitude': [], 'altitude': []},
            'location_wrt_BS': {'d_latitude': [], 'd_longitude': []},
            'distances_to_BS': [],
            'cell_info': {'pci': [], 'ss': [], 'rsrp': [], 'rsrq': [], 'sinr': [], 'band': [], 'freq': []},
            'time_stamp': [], 'date': []}
    lst = sorted(os.listdir(directory))
    # print(len(data))
    for i, f in enumerate(lst):
        new_data = json.load(open(directory + '/' + f))

        data['id'] += [i]

        # original code
        data['location']['longitude'] += [new_data['location']['longitude']]
        data['location']['latitude'] += [new_data['location']['latitude']]
        data['location']['altitude'] += [new_data['location']['altitude']]

        signal_strengths = [x['ss'] for x in new_data['cell_info']]
        data['cell_info']['ss'] += [[] if len(signal_strengths) == 0 else signal_strengths]

        # my additions
        signal_rsrp = [x['rsrp'] for x in new_data['cell_info']]
        data['cell_info']['rsrp'] += [[] if len(signal_strengths) == 0 else signal_rsrp]
        signal_rsrq = [x['rsrq'] for x in new_data['cell_info']]
        data['cell_info']['rsrq'] += [[] if len(signal_strengths) == 0 else signal_rsrq]

        # original code
        data['cell_info']['pci'] += [[x['pci'] for x in new_data['cell_info']]]
        data['cell_info']['band'] += [[x['band'] for x in new_data['cell_info']]]
        data['cell_info']['freq'] += [[x['freq'] for x in new_data['cell_info']]]
        data['carrierName'] += [new_data['carrierName']]
        data['opName'] += [new_data['opName']]

        # my additions
        sum_ss = 0  # sum of all ss values from different BS for measurement i
        for dB in data['cell_info']['ss'][i]:
            sum_ss += dB_to_pow(dB)
        if sum_ss != 0:
            # computing pow(from this BS) / (sum of pow from all other BSs). If only received one signal, set sinr to +300 (very large signal to noise ratio)
            # if only received 2 sigs, one should be the negative of the other
            sinr = [-OUT_OF_BOUNDS if len(data['cell_info']['ss'][i]) == 1 else pow_to_dB(
                dB_to_pow(data['cell_info']['ss'][i][iii]) / (sum_ss - dB_to_pow(data['cell_info']['ss'][i][iii]))) for
                    iii in range(len(data['cell_info']['pci'][i]))]
        data['cell_info']['sinr'] += [[] if len(signal_strengths) == 0 else sinr]

        # original code
        data['time_stamp'] += [timeFormat(new_data['datetime']['time'])]

        data['date'] += [dateFormat(new_data['datetime']['date'])]

        # my additions
        # BS_LOCATIONS_DICT[pci][0] contains latitude data of specific BS
        data['location_wrt_BS']['d_latitude'] += [
            [BS_LOCATIONS_DICT[pci][0] - data['location']['latitude'][i] for pci in data['cell_info']['pci'][i]]]
        data['location_wrt_BS']['d_longitude'] += [
            [BS_LOCATIONS_DICT[pci][1] - data['location']['longitude'][i] for pci in data['cell_info']['pci'][i]]]

        # calculates distance between receiver and each BS using (lat, lon) and a package, the elevation of each BS and geopy
        data['distances_to_BS'] += [[np.sqrt(distance.distance((BS_LOCATIONS_DICT[pci][0], BS_LOCATIONS_DICT[pci][1]), (
        data['location']['latitude'][i], data['location']['longitude'][i])).meters ** 2 + (
                                                         BS_LOCATIONS_DICT[pci][2] - data['location']['altitude'][
                                                     i]) ** 2) for pci in data['cell_info']['pci'][i]]]
    # print(len(data))
    return data


'''
    Uses the dictionary produced by getData to format and write the data into a csv file that resembles
    the raw_data/feature_matrix.csv in Jakob Thrane's code and paper. 
'''


def writeData_formatted(data, dest_dir, date, write='y'):
    if write == 'y':
        csv_feature = open(dest_dir + '/feature_matrix_' + date + '.csv', 'w')
        csv_output = open(dest_dir + '/output_matrix_' + date + '.csv', 'w')
    data_copy = data.copy()
    # note: in the data_subset dict, Longitude comes before Latitude; this is to satisfy the ordering of the feature_matrix
    # , Longitude, Latitude, Speed, Distance, Distance_x, Distance_y, PCI, PCI_64, PCI_65, PCI_302




    # this is taking the index of the maximum ss for each measurement to figure out which base station
    # has the strongest signal
    # index_of_max_ss = [np.argmax(np.asarray(values)) for values in data_copy['cell_info']['ss']]
    m_to_km = 1000
    # below, Distance, Distance_x, Distance_y are taken from the original data at indices such that
    # ss is maximised for the same indices. E.g. data['cell_info']['ss'][0] == [20, -10, -25],
    # data['distances_to_BS'][0] == [100, 200, 300], the corresponding value in 'Distance' would be
    # data_subset['Distance'][0] == 100 / m_to_km.
    # Indexing: ['distances_to_BS'] to get to the dict field, [i] to get to the ith measurement (which has a list
    # of all the distances_to_BS values), [index_of_max_ss[i]] to get the correct index.




    # length of the dataset after expanding each measurement; each pci in each data point counts as one data point, so if
    # data['cell_info']['ss'][k] == [2, -10, -20], this is three data points instead of one data point
    total_len = len(list(itertools.chain.from_iterable(data_copy['cell_info']['ss'])))

    # Assume that Distance_x is \Delta(longitude), Distance_y is \Delta(latitude).
    feature_subset = dict(id=np.linspace(start=0, stop=total_len - 1, num=total_len, endpoint=True),
                          Speed=np.zeros(total_len),
                          Longitude=np.zeros(total_len),
                          Latitude=np.zeros(total_len),
                          Distance=np.array(list(itertools.chain.from_iterable(data_copy['distances_to_BS']))),
                          Distance_x=np.array(list(itertools.chain.from_iterable(data_copy['location_wrt_BS']['d_latitude']))),
                          Distance_y=np.array(list(itertools.chain.from_iterable(data_copy['location_wrt_BS']['d_longitude']))),
                          PCI=np.array(list(itertools.chain.from_iterable(data_copy['cell_info']['pci'])))
                          )
    BS_LOCATIONS_DICT_keys = sorted(list(BS_LOCATIONS_DICT.keys()))
    for PCI_key in BS_LOCATIONS_DICT_keys:
        feature_subset['PCI_' + str(PCI_key)] = np.zeros(total_len)

    feature_PCI_keys = sorted(list(feature_subset.keys())[8:])  # this line must be placed after feature_subset has been constructed

    outer = 0
    for orig_idx, sig in enumerate(data_copy['cell_info']['ss']):
        for sig_iter in range(len(sig)):
            feature_subset['Longitude'][outer], feature_subset['Latitude'][outer] = data_copy['location']['longitude'][orig_idx], data_copy['location']['latitude'][orig_idx]
            feature_subset['PCI_' + str(int(feature_subset['PCI'][outer]))][outer] = 1
            outer += 1
    if feature_PCI_keys != sorted(list(['PCI_{:d}'.format(k) for k in BS_LOCATIONS_DICT_keys])):
        print('WARNING by writeData_formatted: there are missing/extra PCI features in feature_subset. '
              'Ensure that the keys of BS_LOCATIONS_DICT match the PCI keys in feature_subset. ')
    output_subset = dict(SINR=np.array(list(itertools.chain.from_iterable(data_copy['cell_info']['sinr']))),
                         RSRP=np.array(list(itertools.chain.from_iterable(data_copy['cell_info']['rsrp']))),
                         RSRQ=np.array(list(itertools.chain.from_iterable(data_copy['cell_info']['rsrq']))),
                         Power=np.array(list(itertools.chain.from_iterable(data_copy['cell_info']['ss']))))

    PCI_name_list = ['PCI_{:d}'.format(k) for k in BS_LOCATIONS_DICT.keys()]
    if write == 'y':
        csv_feature.write(',Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI,' + ','.join(PCI_name_list) + '\n')
        csv_output.write(',SINR,RSRP,RSRQ,Power\n')
        for i in range(len(feature_subset['id'])):
            # print(type(feature_subset['PCI_' + str(3)]))
            which_PCI = [str(int(feature_subset['PCI_' + str(int(key))][i])) for key in BS_LOCATIONS_DICT_keys]
            # print(which_PCI)
            csv_feature.write('{x},{a},{b},{c},{d},{e},{f},{g},{lst}\n'.
                              format(x=i, a=feature_subset['Longitude'][i],
                                     b=feature_subset['Latitude'][i],
                                     c=feature_subset['Speed'][i],
                                     d=round(feature_subset['Distance'][i], 10),
                                     e=round(feature_subset['Distance_x'][i], 7),
                                     f=round(feature_subset['Distance_y'][i], 7),
                                     g=feature_subset['PCI'][i], lst=','.join(which_PCI)))
            csv_output.write('{x},{a},{b},{c},{d}\n'.
                             format(x=i, a=round(output_subset['SINR'][i], 7), b=round(output_subset['RSRP'][i], 7),
                                    c=round(output_subset['RSRQ'][i], 7), d=round(output_subset['Power'][i], 7)))

    return feature_subset, output_subset


def main():
    direc = '../data/data_05_08_23'
    dat = getData(direc)
    date_global = '05_08_23'
    # print(dat['location_wrt_BS']['d_latitude'])
    dest_name = '../formatted_data'
    # for i in range(len(dat['location_wrt_BS']['d_longitude'])):
    #     print(len(dat['location_wrt_BS']['d_longitude'][i]) - len(dat['distances_to_BS'][i]))
    # print(dat['cell_info']['sinr'])

    # print(len(writeData_formatted(dat, dest_dir=None, date=None)['Longitude']) - len(writeData_formatted(dat, dest_dir=None, date=None)['Distance_y']))
    # print(all(writeData_formatted(dat, dest_dir=None, date=None)['Longitude'] == 0))
    writeData_formatted(dat, dest_dir=dest_name, date=date_global)
    return


if __name__ == '__main__':
    main()
