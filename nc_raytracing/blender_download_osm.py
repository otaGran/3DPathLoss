import requests
import uuid
from os.path import join, dirname
import os
from dotenv import load_dotenv

from blender_test_wraper import splitting_a_line


load_dotenv(join(dirname(__file__), '.env'))

BASE_PATH = os.environ.get('BASE_PATH')
# BLENDER_PATH should be the path I built, since the things are enabled
BLENDER_OSM_DOWNLOAD_PATH = os.environ.get('BLENDER_OSM_DOWNLOAD_PATH')
RES_FILE_NAME = os.environ.get('RES_FILE_NAME')
# IDX_STOP = 10

# http://tc319-srv1.egr.duke.edu:23412/api/map?bbox=-100.702392578125,25.30952262878418,-100.69202423095703,25.318172454833984
# this: -100.702389,25.318173,-100.692025,25.309523
# is:    minLonOut, maxLatOut, maxLonOut, minLatOut
# so, the server request follows minLon, minLat, maxLon, maxLat

f_ptr_res = open(BASE_PATH + 'res3_srv1_whole_us_filtered.txt', 'r')
lines = f_ptr_res.readlines()
f_ptr_res.close()

f_names_osm = [f for f in os.listdir(BLENDER_OSM_DOWNLOAD_PATH)
               if os.path.isfile(BLENDER_OSM_DOWNLOAD_PATH + f) and f != '.DS_Store']
# f is idx_uuid.osm, extract idx:
f_idx_osm = [int(f.split('_')[0]) for f in f_names_osm]
f_ptr_new = open(BASE_PATH + RES_FILE_NAME, 'a')

try:
    for idx, line in enumerate(lines):
        if idx in f_idx_osm:  # do not store duplicate indices
            continue
        minLon, maxLat, maxLon, minLat, perc = splitting_a_line(lll=line, uuid_incl='n')
        url = 'http://tc319-srv1.egr.duke.edu:23412/api/map?bbox={:f},{:f},{:f},{:f}'.format(minLon, minLat, maxLon, maxLat)
        response = requests.get(url)
        osm_text = response.text
        idx_uuid = str(idx) + '_' + str(uuid.uuid4())

        # save osm:
        f_ptr_osm = open(BLENDER_OSM_DOWNLOAD_PATH + idx_uuid + '.osm', 'w')
        f_ptr_osm.writelines(osm_text)
        f_ptr_osm.close()
        temp_arr = [minLon, maxLat, maxLon, minLat, perc, idx_uuid]
        temp_arr = [str(t) for t in temp_arr]
        # save line in res:
        f_ptr_new.write('(' + ','.join(temp_arr[0:4]) + '),' + ','.join(temp_arr[-2:]) + '\n')
        print('DONE', str(idx))
    f_ptr_new.close()
except KeyboardInterrupt:
    f_ptr_new.close()
    raise KeyboardInterrupt()
