from blender_download_osm import splitting_a_line

import requests
import uuid
from os.path import join, dirname
import os
from dotenv import load_dotenv
import numpy as np


def download_from_tc319(blender_osm_download_path, line, f_ptr_save):
    # only works with step1.5.txt files that already have idx_uuid written in
    minLon, maxLat, maxLon, minLat, perc, idx_uuid = splitting_a_line(lll=line, uuid_incl='y')

    url = 'http://tc319-srv1.egr.duke.edu:23412/api/map?bbox={:f},{:f},{:f},{:f}'.format(minLon, minLat, maxLon, maxLat)
    response = requests.get(url)
    osm_text = response.text

    # save osm:
    f_ptr_osm = open(blender_osm_download_path + idx_uuid + '.osm', 'w')
    f_ptr_osm.writelines(osm_text)
    f_ptr_osm.close()
    temp_arr = [minLon, maxLat, maxLon, minLat, perc, idx_uuid]
    temp_arr = [str(t) for t in temp_arr]
    # save line in res:
    f_ptr_save.write('(' + ','.join(temp_arr[0:4]) + '),' + ','.join(temp_arr[-2:]) + '\n')
    f_ptr_save.close()
    return


if __name__ == '__main__':
    load_dotenv(join(dirname(__file__), '..', '.env'))

    BASE_PATH = os.environ.get('BASE_PATH')
    BLENDER_OSM_DOWNLOAD_PATH = os.environ.get('BLENDER_OSM_DOWNLOAD_PATH')
    RES_FILE_NAME = os.environ.get('RES_FILE_NAME')

    f_ptr_res = open(BASE_PATH + 'step1.5.txt', 'r')
    lines = f_ptr_res.readlines()
    f_ptr_res.close()

    for idx, lll in enumerate(lines):
        download_from_tc319(BLENDER_OSM_DOWNLOAD_PATH, line=lll, f_ptr_save=open(join(BASE_PATH, RES_FILE_NAME), 'a'))
