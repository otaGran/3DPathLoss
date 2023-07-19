import subprocess
import concurrent
from concurrent.futures import wait
import os
from os.path import join, dirname
# import uuid
from dotenv import load_dotenv

load_dotenv(join(dirname(__file__), '.env'))
BASE_PATH = os.environ.get('BASE_PATH')
# BLENDER_PATH should be the path I built, since the things are enabled
BLENDER_PATH = os.environ.get('BLENDER_PATH')
BLENDER_COMMAND_LINE_PATH = os.environ.get('BLENDER_COMMAND_LINE_PATH')
BLENDER_OSM_DOWNLOAD_PATH = os.environ.get('BLENDER_OSM_DOWNLOAD_PATH')
START_FROM_IDX = 0
STOP_AT_IDX = -1
NUM_OF_PROCESS = 5
DECIMATE_FACTOR = 10
TERRAIN_OR_PLANE = 'plane'
RES_FILE_NAME = os.environ.get('RES_FILE_NAME')


def splitting_a_line(lll, uuid_incl='n'):
    lll = lll.replace('(', '')
    lll = lll.replace(')', '')
    lll = lll.replace('\n', '')
    lll = lll.split(',')
    # file format: (minLon,maxLat,maxLon,minLat),percent,idx_uuid
    if uuid_incl == 'y':
        minLon, maxLat, maxLon, minLat, perc, idx_uuid = [k for k in lll]
        return float(minLon), float(maxLat), float(maxLon), float(minLat), float(perc), idx_uuid
    else:
        minLon, maxLat, maxLon, minLat, perc = [float(k) for k in lll]
        return minLon, maxLat, maxLon, minLat, perc


if __name__ == '__main__':
    print(BASE_PATH)
    os.makedirs(BASE_PATH + 'height_at_origin/', exist_ok=True)
    os.makedirs(BASE_PATH + 'Bl_terrain_npy/', exist_ok=True)
    os.makedirs(BASE_PATH + 'Bl_building_npy/', exist_ok=True)
    os.makedirs(BASE_PATH + 'Bl_xml_files/', exist_ok=True)
    # f_names_xml = [f for f in os.listdir(BASE_PATH + 'Bl_xml_files/')
    #                 if os.path.isdir(BASE_PATH + 'Bl_xml_files/' + f) and f != '.DS_Store']
    with open(BASE_PATH + RES_FILE_NAME, 'r') as loc_fPtr:
        lines = loc_fPtr.readlines()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_PROCESS) as executor:
        for idx, line in enumerate(lines):
            if idx < START_FROM_IDX:
                continue
            if idx >= STOP_AT_IDX != -1:
                break
            # file format: (minLon,maxLat,maxLon,minLat),percent,idx_uuid\n
            minLonOut, maxLatOut, maxLonOut, minLatOut, percent, idx_uuid = splitting_a_line(lll=line, uuid_incl='y')
            print(' '.join([BLENDER_PATH,  # "--background",
                            "--python",
                            BLENDER_COMMAND_LINE_PATH.replace(' ', '\ '), "--",
                            "--idx", str(idx),
                            "--minLon", str(minLonOut),
                            "--maxLat", str(maxLatOut),
                            "--maxLon", str(maxLonOut),
                            "--minLat", str(minLatOut),
                            "--building_to_area_ratio", str(percent),
                            "--decimate_factor", str(DECIMATE_FACTOR),
                            "--BASE_PATH", str(BASE_PATH).replace(' ', '\ '),
                            "--BLENDER_OSM_DOWNLOAD_PATH", str(BLENDER_OSM_DOWNLOAD_PATH).replace(' ', '\ '),
                            "--idx_uuid", str(idx_uuid),
                            '--terrain_or_plane', TERRAIN_OR_PLANE]))
            futures.append(executor.submit(
                subprocess.run,
                [BLENDER_PATH, "--background",
                 "--python",
                 BLENDER_COMMAND_LINE_PATH, "--",
                 "--idx", str(idx),
                 "--minLon", str(minLonOut),
                 "--maxLat", str(maxLatOut),
                 "--maxLon", str(maxLonOut),
                 "--minLat", str(minLatOut),
                 "--building_to_area_ratio", str(percent),
                 "--decimate_factor", str(DECIMATE_FACTOR),
                 "--BASE_PATH", str(BASE_PATH),
                 "--BLENDER_OSM_DOWNLOAD_PATH", str(BLENDER_OSM_DOWNLOAD_PATH),
                 "--idx_uuid", str(idx_uuid),
                 '--terrain_or_plane', TERRAIN_OR_PLANE],
                capture_output=True, text=True))
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                data = future.result().stdout
                print('\n\n\n\n\n' + str(idx) + '\n' + data + '\n\n\n\n\n')
            except Exception as e:
                print(e)
        wait(futures)
