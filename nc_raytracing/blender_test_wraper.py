import subprocess
import concurrent
from concurrent.futures import wait
import os
from os.path import join, dirname
# import uuid
from dotenv import load_dotenv

load_dotenv(join(dirname(__file__), '.env'))
print(dirname(__file__))
BASE_PATH = os.environ.get('BASE_PATH')
# BLENDER_PATH should be the path I built, since the things are enabled
BLENDER_PATH = os.environ.get('BLENDER_PATH')
BLENDER_COMMAND_LINE_PATH = os.environ.get('BLENDER_COMMAND_LINE_PATH')
BLENDER_OSM_DOWNLOAD_PATH = os.environ.get('BLENDER_OSM_DOWNLOAD_PATH')
START_FROM_IDX = 0
NUM_OF_PROCESS = 8
DECIMATE_FACTOR = 1
RES_FILE_NAME = os.environ.get('RES_FILE_NAME')


def blender_to_xml(minLonOut, maxLatOut, maxLonOut, minLatOut, percent):
    try:
        # run(maxLonOut, minLonOut, maxLatOut, minLatOut, buildingToAreaRatio=percent, run_idx=idx)
        result = subprocess.run(
            [BLENDER_PATH, "--background", "--python",
             "/Users/test/PycharmProjects/3DPathLoss/nc_raytracing/blender_test_command_line.py", "--", "--idx",
             str(idx),
             "--minLon", str(minLonOut), "--maxLat", str(maxLatOut), "--maxLon", str(maxLonOut), "--minLat",
             str(minLatOut),
             "--building_to_area_ratio", str(percent)],
            capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        # f_ptr_error_Exception.write(traceback.format_exc())
        # f_ptr_error_IdxAndPercentBuildings.write(str(idx) + ',' + str(percent) + '\n')
        print(e)


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
    f_names_xml = [f for f in os.listdir(BASE_PATH + 'Bl_xml_files/')
                   if os.path.isdir(BASE_PATH + 'Bl_xml_files/' + f)]
    
    with open(BASE_PATH + RES_FILE_NAME, 'r') as loc_fPtr:
        lines = loc_fPtr.readlines()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_PROCESS) as executor:
        for idx, line in enumerate(lines):
            if idx < START_FROM_IDX:
                continue
            # print(idx)
            # file format: minLon, maxLat, maxLon, minLat, percent, idx_uuid
            minLonOut, maxLatOut, maxLonOut, minLatOut, percent, idx_uuid = splitting_a_line(lll=line, uuid_incl='y')
            #if idx_uuid in f_names_xml:
                #continue
            futures.append(executor.submit(
                subprocess.run,
                [BLENDER_PATH, "--background",
                 "--python",
                 BLENDER_COMMAND_LINE_PATH, "--",
                 "--cycles-device","CPU",
                 "--idx", str(idx),
                 "--minLon", str(minLonOut),
                 "--maxLat", str(maxLatOut),
                 "--maxLon", str(maxLonOut),
                 "--minLat", str(minLatOut),
                 "--building_to_area_ratio", str(percent),
                 "--decimate_factor", str(DECIMATE_FACTOR),
                 "--BASE_PATH", str(BASE_PATH),
                 "--BLENDER_OSM_DOWNLOAD_PATH", str(BLENDER_OSM_DOWNLOAD_PATH),
                 "--idx_uuid", str(idx_uuid)],
                capture_output=True, text=True))
            # if len(futures) % NUM_OF_PROCESS == 0:
            #     wait(futures)
            #     #
            #     s = str(futures[-1].result()).replace('\\n','\n')
            #     print(s)
            #     s = str(futures[-2].result()).replace('\\n','\n')
            #     print(s)
            # -100.702389, 25.318173, -100.692025, 25.309523
            # print(futures[-3].result())
            if idx > 5:
                break
            # pbar.updae(1)
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                data = future.result().stdout
                print('\n\n\n\n\n' + str(idx) + '\n' + data + '\n\n\n\n\n')
            except Exception as e:
                print(e)
        wait(futures)

# print('number of files with bulding to area ratio > ' + str(percent_threshold), count)
# print('percentage of files with bulding to area ratio > ' + str(percent_threshold), count / len(lines))

# exit_routine()
# print('\nDONE\n\n\n\n\n')
#
# except KeyboardInterrupt:
# f_ptr_error_Exception.write(str(datetime.now()) + ': Keyboard Interrupt\n')
# exit_routine()
# print('\nEXIT FROM SCRIPT\n\n\n\n')
