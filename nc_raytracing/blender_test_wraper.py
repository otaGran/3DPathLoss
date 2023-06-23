import subprocess
import concurrent
from concurrent.futures import wait
import os

BASE_PATH = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/res/'
# BLENDER_PATH should be the path I built, since the things are enabled
BLENDER_PATH = "/Users/zeyuli/blender-git/build_darwin/bin/Blender.app/Contents/MacOS/Blender"
BLENDER_COMMAND_LINE_PATH = "/Users/zeyuli/Desktop/Duke/0. Su23_Research/3DPathLoss/nc_raytracing/blender_test_command_line.py"
START_FROM_IDX = 512
NUM_OF_PROCESS = 8
DECIMATE_FACTOR = 1


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


if __name__ == '__main__':
    f_names_xml = [f for f in os.listdir(BASE_PATH + 'Bl_xml_files/')
                   if os.path.isdir(BASE_PATH + 'Bl_xml_files/' + f)]
    with open(BASE_PATH + 'res3_srv1_whole_us_filtered.txt', 'r') as loc_fPtr:
        lines = loc_fPtr.readlines()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_PROCESS) as executor:
        for idx, line in enumerate(lines):
            if idx < START_FROM_IDX:
                continue
            print(idx)
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace('\n', '')
            line = line.split(',')
            # file format: minLon, maxLat, maxLon, minLat
            minLonOut, maxLatOut, maxLonOut, minLatOut, percent = [float(l) for l in line]
            # print('\n\nIndex, starting run seq: ' + str(idx) + '\n\n')
            # inLonOut, maxLatOut, maxLonOut, minLatOut, percent)
            futures.append(executor.submit(
                subprocess.run,
                [BLENDER_PATH, "--background",
                 "--python",
                 BLENDER_COMMAND_LINE_PATH, "--",
                 "--idx", str(idx),
                 "--minLon", str(minLonOut),
                 "--maxLat", str(maxLatOut),
                 "--maxLon", str(maxLonOut),
                 "--minLat",str(minLatOut),
                 "--building_to_area_ratio", str(percent),
                 "--decimate_factor", str(DECIMATE_FACTOR),
                 "--BASE_PATH", str(BASE_PATH)],
                capture_output=True, text=True))
            # if len(futures) % NUM_OF_PROCESS == 0:
            #     wait(futures)
            #     #
            #     s = str(futures[-1].result()).replace('\\n','\n')
            #     print(s)
            #     s = str(futures[-2].result()).replace('\\n','\n')
            #     print(s)
            -100.702389, 25.318173, -100.692025, 25.309523
            # print(futures[-3].result())

            # pbar.updae(1)
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
