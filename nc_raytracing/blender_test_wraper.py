import subprocess
import concurrent
from concurrent.futures import wait

BASE_PATH = '/Users/test/PycharmProjects/3DPathLoss/nc_raytracing/res/'
START_FROM_IDX = 512
NUM_OF_PROCESS = 8


def blender_to_xml(minLonOut, maxLatOut, maxLonOut, minLatOut, percent):
    try:
        # run(maxLonOut, minLonOut, maxLatOut, minLatOut, buildingToAreaRatio=percent, run_idx=idx)
        result = subprocess.run(
            ["/Users/test/blender-git/build_darwin/bin/Blender.app/Contents/MacOS/Blender", "--background", "--python",
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
            futures.append(executor.submit(subprocess.run,
                ["/Users/test/blender-git/build_darwin/bin/Blender.app/Contents/MacOS/Blender", "--background",
                 "--python",
                 "/Users/test/PycharmProjects/3DPathLoss/nc_raytracing/blender_test_command_line.py", "--", "--idx",
                 str(idx),
                 "--minLon", str(minLonOut), "--maxLat", str(maxLatOut), "--maxLon", str(maxLonOut), "--minLat",
                 str(minLatOut),
                 "--building_to_area_ratio", str(percent)],
                capture_output=True, text=True))
            # if len(futures) % NUM_OF_PROCESS == 0:
            #     wait(futures)
            #     #
            #     s = str(futures[-1].result()).replace('\\n','\n')
            #     print(s)
            #     s = str(futures[-2].result()).replace('\\n','\n')
            #     print(s)

                # print(futures[-3].result())

            # pbar.update(1)

# print('number of files with bulding to area ratio > ' + str(percent_threshold), count)
# print('percentage of files with bulding to area ratio > ' + str(percent_threshold), count / len(lines))

# exit_routine()
# print('\nDONE\n\n\n\n\n')
#
# except KeyboardInterrupt:
# f_ptr_error_Exception.write(str(datetime.now()) + ': Keyboard Interrupt\n')
# exit_routine()
# print('\nEXIT FROM SCRIPT\n\n\n\n')
