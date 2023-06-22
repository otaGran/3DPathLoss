import subprocess
import concurrent
from concurrent.futures import wait
import os

BASE_PATH_BLENDER = '/res/'
BASE_PATH_SIONNA = '/Sionna_coverage_maps/coverage_maps_new/'
# BLENDER_PATH should be the path I built, since the things are enabled
# BLENDER_PATH = "/Users/zeyuli/blender-git/build_darwin/bin/Blender.app/Contents/MacOS/Blender"
# START_FROM_IDX = 512
NUM_OF_PROCESS = 8
EXTRA_HEIGHT = 2

if __name__ == '__main__':
    f_names_xml = [f for f in os.listdir(BASE_PATH_BLENDER + 'Bl_xml_files/') if os.path.isdir(f)]
    # f[0:-5] to remove the tiff
    f_names_sig_map = [f[0:-5] for f in os.listdir(BASE_PATH_SIONNA) if os.path.isfile(f)]
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_PROCESS) as executor:
        for idx, f_name_xml in enumerate(f_names_xml):
            if f_name_xml in f_names_sig_map:
                continue
            futures.append(executor.submit(subprocess.run,
                                           ['python', 'xml_to_heatmap_one_run.py',
                                            '--height_file', str(f_name_xml) + '.txt',
                                            '--extra_height', float(EXTRA_HEIGHT),
                                            '--cm_cell_size', float(10),
                                            '--BASE_PATH_BLENDER', str(BASE_PATH_BLENDER),
                                            'BASE_PATH_SIONNA', str(BASE_PATH_SIONNA)],
                                           capture_output=True, text=True))
    wait(futures)
