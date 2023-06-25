import os # Configure which GPU
import time
import argparse
import traceback
from datetime import datetime

from PIL import Image
import tensorflow as tf
import numpy as np

# Import Sionna
import sionna

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, Paths2CIR


"""
    Setting up the environment, including the GPUs
"""
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e) # Avoid warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(1) # Set global random seed for reproducibility

"""
    Read Height file
"""
f_Height = open('Blender_stuff/HeightAtOrigin.txt', 'r')
lines = f_Height.readlines()

"""
    Parse arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('start', nargs='?', default=0, type=int)
parser.add_argument('end', nargs='?', default=len(lines), type=int)
args = parser.parse_args()
print(args.start, args.end)

# log file
f_log = open('Sionna_coverage_maps/logs/siona_cm_log.txt', 'a')
# file storing successful runs
f_run = open('Sionna_coverage_maps/logs/siona_cm_success_rans.txt', 'w')


def include_existing_tiff(f_ptr=f_run, directory='Sionna_coverage_maps/coverage_maps/'):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        print(len(files))
        for f in files:
            f_ptr.write(f + '\n')
            print(f)
        return
    except Exception as e:
        raise e


def get_data_from_HeightFile(line):
    try:
        line = line.replace('(', '')
        line = line.replace(')', '')
        line = line.replace('\n', '')
        line = line.split(',')
        f_name = line[-1]
        line = line[0:-1]  # disregard FileName
        # file format: (minLon, maxLat, maxLon, minLat), building_to_area ratio, height, name
        minLonOut, maxLatOut, maxLonOut, minLatOut, percent, height = [float(l) for l in line]
        return minLonOut, maxLatOut, maxLonOut, minLatOut, percent, height, f_name
    except Exception as e:
        raise e


def cm_routine(extra_height, outer_lines, outer_idx):
    try:
        _,_,_,_,_, height_at_origin, file_name = get_data_from_HeightFile(outer_lines[outer_idx])
        start_loc = time.time()
        scene = load_scene('Blender_stuff/Blender_xml_files/' + file_name + '/' + file_name + '.xml')
        print('load scene time: ', str(time.time() - start_loc))
        scene.tx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="tr38901",
                                  polarization="VH")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="iso",
                                  polarization="cross")
        # Add a transmitter
        tx = Transmitter(name="tx",
                      position=[0,0,height_at_origin+5],
                      orientation=[0,0,0])
        scene.add(tx)

        scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
        scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

        # Compute coverage map
        start_loc = time.time()
        cm = scene.coverage_map(max_depth=8, cm_center=[0, 0, height_at_origin+extraHeight], cm_orientation=[0,0,0], 
                                cm_size=[1000,1000])
        print('compute cm time: ', str(time.time() - start_loc))
        # Visualize coverage in preview
        # scene.preview(coverage_map=cm, resolution=[1000, 600])
        return cm, scene, file_name
    except Exception as e:
        raise e


def save_routine(cm, img_path):
    try:
        cm_tensor = cm.as_tensor()
        cm_2D = cm_tensor.numpy()[0, :, :]
        cm_2D[cm_2D == 0] = np.nan
        cm_2D_dB = 10*np.log10(cm_2D)
        cm_img = Image.fromarray(cm_2D_dB)
        cm_img.save(img_path)
        return
    except Exception as e:
        raise e


def run_routine():
    try:
        print('Calculating for index:', str(idx))
        coverage_map, scene, fName = cm_routine(extra_height=extraHeight, outer_lines=lines, outer_idx=idx)
        print(fName)
        image_path = 'Sionna_coverage_maps/coverage_maps/' + fName + '.tiff'
        save_routine(coverage_map, image_path)
        print('Cumulative time expended:', str(time.time() - start) + ' seconds\n\n')
        return
    except Exception as e:
        raise e


extraHeight = 2  # additional height to add to height above ground
include_existing_tiff()

"""
    Running the routine for producing signal strength heat maps
"""

start = time.time()
f_log.write('\n\n\n-------\nRun started at: ' + str(datetime.now()) + '\n')
for idx in range(args.start, args.end):
    try:
        run_routine()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        break
    except Exception as e:
        data_out = get_data_from_HeightFile(line=lines[idx])
        f_log.write('\n' + str(data_out[-1]) + '\n' +  traceback.format_exc())
        print(e)

print('\nEXIT FROM SCRIPT\n\n\n\n')
include_existing_tiff()
f_run.close()
f_log.close()
