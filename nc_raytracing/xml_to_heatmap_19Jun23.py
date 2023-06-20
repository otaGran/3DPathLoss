"""
    This code is intended to be run on Tingjun Chen's Lab's server and not on my macbook.
"""
import os  # Configure which GPU
import time
import argparse

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
gpu_num = 0  # Use "" to use the CPU
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
        print(e)  # Avoid warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(1)  # Set global random seed for reproducibility

"""
    Read Height file
"""
f_Height = open('Blender_stuff/HeightAtOrigin.txt', 'r')
lines = f_Height.readlines()

"""
    Parse arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('start', nargs='?', const=0, type=int)
parser.add_argument('end', nargs='?', const=len(lines), type=int)
args = parser.parse_args()


def get_data_from_HeightFile(line):
    line = line.replace('(', '')
    line = line.replace(')', '')
    line = line.replace('\n', '')
    line = line.split(',')
    f_name = line[-1]
    line = line[0:-1]  # disregard FileName
    # file format: (minLon, maxLat, maxLon, minLat), building_to_area ratio, height, name
    minLonOut, maxLatOut, maxLonOut, minLatOut, percent, height = [float(l) for l in line]
    return minLonOut, maxLatOut, maxLonOut, minLatOut, percent, height, f_name


def cm_routine(extra_height, outer_lines, outer_idx):
    _, _, _, _, _, height_at_origin, file_name = get_data_from_HeightFile(outer_lines[outer_idx])
    scene = load_scene('Blender_stuff/Blender_xml_files/' + file_name + '/' + file_name + '.xml')
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
                     position=[0, 0, height_at_origin + 5],
                     orientation=[0, 0, 0])
    scene.add(tx)

    scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

    # Compute coverage map
    cm = scene.coverage_map(max_depth=8, cm_center=[0, 0, height_at_origin + extraHeight], cm_orientation=[0, 0, 0],
                            cm_size=[1000, 1000])
    # Visualize coverage in preview
    # scene.preview(coverage_map=cm, resolution=[1000, 600])
    return cm, scene, file_name


def save_routine(cm, img_path):
    cm_tensor = cm.as_tensor()
    cm_2D = cm_tensor.numpy()[0, :, :]
    cm_2D[cm_2D == 0] = np.nan
    cm_2D_dB = 10 * np.log10(cm_2D)
    cm_img = Image.fromarray(cm_2D_dB)
    cm_img.save(img_path)
    return


extraHeight = 2  # additional height to add to height above ground

"""
    Running the routine for producing signal strength heat maps
"""

for idx in range(args.start, args.end):
    coverage_map, scene, fName = cm_routine(extra_height=extraHeight, outer_lines=lines, outer_idx=idx)
    print(fName)
    image_path = 'Sionna_coverage_maps/' + fName + '.tiff'
    save_routine(coverage_map, image_path)