import os # Configure which GPU

import time
import argparse
import traceback
from datetime import datetime

from PIL import Image
import tensorflow as tf
import numpy as np

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

f_name_xml = '6182_40c6a006-c2ce-4261-afee-72dd36fd0262'  # should be idx_uuid

CM_RESOLUTION = 4  # size in meters of each pixel in the coverage map
ANTENNA_PATTERN = 'iso'  # 'tr38901' also possible

# BASE_PATH = 'res_plane/'
BASE_PATH = '/path/to/res_plane/'  # path to res folder, structure should be
# res_folder_git_issue/
#     Bl_building_npy/
#     Bl_terrain_npy/  # should be empty
#     Bl_xml_files/
#     height_at_origin/   # should be empty


BASE_PATH_SIONNA = '/path/to/coverage_map/' 
# BASE_PATH_SIONNA = '/home/yl826/3DPathLoss/nc_raytracing/xml_to_heatmap/git_test/' 
# path to folder that stores coverage_maps
os.makedirs(BASE_PATH_SIONNA, exist_ok=True)


file_name_wo_type = f_name_xml  # idx_uuid, e.g. 6182_40c6a006-c2ce-4261-afee-72dd36fd0262

extra_height = 2  # height of Rx above ground. 

cm_cell_size = 4  # size in meters of each pixel in the coverage map

outer_idx = 0  # this arg doesn't affect anything

cm_num_samples = int(7e6)  # input for the num_samples in scene.coverage_map (sionna method), default 2e6

antenna_pattern = 'iso'



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

tf.random.set_seed(1) # Set global random seed for reproducibility


def generate_coverage_map_config_combination(file_path):

    building_npy = np.load(file_path)
    tx_xy_position = [[-244,244], [-244,-244], [244,244], [244,-244]]
    tx_height = [2]
    
    cm_conf_dict = []
    for cm_conf in tx_xy_position:
        x = cm_conf[0] + 500
        y = cm_conf[1] + 500
        
        x = int(x)
        y = int(y)
        
        # Obtain the max height of the area
        building_height_at_xy_position = np.max(building_npy[x-512:x+512,y-512:y+512])
        
        for height in tx_height:
            cm_conf_dict.append([*cm_conf,height+ building_height_at_xy_position])
    print(cm_conf_dict)
    return cm_conf_dict


def cm_routine(extra_height):
    try:
        file_name = file_name_wo_type
        print("file name", file_name)
        cm_conf_dict = generate_coverage_map_config_combination(
            BASE_PATH + 'Bl_building_npy/' + file_name+'.npy')
        exist = True
        for cm_conf in cm_conf_dict:
            image_path = BASE_PATH_SIONNA + file_name
            for p in cm_conf:
                image_path = image_path+"_"+str(p)
            if not os.path.isfile(image_path+".npy"):
                exist = False
                break
        if exist:
            return
                
        start_loc = time.time()
        scene = load_scene(BASE_PATH + 'Bl_xml_files/' + file_name + '/' + file_name + '.xml')
        print('load scene time: ', str(time.time() - start_loc))
        
        scene.tx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern=antenna_pattern,
                                  polarization="VH")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="iso",
                                  polarization="cross")
        
        for cm_conf in cm_conf_dict:
            image_path = BASE_PATH_SIONNA + file_name
            for p in cm_conf:
                image_path = image_path+"_"+str(p)
            if os.path.isfile(image_path+".npy"):
                print("Skipping already exist files")
                continue
            start_loc = time.time()
            try:
                scene.remove("tx")
            except Exception as e:
                print(e)
                pass
            
            # Add a transmitter
            tx = Transmitter(name="tx",
                          position=cm_conf,
                          orientation=[0, 0, 0])
            scene.add(tx)

            scene.frequency = 3.66e9 # in Hz; implicitly updates RadioMaterials
            
            # Compute coverage map
            cm_only_start = time.time()
            cm = scene.coverage_map(max_depth=8, cm_center=[cm_conf[0], cm_conf[1], extra_height], 
                                    cm_orientation=[0, 0, 0],
                                    cm_cell_size=[cm_cell_size, cm_cell_size], 
                                    cm_size=[512, 512], los=True, reflection=True, diffraction=True, 
                                    num_samples=cm_num_samples)
            
            print('compute cm only time: ', str(time.time() - cm_only_start))
                
            print("images path",image_path)
            
            cm_tensor = cm.as_tensor()
            cm_2D = cm_tensor.numpy()[0, :, :]
            # saving as db and do a flip with axis 0
            np.save(image_path, np.flip(cm_2D,0))
            
            
            print('compute cm whole time: ', str(time.time() - start_loc))
    except Exception as e:
        raise e


try:
   
    cm_routine(extra_height=extra_height)
    
    print('index ' + str(outer_idx), file_name_wo_type + ' DONE\n')
except Exception as e:
    print(e)
    raise e