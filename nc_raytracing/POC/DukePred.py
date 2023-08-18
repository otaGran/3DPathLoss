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
        
# Import Sionna RT components



scene = load_scene("/home/yl826/3DPathLoss/nc_raytracing/POC/res_plane_duke/Bl_xml_files/0_78d03890-be26-47f8-8b81-8485f0d19a83//0_78d03890-be26-47f8-8b81-8485f0d19a83.xml")


scene.tx_array = PlanarArray(num_rows=1,
                      num_cols=1,
                      vertical_spacing=0.5,
                      horizontal_spacing=0.5,
                      pattern="iso",
                      polarization="VH")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                      num_cols=1,
                      vertical_spacing=0.5,
                      horizontal_spacing=0.5,
                      pattern="iso",
                      polarization="cross")





try:
    scene.remove("tx")
except Exception as e:
    print(e)
    pass

# Add a transmitter
tx = Transmitter(name="tx",
              position=(-250.33,248.42,18),
              orientation=[-110,0, 0])
scene.add(tx)

scene.frequency = 3.64e9 # in Hz; implicitly updates RadioMaterials
# #scene.synthetic_array = True 
# # If set to False, ray tracing will be done per antenna element (slower for large arrays)

# # Compute coverage map


cm = scene.coverage_map(max_depth=8, cm_center=[-224,224, 2], 
                        cm_orientation=[0, 0, 0],
                        cm_cell_size=[4, 4], 
                        cm_size=[512, 512], los=True, reflection=True, diffraction=True, 
                        num_samples=6e7)


cm_tensor = cm.as_tensor()
cm_2D = cm_tensor.numpy()[0, :, :]
cm_2D = np.flip(cm_2D,0)
np.save("Duke_Pred_cm.npy",cm_2D)
