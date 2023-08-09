import subprocess
import concurrent
from concurrent.futures import wait, as_completed
import os
import time
import multiprocessing
from PIL import Image
#import tensorflow as tf
import numpy as np
from tqdm import tqdm
# Import Sionna RT components
#from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, Paths2CIR

# testing new Blender_command_line function written on 23. Jun 2023
BASE_PATH_BLENDER = '/dev/shm/res_plane/'
#BASE_PATH_SIONNA = '/dev/shm/coverage_maps_data_aug_Jul20_1000/'
BASE_PATH_SIONNA = '/home/yl826/3DPathLoss/nc_raytracing/cm_200_Aug3/'
# BASE_PATH_SIONNA = '/home/yl826/3DPathLoss/nc_raytracing/Sionna_coverage_maps/coverage_maps_plane_missing_Jul6/'
# un-comment 
# BASE_PATH_BLENDER = 'res/res_23Jun23/'
# BASE_PATH_SIONNA = 'Sionna_coverage_maps/coverage_maps_new_22Jun23/'
# START_FROM_IDX = 512
STOP_AT_IDX = 100


NUM_OF_PROCESS = 2
EXTRA_HEIGHT = 2


def initializer_func(gpu_seq_queue: multiprocessing.Queue, log_level: int) -> None:
    """
    This is a initializer function run after the creation of each process in ProcessPoolExecutor, 
    to set the os env variable to limit the visiablity of GPU for each process inorder to achieve 
    the load balance bewteen diff GPU
    :gpu_seq_queue This is a queue storing the GPU ID as a token, each process will only get 
    """
    import os
    
    gpu_id = gpu_seq_queue.get()
    print("Initlizing the process: %d with GPU: %d"%(os.getpid(),gpu_id))
    
    # Configure which GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    
if __name__ == '__main__':
    
    f_names_xml = [f for f in os.listdir(BASE_PATH_BLENDER + 'Bl_xml_files/')
                   if os.path.isdir(BASE_PATH_BLENDER + 'Bl_xml_files/' + f)]
    print('Number of xml files:', len(f_names_xml))
    
    
    # f[0:-4] to remove the ".npy" from file name
    f_names_sig_map = [f[0:-4] for f in os.listdir(BASE_PATH_SIONNA)
                       if os.path.isfile(BASE_PATH_SIONNA + f)]
    #print('Number of finished signal maps:', len(f_names_sig_map))
    futures = []
    #print(f_names_sig_map)
    
    # Create a GPU ID token Queue
    gpu_seq_queue = multiprocessing.Queue()
    NUMBER_OF_GPU = 2
    for i in range(NUM_OF_PROCESS):
        gpu_seq_queue.put(i%NUMBER_OF_GPU)
    
    # Init pbar
    pbar = tqdm(total=len(f_names_xml), desc='xml_to_heatmap')
    count = 0
    # Init process pool executor
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_PROCESS, initializer=initializer_func, initargs=(gpu_seq_queue,1)) as executor:
        for idx, f_name_xml in enumerate(f_names_xml):
            
            if STOP_AT_IDX < idx: 
                break
            
            if f_name_xml not in f_names_sig_map:  # skip cmaps that have already been generated
                print("yes")
                """
                Creating a subprocess for each job running in process pool
                This is the simples way I can find to free up the GPU memory and do a load balance betwenn each GPU
                """
                futures.append(executor.submit(subprocess.run,
                                                ['python', 'xml_to_heatmap_one_run.py',
                                                 '--file_name_wo_type', str(f_name_xml),
                                                 '--extra_height', str(EXTRA_HEIGHT),
                                                 '--cm_cell_size', str(5),
                                                 '--BASE_PATH_BLENDER', str(BASE_PATH_BLENDER),
                                                 '--BASE_PATH_SIONNA', str(BASE_PATH_SIONNA),
                                                 '--outer_idx', str(idx)],
                                                 capture_output=True, text=True))

            
                print(' '.join(['python', 'xml_to_heatmap_one_run.py',
                                '--file_name_wo_type', str(f_name_xml),
                                '--extra_height', str(EXTRA_HEIGHT),
                                '--cm_cell_size', str(5),
                                '--BASE_PATH_BLENDER', str(BASE_PATH_BLENDER).replace(' ', '\ '),
                                '--BASE_PATH_SIONNA', str(BASE_PATH_SIONNA).replace(' ', '\ '),
                                '--outer_idx', str(idx)]))
    
        for idx, future in enumerate(as_completed(futures)):
            pbar.update(n=1)  
            try:
                data = str(future.result()).replace('\\n','\n')
                print('\n\n\n\n\n' + str(idx) + '\n' + data + '\n\n\n\n\n')
            except Exception as err:
                print(err)
    print('DONE')
    
