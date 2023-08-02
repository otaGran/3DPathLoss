import subprocess
import concurrent
import os
import time
import traceback

import numpy as np

from tqdm import tqdm
import multiprocessing
from concurrent.futures import wait
from threading import Thread
from multiprocessing import Process, Queue


def sionna_run(height_file, extra_height, n_samples, BASE_PATH_BLENDER, BASE_PATH_SPARSE, outer_idx, queue, gpu_id):
    """
        Base path for the Blender files. The height files, pngs, and xml folders should be in BASE_PATH.
        Organisation: BASE_PATH/Bl_terrain_img/, BASE_PATH/Bl_xml_files/, BASE_PATH/height_at_origin/,
    """

    def get_data_from_HeightFile(f_ptr_height):
        try:
            lines = f_ptr_height.readlines()
            line = lines[0]
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

    def raytrace_routine(extra_height, f_ptr_height, n_samples):
        try:
            _, _, _, _, _, height_at_origin, file_name = get_data_from_HeightFile(f_ptr_height=f_ptr_height)
            start_loc = time.time()
            scene = load_scene(BASE_PATH + 'Bl_xml_files/' + file_name + '/' + file_name + '.xml')
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
                             position=[0, 0, height_at_origin + extra_height],
                             orientation=[0, 0, 0])
            scene.add(tx)

            scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials
            scene.synthetic_array = True

            xy_min, xy_max = [-500, -500], [500, 500]
            locs = np.round(np.random.uniform(low=xy_min, high=xy_max, size=(n_samples, 2)))
            ret = np.zeros((1000, 1000))
            ret.fill(np.nan)
            for inner_idx, xy in enumerate(locs):
                rx = Receiver(name="rx" + str(inner_idx),
                              position=[xy[0], xy[1], height_at_origin + extra_height],
                              orientation=[0, 0, 0])
                scene.add(rx)
                currrr = time.time()
                paths = scene.compute_paths(max_depth=6,
                                            method="stochastic",
                                            num_samples=5e6,
                                            seed=1)
                print('raytracing time consumption: ' + str(time.time() - currrr))
                p2c = Paths2CIR(sampling_frequency=15e3, scene=scene)  # path to channel impulse response

                a, tau = p2c(
                    paths.as_tuple())  # a contains some stuff (path coefficients) that eventually becomes power
                ret[int(xy[0]), int(xy[1])] = np.sum(np.squeeze(np.abs(a)))
                scene.remove("rx" + str(inner_idx))

            start_loc = time.time()
            return ret, scene, file_name
        except Exception as e:
            raise e

    def save_routine(cm, img_path):
        try:
            with warnings.catch_warnings(record=True) as _:
                cm_2D_dB = 10. * np.log10(cm)
            # cm_img = Image.fromarray(cm_2D_dB)
            np.save(img_path, cm_2D_dB)
            return
        except Exception as e:
            raise e

    start = time.time()

    try:
        import os
        import sys
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            # tf.config.experimental.set_memory_growth(gpus[1], True)
        except RuntimeError as e:
            print(e)  # Avoid warnings from TensorFlow
        import warnings
        import sionna
        # Import Sionna RT components
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, Paths2CIR

        # For link-level simulations
        from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, \
            CIRDataset
        from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
        from sionna.utils import compute_ber, ebnodb2no, PlotBER
        from sionna.ofdm import KBestDetector, LinearDetector
        from sionna.mimo import StreamManagement

        BASE_PATH = BASE_PATH_BLENDER
        BASE_PATH_SPARSE = BASE_PATH_SPARSE
        f_ptr_H = open(BASE_PATH + 'height_at_origin/' + height_file, 'r')
        pid = os.getpid()
        """
            Setting up the environment, including the GPUs
        """
        gpu_num = 0  # Use "" to use the CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        gpus = tf.config.list_physical_devices('GPU')

        tf.random.set_seed(1)  # Set global random seed for reproducibility
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # print('Calculating for index:', str(idx))

        sparse, scene, fName = raytrace_routine(extra_height=extra_height, f_ptr_height=f_ptr_H,
                                                      n_samples=n_samples)

        # print(fName)
        image_path = BASE_PATH_SPARSE + fName + '.npy'
        save_routine(sparse, image_path)
        tf.keras.backend.clear_session()
        print('Cumulative time expended:', str(time.time() - start) + ' seconds\n\n')
        print('index ' + str(outer_idx), height_file + ' DONE\n')
        queue.put(1)
        sys.exit()
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    return


def consumer(queue, tqdm_size):
    pabar2 = tqdm(total=tqdm_size, position=0, desc="Saving", leave=True)
    while True:
        # get a unit of work
        item = queue.get(block=True)

        pabar2.update(1)
        # check for stop
        if item is None:
            break
    return


def wrap_again(h1, h2, h3, h4, h5, h6, h7, h8):
    print("yes")
    sub_new_process = Process(target=sionna_run, args=(h1,
                                                       h2,
                                                       h3,
                                                       h4,
                                                       h5,
                                                       h6, h7, h8))
    sub_new_process.start()
    sub_new_process.join()
    return


if __name__ == '__main__':
    print("startstartstartstartstartstartstartstartstartstart")
    # testing new Blender_command_line function written on 23. Jun 2023
    BASE_PATH_BLENDER = 'res_0626_without_nonetype_fix/'
    BASE_PATH_SPARSE = 'Sionna_coverage_maps/coverage_sparse/'
    os.makedirs(BASE_PATH_SPARSE, exist_ok=True)

    NUM_OF_PROCESS = 10
    EXTRA_HEIGHT = 2
    NUM_OF_SAMPLES = 20  # number of samples in the sparse heat map
    f_names_xml = [f for f in os.listdir(BASE_PATH_BLENDER + 'Bl_xml_files/')
                   if os.path.isdir(BASE_PATH_BLENDER + 'Bl_xml_files/' + f)]
    print('Number of xml files:', len(f_names_xml))
    # f[0:-5] to remove the tiff
    f_names_sig_map = [f[0:-5] for f in os.listdir(BASE_PATH_SPARSE)
                       if os.path.isfile(BASE_PATH_SPARSE + f)]
    print('Number of finished signal maps:', len(f_names_sig_map))
    futures = []
    
    try:
        m = multiprocessing.Manager()
        queue = m.Queue()
        consumer_process = Thread(target=consumer, args=(queue, len(f_names_xml)))
        consumer_process.start()

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=NUM_OF_PROCESS) as executor, concurrent.futures.ProcessPoolExecutor(
                max_workers=NUM_OF_PROCESS) as executor2:
            for idx, f_name_xml in enumerate(f_names_xml[0: int(len(f_names_xml) / 2)]):
                if idx % 2 == 0:
                    # Running as a function in this file.
                    futures.append(executor.submit(wrap_again,
                                                   str(f_name_xml) + '.txt',
                                                   EXTRA_HEIGHT,
                                                   NUM_OF_SAMPLES,
                                                   BASE_PATH_BLENDER,
                                                   BASE_PATH_SPARSE,
                                                   idx, queue, 0))
                    break
                else:
                    futures.append(executor.submit(wrap_again,
                                                   str(f_name_xml) + '.txt',
                                                   EXTRA_HEIGHT,
                                                   NUM_OF_SAMPLES,
                                                   BASE_PATH_BLENDER,
                                                   BASE_PATH_SPARSE,
                                                   idx, queue, 0))

    except KeyboardInterrupt:
        for job in futures:
            job.cancel()
    finally:
        wait(futures)
        queue.put(None)
        print("Done")
        # consumer_process.join()