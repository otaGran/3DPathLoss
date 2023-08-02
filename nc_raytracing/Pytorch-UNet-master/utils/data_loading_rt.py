import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm





from scipy.constants import speed_of_light
import numpy as np
import matplotlib.pyplot as plt


import math
import os

from scipy import ndimage


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.tiff':
        return np.array(Image.open(filename))
    else:
        return Image.open(filename)


#
# def unique_mask_values(idx, mask_dir, mask_suffix):
#     mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
#     mask = np.asarray(load_image(mask_file))
#     if mask.ndim == 2:
#         return np.unique(mask)
#     elif mask.ndim == 3:
#         mask = mask.reshape(-1, mask.shape[-1])
#         return np.unique(mask, axis=0)
#     else:
#         raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
NUM_OF_POINTS = 200

class RTDataset(Dataset):
    def __init__(self, building_height_map_dir: str, terrain_height_map_dir: str,
                 ground_truth_signal_strength_map_dir: str,sparse_ss_dir: str, scale: float = 1.0, mask_suffix: str = '', pathloss: bool = False, median_filter_size: int = 0):
        np.seterr(divide = 'ignore') 
        self.building_height_map_dir = Path(building_height_map_dir)
        self.terrain_height_map_dir = Path(terrain_height_map_dir)
        self.ground_truth_signal_strength_map_dir = Path(ground_truth_signal_strength_map_dir)
        
        self.sparse_ss_dir = Path(sparse_ss_dir)
        
        self.pathloss = pathloss
        
        self.median_filter_size = median_filter_size

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        ids_gt = [splitext(file)[0] for file in listdir(self.ground_truth_signal_strength_map_dir) if
                  isfile(join(self.ground_truth_signal_strength_map_dir, file)) and not file.startswith(
                      '.')]
        ids_building = [splitext(file)[0].split("_")[0] for file in listdir(self.building_height_map_dir) if
                        isfile(join(self.building_height_map_dir, file)) and not file.startswith(
                            '.')]
        # ids_terrain = [splitext(file)[0].split("_")[0] for file in listdir(self.terrain_height_map_dir) if
        #                isfile(join(self.terrain_height_map_dir, file)) and not file.startswith(
        #                    '.')]
        
        self.ids = ids_gt
        
        
        # Here is a work around of tf variable length length in a single bath problem
        filtered_ids = []
        for file in tqdm(ids_gt):
            tmp = np.load(os.path.join(self.sparse_ss_dir,file.split("\\")[-1]+".npy"))
            if len(tmp) >= NUM_OF_POINTS:
                filtered_ids.append(file)
        self.ids = filtered_ids
                                                                         
        #print(self.ids)
        
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {self.ground_truth_signal_strength_map_dir}, make sure you put your images there')

        # logging.info(f'Creating dataset with {len(self.ids)} examples')
        # logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))
        #
        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    # @staticmethod
    # def preprocess(mask_values, pil_img, scale, is_mask):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    #     pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    #     img = np.asarray(pil_img)
    #
    #     if is_mask:
    #         mask = np.zeros((newH, newW), dtype=np.int64)
    #         for i, v in enumerate(mask_values):
    #             if img.ndim == 2:
    #                 mask[img == v] = i
    #             else:
    #                 mask[(img == v).all(-1)] = i
    #
    #         return mask
    #
    #     else:
    #         if img.ndim == 2:
    #             img = img[np.newaxis, ...]
    #         else:
    #             img = img.transpose((2, 0, 1))
    #
    #         if (img > 1).any():
    #             img = img / 255.0
    #
    #         return img
    def get_ids(self):
        return self.ids

    @staticmethod
    def uma_los(d3d, d2d, dbp, fc, h_b, h_t):
        # 38.901 UMa LOS
        PL1 = 28+22*np.log10(d3d)+20*np.log10(fc)
        PL2 = 28+40*np.log10(d3d)+20*np.log10(fc) - 9*np.log10(dbp**2+(h_b - h_t)**2)
        PL = np.zeros((d3d.shape))
        PL = PL2 # Default pathloss
        PL[(np.greater_equal(d2d,10) & np.less_equal(d2d,dbp))] = PL1[(np.greater_equal(d2d,10) & np.less_equal(d2d,dbp))] # Overwrite if distance is greater than 10 meters or smaller than dbp
        return PL
    
    
    @staticmethod
    def uma_nlos(d3d, d2d, dbp, fc, h_b, h_t):
        # 38901 UMa NLOS
        PL_nlos = 13.54+39.08*np.log10(d3d)+20*np.log10(fc)-0.6*(h_t-1.5)
        PL = np.zeros((d3d.shape))
        PL = np.maximum(RTDataset.uma_los(d3d, d2d, dbp, fc, h_b, h_t), PL_nlos)
        return PL
    
    @staticmethod
    def pathloss_38901(distance, frequency, h_bs=30, h_ut=1.5):
        #print(distance)
        """
            Simple path loss model for computing RSRP based on distance.

            fc: frequency in GHz
            h_b: height of basestation
            h_t: height of UT
        """
        # Constants
        fc = frequency
        h_b =  h_bs # 30 meters
        h_t =  h_ut # 1.5

        # 2D distance 
        d2d = distance

        # 3D distance
        h_e = h_b - h_t # effective height
        d3d = np.sqrt(d2d**2+h_e**2)

        # Breakpoint distance
        dbp =  4*h_b*h_t*fc*10e8/speed_of_light

        loss = RTDataset.uma_nlos(d3d, d2d, dbp, fc, h_b, h_t)
        return loss

    def __getitem__(self, idx):
        name = self.ids[idx]
        name_splited = name.split("_")
        file_name_id_part = name_splited[0]
        tx_height = name_splited[-1]
        tx_x = int(name_splited[-3])+500
        tx_y = (-1 * int(name_splited[-2]))+500
        tx_position = [tx_x // 10, tx_y // 10]
        #print(name)
        #print(tx_position)

        #print(tx_position)
        
        building_height_file = list(self.building_height_map_dir.glob(file_name_id_part + '_*.*'))
        # terrain_height_file = list(self.terrain_height_map_dir.glob(name + '_*.*'))
        ground_truth_file = list(self.ground_truth_signal_strength_map_dir.glob(name + '.npy'))
        
        
        sparse_ss_file = list(self.sparse_ss_dir.glob(name + '.npy'))
        
        
        assert len(
            building_height_file) == 1, f'Either no image or multiple images found for the ID {name}: {building_height_file}'
        # assert len(
        #     terrain_height_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {terrain_height_file}'
        assert len(
            ground_truth_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {ground_truth_file}'

        assert len(
            sparse_ss_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {sparse_ss_file}'
        
        
        
        # Ori image size 1040 * 1040, crop to 1000 * 1000
        building_height_arr = load_image(building_height_file[0])[4:104,4:104]
        # terrain_height_arr = load_image(terrain_height_file[0])[4:104,4:104]



        ground_truth_arr = load_image(ground_truth_file[0])
        
        
        if self.median_filter_size != 0:
            ground_truth_arr = ndimage.median_filter(ground_truth_arr, size = self.median_filter_size)
        
        
        
        sparse_ss_arr = np.array(load_image(sparse_ss_file[0]))
        choice = np.random.choice(len(sparse_ss_arr), NUM_OF_POINTS, replace=True)
        sparse_ss_arr = sparse_ss_arr[choice, :]
        # #Convert the linear power to dB scale
        #sparse_ss_arr = 10 * np.log10(sparse_ss_arr)
        
        #Convert the linear power to dB scale
        ground_truth_arr = 10 * np.log10(ground_truth_arr)
        
        
        ground_truth_arr[ground_truth_arr == np.nan] = -160
        ground_truth_arr[ground_truth_arr == -np.inf] = -160
        ground_truth_arr = np.nan_to_num(ground_truth_arr, nan=0)
        ground_truth_arr[ground_truth_arr >= 0] = 0
        ground_truth_arr[ground_truth_arr <= -160] = -160

        # Since right now GT.size is 100*100 and other two size is 1000 * 1000, just check the input.
        # assert building_height_arr.shape == terrain_height_aimport loggin
        combined_input = np.zeros((3, 100, 100), dtype=float)
        
        
        # Construct the TX position channel
        tx_position_channel = np.full((100, 100), 0, dtype=int)
        tx_position_channel[tx_position[1]][tx_position[0]] = tx_height
        combined_input[1,:, :] = tx_position_channel
        # Construct the Path Loss Model (3GPP TR 308.91 nLos UMa)
        path_loss_heat_map = np.full((100, 100), 0, dtype=float)
        

                
        # Combine all the channels together
        combined_input[0,:, :] = building_height_arr 
        distance = np.arange(0, 145,1)
        path_loss_res =  RTDataset.pathloss_38901(distance,2.14, h_bs=int(tx_height), h_ut=2)
        for row in range(path_loss_heat_map.shape[0]):
            for col in range(path_loss_heat_map.shape[1]):
                # Compute the distance between pixel and tx
                dist = math.sqrt((tx_position[1] - row)**2 + (tx_position[0] - col)**2)
                path_loss_heat_map[row][col] =  -1 * path_loss_res[int(dist)]
        combined_input[2,:, :] = path_loss_heat_map 
         
        
        #combined_input[1,:, :] = terrain_height_arr  
        return {
            'combined_input': torch.as_tensor(combined_input.copy()).float().contiguous(),
            'ground_truth': torch.as_tensor(ground_truth_arr.copy()).long().contiguous(),
            'file_name': name,
            'sparse_ss': torch.as_tensor(sparse_ss_arr.copy()).float().contiguous()
        }
    
    
    


if __name__ == '__main__':
    building_height_map_dir = Path('../../res/Bl_building_npy')
    terrain_height_map_dir = Path('../../res/Bl_terrain_npy')
    ground_truth_signal_strength_map_dir = Path('./coverage_maps')
    sparse_ss_dir = Path('/home/yl826/3DPathLoss/nc_raytracing/jul18_sparse')
    dataset = RTDataset(building_height_map_dir, terrain_height_map_dir, ground_truth_signal_strength_map_dir)
