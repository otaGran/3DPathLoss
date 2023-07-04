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


class RTDataset(Dataset):
    def __init__(self, building_height_map_dir: str, terrain_height_map_dir: str,
                 ground_truth_signal_strength_map_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.building_height_map_dir = Path(building_height_map_dir)
        self.terrain_height_map_dir = Path(terrain_height_map_dir)
        self.ground_truth_signal_strength_map_dir = Path(ground_truth_signal_strength_map_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        ids_gt = [splitext(file)[0].split("_")[0] for file in listdir(self.ground_truth_signal_strength_map_dir) if
                  isfile(join(self.ground_truth_signal_strength_map_dir, file)) and not file.startswith(
                      '.')]
        ids_building = [splitext(file)[0].split("_")[0] for file in listdir(self.building_height_map_dir) if
                        isfile(join(self.building_height_map_dir, file)) and not file.startswith(
                            '.')]
        ids_terrain = [splitext(file)[0].split("_")[0] for file in listdir(self.terrain_height_map_dir) if
                       isfile(join(self.terrain_height_map_dir, file)) and not file.startswith(
                           '.')]

        self.ids = list(set(ids_gt).intersection(ids_building).intersection(ids_terrain))
        print(self.ids)
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

    def __getitem__(self, idx):
        name = self.ids[idx]
        building_height_file = list(self.building_height_map_dir.glob(name + '_*.*'))
        terrain_height_file = list(self.terrain_height_map_dir.glob(name + '_*.*'))
        ground_truth_file = list(self.ground_truth_signal_strength_map_dir.glob(name + '_*.npy'))

        assert len(
            building_height_file) == 1, f'Either no image or multiple images found for the ID {name}: {building_height_file}'
        assert len(
            terrain_height_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {terrain_height_file}'
        assert len(
            ground_truth_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {ground_truth_file}'


        # Ori image size 1040 * 1040, crop to 1000 * 1000
        building_height_arr = load_image(building_height_file[0])[40:1040,40:1040]
        terrain_height_arr = load_image(terrain_height_file[0])[40:1040,40:1040]



        ground_truth_arr = load_image(ground_truth_file[0])
        ground_truth_arr[ground_truth_arr == -np.inf] = -300
        ground_truth_arr = np.nan_to_num(ground_truth_arr, nan=0)
        ground_truth_arr[ground_truth_arr >= 0] = 0


        # Since right now GT.size is 100*100 and other two size is 1000 * 1000, just check the input.
        assert building_height_arr.shape == terrain_height_arr.shape, \
            f'Image and mask {name} should be the same size, but are {building_height_arr.shape} and {terrain_height_arr.shape}'

        # building_height_arr = self.preprocess(building_height_arr)
        # terrain_height_arr = self.preprocess(terrain_height_arr)
        # ground_truth_arr = self.preprocess(self.mask_values, ground_truth_arr, self.scale, is_mask=True)
        combined_input = np.zeros((2, 100, 100))
        combined_input[0,:, :] = building_height_arr[::10, ::10]  # Assign first channel data
        combined_input[1,:, :] = terrain_height_arr[::10, ::10]  # Assign second channel data
        return {
            'combined_input': torch.as_tensor(combined_input.copy()).float().contiguous(),
            'ground_truth': torch.as_tensor(ground_truth_arr.copy()).long().contiguous()
        }


if __name__ == '__main__':
    building_height_map_dir = Path('../../res/Bl_building_npy')
    terrain_height_map_dir = Path('../../res/Bl_terrain_npy')
    ground_truth_signal_strength_map_dir = Path('./coverage_maps')
    dataset = RTDataset(building_height_map_dir, terrain_height_map_dir, ground_truth_signal_strength_map_dir)
