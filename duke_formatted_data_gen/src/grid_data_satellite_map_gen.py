from getImages_Michael import get_Img_all

from grid_data_feature_matrix_Duke import getData_from_csv


def gen_satellite_imgDuke_for_PCI(which_pci):
    feature_dict = getData_from_csv('../grid_data/feature_matrix_grid_PCI{:d}.csv'.format(which_pci))

    mapbox_tile_ID = 'mapbox.satellite'
    pad_fraction_ = 0.9
    zoom_level_ = 17
    cropped_chip_size = 256
    get_Img_all(data_formatted=feature_dict, directory='../grid_data/grid_map_api/PCI{:d}'.format(which_pci),
                mapbox_tile_set_ID=mapbox_tile_ID, cropped_size=cropped_chip_size,
                pad_frac=pad_fraction_, zoom_level=zoom_level_)


if __name__ == '__main__':
    gen_satellite_imgDuke_for_PCI(which_pci=3)
