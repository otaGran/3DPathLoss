from height_map_gen_orig_raw_data import combine_height_maps, get_chip
import pandas as pd
from PIL import Image


"""
    This code generates height map pngs according to (latitude, longitude) pairs specified by the 
    feature_matrix_grid.csv generated by grid_pattern_feature_gen.py. The height maps are fed along
    with the feature matrix into the model to predict the path loss, after which a heat map can be drawn.
"""


def gen_height_map_for_PCI(which_pci):
    chip_dimension = 256 * 1.25
    padded_img = combine_height_maps(pad_fraction=0.2)
    df_feature = pd.read_csv('grid_data/feature_matrix_grid_PCI{:d}.csv'.format(which_pci))
    for r_df in range(len(df_feature)):
        temp_long = df_feature['Latitude'][r_df]
        temp_lati = df_feature['Longitude'][r_df]
        temp_chip = get_chip(padded_img=padded_img, test_lon=temp_long, test_lat=temp_lati, arr_dimension=chip_dimension)
        im = Image.fromarray(temp_chip).convert("L")
        im.save("grid_data/height_maps_grid_data/PCI{:d}/{:d}.png".format(which_pci, r_df))
        print(r_df)


if __name__ == '__main__':
    gen_height_map_for_PCI(which_pci=64)