This repository contains the code from the work detailed in the paper submitted to IEEE Access

```
@article{Thrane2020,
    author       = {Jakob Thrane, Darko Zibar, Henrik L. Christiansen},
    title        = {{Model-aided Deep Learning Method for Path Loss Prediction in Mobile Communication Systems at 2.6 GHz}},
    month        = Jan,
    year         = 2020,
    publisher    = {IEEE},
    journal      = {IEEE Access}
}
```

Previous work is detailed in:

```
@article{Thrane2018,
    author      = {Thrane, Jakob and Artuso, Matteo and Zibar, Darko and Christiansen, Henrik L},
    journal     = {VTC 2018 Fall},
    publisher   = {IEEE}
    title       = {{Drive test minimization using Deep Learning with Bayesian approximation}},
    year        = {2018}
}

```

# Instructions for the code

1. Download the dataset from 
https://ieee-dataport.org/open-access/mobile-communication-system-measurements-and-satellite-images

2. Put the raw data into the `raw_data` folder
    * such that the data is located in:
    * `raw_data\feature_matrix.csv`
    * `raw_data\output_matrix.csv`
    * `raw_data\mapbox_api\*.png`
3. Generate the test and training set using `generate_training_test.py`
4. Run the training of the model by `train.py`, see the script for commandline arguments


# Comments on formatted_data_gen

This directory formats SigCap data in exactly the same way as formatted in Thrane's paper, i.e.
```
,Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI,PCI_3,PCI_4,PCI_6,PCI_7,PCI_20,PCI_40,PCI_184,PCI_237
```

The SigCap data is collected using an Android phone. 

The formatting of the last few columns (from the first PCI_i onwards to the right) depends on the PCI values available. 

The exe file for geckodriver from https://github.com/mozilla/geckodriver/releases should be in venv/bin. 
Depending on the environment, geckodriver may need to be in another directory. Refer to https://github.com/thewati and 
https://medium.com/@watipasomulwafu for more information. 

The data/data_{date} directory must be inside the formatted_data_gen directory. 

Run getData_Michael.py to produce the csv file with formatted data. Run getImages_Michael.py to produce the satellite images
based on the (latitude, longitude) pairs in the data/data_{date} file. 


# Comments on DTU-processing-maps

This directory produces height maps for data collected in DTU (Technical University of Denmark). The DSM_6188_720_2x2 directory mustw
be present because it contains tiff files detailing the height information of DTU. grid_data stores the height maps and the 
feature_matrix which are generated using (latitude, longitude) values that are arranged in a grid. 

grid_pattern_feature_gen.py generates the features in exactly the same format as in Thrane's paper based on gridded locations. 

grid_pattern_height_map_gen.py generates the height maps in exactly the same format as Thrane's satellite images based on 
gridded locations. 

heatmap_prototyping tests code for plotting a heat map of signal strength. 

height_map_gen_orig_raw_data.py generates the height maps in exaclty the same format as Thrane's satellite images based on 
Thrane's (latitude, longitude) measurements provided in IEEE's data portal. 

The directory raw_data must be present in the parent directory of DTU-processing-maps. 