import numpy as np
import pandas as pd

from pyproj import Transformer

# from bokeh.io import output_notebook
from bokeh.plotting import figure, show, gmap
from bokeh.models import ColumnDataSource, LogColorMapper, LinearColorMapper, HoverTool, GMapOptions
import bokeh.palettes as palettes

# top left:
Tx_25832_TL_x, Tx_25832_TL_y = 720000, 6189000
# top right:
Tx_25832_TR_x, Tx_25832_TR_y = 722000, 6189000
# bottom right:
Tx_25832_BR_x, Tx_25832_BR_y = 722000, 6187000
# bottom left:
Tx_25832_BL_x, Tx_25832_BL_y = 720000, 6187000

from_EPSG = 25832
to_EPSG = 4326
transform_obj = Transformer.from_crs(from_EPSG, to_EPSG)  # transform from DTU coordinates to latitude, longitude

x_len = 101
y_len = 201
x_arr = np.linspace(Tx_25832_TL_x, Tx_25832_TR_x, num=x_len)
y_arr = np.linspace(Tx_25832_TL_y, Tx_25832_BL_y, num=y_len)
latitude_arr = np.zeros(len(x_arr) * len(y_arr))
longitude_arr = np.zeros(len(x_arr) * len(y_arr))
displayed_val = np.zeros(len(x_arr) * len(y_arr))
outer_idx = 0

for iii, x in enumerate(x_arr):
    for y in y_arr:
        latitude_arr[outer_idx], longitude_arr[outer_idx] = transform_obj.transform(xx=x, yy=y)
        outer_idx += 1
    displayed_val[iii*len(x_arr):iii*len(x_arr)+len(x_arr)] = iii
col_names = ['LONGITUDE', 'LATITUDE', 'VALUE']
df = pd.DataFrame(list(zip(latitude_arr, longitude_arr, displayed_val)), columns=col_names)

