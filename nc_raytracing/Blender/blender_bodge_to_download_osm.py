"""
This is to quickly download osm files by calling blosm.import_data() even though we don't need the import.
NOTE: run make_temp_res_file before running this file in Blender.
"""
import os
import bpy
import time


RES_PATH_NAME = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/res/res3_srv1_whole_us_filtered_new.txt'
BLENDER_OSM_DOWNLOAD_PATH = "/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/Blender_download_files/osm/"


def delete_all_in_collection():
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # clears all collections
        collections_list = list(bpy.data.collections)
        for col in collections_list:
            bpy.data.collections.remove(col)

        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            bpy.data.textures.remove(block)

        for block in bpy.data.images:
            bpy.data.images.remove(block)
        bpy.ops.outliner.orphans_purge(do_local_ids=True)
        return
    except Exception as e:
        raise e


def add_osm(maxLon, minLon, maxLat, minLat, from_file='n', osmFilepath=None):
    try:
        bpy.data.scenes['Scene'].blosm.osmSource = 'server'
        bpy.data.scenes['Scene'].blosm.dataType = 'osm'
        bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = True

        bpy.data.scenes['Scene'].blosm.maxLon = maxLon
        bpy.data.scenes['Scene'].blosm.minLon = minLon

        bpy.data.scenes['Scene'].blosm.maxLat = maxLat
        bpy.data.scenes['Scene'].blosm.minLat = minLat

        # ensure correct settings:
        # does not import as single object
        bpy.data.scenes["Scene"].blosm.singleObject = False

        # set osm import mode to 3Dsimple
        bpy.data.scenes["Scene"].blosm.mode = '3Dsimple'

        # only import buildings from osm
        bpy.data.scenes["Scene"].blosm.buildings = True
        bpy.data.scenes["Scene"].blosm.water = False
        bpy.data.scenes["Scene"].blosm.forests = False
        bpy.data.scenes["Scene"].blosm.vegetation = False
        bpy.data.scenes["Scene"].blosm.highways = False
        bpy.data.scenes["Scene"].blosm.railways = False

        # import from server
        if from_file == 'n':
            start = time.time()
            bpy.ops.blosm.import_data()
            print('\n\nosm download: ', str(time.time() - start) + ' seconds\n\n')
        if from_file == 'y' and osmFilepath is not None:
            bpy.data.scenes['Scene'].blosm.osmSource = 'file'
            bpy.data.scenes['Scene'].blosm.osmFilepath = osmFilepath
            bpy.ops.blosm.import_data()
        return
    except Exception as e:
        raise e


def splitting_a_line(lll, uuid_incl='n'):
    lll = lll.replace('(', '')
    lll = lll.replace(')', '')
    lll = lll.replace('\n', '')
    lll = lll.split(',')
    # file format: (minLon,maxLat,maxLon,minLat),percent,idx_uuid
    if uuid_incl == 'y':
        minLon, maxLat, maxLon, minLat, perc, idx_uuid = [k for k in lll]
        return float(minLon), float(maxLat), float(maxLon), float(minLat), float(perc), idx_uuid
    else:
        minLon, maxLat, maxLon, minLat, perc = [float(k) for k in lll]
        return minLon, maxLat, maxLon, minLat, perc


def run(lll, use_path_osm='n'):
    delete_all_in_collection()
    minLonOut, maxLatOut, maxLonOut, minLatOut, _, idx_uuid = splitting_a_line(lll=lll, uuid_incl='y')
    loc_args_dict = {'maxLon': maxLonOut, 'minLon': minLonOut, 'maxLat': maxLatOut, 'minLat': minLatOut}
    add_osm(**loc_args_dict, from_file=use_path_osm, osmFilepath=None)

    files = [f for f in os.listdir(BLENDER_OSM_DOWNLOAD_PATH) if os.path.isfile(str(BLENDER_OSM_DOWNLOAD_PATH + f))]
    files.sort(key=lambda f: os.path.getmtime(str(BLENDER_OSM_DOWNLOAD_PATH + f)))
    os.rename(src=BLENDER_OSM_DOWNLOAD_PATH + files[-1], dst=BLENDER_OSM_DOWNLOAD_PATH + idx_uuid + '.osm')


f_ptr_loc = open(RES_PATH_NAME, 'r')
lines = f_ptr_loc.readlines()
# # fff = [f for f in os.listdir(BLENDER_OSM_DOWNLOAD_PATH) if os.path.isfile(str(BLENDER_OSM_DOWNLOAD_PATH + f))]
# # fff.sort(key=lambda f: os.path.getmtime(str(BLENDER_OSM_DOWNLOAD_PATH + f)))
# temp_path = '/Users/zeyuli/Desktop/Duke/6. Electrochem Muser/'
# fff = [f for f in os.listdir(temp_path)]
# fff.sort(key=lambda f: os.path.getmtime(str(temp_path + f)))
# print(fff)
for line in lines:
    run(lll=line, use_path_osm='n')
