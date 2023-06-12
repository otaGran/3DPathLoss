import bpy

# these are necessary for install_package(package_name)
import subprocess
import sys
import os

from PIL import Image
import numpy as np


def install_package(package_name):

    # path to python.exe
    python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
     
    # upgrade pip
    subprocess.call([python_exe, "-m", "ensurepip"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
     
    # install required packages
    subprocess.call([python_exe, "-m", "pip", "install", package_name])

    print("DONE")
    return


def delete_all_in_collection():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge()
    
    # clears all collections
    collections_list = list(bpy.data.collections)
    for col in collections_list:
        bpy.data.collections.remove(col)
    return


def add_osm(maxLon, minLon, maxLat, minLat):
    bpy.data.scenes['Scene'].blosm.dataType = 'osm'
    
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
    
    # import
    bpy.ops.blosm.import_data()
    return


def add_terrain(material_name, maxLon, minLon, maxLat, minLat):
    # toggle to 'terrain' import mode
    bpy.data.scenes['Scene'].blosm.dataType = 'terrain'
    
    # ensure correct settings
    bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = False
    
    # set bounds of import
    bpy.data.scenes['Scene'].blosm.maxLon = maxLon
    bpy.data.scenes['Scene'].blosm.minLon = minLon

    bpy.data.scenes['Scene'].blosm.maxLat = maxLat
    bpy.data.scenes['Scene'].blosm.minLat = minLat
    
    # import
    bpy.ops.blosm.import_data()
    
    # set properties
    terrain_obj = bpy.data.objects['Terrain']
    mat = bpy.data.materials.get(material_name)
    
    if mat is None:
        mat = bpy.data.materials.new(name=material_name)
    terrain_obj.data.materials.append(mat)
    return


def add_plane(material_name, size=1100):
    bpy.ops.mesh.primitive_plane_add(size=size)

    plane_obj = bpy.data.objects['Plane']
    mat = bpy.data.materials.get(material_name)
    
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name=material_name)
    plane_obj.data.materials.append(mat)
    return


def change_material_names(wall_name, roof_name, f_path):
    obj_names = [obj.name for obj in list(bpy.data.objects)]

    # check that there's more than one object
    if len(list(bpy.data.objects)) != 1 and obj_names[-1] != 'Plane':
        # set wall
        bpy.data.objects[obj_names[-1]].active_material_index = 0
        bpy.data.objects[obj_names[-1]].active_material.name = wall_name
        # set roof
        bpy.data.objects[obj_names[-1]].active_material_index = 1
        bpy.data.objects[obj_names[-1]].active_material.name = roof_name
    else:
        print("there's only one object (a plane) in this Scene Collection")

    # export
    bpy.ops.export_scene.mitsuba(filepath=f_path, axis_forward='Y', axis_up='Z')
    return


def terrain_to_png(terrain_img_path):
    bpy.data.objects["Terrain"].select_set(True)

    # compute mesh and vertices from Terrain object
    terrain_dg = bpy.context.evaluated_depsgraph_get()
    terrain_obj = bpy.context.object.evaluated_get(terrain_dg)
    mesh = terrain_obj.to_mesh(depsgraph=terrain_dg)
    vertices = mesh.vertices

    # compute the boundaries of the terrain
    min_x, min_y = mesh.vertices[0].co.x, mesh.vertices[-1].co.y
    max_x, max_y = mesh.vertices[-1].co.x, mesh.vertices[0].co.y

    # compute number of rows and columns in the terrain vertices
    dx = abs(mesh.vertices[0].co.x - mesh.vertices[1].co.x)
    num_x = int(round(abs((max_x - min_x) / dx))) + 1
    num_y = int(round(len(list(mesh.vertices)) / num_x))

    terrain_img_arr = np.zeros((num_y, num_x))

    for row in range(num_y):  # iterate through y, i.e. rows
        terrain_img_arr[row, :] = [vertices[col].co.z for col in range(row*num_x, (row+1)*num_x, 1)]

    terrain_img = Image.fromarray(terrain_img_arr)

    if terrain_img.mode != 'L':
        terrain_img = terrain_img.convert('L')

    terrain_img.save(terrain_img_path)

    bpy.data.objects["Terrain"].select_set(False)
    return


#delete_all_in_collection()

## should follow maxLon, minLon, maxLat, minLat
#diff = 0.001
#loc_args_dict = {'maxLon': -78.9340+diff, 'minLon': -78.9426-diff, 'maxLat': 36.0036+diff, 'minLat': 35.9965-diff}

#add_terrain(material_name='itu_concrete', **loc_args_dict)

#loc_args_dict = {'maxLon': -78.9340, 'minLon': -78.9426, 'maxLat': 36.0036, 'minLat': 35.9965}
#add_osm(**loc_args_dict)

##add_plane(material_name='itu_concrete', size=1100)

#abs_path = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_xml_files/duke_new2/duke_new2.xml'

#change_material_names(wall_name='itu_brick', roof_name='itu_plasterboard', f_path=abs_path)

#terrainImgPATH = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_terrain_img/duke_terrain.png'

#terrain_to_png(terrain_img_path=terrainImgPATH)