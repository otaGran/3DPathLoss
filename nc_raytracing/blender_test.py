import bpy

# these are necessary for install_package(package_name)
import subprocess
import sys
import os
from datetime import datetime
import time

from PIL import Image
import numpy as np


def install_package(package_name):

    # path to python.exe
    python_exe = os.path.join(sys.prefix, 'bin', 'python3.10')
     
    # upgrade pip
    subprocess.call([python_exe, "-m", "ensurepip"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
     
    # install required packages
    subprocess.call([python_exe, "-m", "pip", "install", package_name])

    print("DONE")
    return


def delete_terrain_and_osm_files(PATH_download='/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/Blender_download_files'):
    folder_path_osm = PATH_download + '/osm'  #enter path here
    delete_files_from_directory(folder_path_osm)
    folder_path_terrain = PATH_download + '/terrain'
    delete_files_from_directory(folder_path_terrain)
    return


def delete_files_from_directory(folder_path):
    for filename in os.listdir(folder_path): 
        file_path = os.path.join(folder_path, filename)  
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
#            elif os.path.isdir(file_path):  
#                os.rmdir(file_path)
        except Exception as e:  
            print(f"Error deleting {file_path}: {e}")
    print("Deletion done")
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


def normalise_to_png(arr_to_norm, maxVal):
    
    return maxVal * (arr_to_norm-np.min(arr_to_norm))/(np.max(arr_to_norm)-np.min(arr_to_norm))


def add_osm(maxLon, minLon, maxLat, minLat, from_file='n', osmFilepath=None):
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


def add_terrain(material_name, maxLon, minLon, maxLat, minLat):
    # toggle to 'terrain' import mode
    bpy.data.scenes['Scene'].blosm.dataType = 'terrain'
    
    # ensure correct settings
    bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = False
    bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = True
    
    # set bounds of import
    bpy.data.scenes['Scene'].blosm.maxLon = maxLon
    bpy.data.scenes['Scene'].blosm.minLon = minLon

    bpy.data.scenes['Scene'].blosm.maxLat = maxLat
    bpy.data.scenes['Scene'].blosm.minLat = minLat
    
    # import
    start = time.time()
    bpy.ops.blosm.import_data()
    print('\n\nTerrain download: ', str(time.time() - start) + ' seconds\n\n')
    
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


def change_material_names_and_export(wall_name, roof_name, f_path):
    obj_names = [obj.name for obj in list(bpy.data.objects)]
    if len(obj_names) <= 2:
        return 1
    
#    print(obj_names)
    obj_names = []
    map_name = ''
    for obj in list(bpy.data.objects):
        if 'Camera' not in obj.name and 'Terrain' not in obj.name and 'osm_buildings' not in obj.name:
            obj_names.append(obj.name)
        if '.osm' in obj.name:
            map_name = obj.name
    
    # check that there's more than one object
#    print(list(bpy.data.objects))
    if len(list(bpy.data.objects)) >= 1:
#        print(bpy.data.objects[obj_names[-1]])
        # set wall
        bpy.data.objects[obj_names[-1]].active_material_index = 0
        bpy.data.objects[obj_names[-1]].active_material.name = wall_name
        # set roof
        bpy.data.objects[obj_names[-1]].active_material_index = 1
        bpy.data.objects[obj_names[-1]].active_material.name = roof_name
    else:
        print("there's only one object (a plane) in this Scene Collection")
    # suppress and then turn on terminal output
#    sys.stdout, sys.stderr = os.devnull, os.devnull
    # export
    bpy.ops.export_scene.mitsuba(filepath=f_path, axis_forward='Y', axis_up='Z')
#    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    return 0


def terrain_to_png(terrain_img_path, outer_idx, save='n', normalise_to_256='n'):
    bpy.data.objects["Terrain"].select_set(True)

    # compute mesh and vertices from Terrain object
    terrain_dg = bpy.context.evaluated_depsgraph_get()
    terrain_obj = bpy.context.object.evaluated_get(terrain_dg)
    mesh = terrain_obj.to_mesh(depsgraph=terrain_dg)
    vertices = mesh.vertices

    # compute the boundaries of the terrain
    # min_x: top-left corner; min_y: bottom-right corner
    # max_x: bottom-right corner; max_y: top-left corner
    min_x, min_y = mesh.vertices[0].co.x, mesh.vertices[-1].co.y
    max_x, max_y = mesh.vertices[-1].co.x, mesh.vertices[0].co.y
    
    print('x min and max of terrain mesh: ', min_x, max_x)
    print('y min and max of terrain mesh: ', min_y, max_y)
    
    # compute number of rows and columns in the terrain vertices
    # to do this, compute the difference array. Then, find the index
    # of the first location where the difference is negative. 
    # This index is the number of columns. This works because the vertices
    # array for the terain lists vertices from top to bottom 
    # and from left to right. 
    dx = [mesh.vertices[i].co.x - mesh.vertices[i-1].co.x for i in range(1, len(vertices))]
    
    x_idx = -1
    for idx, delt in enumerate(dx):
        if delt < 0:
            x_idx = idx
            break
    print(x_idx)
    
    if save == 'y':
        num_x = x_idx + 1
        num_y = int(round(len(list(mesh.vertices)) / num_x))
        
        # weird Blender behaviour: this function fails on 
        # initial run upon starting Blender, so I'm running
        # a "test run" using idx = -1 to clear up this stuff
        if num_x * num_y != len(vertices):
            if outer_idx == 1:
                f_ptr_error_Exception.write('Exception at -1 for test run, which is expected\n')
            else:
                print('\n\n incorrect')
                print('row, col: ' + str(num_y), str(num_x) + '\n')
                print('what the size should be:', str(len(vertices)) + '\n')
                raise Exception('terrain_to_png Did not get the correct row_len and col_len')
            
        terrain_img_arr = np.zeros((num_y, num_x))
        
        print(num_x, num_y, num_x*num_y, len(vertices))
#        for idx, vert in enumerate(vertices):
#            print(vert.co.z)
#            if idx == num_x:
#                break
        for row in range(num_y):  # iterate through y, i.e. rows
            terrain_img_arr[row, :] = [vertices[col].co.z for col in range(row*num_x, (row+1)*num_x, 1)]

        if normalise_to_256 == 'y':
            terrain_img_arr = normalise_to_png(terrain_img_arr, 256)
        terrain_img = Image.fromarray(terrain_img_arr)

#        if terrain_img.mode != 'L':
#            terrain_img = terrain_img.convert('L')
        
        terrain_img.save(terrain_img_path)

    bpy.data.objects["Terrain"].select_set(False)
    return min_x, max_x, min_y, max_y


def get_depth():
    """Obtains depth map from Blender render.
    :return: The depth map of the rendered camera view as a numpy array of size (H,W).
    """
    z = bpy.data.images['Viewer Node']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32) # convert to numpy array
    dmap = np.reshape(dmap, (h, w, 4))[:,:,0]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    return dmap


def clear_compositing_nodes():
    bpy.data.scenes['Scene'].use_nodes = True
    tree = bpy.data.scenes['Scene'].node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)
    return


def get_height_at_origin(terrainLim, camera_height=2000, camera_orthoScale=2000, normalise_to_256='n'):
    min_x, max_x, min_y, max_y = terrainLim
    assert min_x < max_x and min_y < max_y
    # add a camera and link it to scene
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    
    curr_camera = bpy.data.cameras["Camera"]
    
    camera_object.location[2] = camera_height  # setting camera height
    curr_camera.type = 'ORTHO'  # setting camera type
    curr_camera.clip_start = 0.1  # setting clipping
    curr_camera.clip_end = camera_height * 5
    curr_camera.ortho_scale = camera_orthoScale  # setting camera scale
    curr_camera.dof.use_dof = False  # do not use, this makes the photo misty
#    curr_camera.track_axis = '-Z'
#    curr_camera.up_axis = 'Y'
    
    bpy.context.scene.camera = bpy.data.objects['Camera']
    
    # enable z data to be passed and use nodes for compositing
    bpy.data.scenes['Scene'].view_layers["ViewLayer"].use_pass_z = True
    bpy.data.scenes['Scene'].use_nodes = True
    
    tree = bpy.data.scenes['Scene'].node_tree
    
    # clear nodes
    clear_compositing_nodes()
    
    image_node = tree.nodes.new(type='CompositorNodeRLayers')
    viewer_node = tree.nodes.new(type='CompositorNodeViewer')
    viewer_node.location = 400, 0
    
    tree.links.new(image_node.outputs["Depth"], viewer_node.inputs['Image'])
    
    tree.nodes['Render Layers'].layer = 'ViewLayer'
    
    bpy.ops.render.render(layer='ViewLayer')

    depth = get_depth()
    select_not_2D = depth > 65500
    depth[select_not_2D] = 2000

    if normalise_to_256 == 'y':
        depth = normalise_to_png(depth, 256)
    depth_img = Image.fromarray(depth)
    
    temp_path = '/Users/zeyuli/Desktop/temp_depth.png'
    if depth_img.mode != 'L':
        depth_img = depth_img.convert('L')
    
    # depth_img.save(temp_path)
    
    depth_flatten = depth.flatten()
    
    selection = np.asarray(np.copy(depth_flatten) < 65500)
    
    depth_final = depth_flatten[selection]
    
    print()
    print('min_depth: ', np.min(depth), '. max_depth: ', np.max(depth_final))
    bpy.data.objects["Camera"].select_set(False)
    print('done with get_height_at_origin\n')
    return


def run(maxLon, minLon, maxLat, minLat, idx):
#    bpy.ops.wm.read_factory_settings(use_empty=True)
    
#    bpy.ops.read_homefile(filepath='/Applications/Blender.app/Contents/Resources/3.3/scripts/startup/bl_app_templates_system/OSM/startup.blend')
#    bpy.ops.wm.read_userpref()
    delete_all_in_collection()

    # should follow maxLon, minLon, maxLat, minLat
    diff = 0.0015
    loc_args_dict = {'maxLon': maxLon+diff, 'minLon': minLon-diff, 'maxLat': maxLat+diff, 'minLat': minLat-diff}

    add_terrain(material_name='itu_concrete', **loc_args_dict)

    loc_args_dict = {'maxLon': maxLon, 'minLon': minLon, 'maxLat': maxLat, 'minLat': minLat}

    # use already-downloaded osm
#    osm_f_path = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/Blender_download_files/osm/map.osm'
#    add_osm(**loc_args_dict, from_file='y', osmFilepath=osm_f_path)
    
    # download from server
    add_osm(**loc_args_dict, from_file='n', osmFilepath=None)

    # do not add plane. Instead, add terrain
    #add_plane(material_name='itu_concrete', size=1100)

    # path of xml file which would be exported in change_material_names_and_export
    i = str(int(idx))
    abs_path = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/Blender_xml_files/' + i + '/' + i + '.xml'

    ret = change_material_names_and_export(wall_name='itu_brick', roof_name='itu_plasterboard', f_path=abs_path)
    if ret != 0:
        return
    
    terrainImgPATH = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/Blender_terrain_img/' + i + '.tiff'
    terrain_limits = terrain_to_png(terrain_img_path=terrainImgPATH, save='y', outer_idx=idx)
    
    get_height_at_origin(terrain_limits, camera_height=2000, camera_orthoScale=2000)
    delete_terrain_and_osm_files()
    return


## running Duke
maxLon, minLon, maxLat, minLat = -78.9340, -78.9426, 36.0036, 35.9965

#run(maxLon, minLon, maxLat, minLat, idx=0)

run(maxLon, minLon, maxLat, minLat, idx=-1)

loc_fPtr = open('/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/res.txt', 'r')
lines = loc_fPtr.readlines()
print(len(lines))

error_path_IdxAndPercentBuildings = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/error_IdxAndPercentBuildings.txt'
error_path_Exception = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/error_Exception.txt'
f_ptr_error_IdxAndPercentBuildings = open(error_path_IdxAndPercentBuildings, 'w')
f_ptr_error_Exception = open(error_path_Exception, 'a')

f_ptr_error_Exception.write('\n\n\n-------\nRun started at: ' + str(datetime.now()) + '\n')

percent_threshold = 0.0001
idx = 0
count = 0
for line in lines:
    if idx > 1:
        break
    line = line.replace('(', '')
    line = line.replace(')', '')
    line = line.replace('\n', '')
    line = line.split(',')
    # file format: minLon, maxLat, maxLon, minLat
    minLon, maxLat, maxLon, minLat, percent = [float(l) for l in line]
    if percent > percent_threshold:
        try:
            run(maxLon, minLon, maxLat, minLat, idx)
        except Exception as e:
            f_ptr_error_Exception.write(str(datetime.now()) + '\n')
            f_ptr_error_Exception.write(str(idx) + ',' + str(percent) + ',' + str(e) + '\n')
            f_ptr_error_IdxAndPercentBuildings.write(str(idx) + ',' + str(percent) + '\n')
            print(e)
        count += 1
    idx += 1
    
print('number of files with bulding to area ratio > ' + str(percent_threshold), count)
print('percentage of files with bulding to area ratio > ' + str(percent_threshold), count / len(lines))

f_ptr_error_Exception.write(str(datetime.now()) + ', last index: ' + str(idx) + '\n')
f_ptr_error_Exception.write('number of OSM files exported: ' + str(count) + '\n')

f_ptr_error_Exception.close()
f_ptr_error_IdxAndPercentBuildings.close()

print('\n\n\nDONE\n\n')
