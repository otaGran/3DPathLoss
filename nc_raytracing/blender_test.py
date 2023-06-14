import bpy

# these are necessary for install_package(package_name)
import subprocess
import sys
import os

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


def delete_all_in_collection():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge()
    
    # clears all collections
    collections_list = list(bpy.data.collections)
    for col in collections_list:
        bpy.data.collections.remove(col)
    return


def add_osm(maxLon, minLon, maxLat, minLat, from_file='n', osmFilepath=None):
    bpy.data.scenes['Scene'].blosm.osmSource = 'server'
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
    
    # import from server
    if from_file == 'n':
        bpy.ops.blosm.import_data()
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


def change_material_names_and_export(wall_name, roof_name, f_path):
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
    
    print('x min and max of terrain mesh: ', min_x, max_x)
    print('y min and max of terrain mesh: ', min_y, max_y)
    
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


def get_height_at_origin(terrainLim, camera_height=2000, camera_orthoScale=2000):
    min_x, max_x, min_y, max_y = terrainLim
    assert min_x < max_x and min_y < max_y
    # add a camera and link it to scene
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    
    curr_camera = bpy.data.cameras["Camera.001"]
    
    camera_object.location[2] = camera_height  # setting camera height
    curr_camera.type = 'ORTHO'  # setting camera type
    curr_camera.clip_start = 0.1  # setting clipping
    curr_camera.clip_end = camera_height * 5
    curr_camera.ortho_scale = camera_orthoScale  # setting camera scale
    curr_camera.dof.use_dof = False  # do not use, this makes the photo misty
#    curr_camera.track_axis = '-Z'
#    curr_camera.up_axis = 'Y'
    
    bpy.context.scene.camera = bpy.data.objects['Camera.001']
    
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
    depth_flatten = depth.flatten()
    
    selection = np.asarray(np.copy(depth_flatten) < 65500)
    print(len(selection))
    depth_final = depth_flatten[selection]
    
    print()
    print(np.min(depth), np.max(depth_final))
    bpy.data.objects["Camera.001"].select_set(False)
    print('done\n')
    return


delete_all_in_collection()

# should follow maxLon, minLon, maxLat, minLat
diff = 0.001
loc_args_dict = {'maxLon': -78.9340+diff, 'minLon': -78.9426-diff, 'maxLat': 36.0036+diff, 'minLat': 35.9965-diff}

add_terrain(material_name='itu_concrete', **loc_args_dict)

loc_args_dict = {'maxLon': -78.9340, 'minLon': -78.9426, 'maxLat': 36.0036, 'minLat': 35.9965}
osm_f_path = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_download_files/osm/map.osm'
add_osm(**loc_args_dict, from_file='y', osmFilepath=osm_f_path)

#add_plane(material_name='itu_concrete', size=1100)

abs_path = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_xml_files/duke_new2/duke_new2.xml'

change_material_names_and_export(wall_name='itu_brick', roof_name='itu_plasterboard', f_path=abs_path)

terrainImgPATH = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_terrain_img/duke_terrain.png'

terrain_limits = terrain_to_png(terrain_img_path=terrainImgPATH)
get_height_at_origin(terrain_limits, camera_height=2000, camera_orthoScale=2000)
