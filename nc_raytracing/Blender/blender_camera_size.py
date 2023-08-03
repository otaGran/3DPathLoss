import bpy
import numpy as np


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


def squarify_photo(arr, trim):
    try:
        arr = np.asarray(arr)
        rrr, ccc = arr.shape
        min_rc = min([rrr, ccc]) - trim
        return arr[int((rrr - min_rc) / 2):int((rrr + min_rc) / 2), int((ccc - min_rc) / 2):int((ccc + min_rc) / 2)]
    except Exception as e:
        raise e


def add_plane(material_name='itu_concrete', size=1000):
    try:
        bpy.ops.mesh.primitive_plane_add(size=size)

        plane_obj = bpy.data.objects['Plane']
        mat = bpy.data.materials.get(material_name)

        if mat is None:
            # create material
            mat = bpy.data.materials.new(name=material_name)
        plane_obj.data.materials.append(mat)
        print(plane_obj.data.materials[0])
        return
    except Exception as e:
        raise e


def get_depth():
    try:
        """Obtains depth map from Blender render.
        :return: The depth map of the rendered camera view as a numpy array of size (H,W).
        """
        z = bpy.data.images['Viewer Node']
        w, h = z.size
        dmap = np.array(z.pixels[:], dtype=np.float32)  # convert to numpy array
        dmap = np.reshape(dmap, (h, w, 4))[:, :, 0]
        dmap = np.rot90(dmap, k=2)
        dmap = np.fliplr(dmap)
        return dmap
    except Exception as e:
        raise e


def clear_compositing_nodes():
    try:
        bpy.data.scenes['Scene'].use_nodes = True
        tree = bpy.data.scenes['Scene'].node_tree
        for node in tree.nodes:
            tree.nodes.remove(node)
        return
    except Exception as e:
        raise e


def add_camera_and_set(camera_height, camera_orthoScale):
    try:
        # note: the camera is slightly "thinner" than the terrain.
        # Increase  camera_orthoScale  to increase the area captured.
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
        return
    except Exception as e:
        raise e


def take_picture_and_get_depth():
    try:
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

        return get_depth()
    except Exception as e:
        raise e


def routine_(c_height):
    delete_all_in_collection()
    add_plane()
    add_camera_and_set(camera_height=c_height, camera_orthoScale=1000)
    return take_picture_and_get_depth()


for idx, c_height in enumerate(range(250, 2500, 10)):
    square = squarify_photo(arr=routine_(c_height), trim=80)
    print(idx)
    if square[0, 0] < 65500:
        print('\n\n\n', square[0, 0])
        print('c_height', c_height, '\n\n\n')
