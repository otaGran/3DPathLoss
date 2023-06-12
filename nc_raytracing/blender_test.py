import bpy


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


def add_plane(material_name, size=1100):
    bpy.ops.mesh.primitive_plane_add(size=size)

    plane_obj = bpy.data.objects['Plane']
    mat = bpy.data.materials.get('itu_concrete')
    
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
    bpy.ops.export_scene.mitsuba(filepath='/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_xml_files/duke_new/duke_new.xml', axis_forward='Y', axis_up='Z')
    return


delete_all_in_collection()

# should follow maxLon, minLon, maxLat, minLat
loc_args = (-78.9340, -78.9426, 36.0036, 35.9965)

add_osm(*loc_args)

add_plane(material_name='itu_concrete', size=1100)

abs_path = '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_xml_files/duke_new/duke_new.xml'

change_material_names(wall_name='itu_brick', roof_name='itu_plasterboard', f_path=abs_path)
