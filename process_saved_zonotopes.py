from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope, zonotope
import pickle
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def add_mesh_to_viz(mesh, name, viz_collection, mat):
    bpy_mesh = bpy.data.meshes.new(name)
    bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=False)
    bpy_mesh.validate()
    bpy_mesh.update()
    bpy_obj = bpy.data.objects.new(name, bpy_mesh)
    bpy_obj.hide_render = False
    bpy_obj.hide_viewport = False
    bpy_obj.keyframe_insert(data_path="hide_render", frame=0)
    bpy_obj.keyframe_insert(data_path="hide_viewport", frame=0)
    viz_collection.objects.link(bpy_obj)
    bpy_obj.data.materials.append(mat)
    return

if __name__ == '__main__':
    with open('saved_jso.pkl', 'rb') as f:
        data = pickle.load(f)

    device = 'cuda'
    link_names = ['joint_4', 'joint_5']
    taken_timestep = 99
    ka = torch.ones(7, device=device) * 0.

    mesh_dict = {}
    mat_dict = {}
    scale = 10.
    for link_name in link_names:
        jso = data[link_name]
        joint_center = ((jso['joint_center'][taken_timestep].slice_all_dep(ka)).Z)[0]
        joint_uncertainty = jso['joint_uncertainty'][taken_timestep] * scale
        joint_uncertainty_rad = jso['uncertainty_rad'][taken_timestep] * scale
        
        # save the position zonotope
        joint_position_zonotope = joint_uncertainty.Z.clone()
        joint_position_zonotope[0] = joint_center
        joint_position_zonotope = zonotope(joint_position_zonotope)
        joint_position_zonotope_patch = joint_position_zonotope.polyhedron_patch().reshape(-1,3)
        joint_position_zonotope_mesh = trimesh.convex.convex_hull(joint_position_zonotope_patch)
        mesh_dict[link_name + '_position'] = joint_position_zonotope_mesh
        mat_dict[link_name + '_position_zonotope'] = joint_position_zonotope.Z.cpu().numpy()
        
        # save the box
        joint_position_zonotope_cube = torch.cat([joint_center.unsqueeze(0), torch.diag(joint_uncertainty.to_interval().sup)], dim=0)
        joint_position_zonotope_cube = zonotope(joint_position_zonotope_cube)
        joint_position_zonotope_cube_patch = joint_position_zonotope_cube.polyhedron_patch().reshape(-1,3)
        joint_position_zonotope_cube_mesh = trimesh.convex.convex_hull(joint_position_zonotope_cube_patch)
        mesh_dict[link_name + '_box'] = joint_position_zonotope_cube_mesh
        mat_dict[link_name + '_position_box_zonotope'] = joint_position_zonotope_cube.Z.cpu().numpy()
        
        # save the sphere
        r = joint_uncertainty_rad.item()
        joint_sphere_mesh = trimesh.primitives.Sphere(radius=r, center=joint_center.cpu().numpy())
        mesh_dict[link_name + '_sphere'] = joint_sphere_mesh
        mat_dict[link_name + '_sphere_radius'] = r
        
    save_plot = True
    if save_plot:
        import matplotlib.pyplot as plt
        lim = 1
        fig, ax = plt.subplots(1,1,subplot_kw=dict(projection='3d'))
        ax.add_collection3d(Poly3DCollection(joint_position_zonotope.polyhedron_patch(),edgecolor='orange',facecolor='orange',alpha=0.15,linewidths=0.5))
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        plt.savefig(f"hi.pdf", format="pdf", dpi=600)
        
    save_mat = False
    if save_mat:
        mat_filename = "joint_position_zonotope.mat"
        from scipy.io import savemat
        savemat(mat_filename, mat_dict)
    
    save_blender = False
    if save_blender: 
        import bpy
        filename = f'joint_position_zonotope_scale{scale}.blend'
        print("Saving blend file to: ", filename)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.save_as_mainfile(filepath=filename)
        bpy.context.preferences.filepaths.save_version = 0
        scene = bpy.data.scenes.get('Scene')
        scene.render.fps = 60
        scene.render.fps_base = 1.0
        scene.render.resolution_x = 600
        scene.render.resolution_y = 600
        scene.render.resolution_percentage = 100
        
        joint_position_mat = bpy.data.materials.new(name="JointPosition")
        joint_position_mat.use_nodes = True
        joint_box_mat = bpy.data.materials.new(name="JointBox")
        joint_box_mat.use_nodes = True
        joint_sphere_mat = bpy.data.materials.new(name="JointSphere")
        joint_sphere_mat.use_nodes = True
        
        viz_collection = bpy.data.collections.new('joint_position_viz')
        scene.collection.children.link(viz_collection)

        for mesh_name, mesh in mesh_dict.items():
            if 'position' in mesh_name:
                mat = joint_position_mat
            elif 'box' in mesh_name:
                mat = joint_box_mat
            elif 'sphere' in mesh_name:
                mat = joint_sphere_mat
            else:
                raise NotImplementedError("?")
            add_mesh_to_viz(mesh, mesh_name, viz_collection, mat)
            
        final_frame = 0
        scene.frame_end = final_frame
        scene.frame_set(0)
        bpy.ops.wm.save_mainfile(compress=True)