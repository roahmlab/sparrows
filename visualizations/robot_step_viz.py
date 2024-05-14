""" Helper class for visualizing the history of planned robot steps.

Usable with pyrender environments which have a scene attribute.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pyrender
from environments.urdf_base import RENDERER

class RobotStepViz:
    def __init__(self,):
        self.mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=(1.0, 1.0, 1.0, 0.3),
        )
        self.mesh_list = []
    
    def render_callback(self, env):
        if len(self.mesh_list) == 0:
            # add the initial mesh
            init_mesh_list = self.get_tm_list(env, env.init_qpos)
            self.mesh_list.append(init_mesh_list)
        
        if env.renderer == RENDERER.PYRENDER or env.renderer == RENDERER.PYRENDER_OFFSCREEN:
            # Create the nodes
            pyrender_mesh_list = []
            total_meshes = len(self.mesh_list)
            for i, mesh in enumerate(self.mesh_list):
                mat = self.mat
                mat.baseColorFactor = (mat.baseColorFactor[0], mat.baseColorFactor[1], mat.baseColorFactor[2], 1.0 - 0.6*(total_meshes-i)/total_meshes)
                pyr_mesh = pyrender.Mesh.from_trimesh(mesh, material=mat)
                pyrender_mesh_list.append(pyr_mesh)
            node_list = [env.scene.add(mesh) for mesh in pyrender_mesh_list]

            # Add the current mesh to the list
            current_mesh_list = self.get_tm_list(env, env.qpos)
            self.mesh_list.append(current_mesh_list)
            return lambda: [env.scene.remove_node(node) for node in node_list]
        
        elif env.renderer == RENDERER.BLENDER:
            import bpy
            
            # Add the current mesh to the list
            current_mesh_list = self.get_tm_list(env, env.qpos)
            self.mesh_list.append(current_mesh_list)
            
            traj_mat = bpy.data.materials.get("RobotTrajMat")
            if traj_mat is None:
                traj_mat = bpy.data.materials.new(name="RobotTrajMat")
                traj_mat.use_nodes = True
                    
            itr = getattr(self, 'itr', 1)
            viz_collection = getattr(self, 'viz_collection', None)
            if viz_collection is None:
                viz_collection = bpy.data.collections.new('robot_traj_viz')
                env.scene.collection.children.link(viz_collection)
                self.viz_collection = viz_collection
            
            viz_subcollection = bpy.data.collections.new(f'robot_traj_viz_{itr}')
            viz_collection.children.link(viz_subcollection)
            current_frame = env.scene.frame_current
            self.all_last_obj = []
            
            for i, mesh in enumerate(current_mesh_list):
                name = f'robot_traj_mesh_step{itr}_{i}'
                bpy_mesh = bpy.data.meshes.new(name)
                bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=False)
                bpy_mesh.validate()
                bpy_mesh.update()
                bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                # bpy_obj.hide_render = True
                # bpy_obj.hide_viewport = True
                bpy_obj.keyframe_insert(data_path="hide_render", frame=0)
                bpy_obj.keyframe_insert(data_path="hide_viewport", frame=0)
                # bpy_obj.hide_render = False
                # bpy_obj.hide_viewport = False
                # bpy_obj.keyframe_insert(data_path="hide_render", frame=1)
                # bpy_obj.keyframe_insert(data_path="hide_viewport", frame=1)
                bpy_obj.data.materials.append(traj_mat)
                viz_subcollection.objects.link(bpy_obj)
                self.all_last_obj.append(bpy_obj)
            
            # for i_meshlist, meshes in enumerate(self.mesh_list):
            #     for i, mesh in enumerate(meshes):
            #         name = f'armtd_fullset_mesh_{itr}_{i}'
            #         bpy_mesh = bpy.data.meshes.new(name)
            #         bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=False)
            #         bpy_mesh.validate()
            #         bpy_mesh.update()
            #         bpy_obj = bpy.data.objects.new(name, bpy_mesh)
            #         bpy_obj.hide_render = True
            #         bpy_obj.hide_viewport = True
            #         bpy_obj.keyframe_insert(data_path="hide_render", frame=0)
            #         bpy_obj.keyframe_insert(data_path="hide_viewport", frame=0)
            #         bpy_obj.hide_render = False
            #         bpy_obj.hide_viewport = False
            #         bpy_obj.keyframe_insert(data_path="hide_render", frame=current_frame)
            #         bpy_obj.keyframe_insert(data_path="hide_viewport", frame=current_frame)
            #         bpy_obj.data.materials.append(traj_mat)
            #         viz_subcollection.objects.link(bpy_obj)
            #         self.all_last_obj.append(bpy_obj)
            self.itr = itr + 1

            def hide_all():
                for obj in self.all_last_obj:
                    obj.hide_render = True
                    obj.hide_viewport = True
                    obj.keyframe_insert(data_path="hide_render")
                    obj.keyframe_insert(data_path="hide_viewport")

            return hide_all
            


    def get_tm_list(self, env, qpos):
        fk = env.robot.visual_trimesh_fk(qpos)
        tm_list = []
        for tm, pose in fk.items():
            mesh = tm.copy()
            mesh.apply_transform(pose)
            tm_list.append(mesh)
        return tm_list
