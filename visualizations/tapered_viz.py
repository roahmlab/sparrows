from __future__ import annotations
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import trimesh
import pyrender
import torch
import numpy as np
from forward_occupancy.SO import make_spheres
from zonopyrobots.robots.utils import normal_vec_to_basis
from typing import TYPE_CHECKING
from environments.urdf_base import RENDERER

if TYPE_CHECKING:
    from typing import List, Union
    from numpy import ndarray as NDArray
    from torch import Tensor
    from trimesh import Trimesh
    from planning.sparrows.sparrows_urdf import SPARROWS_3D_planner
    from environments.urdf_base import KinematicUrdfBase


class SpherePlannerTaperedViz:
    def __init__(
            self,
            planner: SPARROWS_3D_planner,
            plot_full_set: bool = False,
            t_plan: float = 0.5,
            t_full: float = 1.0,
            ) -> None:
        ''' Initialize the sphere visualization with a planner.
        
        Args:
            planner: The planner to use for the visualization callbacks.
            plot_full_set: Whether to plot the full set or just the current timestep (provided by the env).
            t_plan: The time to plot the current timestep.
            t_full: The full time of the trajectory.
        '''
        self.planner = planner
        self.plot_full_set = plot_full_set
        self.g_ka = self.planner.g_ka
        self.ka = None
        self.t_plan = t_plan
        self.t_full = t_full
        self.rs_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.4,
            alphaMode='BLEND',
            baseColorFactor=(0.3, 0.1, 0.7, 0.3),
        )
        self.rs_mat_failsafe = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.4,
            alphaMode='BLEND',
            baseColorFactor=(0.8, 0.0, 0.3, 0.3),
        )
        self.failsafe = False
        # Prepare the fullset mesh generator
        self.fullset_mesh_gen = FullSetMeshGenerator(num_segments=64)

    def set_ka(self, ka: Union[Tensor, NDArray] = None):
        ''' Set the ka value for the visualization.

        Args:
            ka: The ka value to use for the visualization. If None, the visualization will be set to failsafe mode.
        '''
        if ka is not None:
            self.failsafe = False
            self.ka = torch.as_tensor(ka/self.g_ka, dtype=self.planner.dtype, device=self.planner.sphere_device)
            bpz = self.planner.SFO_constraint.centers_bpz
            self.joint_centers = bpz.center_slice_all_dep(self.ka)
            n_joints = self.planner.SFO_constraint.n_joints
            n_time = self.planner.SFO_constraint.n_time
            self.joint_radii = self.planner.SFO_constraint.radii[:n_joints*n_time].view(n_joints, n_time).detach()
        else:
            self.failsafe = True
    
    def render_callback(self, env: KinematicUrdfBase, timestep: float = None):
        ''' Render the visualization callback for the given environment.

        Args:
            env: The environment to render the callback for.
            timestep: The timestep to render the callback for. Required for the step visualization.
        '''
        if self.plot_full_set:
            return self._render_callback_full(env)
        else:
            return self._render_callback_step(env, timestep)

    def _render_callback_full(self, env):
        raise NotImplementedError
        if self.ka is None:
            return None

        # Get the meshes for the full set
        all_meshes = self.fullset_mesh_gen(self.joint_centers, self.joint_radii, self.planner.p_idx, self.planner.spheres_per_link)

        # Create the spheres in the scene for pyrender
        if env.renderer == RENDERER.PYRENDER or env.renderer == RENDERER.PYRENDER_OFFSCREEN:
            pyrender_mesh = pyrender.Mesh.from_trimesh(all_meshes, material=self.rs_mat if not self.failsafe else self.rs_mat_failsafe)
            node = env.scene.add(pyrender_mesh)
            return lambda: env.scene.remove_node(node)
        
        elif env.renderer == RENDERER.BLENDER:
            import bpy
            # if self.failsafe:
            #     for obj in self.all_last_obj:
            #         obj.hide_render = False
            #         obj.hide_viewport = False
            #         obj.keyframe_delete(data_path="hide_render")
            #         obj.keyframe_delete(data_path="hide_viewport")
            if self.failsafe:
                sphere_mat = bpy.data.materials.get("SphereFullsetMatFailsafe")
                if sphere_mat is None:
                    sphere_mat = bpy.data.materials.new(name="SphereFullsetMatFailsafe")
                    sphere_mat.use_nodes = True
            else:
                sphere_mat = bpy.data.materials.get("SphereFullsetMat")
                if sphere_mat is None:
                    sphere_mat = bpy.data.materials.new(name="SphereFullsetMat")
                    sphere_mat.use_nodes = True
            itr = getattr(self, 'itr', 1)
            viz_collection = getattr(self, 'viz_collection', None)
            if viz_collection is None:
                viz_collection = bpy.data.collections.new('sphere_fullset_viz')
                env.scene.collection.children.link(viz_collection)
                self.viz_collection = viz_collection
            
            viz_subcollection = bpy.data.collections.new(f'sphere_fullset_viz_{itr}')
            viz_collection.children.link(viz_subcollection)
            current_frame = env.scene.frame_current
            self.all_last_obj = []
            for i, mesh in enumerate(all_meshes):
                name = f'sphere_fullset_mesh_{itr}_{i}'
                bpy_mesh = bpy.data.meshes.new(name)
                bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=False)
                bpy_mesh.validate()
                bpy_mesh.update()
                bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                bpy_obj.hide_render = True
                bpy_obj.hide_viewport = True
                bpy_obj.keyframe_insert(data_path="hide_render", frame=0)
                bpy_obj.keyframe_insert(data_path="hide_viewport", frame=0)
                bpy_obj.hide_render = False
                bpy_obj.hide_viewport = False
                bpy_obj.keyframe_insert(data_path="hide_render", frame=current_frame)
                bpy_obj.keyframe_insert(data_path="hide_viewport", frame=current_frame)
                bpy_obj.data.materials.append(sphere_mat)
                viz_subcollection.objects.link(bpy_obj)
                self.all_last_obj.append(bpy_obj)
            self.itr = itr + 1

            def hide_all():
                for obj in self.all_last_obj:
                    obj.hide_render = True
                    obj.hide_viewport = True
                    obj.keyframe_insert(data_path="hide_render")
                    obj.keyframe_insert(data_path="hide_viewport")

            return hide_all

    def _render_callback_step(self, env, timestep):
        if self.ka is None:
            return None
        timestep = timestep if not self.failsafe else timestep + self.t_plan

        # Get the meshes for the interpolated spheres
        meshes = tapered_meshes_at_time(timestep, self.joint_centers, self.joint_radii, self.planner.p_idx, t_full=self.t_full)

        # Create the spheres in the scene for pyrender
        if env.renderer == RENDERER.PYRENDER or env.renderer == RENDERER.PYRENDER_OFFSCREEN:
            pyrender_mesh = pyrender.Mesh.from_trimesh(meshes, material=self.rs_mat if not self.failsafe else self.rs_mat_failsafe)
            node = env.scene.add(pyrender_mesh)
            return lambda: env.scene.remove_node(node)
    
        elif env.renderer == RENDERER.BLENDER:
            import bpy
            # if self.failsafe:
            #     for obj in self.all_last_obj:
            #         obj.hide_render = False
            #         obj.hide_viewport = False
            #         obj.keyframe_delete(data_path="hide_render")
            #         obj.keyframe_delete(data_path="hide_viewport")
            if self.failsafe:
                tapered_mat = bpy.data.materials.get("TaperedReachsetMatFailsafe")
                if tapered_mat is None:
                    tapered_mat = bpy.data.materials.new(name="TaperedReachsetMatFailsafe")
                    tapered_mat.use_nodes = True
            else:
                tapered_mat = bpy.data.materials.get("TaperedReachsetMat")
                if tapered_mat is None:
                    tapered_mat = bpy.data.materials.new(name="TaperedReachsetMat")
                    tapered_mat.use_nodes = True
            itr = getattr(self, 'itr', 1)
            viz_collection = getattr(self, 'viz_collection', None)
            if viz_collection is None:
                viz_collection = bpy.data.collections.new('tapered_reachset_viz')
                env.scene.collection.children.link(viz_collection)
                self.viz_collection = viz_collection
            
            viz_subcollection = bpy.data.collections.new(f'tapered_reachset_viz_{itr}')
            viz_collection.children.link(viz_subcollection)
            current_frame = env.scene.frame_current
            self.all_last_obj = []
            for i, mesh in enumerate(meshes):
                name = f'tapered_fullset_mesh_{itr}_{i}'
                bpy_mesh = bpy.data.meshes.new(name)
                bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=False)
                bpy_mesh.validate()
                bpy_mesh.update()
                bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                bpy_obj.hide_render = True
                bpy_obj.hide_viewport = True
                bpy_obj.keyframe_insert(data_path="hide_render", frame=0)
                bpy_obj.keyframe_insert(data_path="hide_viewport", frame=0)
                bpy_obj.hide_render = False
                bpy_obj.hide_viewport = False
                bpy_obj.keyframe_insert(data_path="hide_render", frame=current_frame)
                bpy_obj.keyframe_insert(data_path="hide_viewport", frame=current_frame)
                bpy_obj.data.materials.append(tapered_mat)
                viz_subcollection.objects.link(bpy_obj)
                self.all_last_obj.append(bpy_obj)
            self.itr = itr + 1

            def hide_all():
                for obj in self.all_last_obj:
                    obj.hide_render = True
                    obj.hide_viewport = True
                    obj.keyframe_insert(data_path="hide_render")
                    obj.keyframe_insert(data_path="hide_viewport")

            return hide_all
    # def _build_blender_material(self, env, main_rgba, failsafe_rgba):
        


class FullSetMeshGenerator:
    ''' Mesh generator for the full set visualization of the SFO. '''
    def __init__(
            self,
            num_segments: int = 64,
            ) -> None:
        ''' Initialize the mesh generator for the full set visualization. 
        
        Args:
            num_segments: The number of segments to use for the sphere mesh, which is split into two hemispheres.
        '''
        self.num_segments = num_segments
        temp = trimesh.creation.uv_sphere(count=(self.num_segments/2, self.num_segments))
        num_faces = len(temp.faces)
        faces = temp.faces[:int(num_faces/2)]
        num_verts = np.max(faces)+1
        vertices = np.empty((num_verts,3))
        vertices[:,0] = temp.vertices[:num_verts,0]
        vertices[:,1] = -temp.vertices[:num_verts,2]
        vertices[:,2] = temp.vertices[:num_verts,1]
        self.hemisphere_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    def __call__(
            self,
            joint_centers: Tensor,
            joint_radii: Tensor,
            joint_pair_idxs: Tensor,
            spheres_per_pair: int,
            ) -> List[Trimesh]:
        ''' Generate the meshes for the full set visualization.

        Args:
            joint_centers: The centers of the joint spheres. This is shape (n_joints, n_times, 3).
            joint_radii: The radii of the joint spheres. This is shape (n_joints, n_times).
            joint_pair_idxs: The indices of the joint pairs. This is shape (2, n_links).
            spheres_per_pair: The number of spheres to use for each link.
        
        Returns:
            A list of meshes, one for each sphere forward set.
        '''
        n_times = joint_centers.shape[1]
        # consider only half the number of spheres
        # get all spheres
        centers = joint_centers[joint_pair_idxs]
        radii = joint_radii[joint_pair_idxs]
        link_centers, link_radii = make_spheres(centers[0], centers[1], radii[0], radii[1], n_spheres=spheres_per_pair)
        centers = torch.cat((joint_centers, link_centers.transpose(1,2).reshape(-1,n_times,3)), dim=0).cpu().numpy()
        radii = torch.cat((joint_radii, link_radii.transpose(1,2).reshape(-1,n_times)), dim=0).cpu().numpy()
        all_meshes = []
        for path_centers, path_radii in zip(centers, radii):
            # only consider the outermost spheres
            outer_mask = np.any(np.linalg.norm(path_centers[:,None,:] - path_centers[None,:,:], axis=2) + (path_radii[:,None] - path_radii[None,:]) < 0, axis=1)
            outer_mask = np.logical_not(outer_mask)
            path_centers = path_centers[outer_mask]
            path_radii = path_radii[outer_mask]
            if len(path_centers) == 1:
                radius = np.max(path_radii)
                all_meshes.append(trimesh.primitives.Sphere(radius=radius, center=path_centers[0]))
                continue
            d = path_centers[1:] - path_centers[:-1]
            if d.sum() == 0:
                radius = np.max(path_radii)
                all_meshes.append(trimesh.primitives.Sphere(radius=radius, center=path_centers[0]))
                continue
            c = np.zeros_like(path_centers)
            c[:-1] = d; c[1:] = d
            dirs = c / np.linalg.norm(c, axis=1, keepdims=True)
            basis = [normal_vec_to_basis(dir) for dir in dirs]
            angles = np.linspace(0, 2*np.pi, self.num_segments)
            x_comp = np.cos(angles)
            y_comp = np.sin(angles)
            circle_points = [(basis[i][:,0:1]*x_comp + basis[i][:,1:2]*y_comp).T*path_radii[i] + path_centers[i] for i in range(len(basis))]
            circle_points = np.vstack(circle_points)
            n_circles = len(path_radii)
            faces = np.empty((6, (n_circles-1)*self.num_segments-1))
            faces[:] = np.arange(0, (n_circles-1)*self.num_segments-1)
            faces[[1,2,5]] += self.num_segments
            faces[[1,4,5]] += 1
            faces = faces.reshape(2,3,-1).transpose((0,2,1)).reshape(-1,3)
            central_mesh = trimesh.Trimesh(vertices=circle_points, faces=faces)

            # Generate the endcaps
            # Create a transformation matrix from the origin to the first basis flipped 180 about x
            transformation_0 = np.eye(4)
            transformation_0[:3,:3] = basis[0] * path_radii[0]
            transformation_0[:3,3] = path_centers[0]
            transformation_0[:3,[0,2]] *= -1

            # Create a transformation matrix from the origin to the last basis
            transformation_1 = np.eye(4)
            transformation_1[:3,:3] = basis[-1] * path_radii[-1]
            transformation_1[:3,3] = path_centers[-1]
            
            # Combine it all
            endcap_0 = self.hemisphere_mesh.copy()
            endcap_0.apply_transform(transformation_0)
            endcap_1 = self.hemisphere_mesh.copy()
            endcap_1.apply_transform(transformation_1)

            # not watertight but close enough for viz
            full_mesh = trimesh.util.concatenate([central_mesh, endcap_0, endcap_1])
            all_meshes.append(full_mesh)
        return all_meshes


def tapered_meshes_at_time(
        timestep: float,
        joint_centers: Tensor,
        joint_radii: Tensor, 
        joint_pair_idxs: Tensor,
        t_full: float = 1.0,
        ) -> List[Trimesh]:
    ''' Generate the meshes for visualization at a given timestep. Interpolates so not technically correct for usage, but is visually representative.
    
    Args:
        timestep: The timestep to generate the meshes for.
        joint_centers: The centers of the joint spheres. This is shape (n_joints, n_times, 3).
        joint_radii: The radii of the joint spheres. This is shape (n_joints, n_times).
        joint_pair_idxs: The indices of the joint pairs. This is shape (2, n_links).
        spheres_per_pair: The number of spheres to use for each link.
        t_full: The full time of the trajectory.
        
    Returns:
        A list of meshes, one for each sphere forward set.
    '''
    # Interpolate the spheres
    idx = timestep / (t_full/joint_centers.shape[1])
    idx = min(max(idx-1, 0), joint_centers.shape[1]-1)
    # interpolate the idx
    idx_1 = int(idx)
    idx_2 = idx_1 + 1 if idx_1 < joint_centers.shape[1]-1 else idx_1
    prop_2 = idx - idx_1

    # get the proportion of each value to interpolate
    joint_centers = joint_centers[:,idx_1]*(1-prop_2) + joint_centers[:,idx_2]*prop_2
    joint_radii = joint_radii[:,idx_1]*(1-prop_2) + joint_radii[:,idx_2]*prop_2
    joint_centers = joint_centers.cpu().numpy()
    joint_radii = joint_radii.cpu().numpy()
    joint_spheres = np.array([trimesh.primitives.Sphere(radius=radius, center=center) for center, radius in zip(joint_centers, joint_radii)])
    
    jp_idxs = joint_pair_idxs.cpu().numpy()
    pairs = joint_spheres[jp_idxs].T
    tcapsules = [(spheres[0] + spheres[1]).convex_hull for spheres in pairs]

    return tcapsules
