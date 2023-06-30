from __future__ import annotations
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import trimesh
import pyrender
import torch
import zonopy as zp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    from numpy import ndarray as NDArray
    from torch import Tensor
    from environments.urdf_base import KinematicUrdfBase
    from planning.armtd.armtd_3d_urdf import ARMTD_3D_planner


class FOViz:
    def __init__(
            self,
            planner: ARMTD_3D_planner,
            plot_full_set: bool = False,
            addbuffer: float = 0.0,
            t_plan: float = 0.5,
            t_full: float = 1.0,
            ) -> None:
        ''' Initialize the forward occupancy visualization with a planner.
        
        Args:
            planner (ARMTD_3D_planner): The planner.
            plot_full_set (bool, optional): Whether to plot the full set. Defaults to False.
            addbuffer (float, optional): The buffer to add to the set. Defaults to 0.0.
            t_plan (float, optional): The planning time. Defaults to 0.5.
            t_full (float, optional): The full time. Defaults to 1.0.
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
        if addbuffer > 0:
            Z = torch.eye(3, dtype=self.planner.dtype, device=self.planner.device) * addbuffer
            Z = torch.vstack((torch.zeros(3, dtype=self.planner.dtype, device=self.planner.device), Z))
            self.buffer = zp.polyZonotope(Z, copy_Z=False, dtype=self.planner.dtype, device=self.planner.device)
        else:
            self.buffer = zp.polyZonotope.zeros(3, dtype=self.planner.dtype, device=self.planner.device)

    def set_ka(self, ka: Union[Tensor, NDArray] = None):
        ''' Set the ka value for the visualization.

        Args:
            ka: The ka value to use for the visualization. If None, the visualization will be set to failsafe mode.
        '''
        if ka is not None:
            self.failsafe = False
            self.ka = torch.as_tensor(ka/self.g_ka, dtype=self.planner.dtype, device=self.planner.device)
            self.FO_links = self.planner.FO_links
            self.n_frs_timesteps = self.FO_links[0].batch_shape[0]
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
        if self.ka is None:
            return None
        
        meshes = []
        for occupancy in self.FO_links:
            occupancy = occupancy + self.buffer
            from zonopy import batchPolyZonotope
            if isinstance(occupancy, batchPolyZonotope):
                all_patch = []
                for i in range(self.FO_links[0].batch_shape[0]):
                    patch = occupancy[i].slice_all_dep(self.ka).reduce(4).polyhedron_patch()
                    all_patch.append(patch.reshape(-1,3))
                all_patch = torch.cat(all_patch, dim=0)
            else:
                occupancy = occupancy.slice_all_dep(self.ka)
                all_patch = occupancy.reduce(4).polyhedron_patch().reshape(-1,3)
            mesh = trimesh.convex.convex_hull(all_patch)
            meshes.append(mesh)

        pyrender_mesh = pyrender.Mesh.from_trimesh(meshes, material=self.rs_mat if not self.failsafe else self.rs_mat_failsafe, smooth=False)
        node = env.scene.add(pyrender_mesh)
        return lambda: env.scene.remove_node(node)

    def _render_callback_step(self, env, timestep):
        if self.ka is None:
            return None
        timestep = timestep if not self.failsafe else timestep + self.t_plan
        idx = timestep / (self.t_full/self.n_frs_timesteps)
        idx = int(idx)

        meshes = []
        for occupancy in self.FO_links:
            try:
                occupancy = occupancy[idx] + self.buffer
            except:
                occupancy = occupancy + self.buffer
            occupancy = occupancy.slice_all_dep(self.ka)
            patch = occupancy.reduce(4).polyhedron_patch()
            mesh = trimesh.convex.convex_hull(patch.reshape(-1,3))
            meshes.append(mesh)
        
        pyrender_mesh = pyrender.Mesh.from_trimesh(meshes, material=self.rs_mat if not self.failsafe else self.rs_mat_failsafe, smooth=False)
        node = env.scene.add(pyrender_mesh)
        return lambda: env.scene.remove_node(node)
