""" Helper class for visualizing the history of planned robot steps.

Usable with pyrender environments which have a scene attribute.
"""
import pyrender

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

    def get_tm_list(self, env, qpos):
        fk = env.robot.visual_trimesh_fk(qpos)
        tm_list = []
        for tm, pose in fk.items():
            mesh = tm.copy()
            mesh.apply_transform(pose)
            tm_list.append(mesh)
        return tm_list
