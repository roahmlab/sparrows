import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environments.urdf_base import KinematicUrdfBase, STEP_TYPE, RENDERER
import numpy as np
import trimesh

class KinematicUrdfWithObstacles(KinematicUrdfBase):
    """
    Kinematic URDF with obstacles
    """
    def __init__(self,
                 n_obs = 1, # number of obstacles
                 obs_size_max = [0.1,0.1,0.1], # maximum size of randomized obstacles in xyz
                 obs_size_min = [0.1,0.1,0.1], # minimum size of randomized obstacle in xyz
                 obs_gen_buffer = 1e-2, # buffer around obstacles generated
                 goal_threshold = 0.1, # goal threshold
                 goal_use_mesh = False, # use mesh for goal
                 velocity_threshold = 0.1, # velocity threshold
                 verbose_obstacle_collision = False, # verbose obstacle collision
                 info_nearest_obstacle_dist = False, # output nearest obstacle distance
                 viz_goal = True, # visualize goal
                 **kwargs):
        '''Initialize the environment with obstacles
        
        Args:
            n_obs: int
                Number of obstacles.
                This isn't guaranteed to be the number of obstacles in the environment, but is the number of obstacles that will be generated.
                This must be at least 1; however, any number of obstacles can be loaded during reset.
            obs_size_max: list
                Maximum size of the obstacles in xyz.
                This can be a list of 3 floats or a single float.
                If a single float is provided, then the same size will be used for all obstacles.
            obs_size_min: list
                Minimum size of the obstacles in xyz.
                This can be a list of 3 floats or a single float.
                If a single float is provided, then the same size will be used for all obstacles.
            obs_gen_buffer: float
                Buffer around the obstacles generated for robot initialization.
            goal_threshold: float
                Threshold at which the goal is considered reached based on the l2 norm of the joint positions to goal joint positions.
            goal_use_mesh: bool
                Use the mesh for the goal visualization.
            velocity_threshold: float
                Threshold for the velocity of the robot to be considered successful at the goal.
                If it is less than this value, then the robot is considered successful.
            verbose_obstacle_collision: bool
                If true, then the collision pairs will be output in the info dictionary.
            info_nearest_obstacle_dist: bool
                If true, then the nearest obstacle distance will be output in the info dictionary.
            viz_goal: bool
                If true, then the goal will be visualized.
        '''
        super().__init__(**kwargs)

        self.n_obs = n_obs
        self.obs_size_range = np.array([obs_size_min, obs_size_max])
        if len(self.obs_size_range) != 3:
            self.obs_size_range = np.broadcast_to(np.expand_dims(self.obs_size_range,1), (2, self.n_obs, 3))
        self.obs_gen_buffer = obs_gen_buffer
        self.goal_threshold = goal_threshold
        self.goal_use_mesh = goal_use_mesh
        self.velocity_threshold = velocity_threshold
        self.verbose_obstacle_collision = verbose_obstacle_collision
        self.info_nearest_obstacle_dist = info_nearest_obstacle_dist
        if self.use_bb_collision is True and self.info_nearest_obstacle_dist:
            import warnings
            warnings.warn('Nearest obstacle distance will be based on the bounding boxes and may not be accurate!')
        self.qgoal = np.zeros(self.dof)

        self.obstacle_collision_manager = None
        self.obs_list = []
        self.obs_size = None
        self.obs_pos = None

        self.viz_goal = viz_goal

    def reset(self,
              qgoal: np.ndarray = None,
              obs_pos: np.ndarray = None,
              obs_size: np.ndarray = None,
              **kwargs):
        '''Reset the environment with obstacles

        Args:
            qgoal: np.ndarray, Optional
                Goal position for the robot.
                If not provided, then a random goal position will be generated.
            obs_pos: np.ndarray, Optional
                Position of the obstacles.
                If not provided, then random positions will be generated.
            obs_size: np.ndarray, Optional
                Size of the obstacles.
                If not provided, then random sizes will be generated.
            **kwargs: dict
                Additional keyword arguments to pass to the reset function and the super class reset function.
        '''
        # Setup the initial environment
        super().reset(**kwargs)

        # Generate the goal position
        if qgoal is not None:
            self.qgoal = self._wrap_cont_joints(qgoal)
        else:
            # Generate a random position for each of the joints
            # Try 10 times until there is no self collision
            self.qgoal = self._generate_free_configuration(n_tries=10)

        # generate the obstacle sizes we want
        if obs_size is None:
            obs_size = self.np_random.uniform(low=self.obs_size_range[0], high=self.obs_size_range[1])
        obs_size = np.asarray(obs_size)

        # if no position is provided, generate
        if obs_pos is None:
            # create temporary collision managers for start and end
            start_collision_manager = trimesh.collision.CollisionManager()
            goal_collision_manager = trimesh.collision.CollisionManager()
            start_fk = self.robot.link_fk(self.qpos)
            goal_fk = self.robot.link_fk(self.qgoal)
            for (link, start_tf), goal_tf in zip(start_fk.items(), goal_fk.values()):
                if link.collision_mesh is not None:
                    mesh = link.collision_mesh.bounding_box
                    start_collision_manager.add_object(link.name, mesh, transform=start_tf)
                    goal_collision_manager.add_object(link.name, mesh, transform=goal_tf)
                    
            # For each of the obstacles, determine a position.
            # Generate arm configurations and use the end effector location as a candidate position
            # compute non-inf bounds
            pos_lim = np.copy(self.pos_lim)
            pos_lim[np.isneginf(pos_lim)] = -np.pi*3
            pos_lim[np.isposinf(pos_lim)] = np.pi*3
            def pos_helper(mesh):
                new_pos = self.np_random.uniform(low=pos_lim[0], high=pos_lim[1])
                new_pos = self._wrap_cont_joints(new_pos)
                fk_dict = self.robot.link_fk(new_pos, links=self.robot.end_links)
                cand_pos = np.array(list(fk_dict.values()))
                cand_pos[...,0:3,0:3] = np.eye(3)
                if len(cand_pos) > 1:
                    self.np_random.shuffle(cand_pos)
                for pose in cand_pos:
                    coll_start = start_collision_manager.in_collision_single(mesh, pose)
                    coll_end = goal_collision_manager.in_collision_single(mesh, pose)
                    if not (coll_start or coll_end):
                        return pose[0:3,3]
                return None
            
            # generate obstacle
            obs_pos_list = []
            for size in obs_size:
                # Try 10 times per obstacle
                pos = None
                for _ in range(10):
                    cand_mesh = trimesh.primitives.Box(extents=size + self.obs_gen_buffer * 2)
                    pos = pos_helper(cand_mesh)
                    if pos is not None:
                        break
                if pos is None:
                    raise RuntimeError
                obs_pos_list.append(pos)
            obs_pos = np.array(obs_pos_list)
        obs_pos = np.asarray(obs_pos)
        
        # add the obstacles to the obstacle collision manager
        self.obs_pos = obs_pos
        self.obs_size = obs_size
        self.obstacle_collision_manager = trimesh.collision.CollisionManager()
        self.obs_meshes = {}
        for i, (pos, size) in enumerate(zip(obs_pos, obs_size)):
            obs = trimesh.primitives.Box(extents=size)
            pose = np.eye(4)
            pose[0:3,3] = pos
            obs_name = f'obs{i}'
            self.obstacle_collision_manager.add_object(obs_name, obs, transform=pose)
            self.obs_meshes[obs_name] = (obs, pose)
        
        # reset the pyrender becuase I don't feel like tracking obstacles
        self.close()
        
        # return observations
        return self.get_observations()
    
    def get_reward(self, action):
        '''Get the reward for the current state

        Args:
            action: np.ndarray
                Action taken by the robot

        Returns:
            float: Reward for the current state
        '''
        # Get the position and goal then calculate distance to goal
        collision = self.info['collision_info']['in_collision']
        dist = np.linalg.norm(self._wrap_cont_joints(self.qpos - self.qgoal))
        success = (dist < self.goal_threshold) and (not collision) and (np.linalg.norm(self.qvel) < self.velocity_threshold)

        self.done = self.done or success or collision

        if success:
            return 1
        else:
            return 0
    
    def get_observations(self):
        '''Get a deepcopy of the observations for the current state
        
        Returns:
            dict: Observations for the current state
        '''
        observations = super().get_observations()
        observations['qgoal'] = self.qgoal
        observations['obstacle_pos'] = self.obs_pos
        observations['obstacle_size'] = self.obs_size
        return observations
    
    def _create_pyrender_scene(self, *args, obs_mat=None, obs_wireframe=True, goal_mat=None, goal_wireframe=True, **kwargs):
        '''Create the pyrender scene for the environment

        Args:
            *args: list
                Arguments to pass to the super class create pyrender scene function
            obs_mat: pyrender.Material, Optional
                Material to use for the obstacles
            obs_wireframe: bool, Optional
                If true, then the obstacles will be rendered with wireframes as well
            goal_mat: pyrender.Material, Optional
                Material to use for the goal
            goal_wireframe: bool, Optional
                If true, then the goal will be rendered wire wireframe as well
            **kwargs: dict
                Additional keyword arguments to pass to the super class create pyrender scene function
        '''
        import pyrender
        super()._create_pyrender_scene(*args, **kwargs)
        # add the obstacles
        if obs_mat is None:
            obs_mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.4,
                alphaMode='BLEND',
                baseColorFactor=(1.0, 0.0, 0.0, 0.3),
            )
        for mesh, pose in self.obs_meshes.values():
            if obs_wireframe:
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, wireframe=True, material=obs_mat)
                self.scene.add(pyrender_mesh, pose=pose)
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=obs_mat)
            self.scene.add(pyrender_mesh, pose=pose)
        # Add the goal
        if self.viz_goal:
            if goal_mat is None:
                goal_mat = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='BLEND',
                    baseColorFactor=(0.0, 1.0, 0.0, 0.3),
                )
            fk = self.robot.visual_trimesh_fk(self.qgoal)
            for tm, pose in fk.items():
                if self.goal_use_mesh:
                    if goal_wireframe:
                        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False, wireframe=True, material=goal_mat)
                        self.scene.add(mesh, pose=pose)
                    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False, material=goal_mat)
                    self.scene.add(mesh, pose=pose)
                else:
                    if goal_wireframe:
                        mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False, wireframe=True, material=goal_mat)
                        self.scene.add(mesh, pose=pose)
                    mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False, material=goal_mat)
                    self.scene.add(mesh, pose=pose)

    def _create_blender_scene(self, *args, **kwargs):
        import bpy
        super()._create_blender_scene(*args, **kwargs)
        # add the obstacles
        # obs_mat = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.4,
        #     alphaMode='BLEND',
        #     baseColorFactor=(1.0, 0.0, 0.0, 0.3),
        # )
        obs_mat = bpy.data.materials.new(name="ObstacleMaterial")
        ### Pre-set Color
        obs_mat.use_nodes = True
        obs_mat.shadow_method = 'NONE'
        obs_mat.show_transparent_back = False
        obs_mat.blend_method = 'BLEND'
        obs_principled = obs_mat.node_tree.nodes["Principled BSDF"]
        obs_principled.inputs["Base Color"].default_value = (0.947, 0.156, 0.098, 0.5)
        obs_principled.inputs["Alpha"].default_value = 0.5
        ###
        # Add the obstacles
        obs_collection = bpy.data.collections.new('Obstacles')
        self.scene.collection.children.link(obs_collection)
        for i, (mesh, pose) in enumerate(self.obs_meshes.values()):
            bpy_mesh = bpy.data.meshes.new(f"obstacle_{i}")
            bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=True)
            bpy_obj = bpy.data.objects.new(f"obstacle_{i}", bpy_mesh)
            bpy_obj.rotation_mode = 'QUATERNION'
            bpy_obj.location = pose[:3,3]
            bpy_obj.rotation_quaternion = trimesh.transformations.quaternion_from_matrix(pose)
            bpy_obj.keyframe_insert(data_path="location", frame=0)
            bpy_obj.keyframe_insert(data_path="rotation_quaternion", frame=0)
            bpy_obj.data.materials.append(obs_mat)
            obs_collection.objects.link(bpy_obj)
        # Add the goal
        if self.viz_goal:
            # goal_mat = pyrender.MetallicRoughnessMaterial(
            #     metallicFactor=0.2,
            #     alphaMode='BLEND',
            #     baseColorFactor=(0.0, 1.0, 0.0, 0.3),
            # )
            goal_mat = bpy.data.materials.new(name="GoalMaterial")
            ### Pre-set Color
            goal_mat.shadow_method = 'NONE'
            goal_mat.show_transparent_back = False
            hex_value = 0x58AF4A
            b = (hex_value & 0xFF) / 255.0
            g = ((hex_value >> 8) & 0xFF) / 255.0
            r = ((hex_value >> 16) & 0xFF) / 255.0
            goal_mat.use_nodes = True
            goal_mat.blend_method = 'BLEND'
            goal_principled = goal_mat.node_tree.nodes["Principled BSDF"]
            goal_principled.inputs["Base Color"].default_value = (r, g, b, 0.5)
            goal_principled.inputs["Alpha"].default_value = 0.5
            ###
            goal_collection = bpy.data.collections.new('Goal')
            self.scene.collection.children.link(goal_collection)
            fk = self.robot.visual_trimesh_fk(self.qgoal)
            for i, (tm, pose) in enumerate(fk.items()):
                mesh = tm if self.goal_use_mesh else tm.bounding_box
                name = f'robot_goal_{i}'
                bpy_mesh = bpy.data.meshes.new(name)
                bpy_mesh.from_pydata(mesh.vertices, [], mesh.faces, shade_flat=True)
                bpy_obj = bpy.data.objects.new(name, bpy_mesh)
                bpy_obj.rotation_mode = 'QUATERNION'
                bpy_obj.location = pose[:3,3]
                bpy_obj.rotation_quaternion = trimesh.transformations.quaternion_from_matrix(pose)
                bpy_obj.keyframe_insert(data_path="location", frame=0)
                bpy_obj.keyframe_insert(data_path="rotation_quaternion", frame=0)
                bpy_obj.data.materials.append(goal_mat)
                goal_collection.objects.link(bpy_obj)

    def _collision_check_obstacle(self, fk_dict, use_bb=True):
        '''Check for collisions with the obstacles

        Args:
            fk_dict: dict
                Dictionary of link names to transforms
            use_bb: bool
                If true, then use the bounding box for collision checking

        Returns:
            bool: True if there is a collision
            dict: Dictionary of collision pairs
            float: Distance to the nearest obstacle
        '''
        # Use the collision manager to check each of these positions
        if use_bb:
            rob_collision_manager = self.robot_collision_manager_bb
            obs_collision_manager = self.obstacle_collision_manager#_bb
        else:
            rob_collision_manager = self.robot_collision_manager
            obs_collision_manager = self.obstacle_collision_manager
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                rob_collision_manager.set_transform(name, transform)
        if self.info_nearest_obstacle_dist:
            distance, name_pairs = rob_collision_manager.min_distance_other(obs_collision_manager, return_names=True)
            return distance <= 0, name_pairs, distance
        else:
            collision, name_pairs = rob_collision_manager.in_collision_other(obs_collision_manager, return_names=True)
            return collision, name_pairs, None

    def collision_check(self, batch_fk_dict):
        '''Check for collisions with the obstacles

        Args:
            batch_fk_dict: dict
                Dictionary of link names to transforms

        Returns:
            dict: Dictionary of collision information
        '''
        
        out = {}
        out['obstacle_collision'] = False
        collision_pairs = {}
        distance_pair = {'dist': np.inf}
        for i in range(self.collision_discretization):
            time = float(i)/self.collision_discretization * self.t_step
            # Comprehend a name: transform from the batch link fk dictionary
            fk_dict = {link.name: pose[i] for link, pose in batch_fk_dict.items()}
            collision, pairs, distance = self._collision_check_obstacle(fk_dict, use_bb=True)
            # Refine the collision if we aren't using bb collision
            if not self.use_bb_collision and (collision or self.info_nearest_obstacle_dist):
                collision, pairs, distance = self._collision_check_obstacle(fk_dict, use_bb=False)
            if self.info_nearest_obstacle_dist:
                if distance < distance_pair['dist']:
                    distance_pair['time'] = time
                    distance_pair['pairs'] = pairs
                    distance_pair['dist'] = distance
            if self.verbose_obstacle_collision and collision:
                collision_pairs[time] = pairs
            elif collision:
                out['in_collision'] = True
                out['obstacle_collision'] = True
                break
        if len(collision_pairs) > 0:
            out['in_collision'] = True
            out['obstacle_collision'] = collision_pairs
        if self.info_nearest_obstacle_dist:
            out['nearest_obstacle_dist'] = distance_pair
        return out

if __name__ == '__main__':

    # Load robot
    import os
    import zonopyrobots as zpr
    basedirname = os.path.dirname(zpr.__file__)

    print('Loading Robot')
    # This is hardcoded for now
    rob = zpr.ZonoArmRobot(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'))
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    test = KinematicUrdfWithObstacles(robot = rob.robot, step_type='integration', check_joint_limits=True, check_self_collision=True, use_bb_collision=True, render_mesh=True, reopen_on_close=False, n_obs=5, render_fps=30, render_frames=10)
    # test.render()
    test.reset()
    # test.reset(
    #     qpos=np.array([ 3.1098, -0.9964, -0.2729, -2.3615,  0.2724, -1.6465, -0.5739]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([-1.9472,  1.4003, -1.3683, -1.1298,  0.7062, -1.0147, -1.1896]),
    #     obs_pos=[
    #         np.array([ 0.3925, -0.7788,  0.2958]),
    #         np.array([0.3550, 0.3895, 0.3000]),
    #         np.array([-0.0475, -0.1682, -0.7190]),
    #         np.array([0.3896, 0.5005, 0.7413]),
    #         np.array([0.4406, 0.1859, 0.1840]),
    #         np.array([ 0.1462, -0.6461,  0.7416]),
    #         np.array([-0.4969, -0.5828,  0.1111]),
    #         np.array([-0.0275,  0.1137,  0.6985]),
    #         np.array([ 0.4139, -0.1155,  0.7733]),
    #         np.array([ 0.5243, -0.7838,  0.4781])
    #         ])
    # test.reset(
    #     qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
    #     obs_pos=[
    #         np.array([0.65,-0.46,0.33]),
    #         np.array([0.5,-0.43,0.3]),
    #         np.array([0.47,-0.45,0.15]),
    #         np.array([-0.3,0.2,0.23]),
    #         np.array([0.3,0.2,0.31])
    #         ])
    # plt.figure()
    # plt.ion()
    for i in range(302):
        a = test.step(np.random.random(7)-0.5)
        test.render()
        # im, depth = test.render()
        # for i in range(0,len(im),10):
        #     plt.imshow(im[i])
        #     plt.draw()
        #     plt.pause(0.01)
        # test.render(render_fps=5)
        # print(a)

    print("hi")

    # env = Locked_Arm_3D(n_obs=3,T_len=50,interpolate=True,locked_idx=[1,2],locked_qpos = [0,0])
    # for _ in range(3):
    #     for _ in range(10):
    #         env.step(torch.rand(env.dof))
    #         env.render()
    #         #env.reset()