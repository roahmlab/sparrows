# TODO VALIDATE

import torch
import numpy as np
import zonopy as zp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from forward_occupancy.JRS import OfflineJRS
from forward_occupancy.SO import sphere_occupancy, make_spheres
import cyipopt

import time

T_PLAN, T_FULL = 0.5, 1.0

from typing import List
from planning.sparrows.sphere_nlp_problem import OfflineSparrowsConstraints
from planning.common.base_armtd_nlp_problem import BaseArmtdNlpProblem
import zonopyrobots as zpr
from zonopyrobots import ZonoArmRobot

from distance_net.compute_vertices_from_generators import compute_edges_from_generators

class SPARROWS_3D_planner():
    def __init__(self,
                 robot: zpr.ZonoArmRobot,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 200,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 sphere_device: torch.device = torch.device('cpu'),
                 spheres_per_link: int = 5,
                 use_weighted_cost: bool = False,
                 joint_radius_override: dict = {},
                 filter_links: bool = False,
                 check_self_collisions: bool = False,
                 linear_solver: str = 'ma27',
                 ):
        '''SPARROWS 3D receding horizon arm robot planner
        
        Args:
            robot: zpr.ZonoArmRobot
                The robot to plan for.
            zono_order: int, Optional
                The order of the zonotope to use for the forward occupancy. Default is 2.
                This usually doesn't need to be changed
            max_combs: int, Optional
                The maximum number of combinations to use for the forward occupancy. Default is 200.
                This usually doesn't need to be changed
            dtype: torch.dtype, Optional
                The datatype to use for the calculations. Default is torch.float.
                This should match the datatype of the robot.
            device: torch.device, Optional
                The device to use for the calculations. Default is torch.device('cpu').
            sphere_device: torch.device, Optional
                The device to use for the sphere calculations. Default is torch.device('cpu').
            spheres_per_link: int, Optional
                The number of spheres to use per link for the sphere calculations. Default is 5.
            use_weighted_cost: bool, Optional
                Whether to use a weighted cost for the NLP. Default is False.
            joint_radius_override: dict, Optional
                A dictionary of joint radius overrides. Default is {}.
                This is used to override the joint radius for the forward occupancy calculations.
            filter_links: bool, Optional
                Whether to filter out links that are too far away from obstacles. Default is False.
            check_self_collisions: bool, Optional
                Whether to check for self collisions. Default is False.
            linear_solver: str, Optional
                The linear solver to use for the NLP. Default is 'ma27'.

        '''
        self.dtype, self.device = dtype, device
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype

        self.robot = robot
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS = OfflineJRS(dtype=self.dtype,device=self.device)
        self.linear_solver = linear_solver
        
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)

        self._setup_robot(robot)
        self.sphere_device = sphere_device
        self.spheres_per_link = spheres_per_link
        self.SFO_constraint = OfflineSparrowsConstraints(dtype=dtype, device=sphere_device, n_params=self.dof)
        self.joint_radius_override = joint_radius_override
        self.filter_links = filter_links
        self.check_self_collisions = check_self_collisions
        self.__preprocess_SO_constants()

        # Prepare the nlp
        self.g_ka = np.ones((self.dof),dtype=self.np_dtype) * self.JRS.g_ka
        # Radius of maximum steady state oscillation that can occur for the given g_ka
        self.osc_rad = self.g_ka*(T_PLAN**2)/8
        self.nlp_problem_obj = BaseArmtdNlpProblem(self.dof,
                                         self.g_ka,
                                         self.pos_lim, 
                                         self.vel_lim,
                                         self.continuous_joints,
                                         self.pos_lim_mask,
                                         self.dtype,
                                         T_PLAN,
                                         T_FULL,
                                         weight_joint_costs = use_weighted_cost)
    
    def _setup_robot(self, robot: zpr.ZonoArmRobot):
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.vel_lim = np.clip(robot.np.vel_lim, a_min=None, a_max=self.JRS.g_ka * T_PLAN)
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        pass

    def _generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i,device=self.device),2) for i in range(max_combs+1)]
    
    def __preprocess_SO_constants(self):
        ## Preprocess some constraint constants
        joint_occ, link_joint_pairs, _ = sphere_occupancy({}, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)
        # Flatten out all link joint pairs so we can just process that
        joint_pairs = []
        for pairs in link_joint_pairs.values():
            joint_pairs.extend(pairs)
        joints_idxs = {name: i for i, name in enumerate(joint_occ.keys())}
        # convert to indices
        p_idx = torch.empty((2, len(joint_pairs)), dtype=int, device=self.sphere_device)
        for i, (joint1, joint2) in enumerate(joint_pairs):
            p_idx[0][i] = joints_idxs[joint1]
            p_idx[1][i] = joints_idxs[joint2]
        self.p_idx = p_idx.to(device=self.sphere_device)

        ## Preprocess self collision constants
        if self.check_self_collisions:
            self.__preprocess_self_collision_constants(joint_occ, joint_pairs, joints_idxs)
    
    def __preprocess_self_collision_constants(self, joint_occ, joint_pairs, joints_idxs):
        # we want to ignore joints that are too close
        joint_centers = torch.stack([centers.c for centers, _ in joint_occ.values()])
        joint_radii = torch.stack([radius for _, radius in joint_occ.values()])*2 # this needs a better multiplier or constant to add based on the JRS or FO
        self_collision_joint_joint = (torch.linalg.vector_norm(joint_centers.unsqueeze(0) - joint_centers.unsqueeze(1), dim=2) > joint_radii)
        # we want all link pairs that don't share joints
        self._self_collision_link_link = [None]*len(joint_pairs)
        self._self_collision_link_joint = [None]*len(joint_pairs)
        for outer_i,jp_outer in enumerate(joint_pairs):
            # get collidable joints
            collidable_vec = torch.logical_and(self_collision_joint_joint[joints_idxs[jp_outer[0]]], self_collision_joint_joint[joints_idxs[jp_outer[1]]])
            def collidable(jp_inner):
                return collidable_vec[joints_idxs[jp_inner[0]]] and collidable_vec[joints_idxs[jp_inner[1]]]
            jp_mask = torch.zeros(len(joint_pairs), dtype=torch.bool, device=self.sphere_device)
            jp_idxs = [inner_i for inner_i, jp_inner in enumerate(joint_pairs) if (jp_inner[0] not in jp_outer and jp_inner[1] not in jp_outer and collidable(jp_inner))]
            jp_mask[jp_idxs] = True
            joint_mask = torch.zeros(len(joint_occ), dtype=torch.bool, device=self.sphere_device)
            joint_idxs = torch.unique(self.p_idx[:,jp_idxs])
            joint_mask[joint_idxs] = True
            self._self_collision_link_link[outer_i] = jp_mask
            self._self_collision_link_joint[outer_i] = joint_mask
        
        # Collision pairs between links and joints
        self._self_collision_link_link = torch.stack(self._self_collision_link_link).tril() # check up to diag
        self._self_collision_link_joint = torch.stack(self._self_collision_link_joint) # check full row
        self._self_collision_joint_joint = self_collision_joint_joint.tril() # check up to diag

    def _prepare_SO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope,
                                ):
        ### Process the obstacles
        dist_net_time = time.perf_counter()

        # Compute hyperplanes from buffered obstacles generators
        hyperplanes_A, hyperplanes_b = obs_zono.to(device=self.sphere_device).polytope(self.combs)
        # Compute vertices from buffered obstacles generators
        v1, v2 = compute_edges_from_generators(obs_zono.Z[...,0:1,:], obs_zono.Z[...,1:,:], hyperplanes_A, hyperplanes_b.unsqueeze(-1))
        # combine to one input for the NN
        obs_tuple = (hyperplanes_A, hyperplanes_b, v1, v2)

        ### get the forward occupancy
        SFO_gen_time = time.perf_counter()
        joint_occ, _, _ = sphere_occupancy(JRS_R, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)
        n_timesteps = JRS_R[0].batch_shape[0]

        # Batch construction
        centers_bpz = zp.stack([pz for pz, _ in joint_occ.values()])
        radii = torch.stack([r for _, r in joint_occ.values()])

        ## Get overapproximative spheres over all timesteps if needed
        if self.filter_links or self.check_self_collisions:
            center_check_int = centers_bpz.to_interval()
            center_check = center_check_int.center()
            rad_check = radii + torch.linalg.vector_norm(center_check_int.rad(),dim=-1)
            joint_centers_check = center_check[self.p_idx]
            joint_radii_check = rad_check[self.p_idx]
            points_check, radii_check = make_spheres(joint_centers_check[0], joint_centers_check[1], joint_radii_check[0], joint_radii_check[1], n_spheres=self.spheres_per_link)

        ## Discard pairs that are too far to be a problem
        obs_link_mask = None
        if self.filter_links:
            dists_out, _ = self.SFO_constraint.NN_fun(points_check.view(-1,3), obs_tuple)
            obs_link_mask = torch.any((dists_out.view(radii_check.shape[0],-1) < radii_check.view(radii_check.shape[0],-1)), dim=-1)
        ## End discard pairs

        ### Self collsion
        # First get each link to link distance
        # spheres[0] are (link_idx, time_idx, sphere_idx, dim)
        # spheres[1] are (link_idx, time_idx, sphere_idx)
        # link_spheres_c are (time_idx, sphere_idx, dim)
        # link_spheres_r are (time_idx, sphere_idx)
        self_collision = None
        if self.check_self_collisions:
            link_link_mask = self._self_collision_link_link.clone()
            for link_idx, (link_spheres_c, link_spheres_r) in enumerate(zip(points_check, radii_check)):
                comp_spheres_c = points_check[self._self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx, dim)
                comp_spheres_r = radii_check[self._self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx)
                delta = link_spheres_c[None,...,None,:] - comp_spheres_c[...,None,:,:]
                dists = torch.linalg.vector_norm(delta, dim=-1)
                surf_dists = dists - link_spheres_r.unsqueeze(-1) - comp_spheres_r.unsqueeze(-2) # (n_comp_links, time_idx, sphere_idx(self), sphere_idx)
                link_link_mask[link_idx][self._self_collision_link_link[link_idx]] = torch.sum(surf_dists < 0,dim=(-1,-2,-3),dtype=bool)
            
            # Then get each link to joint distance
            link_joint_mask = self._self_collision_link_joint.clone()
            for link_idx, (link_spheres_c, link_spheres_r) in enumerate(zip(points_check, radii_check)):
                comp_spheres_c = center_check[self._self_collision_link_joint[link_idx]].unsqueeze(-2) # (n_comp_links, time_idx, 1, dim)
                comp_spheres_r = rad_check[self._self_collision_link_joint[link_idx]].unsqueeze(-1) # (n_comp_links, time_idx, 1)
                delta = link_spheres_c.unsqueeze(0) - comp_spheres_c
                dists = torch.linalg.vector_norm(delta, dim=-1)
                surf_dists = dists - comp_spheres_r - link_spheres_r # (n_comp_links, time_idx, sphere_idx)
                link_joint_mask[link_idx][self._self_collision_link_joint[link_idx]] = torch.sum(surf_dists < 0,dim=(-1,-2),dtype=bool)

            # Then get each joint to joint distance
            # joint_centers_all are (joint_idx, time_idx, dim)
            # joint_radii_all are (joint_idx, time_idx)
            joint_joint_mask = self._self_collision_joint_joint.clone()
            for joint_idx, (joint_spheres_c, joint_spheres_r) in enumerate(zip(center_check, rad_check)):
                comp_spheres_c = center_check[self._self_collision_joint_joint[joint_idx]] # (n_comp_joints, time_idx, dim)
                comp_spheres_r = rad_check[self._self_collision_joint_joint[joint_idx]] # (n_comp_joints, time_idx)
                delta = joint_spheres_c.unsqueeze(0) - comp_spheres_c
                dists = torch.linalg.vector_norm(delta, dim=-1)
                surf_dists = dists - comp_spheres_r - joint_spheres_r # (n_comp_joints, time_idx)
                joint_joint_mask[joint_idx][self._self_collision_joint_joint[joint_idx]] = torch.sum(surf_dists < 0,dim=(-1),dtype=bool)
            
            self_collision = (link_link_mask, link_joint_mask, joint_joint_mask)

        # output range
        out_g_ka = self.g_ka

        # Build the constraint
        self.SFO_constraint.set_params(centers_bpz, radii, self.p_idx, obs_link_mask, out_g_ka, self.spheres_per_link, n_timesteps, obs_tuple, self_collision)

        final_time = time.perf_counter()
        out_times = {
            'SFO_gen': final_time - SFO_gen_time,
            'distance_prep_net': SFO_gen_time - dist_net_time,
        }
        return self.SFO_constraint, out_times

    def trajopt(self, qpos, qvel, waypoint, ka_0, SFO_constraint, time_limit=None, t_final_thereshold=0.):
        self.nlp_problem_obj.reset(qpos, qvel, waypoint.pos, SFO_constraint, t_final_thereshold=t_final_thereshold, qdgoal=waypoint.vel)
        n_constraints = self.nlp_problem_obj.M

        nlp = cyipopt.Problem(
        n = self.dof,
        m = n_constraints,
        problem_obj=self.nlp_problem_obj,
        lb = [-1]*self.dof,
        ub = [1]*self.dof,
        cl = [-1e20]*n_constraints,
        cu = [-1e-6]*n_constraints,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes') # Silent Banner
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-5)
        nlp.add_option('linear_solver', self.linear_solver)
        # nlp.add_option('derivative_test', 'first-order')
        # nlp.add_option('derivative_test_perturbation', 1e-5)
        

        if time_limit is not None:
            nlp.add_option('max_wall_time', max(time_limit, 0.01))

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)
        if self.info['status'] == -12:
            raise ValueError(f"Invalid option or solver specified. Perhaps {self.linear_solver} isn't available?")
        self.final_cost = self.info['obj_val'] if self.info['status'] == 0 else None      
        return SFO_constraint.g_ka * k_opt, self.info['status'], self.nlp_problem_obj.constraint_times
        
    def plan(self,qpos, qvel, waypoint, obs, ka_0 = None, time_limit=None, t_final_thereshold=0.):
        '''Plan a trajectory for the robot

        Args:
            qpos: np.ndarray
                The current joint positions of the robot
            qvel: np.ndarray
                The current joint velocities of the robot
            waypoint: np.ndarray
                The waypoint to plan to
            obs: tuple
                The obstacles in the environment
                The first element is the positions of the obstacles in rows of xyz
                The second element is the sizes of the obstacles in rows of xyz
            ka_0: np.ndarray, Optional
                The initial guess for the joint accelerations. Default is None.
            time_limit: float, Optional
                The time limit for the optimization. Default is None.
            t_final_thereshold: float, Optional
                The final time threshold for the optimization. Default is 0.

        Returns:
            np.ndarray: The joint accelerations
            int: The status of the optimization (0 is success)
            dict: The timing statistics
        '''
        # prepare the JRS
        JRS_process_time_start = time.perf_counter()
        _, JRS_R = self.JRS(qpos, qvel, self.joint_axis)
        JRS_process_time = time.perf_counter() - JRS_process_time_start

        # Create obs zonotopes
        obs_Z = torch.cat((
            torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(-2),
            torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))/2.
            ), dim=-2)
        obs_zono = zp.batchZonotope(obs_Z)

        # Compute FO
        SFO_constraint, SFO_times = self._prepare_SO_constraints(JRS_R, obs_zono)
        preparation_time = time.perf_counter() - JRS_process_time_start
        if time_limit is not None:
            time_limit -= preparation_time
            
        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, waypoint, ka_0, SFO_constraint, time_limit=time_limit, t_final_thereshold=t_final_thereshold)
        trajopt_time = time.perf_counter() - trajopt_time
        
        timing_stats = {
            'cost': self.final_cost,
            'nlp': trajopt_time, 
            'total_prepartion_time': preparation_time,
            'JRS_process_time': JRS_process_time,
            'constraint_times': constraint_times,
            'num_constraint_evaluations': self.nlp_problem_obj.num_constraint_evaluations,
            'num_jacobian_evaluations': self.nlp_problem_obj.num_jacobian_evaluations,
        }
        timing_stats.update(SFO_times)
        
        return k_opt, flag, timing_stats


if __name__ == '__main__':
    from environments.urdf_obstacle import KinematicUrdfWithObstacles
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
    # if False:
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float
        # dtype = torch.double
    else:
        device = 'cpu'
        dtype = torch.float
        # dtype = torch.double

    ##### LOAD ROBOT #####
    import os

    print('Loading Robot')
    # This is hardcoded for now
    import zonopyrobots as robots2
    basedirname = os.path.dirname(robots2.__file__)

    robots2.DEBUG_VIZ = False
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), device=device, dtype=dtype, create_joint_occupancy=True)
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    ##### SET ENVIRONMENT #####
    env = KinematicUrdfWithObstacles(
        robot=rob.urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=False,
        use_bb_collision=False,
        render_mesh=True,
        reopen_on_close=False,
        obs_size_min = [0.2,0.2,0.2],
        obs_size_max = [0.2,0.2,0.2],
        obs_gen_buffer = 0.01,
        info_nearest_obstacle_dist = True,
        renderer = 'blender',
        n_obs=5,
        )
    # obs = env.reset()
    obs = env.reset(
        qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
        # qpos=np.array([0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428])-0.2,
        qvel=np.array([0,0,0,0,0,0,0.]),
        qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
        obs_pos=[
            np.array([0.65,-0.46,0.33]),
            np.array([0.5,-0.43,0.3]),
            np.array([0.47,-0.45,0.15]),
            np.array([-0.3,0.2,0.23]),
            np.array([0.3,0.2,0.31])
            ])
    # obs = env.reset(
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
    from environments.fullstep_recorder import FullStepRecorder
    recorder = FullStepRecorder(env, path=os.path.join(os.getcwd(),'spheres.mp4'))

    ##### 2. RUN ARMTD #####
    joint_radius_override = {
        'joint_1': torch.tensor(0.0503305, dtype=dtype, device=device),
        'joint_2': torch.tensor(0.0630855, dtype=dtype, device=device),
        'joint_3': torch.tensor(0.0463565, dtype=dtype, device=device),
        'joint_4': torch.tensor(0.0634475, dtype=dtype, device=device),
        'joint_5': torch.tensor(0.0352165, dtype=dtype, device=device),
        'joint_6': torch.tensor(0.0542545, dtype=dtype, device=device),
        'joint_7': torch.tensor(0.0364255, dtype=dtype, device=device),
        'end_effector': torch.tensor(0.0394685, dtype=dtype, device=device),
    }
    # joint_radius_override = {
    #     'joint_1_rob1': torch.tensor(0.0503305, dtype=dtype, device=device),
    #     'joint_2_rob1': torch.tensor(0.0630855, dtype=dtype, device=device),
    #     'joint_3_rob1': torch.tensor(0.0463565, dtype=dtype, device=device),
    #     'joint_4_rob1': torch.tensor(0.0634475, dtype=dtype, device=device),
    #     'joint_5_rob1': torch.tensor(0.0352165, dtype=dtype, device=device),
    #     'joint_6_rob1': torch.tensor(0.0542545, dtype=dtype, device=device),
    #     'joint_7_rob1': torch.tensor(0.0364255, dtype=dtype, device=device),
    #     'end_effector_rob1': torch.tensor(0.0394685, dtype=dtype, device=device),
    #     'joint_1_rob2': torch.tensor(0.0503305, dtype=dtype, device=device),
    #     'joint_2_rob2': torch.tensor(0.0630855, dtype=dtype, device=device),
    #     'joint_3_rob2': torch.tensor(0.0463565, dtype=dtype, device=device),
    #     'joint_4_rob2': torch.tensor(0.0634475, dtype=dtype, device=device),
    #     'joint_5_rob2': torch.tensor(0.0352165, dtype=dtype, device=device),
    #     'joint_6_rob2': torch.tensor(0.0542545, dtype=dtype, device=device),
    #     'joint_7_rob2': torch.tensor(0.0364255, dtype=dtype, device=device),
    #     'end_effector_rob2': torch.tensor(0.0394685, dtype=dtype, device=device),
    # }
    planner = SPARROWS_3D_planner(rob, device=device, sphere_device=device, dtype=dtype, use_weighted_cost=False, 
        joint_radius_override=joint_radius_override, spheres_per_link=5, filter_links=True, check_self_collisions=False)

    from visualizations.sphere_viz import SpherePlannerViz
    plot_full_set = False
    sphereviz = SpherePlannerViz(planner, plot_full_set=plot_full_set, t_full=T_FULL)
    env.add_render_callback('spheres', sphereviz.render_callback, needs_time=not plot_full_set)

    use_last_k = False
    t_armtd = []
    T_NLP = []
    T_SFO = []
    T_NET_PREP = []
    T_PREPROC = []
    T_CONSTR_E = []
    N_EVALS = []
    # n_steps = 100
    # for _ in range(n_steps):
    #     ts = time.time()
    #     qpos, qvel, qgoal = obs['qpos'], obs['qvel'], obs['qgoal']
    #     obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
    #     ka, flag, tnlp, tsfo, tdnp, tpreproc, tconstraint_evals = planner.plan(qpos, qvel, qgoal, obstacles)
    #     t_elasped = time.time()-ts
    #     #print(f'Time elasped for ARMTD-3d:{t_elasped}')
    #     T_NLP.append(tnlp)
    #     T_SFO.append(tsfo)
    #     T_NET_PREP.append(tdnp)
    #     T_PREPROC.append(tpreproc)
    #     T_CONSTR_E.extend(tconstraint_evals)
    #     N_EVALS.append(len(tconstraint_evals))
    #     t_armtd.append(t_elasped)
    #     if flag != 0:
    #         print("executing failsafe!")
    #         ka = (0 - qvel)/(T_FULL - T_PLAN)
    #     obs, rew, done, info = env.step(ka)
    #     # env.step(ka,flag)
    #     # assert(not info['collision_info']['in_collision'])
    #     # env.render()
    # from scipy import stats
    # print(f'Total time elasped for ARMTD-3D with {n_steps} steps: {stats.describe(t_armtd)}')
    # print("Per step")
    # print(f'NLP: {stats.describe(T_NLP)}')
    # print(f'constraint evals: {stats.describe(T_CONSTR_E)}')
    # print(f'number of constraint evals: {stats.describe(N_EVALS)}')
    # print(f'Distance net prep: {stats.describe(T_NET_PREP)}')
    # print(f'SFO generation: {stats.describe(T_SFO)}')
    # print(f'JRS preprocessing: {stats.describe(T_PREPROC)}')
    
    #### Setup Waypoint Generator ####
    # Goal doesn't change so just set it here
    from planning.common.waypoints import GoalWaypointGenerator
    waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*3)

    total_stats = {}
    n_steps = 10
    ka = np.zeros(rob.dof)
    from tqdm import tqdm
    for _ in tqdm(range(n_steps)):
        ts = time.time()
        qpos, qvel = obs['qpos'], obs['qvel']
        obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
        wp = waypoint_generator.get_waypoint(qpos, qvel)
        ka, flag, stats = planner.plan(qpos, qvel, wp, obstacles, ka_0=(ka if use_last_k else None))
        if flag == 0:
            sphereviz.set_ka(ka)
        else:
            sphereviz.set_ka(None)

        t_elasped = time.time()-ts
        t_armtd.append(t_elasped)
        
        for k, v in stats.items():
            if v is None:
                continue
            if k not in total_stats:
                total_stats[k] = []
            if k == 'constraint_times':
                total_stats[k] += v
                N_EVALS.append(len(v))
            else:
                total_stats[k].append(v)
                
        if flag != 0:
            print("executing failsafe!")
            ka = (0 - qvel)/(T_FULL - T_PLAN)
        obs, rew, done, info = env.step(ka)
        # env.step(ka,flag)
        assert(not info['collision_info']['in_collision'])
        # env.render()
        recorder.capture_frame()
    from scipy import stats
    print(f'Total time elasped for ARMTD-3D with {n_steps} steps: {stats.describe(t_armtd)}')
    print("Per step")
    for k, v in total_stats.items():
        print(f'{k}: {stats.describe(v)}')
    print(f'number of constraint evals: {stats.describe(N_EVALS)}')
    # env.spin()
    recorder.close()