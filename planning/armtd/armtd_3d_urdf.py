# TODO VALIDATE
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import torch
import numpy as np
import zonopy as zp
from zonopy import batchZonotope
from forward_occupancy.JRS import OfflineJRS
from forward_occupancy.FO import forward_occupancy as forward_occupancy
import cyipopt


import time

T_PLAN, T_FULL = 0.5, 1.0

from typing import List
from planning.common.base_armtd_nlp_problem import BaseArmtdNlpProblem
from planning.armtd.armtd_nlp_problem import OfflineArmtdFoConstraints
import zonopyrobots as zpr

class ARMTD_3D_planner():
    def __init__(self,
                 robot: zpr.ZonoArmRobot,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 200,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 include_end_effector: bool = False,
                 use_weighted_cost: bool = False,
                 filter_obstacles: bool = False,
                 linear_solver: str = 'ma27',
                 ):
        '''ARMTD 3D receding horizon arm robot planner
        
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
            include_end_effector: bool, Optional
                Whether to include the end effector in the forward occupancy calculations. Default is False.
            use_weighted_cost: bool, Optional
                Whether to use a weighted cost for the NLP. Default is False.
            filter_obstacles: bool, Optional
                Whether to filter out obstacles that are not in reach of the robot. Default is False.
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
        self.include_end_effector = include_end_effector

        self._setup_robot(robot)

        # Prepare the nlp
        self.filter_obstacles = filter_obstacles
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
        
    def _prepare_FO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope,
                                ):
        # constant
        n_obs = len(obs_zono)

        ### get the forward occupancy
        FO_gen_time = time.perf_counter()
        FO_links, _ = forward_occupancy(JRS_R, self.robot, self.zono_order)
        # let's assume we can ignore the base link and convert to a list of pz's
        # end effector link is a thing???? (n actuated_joints = 7, n links = 8)
        FO_links = list(FO_links.values())[1:]
        if not self.include_end_effector:
            FO_links = FO_links[:-1]
        self.FO_links = FO_links
        # two more constants
        n_links = len(FO_links)
        n_frs_timesteps = FO_links[0].batch_shape[0]

        ### begin generating constraints
        constraint_gen_time = time.perf_counter()
        out_A = np.empty((n_links),dtype=object)
        out_b = np.empty((n_links),dtype=object)
        out_g_ka = self.g_ka
        out_FO_links = np.empty((n_links),dtype=object)

        # Get the obstacle Z matrix
        obs_Z = obs_zono.Z.unsqueeze(1) # expand dim for timesteps

        obs_in_reach_idx = torch.zeros(n_obs,dtype=bool,device=self.device)
        for idx, FO_link_zono in enumerate(FO_links):
            # buffer the obstacle by Grest
            obs_buff_Grest_Z = torch.cat((obs_Z.expand(-1, n_frs_timesteps, -1, -1), FO_link_zono.Grest.expand(n_obs, -1, -1, -1)), -2)
            obs_buff_Grest = zp.batchZonotope(obs_buff_Grest_Z)
            # polytope with prefegenerated combinations
            # TODO: merge into the function w/ a cache
            A_Grest, b_Grest  = obs_buff_Grest.polytope(self.combs)

            if self.filter_obstacles:
                # Further overapproximate to identify which obstacles we need to care about
                FO_overapprox_center = zp.batchZonotope(FO_link_zono.Z[...,:FO_link_zono.n_dep_gens+1, :].expand(n_obs, -1, -1, -1))
                obs_buff_overapprox = obs_buff_Grest + (-FO_overapprox_center)
                # polytope with prefegenerated combinations
                # TODO: merge into the function w/ a cache
                _, b_obs_overapprox = obs_buff_overapprox.reduce(3).polytope(self.combs)

                obs_in_reach_idx += (torch.min(b_obs_overapprox.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)
            # only save the FO center
            # TODO: add dtype and device args
            out_FO_links[idx] = zp.batchPolyZonotope(FO_link_zono.Z[...,:FO_link_zono.n_dep_gens+1, :], FO_link_zono.n_dep_gens, FO_link_zono.expMat, FO_link_zono.id).cpu()
            out_A[idx] = A_Grest.cpu()
            out_b[idx] = b_Grest.cpu()

        # Select for the obs in reach only
        if self.filter_obstacles:
            for idx in range(n_links):
                out_A[idx] = out_A[idx][obs_in_reach_idx.cpu()]
                out_b[idx] = out_b[idx][obs_in_reach_idx.cpu()]
            out_n_obs_in_frs = int(obs_in_reach_idx.sum())
        else:
            out_n_obs_in_frs = n_obs
        final_time = time.perf_counter()
        out_times = {
            'FO_gen': constraint_gen_time - FO_gen_time,
            'constraint_gen': final_time - constraint_gen_time,
        }
        FO_constraint = OfflineArmtdFoConstraints(dtype=self.dtype)
        FO_constraint.set_params(out_FO_links, out_A, out_b, out_g_ka, out_n_obs_in_frs, self.dof)
        return FO_constraint, out_times

    def trajopt(self, qpos, qvel, waypoint, ka_0, FO_constraint, time_limit=None, t_final_thereshold=0.):
        # Moved to another file
        self.nlp_problem_obj.reset(qpos, qvel, waypoint.pos, FO_constraint, t_final_thereshold=t_final_thereshold, qdgoal=waypoint.vel)
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
        if time_limit is not None:
            nlp.add_option('max_wall_time', max(time_limit, 0.001))

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)
        if self.info['status'] == -12:
            raise ValueError(f"Invalid option or solver specified. Perhaps {self.linear_solver} isn't available?")
        self.final_cost = self.info['obj_val'] if self.info['status'] == 0 else None
        return FO_constraint.g_ka * k_opt, self.info['status'], self.nlp_problem_obj.constraint_times
        
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
        obs_zono = batchZonotope(obs_Z)

        # Compute FO
        FO_constraint, FO_times = self._prepare_FO_constraints(JRS_R, obs_zono)
        preparation_time = time.perf_counter() - JRS_process_time_start

        # preproc_time, FO_gen_time, constraint_time = self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)
        if time_limit is not None:
            time_limit = time_limit - preparation_time

        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, waypoint, ka_0, FO_constraint, time_limit=time_limit, t_final_thereshold=t_final_thereshold)
        trajopt_time = time.perf_counter() - trajopt_time
        stats = {
            'cost': self.final_cost,
            'nlp': trajopt_time,
            'total_preparation_time': preparation_time,  
            'constraint_gen': FO_times['constraint_gen'],
            'FO_gen': FO_times['FO_gen'],
            'JRS_process_time': JRS_process_time,
            'constraint_times': constraint_times,
            'num_constraint_evaluations': self.nlp_problem_obj.num_constraint_evaluations,
            'num_jacobian_evaluations': self.nlp_problem_obj.num_jacobian_evaluations,
        }
        return k_opt, flag, stats


if __name__ == '__main__':
    from environments.urdf_obstacle import KinematicUrdfWithObstacles
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### LOAD ROBOT #####
    import os
    import zonopyrobots as zpr
    basedirname = os.path.dirname(zpr.__file__)

    print('Loading Robot')
    # This is hardcoded for now
    rob = zpr.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), device=device, dtype=dtype)
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    ##### SET ENVIRONMENT #####
    env = KinematicUrdfWithObstacles(
        robot=rob.urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=False,
        use_bb_collision=True,
        render_mesh=True,
        reopen_on_close=False,
        obs_size_min = [0.2,0.2,0.2],
        obs_size_max = [0.2,0.2,0.2],
        renderer = 'pyrender-offscreen',
        render_frames = 15,
        render_fps = 30,
        n_obs=5,
        )
    # obs = env.reset()
    obs = env.reset(
        qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
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
    recorder = FullStepRecorder(env, path=os.path.join(os.getcwd(),'armtd.mp4'))

    ##### 2. RUN ARMTD #####    
    planner = ARMTD_3D_planner(rob, device=device, dtype=dtype, use_weighted_cost=False)
    plot_full_set = True
    from visualizations.fo_viz import FOViz
    sphereviz = FOViz(planner, plot_full_set=plot_full_set, t_full=T_FULL)
    env.add_render_callback('foviz', sphereviz.render_callback, needs_time=not plot_full_set)
    use_last_k = False
    total_stats = {}

    #### Setup Waypoint Generator ####
    # Goal doesn't change so just set it here
    from planning.common.waypoints import GoalWaypointGenerator
    waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*3)

    t_armtd = []
    N_EVALS = []
    n_steps = 100
    ka = np.zeros(rob.dof)
    for _ in range(n_steps):
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