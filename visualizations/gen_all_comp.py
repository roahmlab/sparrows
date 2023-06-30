# TODO VALIDATE

import torch
import numpy as np
import zonopy as zp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import time

T_PLAN, T_FULL = 0.5, 1.0

from planning.sparrows.sparrows_urdf import SPARROWS_3D_planner

## OVERRIDE TO ADD THE VIZ STUFF
class extended_spheres(SPARROWS_3D_planner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_SO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope,
                                ):
        ### Get the FO links for viz
        from forward_occupancy.FO import forward_occupancy
        FO_links, _ = forward_occupancy(JRS_R, self.robot, self.zono_order)
        # let's assume we can ignore the base link and convert to a list of pz's
        # end effector link is a thing???? (n actuated_joints = 7, n links = 8)
        FO_links = list(FO_links.values())[1:]
        FO_links = FO_links[:-1]
        self.FO_links = FO_links

        ### Return the normal function
        return super()._prepare_SO_constraints(JRS_R, obs_zono)

if __name__ == '__main__':
    from environments.urdf_obstacle import KinematicUrdfWithObstacles
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
    # if False:
        device = 'cuda:0'
        #device = 'cpu'
        # dtype = torch.float
        dtype = torch.float
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
        check_self_collision=True,
        use_bb_collision=False,
        verbose_obstacle_collision=True,
        verbose_self_collision=True,
        render_mesh=True,
        reopen_on_close=False,
        obs_size_min = [0.2,0.2,0.2],
        obs_size_max = [0.2,0.2,0.2],
        obs_gen_buffer = 0.01,
        info_nearest_obstacle_dist = True,
        renderer = 'pyrender-offscreen',
        render_frames = 30,
        render_fps = 60,
        seed=42,
        n_obs=1,
        viz_goal=False,
        )
    # obs = env.reset()
    temp_qpos = np.array([ 2.14326,  1.07385, -0.18000,  1.55052, -1.37706,  0.21373,  2.53945, ])
    temp_qgoal = np.array([ 2.14326,  1.17810, -0.02793,  1.85878, -1.44164,  0.17453,  1.99840, ])
    temp_qvel = (temp_qgoal - temp_qpos)*2
    obs = env.reset(obs_pos=[np.array([10,10,10])],
        qpos=temp_qpos,
        qvel=temp_qvel,
        qgoal = temp_qgoal,)
    # obs = env.reset(
    #     # qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
    #     qpos=np.array([0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428])-0.2,
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
    #     obs_pos=[
    #         np.array([0.65,-0.46,0.33]),
    #         np.array([0.5,-0.43,0.3]),
    #         np.array([0.47,-0.45,0.15]),
    #         np.array([-0.3,0.2,0.23]),
    #         np.array([0.3,0.2,0.31])
    #         ])
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
    recorder_fo = FullStepRecorder(env, path=os.path.join(os.getcwd(),'mix.fo.mp4'))
    recorder_sphere = FullStepRecorder(env, path=os.path.join(os.getcwd(),'mix.sphere.mp4'))

    ##### 2. RUN ARMTD #####
    import yaml
    fp = "kinova_scenarios/extras/joint_radius_constraint_1arm.yml"
    with open(fp) as stream:
        try:
            joint_rad_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    joint_radius_override = {k: torch.tensor(v, dtype=dtype, device=device) for k,v in joint_rad_yaml['joint_radius_override'].items()}
    # joint_radius_override = {
    #     'joint_1': torch.tensor(0.0503305, dtype=dtype, device=device),
    #     'joint_2': torch.tensor(0.0630855, dtype=dtype, device=device),
    #     'joint_3': torch.tensor(0.0463565, dtype=dtype, device=device),
    #     'joint_4': torch.tensor(0.0634475, dtype=dtype, device=device),
    #     'joint_5': torch.tensor(0.0352165, dtype=dtype, device=device),
    #     'joint_6': torch.tensor(0.0542545, dtype=dtype, device=device),
    #     'joint_7': torch.tensor(0.0364255, dtype=dtype, device=device),
    #     'end_effector': torch.tensor(0.0394685, dtype=dtype, device=device),
    # }
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
    planner = extended_spheres(rob, device=device, sphere_device=device, dtype=dtype, use_weighted_cost=False, 
        joint_radius_override=joint_radius_override, spheres_per_link=5, filter_links=False, check_self_collisions=False)
    
    plot_full_set = True
    from visualizations.sphere_viz import SpherePlannerViz
    from visualizations.fo_viz import FOViz
    sphereviz = SpherePlannerViz(planner, plot_full_set=plot_full_set, t_full=T_FULL)
    foviz = FOViz(planner, plot_full_set=plot_full_set, t_full=T_FULL)
    
    use_last_k = False
    t_armtd = []
    T_NLP = []
    T_SFO = []
    T_NET_PREP = []
    T_PREPROC = []
    T_CONSTR_E = []
    N_EVALS = []
    
    #### Setup Waypoint Generator ####
    # Goal doesn't change so just set it here
    from planning.common.waypoints import GoalWaypointGenerator
    waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*3)

    import tqdm
    total_stats = {}
    n_steps = 2
    ka = np.zeros(rob.dof)
    for _ in tqdm.tqdm(range(n_steps)):
        ts = time.time()
        qpos, qvel = obs['qpos'], obs['qvel']
        obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
        wp = waypoint_generator.get_waypoint(qpos, qvel)
        ka, flag, stats = planner.plan(qpos, qvel, wp, obstacles, ka_0=(ka if use_last_k else None))
        if flag == 0:
            sphereviz.set_ka(ka)
            foviz.set_ka(ka)
        else:
            sphereviz.set_ka(None)
            foviz.set_ka(None)

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
        # if(info['collision_info']['in_collision']):
        #     env.spin()
        #     assert False
        # env.render()
        # env.spin(wait_for_enter=True)
        
        env.add_render_callback('foviz', foviz.render_callback, needs_time=not plot_full_set)
        recorder_fo.capture_frame()
        env.remove_render_callback('foviz')

        env.add_render_callback('spheres', sphereviz.render_callback, needs_time=not plot_full_set)
        recorder_sphere.capture_frame()
        env.remove_render_callback('spheres')
        # if done:
        #     break
    from scipy import stats
    print(f'Total time elasped for ARMTD-3D with {n_steps} steps: {stats.describe(t_armtd)}')
    print("Per step")
    for k, v in total_stats.items():
        print(f'{k}: {stats.describe(v)}')
    print(f'number of constraint evals: {stats.describe(N_EVALS)}')
    # env.spin()
    recorder_fo.close()
    recorder_sphere.close()