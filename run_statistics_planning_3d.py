import torch
import argparse
import numpy as np
import time
import json
import random
from tqdm import tqdm
from environments.urdf_obstacle import KinematicUrdfWithObstacles
from environments.fullstep_recorder import FullStepRecorder
from planning.armtd.armtd_3d_urdf import ARMTD_3D_planner
from planning.sparrows.sparrows_urdf import SPARROWS_3D_planner
from planning.common.waypoints import GoalWaypointGenerator
from visualizations.fo_viz import FOViz
from visualizations.sphere_viz import SpherePlannerViz
import os
T_PLAN, T_FULL = 0.5, 1.0

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def evaluate_planner(planner, 
                     planner_name='sphere', 
                     n_envs=1000, 
                     n_steps=150, 
                     n_links=7, 
                     n_obs=5, 
                     save_success_trial_id=False, 
                     video=False, 
                     reachset_viz=False, 
                     time_limit=0.5, 
                     detail=True, 
                     t_final_thereshold=0.,
                     check_self_collision=False
                    ):
    t_armtd = 0.0
    num_success = 0
    num_collision = 0
    num_no_solution = 0
    num_step = 0
    t_armtd_list = []
    t_success_list = []
    
    planner_stats = {}
    if save_success_trial_id:
        success_episodes = []
    if video:
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        video_folder = f'planning_videos/{planner_name}/3d{n_links}links{n_obs}obs'
        if reachset_viz:
            video_folder += '_reachset'
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    if detail:
        import pickle
        planning_details = {}
        trial_details = {}
    
    env_args = dict(
            step_type='integration',
            check_joint_limits=True,
            check_self_collision=check_self_collision,
            use_bb_collision=False,
            render_mesh=True,
            reopen_on_close=False,
            obs_size_min = [0.2,0.2,0.2],
            obs_size_max = [0.2,0.2,0.2],
            n_obs=n_obs,
            renderer = 'pyrender-offscreen',
            info_nearest_obstacle_dist = False,
            obs_gen_buffer = 0.01
        )
    env = KinematicUrdfWithObstacles(
            robot=rob.urdf,
            **env_args
        )
    if video and reachset_viz:
        if 'sphere' in planner_name:
            viz = SpherePlannerViz(planner, plot_full_set=True, t_full=T_FULL)
        elif 'armtd' in planner_name:
            viz = FOViz(planner, plot_full_set=True, t_full=T_FULL)
        else:
            raise NotImplementedError(f"Visualizer for {planner_name} type has not been implemented yet.")
        env.add_render_callback('spheres', viz.render_callback, needs_time=False)
    for i_env in tqdm(range(n_envs)):
        set_random_seed(i_env)
        obs = env.reset()
        waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*3)
        if detail:
            planning_details[i_env] = {
                'initial': obs,
                'trajectory': {
                    'k': [],
                    'flag': [],
                    'nearest_distance': []
                }
            }
        t_curr_trial = []
        if video:
            video_path = os.path.join(video_folder, f'video{i_env}.mp4')
            video_recorder = FullStepRecorder(env, path=video_path)
        was_stuck = False
        force_fail_safe = False
        for i_step in range(n_steps):
            qpos, qvel, qgoal = obs['qpos'], obs['qvel'], obs['qgoal']
            obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
            waypoint = waypoint_generator.get_waypoint(qpos, qvel)
            ts = time.time()
            ka, flag, planner_stat = planner.plan(qpos, qvel, waypoint, obstacles, time_limit=time_limit, t_final_thereshold=t_final_thereshold)
            t_elasped = time.time()-ts
            t_armtd += t_elasped
            t_armtd_list.append(t_elasped)
            t_curr_trial.append(t_elasped)
            for key in planner_stat:
                if planner_stat[key] is None:
                    continue      
                if key in planner_stats:
                    if isinstance(planner_stat[key], list):
                        planner_stats[key] += planner_stat[key]
                    else:
                        planner_stats[key].append(planner_stat[key])
                else:
                    if isinstance(planner_stat[key], list):
                        planner_stats[key] = planner_stat[key]
                    else:
                        planner_stats[key] = [planner_stat[key]]

            if flag != 0:
                ka = (0 - qvel)/(T_FULL - T_PLAN)
            if force_fail_safe:
                ka = (0 - qvel)/(T_FULL - T_PLAN)
                force_fail_safe = False
            else:
                force_fail_safe = (flag == 0) and planner.nlp_problem_obj.use_t_final and (np.sqrt(planner.final_cost) < env.goal_threshold)
                
            if video and reachset_viz:
                if flag == 0:
                    viz.set_ka(ka)
                else:
                    viz.set_ka(None)
            obs, reward, done, info = env.step(ka)
            if video:
                video_recorder.capture_frame()
            if detail:
                planning_details[i_env]['trajectory']['k'].append(ka)
                planning_details[i_env]['trajectory']['flag'].append(flag)
            
            num_step += 1
            if info['collision_info']['in_collision']:
                num_collision += 1
                break
            elif reward == 1:
                num_success += 1
                t_success_list += t_curr_trial
                if save_success_trial_id:
                    success_episodes.append(i_env)
                break
            elif done:
                break

            if flag != 0:
                if was_stuck:
                    num_step -= 1
                    break
                else:
                    was_stuck = True
                    if flag > 0 or flag == -5:
                        num_no_solution += 1
            else:
                was_stuck = False
        if detail:
            trial_details[i_env] = {'success': reward == 1, 'length': i_step+1, 'collision': info['collision_info']['in_collision']}
            planning_details[i_env].update(
                trial_details[i_env]
            )
        if video:
            video_recorder.close()
        

    planner_stats_summary = {}
    for key in planner_stats:
        planner_stats_summary[key] = {
            'mean': np.mean(planner_stats[key]),
            'std': np.std(planner_stats[key]),
            'max': float(np.max(planner_stats[key]))
        }
    stats = {
        'planner': planner_name,
        'n_trials': n_envs,
        'n_links': n_links,
        'n_obs':n_obs,
        'time_limit': time_limit,
        't_final_thereshold': t_final_thereshold,
        'num_success': num_success,
        'num_collision': num_collision,
        'mean planning time': np.mean(np.array(t_armtd_list)),
        'std planning time': np.std(np.array(t_armtd_list)),
        'mean planning time for success trials': np.mean(np.array(t_success_list)),
        'std planning time for success trials': np.std(np.array(t_success_list)),
        'total planning time': t_armtd,
        'num_no_solution': num_no_solution,
        'num_step': num_step,
        'planner_stats': planner_stats_summary,
    } 
    if detail:
        stats.update({'trial_details': trial_details})
    stats.update({'env_args': env_args})
        
    with open(f"planning_results/3d{n_links}links{n_obs}obs/{planner_name}_stats_3d{n_links}links{n_envs}trials{n_obs}obs{n_steps}steps_{time_limit}limit.json", 'w') as f:
        if save_success_trial_id:
            stats['success_episodes'] = success_episodes
        json.dump(stats, f, indent=2)
    if detail:
        with open(f"planning_results/3d{n_links}links{n_obs}obs/{planner_name}_stats_3d{n_links}links{n_envs}trials{n_obs}obs{n_steps}steps_{time_limit}limit.pkl", 'wb') as f:
            pickle.dump(planning_details, f)
        
    return stats
    

def read_params():
    parser = argparse.ArgumentParser(description="Arm Planning")
    # general setting
    parser.add_argument("--planner", type=str, default="sphere") # "armtd", "sphere"
    parser.add_argument('--robot_type', type=str, default="branched")
    parser.add_argument('--n_robots', type=int, default=1)
    parser.add_argument('--n_links', type=int, default=7)
    parser.add_argument('--n_dims', type=int, default=3)
    parser.add_argument('--n_obs', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    # visualization settings
    parser.add_argument('--video',  action='store_true')
    parser.add_argument('--reachset',  action='store_true')
        
    # optimization info
    parser.add_argument('--num_spheres', type=int, default=5)
    parser.add_argument('--time_limit',  type=float, default=1e20)
    parser.add_argument('--t_final_thereshold', type=float, default=0.2)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--check_self_collision', action='store_true')
    parser.add_argument('--solver', type=str, default="ma27")

    # results info
    parser.add_argument('--save_success', action='store_true') # whether to save success trial id
    parser.add_argument('--detail', action='store_true') # whether to save trajetcory detail
    
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    
    params = read_params()
    planner_name = params.planner
    device = torch.device(params.device)
    n_links = params.n_links * params.n_robots

    print(f"Running {params.n_envs}trials of {planner_name} 3D{n_links}Links{params.n_obs}obs with {params.n_steps} step limit and {params.time_limit}s time limit each step")
    print(f"Using device {device}")
    
    planning_result_dir = f'planning_results/3d{n_links}links{params.n_obs}obs'
    if not os.path.exists(planning_result_dir):
        os.makedirs(planning_result_dir)
    
    stats = {}
    import zonopyrobots as robots2
    robots2.DEBUG_VIZ = False
    basedirname = os.path.dirname(robots2.__file__)
    name_mapping = {2: "two", 3: "three"}
    if params.n_robots in name_mapping:
        robot_path = f'robots/assets/robots/kinova_arm/kinova_{name_mapping[params.n_robots]}_arm_{params.robot_type}.urdf'
    else:
        robot_path = 'robots/assets/robots/kinova_arm/gen3.urdf'
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, robot_path), device=device, create_joint_occupancy=True)

    if planner_name == 'armtd':
        planner = ARMTD_3D_planner(
            rob, 
            device=device,
            filter_obstacles=params.filter,
            linear_solver=params.solver,
        )
        filter_identifier = 'F' if params.filter else ''
        self_collision_identifier = '_SC' if params.check_self_collision else ''
        stats['armtd'] = evaluate_planner(
            planner=planner,
            planner_name=f'armtd{filter_identifier}{self_collision_identifier}_{params.n_robots}{params.robot_type}_t{params.time_limit}', 
            n_envs=params.n_envs, 
            n_steps=params.n_steps, 
            n_links=n_links, 
            n_obs=params.n_obs,
            save_success_trial_id=params.save_success, 
            video=params.video,
            reachset_viz=params.reachset,
            time_limit=params.time_limit,
            detail=params.detail,
            t_final_thereshold=params.t_final_thereshold,
            check_self_collision=params.check_self_collision,
        )
        
    if planner_name == 'sphere':
        joint_radius_override_base = {
                'joint_1': torch.tensor(0.0503305, dtype=torch.float, device=device),
                'joint_2': torch.tensor(0.0630855, dtype=torch.float, device=device),
                'joint_3': torch.tensor(0.0463565, dtype=torch.float, device=device),
                'joint_4': torch.tensor(0.0634475, dtype=torch.float, device=device),
                'joint_5': torch.tensor(0.0352165, dtype=torch.float, device=device),
                'joint_6': torch.tensor(0.0542545, dtype=torch.float, device=device),
                'joint_7': torch.tensor(0.0364255, dtype=torch.float, device=device),
                'end_effector': torch.tensor(0.0394685, dtype=torch.float, device=device),
            }
        if params.n_robots > 1:
            joint_radius_override = {}
            for i_robot in range(params.n_robots):
                for k, v in joint_radius_override_base.items():
                    joint_radius_override[k + f'_rob{i_robot+1}'] = v
        else:
            joint_radius_override = joint_radius_override_base
        planner = SPARROWS_3D_planner(
            rob, 
            device=device, 
            sphere_device=device, 
            spheres_per_link=params.num_spheres,
            joint_radius_override=joint_radius_override,
            filter_links=params.filter,
            check_self_collisions=params.check_self_collision,
            linear_solver=params.solver,
        )
        filter_identifier = 'F' if params.filter else ''
        self_collision_identifier = '_SC' if params.check_self_collision else ''
        stats['spheres_armtd'] = evaluate_planner(
            planner=planner,
            planner_name=f'sphere{filter_identifier}{self_collision_identifier}_{params.n_robots}{params.robot_type}_t{params.time_limit}', 
            n_envs=params.n_envs, 
            n_steps=params.n_steps, 
            n_links=n_links, 
            n_obs=params.n_obs,
            save_success_trial_id=params.save_success, 
            video=params.video,
            reachset_viz=params.reachset,
            time_limit=params.time_limit,
            detail=params.detail,
            t_final_thereshold=params.t_final_thereshold,
            check_self_collision=params.check_self_collision,
        )