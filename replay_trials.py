import torch
import argparse
import numpy as np
import time
import json
import random
from tqdm import tqdm
from environments.urdf_obstacle import KinematicUrdfWithObstacles
from environments.fullstep_recorder import FullStepRecorder
import os
import pickle

def replay_trials(trial_ids=[], video=True, filename=''):
    num_success = 0
    num_collision = 0
    
    with open(f"{filename}.json", 'r') as f:
        env_args = json.load(f)["env_args"]
    with open(f"{filename}.pkl", 'rb') as f:
        trajectory_data = pickle.load(f)
    
    if video:
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        n_obs = env_args['n_obs']
        video_folder = f'planning_videos/{filename}/3d7links{n_obs}obs'
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    
    env = KinematicUrdfWithObstacles(
            robot=rob.urdf,
            **env_args
        )
    for i_env in tqdm(trial_ids):
        trial_data = trajectory_data[i_env]
        initial = trial_data['initial']
        initial['obs_pos'] = initial['obstacle_pos']
        initial['obs_size'] = initial['obstacle_size']
        obs = env.reset(
            qpos=initial['qpos'],
            qvel=initial['qvel'],
            qgoal=initial['qgoal'],
            obs_pos=initial['obstacle_pos'],
            obs_size=initial['obstacle_size']
        )
        
        trial_k = trial_data['trajectory']['k']
        if video:
            video_path = os.path.join(video_folder, f'video{i_env}.mp4')
            video_recorder = FullStepRecorder(env, path=video_path)
        was_stuck = False
        for i_step in range(len(trial_k)):
            ka = trial_k[i_step]
            obs, reward, done, info = env.step(ka)
            if video:
                video_recorder.capture_frame()
            if info['collision_info']['in_collision']:
                num_collision += 1
            elif reward == 1:
                # print("SUCCESS!")
                num_success += 1
        if video:
            video_recorder.close()
    print(f"num_success: {num_success}, num_collision: {num_collision}")
        
    return 
    

def read_params():
    parser = argparse.ArgumentParser(description="Planning")
    # general env setting
    parser.add_argument("--planner", type=str, default='sphere') # armtd, sphere
    parser.add_argument('--n_links', type=int, default=7)
    parser.add_argument('--n_dims', type=int, default=3)
    parser.add_argument('--n_obs', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=150)
    parser.add_argument('--video',  action='store_true')
    parser.add_argument('--time_limit',  type=float, default=0.15)
    
    # optimization info
    parser.add_argument('--buffer_size', type=float, default=0.03)
    parser.add_argument('--num_spheres', type=int, default=5)
    
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    
    params = read_params()
    planner_name = params.planner
    planning_result_dir = f'planning_results/3d7links{params.n_obs}obs'

    import zonopyrobots as robots2
    robots2.DEBUG_VIZ = False
    basedirname = os.path.dirname(robots2.__file__)
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), create_joint_occupancy=True)

    if planner_name == 'armtd' or planner_name == 'all' or planner_name == 'both':
        filename = f'armtd_t{params.time_limit}'
    elif planner_name == 'sphere' or planner_name == 'all' or planner_name == 'both':
        filename=f'spheres_armtd_s{params.num_spheres}_t{params.time_limit}'
    filename = f'{filename}_stats_3d7links{params.n_envs}trials{params.n_obs}obs{params.n_steps}steps_{params.time_limit}limit'

    replay_trials(
        trial_ids=range(20),
        video=True,
        filename=os.path.join(planning_result_dir, filename))