#!/usr/bin/env python3
import torch
from environments.urdf_obstacle import KinematicUrdfWithObstacles
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import pickle

PREFIX = '' #'scenario_'

def interp_waypoints(start: np.ndarray, goal: np.ndarray, steps: int = 50):
    """Interpolate between two waypoints."""
    ret = np.array(
        [
            (i / (steps - 1)) * goal + (1 - (i / (steps - 1))) * start
            for i in range(steps)
        ]
    )
    vel = np.repeat(((goal - start) / 0.5)[np.newaxis, :], steps, axis=0)
    ret = np.stack([ret, vel], axis=0)
    return ret

def replay_trajectory(trial_data, indice, rob, n_obs):
    n_obs = len(trial_data[indice]["initial"]["obstacle_pos"])
    blender_folder = f'{PREFIX}planning_blender_files/mpot/{n_obs}obs'
    if not os.path.exists(blender_folder):
        os.makedirs(blender_folder)
    env = KinematicUrdfWithObstacles(
        robot=rob.urdf,
        step_type="direct",
        check_joint_limits=True,
        check_self_collision=False,
        use_bb_collision=False,
        render_mesh=True,
        goal_use_mesh=True,
        reopen_on_close=False,
        obs_size_min=[0.2, 0.2, 0.2],
        obs_size_max=[0.2, 0.2, 0.2],
        obs_gen_buffer=0.01,
        info_nearest_obstacle_dist=True,
        renderer='blender',
        n_obs=n_obs,
        renderer_kwargs = {'filename': os.path.join(blender_folder, f"mpot_random_scene{indice}.blend")}
    )
    
    trial_initial = trial_data[indice]['initial']
    obs = env.reset(
            qpos=np.array(trial_initial["qpos"]),
            qvel=np.zeros(7),
            qgoal=np.array(trial_initial["qgoal"]),
            obs_pos=np.array(trial_initial["obstacle_pos"]),
            obs_size=np.array(trial_initial["obstacle_size"]),
        )
    waypoints = np.array(trial_data[indice]["trajectory"])
    waypoints = waypoints[0][:,:7]

    for i in range(waypoints.shape[0] - 1):
        traj = interp_waypoints(waypoints[i], waypoints[i + 1])
        obs, rew, done, info = env.step(traj)
        env.render()
        if done:
            print(f"Indice {indice}: reward = {rew}")
            break
    env.close()

        
    return
    

if __name__ == "__main__":
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float
    else:
        device = "cpu"
        dtype = torch.float

    ##### LOAD ROBOT #####
    import os
    print("Loading Robot")
    import zonopyrobots as robots2
    basedirname = os.path.dirname(robots2.__file__)
    robots2.DEBUG_VIZ = False
    rob = robots2.ZonoArmRobot.load(
        os.path.join(basedirname, "robots/assets/robots/kinova_arm/gen3.urdf"),
        device=device,
        dtype=dtype,
        create_joint_occupancy=True,
    )
    
    random_scenarios = True
    if random_scenarios:
        n_obs = 20
        with open(f"{PREFIX}planning_results/3d7links{n_obs}obs/mpot_stats_3d7links100trials{n_obs}obs150steps_1e+20limit.pkl", "rb") as f:
            trial_data = pickle.load(f)
        # # n_obs, indice
        # # 10   , 10
        # # 20   ,  5
        # # 40   ,  3
        indices = [19 ]
        for id in indices:
            replay_trajectory(trial_data=trial_data, indice=id, rob=rob, n_obs=n_obs)
    else:
        with open(f"{PREFIX}planning_results/mpot_stats_3d7links14trials12obs150steps_1e+20limit.pkl", "rb") as f:
            trial_data = pickle.load(f)
        for indice in range(1,15):
            replay_trajectory(trial_data=trial_data, indice=indice, rob=rob)
            
        
    

