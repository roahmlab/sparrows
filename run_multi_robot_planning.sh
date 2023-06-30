time_limits=(0.5 1.0)
n_obss=(5 10 15)
n_envss=100
n_robotss=(2 3) 
robot_types=("branched") 
solver="ma27"

for n_robots in "${n_robotss[@]}";do
    for time_limit in "${time_limits[@]}";do
        for n_obs in "${n_obss[@]}";do
            for robot_type in "${robot_types[@]}";do
                python run_statistics_planning_3d.py --planner armtd --n_obs $n_obs --time_limit $time_limit --n_envs $n_envss --detail --t_final_thereshold 0.2 --n_robot $n_robots --robot_type $robot_type --solver $solver
                python run_statistics_planning_3d.py --planner sphere --n_obs $n_obs --time_limit $time_limit --n_envs $n_envss --detail --t_final_thereshold 0.2 --n_robot $n_robots --robot_type $robot_type --solver $solver
            done
        done
    done
done
