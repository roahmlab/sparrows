time_limits=(0.15 0.25 0.5) 
n_obss=(10 20 40) 
n_envss=100
solver="ma27"

for time_limit in "${time_limits[@]}";do
    for n_obs in "${n_obss[@]}";do
        python run_statistics_planning_3d.py --planner sphere --save_success --n_obs $n_obs --time_limit $time_limit --n_envs $n_envss --t_final_thereshold 0.2 --detail --solver $solver
        python run_statistics_planning_3d.py --planner armtd --save_success --n_obs $n_obs --time_limit $time_limit --n_envs $n_envss  --t_final_thereshold 0.2 --detail --solver $solver
    done
done