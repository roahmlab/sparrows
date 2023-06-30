time_limits=(0.5)
solver="ma27"

for time_limit in "${time_limits[@]}";do
    python run_scenario_planning_3d.py --planner sphere --save_success --time_limit $time_limit --t_final_thereshold 0.2 --detail --solver $solver
    python run_scenario_planning_3d.py --planner armtd --save_success --time_limit $time_limit --t_final_thereshold 0.2 --detail --solver $solver
done