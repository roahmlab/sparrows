import json

def print_mean_constraint_time(
    k_range = 24,
    time_limit = 0.25,
    planner = 'armtd', # 'sphere'
):
    toprint_numbers = []
    
    for n_obs in [10, 20, 40]:
        filename = f'very_important_results/no_filter_planning_results/planning_results_pi_{k_range}/3d7links{n_obs}obs/{planner}_1branched_t{time_limit}_stats_3d7links100trials{n_obs}obs150steps_{time_limit}limit.json'
        with open(filename, 'r') as f:
            stats = json.load(f)
        
        if "constraint_times" in stats["planner_stats"]:
            stats = stats["planner_stats"]["constraint_times"]
            toprint_numbers.append(stats["mean"])
            toprint_numbers.append(stats["std"])     
    
    toprint_str = ""
    i = 0
    while i <= len(toprint_numbers) - 1:
        if toprint_numbers[i] is not None:
            toprint_str += f"& {toprint_numbers[i] * 1000:.1f} ± {toprint_numbers[i+1] * 1000:.1f} "
            i += 2
        else:
            toprint_str += f"& - "
            i += 1
    print(toprint_str)


def print_mean_planning_time(
    k_range = 24,
    time_limit = 0.25,
    planner = 'armtd', # 'sphere'
):
    toprint_numbers = []
    keys = ["mean planning time", "std planning time"]
    
    for n_obs in [10, 20, 40]:
        filename = f'very_important_results/no_filter_planning_results/planning_results_pi_{k_range}/3d7links{n_obs}obs/{planner}_1branched_t{time_limit}_stats_3d7links100trials{n_obs}obs150steps_{time_limit}limit.json'
        with open(filename, 'r') as f:
            stats = json.load(f)
        
        for k in keys:
            toprint_numbers.append(stats[k])     
                    
    toprint_str = ""
    i = 0
    while i <= len(toprint_numbers) - 1:
        if toprint_numbers[i] is not None:
            toprint_str += f"& {toprint_numbers[i]:.2f} ± {toprint_numbers[i+1]:.2f} "
            i += 2
        else:
            toprint_str += f"& - "
            i += 1
    print(toprint_str)

def print_num_cons_evals(
    k_range = 24,
    time_limit = 0.25,
    planner = 'armtd', # 'sphere'
):  
    toprint_numbers = []
    keys = ["num_constraint_evaluations", "num_jacobian_evaluations"]
    metrics = ["mean", "std"]
    
    for k in keys:
        for n_obs in [10, 20, 40]:
            filename = f'very_important_results/no_filter_planning_results/planning_results_pi_{k_range}/3d7links{n_obs}obs/{planner}_1branched_t{time_limit}_stats_3d7links100trials{n_obs}obs150steps_{time_limit}limit.json'
            with open(filename, 'r') as f:
                stats = json.load(f)
            stats = stats["planner_stats"]
        
            if k in stats:
                for m in metrics:
                    toprint_numbers.append(stats[k][m])
            else:
                toprint_numbers.append(None)        
                    
    toprint_str = ""
    i = 0
    while i <= len(toprint_numbers) - 1:
        if toprint_numbers[i] is not None:
            toprint_str += f"& {toprint_numbers[i]:.1f} ± {toprint_numbers[i+1]:.1f} "
            i += 2
        else:
            toprint_str += f"& - "
            i += 1
    print(toprint_str)
        
        
        

if __name__ == '__main__':
    do_print_num_cons_evals = False
    if do_print_num_cons_evals:
        for k_range in [24, 6]:
            for planner in ['sphere', 'armtd']:
                print_num_cons_evals(k_range=k_range, time_limit=0.25, planner=planner)

    do_print_mean_planning_time = True
    if do_print_mean_planning_time:
        for k_range in [24, 6]:
            for planner in ['sphere', 'armtd']:
                print_mean_planning_time(k_range=k_range, time_limit=0.15, planner=planner)
                
    print_cons_time = False
    if print_cons_time:
        for k_range in [24, 6]:
            for planner in ['sphere', 'armtd']:
                print_mean_constraint_time(k_range=k_range, time_limit=0.25, planner=planner)