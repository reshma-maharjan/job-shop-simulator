"""
Heuristic scheduling algorithms for job shop with machine constraints and tool requirements
"""
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Set

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what we know exists for sure
from environment.job_shop_env import (
    Job, MachineSpec, Operation, JobShopEnvironment, Action
)

def apply_spt_algorithm(jobs: List[Job], machine_specs: List[MachineSpec] = None):
    """
    Shortest Processing Time algorithm implementation for job shop scheduling
    
    Parameters:
    jobs: List of Job objects
    machine_specs: List of MachineSpec objects for tool-aware scheduling
    
    Returns:
    Dict containing the schedule and makespan
    """
    # Create the environment directly
    env = JobShopEnvironment(jobs, machine_specs)
    
    # Reset the environment
    env.reset()
    
    # Scheduling algorithm logic
    steps = 0
    
    while not env.is_done():
        steps += 1
        # Get possible actions from the environment
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            print(f"No valid actions available at step {steps}")
            break
        
        # For SPT, find the operation with the shortest processing time
        shortest_duration = float('inf')
        selected_action = None
        
        for action in possible_actions:
            job_id = action.job
            op_id = action.operation
            machine_id = action.machine
            duration = jobs[job_id].operations[op_id].duration
            
            # Consider tool change time if machine specs are provided
            if machine_specs:
                op = jobs[job_id].operations[op_id]
                current_tools = env.get_current_tools(machine_id)
                
                # Get tool change time
                tool_change_duration, _, _ = env.calculate_tool_change_time(
                    machine_id, op.required_tools, current_tools
                )
                
                # Add tool change time to processing duration
                duration += tool_change_duration
            
            if duration < shortest_duration:
                shortest_duration = duration
                selected_action = action
        
        if selected_action is not None:
            # Take the step using the selected action
            env.step(selected_action)
        else:
            print(f"Failed to select an action at step {steps}")
            break
    
    print(f"SPT algorithm - Steps: {steps}")
    print(f"Final makespan: {env.total_time}")
    
    # Generate visualization
    env.generate_html_gantt("spt_schedule.html")
    
    return {
        'schedule': env.schedule_entries,
        'makespan': env.total_time,
        'metrics': env.get_performance_metrics()
    }

def apply_mwr_algorithm(jobs: List[Job], machine_specs: List[MachineSpec] = None):
    """
    Most Work Remaining algorithm implementation for job shop scheduling
    
    Parameters:
    jobs: List of Job objects
    machine_specs: List of MachineSpec objects for tool-aware scheduling
    
    Returns:
    Dict containing the schedule and makespan
    """
    # Calculate initial total work for each job
    initial_work = {}
    for job_id, job in enumerate(jobs):
        total_time = sum(op.duration for op in job.operations)
        initial_work[job_id] = total_time
    
    # Create the environment directly
    env = JobShopEnvironment(jobs, machine_specs)
    
    # Reset the environment
    env.reset()
    
    # Scheduling algorithm logic
    steps = 0
    
    while not env.is_done():
        steps += 1
        # Get possible actions from the environment
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            print(f"No valid actions available at step {steps}")
            break
        
        # For MWR, calculate remaining work for eligible jobs
        job_remaining_work = {}
        for action in possible_actions:
            job_id = action.job
            
            # Get the next operation index for this job
            next_op_idx = int(env.current_state.next_operation_for_job[job_id])
            
            # Calculate remaining work for this job
            remaining_work = sum(
                jobs[job_id].operations[i].duration 
                for i in range(next_op_idx, len(jobs[job_id].operations))
            )
            
            # Store the remaining work for this job
            if job_id not in job_remaining_work or remaining_work > job_remaining_work[job_id]:
                job_remaining_work[job_id] = remaining_work
        
        # Select job with most work remaining
        if job_remaining_work:
            selected_job = max(job_remaining_work.items(), key=lambda x: x[1])[0]
            
            # Find an action for this job
            actions_for_job = [a for a in possible_actions if a.job == selected_job]
            if actions_for_job:
                selected_action = actions_for_job[0]
                
                # Take the step using the selected action
                env.step(selected_action)
            else:
                print(f"No valid actions for selected job {selected_job}")
                break
        else:
            print(f"Failed to calculate remaining work at step {steps}")
            break
    
    print(f"MWR algorithm - Steps: {steps}")
    print(f"Final makespan: {env.total_time}")
    
    # Generate visualization
    env.generate_html_gantt("mwr_schedule.html")
    
    return {
        'schedule': env.schedule_entries,
        'makespan': env.total_time,
        'metrics': env.get_performance_metrics()
    }

def apply_min_tool_change_algorithm(jobs: List[Job], machine_specs: List[MachineSpec]):
    """
    Minimum Tool Change algorithm implementation for job shop scheduling
    
    Parameters:
    jobs: List of Job objects
    machine_specs: List of MachineSpec objects (required for this algorithm)
    
    Returns:
    Dict containing the schedule and makespan
    """
    if not machine_specs:
        raise ValueError("Machine specifications with tool information are required for this algorithm")
    
    # Create the environment directly
    env = JobShopEnvironment(jobs, machine_specs)
    
    # Reset the environment
    env.reset()
    
    # Scheduling algorithm logic
    steps = 0
    
    while not env.is_done():
        steps += 1
        # Get possible actions from the environment
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            print(f"No valid actions available at step {steps}")
            break
        
        # For Min Tool Change, find the operation requiring minimal tool changes
        min_tool_changes = float('inf')
        min_tool_change_duration = float('inf')
        selected_action = None
        
        for action in possible_actions:
            job_id = action.job
            op_id = action.operation
            machine_id = action.machine
            op = jobs[job_id].operations[op_id]
            
            # Get current tools on the machine
            current_tools = env.get_current_tools(machine_id)
            
            # Calculate tool changes required
            tool_change_duration, tools_to_add, tools_to_remove = env.calculate_tool_change_time(
                machine_id, op.required_tools, current_tools
            )
            
            # Count total tool changes
            total_changes = len(tools_to_add) + len(tools_to_remove)
            
            # Prefer operations with fewer tool changes
            # If equal, prefer shorter tool change duration
            if total_changes < min_tool_changes or (
                total_changes == min_tool_changes and tool_change_duration < min_tool_change_duration
            ):
                min_tool_changes = total_changes
                min_tool_change_duration = tool_change_duration
                selected_action = action
        
        if selected_action is not None:
            # Take the step using the selected action
            env.step(selected_action)
        else:
            print(f"Failed to select an action at step {steps}")
            break
    
    print(f"Min Tool Change algorithm - Steps: {steps}")
    print(f"Final makespan: {env.total_time}")
    
    # Generate visualization
    env.generate_html_gantt("min_tool_change_schedule.html")
    
    return {
        'schedule': env.schedule_entries,
        'makespan': env.total_time,
        'metrics': env.get_performance_metrics()
    }

def compare_algorithms(jobs, machine_specs):
    """
    Compare different scheduling algorithms
    
    Parameters:
    jobs: List of Job objects
    machine_specs: List of MachineSpec objects
    
    Returns:
    Dict containing results for each algorithm
    """
    results = {}
    
    print("Running SPT algorithm...")
    results['SPT'] = apply_spt_algorithm(jobs, machine_specs)
    
    print("\nRunning MWR algorithm...")
    results['MWR'] = apply_mwr_algorithm(jobs, machine_specs)
    
    if machine_specs:
        print("\nRunning Min Tool Change algorithm...")
        results['MinToolChange'] = apply_min_tool_change_algorithm(jobs, machine_specs)
    
    # Find the best algorithm based on makespan
    best_alg = min(results.items(), key=lambda x: x[1]['makespan'])
    
    print("\nAlgorithm Comparison Results:")
    print("=============================")
    for alg, result in results.items():
        print(f"{alg}: Makespan = {result['makespan']}")
        
        # Print additional metrics if available
        if 'metrics' in result:
            util = result['metrics'].get('avg_machine_utilization', 0) * 100
            tool_changes = result['metrics'].get('total_tool_changes', 0)
            tool_change_time = result['metrics'].get('total_tool_change_time', 0)
            
            print(f"  - Avg Machine Utilization: {util:.1f}%")
            print(f"  - Total Tool Changes: {tool_changes}")
            print(f"  - Total Tool Change Time: {tool_change_time}")
    
    print(f"\nBest algorithm: {best_alg[0]} with makespan {best_alg[1]['makespan']}")
    
    return results

# Example of how to run this code
if __name__ == "__main__":
    # Create example jobs and machine specs
    machine_specs = [
        MachineSpec(
            max_slots=4,
            compatible_tools={1, 2, 3, 4, 5, 6},
            tool_matrix=np.array([
                [0, 8, 10, 12, 15, 18, 20],
                [6, 0, 5, 8, 10, 12, 15],
                [8, 5, 0, 6, 8, 10, 12],
                [10, 8, 6, 0, 5, 8, 10],
                [12, 10, 8, 5, 0, 6, 8],
                [15, 12, 10, 8, 6, 0, 5],
                [18, 15, 12, 10, 8, 5, 0]
            ])
        ),
        MachineSpec(
            max_slots=2,
            compatible_tools={2, 3, 4},
            tool_matrix=np.array([
                [0, 0, 15, 18, 20, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [12, 0, 0, 10, 12, 0, 0],
                [15, 0, 10, 0, 8, 0, 0],
                [18, 0, 12, 8, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        ),
        MachineSpec(
            max_slots=3,
            compatible_tools={1, 3, 4, 5},
            tool_matrix=np.array([
                [0, 10, 0, 12, 15, 18, 0],
                [8, 0, 0, 10, 12, 15, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [10, 8, 0, 0, 8, 10, 0],
                [12, 10, 0, 8, 0, 8, 0],
                [15, 12, 0, 10, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        )
    ]

    jobs = [
        Job(operations=[
            Operation(duration=4, machine=0, required_tools={1, 2}),
            Operation(duration=3, machine=1, required_tools={2, 3}),
            Operation(duration=5, machine=2, required_tools={3, 4})
        ]),
        Job(operations=[
            Operation(duration=3, machine=1, required_tools={3, 4}),
            Operation(duration=6, machine=0, required_tools={1, 6}),
            Operation(duration=4, machine=2, required_tools={1, 3})
        ]),
        Job(operations=[
            Operation(duration=2, machine=2, required_tools={4, 5}),
            Operation(duration=3, machine=0, required_tools={2, 6})
        ]),
        Job(operations=[
            Operation(duration=5, machine=0, required_tools={1, 2}),
            Operation(duration=4, machine=1, required_tools={2, 4}),
            Operation(duration=3, machine=2, required_tools={3, 5}),
            Operation(duration=4, machine=0, required_tools={1, 6})
        ])
    ]
    
    # Add operation dependencies
    jobs[1].operations[1].dependent_operations = [(1, 0)]
    jobs[2].operations[1].dependent_operations = [(0, 1)]
    jobs[3].operations[2].dependent_operations = [(1, 1)]

    # Add job dependencies
    jobs[2].dependent_jobs = [0]
    jobs[3].dependent_jobs = [1]
    
    # Run the comparison
    results = compare_algorithms(jobs, machine_specs)
    
    print("Algorithm execution complete.")