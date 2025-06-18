import os
import sys
import logging
import numpy as np
from collections import defaultdict
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import necessary classes from your original module
# Adjust these paths as needed for your environment
# Add the parent directory to Python path to find the 'algorithms' module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append('/workspaces/job-shop-simulator')


from per_jsp.python.per_jsp.environment.job_shop_environment import (
    Job, Operation, JobShopEnvironment, MachineSpec, 
    get_state, run_episode_qlearning_shaped, GanttChartGenerator
)
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.python.per_jsp.environment.jsp_env_gym import JobShopGym

class TaillardJobShopLoader:
    """Loads standard Taillard job shop instances and adapts them for the tool-aware environment."""
    
    @staticmethod
    def load_file(file_path: str) -> str:
        """Load and read the Taillard format file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Failed to open file: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_from_string(data_string: str) -> Tuple[List[Job], int]:
        """Parse Taillard data from a string and create Job objects."""
        # Split the data into lines and remove empty lines
        lines = [line.strip() for line in data_string.split('\n') if line.strip()]

        # Parse number of jobs and machines from first line
        parts = lines[0].split()
        if len(parts) >= 2:
            num_jobs, num_machines = map(int, parts[:2])
        else:
            raise ValueError("Invalid header format in Taillard data")
            
        logger.info(f"Parsing Taillard data: {num_jobs} jobs, {num_machines} machines")

        # Parse processing times
        processing_times = []
        for i in range(1, num_jobs + 1):
            if i < len(lines):
                times = list(map(int, lines[i].split()))
                if len(times) != num_machines:
                    logger.warning(f"Expected {num_machines} times for job {i-1}, got {len(times)}")
                processing_times.append(times)

        # Parse machine assignments
        machine_assignments = []
        for i in range(num_jobs + 1, 2 * num_jobs + 1):
            if i < len(lines):
                # Convert machine numbers from 1-indexed to 0-indexed
                machines = list(map(lambda x: int(x) - 1, lines[i].split()))
                if len(machines) != num_machines:
                    logger.warning(f"Expected {num_machines} machines for job {i-num_jobs-1}, got {len(machines)}")
                machine_assignments.append(machines)

        # Create Job objects
        jobs = []
        for job_id in range(num_jobs):
            operations = []
            for op_idx in range(num_machines):
                if op_idx < len(processing_times[job_id]) and op_idx < len(machine_assignments[job_id]):
                    duration = processing_times[job_id][op_idx]
                    machine = machine_assignments[job_id][op_idx]
                    
                    # Create operation with empty tool requirements
                    operation = Operation(
                        duration=duration,
                        machine=machine,
                        required_tools=set(),  # No tools required for standard instances
                        eligible_machines={machine}  # Only assigned machine is eligible
                    )
                    
                    # Add dependency on previous operation in the same job
                    if op_idx > 0:
                        operation.dependent_operations = [(job_id, op_idx - 1)]
                        
                    operations.append(operation)
            
            jobs.append(Job(operations=operations))

        # Calculate makespan estimate (sum of all processing times / number of machines)
        total_processing_time = sum(op.duration for job in jobs for op in job.operations)
        makespan_estimate = total_processing_time // num_machines
        
        return jobs, makespan_estimate
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Tuple[List[Job], int]:
        """Load a Taillard instance from file."""
        data = cls.load_file(file_path)
        return cls.load_from_string(data)

def get_state(env: JobShopEnvironment):
    """Convert environment state to tuple for Q-learning."""
    return tuple(env.current_state.next_operation_for_job.tolist())

def run_q_learning_algorithm(env: JobShopEnvironment, num_episodes: int = 1000, 
                           alpha: float = 0.1, epsilon: float = 0.2, gamma: float = 0.9):
    """
    Run Q-learning algorithm on the job shop environment (simplified for standard instances).
    
    Args:
        env: The JobShopEnvironment instance
        num_episodes: Number of training episodes
        alpha: Learning rate
        epsilon: Exploration rate
        gamma: Discount factor
        
    Returns:
        Tuple of (Q-values, best schedule entries, best makespan)
    """
    Q = defaultdict(dict)
    best_makespan = float('inf')
    best_schedule = None
    makespan_history = []
    
    print(f"Training Q-learning for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        env.reset()
        total_reward = 0.0
        last_state = None
        last_action = None
        
        # Current epsilon with decay
        current_epsilon = max(0.01, epsilon * (1 - episode/num_episodes))
        
        # Run one episode
        while not env.is_done():
            s = get_state(env)
            possible_actions = env.get_possible_actions()
            
            if not possible_actions:
                break
                
            possible_jobs = set(a.job for a in possible_actions)
            
            # Epsilon-greedy job selection
            if random.random() < current_epsilon:
                chosen_job = random.choice(list(possible_jobs))
            else:
                best_q = -math.inf
                chosen_job = None
                for job_id in possible_jobs:
                    val = Q[s].get(job_id, 0.0)
                    if val > best_q:
                        best_q = val
                        chosen_job = job_id
            
            # Select action for chosen job
            valid_for_job = [a for a in possible_actions if a.job == chosen_job]
            if not valid_for_job:
                break
                
            chosen_action = valid_for_job[0]
            
            try:
                env.step(chosen_action)
            except Exception as e:
                print(f"Error during step: {e}")
                break
            
            # Simple reward function - no tool change costs
            r_step = 10.0
            
            # Update Q-values
            if last_state is not None and last_action is not None:
                old_q = Q[last_state].get(last_action, 0.0)
                s_next = get_state(env)
                max_next_q = max(Q[s_next].values()) if s_next in Q and Q[s_next] else 0.0
                new_q = old_q + alpha * (r_step + gamma * max_next_q - old_q)
                Q[last_state][last_action] = new_q
            
            last_state = s
            last_action = chosen_job
            total_reward += r_step
        
        # Final makespan and reward
        makespan = env.total_time
        makespan_history.append(makespan)
        
        # Track best solution
        if makespan < best_makespan and makespan > 0:
            best_makespan = makespan
            best_schedule = env.schedule_entries[:]
        
        # Progress reporting
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{num_episodes}, Makespan: {makespan}, Best: {best_makespan}")
    
    return Q, best_schedule, best_makespan

def run_priority_rules(env: JobShopEnvironment) -> Dict[str, int]:
    """
    Run different priority rules and return their makespans.
    
    Args:
        env: The JobShopEnvironment instance
        
    Returns:
        Dictionary mapping rule names to their makespans
    """
    rules = {
        'SPT': lambda a: env.jobs[a.job].operations[a.operation].duration,  # Shortest Processing Time
        'LPT': lambda a: -env.jobs[a.job].operations[a.operation].duration,  # Longest Processing Time
        'FIFO': lambda a: a.job,  # First Job First
    }
    
    results = {}
    
    for rule_name, key_func in rules.items():
        env.reset()
        
        while not env.is_done():
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break
                
            # Choose action based on priority rule
            chosen_action = min(possible_actions, key=key_func)
            
            try:
                env.step(chosen_action)
            except Exception as e:
                print(f"Error in {rule_name} rule: {e}")
                break
        
        results[rule_name] = env.total_time
        print(f"{rule_name} rule makespan: {env.total_time}")
    
    return results

def create_empty_machine_specs(num_machines: int) -> List[MachineSpec]:
    """
    Create machine specifications with no tool constraints.
    This prevents the IndexError in calculate_tool_change_time.
    
    Args:
        num_machines: Number of machines in the problem
        
    Returns:
        List of MachineSpec objects with no tool constraints
    """
    return [MachineSpec(max_slots=-1) for _ in range(num_machines)]

def main():
    """Main function to demonstrate the integrated solution."""
    # Example standard format instance (your 5x5 example)
    example_data = """5 5
23 54 12 65 34
45 76 23 87 55
12 43 67 33 76
56 32 49 64 21
78 61 53 71 47
1 3 2 4 5
2 4 1 5 3
3 5 4 1 2
4 1 3 2 5
5 2 1 3 4"""
    
    # Load jobs using the Taillard loader
    print("\nLoading standard job shop instance...")
    #jobs, makespan_estimate = TaillardJobShopLoader.load_from_string(example_data)
    jobs, makespan_estimate = TaillardJobShopGenerator.load_problem("/workspaces/job-shop-simulator/per_jsp/data/taillard_instances/ta01.txt")
    
    print(f"Created {len(jobs)} jobs")
    print(f"Estimated lower bound on makespan: {makespan_estimate}")
    
    # Get number of machines from first job
    num_machines = max(op.machine for job in jobs for op in job.operations) + 1
    
    # Create empty machine specs to avoid IndexError
    machine_specs = create_empty_machine_specs(num_machines)
    # print('Machine specs created with no tool constraints.')
    # print(f"Number of machines: {num_machines}")
    # print(f"Machine specs: {machine_specs}")
    # print(f"Jobs: {jobs}")
    
    # Create environment with empty machine specifications
    print(f"\nCreating job shop environment with {num_machines} machines...")
    env = JobShopEnvironment(jobs, machine_specs)
    
    # Try different priority rules
    print("\nRunning priority rules...")
    priority_results = run_priority_rules(env)
    
    # Run Q-learning
    print("\nRunning Q-learning algorithm...")
    Q, best_schedule, best_makespan = run_q_learning_algorithm(
        env, 
        num_episodes=3000,  # Reduced for demonstration
        alpha=0.1,
        epsilon=0.3,
        gamma=0.9
    )
    
    # Restore best solution
    if best_schedule:
        env.schedule_entries = best_schedule
        env.total_time = best_makespan
    
    # Compare results
    print("\nResults Comparison:")
    print(f"Estimated lower bound: {makespan_estimate}")
    for rule, makespan in priority_results.items():
        print(f"{rule} rule: {makespan}")
    print(f"Q-learning: {best_makespan}")
    
    # Print final schedule
    print("\nFinal schedule:")
    env.print_schedule(show_critical_path=True)
    
    # Generate Gantt chart
    try:
        env.generate_html_gantt("standard_jobshop_solution.html")
        print("Gantt chart saved to standard_jobshop_solution.html")
    except Exception as e:
        print(f"Error generating Gantt chart: {e}")
    
    # Print performance metrics
    try:
        metrics = env.get_performance_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    print(f"  {key}: {np.mean(value):.2f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error generating performance metrics: {e}")

def train_with_stable_baselines_PPO():
    """Train a JobShop environment using PPO from stable_baselines3 without tensorboard."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        print("\nLoading standard job shop instance...")
        #jobs, makespan_estimate = TaillardJobShopLoader.load_from_string(example_data)
        jobs, makespan_estimate = TaillardJobShopGenerator.load_problem("/workspaces/job-shop-simulator/per_jsp/data/taillard_instances/ta01.txt")
        
        print(f"Created {len(jobs)} jobs")
        print(f"Estimated lower bound on makespan: {makespan_estimate}")
        
        # Get number of machines from first job
        num_machines = max(op.machine for job in jobs for op in job.operations) + 1
        
        # Create empty machine specs to avoid IndexError
        machine_specs = create_empty_machine_specs(num_machines)
        
        # Create gym environment
        env = JobShopGym(jobs, machine_specs)
        
        # Check that environment follows Gym API
        check_env(env, warn=True)
        
        print("Environment check passed. Creating PPO model...")
        
        # Create the agent with tensorboard logging DISABLED
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=None,  # Set to None to disable tensorboard
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Train the agent
        print("Beginning training...")
        model.learn(total_timesteps=10000)
        print("Training completed")
        
        # Save the agent
        model.save("jobshop_ppo")
        print("Model saved to jobshop_ppo")
        
        # Test the trained agent
        print("Testing trained agent...")
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Trained agent - Total reward: {total_reward:.2f}")
        print(f"Trained agent - Final makespan: {info['makespan']}")
        
        # Generate Gantt chart for the final schedule
        env.env.generate_html_gantt("trained_ppo_schedule.html")
        print("Gantt chart saved to trained_ppo_schedule.html")
        
    except Exception as e:
        print(f"Error during PPO training: {e}")
        import traceback
        traceback.print_exc()

#train_with_stable_baselines_PPO()

# Run the example if executed directly
if __name__ == "__main__":
    main()
    train_with_stable_baselines_PPO()