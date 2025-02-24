#!/usr/bin/env python3
"""
Main entry point for the Job Shop Scheduling System.
Handles problem generation, solving, and result visualization.
"""

import logging
import argparse
import json
import time
import random
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
import os
from pathlib import Path

from per_jsp.python.per_jsp.environment.job_shop_environment import (
    JobShopEnvironment,
    Action,
    ScheduleEntry,
    Job,
)
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.python.per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator
from per_jsp.python.per_jsp.environment.job_shop_automatic_generator import AutomaticJobShopGenerator
from per_jsp.python.per_jsp.environment.structs import GenerationParams

# Import algorithms
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.algorithms.greedy import GreedyScheduler
from per_jsp.python.per_jsp.algorithms.q_learning import QLearningScheduler
from per_jsp.python.per_jsp.algorithms.ppo_2 import PPOScheduler
from per_jsp.python.per_jsp.algorithms.dqn_1 import DQNScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types in nested structures to Python native types.

    Args:
        obj: The object to convert

    Returns:
        The converted object with numpy types replaced by native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def ensure_output_directory(output_dir: str = "output") -> Path:
    """
    Ensure the output directory exists, create it if it doesn't.
    
    Args:
        output_dir: Directory path for outputs
        
    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def get_output_path(filename: str, output_dir: Optional[str] = None) -> Path:
    """
    Get the full path for an output file in the output directory.
    
    Args:
        filename: Name of the output file
        output_dir: Optional custom output directory
        
    Returns:
        Full path for the output file
    """
    if output_dir is None:
        output_dir = "output"
    
    output_path = ensure_output_directory(output_dir)
    return output_path / filename

def save_solution(
        filename: str,
        actions: List[Action],
        makespan: int,
        schedule_data: List[List[ScheduleEntry]],
        algorithm: str,
        solve_time: float,
        metrics: Dict,
        gantt_data: Optional[Dict],
        params: Optional[dict] = None) -> None:
    """
    Save the complete solution data to a JSON file in the output directory.
    
    Args:
        filename: Output file name
        actions: List of scheduling actions taken
        makespan: Total completion time
        schedule_data: Detailed schedule for each machine
        algorithm: Name of the algorithm used
        solve_time: Time taken to solve
        metrics: Performance metrics
        gantt_data: Gantt chart visualization data
        params: Algorithm parameters used
        output_dir: Directory to save the output
    """

    output_dir=Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    solution = {
        "metadata": {
            "algorithm": algorithm,
            "solve_time": solve_time,
            "parameters": params or {}
        },
        "results": {
            "makespan": makespan,
            "num_actions": len(actions),
            "actions": [{"job": int(a.job), "machine": int(a.machine), "operation": int(a.operation)}
                        for a in actions],
            "schedule": [
                [{"job": int(e.job),
                  "operation": int(e.operation),
                  "start": int(e.start),
                  "duration": int(e.duration)} for e in machine_schedule]
                for machine_schedule in schedule_data
            ],
            "metrics": metrics
        }
    }

    if gantt_data:
        solution["results"]["gantt_data"] = gantt_data

    # Convert any remaining numpy types
    solution = convert_numpy_types(solution)

    with open(output_path, 'w') as f:
        json.dump(solution, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Solution saved to {output_path}")

def format_metrics(metrics: Dict) -> str:
    """
    Format metrics dictionary for console output.

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        Formatted string of metrics
    """
    output = []
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            if key == "machine_utilization":
                for i, util in enumerate(value):
                    output.append(f"Machine {i} utilization: {util*100:.1f}%")
            else:
                output.append(f"{key}: {value}")
        elif isinstance(value, float):
            output.append(f"{key}: {value:.2f}")
        else:
            output.append(f"{key}: {value}")
    return "\n".join(output)

def save_gantt_visualization(gantt_data: Dict, filename: str, output_dir: str = "output") -> None:
    """
    Save Gantt chart data for visualization in the output directory.
    
    Args:
        gantt_data: Gantt chart data dictionary
        filename: Output file name
        output_dir: Directory to save the output
    """
    output_dir=Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(gantt_data, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Gantt visualization data saved to {output_path}")

def save_html_gantt(env, filename: str) -> None:
    """
    Save HTML Gantt chart to the output directory.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    env.generate_html_gantt(str(output_path))
    logger.info(f"HTML Gantt chart visualization saved to {output_path}")    

def run_automatic_generation(args) -> Tuple[List[Job], int]:
    """
    Run job shop with automatically generated instance.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (generated jobs, estimated makespan)
    """
    params = GenerationParams(
        num_jobs=args.num_jobs,
        num_machines=args.num_machines,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dependency_density=args.dependency_density,
        max_dependencies_per_job=args.max_deps,
        long_job_rate=0.1,
        long_job_factor=2.0,
        output_file=args.output_prefix + "_instance.json" if args.save_instance else None
    )

    jobs, estimated_makespan = AutomaticJobShopGenerator.generate(params)
    logger.info(f"Generated problem with estimated makespan: {estimated_makespan}")

    return jobs, estimated_makespan

def run_taillard_instance(args) -> Tuple[List[Job], int]:
    """
    Run job shop with Taillard benchmark instance.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (loaded jobs, estimated makespan)
    """
    jobs, estimated_makespan = TaillardJobShopGenerator.load_problem(args.taillard_file)
    logger.info(f"Loaded Taillard instance with estimated makespan: {estimated_makespan}")

    return jobs, estimated_makespan

def run_manual_instance(args) -> Tuple[List[Job], int]:
    """
    Run job shop with manually specified instance.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (loaded jobs, estimated makespan)
    """
    jobs, estimated_makespan = ManualJobShopGenerator.generate_from_file(args.input_json)
    logger.info(f"Loaded manual instance with estimated makespan: {estimated_makespan}")

    return jobs, estimated_makespan

def create_scheduler(args) -> BaseScheduler:
    """
    Create a scheduler based on command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Configured scheduler instance

    Raises:
        ValueError: If unknown algorithm specified
    """
    if args.algorithm == "greedy":
        return GreedyScheduler(use_longest=args.use_longest)
    elif args.algorithm == "q-learning":
        return  QLearningScheduler(
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=1.0,
            episodes=1300
        )
    elif args.algorithm == "dqn":
        return DQNScheduler(
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=1.0,
            episodes=500
        )
    elif args.algorithm == "ppo":
        return PPOScheduler(
            learning_rate=args.ppo_lr,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            episodes=args.episodes
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(description="Job Shop Scheduling System")

    # Generation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true", help="Use automatic generation")
    group.add_argument("--taillard", action="store_true", help="Use Taillard instance")
    group.add_argument("--manual", action="store_true", help="Use manual JSON specification")

    # Algorithm selection
    parser.add_argument("--algorithm", type=str, 
                   choices=["greedy", "q-learning", "ppo", "dqn"],
                   default="greedy", help="Algorithm to use")

    # Greedy algorithm parameters
    parser.add_argument("--use-longest", action="store_true",
                        help="Use longest duration first (greedy only)")

    # Q-learning parameters
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate for Q-learning")
    parser.add_argument("--discount-factor", type=float, default=0.95,
                        help="Discount factor for Q-learning")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Exploration rate for Q-learning")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Number of episodes for Q-learning")
    # DQN parameters
    parser.add_argument("--dqn-hidden-size", type=int, default=128,
                    help="Hidden layer size for DQN")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer size for DQN")
    parser.add_argument("--target-update", type=int, default=10,
                        help="Steps between target network updates for DQN")    
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DQN training")
    
    
    # PPO parameters
    parser.add_argument("--ppo-lr", type=float, default=3e-4,
                    help="Learning rate for PPO")
    parser.add_argument("--ppo-epochs", type=int, default=5,
                    help="Number of PPO epochs per update")
    parser.add_argument("--ppo-batch-size", type=int, default=32,
                    help="PPO batch size")

    # Automatic generation parameters
    parser.add_argument("--num-jobs", type=int, default=10)
    parser.add_argument("--num-machines", type=int, default=3)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=100)
    parser.add_argument("--dependency-density", type=float, default=0.3)
    parser.add_argument("--max-deps", type=int, default=3)

    # File inputs/outputs
    parser.add_argument("--taillard-file", type=str, help="Taillard instance file")
    parser.add_argument("--input-json", type=str, help="Input JSON file for manual specification")
    parser.add_argument("--output-prefix", type=str, default="output",
                        help="Prefix for output files")
    parser.add_argument("--save-instance", action="store_true",
                        help="Save generated instance to JSON")

    # Solver parameters
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, help="Random seed")

    # Visualization and output options
    parser.add_argument("--show-critical-path", action="store_true",
                        help="Show critical path in schedule output")
    parser.add_argument("--save-gantt", action="store_true",
                        help="Save Gantt chart data")
    parser.add_argument("--detailed-metrics", action="store_true",
                        help="Show detailed performance metrics")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate HTML Gantt chart visualization")
    return parser


def main() -> int:
    """Main entry point for the job shop scheduling system."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    try:

        # Generate or load jobs based on mode
        if args.auto:
            jobs, estimated_makespan = run_automatic_generation(args)
        elif args.taillard:
            if not args.taillard_file:
                parser.error("--taillard-file required with --taillard mode")
            jobs, estimated_makespan = run_taillard_instance(args)
        else:  # manual
            if not args.input_json:
                parser.error("--input-json required with --manual mode")
            jobs, estimated_makespan = run_manual_instance(args)

        
        # Create environment and scheduler
        env = JobShopEnvironment(jobs)
        scheduler = create_scheduler(args)

        logger.info(f"Created environment with {len(jobs)} jobs and {env.num_machines} machines")
        logger.info(f"Using algorithm: {args.algorithm}")

        # Solve the problem
        start_time = time.time()
        actions, makespan = scheduler.solve(env, args.max_steps)
        solve_time = time.time() - start_time

        # Get performance metrics
        metrics = env.get_performance_metrics()

        # Generate Gantt data if requested
        gantt_data = env.generate_gantt_data() if args.save_gantt else None

        # Display results
        env.print_schedule(show_critical_path=args.show_critical_path)

        if args.detailed_metrics:
            logger.info("\nDetailed Performance Metrics:")
            logger.info(format_metrics(metrics))

        # Save solution with enhanced data
        solution_file = f"{args.output_prefix}_solution.json"
        algorithm_params = {
            "algorithm": args.algorithm,
            "max_steps": args.max_steps
        }

        if args.algorithm == "greedy":
            algorithm_params["use_longest"] = args.use_longest
        elif args.algorithm == "q-learning":
            algorithm_params.update({
                "learning_rate": args.learning_rate,
                "discount_factor": args.discount_factor,
                "epsilon": args.epsilon,
                "episodes": args.episodes
            })
        elif args.algorithm == "dqn":
            algorithm_params.update({
                "learning_rate": args.learning_rate,
                "discount_factor": args.discount_factor,
                "epsilon": args.epsilon,
                "episodes": args.episodes,
                "hidden_size": args.dqn_hidden_size,
                "buffer_size": args.buffer_size,
                "target_update": args.target_update,
                "batch_size": args.batch_size
            })
        elif args.algorithm == "ppo":
            algorithm_params.update({
                "learning_rate": args.ppo_lr,
                "epochs": args.ppo_epochs,
                "batch_size": args.ppo_batch_size,
                "episodes": args.episodes
            })

        #save solution
        solution_file = f"{args.output_prefix}_solution.json"
        save_solution(
            solution_file,
            actions,
            makespan,
            env.get_schedule_data(),
            args.algorithm,
            solve_time,
            metrics,
            gantt_data,
            algorithm_params
        )

        # Save Gantt visualization data if requested
        if args.save_gantt:
            gantt_file = f"{args.output_prefix}_gantt.json"
            save_gantt_visualization(gantt_data, gantt_file)

        if args.visualize:
            html_file = f"{args.output_prefix}_gantt.html"
            save_html_gantt(env, html_file)
         

        # Print summary
        logger.info("\nSolution Summary:")
        logger.info(f"Algorithm: {args.algorithm}")
        logger.info(f"Solve time: {solve_time:.2f} seconds")
        logger.info(f"Makespan: {makespan}")
        logger.info(f"Estimated lower bound: {estimated_makespan}")
        logger.info(f"Number of actions: {len(actions)}")
        logger.info(f"Average machine utilization: {metrics['avg_machine_utilization']*100:.1f}%")
        logger.info(f"Critical path length: {metrics['critical_path_length']}")
        logger.info(f"Solution saved to: {solution_file}")

        if args.save_gantt:
            logger.info(f"Gantt data saved to: {gantt_file}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    exit(main())