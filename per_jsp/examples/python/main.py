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

from per_jsp.environment.job_shop_environment import (
    JobShopEnvironment,
    Action,
    ScheduleEntry,
    Job,
)
from per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator
from per_jsp.environment.job_shop_automatic_generator import AutomaticJobShopGenerator
from per_jsp.environment.structs import GenerationParams

# Import algorithms
from per_jsp.algorithms.base import BaseScheduler
from per_jsp.algorithms.greedy import GreedyScheduler
from per_jsp.algorithms.q_learning import QLearningScheduler

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
    Save the complete solution data to a JSON file.

    Args:
        filename: Output file path
        actions: List of scheduling actions taken
        makespan: Total completion time
        schedule_data: Detailed schedule for each machine
        algorithm: Name of the algorithm used
        solve_time: Time taken to solve
        metrics: Performance metrics
        gantt_data: Gantt chart visualization data
        params: Algorithm parameters used
    """
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

    with open(filename, 'w') as f:
        json.dump(solution, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Solution saved to {filename}")

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

def save_gantt_visualization(gantt_data: Dict, filename: str) -> None:
    """
    Save Gantt chart data for visualization.

    Args:
        gantt_data: Gantt chart data dictionary
        filename: Output file path
    """
    with open(filename, 'w') as f:
        json.dump(gantt_data, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Gantt visualization data saved to {filename}")

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
        return QLearningScheduler(
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon=args.epsilon,
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
    parser.add_argument("--algorithm", type=str, choices=["greedy", "q-learning"],
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
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of episodes for Q-learning")

    # Automatic generation parameters
    parser.add_argument("--num-jobs", type=int, default=10)
    parser.add_argument("--num-machines", type=int, default=5)
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
            visualization_file = f"{args.output_prefix}_gantt.html"
            env.generate_html_gantt(visualization_file)
            logger.info(f"Gantt chart visualization saved to {visualization_file}")

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