"""
Main entry point for the Job Shop Scheduling System.
Handles problem generation, solving, and result visualization for FIFO, MWKR, and SPT schedulers.
"""

import logging
import argparse
import json
import time
import random
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path

from per_jsp.python.per_jsp.environment.job_shop_environment import (
    JobShopEnvironment,
    Action,
    ScheduleEntry,
    Job,
)
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.python.per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator

# Import schedulers
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.algorithms.fifo import FIFOScheduler
from per_jsp.python.per_jsp.algorithms.mwkr import MWKRScheduler
from per_jsp.python.per_jsp.algorithms.spt import SPTScheduler

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
    """Convert numpy types to Python native types."""
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

def save_solution(filename: str, actions: List[Action], makespan: int, 
                 schedule_data: List[List[ScheduleEntry]], algorithm: str, 
                 solve_time: float, metrics: Dict, gantt_data: Optional[Dict] = None, 
                 params: Optional[dict] = None) -> None:
    """Save solution data to JSON file."""
    output_dir = Path("output")
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

    # Convert numpy types
    solution = convert_numpy_types(solution)

    with open(output_path, 'w') as f:
        json.dump(solution, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Solution saved to {output_path}")

def create_scheduler(algorithm: str) -> BaseScheduler:
    """Create scheduler instance based on algorithm name."""
    schedulers = {
        "fifo": FIFOScheduler,
        "mwkr": MWKRScheduler,
        "spt": SPTScheduler
    }
    
    if algorithm not in schedulers:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return schedulers[algorithm]()

def run_scheduler(env: JobShopEnvironment, scheduler: BaseScheduler, 
                 max_steps: int, algorithm: str, show_critical_path: bool = False) -> Dict:
    """Run a scheduler and return results."""
    logger.info(f"Running {algorithm.upper()} scheduler...")
    
    start_time = time.time()
    actions, makespan = scheduler.solve(env, max_steps)
    solve_time = time.time() - start_time
    
    metrics = env.get_performance_metrics()
    
    # Display results
    env.print_schedule(show_critical_path=show_critical_path)
    
    return {
        "actions": actions,
        "makespan": makespan,
        "solve_time": solve_time,
        "metrics": metrics
    }

def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure and return argument parser."""
    parser = argparse.ArgumentParser(description="Job Shop Scheduling System")
    
    # Problem instance source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--taillard", action="store_true", help="Use Taillard instance")
    group.add_argument("--manual", action="store_true", help="Use manual JSON specification")
    
    # Algorithm selection
    parser.add_argument("--algorithms", type=str, nargs="+", 
                       choices=["fifo", "mwkr", "spt"], default=["fifo"],
                       help="Algorithms to run")
    
    # File inputs/outputs
    parser.add_argument("--taillard-file", type=str, help="Taillard instance file")
    parser.add_argument("--input-json", type=str, help="Input JSON file for manual specification")
    parser.add_argument("--output-prefix", type=str, default="output",
                       help="Prefix for output files")
    
    # Solver parameters
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Visualization and output options
    parser.add_argument("--show-critical-path", action="store_true",
                       help="Show critical path in schedule output")
    parser.add_argument("--save-gantt", action="store_true",
                       help="Save Gantt chart data")
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
        # Load jobs based on mode
        if args.taillard:
            if not args.taillard_file:
                parser.error("--taillard-file required with --taillard mode")
            jobs, estimated_makespan = TaillardJobShopGenerator.load_problem(args.taillard_file)
            logger.info(f"Loaded Taillard instance with estimated makespan: {estimated_makespan}")
        else:  # manual
            if not args.input_json:
                parser.error("--input-json required with --manual mode")
            jobs, estimated_makespan = ManualJobShopGenerator.generate_from_file(args.input_json)
            logger.info(f"Loaded manual instance with estimated makespan: {estimated_makespan}")
        
        results = {}
        for algorithm in args.algorithms:
            # Create fresh environment for each algorithm
            env = JobShopEnvironment(jobs)
            scheduler = create_scheduler(algorithm)
            
            # Run scheduler
            result = run_scheduler(
                env, 
                scheduler, 
                args.max_steps,
                algorithm,
                args.show_critical_path
            )
            results[algorithm] = result
            
            # Generate and save Gantt data if requested
            gantt_data = env.generate_gantt_data() if args.save_gantt else None
            
            # Save individual solution
            solution_file = f"{args.output_prefix}_{algorithm}_solution.json"
            save_solution(
                solution_file,
                result["actions"],
                result["makespan"],
                env.get_schedule_data(),
                algorithm,
                result["solve_time"],
                result["metrics"],
                gantt_data,
                {"max_steps": args.max_steps}
            )
            
            # Generate visualization if requested
            if args.visualize:
                visualization_file = f"{args.output_prefix}_{algorithm}_gantt.html"
                env.generate_html_gantt(str(output_dir / visualization_file))
                logger.info(f"Gantt chart visualization saved to {visualization_file}")
            
            # Print summary
            logger.info(f"\n{algorithm.upper()} Solution Summary:")
            logger.info(f"Solve time: {result['solve_time']:.2f} seconds")
            logger.info(f"Makespan: {result['makespan']}")
            logger.info(f"Estimated lower bound: {estimated_makespan}")
            logger.info(f"Number of actions: {len(result['actions'])}")
            logger.info(f"Average machine utilization: {result['metrics']['avg_machine_utilization']*100:.1f}%")
        
        # Save comparative results
        comparative_results = {
            'instance_info': {
                'num_jobs': len(jobs),
                'num_machines': env.num_machines,
                'estimated_makespan': estimated_makespan
            },
            'results': {
                algo: {
                    'makespan': data['makespan'],
                    'solve_time': data['solve_time'],
                    'utilization': data['metrics']['avg_machine_utilization']
                }
                for algo, data in results.items()
            }
        }
        
        
        with open(output_dir / 'comparative_results.json', 'w') as f:
            json.dump(comparative_results, f, indent=2, cls=NumpyJSONEncoder)
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())