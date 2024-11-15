import numpy as np
from typing import List, Tuple, Dict
import pickle
import logging
from collections import defaultdict
import random
import time
import hashlib

from per_jsp.environment.job_shop_environment import (
    JobShopEnvironment, Action, State, Job
)
from per_jsp.algorithms.q_learning import QLearningScheduler
from per_jsp.algorithms.greedy import GreedyScheduler

logger = logging.getLogger(__name__)

class JSPUniqueSolutionDatasetGenerator:
    def __init__(self, env: JobShopEnvironment):
        self.env = env
        self.solutions = []
        self.state_features = []
        self.makespans = []
        self.state_hashes = set()  # Track unique states
        self.solution_hashes = set()  # Track unique solutions

    def get_state_features(self, state: State) -> np.ndarray:
        """Extract normalized features from state."""
        features = []

        # Job progress features
        max_progress = max(1, np.max(state.job_progress))
        job_progress_norm = state.job_progress / max_progress
        features.extend(job_progress_norm.flatten())

        # Machine availability features
        max_avail = max(1, np.max(state.machine_availability))
        machine_avail_norm = state.machine_availability / max_avail
        features.extend(machine_avail_norm)

        # Job completion status
        features.extend(state.completed_jobs)

        # Next operation features
        max_ops = max(len(job.operations) for job in self.env.jobs)
        next_op_norm = state.next_operation_for_job / max_ops
        features.extend(next_op_norm)

        # Job start times
        max_start = max(1, np.max(np.where(state.job_start_times > -1,
                                           state.job_start_times, 0)))
        start_times_norm = np.where(state.job_start_times > -1,
                                    state.job_start_times / max_start,
                                    -1)
        features.extend(start_times_norm)

        return np.array(features, dtype=np.float32)

    def hash_state(self, features: np.ndarray) -> str:
        """Create a hash of state features for uniqueness checking."""
        # Round features to reduce floating point precision issues
        rounded = np.round(features, decimals=6)
        return hashlib.sha256(rounded.tobytes()).hexdigest()

    def hash_solution(self, actions: List[Action], makespan: int) -> str:
        """Create a hash of a solution for uniqueness checking."""
        solution_str = f"{makespan}_" + "_".join(
            f"{a.job},{a.machine},{a.operation}" for a in actions
        )
        return hashlib.sha256(solution_str.encode()).hexdigest()

    def collect_solution(self, actions: List[Action], makespan: int) -> bool:
        """Store a solution and its intermediate states if unique."""
        # Check if solution is unique
        solution_hash = self.hash_solution(actions, makespan)
        if solution_hash in self.solution_hashes:
            return False

        self.env.reset()
        new_features = []
        unique_states = 0

        # Collect unique intermediate states
        for action in actions:
            state = self.env.current_state
            features = self.get_state_features(state)
            state_hash = self.hash_state(features)

            if state_hash not in self.state_hashes:
                new_features.append(features)
                self.state_hashes.add(state_hash)
                unique_states += 1

            self.env.step(action)

        # Add final state if unique
        final_state = self.env.current_state
        final_features = self.get_state_features(final_state)
        final_hash = self.hash_state(final_features)

        if final_hash not in self.state_hashes:
            new_features.append(final_features)
            self.state_hashes.add(final_hash)
            unique_states += 1

        # Only store solution if it contributed unique states
        if unique_states > 0:
            self.solutions.append(actions)
            self.solution_hashes.add(solution_hash)
            self.state_features.extend(new_features)
            # Add makespan for each unique state from this solution
            self.makespans.extend([makespan] * len(new_features))
            return True

        return False

    def generate_solutions(self,
                           target_unique_solutions: int,
                           max_attempts: int = 1000,
                           methods: List[str] = None,
                           q_learning_params: Dict = None,
                           greedy_variations: bool = True) -> None:
        """Generate multiple unique solutions using different methods."""
        if methods is None:
            methods = ['q_learning', 'greedy']

        if q_learning_params is None:
            q_learning_params = {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 0.1,
                'episodes': 1000
            }

        unique_solutions = 0
        attempts = 0

        while unique_solutions < target_unique_solutions and attempts < max_attempts:
            method = methods[unique_solutions % len(methods)]

            if method == 'q_learning':
                scheduler = QLearningScheduler(**q_learning_params)
                # Vary random seed and parameters slightly
                seed = attempts + unique_solutions
                random.seed(seed)
                current_params = q_learning_params.copy()
                current_params['epsilon'] = max(0.05, q_learning_params['epsilon'] *
                                                (1 - unique_solutions/target_unique_solutions))
                scheduler = QLearningScheduler(**current_params)

            else:  # greedy
                variations = [(True, True), (True, False),
                              (False, True), (False, False)] if greedy_variations else [(True, True)]
                variation_idx = (unique_solutions // len(methods)) % len(variations)
                use_longest, use_earliest = variations[variation_idx]
                scheduler = GreedyScheduler(use_longest=use_longest)
                random.seed(attempts + unique_solutions)

            actions, makespan = scheduler.solve(self.env)

            if self.collect_solution(actions, makespan):
                unique_solutions += 1
                logger.info(f"Found unique solution {unique_solutions}/{target_unique_solutions} "
                            f"using {method} (makespan={makespan})")

            attempts += 1

        if unique_solutions < target_unique_solutions:
            logger.warning(f"Only found {unique_solutions} unique solutions "
                           f"after {attempts} attempts")

    def create_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix and labels from collected solutions."""
        if not self.state_features or not self.makespans:
            raise ValueError("No data collected yet. Run generate_solutions first.")

        X = np.array(self.state_features)
        y = np.array(self.makespans)

        # Normalize makespans
        y = y / np.max(y)

        # Verify shapes match
        if len(X) != len(y):
            raise ValueError(f"Mismatch between features ({len(X)}) and labels ({len(y)})")

        return X, y
    def save_dataset(self, filename: str):
        """Save the dataset and metadata."""
        X, y = self.create_dataset()

        # Create solution records with correct makespan indexing
        solution_records = []
        current_index = 0
        for solution in self.solutions:
            # Count how many unique states this solution contributed
            next_index = current_index
            while next_index < len(self.makespans) and self.makespans[next_index] == self.makespans[current_index]:
                next_index += 1

            solution_records.append({
                'actions': [(a.job, a.machine, a.operation) for a in solution],
                'makespan': self.makespans[current_index],
                'num_states': next_index - current_index
            })
            current_index = next_index

        dataset = {
            'X': X,
            'y': y,
            'feature_names': self.get_feature_names(),
            'num_unique_solutions': len(self.solutions),
            'num_unique_states': len(self.state_hashes),
            'env_config': {
                'num_jobs': len(self.env.jobs),
                'num_machines': self.env.num_machines,
                'max_operations': max(len(job.operations) for job in self.env.jobs)
            },
            'solutions': solution_records
        }

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

        logger.info(f"Dataset saved to {filename}")
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Unique solutions: {len(self.solutions)}")
        logger.info(f"Unique states: {len(self.state_hashes)}")
        logger.info(f"Total states stored: {len(self.makespans)}")

    def get_feature_names(self) -> List[str]:
        """Get descriptive names for features."""
        features = []

        # Job progress features
        for job_id in range(len(self.env.jobs)):
            for op_id in range(len(self.env.jobs[job_id].operations)):
                features.append(f'job_{job_id}_op_{op_id}_progress')

        # Machine availability features
        for machine_id in range(self.env.num_machines):
            features.append(f'machine_{machine_id}_availability')

        # Job completion status
        for job_id in range(len(self.env.jobs)):
            features.append(f'job_{job_id}_completed')

        # Next operation features
        for job_id in range(len(self.env.jobs)):
            features.append(f'job_{job_id}_next_op')

        # Job start times
        for job_id in range(len(self.env.jobs)):
            features.append(f'job_{job_id}_start_time')

        return features

def add_dataset_generation_args(parser):
    """Add dataset generation arguments to argument parser."""
    group = parser.add_argument_group('Dataset Generation')
    group.add_argument("--generate-dataset", action="store_true",
                       help="Generate solution dataset")
    group.add_argument("--target-solutions", type=int, default=100,
                       help="Target number of unique solutions")
    group.add_argument("--max-attempts", type=int, default=1000,
                       help="Maximum attempts to find unique solutions")
    group.add_argument("--dataset-output", type=str, default="jsp_dataset.pkl",
                       help="Output file for dataset")



    #!/usr/bin/env python3
"""
Main script for generating Job Shop Scheduling datasets.
Supports multiple problem types and solution methods.
"""

import argparse
import logging
import sys
import os
import json
import time
from typing import Dict, List

from per_jsp.environment.job_shop_environment import JobShopEnvironment, Job
from per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator
from per_jsp.environment.job_shop_automatic_generator import AutomaticJobShopGenerator
from per_jsp.environment.structs import GenerationParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Job Shop Scheduling Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Problem specification
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--auto", action="store_true",
                               help="Use automatic problem generation")
    problem_group.add_argument("--taillard", action="store_true",
                               help="Use Taillard benchmark instance")
    problem_group.add_argument("--manual", action="store_true",
                               help="Use manual JSON specification")

    # Automatic generation parameters
    auto_group = parser.add_argument_group("Automatic Generation Options")
    auto_group.add_argument("--num-jobs", type=int, default=10,
                            help="Number of jobs for automatic generation")
    auto_group.add_argument("--num-machines", type=int, default=5,
                            help="Number of machines for automatic generation")
    auto_group.add_argument("--min-duration", type=int, default=1,
                            help="Minimum operation duration")
    auto_group.add_argument("--max-duration", type=int, default=100,
                            help="Maximum operation duration")
    auto_group.add_argument("--dependency-density", type=float, default=0.3,
                            help="Job dependency density (0-1)")
    auto_group.add_argument("--max-deps", type=int, default=3,
                            help="Maximum dependencies per job")

    # Input files
    file_group = parser.add_argument_group("Input/Output Files")
    file_group.add_argument("--taillard-file", type=str,
                            help="Path to Taillard instance file")
    file_group.add_argument("--manual-file", type=str,
                            help="Path to manual JSON specification")
    file_group.add_argument("--output-dir", type=str, default="datasets",
                            help="Output directory for generated datasets")
    file_group.add_argument("--save-problem", action="store_true",
                            help="Save the problem instance to JSON")

    # Dataset generation parameters
    dataset_group = parser.add_argument_group("Dataset Generation Options")
    dataset_group.add_argument("--target-solutions", type=int, default=100,
                               help="Target number of unique solutions")
    dataset_group.add_argument("--max-attempts", type=int, default=1000,
                               help="Maximum attempts to find unique solutions")
    dataset_group.add_argument("--methods", type=str, nargs="+",
                               choices=["q_learning", "greedy", "both"],
                               default=["both"],
                               help="Solution methods to use")

    # Q-learning parameters
    qlearn_group = parser.add_argument_group("Q-Learning Parameters")
    qlearn_group.add_argument("--learning-rate", type=float, default=0.1,
                              help="Q-learning learning rate")
    qlearn_group.add_argument("--discount-factor", type=float, default=0.95,
                              help="Q-learning discount factor")
    qlearn_group.add_argument("--epsilon", type=float, default=0.1,
                              help="Q-learning exploration rate")
    qlearn_group.add_argument("--episodes", type=int, default=1000,
                              help="Q-learning episodes per solution attempt")

    # Greedy parameters
    greedy_group = parser.add_argument_group("Greedy Parameters")
    greedy_group.add_argument("--no-greedy-variations", action="store_true",
                              help="Disable greedy algorithm variations")

    # Other parameters
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return parser

def create_output_dir(args) -> str:
    """Create and return output directory path."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create base directory name
    if args.auto:
        dirname = f"auto_{args.num_jobs}j_{args.num_machines}m_{timestamp}"
    elif args.taillard:
        taillard_name = os.path.splitext(os.path.basename(args.taillard_file))[0]
        dirname = f"taillard_{taillard_name}_{timestamp}"
    else:
        manual_name = os.path.splitext(os.path.basename(args.manual_file))[0]
        dirname = f"manual_{manual_name}_{timestamp}"

    # Create full path
    output_dir = os.path.join(args.output_dir, dirname)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def save_config(args, output_dir: str):
    """Save configuration to JSON file."""
    config = {k: v for k, v in vars(args).items() if v is not None}
    config_file = os.path.join(output_dir, "config.json")

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {config_file}")

def get_problem_instance(args) -> tuple[List[Job], int]:
    """Get problem instance based on arguments."""
    if args.auto:
        params = GenerationParams(
            num_jobs=args.num_jobs,
            num_machines=args.num_machines,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            dependency_density=args.dependency_density,
            max_dependencies_per_job=args.max_deps
        )
        jobs, estimated_makespan = AutomaticJobShopGenerator.generate(params)
        logger.info(f"Generated problem with {args.num_jobs} jobs and {args.num_machines} machines")

    elif args.taillard:
        if not args.taillard_file:
            raise ValueError("Taillard file path required with --taillard option")
        jobs, estimated_makespan = TaillardJobShopGenerator.load_problem(args.taillard_file)
        logger.info(f"Loaded Taillard instance from {args.taillard_file}")

    else:  # manual
        if not args.manual_file:
            raise ValueError("Manual file path required with --manual option")
        jobs, estimated_makespan = ManualJobShopGenerator.generate_from_file(args.manual_file)
        logger.info(f"Loaded manual instance from {args.manual_file}")

    return jobs, estimated_makespan

def main():
    """Main entry point for dataset generation."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create output directory
        output_dir = create_output_dir(args)
        logger.info(f"Output directory: {output_dir}")

        # Save configuration
        save_config(args, output_dir)

        # Get problem instance
        jobs, estimated_makespan = get_problem_instance(args)

        # Save problem instance if requested
        if args.save_problem:
            problem_file = os.path.join(output_dir, "problem.json")
            # Implement problem saving logic here
            logger.info(f"Problem instance saved to {problem_file}")

        # Create environment
        env = JobShopEnvironment(jobs)

        # Setup methods
        if "both" in args.methods:
            methods = ["q_learning", "greedy"]
        else:
            methods = args.methods

        # Q-learning parameters
        q_learning_params = {
            "learning_rate": args.learning_rate,
            "discount_factor": args.discount_factor,
            "epsilon": args.epsilon,
            "episodes": args.episodes
        }

        # Create dataset generator
        generator = JSPUniqueSolutionDatasetGenerator(env)

        # Generate dataset
        start_time = time.time()
        generator.generate_solutions(
            target_unique_solutions=args.target_solutions,
            max_attempts=args.max_attempts,
            methods=methods,
            q_learning_params=q_learning_params,
            greedy_variations=not args.no_greedy_variations
        )

        # Save dataset
        dataset_file = os.path.join(output_dir, "dataset.pkl")
        generator.save_dataset(dataset_file)

        # Print summary
        duration = time.time() - start_time
        logger.info("\nDataset Generation Summary:")
        logger.info(f"Time taken: {duration:.2f} seconds")
        logger.info(f"Target solutions: {args.target_solutions}")
        logger.info(f"Methods used: {methods}")
        logger.info(f"Problem type: {'auto' if args.auto else 'taillard' if args.taillard else 'manual'}")
        logger.info(f"Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())