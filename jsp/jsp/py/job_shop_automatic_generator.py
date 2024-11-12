import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
import json
from collections import defaultdict

from job_shop_environment import Job
from job_shop_manual_generator import ManualJobShopGenerator

logger = logging.getLogger(__name__)

@dataclass
class GenerationParams:
    """Parameters for generating job shop problems."""
    num_jobs: int
    num_machines: int
    min_duration: int
    max_duration: int
    dependency_density: float
    max_dependencies_per_job: int
    long_job_rate: float = 0.1
    long_job_factor: float = 3.0
    output_file: Optional[str] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.num_jobs <= 0:
            raise ValueError("Number of jobs must be positive")
        if self.num_machines <= 0:
            raise ValueError("Number of machines must be positive")
        if self.min_duration <= 0:
            raise ValueError("Minimum duration must be positive")
        if self.max_duration < self.min_duration:
            raise ValueError("Maximum duration must be greater than or equal to minimum duration")
        if not 0.0 <= self.dependency_density <= 1.0:
            raise ValueError("Dependency density must be between 0.0 and 1.0")
        if not 0.0 <= self.long_job_rate <= 1.0:
            raise ValueError("Long job rate must be between 0.0 and 1.0")
        if self.long_job_factor < 1.0:
            raise ValueError("Long job factor must be greater than 1.0")

class AutomaticJobShopGenerator:
    """Generator for random job shop scheduling problems."""

    @staticmethod
    def generate_dependencies(
            num_jobs: int,
            dependency_density: float,
            max_dependencies_per_job: int) -> List[List[int]]:
        """
        Generate random dependencies between jobs.

        Args:
            num_jobs: Number of jobs
            dependency_density: Probability of creating dependencies
            max_dependencies_per_job: Maximum dependencies per job

        Returns:
            List of lists containing job dependencies
        """
        dependencies = [[] for _ in range(num_jobs)]

        for i in range(num_jobs):
            max_possible_deps = min(i, max_dependencies_per_job)
            if max_possible_deps == 0:
                continue

            # Determine number of dependencies based on density
            num_deps = 0
            if random.random() < dependency_density:
                num_deps = random.randint(0, max_possible_deps)

            if num_deps > 0:
                # Create list of possible dependencies (jobs that come before current job)
                possible_deps = list(range(i))
                random.shuffle(possible_deps)
                dependencies[i] = sorted(possible_deps[:num_deps])

        # Verify no circular dependencies
        if AutomaticJobShopGenerator.detect_circular_dependencies(dependencies):
            logger.warning("Circular dependencies detected, regenerating dependencies")
            return AutomaticJobShopGenerator.generate_dependencies(
                num_jobs, dependency_density, max_dependencies_per_job
            )

        return dependencies

    @staticmethod
    def detect_circular_dependencies(dependencies: List[List[int]]) -> bool:
        """
        Detect if there are any circular dependencies in the job graph.

        Args:
            dependencies: List of lists containing job dependencies

        Returns:
            True if circular dependencies exist, False otherwise
        """
        num_jobs = len(dependencies)
        visited = [0] * num_jobs  # 0: unvisited, 1: in progress, 2: done

        def has_cycle(job: int) -> bool:
            if visited[job] == 1:  # Node being visited - cycle detected
                return True
            if visited[job] == 2:  # Node already fully explored
                return False

            visited[job] = 1  # Mark as being visited

            # Check all dependencies
            for dep in dependencies[job]:
                if has_cycle(dep):
                    return True

            visited[job] = 2  # Mark as fully explored
            return False

        # Check each job
        for job in range(num_jobs):
            if visited[job] == 0:  # Unvisited
                if has_cycle(job):
                    return True

        return False

    @staticmethod
    def generate_durations(
            num_jobs: int,
            min_duration: int,
            max_duration: int,
            long_job_rate: float = 0.1,
            long_job_factor: float = 3.0) -> List[int]:
        """
        Generate random durations for jobs.

        Args:
            num_jobs: Number of jobs
            min_duration: Minimum operation duration
            max_duration: Maximum operation duration
            long_job_rate: Probability of generating a long job
            long_job_factor: Multiplier for long job durations

        Returns:
            List of job durations
        """
        durations = []

        for _ in range(num_jobs):
            if random.random() < long_job_rate:
                # Generate a long job
                duration = random.randint(
                    int(min_duration * long_job_factor),
                    int(max_duration * long_job_factor)
                )
            else:
                # Generate a normal job
                duration = random.randint(min_duration, max_duration)
            durations.append(duration)

        return durations

    @staticmethod
    def assign_machines(
            num_jobs: int,
            num_machines: int,
            balance_factor: float = 0.8) -> List[int]:
        """
        Assign machines to jobs with optional load balancing.

        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            balance_factor: Factor for load balancing (0-1)

        Returns:
            List of machine assignments
        """
        assignments = []
        machine_loads = [0] * num_machines

        for _ in range(num_jobs):
            if random.random() < balance_factor:
                # Assign to least loaded machine
                machine = machine_loads.index(min(machine_loads))
            else:
                # Random assignment
                machine = random.randint(0, num_machines - 1)

            assignments.append(machine)
            machine_loads[machine] += 1

        return assignments

    @staticmethod
    def create_job_shop_data(
            num_jobs: int,
            num_machines: int,
            durations: List[int],
            dependencies: List[List[int]]) -> Dict:
        """
        Create job shop data structure from generated components.

        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            durations: List of job durations
            dependencies: List of job dependencies

        Returns:
            Dictionary containing job shop data
        """
        machine_assignments = AutomaticJobShopGenerator.assign_machines(num_jobs, num_machines)

        job_shop_data = {
            "metadata": {
                "numJobs": num_jobs,
                "numMachines": num_machines
            },
            "jobs": []
        }

        for i in range(num_jobs):
            job_data = {
                "id": i,
                "duration": durations[i],
                "machine": machine_assignments[i],
                "dependencies": dependencies[i]
            }
            job_shop_data["jobs"].append(job_data)

        return job_shop_data

    @staticmethod
    def calculate_makespan(job_shop_data: Dict) -> int:
        """
        Calculate estimated makespan for the generated problem.

        Args:
            job_shop_data: Dictionary containing job shop data

        Returns:
            Estimated makespan
        """
        num_machines = job_shop_data["metadata"]["numMachines"]
        machine_loads = [0] * num_machines
        job_end_times = [0] * len(job_shop_data["jobs"])

        # Calculate machine loads and critical path
        for job in job_shop_data["jobs"]:
            job_id = job["id"]
            duration = job["duration"]
            machine = job["machine"]

            # Consider dependencies
            start_time = 0
            for dep_id in job["dependencies"]:
                start_time = max(start_time, job_end_times[dep_id])

            # Update job end time
            job_end_times[job_id] = start_time + duration

            # Update machine load
            machine_loads[machine] += duration

        # Consider both critical path and machine loads
        critical_path = max(job_end_times)
        max_machine_load = max(machine_loads)

        return max(critical_path, max_machine_load)

    @classmethod
    def generate(cls, params: GenerationParams) -> Tuple[List[Job], int]:
        """
        Generate a random job shop problem.

        Args:
            params: Generation parameters

        Returns:
            Tuple of (jobs list, estimated makespan)
        """
        logger.info(f"Generating random job shop problem with {params.num_jobs} jobs "
                    f"and {params.num_machines} machines")

        # Generate problem components
        dependencies = cls.generate_dependencies(
            params.num_jobs,
            params.dependency_density,
            params.max_dependencies_per_job
        )

        durations = cls.generate_durations(
            params.num_jobs,
            params.min_duration,
            params.max_duration,
            params.long_job_rate,
            params.long_job_factor
        )

        # Create data structure
        job_shop_data = cls.create_job_shop_data(
            params.num_jobs,
            params.num_machines,
            durations,
            dependencies
        )

        # Save to file if specified
        if params.output_file:
            with open(params.output_file, 'w') as f:
                json.dump(job_shop_data, f, indent=2)
            logger.info(f"Saved generated instance to {params.output_file}")

        # Convert to Job objects and calculate makespan
        makespan = cls.calculate_makespan(job_shop_data)
        return ManualJobShopGenerator.generate_from_data(job_shop_data)

    @classmethod
    def generate_default(
            cls,
            num_jobs: int,
            num_machines: int,
            output_file: Optional[str] = None) -> Tuple[List[Job], int]:
        """
        Generate a job shop problem with default parameters.

        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            output_file: Optional file to save the instance

        Returns:
            Tuple of (jobs list, estimated makespan)
        """
        params = GenerationParams(
            num_jobs=num_jobs,
            num_machines=num_machines,
            min_duration=1,
            max_duration=100,
            dependency_density=0.3,
            max_dependencies_per_job=3,
            long_job_rate=0.1,
            long_job_factor=2.0,
            output_file=output_file
        )

        return cls.generate(params)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Example 1: Generate with default parameters
    jobs1, makespan1 = AutomaticJobShopGenerator.generate_default(
        num_jobs=5,
        num_machines=3,
        output_file="default_instance.json"
    )
    print(f"\nDefault generation - Makespan: {makespan1}")

    # Example 2: Generate with custom parameters
    params = GenerationParams(
        num_jobs=8,
        num_machines=4,
        min_duration=5,
        max_duration=50,
        dependency_density=0.4,
        max_dependencies_per_job=3,
        long_job_rate=0.2,
        long_job_factor=2.5,
        output_file="custom_instance.json"
    )

    jobs2, makespan2 = AutomaticJobShopGenerator.generate(params)
    print(f"\nCustom generation - Makespan: {makespan2}")

    # Print some statistics
    def print_job_stats(jobs: List[Job], label: str):
        print(f"\n{label} Statistics:")
        print(f"Number of jobs: {len(jobs)}")
        print(f"Average duration: {np.mean([op.duration for job in jobs for op in job.operations]):.2f}")
        print("Machine utilization:", end=" ")
        machine_loads = defaultdict(int)
        for job in jobs:
            for op in job.operations:
                machine_loads[op.machine] += op.duration
        for machine, load in sorted(machine_loads.items()):
            print(f"M{machine}:{load}", end=" ")
        print()

    print_job_stats(jobs1, "Default Instance")
    print_job_stats(jobs2, "Custom Instance")