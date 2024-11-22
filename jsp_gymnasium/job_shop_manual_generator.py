# manual_generator.py

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import json
from collections import defaultdict
from pathlib import Path
from job_shop_env import Job, Operation, JobShopGymEnv

logger = logging.getLogger(__name__)

class ManualGymGenerator:
    """Generator for manual job shop instances compatible with Gymnasium environment."""
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        while current_dir.name != "job-shop-simulator" and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        return current_dir

    @staticmethod
    def get_problem_path(filename: str) -> Path:
        """Get the full path to a problem instance file."""
        # Get project root
        project_root = ManualGymGenerator.get_project_root()
        
        # Check multiple possible locations
        possible_paths = [
            Path(filename),  # Direct path
            project_root / "per_jsp" / "data" / "problem_instances" / filename,
            project_root / "per_jsp" / "environments" / filename,
            project_root / "per_jsp" / "data" / filename
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found problem file at: {path}")
                return path
                
        # If not found, provide detailed error
        error_msg = f"Problem file '{filename}' not found. Searched in:\n"
        for path in possible_paths:
            error_msg += f"  - {path}\n"
        raise FileNotFoundError(error_msg)

    @staticmethod
    def list_available_problems() -> List[str]:
        """List all available problem instances."""
        project_root = ManualGymGenerator.get_project_root()
        
        # Search directories
        search_dirs = [
            project_root / "per_jsp" / "data" / "problem_instances",
            project_root / "per_jsp" / "environments",
            project_root / "per_jsp" / "data"
        ]
        
        problems = []
        for directory in search_dirs:
            if directory.exists():
                problems.extend(f.name for f in directory.glob("*.json"))
                
        return sorted(set(problems))  # Remove duplicates

    @staticmethod
    def detect_circular_dependencies(job_dependencies: List[List[int]]) -> bool:
        """Detect if there are any circular dependencies in the job graph."""
        visited = [0] * len(job_dependencies)  # 0: unvisited, 1: visiting, 2: visited
        recursion_stack = [0] * len(job_dependencies)

        def has_cycle(job: int) -> bool:
            if visited[job] == 0:
                visited[job] = 1
                recursion_stack[job] = 1

                for dep_job in job_dependencies[job]:
                    if dep_job >= len(job_dependencies):
                        logger.error(f"Invalid job dependency: Job {job} depends on non-existent job {dep_job}")
                        raise ValueError("Invalid job dependency reference")

                    if visited[dep_job] == 0 and has_cycle(dep_job):
                        return True
                    elif recursion_stack[dep_job] == 1:
                        logger.error(f"Circular dependency detected involving job {dep_job}")
                        return True

            recursion_stack[job] = 0
            visited[job] = 2
            return False

        for i in range(len(job_dependencies)):
            if visited[i] == 0 and has_cycle(i):
                return True
        return False

    @staticmethod
    def validate_job_shop(jobs: List[Job], num_machines: int):
        """Validate the job shop configuration."""
        logger.info("Performing job shop validation")

        machine_operations = defaultdict(set)

        for job_id, job in enumerate(jobs):
            for op_id, op in enumerate(job.operations):
                if op.machine >= num_machines:
                    logger.error(f"Job {job_id} Operation {op_id} assigned to invalid machine {op.machine}")
                    raise ValueError(f"Invalid machine assignment for Job {job_id} Operation {op_id}")

                machine_operations[op.machine].add(job_id)

                if op.duration <= 0:
                    logger.error(f"Job {job_id} Operation {op_id} has invalid duration {op.duration}")
                    raise ValueError(f"Invalid duration for Job {job_id} Operation {op_id}")

                for dep_job_id, dep_op_id in op.dependent_operations:
                    if dep_job_id >= len(jobs):
                        logger.error(f"Invalid operation dependency: Job {job_id} -> Job {dep_job_id}")
                        raise ValueError("Invalid operation dependency")

                    dep_job = jobs[dep_job_id]
                    if dep_op_id >= len(dep_job.operations):
                        logger.error(f"Invalid operation dependency: Job {job_id} Op {op_id} -> "
                                   f"Job {dep_job_id} Op {dep_op_id}")
                        raise ValueError("Invalid operation dependency")

        logger.info("Job shop validation complete")
        return True

    @classmethod
    def create_env_from_data(cls, job_shop_data: Dict, render_mode: Optional[str] = None) -> JobShopGymEnv:
        """Create a Gymnasium environment from job shop data."""
        logger.info("Creating environment from job shop data")
        
        num_jobs = job_shop_data["metadata"]["numJobs"]
        num_machines = job_shop_data["metadata"]["numMachines"]
        
        # Convert to job dependencies format
        job_dependencies = [[] for _ in range(num_jobs)]
        for job_data in job_shop_data["jobs"]:
            job_dependencies[job_data["id"]] = job_data.get("dependencies", [])
            
        # Check for circular dependencies
        if cls.detect_circular_dependencies(job_dependencies):
            raise ValueError("Circular dependencies detected in job configuration")
            
        # Create jobs
        jobs = [Job() for _ in range(num_jobs)]
        for job_data in job_shop_data["jobs"]:
            job = jobs[job_data["id"]]
            
            # Create operation
            op = Operation(
                duration=job_data["duration"],
                machine=job_data["machine"],
                eligible_machines={job_data["machine"]},
                dependent_operations=[]
            )
            
            # Add operation dependencies
            if "operationDependencies" in job_data:
                for op_dep in job_data["operationDependencies"]:
                    for dep_job_id in op_dep["dependencies"]:
                        op.dependent_operations.append((dep_job_id, 0))
            
            job.operations = [op]
            job.dependent_jobs = job_dependencies[job_data["id"]]
            
        # Validate configuration
        cls.validate_job_shop(jobs, num_machines)
        
        # Create environment
        env = JobShopGymEnv(jobs, render_mode=render_mode)
        
        logger.info(f"Created environment with {len(jobs)} jobs and {num_machines} machines")
        return env

    @classmethod
    def create_env_from_file(cls, file_path: str, render_mode: Optional[str] = None) -> JobShopGymEnv:
        """Create a Gymnasium environment from a JSON file."""
        logger.info(f"Creating environment from file: {file_path}")
        
        try:
            # Get full path to problem file
            problem_path = cls.get_problem_path(file_path)
            logger.info(f"Using problem file at: {problem_path}")
            
            with open(problem_path, 'r') as f:
                job_shop_data = json.load(f)
                
            return cls.create_env_from_data(job_shop_data, render_mode)
            
        except FileNotFoundError as e:
            logger.error(f"Problem file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in problem file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating environment: {e}")
            raise

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # List available problems
        print("\nAvailable problem instances:")
        problems = ManualGymGenerator.list_available_problems()
        if problems:
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("  No problem instances found")
            
        # Example job shop data
        example_data = {
            "metadata": {
                "numJobs": 3,
                "numMachines": 2,
                "name": "Example Problem"
            },
            "jobs": [
                {
                    "id": 0,
                    "duration": 4,
                    "machine": 0,
                    "dependencies": []
                },
                {
                    "id": 1,
                    "duration": 3,
                    "machine": 1,
                    "dependencies": [0]
                },
                {
                    "id": 2,
                    "duration": 5,
                    "machine": 0,
                    "dependencies": [1]
                }
            ]
        }

        # Create environment from example data
        print("\nTesting with example data:")
        env = ManualGymGenerator.create_env_from_data(example_data)
        
        # Test the environment
        print("\nTesting environment:")
        obs = env.reset()
        print("\nInitial state:")
        env.render()
        
        # Try a few random actions
        print("\nTesting random actions:")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"\nStep {i+1}:")
            print(f"Action: {env._decode_action(action)}")
            print(f"Reward: {reward}")
            print(f"Info: {info}")
            env.render()
            
            if done:
                break
                
    except Exception as e:
        logger.error(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    