# Job_shop_taillard_generator.py

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
from job_shop_env import Job, Operation, JobShopGymEnv

logger = logging.getLogger(__name__)

class TaillardGymGenerator:
    """Generator for Taillard job shop instances compatible with Gymnasium environment."""
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        while current_dir.name != "job-shop-simulator" and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        return current_dir

    @staticmethod
    def get_taillard_path(instance_name: str) -> str:
        """Get the full path to a Taillard instance file."""
        project_root = TaillardGymGenerator.get_project_root()
        
        # Updated path to match project structure
        taillard_path = project_root / "per_jsp" / "data" / "taillard_instances" / instance_name
        
        if not taillard_path.exists():
            raise FileNotFoundError(f"Taillard instance not found: {taillard_path}")
            
        return str(taillard_path)

    @staticmethod
    def list_available_instances() -> List[str]:
        """List all available Taillard instances in the project."""
        project_root = TaillardGymGenerator.get_project_root()
        # Updated path to match project structure
        taillard_dir = project_root / "per_jsp" / "data" / "taillard_instances"
        
        logger.info(f"Looking for Taillard instances in: {taillard_dir}")
        
        if not taillard_dir.exists():
            logger.warning(f"Taillard instances directory not found at: {taillard_dir}")
            return []
        
        # List all .txt files in the directory
        instances = [f.name for f in taillard_dir.glob("ta*.txt")]
        logger.info(f"Found {len(instances)} Taillard instances")
        
        return sorted(instances)

    @staticmethod
    def parse_taillard_file(file_path: str) -> Tuple[List[List[int]], List[List[int]]]:
        """Parse a Taillard format file into processing times and machine orders."""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse dimensions
        num_jobs, num_machines = map(int, lines[0].split())
        
        # Parse processing times
        processing_times = []
        current_line = 1
        for _ in range(num_jobs):
            times = list(map(int, lines[current_line].split()))
            processing_times.append(times)
            current_line += 1
            
        # Parse machine orders (convert to 0-based indexing)
        machine_orders = []
        for _ in range(num_jobs):
            machines = list(map(lambda x: int(x) - 1, lines[current_line].split()))
            machine_orders.append(machines)
            current_line += 1
            
        return processing_times, machine_orders

    @staticmethod
    def create_jobs(processing_times: List[List[int]], 
                   machine_orders: List[List[int]]) -> List[Job]:
        """Create Job objects from Taillard data."""
        jobs = []
        num_jobs = len(processing_times)
        num_machines = len(processing_times[0])
        
        for job_idx in range(num_jobs):
            operations = []
            for op_idx in range(num_machines):
                machine = machine_orders[job_idx][op_idx]
                duration = processing_times[job_idx][op_idx]
                
                operation = Operation(
                    duration=duration,
                    machine=machine,
                    eligible_machines={machine},
                    dependent_operations=[]
                )
                
                if op_idx > 0:
                    operation.dependent_operations.append((job_idx, op_idx - 1))
                    
                operations.append(operation)
            
            job = Job(operations=operations)
            jobs.append(job)
            
        return jobs

    @classmethod
    def create_env_from_instance(cls, instance_name: str, render_mode: Optional[str] = None) -> JobShopGymEnv:
        """Create a Gymnasium environment from a Taillard instance name."""
        logger.info(f"Loading Taillard instance: {instance_name}")
        
        file_path = cls.get_taillard_path(instance_name)
        logger.info(f"Using instance file: {file_path}")
        
        processing_times, machine_orders = cls.parse_taillard_file(file_path)
        jobs = cls.create_jobs(processing_times, machine_orders)
        
        env = JobShopGymEnv(jobs, render_mode=render_mode)
        
        logger.info(f"Created environment from {instance_name} with {len(jobs)} jobs, "
                   f"{len(jobs[0].operations)} machines per job")
        
        return env

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        project_root = TaillardGymGenerator.get_project_root()
        print(f"\nProject root directory: {project_root}")
        
        # Updated path to match project structure
        taillard_dir = project_root / "per_jsp" / "data" / "taillard_instances"
        print(f"Taillard instances directory: {taillard_dir}")
        
        if taillard_dir.exists():
            print("Taillard instances directory found")
        else:
            print("WARNING: Taillard instances directory not found!")
            return
        
        # List available instances
        instances = TaillardGymGenerator.list_available_instances()
        if instances:
            print("\nAvailable Taillard instances:")
            for instance in instances:
                print(f"  {instance}")
                # Print file size and first few lines
                with open(taillard_dir / instance, 'r') as f:
                    content = f.readlines()
                    size = Path(taillard_dir / instance).stat().st_size
                    print(f"    Size: {size} bytes")
                    print(f"    First line: {content[0].strip()}")
            
            # Create environment from first instance
            instance_name = instances[0]
            print(f"\nLoading instance: {instance_name}")
            
            env = TaillardGymGenerator.create_env_from_instance(instance_name)
            
            # Test environment
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
        else:
            print("\nNo Taillard instances found in the directory")
            print("\nPlease ensure the following:")
            print("1. The directory contains Taillard instance files (ta*.txt)")
            print("2. You have appropriate read permissions")
            print(f"3. Check the contents of: {taillard_dir}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()