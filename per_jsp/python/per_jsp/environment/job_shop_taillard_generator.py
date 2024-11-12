from pathlib import Path
from typing import List, Tuple, Dict
import logging
from per_jsp.environment.job_shop_environment import Job
from per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator

logger = logging.getLogger(__name__)

class TaillardJobShopGenerator:
    @staticmethod
    def convert_to_job_shop_data(problem_data: str) -> Dict:
        """Convert Taillard format data to job shop data structure."""
        # Split the data into lines and remove empty lines
        lines = [line.strip() for line in problem_data.split('\n') if line.strip()]

        # Parse number of jobs and machines from first line
        num_jobs, num_machines = map(int, lines[0].split())
        logger.info(f"Converting Taillard data: {num_jobs} jobs, {num_machines} machines")

        # Initialize job shop data structure
        job_shop_data = {
            "metadata": {
                "numJobs": num_jobs * num_machines,  # Each operation becomes a job
                "numMachines": num_machines
            },
            "jobs": []
        }

        # Parse processing times
        processing_times = []
        current_line = 1
        for _ in range(num_jobs):
            times = list(map(int, lines[current_line].split()))
            processing_times.append(times)
            current_line += 1

        # Parse machine orders (1-based in input)
        machine_orders = []
        for _ in range(num_jobs):
            machines = list(map(lambda x: int(x) - 1, lines[current_line].split()))  # Convert to 0-based
            machine_orders.append(machines)
            current_line += 1

        # Convert each operation to a job
        job_counter = 0
        for i in range(num_jobs):
            for j in range(num_machines):
                job_data = {
                    "id": job_counter,
                    "duration": processing_times[i][j],
                    "machine": machine_orders[i][j],
                    "dependencies": [i * num_machines + (j - 1)] if j > 0 else []
                }

                logger.debug(f"Created job {job_counter} (original job {i}, op {j}) "
                             f"with duration {job_data['duration']} on machine {job_data['machine']}")

                job_shop_data["jobs"].append(job_data)
                job_counter += 1

        logger.info(f"Created {len(job_shop_data['jobs'])} jobs from {num_jobs} "
                    f"original jobs with {num_machines} machines each")

        return job_shop_data

    @staticmethod
    def load_file(file_path: str) -> str:
        """Load and read the Taillard format file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Failed to open file: {file_path}")

        with open(path, 'r') as f:
            return f.read()

    @classmethod
    def load_problem(cls, file_path: str) -> Tuple[List[Job], int]:
        """Load a Taillard instance from file."""
        logger.info(f"Loading Taillard instance from {file_path}")

        # Load and parse the file
        problem_data = cls.load_file(file_path)
        job_shop_data = cls.convert_to_job_shop_data(problem_data)

        # Generate job shop using ManualJobShopGenerator
        jobs, makespan_estimate = ManualJobShopGenerator.generate_from_data(job_shop_data)

        logger.info(f"Successfully loaded Taillard instance with "
                    f"{job_shop_data['metadata']['numJobs']} jobs and "
                    f"{job_shop_data['metadata']['numMachines']} machines")

        # Verify the generated data
        cls.verify_jobs_data(jobs)

        return jobs, makespan_estimate

    @staticmethod
    def verify_jobs_data(jobs: List[Job]):
        """Verify the integrity of the generated jobs data."""
        if not jobs:
            raise ValueError("Jobs vector is empty")

        for i, job in enumerate(jobs):
            if not job.operations:
                raise ValueError(f"Job {i} has no operations")

            # Verify each operation
            for j, op in enumerate(job.operations):
                if op.duration <= 0:
                    raise ValueError(f"Job {i} operation {j} has invalid duration: {op.duration}")

                if op.machine < 0:
                    raise ValueError(f"Job {i} operation {j} has invalid machine: {op.machine}")

                if len(op.eligible_machines) != 1:
                    raise ValueError(f"Job {i} operation {j} has invalid number of "
                                     f"eligible machines: {len(op.eligible_machines)}")

                logger.debug(f"Job {i} Op {j} - Duration: {op.duration}, Machine: {op.machine}")

            # Verify operation sequence
            for j in range(1, len(job.operations)):
                has_sequential_dependency = False
                for dep_job, dep_op in job.operations[j].dependent_operations:
                    if dep_job == i and dep_op == j - 1:
                        has_sequential_dependency = True
                        break

                if not has_sequential_dependency:
                    logger.warning(f"Job {i} Op {j} may not properly depend on previous operation")

        logger.info("Job data verification completed successfully")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example Taillard format data
    example_data = """4 3
1 3 2
2 1 4
2 3 1
1 2 3
3 2 1
1 2 3
2 1 3
3 2 1"""

    # Save example data to a temporary file
    with open("example_taillard.txt", "w") as f:
        f.write(example_data)

    # Load and process the example
    try:
        jobs, makespan = TaillardJobShopGenerator.load_problem("example_taillard.txt")
        print(f"Loaded job shop problem with estimated makespan: {makespan}")
        for i, job in enumerate(jobs):
            print(f"\nJob {i}:")
            print(f"Dependencies: {job.dependent_jobs}")
            for j, op in enumerate(job.operations):
                print(f"  Operation {j}: Duration={op.duration}, Machine={op.machine}")
    except Exception as e:
        print(f"Error loading problem: {e}")