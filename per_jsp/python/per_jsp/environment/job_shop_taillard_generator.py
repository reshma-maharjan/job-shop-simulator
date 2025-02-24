from pathlib import Path
from typing import List, Tuple, Dict
import logging
from per_jsp.python.per_jsp.environment.job_shop_environment import Job, Operation
from per_jsp.python.per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator

logger = logging.getLogger(__name__)

class TaillardJobShopGenerator:
    @staticmethod
    def load_file(file_path: str) -> str:
        """Load and read the Taillard format file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Failed to open file: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def convert_to_job_shop_data(problem_data: str) -> Dict:
        """Convert Taillard format data to job shop data structure."""
        # Split the data into lines and remove empty lines
        lines = [line.strip() for line in problem_data.split('\n') if line.strip()]

        # Parse number of jobs and machines from first line
        num_jobs, num_machines = map(int, lines[0].split())
        logger.info(f"Converting Taillard data: {num_jobs} jobs, {num_machines} machines")

        # First section: Processing times matrix (lines 1 to num_jobs)
        processing_times = []
        for i in range(num_jobs):
            times = list(map(int, lines[i + 1].split()))
            if len(times) != num_machines:
                raise ValueError(f"Invalid processing times for job {i}: expected {num_machines} values")
            processing_times.append(times)

        # Second section: Machine orders matrix (lines num_jobs+1 to 2*num_jobs)
        machine_orders = []
        machine_section_start = 1 + num_jobs
        for i in range(num_jobs):
            # Convert machine numbers from 1-based to 0-based indexing
            machines = list(map(lambda x: int(x) - 1, lines[machine_section_start + i].split()))
            if len(machines) != num_machines:
                raise ValueError(f"Invalid machine sequence for job {i}: expected {num_machines} values")
            machine_orders.append(machines)

        # Create job shop data structure
        job_shop_data = {
            "metadata": {
                "numJobs": num_jobs,
                "numMachines": num_machines
            },
            "jobs": []
        }

        # Create jobs with their operations in sequence
        for job_id in range(num_jobs):
            operations = []

            # Create each operation for this job
            for op_idx in range(num_machines):
                # Get the processing time and machine for this operation
                duration = processing_times[job_id][op_idx]
                machine = machine_orders[job_id][op_idx]

                operation = {
                    "duration": duration,
                    "machine": machine,
                    "eligible_machines": {machine},
                    "dependent_operations": []
                }

                # Add dependency on previous operation in the same job
                if op_idx > 0:
                    operation["dependent_operations"].append((job_id, op_idx - 1))

                operations.append(operation)

            job_data = {
                "id": job_id,
                "operations": operations,
                "dependencies": []  # Taillard instances don't have job-level dependencies
            }
            job_shop_data["jobs"].append(job_data)

        # Verify parsing is correct by logging first job
        first_job = job_shop_data["jobs"][0]
        logger.debug("\nVerifying first job parsing:")
        for op_idx, op in enumerate(first_job["operations"]):
            logger.debug(f"Op {op_idx}: Duration={op['duration']}, "
                         f"Original Machine={machine_orders[0][op_idx]+1}, "  # Convert back to 1-based for verification
                         f"Parsed Machine={op['machine']+1}")  # Convert back to 1-based for verification

        logger.info(f"Created {len(job_shop_data['jobs'])} jobs with {num_machines} operations each")
        return job_shop_data

    @staticmethod
    def verify_taillard_format(lines: List[str], num_jobs: int, num_machines: int):
        """Verify the Taillard instance format is correct."""
        expected_lines = 1 + 2 * num_jobs  # Header + processing times + machine sequences

        if len(lines) < expected_lines:
            raise ValueError(
                f"Invalid file format: expected {expected_lines} lines, "
                f"got {len(lines)}"
            )

        # Verify all lines have the correct number of values
        for i, line in enumerate(lines[1:expected_lines]):
            values = line.split()
            if len(values) != num_machines:
                row_type = "processing time" if i < num_jobs else "machine sequence"
                row_num = i % num_jobs
                raise ValueError(
                    f"Invalid {row_type} row {row_num}: "
                    f"expected {num_machines} values, got {len(values)}"
                )

    @classmethod
    def create_jobs_from_data(cls, job_shop_data: Dict) -> List[Job]:
        """Create Job instances directly without using ManualJobShopGenerator."""
        jobs = []
        for job_data in job_shop_data["jobs"]:
            operations = []
            for op_data in job_data["operations"]:
                operation = Operation(
                    duration=op_data["duration"],
                    machine=op_data["machine"],
                    eligible_machines=op_data["eligible_machines"],
                    dependent_operations=op_data["dependent_operations"]
                )
                operations.append(operation)

            job = Job(
                operations=operations,
                dependent_jobs=job_data["dependencies"]
            )
            jobs.append(job)
        return jobs

    @classmethod
    def load_problem(cls, file_path: str) -> Tuple[List[Job], int]:
        """Load a Taillard instance from file."""
        logger.info(f"Loading Taillard instance from {file_path}")

        # Load and verify the file
        problem_data = cls.load_file(file_path)
        lines = [line.strip() for line in problem_data.split('\n') if line.strip()]

        # Parse dimensions
        try:
            num_jobs, num_machines = map(int, lines[0].split())
        except (ValueError, IndexError) as e:
            raise ValueError("Invalid header format: expected 'num_jobs num_machines'") from e

        # Verify format before parsing
        cls.verify_taillard_format(lines, num_jobs, num_machines)

        # Convert to job shop data
        job_shop_data = cls.convert_to_job_shop_data(problem_data)

        # Create jobs directly
        jobs = cls.create_jobs_from_data(job_shop_data)

        # Calculate makespan estimate (sum of all processing times / number of machines)
        total_processing_time = sum(
            op.duration for job in jobs for op in job.operations
        )
        makespan_estimate = total_processing_time // num_machines

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

def verify_taillard_instance():
    """Verify Taillard instance parsing."""
    logger.info("Verifying Taillard instance parsing")

    # Load first few lines of ta01
    with open("/home/per/jsp/per_jsp/data/taillard_instances/ta01.txt", 'r') as f:
        content = f.readlines()

    logger.info("First few lines of ta01:")
    logger.info("\n".join(content))

    # Parse and verify structure
    problem_data = TaillardJobShopGenerator.load_file("/workspaces/job-shop-simulator/per_jsp/data/taillard_instances/ta01.txt")
    jobs, _ = TaillardJobShopGenerator.load_problem("/workspaces/job-shop-simulator/per_jsp/data/taillard_instances/ta01.txt")

    # Check first job's operations
    logger.info("\nFirst job operations:")
    for i, op in enumerate(jobs[0].operations):
        logger.info(f"Op {i}: Machine {op.machine}, Duration {op.duration}, "
                    f"Dependencies {op.dependent_operations}")

if __name__ == "__main__":
    verify_taillard_instance()