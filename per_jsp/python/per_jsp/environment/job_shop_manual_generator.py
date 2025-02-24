import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import logging
import json
from collections import defaultdict, deque

from per_jsp.python.per_jsp.environment.job_shop_environment import Operation, Job, MachineSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManualJobShopGenerator:
    @staticmethod
    def detect_circular_dependencies(job_dependencies: List[List[int]]) -> bool:
        """
        Detect if there are any circular dependencies in the job graph.
        Uses depth-first search to detect cycles.
        """
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
    def identify_dependency_chains(jobs: List[Job], job_dependencies: List[List[int]]) -> List[List[int]]:
        """
        Identify chains of dependent jobs using topological sort and path finding.
        Returns a list of dependency chains.
        """
        num_jobs = len(jobs)
        in_degree = [0] * num_jobs
        chains = []

        # Calculate in-degree for each job
        for deps in job_dependencies:
            for dep in deps:
                in_degree[dep] += 1

        # Find all root jobs (no incoming dependencies)
        roots = [i for i, deg in enumerate(in_degree) if deg == 0]

        def find_longest_path(start: int) -> List[int]:
            distances = [-float('inf')] * num_jobs
            distances[start] = 0
            queue = deque([start])
            predecessors = [-1] * num_jobs

            while queue:
                current = queue.popleft()
                current_dist = distances[current]

                # Check all jobs that depend on current
                for next_job, deps in enumerate(job_dependencies):
                    if current in deps:
                        if distances[next_job] < current_dist + 1:
                            distances[next_job] = current_dist + 1
                            predecessors[next_job] = current
                            queue.append(next_job)

            # Find the end of the longest path
            end_job = max(range(num_jobs), key=lambda x: distances[x])

            # Reconstruct the path
            path = []
            current = end_job
            while current != -1:
                path.append(current)
                current = predecessors[current]

            return list(reversed(path))

        # Find longest path starting from each root
        for root in roots:
            chain = find_longest_path(root)
            if chain:
                chains.append(chain)

        return chains

    @staticmethod
    def calculate_chain_conflict(jobs: List[Job], chain1: List[int], chain2: List[int]) -> int:
        """Calculate conflict score between two chains based on shared machine and tool usage."""
        machines1 = set()
        machines2 = set()
        tools1 = set()
        tools2 = set()

        for job_id in chain1:
            for op in jobs[job_id].operations:
                machines1.add(op.machine)
                tools1.update(op.required_tools)
        for job_id in chain2:
            for op in jobs[job_id].operations:
                machines2.add(op.machine)
                tools2.update(op.required_tools)

        machine_conflicts = len(machines1.intersection(machines2))
        tool_conflicts = len(tools1.intersection(tools2))
        
        return machine_conflicts + tool_conflicts

    @staticmethod
    def calculate_chain_duration(jobs: List[Job], chain: List[int], machine_specs: Optional[List[MachineSpec]] = None) -> int:
        """Calculate the total duration of a chain including machine conflicts and tool changes."""
        duration = 0
        used_machines = set()
        required_tools = defaultdict(set)  # machine -> set of tools

        for job_id in chain:
            # Add operation duration
            job_duration = sum(op.duration for op in jobs[job_id].operations)
            duration += job_duration

            # Track machine and tool usage
            for op in jobs[job_id].operations:
                used_machines.add(op.machine)
                required_tools[op.machine].update(op.required_tools)

        # Add overhead for machine switches
        duration += len(used_machines) * 10

        # Add estimated tool change overhead if machine specs provided
        if machine_specs:
            for machine, tools in required_tools.items():
                if machine < len(machine_specs) and machine_specs[machine].tool_matrix is not None:
                    # Estimate tool change time based on matrix
                    tool_matrix = machine_specs[machine].tool_matrix
                    avg_change_time = np.mean(tool_matrix[tool_matrix > 0])
                    duration += len(tools) * avg_change_time

        return duration

    @staticmethod
    def calculate_optimal_makespan(jobs: List[Job], 
                                 job_dependencies: List[List[int]], 
                                 num_machines: int,
                                 machine_specs: Optional[List[MachineSpec]] = None) -> int:
        """Calculate the optimal makespan considering dependencies, machine conflicts, and tool changes."""
        # Identify dependency chains
        chains = ManualJobShopGenerator.identify_dependency_chains(jobs, job_dependencies)

        # Calculate base workload
        total_workload = 0
        machine_workload = [0] * num_machines
        tool_workload = defaultdict(int)  # tool -> total usage time

        for job in jobs:
            for op in job.operations:
                total_workload += op.duration
                machine_workload[op.machine] += op.duration
                for tool in op.required_tools:
                    tool_workload[tool] += op.duration

        # Calculate chain durations with conflicts
        max_chain_duration = 0
        for chain in chains:
            chain_duration = ManualJobShopGenerator.calculate_chain_duration(jobs, chain, machine_specs)
            max_chain_duration = max(max_chain_duration, chain_duration)

            # Consider conflicts with other chains
            for other_chain in chains:
                if chain != other_chain:
                    conflict_score = ManualJobShopGenerator.calculate_chain_conflict(
                        jobs, chain, other_chain)
                    if conflict_score > 0:
                        chain_duration += conflict_score * 15

            max_chain_duration = max(max_chain_duration, chain_duration)

        # Get maximum machine workload
        max_machine_workload = max(machine_workload)
        avg_workload = total_workload / num_machines

        # Consider tool change overhead
        tool_change_overhead = 0
        if machine_specs:
            for spec in machine_specs:
                if spec.tool_matrix is not None:
                    # Estimate average tool change time
                    avg_change_time = np.mean(spec.tool_matrix[spec.tool_matrix > 0])
                    tool_change_overhead += avg_change_time * len(tool_workload)

        # Calculate lower bound considering all factors
        lower_bound = max(
            max_chain_duration,
            max_machine_workload + tool_change_overhead,
            int(np.ceil(avg_workload + tool_change_overhead))
        )

        logger.info(f"Max chain duration (with conflicts): {max_chain_duration}")
        logger.info(f"Max machine workload: {max_machine_workload}")
        logger.info(f"Average workload per machine: {avg_workload:.2f}")
        if tool_change_overhead > 0:
            logger.info(f"Estimated tool change overhead: {tool_change_overhead}")

        return lower_bound

    @classmethod
    def generate_from_data(cls, job_shop_data: Dict) -> Tuple[List[Job], List[MachineSpec], int]:
        """Generate job shop problem from JSON data structure."""
        logger.info("Starting to generate job shop from data structure")

        num_jobs = job_shop_data["metadata"]["numJobs"]
        num_machines = job_shop_data["metadata"]["numMachines"]
        num_tools = job_shop_data["metadata"].get("numTools", 0)

        logger.info(f"Data loaded: {num_jobs} jobs, {num_machines} machines, {num_tools} tools")

        # Parse machine specifications
        machine_specs = []
        if "machine_specs" in job_shop_data:
            for spec_data in job_shop_data["machine_specs"]:
                tooling = spec_data.get("tooling", {})
                spec = MachineSpec(
                    max_slots=tooling.get("max_slots", -1),
                    tool_matrix=np.array(tooling["tool_matrix"]) if "tool_matrix" in tooling else None,
                    compatible_tools=set(tooling.get("compatible_tools", []))
                )
                machine_specs.append(spec)

        # Create jobs with tool requirements
        job_dependencies = [[] for _ in range(num_jobs)]
        jobs = [Job() for _ in range(num_jobs)]
        
        for job_data in job_shop_data["jobs"]:
            job_id = job_data["id"]
            job = jobs[job_id]
            job_dependencies[job_id] = job_data.get("dependencies", [])

            # Create operation with tool requirements
            op = Operation(
                duration=job_data["duration"],
                machine=job_data["machine"],
                required_tools=set(job_data.get("required_tools", []))
            )
            op.eligible_machines = {op.machine}

            # Add operation dependencies
            if "operationDependencies" in job_data:
                for op_dep in job_data["operationDependencies"]:
                    for dep_job_id in op_dep["dependencies"]:
                        op.dependent_operations.append((dep_job_id, 0))

            job.operations = [op]
            job.dependent_jobs = job_dependencies[job_id]

        # Check for circular dependencies
        if cls.detect_circular_dependencies(job_dependencies):
            logger.error("Circular dependencies detected in job dependencies")
            raise ValueError("Circular dependencies detected")

        # Validate and compute makespan
        cls.validate_job_shop(jobs, num_machines, machine_specs)
        optimal_makespan = cls.calculate_optimal_makespan(
            jobs, job_dependencies, num_machines, machine_specs)

        logger.info(f"Job shop generation complete with {len(jobs)} jobs")
        logger.info(f"Optimal makespan: {optimal_makespan}")

        return jobs, machine_specs, optimal_makespan

    @classmethod
    def generate_from_file(cls, filename: str) -> Tuple[List[Job], List[MachineSpec], int]:
        """Generate job shop problem from JSON file."""
        logger.info(f"Starting to generate job shop from file: {filename}")

        with open(filename, 'r') as f:
            job_shop_data = json.load(f)

        return cls.generate_from_data(job_shop_data)

    @staticmethod
    def validate_job_shop(jobs: List[Job], num_machines: int, 
                         machine_specs: Optional[List[MachineSpec]] = None):
        """Validate the job shop configuration."""
        logger.info("Performing final job shop validation")

        machine_operations = defaultdict(set)
        machine_tools = defaultdict(set)

        for job_id, job in enumerate(jobs):
            for op_id, op in enumerate(job.operations):
                if op.machine >= num_machines:
                    logger.error(f"Job {job_id} Operation {op_id} assigned to invalid machine {op.machine}")
                    raise ValueError(f"Invalid machine assignment for Job {job_id} Operation {op_id}")

                machine_operations[op.machine].add(job_id)
                machine_tools[op.machine].update(op.required_tools)

                if op.duration <= 0:
                    logger.error(f"Job {job_id} Operation {op_id} has invalid duration {op.duration}")
                    raise ValueError(f"Invalid duration for Job {job_id} Operation {op_id}")

                # Validate tool requirements against machine specifications
                if machine_specs and op.machine < len(machine_specs):
                    spec = machine_specs[op.machine]
                    if spec.compatible_tools is not None:
                        invalid_tools = op.required_tools - spec.compatible_tools
                        if invalid_tools:
                            logger.error(f"Job {job_id} Operation {op_id} requires incompatible "
                                       f"tools {invalid_tools} for machine {op.machine}")
                            raise ValueError(f"Invalid tool requirements for Job {job_id} Operation {op_id}")

                for dep_job_id, dep_op_id in op.dependent_operations:
                    if dep_job_id >= len(jobs):
                        logger.error(f"Invalid operation dependency: Job {job_id} -> Job {dep_job_id}")
                        raise ValueError("Invalid operation dependency")

                    dep_job = jobs[dep_job_id]
                    if dep_op_id >= len(dep_job.operations):
                        logger.error(f"Invalid operation dependency: Job {job_id} Op {op_id} -> "
                                     f"Job {dep_job_id} Op {dep_op_id}")
                        raise ValueError("Invalid operation dependency")

        # Validate machine utilization and tool requirements
        for machine_id in range(num_machines):
            if not machine_operations[machine_id]:
                logger.warning(f"Machine {machine_id} has no assigned operations")
            else:
                logger.debug(f"Machine {machine_id} is used by {len(machine_operations[machine_id])} jobs")
                
            if machine_specs and machine_id < len(machine_specs):
                spec = machine_specs[machine_id]
                if spec.max_slots != -1:
                    max_concurrent_tools = max(
                        len(op.required_tools)
                        for job in jobs
                        for op in job.operations
                        if op.machine == machine_id
                    )
                    if max_concurrent_tools > spec.max_slots:
                        logger.error(f"Machine {machine_id} requires {max_concurrent_tools} concurrent tools "
                                   f"but only has {spec.max_slots} slots")
                        raise ValueError(f"Insufficient tool slots on machine {machine_id}")

        logger.info("Job shop validation complete - configuration is valid")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example job shop data
    example_data = {
        "metadata": {
            "numJobs": 5,
            "numMachines": 3,
            "numTools": 4
        },
        "machine_specs": [
            {
                "tooling": {
                    "max_slots": 3,
                    "compatible_tools": [1, 2, 3],
                    "tool_matrix": [
                        [0, 10, 15, 20],  # From empty
                        [10, 0, 12, 18],  # From tool 1
                        [15, 12, 0, 15],  # From tool 2
                        [20, 18, 15, 0]   # From tool 3
                    ]
                }
            },
            {
                "tooling": {
                    "max_slots": 2,
                    "compatible_tools": [2, 3, 4],
                    "tool_matrix": [
                        [0, 0, 12, 15],   # From empty
                        [0, 0, 0, 0],     # From tool 1 (not compatible)
                        [12, 0, 0, 12],   # From tool 2
                        [15, 0, 12, 0]    # From tool 3
                    ]
                }
            },
            {
                "tooling": {
                    "max_slots": 4,
                    "compatible_tools": [1, 2, 3, 4],
                    "tool_matrix": [
                        [0, 10, 15, 20, 25],  # From empty
                        [10, 0, 12, 18, 22],  # From tool 1
                        [15, 12, 0, 15, 20],  # From tool 2
                        [20, 18, 15, 0, 12],  # From tool 3
                        [25, 22, 20, 12, 0]   # From tool 4
                    ]
                }
            }
        ],
        "jobs": [
            {
                "id": 0,
                "duration": 20,
                "machine": 0,
                "required_tools": [1, 2],
                "dependencies": []
            },
            {
                "id": 1,
                "duration": 25,
                "machine": 1,
                "required_tools": [2, 3],
                "dependencies": [0]
            },
            {
                "id": 2,
                "duration": 15,
                "machine": 2,
                "required_tools": [1, 3],
                "dependencies": [0]
            },
            {
                "id": 3,
                "duration": 30,
                "machine": 0,
                "required_tools": [2, 3],
                "dependencies": [1]
            },
            {
                "id": 4,
                "duration": 20,
                "machine": 2,
                "required_tools": [3, 4],
                "dependencies": [2]
            }
        ]
    }

    # Generate job shop from data
    try:
        jobs, machine_specs, makespan = ManualJobShopGenerator.generate_from_data(example_data)
        print(f"\nGenerated job shop problem with estimated makespan: {makespan}")
        
        print("\nJobs Configuration:")
        for i, job in enumerate(jobs):
            print(f"\nJob {i}:")
            print(f"Dependencies: {job.dependent_jobs}")
            for j, op in enumerate(job.operations):
                print(f"  Operation {j}:")
                print(f"    Duration: {op.duration}")
                print(f"    Machine: {op.machine}")
                print(f"    Required Tools: {op.required_tools}")

        print("\nMachine Specifications:")
        for i, spec in enumerate(machine_specs):
            print(f"\nMachine {i}:")
            print(f"  Max Tool Slots: {spec.max_slots}")
            print(f"  Compatible Tools: {spec.compatible_tools}")
            if spec.tool_matrix is not None:
                print(f"  Tool Change Matrix Shape: {spec.tool_matrix.shape}")

    except Exception as e:
        print(f"Error generating job shop: {e}")