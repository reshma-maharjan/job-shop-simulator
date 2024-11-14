from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
# you need pip install dot
# apt install graphviz


@dataclass
class Operation:
    job: int
    op: int
    start: int
    duration: int
    end: int


class JobSchedulingProblem:
    def __init__(self):
        self.num_jobs = 0
        self.num_machines = 0
        self.dependencies = {}
        self.durations = []
        self.operation_dependencies = {}

    def parse_file_content(self, content: str):
        """Parse the problem definition from file content."""
        sections = {
            'METADATA': '',
            'JOB DEPENDENCIES': '',
            'OPERATION DURATIONS': '',
            'OPERATION DEPENDENCIES': ''
        }

        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check for section starts and ends
            for section in sections.keys():
                if line == f'[{section}]':
                    current_section = section
                    continue
                elif line == f'[{section} END]':
                    current_section = None
                    continue

            if current_section:
                sections[current_section] += line + '\n'

        self._parse_metadata(sections['METADATA'])
        self._parse_job_dependencies(sections['JOB DEPENDENCIES'])
        self._parse_durations(sections['OPERATION DURATIONS'])
        self._parse_operation_dependencies(sections['OPERATION DEPENDENCIES'])

    def _parse_metadata(self, metadata: str):
        """Parse metadata section."""
        for item in metadata.split():
            if ':' in item:
                key, value = item.split(':')
                if key == 'num_jobs':
                    self.num_jobs = int(value)
                elif key == 'num_machines':
                    self.num_machines = int(value)

    def _parse_job_dependencies(self, deps: str):
        """Parse job dependencies section."""
        for line in deps.strip().split('\n'):
            if not line or line.startswith('#'):
                continue

            parts = line.split(':', 1)
            if len(parts) == 2:
                try:
                    job = int(parts[0])
                    deps_str = parts[1].strip()
                    if deps_str:
                        self.dependencies[job] = [int(x.strip()) for x in deps_str.split(',')]
                    else:
                        self.dependencies[job] = []
                except ValueError:
                    continue

    def _parse_durations(self, durations: str):
        """Parse operation durations section."""
        numbers = []
        for line in durations.strip().split('\n'):
            numbers.extend([int(x) for x in line.split() if x.isdigit()])
        self.durations = numbers

    def _parse_operation_dependencies(self, op_deps: str):
        """Parse operation dependencies section."""
        for line in op_deps.strip().split('\n'):
            if line.startswith('#') or not line:
                continue

            parts = line.split(':', 2)
            if len(parts) >= 2:
                try:
                    op_index = int(parts[0])
                    deps = [int(x.strip()) for x in parts[-1].split(',') if x.strip()]
                    self.operation_dependencies[op_index] = deps
                except ValueError:
                    continue

    def validate_sequence(self, sequence: list) -> tuple[bool, str]:
        """Validate a given job sequence."""
        if len(sequence) != self.num_jobs:
            return False, f"Sequence length ({len(sequence)}) does not match number of jobs ({self.num_jobs})"

        if len(set(sequence)) != len(sequence):
            return False, "Sequence contains duplicate jobs"

        if not all(0 <= job < self.num_jobs for job in sequence):
            return False, "Sequence contains invalid job numbers"

        completed = set()
        for job in sequence:
            if job in self.dependencies:
                for dep_job in self.dependencies[job]:
                    if dep_job not in completed:
                        return False, f"Job {job} scheduled before its dependency {dep_job}"
            completed.add(job)

        return True, "Sequence is valid"

    def calculate_makespan(self, sequence: list) -> int:
        """Calculate the makespan for a given sequence."""
        job_completion_times = {i: 0 for i in range(self.num_jobs)}

        for job in sequence:
            start_time = 0
            if job in self.dependencies:
                for dep_job in self.dependencies[job]:
                    start_time = max(start_time, job_completion_times[dep_job])

            job_completion_times[job] = start_time + self.durations[job]

        return max(job_completion_times.values())


class ScheduleParser:
    def __init__(self):
        self.machine_schedules = {}
        self.total_time = 0
        self.best_makespan = 0

    def parse_machine_schedule(self, text: str):
        """Parse the machine schedule from text."""
        machine_parts = text.split("Total time:")
        schedule_text = machine_parts[0]
        metadata = machine_parts[1] if len(machine_parts) > 1 else ""

        if "Best makespan achieved:" in metadata:
            self.best_makespan = int(metadata.split("Best makespan achieved:")[1].strip())
        if len(machine_parts) > 1:
            self.total_time = int(metadata.split("Best makespan achieved:")[0].strip())

        for line in schedule_text.split('\n'):
            if not line.strip():
                continue

            if line.startswith("Machine"):
                current_machine = int(line.split(':')[0].split()[-1])
                self.machine_schedules[current_machine] = []

                operations = re.findall(r'\(Job (\d+), Op (\d+), Start: (\d+), Duration: (\d+)\)', line)

                for job, op, start, duration in operations:
                    operation = Operation(
                        job=int(job),
                        op=int(op),
                        start=int(start),
                        duration=int(duration),
                        end=int(start) + int(duration)
                    )
                    self.machine_schedules[current_machine].append(operation)

    def get_job_sequence(self) -> List[int]:
        """Extract the job sequence from the schedule."""
        all_jobs = set()
        sequence = []

        max_time = max(op.end for machine in self.machine_schedules.values() for op in machine)

        for t in range(max_time + 1):
            for machine_id in sorted(self.machine_schedules.keys()):
                for op in self.machine_schedules[machine_id]:
                    if op.start == t and op.job not in all_jobs:
                        sequence.append(op.job)
                        all_jobs.add(op.job)

        return sequence

    def validate_schedule(self) -> List[str]:
        """Validate the schedule for conflicts."""
        issues = []

        # Check for machine overlaps
        for machine_id, operations in self.machine_schedules.items():
            sorted_ops = sorted(operations, key=lambda x: x.start)
            for i in range(len(sorted_ops) - 1):
                if sorted_ops[i].end > sorted_ops[i + 1].start:
                    issues.append(
                        f"Machine {machine_id}: Overlap between Job {sorted_ops[i].job} and Job {sorted_ops[i + 1].job}")

        # Check for simultaneous job execution
        all_ops = [(op.job, op.start, op.end, machine_id)
                   for machine_id, ops in self.machine_schedules.items()
                   for op in ops]

        for i, (job1, start1, end1, m1) in enumerate(all_ops):
            for job2, start2, end2, m2 in all_ops[i + 1:]:
                if job1 == job2 and m1 != m2:
                    if not (end1 <= start2 or end2 <= start1):
                        issues.append(f"Job {job1} scheduled simultaneously on machines {m1} and {m2}")

        return issues


def read_problem_from_string(content: str) -> JobSchedulingProblem:
    """Create a JobSchedulingProblem instance from a string."""
    problem = JobSchedulingProblem()
    problem.parse_file_content(content)
    return problem


def validate_solution(problem_content: str, sequence: list) -> tuple[bool, str, int]:
    """Validate a solution sequence for a given problem."""
    problem = read_problem_from_string(problem_content)
    is_valid, message = problem.validate_sequence(sequence)
    makespan = problem.calculate_makespan(sequence) if is_valid else -1
    return is_valid, message, makespan


def parse_solution_string(solution_str: str) -> list:
    """Parse a solution string into a list of job numbers."""
    jobs = solution_str.split(',')
    return [int(job.strip().split('Job')[1]) for job in jobs]


def find_missing_jobs(sequence: list, num_jobs: int):
    """Find missing jobs in a sequence."""
    expected_jobs = set(range(num_jobs))
    actual_jobs = set(sequence)
    missing_jobs = expected_jobs - actual_jobs
    return sorted(list(missing_jobs))


def create_directed_dependency_graph(problem_content: str):
    # Parse problem
    problem = read_problem_from_string(problem_content)

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for i in range(problem.num_jobs):
        G.add_node(i, duration=problem.durations[i])

    for job, deps in problem.dependencies.items():
        for dep in deps:
            G.add_edge(dep, job)

    # Set up the plot
    plt.figure(figsize=(30, 20), facecolor='white')

    # Use hierarchical layout
    pos = nx.nx_pydot.pydot_layout(G, prog='dot', root=0)

    # Convert positions to numpy arrays for manipulation
    pos = {node: np.array([x, y]) for node, (x, y) in pos.items()}

    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        if in_degree == 0:
            node_colors.append('#90EE90')  # Start nodes (green)
            node_sizes.append(2000)
        elif out_degree == 0:
            node_colors.append('#FFB6C1')  # End nodes (pink)
            node_sizes.append(2000)
        elif in_degree + out_degree > 3:
            node_colors.append('#FFA500')  # Critical nodes (orange)
            node_sizes.append(2500)
        else:
            node_colors.append('#ADD8E6')  # Standard nodes (light blue)
            node_sizes.append(1500)

    # Draw edges with enhanced arrows
    edge_colors = ['#404040' for _ in G.edges()]

    # Draw edges with larger arrows and curves
    nx.draw_networkx_edges(G, pos,
                           edge_color=edge_colors,
                           width=1.5,
                           arrowsize=25,
                           alpha=0.7,
                           connectionstyle="arc3,rad=0.2",
                           arrowstyle='->',
                           min_source_margin=25,
                           min_target_margin=25)

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos,
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   edgecolors='black',
                                   linewidths=2)

    # Add labels with white background
    labels = {node: f'Job {node}\n({G.nodes[node]["duration"]})' for node in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels,
                            font_size=10,
                            font_weight='bold',
                            bbox=dict(facecolor='white',
                                      edgecolor='black',
                                      alpha=0.8,
                                      pad=0.5,
                                      boxstyle='round'))

    # Add legend with directional information
    legend_elements = [
        mpatches.Patch(color='#90EE90', label='Start Jobs (No Dependencies)'),
        mpatches.Patch(color='#FFB6C1', label='End Jobs (No Dependents)'),
        mpatches.Patch(color='#FFA500', label='Critical Jobs (Many Connections)'),
        mpatches.Patch(color='#ADD8E6', label='Standard Jobs'),
        plt.Line2D([0], [0], color='#404040', marker='>', markersize=15,
                   label='Dependency Direction', linestyle='-', linewidth=2)
    ]

    plt.legend(handles=legend_elements,
               title='Graph Legend\nArrows show dependencies: Job A → Job B means B depends on A',
               loc='upper left',
               bbox_to_anchor=(1, 1),
               fontsize=12,
               title_fontsize=14)

    # Add title with explanation
    plt.title(
        'Job Dependency Graph\nDirected arrows show dependency flow: A → B means Job B depends on Job A\n(Numbers in parentheses show job duration)',
        pad=20, fontsize=16, fontweight='bold')

    # Set margins and layout
    plt.margins(0.2)
    plt.tight_layout()

    # Save with high resolution
    plt.savefig('job_dependencies_directed.png',
                format='png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

    # Print analysis focusing on dependency chains
    print("Dependency Chain Analysis:")
    print("-------------------------")

    # Find and print all paths from start nodes to end nodes
    start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    print("\nStart Nodes (Entry Points):", [f"Job {n}" for n in start_nodes])
    print("End Nodes (Exit Points):", [f"Job {n}" for n in end_nodes])

    print("\nLongest Dependency Chains:")
    for start in start_nodes:
        for end in end_nodes:
            try:
                paths = list(nx.all_simple_paths(G, start, end))
                if paths:
                    longest_path = max(paths, key=len)
                    total_duration = sum(problem.durations[node] for node in longest_path)
                    print(f"\nChain from Job {start} to Job {end}:")
                    print(" → ".join(f"Job {node}" for node in longest_path))
                    print(f"Total duration: {total_duration}")
            except nx.NetworkXNoPath:
                continue

    # Calculate and print critical path
    try:
        critical_path = nx.dag_longest_path(G, weight=lambda u, v, d: problem.durations[v])
        print("\nCritical Path:")
        print(" → ".join(f"Job {node}" for node in critical_path))
        critical_path_duration = sum(problem.durations[node] for node in critical_path)
        print(f"Critical Path Duration: {critical_path_duration}")
    except:
        print("\nCould not determine critical path")


if __name__ == "__main__":
    # Read problem file
    with open('doris.csv', 'r') as f:
        problem_content = f.read()

    create_directed_dependency_graph(problem_content)
    print("\nGraph has been saved as 'job_dependencies.png'")

    # Read and parse machine schedule
    schedule_text = """Machine 0: (Job 0, Op 0, Start: 0, Duration: 49) (Job 6, Op 0, Start: 73, Duration: 10) (Job 12, Op 0, Start: 145, Duration: 60) (Job 18, Op 0, Start: 253, Duration: 24) 
Machine 1: (Job 1, Op 0, Start: 49, Duration: 64) (Job 13, Op 0, Start: 145, Duration: 0) (Job 7, Op 0, Start: 283, Duration: 0) (Job 31, Op 0, Start: 283, Duration: 70) 
Machine 2: (Job 14, Op 0, Start: 145, Duration: 0) (Job 2, Op 0, Start: 145, Duration: 36) (Job 8, Op 0, Start: 283, Duration: 24) (Job 32, Op 0, Start: 307, Duration: 10) (Job 26, Op 0, Start: 353, Duration: 72) 
Machine 3: (Job 3, Op 0, Start: 73, Duration: 12) (Job 33, Op 0, Start: 283, Duration: 10) 
Machine 4: (Job 10, Op 0, Start: 49, Duration: 24) (Job 4, Op 0, Start: 73, Duration: 210) 
Machine 5: (Job 5, Op 0, Start: 73, Duration: 24) (Job 11, Op 0, Start: 97, Duration: 48) (Job 17, Op 0, Start: 205, Duration: 48) 
Total time: 425
Best makespan achieved: 425"""

    # Parse and analyze schedule
    parser = ScheduleParser()
    parser.parse_machine_schedule(schedule_text)

    print("\nSchedule Analysis:")
    print("-----------------")

    # Validate schedule for conflicts
    issues = parser.validate_schedule()
    if issues:
        print("\nSchedule Issues Found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo scheduling conflicts found.")

    # Extract and display job sequence
    sequence = parser.get_job_sequence()
    print(f"\nExtracted Job Sequence:")
    job_sequence_str = ", ".join(f"Job {job}" for job in sequence)
    print(job_sequence_str)
    missing = find_missing_jobs(sequence, 36)

    print("Missing jobs:", missing)
    print("\nPresent jobs:", sorted(sequence))
    print("\nSequence length:", len(sequence))
    print("Expected jobs: 36")
    print("Number of missing jobs:", len(missing))

    # Validate against problem constraints
    is_valid, message, makespan = validate_solution(problem_content, sequence)
    print(f"\nProblem Constraint Validation:")
    print(f"Valid: {is_valid}")
    print(f"Message: {message}")
    if is_valid:
        print(f"Calculated Makespan: {makespan}")
        print(f"Reported Makespan: {parser.best_makespan}")
        if makespan != parser.best_makespan:
            print(f"Warning: Calculated makespan differs from reported makespan!")

    # Display schedule statistics
    print(f"\nSchedule Statistics:")
    print(f"Total Scheduled Time: {parser.total_time}")
    print(f"Number of Machines Used: {len(parser.machine_schedules)}")
    total_ops = sum(len(ops) for ops in parser.machine_schedules.values())
    print(f"Total Operations Scheduled: {total_ops}")