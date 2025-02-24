import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
import json
import logging
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_MACHINES = 100

@dataclass
class Operation:
    duration: int
    machine: int
    eligible_machines: set = field(default_factory=set)
    dependent_operations: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.eligible_machines:
            self.eligible_machines = {self.machine}
        if not self.dependent_operations:
            self.dependent_operations = []

@dataclass
class Job:
    operations: List[Operation] = field(default_factory=list)
    dependent_jobs: List[int] = field(default_factory=list)

@dataclass
class Action:
    job: int
    machine: int
    operation: int

@dataclass
class State:
    job_progress: np.ndarray  # 2D array [num_jobs, max_operations]
    machine_availability: np.ndarray  # 1D array [num_machines]
    next_operation_for_job: np.ndarray  # 1D array [num_jobs]
    completed_jobs: np.ndarray  # 1D array [num_jobs]
    job_start_times: np.ndarray  # 1D array [num_jobs]

    @classmethod
    def create(cls, num_jobs: int, num_machines: int, max_operations: int):
        return cls(
            job_progress=np.zeros((num_jobs, max_operations), dtype=np.int32),
            machine_availability=np.zeros(num_machines, dtype=np.int32),
            next_operation_for_job=np.zeros(num_jobs, dtype=np.int32),
            completed_jobs=np.zeros(num_jobs, dtype=bool),
            job_start_times=np.full(num_jobs, -1, dtype=np.int32)
        )

@dataclass
class ScheduleEntry:
    job: int
    operation: int
    machine: int
    start: int
    duration: int

class Dependencies:
    """Class to manage dependency tracking structures."""
    def __init__(self, num_jobs: int):
        # Job dependency tracking
        self.job_masks = np.ones((num_jobs, num_jobs), dtype=bool)
        self.pending_job_deps = np.zeros(num_jobs, dtype=np.int32)
        self.reverse_job_deps = defaultdict(set)

        # Operation dependency tracking
        self.op_masks = {}
        self.pending_op_deps = {}
        self.reverse_op_deps = defaultdict(set)

        # Active entity tracking
        self.active_jobs = set(range(num_jobs))
        self.active_operations = {}

    def add_job_dependency(self, job_id: int, dep_job_id: int):
        """Add a job dependency."""
        self.job_masks[job_id][dep_job_id] = False
        self.pending_job_deps[job_id] += 1
        self.reverse_job_deps[dep_job_id].add(job_id)

    def add_operation_dependency(self, job_id: int, op_id: int,
                                 dep_job_id: int, dep_op_id: int):
        """Add an operation dependency."""
        key = (job_id, op_id)
        dep_key = (dep_job_id, dep_op_id)

        if key not in self.op_masks:
            self.op_masks[key] = {}
            self.pending_op_deps[key] = 0

        self.op_masks[key][dep_key] = False
        self.pending_op_deps[key] += 1
        self.reverse_op_deps[dep_key].add(key)

    def satisfy_job_dependency(self, job_id: int):
        """Mark a job's dependencies as satisfied."""
        for dep_job_id in self.reverse_job_deps[job_id]:
            if not self.job_masks[dep_job_id][job_id]:
                self.job_masks[dep_job_id][job_id] = True
                self.pending_job_deps[dep_job_id] -= 1

    def satisfy_operation_dependency(self, job_id: int, op_id: int):
        """Mark an operation's dependencies as satisfied."""
        key = (job_id, op_id)
        for dep_key in self.reverse_op_deps[key]:
            if key in self.op_masks[dep_key] and not self.op_masks[dep_key][key]:
                self.op_masks[dep_key][key] = True
                self.pending_op_deps[dep_key] -= 1

    def is_job_ready(self, job_id: int) -> bool:
        """Check if a job is ready (all dependencies satisfied)."""
        return self.pending_job_deps[job_id] == 0

    def is_operation_ready(self, job_id: int, op_id: int) -> bool:
        """Check if an operation is ready (all dependencies satisfied)."""
        key = (job_id, op_id)
        return key not in self.pending_op_deps or self.pending_op_deps[key] == 0

    def remove_completed_job(self, job_id: int):
        """Remove a completed job from active tracking."""
        self.active_jobs.remove(job_id)
        if job_id in self.active_operations:
            del self.active_operations[job_id]


class JobShopEnvironment:
    def __init__(self, jobs: List[Job]):
        if not jobs:
            raise ValueError("Jobs list cannot be empty")

        self.jobs = jobs
        self.num_machines = 0
        self.total_time = 0

        # Find number of machines
        for job in jobs:
            for op in job.operations:
                self.num_machines = max(self.num_machines, op.machine + 1)

        if self.num_machines == 0 or self.num_machines > MAX_MACHINES:
            raise ValueError("Invalid number of machines")

        # Find max operations
        max_operations = max(len(job.operations) for job in jobs)

        # Initialize state
        self.current_state = State.create(
            num_jobs=len(jobs),
            num_machines=self.num_machines,
            max_operations=max_operations
        )

        # Initialize tracking structures
        self.dependencies = Dependencies(len(jobs))
        self._initialize_dependencies()
        self.action_history = []
        self.action_lookup = self._precompute_actions()

        # Initialize machine mappings
        self.machine_to_jobs = self._create_machine_job_mapping()
        self.operation_to_machine = self._create_operation_machine_mapping()

        # Ready state tracking
        self.ready_jobs = np.zeros(len(jobs), dtype=bool)
        self.ready_operations = defaultdict(set)

        # Initial update
        self._update_ready_states()

    def _initialize_dependencies(self):
        """Initialize all dependencies from jobs."""
        for job_id, job in enumerate(self.jobs):
            # Add job dependencies
            for dep_job_id in job.dependent_jobs:
                self.dependencies.add_job_dependency(job_id, dep_job_id)

            # Add operation dependencies
            for op_id, op in enumerate(job.operations):
                for dep_job_id, dep_op_id in op.dependent_operations:
                    self.dependencies.add_operation_dependency(
                        job_id, op_id, dep_job_id, dep_op_id)

            # Initialize active operations
            self.dependencies.active_operations[job_id] = set(range(len(job.operations)))

    def _create_machine_job_mapping(self) -> Dict[int, Set[int]]:
        """Create mapping from machines to jobs that use them."""
        mapping = defaultdict(set)
        for job_id, job in enumerate(self.jobs):
            for op in job.operations:
                mapping[op.machine].add(job_id)
        return dict(mapping)

    def _create_operation_machine_mapping(self) -> Dict[Tuple[int, int], int]:
        """Create mapping from (job, operation) to machine."""
        mapping = {}
        for job_id, job in enumerate(self.jobs):
            for op_id, op in enumerate(job.operations):
                mapping[(job_id, op_id)] = op.machine
        return mapping

    def _precompute_actions(self) -> List[List[List[Action]]]:
        """Precompute all possible actions."""
        actions = []
        for job_id, job in enumerate(self.jobs):
            job_actions = []
            for op_id, op in enumerate(job.operations):
                op_actions = [
                    Action(job_id, m, op_id)
                    for m in range(self.num_machines)
                    if m in op.eligible_machines
                ]
                job_actions.append(op_actions)
            actions.append(job_actions)
        return actions

    def _update_ready_states(self):
        """Update ready states using dependency tracking."""
        # Update ready jobs
        for job_id in self.dependencies.active_jobs:
            if not self.current_state.completed_jobs[job_id]:
                self.ready_jobs[job_id] = self.dependencies.is_job_ready(job_id)
            else:
                self.ready_jobs[job_id] = False

        # Update ready operations
        self.ready_operations.clear()
        for job_id in self.dependencies.active_jobs:
            if not self.ready_jobs[job_id]:
                continue

            current_op = self.current_state.next_operation_for_job[job_id]
            if current_op >= len(self.jobs[job_id].operations):
                continue

            if self.dependencies.is_operation_ready(job_id, current_op):
                self.ready_operations[job_id].add(current_op)

    def get_possible_actions(self) -> List[Action]:   # get all possible actions in current state 
        """Get all possible actions in current state."""
        possible_actions = []

        for job_id in self.dependencies.active_jobs:
            if not self.ready_jobs[job_id]:
                continue

            for op_id in self.ready_operations[job_id]:
                if self.current_state.job_progress[job_id, op_id] == 0:
                    possible_actions.extend(self.action_lookup[job_id][op_id])

        return possible_actions
    
    def step(self, action: Action) -> State:
        """Execute an action and update the environment state."""
        # Verify action validity before execution
        if action.job not in self.dependencies.active_jobs:
            raise ValueError(f"Invalid action: Job {action.job} is not active")

        current_op = self.current_state.next_operation_for_job[action.job]
        if action.operation != current_op:
            raise ValueError(f"Invalid action: Expected operation {current_op} for job {action.job}, got {action.operation}")

        op = self.jobs[action.job].operations[action.operation]
        if action.machine not in op.eligible_machines:
            raise ValueError(f"Invalid action: Machine {action.machine} not eligible for job {action.job} operation {action.operation}")

        # Calculate start time considering all constraints
        start_time = self.current_state.machine_availability[action.machine]

        # Consider job dependencies
        for dep_job in self.jobs[action.job].dependent_jobs:
            dep_job_last_op = len(self.jobs[dep_job].operations) - 1
            dep_end_time = self.current_state.job_progress[dep_job, dep_job_last_op]
            if dep_end_time == 0:  # Dependency not completed
                raise ValueError(f"Invalid action: Dependent job {dep_job} not completed")
            start_time = max(start_time, dep_end_time)

        # Check previous operation
        if action.operation > 0:
            prev_end_time = self.current_state.job_progress[action.job, action.operation - 1]
            if prev_end_time == 0:  # Previous operation not completed
                raise ValueError(f"Invalid action: Previous operation {action.operation - 1} not completed")
            start_time = max(start_time, prev_end_time)

        # Consider operation dependencies
        for dep_job, dep_op in op.dependent_operations:
            dep_end_time = self.current_state.job_progress[dep_job, dep_op]
            if dep_end_time == 0:  # Dependency not completed
                raise ValueError(f"Invalid action: Dependent operation ({dep_job}, {dep_op}) not completed")
            start_time = max(start_time, dep_end_time)

        # Update state
        end_time = start_time + op.duration

        # Record job start time if this is the first operation
        if self.current_state.job_start_times[action.job] == -1:
            self.current_state.job_start_times[action.job] = start_time

        # Update progress and machine availability
        self.current_state.job_progress[action.job, action.operation] = end_time
        self.current_state.machine_availability[action.machine] = end_time

        # Update operation completion
        if self.current_state.next_operation_for_job[action.job] == action.operation:
            self.current_state.next_operation_for_job[action.job] += 1

            # Satisfy operation dependencies
            self.dependencies.satisfy_operation_dependency(action.job, action.operation)

            # Check job completion
            if self.current_state.next_operation_for_job[action.job] == len(self.jobs[action.job].operations):
                self.current_state.completed_jobs[action.job] = True
                self.dependencies.remove_completed_job(action.job)
                self.dependencies.satisfy_job_dependency(action.job)

        # Update total time and action history
        self.total_time = max(self.total_time, end_time)
        self.action_history.append(action)

        # Update ready states
        self._update_ready_states()

        # Verify state consistency if in debug mode
        if logger.getEffectiveLevel() <= logging.DEBUG:
            errors = self.verify_state()
            if errors:
                logger.error("Environment verification failed after step:\n" + "\n".join(errors))
                raise RuntimeError("Environment state verification failed")

        return self.current_state



    def reset(self):
        """Reset the environment to initial state."""
        # Reset state arrays
        self.current_state.job_progress.fill(0)
        self.current_state.machine_availability.fill(0)
        self.current_state.next_operation_for_job.fill(0)
        self.current_state.completed_jobs.fill(False)
        self.current_state.job_start_times.fill(-1)

        # Reset counters
        self.total_time = 0
        self.action_history.clear()

        # Reinitialize dependencies
        self.dependencies = Dependencies(len(self.jobs))
        self._initialize_dependencies()

        # Reset ready states
        self.ready_jobs.fill(False)
        self.ready_operations.clear()
        self._update_ready_states()

    def is_done(self) -> bool:
        """Check if all jobs are completed."""
        return not bool(self.dependencies.active_jobs)

    def get_schedule_data(self) -> List[List[ScheduleEntry]]:
        """Get the complete schedule data for visualization."""
        schedule_data = [[] for _ in range(self.num_machines)]

        for action in self.action_history:
            op = self.jobs[action.job].operations[action.operation]
            end = self.current_state.job_progress[action.job, action.operation]
            entry = ScheduleEntry(
                job=action.job,
                operation=action.operation,
                machine=action.machine,
                start=end - op.duration,
                duration=op.duration
            )
            schedule_data[action.machine].append(entry)

        return schedule_data

    def get_machine_utilization(self) -> List[float]:
        """Calculate machine utilization percentages."""
        total_time = max(1, self.total_time)  # Avoid division by zero
        machine_busy_time = np.zeros(self.num_machines)

        for action in self.action_history:
            op = self.jobs[action.job].operations[action.operation]
            machine_busy_time[action.machine] += op.duration

        return [busy_time / total_time for busy_time in machine_busy_time]





    def get_critical_path(self) -> List[Tuple[int, int]]:
        """Identify the critical path of operations."""
        critical_path = []
        end_times = {(j, o): self.current_state.job_progress[j, o]
                     for j in range(len(self.jobs))
                     for o in range(len(self.jobs[j].operations))}

        # Find last operation
        last_job, last_op = max(end_times.items(), key=lambda x: x[1])[0]
        current = (last_job, last_op)

        while current:
            critical_path.append(current)
            job_idx, op_idx = current

            # Find the operation that determined this operation's start time
            op = self.jobs[job_idx].operations[op_idx]
            end_time = self.current_state.job_progress[job_idx, op_idx]
            start_time = end_time - op.duration

            # Check previous operation in same job
            prev_op = None
            if op_idx > 0:
                prev_end = self.current_state.job_progress[job_idx, op_idx - 1]
                if prev_end == start_time:
                    prev_op = (job_idx, op_idx - 1)

            # Check dependent operations
            for dep_job, dep_op in op.dependent_operations:
                dep_end = self.current_state.job_progress[dep_job, dep_op]
                if dep_end == start_time:
                    prev_op = (dep_job, dep_op)

            current = prev_op

        return list(reversed(critical_path))

    def print_schedule(self, show_critical_path: bool = False):
        """Print the current schedule in a readable format."""
        schedule_data = self.get_schedule_data()
        critical_path = self.get_critical_path() if show_critical_path else []

        print("\nSchedule:")
        for i, machine_schedule in enumerate(schedule_data):
            util = self.get_machine_utilization()[i] * 100
            print(f"\nMachine {i} (Utilization: {util:.1f}%):")

            for entry in machine_schedule:
                is_critical = (entry.job, entry.operation) in critical_path
                critical_marker = "*" if is_critical else " "

                print(f"{critical_marker}[Job {entry.job}, Op {entry.operation}] "
                      f"Start: {entry.start:4d}, Duration: {entry.duration:3d}, "
                      f"End: {entry.start + entry.duration:4d}")

        print(f"\nTotal makespan: {self.total_time}")
        if show_critical_path:
            print("\nCritical path operations marked with *")

    def generate_gantt_data(self) -> dict:
        """Generate data for Gantt chart visualization."""
        schedule_data = self.get_schedule_data()
        critical_path = set(self.get_critical_path())

        gantt_data = {
            "machines": [],
            "jobs": [],
            "operations": [],
            "critical_path": list(critical_path)
        }

        for machine_id, machine_schedule in enumerate(schedule_data):
            machine_data = {
                "id": machine_id,
                "utilization": self.get_machine_utilization()[machine_id],
                "operations": []
            }

            for entry in machine_schedule:
                op_data = {
                    "job": entry.job,
                    "operation": entry.operation,
                    "start": entry.start,
                    "duration": entry.duration,
                    "is_critical": (entry.job, entry.operation) in critical_path
                }
                machine_data["operations"].append(op_data)

            gantt_data["machines"].append(machine_data)

        return gantt_data

    def get_state_features(self) -> np.ndarray:
        """Get a feature vector representing the current state (for ML algorithms)."""
        features = []

        # Job progress features
        job_progress_flat = self.current_state.job_progress.flatten()
        features.extend(job_progress_flat / max(1, np.max(job_progress_flat)))

        # Machine availability features
        machine_avail = self.current_state.machine_availability
        features.extend(machine_avail / max(1, np.max(machine_avail)))

        # Job completion status
        features.extend(self.current_state.completed_jobs)

        # Machine utilization
        features.extend(self.get_machine_utilization())

        # Dependency satisfaction
        features.extend(self.dependency_tracker.pending_job_deps /
                        max(1, np.max(self.dependency_tracker.pending_job_deps)))

        return np.array(features, dtype=np.float32)

    def get_performance_metrics(self) -> dict:
        """Calculate various performance metrics for the current schedule."""
        metrics = {
            "makespan": self.total_time,
            "machine_utilization": self.get_machine_utilization(),
            "avg_machine_utilization": np.mean(self.get_machine_utilization()),
            "num_operations": len(self.action_history),
            "critical_path_length": len(self.get_critical_path()),
            "completed_jobs": int(np.sum(self.current_state.completed_jobs))
        }

        # Calculate average waiting time
        waiting_times = []
        for action in self.action_history:
            op = self.jobs[action.job].operations[action.operation]
            end = self.current_state.job_progress[action.job, action.operation]
            waiting_time = end - op.duration
            waiting_times.append(waiting_time)

        metrics["avg_waiting_time"] = np.mean(waiting_times)
        metrics["max_waiting_time"] = np.max(waiting_times)

        return metrics

    def save_state(self, filename: str):
        """Save the current state and schedule to a file."""
        state_data = {
            "metadata": {
                "num_jobs": len(self.jobs),
                "num_machines": self.num_machines,
                "total_time": self.total_time
            },
            "schedule": {
                "actions": [(a.job, a.machine, a.operation) for a in self.action_history],
                "machine_schedules": self.get_schedule_data(),
                "critical_path": self.get_critical_path()
            },
            "metrics": self.get_performance_metrics()
        }

        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2, cls=NumpyEncoder)

    @classmethod
    def load_state(cls, filename: str, jobs: List[Job]) -> 'JobShopEnvironment':
        """Load a saved state into a new environment."""
        with open(filename, 'r') as f:
            state_data = json.load(f)

        env = cls(jobs)

        # Replay actions to reconstruct state
        for job, machine, operation in state_data["schedule"]["actions"]:
            action = Action(job, machine, operation)
            env.step(action)

        return env

    def get_incomplete_jobs(self) -> List[int]:
        """Get list of jobs that haven't been completed yet."""
        incomplete_jobs = []
        for job_id in range(len(self.jobs)):
            if not self.current_state.completed_jobs[job_id]:
                incomplete_jobs.append(job_id)
        return incomplete_jobs

    def is_job_executable(self, job_id: int) -> bool:
        """Check if a job can be executed in the current state."""
        if self.current_state.completed_jobs[job_id]:
            return False

        current_op = self.current_state.next_operation_for_job[job_id]
        if current_op >= len(self.jobs[job_id].operations):
            return False

        # Check dependencies
        return self.dependencies.is_job_ready(job_id) and \
            self.dependencies.is_operation_ready(job_id, current_op)

    def get_executable_jobs(self) -> List[int]:
        """Get list of jobs that can be executed in the current state."""
        return [job_id for job_id in range(len(self.jobs))
                if self.is_job_executable(job_id)]

    def generate_html_gantt(self, filename: str):
        """Generate an HTML file containing the Gantt chart visualization."""
        schedule_data = self.get_schedule_data()
        critical_path = set(self.get_critical_path())

        # Generate colors for jobs
        job_colors = [
            '#4299E1', '#48BB78', '#ECC94B', '#9F7AEA', '#ED64A6',
            '#667EEA', '#F56565', '#ED8936', '#38B2AC', '#4FD1C5'
        ]

        # Calculate timeline dimensions
        max_time = self.total_time
        time_scale = max(100, max_time + 20)  # Add padding

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Job Shop Schedule Gantt Chart</title>
            <style>
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }}
                .machine-row {{ display: flex; margin: 10px 0; align-items: center; }}
                .machine-label {{ width: 120px; flex-shrink: 0; }}
                .timeline {{ flex-grow: 1; height: 40px; background: #f0f0f0; position: relative; }}
                .operation {{ 
                    position: absolute; height: 100%; 
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-size: 12px; 
                }}
                .critical-path {{ border: 2px solid #E53E3E; }}
                .time-scale {{ display: flex; margin-left: 120px; }}
                .time-mark {{ flex-grow: 1; text-align: center; font-size: 12px; }}
                .legend {{ margin-top: 20px; display: flex; gap: 20px; flex-wrap: wrap; }}
                .legend-item {{ display: flex; align-items: center; gap: 8px; }}
                .legend-color {{ width: 20px; height: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Job Shop Schedule Gantt Chart</h2>
                
                <!-- Time scale -->
                <div class="time-scale">
                    {' '.join(f'<div class="time-mark">{t}</div>'
                              for t in range(0, time_scale + 1, max(1, time_scale // 10)))}
                </div>
                
                <!-- Machine rows -->
                {self._generate_machine_rows_html(schedule_data, critical_path, job_colors, time_scale)}
                
                <!-- Legend -->
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="border: 2px solid #E53E3E"></div>
                        <span>Critical Path</span>
                    </div>
                    {self._generate_legend_html(job_colors)}
                </div>
            </div>
        </body>
        </html>
        """

        with open(filename, 'w') as f:
            f.write(html_content)

        logger.info(f"Gantt chart saved to {filename}")

    def _generate_machine_rows_html(self, schedule_data, critical_path, job_colors, time_scale):
        """Generate HTML for machine rows in Gantt chart."""
        rows = []
        for machine_id, machine_schedule in enumerate(schedule_data):
            util = self.get_machine_utilization()[machine_id] * 100

            operations_html = []
            for entry in machine_schedule:
                is_critical = (entry.job, entry.operation) in critical_path
                critical_class = ' critical-path' if is_critical else ''
                color = job_colors[entry.job % len(job_colors)]

                left = (entry.start / time_scale) * 100
                width = (entry.duration / time_scale) * 100

                operations_html.append(
                    f'<div class="operation{critical_class}" '
                    f'style="left: {left}%; width: {width}%; background-color: {color}">'
                    f'J{entry.job}-{entry.operation}'
                    f'</div>'
                )

            rows.append(
                f'<div class="machine-row">'
                f'<div class="machine-label">Machine {machine_id}<br/>'
                f'<small>({util:.1f}% util)</small></div>'
                f'<div class="timeline">{"".join(operations_html)}</div>'
                f'</div>'
            )

        return '\n'.join(rows)

    def _generate_legend_html(self, job_colors):
        """Generate HTML for the job color legend."""
        legend_items = []
        unique_jobs = set(entry.job for machine in self.get_schedule_data()
                          for entry in machine)

        for job_id in sorted(unique_jobs):
            color = job_colors[job_id % len(job_colors)]
            legend_items.append(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background-color: {color}"></div>'
                f'<span>Job {job_id}</span>'
                f'</div>'
            )

        return '\n'.join(legend_items)


    def _verify_operation_sequence(self) -> List[str]:
        """Verify that operations execute in correct sequence."""
        errors = []
        for job_id, job in enumerate(self.jobs):
            for op_idx in range(1, len(job.operations)):
                # Get current and previous operation end times
                curr_end = self.current_state.job_progress[job_id, op_idx]
                prev_end = self.current_state.job_progress[job_id, op_idx - 1]

                # Skip if operations haven't been scheduled yet
                if curr_end == 0:
                    continue

                # Previous operation must finish before current starts
                curr_start = curr_end - job.operations[op_idx].duration
                if curr_start < prev_end:
                    errors.append(
                        f"Operation sequence violated for Job {job_id}: "
                        f"Op {op_idx} starts at {curr_start} before Op {op_idx-1} ends at {prev_end}"
                    )
        return errors

    def _verify_machine_capacity(self) -> List[str]:
        """Verify no machine overlap in the schedule."""
        errors = []
        for machine in range(self.num_machines):
            # Get all operations on this machine
            machine_ops = []
            for action in self.action_history:
                if action.machine == machine:
                    op = self.jobs[action.job].operations[action.operation]
                    end_time = self.current_state.job_progress[action.job, action.operation]
                    start_time = end_time - op.duration
                    machine_ops.append((start_time, end_time, action.job, action.operation))

            # Sort by start time
            machine_ops.sort()

            # Check for overlaps
            for i in range(1, len(machine_ops)):
                curr_start = machine_ops[i][0]
                prev_end = machine_ops[i-1][1]
                if curr_start < prev_end:
                    errors.append(
                        f"Machine capacity violated on Machine {machine}: "
                        f"Job {machine_ops[i][2]} Op {machine_ops[i][3]} starts at {curr_start} "
                        f"before Job {machine_ops[i-1][2]} Op {machine_ops[i-1][3]} ends at {prev_end}"
                    )
        return errors

    def _verify_dependencies(self) -> List[str]:
        """Verify all operation dependencies are respected."""
        errors = []
        for job_id, job in enumerate(self.jobs):
            for op_idx, op in enumerate(job.operations):
                end_time = self.current_state.job_progress[job_id, op_idx]
                # Skip if operation hasn't been scheduled yet
                if end_time == 0:
                    continue

                start_time = end_time - op.duration

                # Check operation dependencies
                for dep_job, dep_op in op.dependent_operations:
                    dep_end = self.current_state.job_progress[dep_job, dep_op]
                    if start_time < dep_end:
                        errors.append(
                            f"Dependency violated: Job {job_id} Op {op_idx} starts at {start_time} "
                            f"before dependent Job {dep_job} Op {dep_op} ends at {dep_end}"
                        )
        return errors

    def _verify_state_consistency(self) -> List[str]:
        """Verify the environment's state is consistent."""
        errors = []

        # Verify next_operation_for_job
        for job_id in range(len(self.jobs)):
            next_op = self.current_state.next_operation_for_job[job_id]
            if next_op > 0:
                # All previous operations should be completed
                for op_idx in range(next_op):
                    if self.current_state.job_progress[job_id, op_idx] == 0:
                        errors.append(
                            f"State inconsistency: Job {job_id} next_op is {next_op} "
                            f"but operation {op_idx} is not completed"
                        )

        # Verify completed_jobs
        for job_id in range(len(self.jobs)):
            if self.current_state.completed_jobs[job_id]:
                # All operations should be completed
                for op_idx in range(len(self.jobs[job_id].operations)):
                    if self.current_state.job_progress[job_id, op_idx] == 0:
                        errors.append(
                            f"State inconsistency: Job {job_id} marked complete "
                            f"but operation {op_idx} is not completed"
                        )

        # Verify machine_availability
        for machine in range(self.num_machines):
            last_end_time = 0
            for action in self.action_history:
                if action.machine == machine:
                    end_time = self.current_state.job_progress[action.job, action.operation]
                    if end_time > last_end_time:
                        last_end_time = end_time

            if self.current_state.machine_availability[machine] != last_end_time:
                errors.append(
                    f"State inconsistency: Machine {machine} availability is "
                    f"{self.current_state.machine_availability[machine]} but last operation "
                    f"ends at {last_end_time}"
                )

        return errors

    def verify_state(self) -> List[str]:
        """Run all state verifications."""
        errors = []
        errors.extend(self._verify_operation_sequence())
        errors.extend(self._verify_machine_capacity())
        errors.extend(self._verify_dependencies())
        errors.extend(self._verify_state_consistency())
        return errors


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)





# Example usage
if __name__ == "__main__":
    # Create sample jobs with dependencies
    job1 = Job(operations=[
        Operation(duration=3, machine=0),
        Operation(duration=2, machine=1)
    ])

    job2 = Job(operations=[
        Operation(duration=2, machine=1),
        Operation(duration=4, machine=0)
    ])
    job2.dependent_jobs = [0]  # Job 2 depends on Job 1

    # Create environment
    env = JobShopEnvironment([job1, job2])

    # Run schedule with performance tracking
    start_time = time.time()
    steps = 0

    while not env.is_done():
        possible_actions = env.get_possible_actions()
        if possible_actions:
            # Choose shortest duration action
            chosen_action = min(
                possible_actions,
                key=lambda a: env.jobs[a.job].operations[a.operation].duration
            )
            env.step(chosen_action)
            steps += 1

    end_time = time.time()

    # Print detailed results
    print(f"\nScheduling completed in {end_time - start_time:.4f} seconds")
    print(f"Steps taken: {steps}")
    env.print_schedule(show_critical_path=True)

    metrics = env.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save the solution
    env.save_state("example_solution.json")