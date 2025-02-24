import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import json
import logging
import time
from collections import defaultdict
from copy import deepcopy
import itertools
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_MACHINES = 100

@dataclass
class Operation:
    """Represents a processing operation with integrated tool requirements"""
    duration: int
    machine: int
    required_tools: Set[int] = field(default_factory=set)
    eligible_machines: Set[int] = field(default_factory=set)
    dependent_operations: List[Tuple[int, int]] = field(default_factory=list)
    tools_to_remove: Set[int] = field(default_factory=set)
    
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
    """Represents a scheduling action"""
    job: int
    machine: int
    operation: int

@dataclass
class ScheduleEntry:
    """Schedule entry with detailed timing information"""
    job: int
    operation: int
    machine: int
    start: int
    duration: int  # Total duration including tool changes
    process_duration: int  # Just the processing time
    tool_change_duration: int  # Just the tool change time
    tools_added: Set[int]
    tools_removed: Set[int]

@dataclass
class State:
    job_progress: np.ndarray  # [num_jobs, max_operations]
    machine_availability: np.ndarray  # [num_machines]
    next_operation_for_job: np.ndarray  # [num_jobs]
    completed_jobs: np.ndarray  # [num_jobs]
    job_start_times: np.ndarray  # [num_jobs]
    tool_state: np.ndarray  # [num_machines, num_tools]

    @classmethod
    def create(cls, num_jobs: int, num_machines: int, max_operations: int, num_tools: Optional[int] = None):
        return cls(
            job_progress=np.zeros((num_jobs, max_operations), dtype=np.int32),
            machine_availability=np.zeros(num_machines, dtype=np.int32),
            next_operation_for_job=np.zeros(num_jobs, dtype=np.int32),
            completed_jobs=np.zeros(num_jobs, dtype=bool),
            job_start_times=np.full(num_jobs, -1, dtype=np.int32),
            tool_state=np.zeros((num_machines, num_tools), dtype=bool) if num_tools else np.array([])
        )

@dataclass
class MachineSpec:
    """Specification for a machine's tooling capabilities"""
    max_slots: int = -1  # -1 means unlimited
    compatible_tools: Optional[Set[int]] = None  # None means all tools compatible
    tool_matrix: Optional[np.ndarray] = None  # Time matrix for tool changes

class Dependencies:
    """Manages dependency tracking"""
    def __init__(self, num_jobs: int):
        self.job_masks = np.ones((num_jobs, num_jobs), dtype=bool)
        self.pending_job_deps = np.zeros(num_jobs, dtype=np.int32)
        self.reverse_job_deps = defaultdict(set)
        self.op_masks = {}
        self.pending_op_deps = {}
        self.reverse_op_deps = defaultdict(set)
        self.active_jobs = set(range(num_jobs))
        self.active_operations = {}

    def add_job_dependency(self, job_id: int, dep_job_id: int):
        self.job_masks[job_id][dep_job_id] = False
        self.pending_job_deps[job_id] += 1
        self.reverse_job_deps[dep_job_id].add(job_id)

    def add_operation_dependency(self, job_id: int, op_id: int, 
                               dep_job_id: int, dep_op_id: int):
        key = (job_id, op_id)
        dep_key = (dep_job_id, dep_op_id)
        
        if key not in self.op_masks:
            self.op_masks[key] = {}
            self.pending_op_deps[key] = 0

        self.op_masks[key][dep_key] = False
        self.pending_op_deps[key] += 1
        self.reverse_op_deps[dep_key].add(key)

    def satisfy_job_dependency(self, job_id: int):
        for dep_job_id in self.reverse_job_deps[job_id]:
            if not self.job_masks[dep_job_id][job_id]:
                self.job_masks[dep_job_id][job_id] = True
                self.pending_job_deps[dep_job_id] -= 1

    def satisfy_operation_dependency(self, job_id: int, op_id: int):
        key = (job_id, op_id)
        for dep_key in self.reverse_op_deps[key]:
            if key in self.op_masks[dep_key] and not self.op_masks[dep_key][key]:
                self.op_masks[dep_key][key] = True
                self.pending_op_deps[dep_key] -= 1

    def is_job_ready(self, job_id: int) -> bool:
        return self.pending_job_deps[job_id] == 0

    def is_operation_ready(self, job_id: int, op_id: int) -> bool:
        key = (job_id, op_id)
        return key not in self.pending_op_deps or self.pending_op_deps[key] == 0

    def remove_completed_job(self, job_id: int):
        self.active_jobs.remove(job_id)
        if job_id in self.active_operations:
            del self.active_operations[job_id]

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

class JobShopEnvironment:
    def __init__(self, jobs: List[Job], machine_specs: Optional[List[MachineSpec]] = None):
        if not jobs:
            raise ValueError("Jobs list cannot be empty")
            
        self.jobs = jobs
        self.machine_specs = machine_specs or []
        
        # Find number of machines
        self.num_machines = max(
            max(op.machine + 1 for job in self.jobs for op in job.operations),
            0
        )
        
        if self.num_machines == 0 or self.num_machines > MAX_MACHINES:
            raise ValueError("Invalid number of machines")
            
        # Find number of tools
        self.num_tools = 0
        if machine_specs:
            for job in jobs:
                for op in job.operations:
                    if op.required_tools:
                        self.num_tools = max(self.num_tools, max(op.required_tools)) if op.required_tools else self.num_tools
            self.num_tools += 1
        
        # Find max operations
        max_operations = max(len(job.operations) for job in self.jobs)
        
        # Initialize state
        self.current_state = State.create(
            num_jobs=len(self.jobs),
            num_machines=self.num_machines,
            max_operations=max_operations,
            num_tools=self.num_tools if machine_specs else None
        )
        
        # Initialize tracking structures
        self.dependencies = Dependencies(len(self.jobs))
        self._initialize_dependencies()
        self.schedule_entries = []
        
        # Initialize machine mappings
        self.machine_to_jobs = defaultdict(set)
        self.operation_to_machine = {}
        self._create_machine_mappings()
        
        # Ready state tracking
        self.ready_jobs = np.zeros(len(self.jobs), dtype=bool)
        self.ready_operations = defaultdict(set)
        
        # Performance tracking
        self.total_time = 0
        
        # Initial update
        self._update_ready_states()

    def _initialize_dependencies(self):
        for job_id, job in enumerate(self.jobs):
            for dep_job_id in job.dependent_jobs:
                self.dependencies.add_job_dependency(job_id, dep_job_id)

            for op_id, op in enumerate(job.operations):
                for dep_job_id, dep_op_id in op.dependent_operations:
                    self.dependencies.add_operation_dependency(
                        job_id, op_id, dep_job_id, dep_op_id)

            self.dependencies.active_operations[job_id] = set(range(len(job.operations)))

    def _create_machine_mappings(self):
        for job_id, job in enumerate(self.jobs):
            for op_id, op in enumerate(job.operations):
                self.machine_to_jobs[op.machine].add(job_id)
                self.operation_to_machine[(job_id, op_id)] = op.machine

    def calculate_tool_change_time(self, machine_id: int, required_tools: Set[int], 
                                 current_tools: Set[int]) -> Tuple[int, Set[int], Set[int]]:
        """Calculate optimal tool changes and duration"""
        spec = self.machine_specs[machine_id]
        if not spec or spec.max_slots == -1:
            return 0, set(), set()
            
        # If all required tools are already mounted, no changes needed
        if required_tools.issubset(current_tools):
            return 0, set(), set()
            
        tools_to_add = required_tools - current_tools
        tools_to_remove = set()
        
        # Calculate optimal tool changes
        if spec.max_slots != -1:
            # Keep track of current slot usage
            slots_used = len(current_tools)
            slots_needed = len(tools_to_add)
            available_slots = spec.max_slots - slots_used
            
            # If we need more slots than available
            if available_slots < slots_needed:
                # Calculate how many tools we need to remove
                tools_to_free = slots_needed - available_slots
                
                # Get removable tools (tools not needed for this operation)
                removable_tools = current_tools - required_tools
                
                if spec.tool_matrix is not None:
                    # Sort by removal time + potential future add time
                    def removal_cost(tool):
                        remove_time = spec.tool_matrix[tool][0]
                        potential_add_time = min(spec.tool_matrix[0][t] for t in tools_to_add)
                        return remove_time + potential_add_time
                    
                    removable_tools = sorted(removable_tools, key=removal_cost)
                
                tools_to_remove = set(list(removable_tools)[:tools_to_free])
        
        # Calculate minimum duration for tool changes
        duration = 0
        if spec.tool_matrix is not None:
            # First remove all tools that need to be removed
            for tool in tools_to_remove:
                duration += spec.tool_matrix[tool][0]
            
            # Then add all new tools needed
            for tool in tools_to_add:
                duration += spec.tool_matrix[0][tool]
        
        return duration, tools_to_add, tools_to_remove

    def _update_ready_states(self):
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

    def get_possible_actions(self) -> List[Action]:
        possible_actions = []

        for job_id in self.dependencies.active_jobs:
            if not self.ready_jobs[job_id]:
                continue

            for op_id in self.ready_operations[job_id]:
                if self.current_state.job_progress[job_id, op_id] == 0:
                    op = self.jobs[job_id].operations[op_id]
                    
                    for machine_id in op.eligible_machines:
                        # Validate tool constraints
                        current_tools = set(np.where(self.current_state.tool_state[machine_id])[0])
                        spec = self.machine_specs[machine_id]
                        
                        if spec:
                            # Check tool compatibility
                            if spec.compatible_tools is not None:
                                if not op.required_tools.issubset(spec.compatible_tools):
                                    continue
                            
                            # Check capacity
                            if spec.max_slots != -1:
                                tools_to_add = op.required_tools - current_tools
                                removable_tools = current_tools - op.required_tools
                                needed_space = len(current_tools) + len(tools_to_add) - len(removable_tools)
                                if needed_space > spec.max_slots:
                                    continue
                        
                        possible_actions.append(Action(
                            job=job_id,
                            machine=machine_id,
                            operation=op_id
                        ))

        return possible_actions

    def step(self, action: Action) -> State:
        """Execute an action and update the environment state."""
        if action.job not in self.dependencies.active_jobs:
            raise ValueError(f"Invalid action: Job {action.job} is not active")

        current_op = self.current_state.next_operation_for_job[action.job]
        if action.operation != current_op:
            raise ValueError(f"Invalid action: Expected operation {current_op}")

        op = self.jobs[action.job].operations[action.operation]
        if action.machine not in op.eligible_machines:
            raise ValueError(f"Invalid action: Machine {action.machine} not eligible")

        # Calculate earliest possible start time
        start_time = self.current_state.machine_availability[action.machine]

        # Check previous operation completion FIRST
        if action.operation > 0:
            prev_end = self.current_state.job_progress[action.job, action.operation - 1]
            if prev_end == 0:
                raise ValueError("Previous operation not completed")
            start_time = max(start_time, prev_end)
            logger.debug(f"Job {action.job} Op {action.operation} waiting for prev op: new start {start_time}")

        # Calculate tool changes
        current_tools = self.get_current_tools(action.machine)
        tool_change_duration, tools_to_add, tools_to_remove = self.calculate_tool_change_time(
            action.machine, op.required_tools, current_tools
        )

        # The entire operation (tool change + processing) is treated as one unit
        total_duration = tool_change_duration + op.duration
        
        # Check other dependencies AFTER determining total duration
        for dep_job in self.jobs[action.job].dependent_jobs:
            last_op = len(self.jobs[dep_job].operations) - 1
            dep_end = self.current_state.job_progress[dep_job, last_op]
            if dep_end == 0:
                raise ValueError(f"Dependent job {dep_job} not completed")
            start_time = max(start_time, dep_end)
            logger.debug(f"Job {action.job} Op {action.operation} waiting for job {dep_job}: new start {start_time}")

        for dep_job, dep_op in op.dependent_operations:
            dep_end = self.current_state.job_progress[dep_job, dep_op]
            if dep_end == 0:
                raise ValueError(f"Dependent operation ({dep_job}, {dep_op}) not completed")
            start_time = max(start_time, dep_end)
            logger.debug(f"Job {action.job} Op {action.operation} waiting for op {dep_job}-{dep_op}: new start {start_time}")

        # Update tool state
        for tool in tools_to_remove:
            self.current_state.tool_state[action.machine, tool] = False
        for tool in tools_to_add:
            self.current_state.tool_state[action.machine, tool] = True

        # Operation end time includes both tool change and processing
        end_time = start_time + total_duration

        # Record start time for job if this is first operation
        if self.current_state.job_start_times[action.job] == -1:
            self.current_state.job_start_times[action.job] = start_time

        # Update state with combined duration
        self.current_state.job_progress[action.job, action.operation] = end_time
        self.current_state.machine_availability[action.machine] = end_time

        # Create ONE schedule entry that includes both tool change and processing
        schedule_entry = ScheduleEntry(
            job=action.job,
            operation=action.operation,
            machine=action.machine,
            start=start_time,
            duration=total_duration,
            process_duration=op.duration,
            tool_change_duration=tool_change_duration,
            tools_added=tools_to_add,
            tools_removed=tools_to_remove
        )
        self.schedule_entries.append(schedule_entry)

        # Update completion states - ONE update for entire operation
        if self.current_state.next_operation_for_job[action.job] == action.operation:
            self.current_state.next_operation_for_job[action.job] += 1
            self.dependencies.satisfy_operation_dependency(action.job, action.operation)

            if self.current_state.next_operation_for_job[action.job] == len(self.jobs[action.job].operations):
                self.current_state.completed_jobs[action.job] = True
                self.dependencies.remove_completed_job(action.job)
                self.dependencies.satisfy_job_dependency(action.job)

        # Update total time
        self.total_time = max(self.total_time, end_time)
        self._update_ready_states()

        return self.current_state
        # Validate action
        if action.job not in self.dependencies.active_jobs:
            raise ValueError(f"Invalid action: Job {action.job} is not active")

        current_op = self.current_state.next_operation_for_job[action.job]
        if action.operation != current_op:
            raise ValueError(f"Invalid action: Expected operation {current_op}")

        op = self.jobs[action.job].operations[action.operation]
        if action.machine not in op.eligible_machines:
            raise ValueError(f"Invalid action: Machine {action.machine} not eligible")

        # Calculate start time with debugging
        start_time = self.current_state.machine_availability[action.machine]
        start_time = self.debug_start_time_calculation(action, start_time)

        # Calculate tool changes
        current_tools = set(np.where(self.current_state.tool_state[action.machine])[0])
        tool_change_duration, tools_to_add, tools_to_remove = self.calculate_tool_change_time(
            action.machine, op.required_tools, current_tools
        )

        # Total duration includes tool change
        total_duration = tool_change_duration + op.duration

        # Check previous operation completion
        if action.operation > 0:
            prev_end = self.current_state.job_progress[action.job, action.operation - 1]
            if prev_end == 0:
                raise ValueError("Previous operation not completed")
            start_time = max(start_time, prev_end)

        # Check job dependencies
        for dep_job in self.jobs[action.job].dependent_jobs:
            dep_job_last_op = len(self.jobs[dep_job].operations) - 1
            dep_end = self.current_state.job_progress[dep_job, dep_job_last_op]
            if dep_end == 0:
                raise ValueError(f"Dependent job {dep_job} not completed")
            start_time = max(start_time, dep_end)

        # Check operation dependencies
        for dep_job, dep_op in op.dependent_operations:
            dep_end = self.current_state.job_progress[dep_job, dep_op]
            if dep_end == 0:
                raise ValueError(f"Dependent operation ({dep_job}, {dep_op}) not completed")
            start_time = max(start_time, dep_end)

        # Update tool state
        for tool in tools_to_remove:
            self.current_state.tool_state[action.machine, tool] = False
        for tool in tools_to_add:
            self.current_state.tool_state[action.machine, tool] = True

        # Calculate end time and update state
        end_time = start_time + total_duration

        # Record job start time if first operation
        if self.current_state.job_start_times[action.job] == -1:
            self.current_state.job_start_times[action.job] = start_time

        # Update progress and machine availability
        self.current_state.job_progress[action.job, action.operation] = end_time
        self.current_state.machine_availability[action.machine] = end_time

        # Create schedule entry
        schedule_entry = ScheduleEntry(
            job=action.job,
            operation=action.operation,
            machine=action.machine,
            start=start_time,
            duration=total_duration,
            process_duration=op.duration,
            tool_change_duration=tool_change_duration,
            tools_added=tools_to_add,
            tools_removed=tools_to_remove
        )
        self.schedule_entries.append(schedule_entry)

        # Update operation completion state
        if self.current_state.next_operation_for_job[action.job] == action.operation:
            self.current_state.next_operation_for_job[action.job] += 1
            self.dependencies.satisfy_operation_dependency(action.job, action.operation)

            # Check job completion
            if self.current_state.next_operation_for_job[action.job] == len(self.jobs[action.job].operations):
                self.current_state.completed_jobs[action.job] = True
                self.dependencies.remove_completed_job(action.job)
                self.dependencies.satisfy_job_dependency(action.job)

        # Update total time
        self.total_time = max(self.total_time, end_time)

        # Update ready states
        self._update_ready_states()

        return self.current_state

    def reset(self):
        """Reset the environment to initial state."""
        self.current_state.job_progress.fill(0)
        self.current_state.machine_availability.fill(0)
        self.current_state.next_operation_for_job.fill(0)
        self.current_state.completed_jobs.fill(False)
        self.current_state.job_start_times.fill(-1)
        self.current_state.tool_state.fill(False)

        self.total_time = 0
        self.schedule_entries.clear()

        self.dependencies = Dependencies(len(self.jobs))
        self._initialize_dependencies()

        self.ready_jobs.fill(False)
        self.ready_operations.clear()
        self._update_ready_states()

    def is_done(self) -> bool:
        """Check if all jobs are completed."""
        return not bool(self.dependencies.active_jobs)

    def get_schedule_data(self) -> List[List[ScheduleEntry]]:
        """Get the complete schedule data organized by machine."""
        schedule_data = [[] for _ in range(self.num_machines)]
        for entry in self.schedule_entries:
            schedule_data[entry.machine].append(entry)
        return schedule_data

    def get_machine_utilization(self) -> List[float]:
        """Calculate machine utilization percentages."""
        total_time = max(1, self.total_time)
        machine_busy_time = np.zeros(self.num_machines)

        for entry in self.schedule_entries:
            machine_busy_time[entry.machine] += entry.duration

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

    def get_performance_metrics(self) -> dict:
        """Calculate schedule performance metrics."""
        metrics = {
            "makespan": self.total_time,
            "machine_utilization": self.get_machine_utilization(),
            "avg_machine_utilization": np.mean(self.get_machine_utilization()),
            "num_operations": len(self.schedule_entries),
            "critical_path_length": len(self.get_critical_path()),
            "completed_jobs": int(np.sum(self.current_state.completed_jobs))
        }

        process_times = []
        tool_change_times = []
        waiting_times = []

        for entry in self.schedule_entries:
            process_times.append(entry.process_duration)
            if entry.tool_change_duration > 0:
                tool_change_times.append(entry.tool_change_duration)
            waiting_time = entry.start - max(
                [0] + [self.current_state.job_progress[entry.job, op_idx]
                      for op_idx in range(entry.operation)]
            )
            waiting_times.append(waiting_time)

        metrics.update({
            "avg_process_time": np.mean(process_times) if process_times else 0,
            "total_tool_changes": len(tool_change_times),
            "total_tool_change_time": sum(tool_change_times),
            "avg_tool_change_time": np.mean(tool_change_times) if tool_change_times else 0,
            "avg_waiting_time": np.mean(waiting_times) if waiting_times else 0,
            "max_waiting_time": max(waiting_times) if waiting_times else 0
        })

        return metrics

    def print_schedule(self, show_critical_path: bool = False):
        """Print the current schedule in a readable format."""
        schedule_data = self.get_schedule_data()
        critical_path = self.get_critical_path() if show_critical_path else []

        print("\nSchedule:")
        for i, machine_schedule in enumerate(schedule_data):
            util = self.get_machine_utilization()[i] * 100
            print(f"\nMachine {i} (Utilization: {util:.1f}%):")

            for entry in sorted(machine_schedule, key=lambda x: x.start):
                is_critical = (entry.job, entry.operation) in critical_path
                critical_marker = "*" if is_critical else " "
                
                print(f"{critical_marker}[Job {entry.job}, Op {entry.operation}] "
                      f"Start: {entry.start:4d}, Duration: {entry.duration:3d} "
                      f"(Process: {entry.process_duration}, Tool Change: {entry.tool_change_duration})")
                if entry.tools_added or entry.tools_removed:
                    print(f"    Tools Added: {entry.tools_added}, Removed: {entry.tools_removed}")

        print(f"\nTotal makespan: {self.total_time}")
        if show_critical_path:
            print("\nCritical path operations marked with *")

    def generate_html_gantt(self, filename: str):
        """Generate an HTML file containing the Gantt chart visualization."""
        schedule_data = self.get_schedule_data()
        critical_path = set(self.get_critical_path())
        
        # Generate using the GanttChartGenerator
        GanttChartGenerator.generate_html(
            schedule_data=schedule_data,
            critical_path=critical_path,
            total_time=self.total_time,
            machine_utilization=self.get_machine_utilization(),
            filename=filename
        )



    # Add this method to JobShopEnvironment class
    def analyze_schedule_gaps(self):
        """Analyze gaps in the schedule and their causes"""
        for machine_id in range(self.num_machines):
            entries = [e for e in self.schedule_entries if e.machine == machine_id]
            entries.sort(key=lambda x: x.start)
            
            print(f"\nAnalyzing Machine {machine_id}:")
            last_end = 0
            for entry in entries:
                if entry.start > last_end:
                    gap = entry.start - last_end
                    print(f"  Gap of {gap} units before Job {entry.job}, Op {entry.operation}")
                    
                    # Check dependencies
                    op = self.jobs[entry.job].operations[entry.operation]
                    if entry.operation > 0:
                        prev_end = self.current_state.job_progress[entry.job, entry.operation - 1]
                        print(f"    Previous op in job ends at: {prev_end}")
                    
                    for dep_job, dep_op in op.dependent_operations:
                        dep_end = self.current_state.job_progress[dep_job, dep_op]
                        print(f"    Dependency (Job {dep_job}, Op {dep_op}) ends at: {dep_end}")
                        
                    # Check tool state at this point
                    current_tools = set(np.where(self.current_state.tool_state[machine_id])[0])
                    print(f"    Required tools: {op.required_tools}")
                    print(f"    Current tools: {current_tools}")
                
                last_end = entry.start + entry.duration

    def get_current_tools(self, machine_id: int) -> Set[int]:
        """Get currently mounted tools as regular integers"""
        return {int(i) for i in np.where(self.current_state.tool_state[machine_id])[0]}

    # Add to step method before calculating start time
    def debug_start_time_calculation(self, action: Action, start_time: int):
        """Debug the start time calculation"""
        print(f"\nCalculating start time for Job {action.job}, Op {action.operation}")
        print(f"Initial start time from machine availability: {start_time}")
        
        if action.operation > 0:
            prev_end = self.current_state.job_progress[action.job, action.operation - 1]
            print(f"Previous operation in job ends at: {prev_end}")
            start_time = max(start_time, prev_end)
            print(f"After considering previous op: {start_time}")
        
        for dep_job in self.jobs[action.job].dependent_jobs:
            dep_end = self.current_state.job_progress[dep_job, -1]
            print(f"Job dependency {dep_job} ends at: {dep_end}")
            start_time = max(start_time, dep_end)
            print(f"After considering job dependency: {start_time}")
        
        op = self.jobs[action.job].operations[action.operation]
        for dep_job, dep_op in op.dependent_operations:
            dep_end = self.current_state.job_progress[dep_job, dep_op]
            print(f"Operation dependency ({dep_job}, {dep_op}) ends at: {dep_end}")
            start_time = max(start_time, dep_end)
            print(f"After considering operation dependency: {start_time}")
        
        return start_time

    def verify_state(self) -> List[str]:
        """Run all state verifications."""
        errors = []
        # Verify operation sequence
        for job_id, job in enumerate(self.jobs):
            for op_idx in range(1, len(job.operations)):
                curr_end = self.current_state.job_progress[job_id, op_idx]
                prev_end = self.current_state.job_progress[job_id, op_idx - 1]
                
                if curr_end > 0:
                    curr_start = curr_end - job.operations[op_idx].duration
                    if curr_start < prev_end:
                        errors.append(
                            f"Operation sequence violated for Job {job_id}: "
                            f"Op {op_idx} starts at {curr_start} before Op {op_idx-1} ends at {prev_end}"
                        )

        # Verify machine capacity
        for m_id in range(self.num_machines):
            machine_entries = [e for e in self.schedule_entries if e.machine == m_id]
            machine_entries.sort(key=lambda x: x.start)
            
            for i in range(1, len(machine_entries)):
                if machine_entries[i].start < machine_entries[i-1].start + machine_entries[i-1].duration:
                    errors.append(
                        f"Machine capacity violated on Machine {m_id}: "
                        f"Job {machine_entries[i].job} Op {machine_entries[i].operation} starts at "
                        f"{machine_entries[i].start} before previous operation ends at "
                        f"{machine_entries[i-1].start + machine_entries[i-1].duration}"
                    )

        # Verify tool constraints
        if self.machine_specs:
            for machine_id in range(self.num_machines):
                spec = self.machine_specs[machine_id]
                current_tools = set(np.where(self.current_state.tool_state[machine_id])[0])
                
                if spec.max_slots != -1 and len(current_tools) > spec.max_slots:
                    errors.append(
                        f"Tool constraint violated: Machine {machine_id} has {len(current_tools)} "
                        f"tools mounted, exceeding maximum of {spec.max_slots}"
                    )
                
                if spec.compatible_tools is not None:
                    invalid_tools = current_tools - spec.compatible_tools
                    if invalid_tools:
                        errors.append(
                            f"Tool compatibility violated: Machine {machine_id} has incompatible "
                            f"tools mounted: {invalid_tools}"
                        )

        return errors

def get_state(env: JobShopEnvironment):
    """Convert environment state to tuple for Q-learning."""
    return tuple(env.current_state.next_operation_for_job.tolist())

def run_episode_qlearning_shaped(env: JobShopEnvironment, Q, alpha=0.1, 
                               epsilon=0.2, gamma=0.9,
                               process_reward=20.0,
                               tool_change_cost=2.0,
                               duration_penalty=1.0,
                               final_makespan_scale=10.0):
    """Run one Q-learning episode with shaped rewards accounting for tool changes."""
    env.reset()
    total_step_reward = 0.0
    last_state = None
    last_action = None

    while not env.is_done():
        s = get_state(env)
        possible_actions = env.get_possible_actions()
        if not possible_actions:
            total_step_reward -= 5.0
            break

        possible_jobs = set(a.job for a in possible_actions)

        # Epsilon-greedy job selection
        if random.random() < epsilon:
            chosen_job = random.choice(list(possible_jobs))
        else:
            best_q = -math.inf
            chosen_job = None
            for job_id in possible_jobs:
                val = Q[s].get(job_id, 0.0)
                if val > best_q:
                    best_q = val
                    chosen_job = job_id

        valid_for_job = [a for a in possible_actions if a.job == chosen_job]
        if not valid_for_job:
            total_step_reward -= 2.0
            break

        chosen_action = valid_for_job[0]
        old_state = env.current_state
        new_state = env.step(chosen_action)
        
        # Calculate reward based on the schedule entry
        schedule_entry = env.schedule_entries[-1]
        r_step = (process_reward 
                 - (duration_penalty * schedule_entry.process_duration)
                 - (tool_change_cost * schedule_entry.tool_change_duration))
        
        total_step_reward += r_step

        if last_state is not None and last_action is not None:
            old_q = Q[last_state].get(last_action, 0.0)
            s_next = get_state(env)
            max_next_q = max(Q[s_next].values()) if s_next in Q and Q[s_next] else 0.0
            new_q = old_q + alpha*(r_step + gamma*max_next_q - old_q)
            Q[last_state][last_action] = new_q

        last_state = s
        last_action = chosen_job

    makespan = env.total_time
    final_reward = -(makespan / final_makespan_scale)
    total_reward = total_step_reward + final_reward

    if last_state is not None and last_action is not None:
        old_q = Q[last_state].get(last_action, 0.0)
        new_q = old_q + alpha*(final_reward - old_q)
        Q[last_state][last_action] = new_q

    return makespan, total_reward

def train_qlearning_shaped(env: JobShopEnvironment,
                          num_episodes=2000,
                          alpha=0.1,
                          epsilon=0.2,
                          gamma=0.9,
                          process_reward=20.0,
                          tool_change_cost=2.0,
                          duration_penalty=1.0,
                          final_makespan_scale=10.0,
                          epsilon_decay=0.999,
                          save_interval=100):
    """Train Q-learning policy with integrated tool change costs."""
    Q = defaultdict(dict)
    best_makespan = float('inf')
    best_schedule = None

    for episode in range(1, num_episodes + 1):
        makespan, ep_reward = run_episode_qlearning_shaped(
            env, Q,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            process_reward=process_reward,
            tool_change_cost=tool_change_cost,
            duration_penalty=duration_penalty,
            final_makespan_scale=final_makespan_scale
        )

        # Track best solution
        if makespan < best_makespan and makespan > 0:
            best_makespan = makespan
            best_schedule = env.schedule_entries[:]

        if episode % 10 == 0:
            print(f"Episode {episode}, Makespan={makespan}, "
                  f"Best={best_makespan}, Eps={epsilon:.3f}, "
                  f"EpReward={ep_reward:.2f}")

        if episode % save_interval == 0 and best_schedule is not None:
            save_best_solution(env, best_schedule, best_makespan)

        epsilon = max(0.01, epsilon * epsilon_decay)

    return Q, best_schedule, best_makespan

def save_best_solution(env: JobShopEnvironment, schedule: List[ScheduleEntry], 
                      best_makespan: float, filename: str = "best_solution.json"):
    """Save the best schedule found to a JSON file."""
    data = {
        "best_makespan": best_makespan,
        "schedule": [],
        "metrics": env.get_performance_metrics()
    }

    # Group schedule by machine
    machine_schedules = defaultdict(list)
    for entry in schedule:
        machine_schedules[entry.machine].append({
            "job": entry.job,
            "operation": entry.operation,
            "start": entry.start,
            "duration": entry.duration,
            "process_duration": entry.process_duration,
            "tool_change_duration": entry.tool_change_duration,
            "tools_added": list(entry.tools_added),
            "tools_removed": list(entry.tools_removed)
        })

    # Add sorted machine schedules
    for machine_id in sorted(machine_schedules.keys()):
        data["schedule"].append({
            "machine": machine_id,
            "operations": sorted(machine_schedules[machine_id], key=lambda x: x["start"])
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)

    print(f"Best solution saved to {filename} with makespan={best_makespan}")

class GanttChartGenerator:
    """Generates HTML Gantt chart visualizations."""
    
    @staticmethod
    def generate_html(schedule_data: List[List[ScheduleEntry]], 
                     critical_path: Set[Tuple[int, int]], 
                     total_time: int,
                     machine_utilization: List[float],
                     filename: str):
        """Generate an HTML Gantt chart."""
        job_colors = [
            '#4299E1', '#48BB78', '#ECC94B', '#9F7AEA', '#ED64A6',
            '#667EEA', '#F56565', '#ED8936', '#38B2AC', '#4FD1C5'
        ]

        time_scale = max(100, total_time + 20)

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
                .tool-change {{
                    background: repeating-linear-gradient(
                        45deg,
                        rgba(0, 0, 0, 0.1),
                        rgba(0, 0, 0, 0.1) 10px,
                        rgba(0, 0, 0, 0.2) 10px,
                        rgba(0, 0, 0, 0.2) 20px
                    );
                }}
                .critical-path {{ border: 2px solid #E53E3E; }}
                .time-scale {{ display: flex; margin-left: 120px; }}
                .time-mark {{ flex-grow: 1; text-align: center; font-size: 12px; }}
                .legend {{ margin-top: 20px; display: flex; gap: 20px; flex-wrap: wrap; }}
                .legend-item {{ display: flex; align-items: center; gap: 8px; }}
                .legend-color {{ width: 20px; height: 20px; }}
                .tool-info {{ font-size: 10px; margin-top: 2px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Job Shop Schedule Gantt Chart</h2>
                
                <div class="time-scale">
                    {' '.join(f'<div class="time-mark">{t}</div>'
                             for t in range(0, time_scale + 1, max(1, time_scale // 10)))}
                </div>
                
                {GanttChartGenerator._generate_machine_rows_html(schedule_data, critical_path, 
                                                               job_colors, time_scale, machine_utilization)}
                
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="border: 2px solid #E53E3E"></div>
                        <span>Critical Path</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color tool-change" style="width: 40px; background: #999"></div>
                        <span>Tool Change</span>
                    </div>
                    {GanttChartGenerator._generate_legend_html(schedule_data, job_colors)}
                </div>
            </div>
        </body>
        </html>
        """

        with open(filename, 'w') as f:
            f.write(html_content)

        print(f"Gantt chart saved to {filename}")

    @staticmethod
    def _generate_machine_rows_html(schedule_data, critical_path, job_colors, 
                                  time_scale, machine_utilization):
        rows = []
        for machine_id, machine_schedule in enumerate(schedule_data):
            util = machine_utilization[machine_id] * 100

            operations_html = []
            for entry in sorted(machine_schedule, key=lambda x: x.start):
                is_critical = (entry.job, entry.operation) in critical_path
                critical_class = ' critical-path' if is_critical else ''
                has_tool_change = ' tool-change' if entry.tool_change_duration > 0 else ''
                color = job_colors[entry.job % len(job_colors)]

                left = (entry.start / time_scale) * 100
                width = (entry.duration / time_scale) * 100

                tool_info = ""
                if entry.tools_added or entry.tools_removed:
                    tool_info = (f'<div class="tool-info">Added: {entry.tools_added}<br/>'
                               f'Removed: {entry.tools_removed}</div>')

                operations_html.append(
                    f'<div class="operation{critical_class}{has_tool_change}" '
                    f'style="left: {left}%; width: {width}%; background-color: {color}">'
                    f'J{entry.job}-{entry.operation}'
                    f'{tool_info}'
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

    @staticmethod
    def _generate_legend_html(schedule_data, job_colors):
        legend_items = []
        unique_jobs = set(entry.job for machine in schedule_data
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

if __name__ == "__main__":
    # Example usage
    machine_specs = [
        MachineSpec(
            max_slots=4,
            compatible_tools={1, 2, 3, 4, 5, 6},
            tool_matrix=np.array([
                [0, 8, 10, 12, 15, 18, 20],
                [6, 0, 5, 8, 10, 12, 15],
                [8, 5, 0, 6, 8, 10, 12],
                [10, 8, 6, 0, 5, 8, 10],
                [12, 10, 8, 5, 0, 6, 8],
                [15, 12, 10, 8, 6, 0, 5],
                [18, 15, 12, 10, 8, 5, 0]
            ])
        ),
        MachineSpec(
            max_slots=2,
            compatible_tools={2, 3, 4},
            tool_matrix=np.array([
                [0, 0, 15, 18, 20, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [12, 0, 0, 10, 12, 0, 0],
                [15, 0, 10, 0, 8, 0, 0],
                [18, 0, 12, 8, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        ),
        MachineSpec(
            max_slots=3,
            compatible_tools={1, 3, 4, 5},
            tool_matrix=np.array([
                [0, 10, 0, 12, 15, 18, 0],
                [8, 0, 0, 10, 12, 15, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [10, 8, 0, 0, 8, 10, 0],
                [12, 10, 0, 8, 0, 8, 0],
                [15, 12, 0, 10, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        )
    ]

    jobs = [
        Job(operations=[
            Operation(duration=4, machine=0, required_tools={1, 2}),
            Operation(duration=3, machine=1, required_tools={2, 3}),
            Operation(duration=5, machine=2, required_tools={3, 4})
        ]),
        Job(operations=[
            Operation(duration=3, machine=1, required_tools={3, 4}),
            Operation(duration=6, machine=0, required_tools={1, 6}),
            Operation(duration=4, machine=2, required_tools={1, 3})
        ]),
        Job(operations=[
            Operation(duration=2, machine=2, required_tools={4, 5}),
            Operation(duration=3, machine=0, required_tools={2, 6})
        ]),
        Job(operations=[
            Operation(duration=5, machine=0, required_tools={1, 2}),
            Operation(duration=4, machine=1, required_tools={2, 4}),
            Operation(duration=3, machine=2, required_tools={3, 5}),
            Operation(duration=4, machine=0, required_tools={1, 6})
        ])
    ]

    # Add operation dependencies
    jobs[1].operations[1].dependent_operations = [(1, 0)]
    jobs[2].operations[1].dependent_operations = [(0, 1)]
    jobs[3].operations[2].dependent_operations = [(1, 1)]

    # Add job dependencies
    jobs[2].dependent_jobs = [0]
    jobs[3].dependent_jobs = [1]

    env = JobShopEnvironment(jobs, machine_specs)

    # Train the scheduler
    Q, best_schedule, best_makespan = train_qlearning_shaped(
        env,
        num_episodes=2000,
        alpha=0.1,
        epsilon=0.3,
        gamma=0.9,
        process_reward=20.0,
        tool_change_cost=2.0,
        duration_penalty=1.0,
        final_makespan_scale=10.0,
        epsilon_decay=0.999,
        save_interval=100
    )

    print(f"\nTraining complete")
    print(f"Best makespan found: {best_makespan}")

    # Analyze the schedule
    print("\nAnalyzing schedule for gaps and issues:")
    env.analyze_schedule_gaps()
    
    # Check for any state violations
    errors = env.verify_state()
    if errors:
        print("\nFound schedule violations:")
        for error in errors:
            print(f"  {error}")
    else:
        print("\nNo schedule violations found")
    
    # Visualize final schedule
    env.print_schedule(show_critical_path=True)
    env.generate_html_gantt("final_schedule.html")