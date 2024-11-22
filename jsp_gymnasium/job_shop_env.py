# job_shop_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
import json
import logging
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
    job_progress: np.ndarray  
    machine_availability: np.ndarray  
    next_operation_for_job: np.ndarray  
    completed_jobs: np.ndarray  
    job_start_times: np.ndarray  

    @classmethod
    def create(cls, num_jobs: int, num_machines: int, max_operations: int):
        return cls(
            job_progress=np.zeros((num_jobs, max_operations), dtype=np.int32),
            machine_availability=np.zeros(num_machines, dtype=np.int32),
            next_operation_for_job=np.zeros(num_jobs, dtype=np.int32),
            completed_jobs=np.zeros(num_jobs, dtype=bool),
            job_start_times=np.full(num_jobs, -1, dtype=np.int32)
        )

class Dependencies:
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

    def add_operation_dependency(self, job_id: int, op_id: int, dep_job_id: int, dep_op_id: int):
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

class JobShopGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, jobs: List[Job], render_mode: Optional[str] = None):
        super().__init__()
        
        self.jobs = jobs
        self.render_mode = render_mode
        self.num_machines = 0
        self.total_time = 0
        
        for job in jobs:
            for op in job.operations:
                self.num_machines = max(self.num_machines, op.machine + 1)
                
        if self.num_machines == 0 or self.num_machines > MAX_MACHINES:
            raise ValueError("Invalid number of machines")
            
        self.max_operations = max(len(job.operations) for job in jobs)
        
        self.action_space = spaces.Discrete(len(jobs) * self.num_machines * self.max_operations)
        
        obs_dim = (
            (len(jobs) * self.max_operations) +  
            self.num_machines +                   
            len(jobs) +                          
            len(jobs) +                          
            len(jobs)                            
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.reset()

    def _initialize_dependencies(self):
        for job_id, job in enumerate(self.jobs):
            for dep_job_id in job.dependent_jobs:
                self.dependencies.add_job_dependency(job_id, dep_job_id)
            
            for op_id, op in enumerate(job.operations):
                for dep_job_id, dep_op_id in op.dependent_operations:
                    self.dependencies.add_operation_dependency(
                        job_id, op_id, dep_job_id, dep_op_id)
            
            self.dependencies.active_operations[job_id] = set(range(len(job.operations)))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.current_state = State.create(
            num_jobs=len(self.jobs),
            num_machines=self.num_machines,
            max_operations=self.max_operations
        )
        
        self.dependencies = Dependencies(len(self.jobs))
        self._initialize_dependencies()
        self.action_history = []
        
        self.total_time = 0
        
        observation = self._get_observation()
        
        return observation, {}

    def _get_observation(self) -> np.ndarray:
        return np.concatenate([
            self.current_state.job_progress.flatten(),
            self.current_state.machine_availability,
            self.current_state.next_operation_for_job,
            self.current_state.completed_jobs.astype(np.float32),
            self.current_state.job_start_times
        ])

    def _decode_action(self, action_id: int) -> Action:
        job_id = action_id // (self.num_machines * self.max_operations)
        remainder = action_id % (self.num_machines * self.max_operations)
        machine_id = remainder // self.max_operations
        operation_id = remainder % self.max_operations
        return Action(job_id, machine_id, operation_id)

    def _is_valid_action(self, action: Action) -> bool:
        if action.job >= len(self.jobs) or action.operation >= len(self.jobs[action.job].operations):
            return False
            
        if self.current_state.completed_jobs[action.job]:
            return False
            
        if action.operation != self.current_state.next_operation_for_job[action.job]:
            return False
            
        op = self.jobs[action.job].operations[action.operation]
        if action.machine not in op.eligible_machines:
            return False
            
        return True

    def _execute_action(self, action: Action):
        op = self.jobs[action.job].operations[action.operation]
        start_time = self.current_state.machine_availability[action.machine]

        # Consider job dependencies
        for dep_job in self.jobs[action.job].dependent_jobs:
            dep_job_last_op = len(self.jobs[dep_job].operations) - 1
            start_time = max(start_time,
                           self.current_state.job_progress[dep_job, dep_job_last_op])

        # Check previous operation
        if action.operation > 0:
            start_time = max(start_time,
                           self.current_state.job_progress[action.job, action.operation - 1])

        # Consider operation dependencies
        for dep_job, dep_op in op.dependent_operations:
            start_time = max(start_time,
                           self.current_state.job_progress[dep_job, dep_op])

        # Update state
        end_time = start_time + op.duration
        if self.current_state.job_start_times[action.job] == -1:
            self.current_state.job_start_times[action.job] = start_time

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

        self.total_time = max(self.total_time, end_time)
        self.action_history.append(action)

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = self._decode_action(action_id)
        
        if not self._is_valid_action(action):
            return (
                self._get_observation(),
                -1000.0,
                False,
                False,
                {"valid_action": False}
            )
            
        self._execute_action(action)
        
        reward = self._calculate_reward()
        
        done = self.is_done()
        
        info = {
            "valid_action": True,
            "makespan": self.total_time,
            "completed_jobs": int(np.sum(self.current_state.completed_jobs))
        }
        
        return self._get_observation(), reward, done, False, info

    def _calculate_reward(self) -> float:
        completed_jobs = np.sum(self.current_state.completed_jobs)
        total_jobs = len(self.jobs)
        
        # Base reward structure
        reward = 0
        
        # Reward for completing jobs
        completion_reward = (completed_jobs / total_jobs) * 1000
        reward += completion_reward
        
        # Penalize makespan only when jobs are completed
        if completed_jobs > 0:
            makespan_penalty = -self.total_time / 100
            reward += makespan_penalty
        
        # Machine utilization bonus
        machine_utilization = np.mean(self.get_machine_utilization())
        utilization_reward = machine_utilization * 200
        reward += utilization_reward
        
        # Add completion bonus
        if completed_jobs == total_jobs:
            reward += 2000  # Large bonus for complete solutions
            
        # Add penalty for very long makespans
        if self.total_time > 1000:  # Adjust threshold based on your problem
            reward -= (self.total_time - 1000) / 10
            
        return float(reward)

    def get_machine_utilization(self) -> List[float]:
        total_time = max(1, self.total_time)
        machine_busy_time = np.zeros(self.num_machines)
        
        for action in self.action_history:
            op = self.jobs[action.job].operations[action.operation]
            machine_busy_time[action.machine] += op.duration
            
        return [busy_time / total_time for busy_time in machine_busy_time]

    def is_done(self) -> bool:
        return not bool(self.dependencies.active_jobs)

    def render(self, mode="human"):
        if mode == "human":
            print("\nSchedule:")
            for i in range(self.num_machines):
                util = self.get_machine_utilization()[i] * 100
                print(f"\nMachine {i} (Utilization: {util:.1f}%):")
                
                # Get operations scheduled on this machine
                machine_ops = [
                    (action, self.jobs[action.job].operations[action.operation])
                    for action in self.action_history
                    if action.machine == i
                ]
                
                for action, op in machine_ops:
                    end_time = self.current_state.job_progress[action.job, action.operation]
                    start_time = end_time - op.duration
                    print(f" [Job {action.job}, Op {action.operation}] "
                          f"Start: {start_time:4d}, Duration: {op.duration:3d}, "
                          f"End: {end_time:4d}")
            
            print(f"\nTotal makespan: {self.total_time}")

    def close(self):
        pass