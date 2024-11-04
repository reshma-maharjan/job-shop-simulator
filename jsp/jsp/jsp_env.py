from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import jobshop
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ObservationSpace(ABC):
    @abstractmethod
    def get_observation_space(self, env: JobShopGymEnv) -> spaces.Space:
        pass

    @abstractmethod
    def get_observation(self, env: JobShopGymEnv) -> np.ndarray:
        pass


class DefaultObservationSpace(ObservationSpace):
    def get_observation_space(self, env: JobShopGymEnv) -> spaces.Space:
        num_jobs = len(env.env.getJobs())
        num_machines = env.env.getNumMachines()
        max_operations = max(len(job.operations) for job in env.env.getJobs())

        low = np.zeros(num_jobs * max_operations + num_jobs * 3 + num_machines)
        high = np.inf * np.ones(num_jobs * max_operations + num_jobs * 3 + num_machines)

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def get_observation(self, env: JobShopGymEnv) -> np.ndarray:
        state: jobshop.JobShopState = env.env.getState()
        job_progress = np.array(state.jobProgress, copy=False).flatten()
        completed_jobs = np.array(state.completedJobs, dtype=np.float32)
        job_start_times = np.array(state.jobStartTimes, dtype=np.float32)
        machine_availability = np.array(state.machineAvailability, dtype=np.float32)
        next_operation_for_job = np.array(state.nextOperationForJob, dtype=np.float32)

        return np.concatenate([
            job_progress,
            completed_jobs,
            job_start_times,
            machine_availability,
            next_operation_for_job
        ])


class NormalizedObservationSpace(ObservationSpace):
    def get_observation_space(self, env: JobShopGymEnv) -> spaces.Space:
        num_jobs = len(env.env.getJobs())
        num_machines = env.env.getNumMachines()
        max_operations = max(len(job.operations) for job in env.env.getJobs())

        return spaces.Box(low=0, high=1, shape=(num_jobs * max_operations + num_jobs * 3 + num_machines,),
                          dtype=np.float32)

    def get_observation(self, env: JobShopGymEnv) -> np.ndarray:
        state: jobshop.JobShopState = env.env.getState()
        total_time = env.env.getTotalTime()
        max_time = sum(op.duration for job in env.env.getJobs() for op in job.operations)

        job_progress = np.array(state.jobProgress, copy=False).flatten() / max_time
        completed_jobs = np.array(state.completedJobs, dtype=np.float32)
        job_start_times = np.array(state.jobStartTimes, dtype=np.float32) / max_time
        machine_availability = np.array(state.machineAvailability, dtype=np.float32) / max_time
        next_operation_for_job = np.array(state.nextOperationForJob, dtype=np.float32) / max(
            len(job.operations) for job in env.env.getJobs())

        return np.concatenate([
            job_progress,
            completed_jobs,
            job_start_times,
            machine_availability,
            next_operation_for_job
        ])


class RewardFunction(ABC):
    @abstractmethod
    def calculate_reward(self, env: JobShopGymEnv, done: bool) -> float:
        pass


class MakespanRewardFunction(RewardFunction):
    def calculate_reward(self, env: JobShopGymEnv, done: bool) -> float:
        if done:
            return -env.env.getTotalTime()
        return 0


class ProgressRewardFunction(RewardFunction):
    def __init__(self, completion_bonus: float = 1000):
        self.completion_bonus = completion_bonus
        self.last_progress = 0
        self.utilization_weight = 0.5  # Match C++ utilization reward weight

    def calculate_reward(self, env: JobShopGymEnv, done: bool) -> float:
        state = env.env.getState()
        current_progress = sum(state.nextOperationForJob) / sum(len(job.operations) for job in env.env.getJobs())

        # Match C++ reward calculation components
        progress_reward = (current_progress - self.last_progress) * 100
        self.last_progress = current_progress

        # Add machine utilization component to match C++ implementation
        total_time = env.env.getTotalTime()
        if total_time > 0:
            utilization_reward = sum(state.machineAvailability) / (total_time * len(state.machineAvailability))
            utilization_reward *= self.utilization_weight
        else:
            utilization_reward = 0

        if done:
            return progress_reward + self.completion_bonus - total_time + utilization_reward
        return progress_reward + utilization_reward

def validate_schedule(env: jobshop.JobShopEnvironment) -> Dict[str, Any]:
    """
    Validate schedule completeness and correctness.
    Returns metrics dict if valid, raises ValueError if invalid.
    """
    state = env.getState()
    jobs = env.getJobs()

    # Check job completion
    incomplete_jobs = [i for i, completed in enumerate(state.completedJobs) if not completed]
    if incomplete_jobs:
        raise ValueError(f"Incomplete jobs: {incomplete_jobs}")

    # Check operation completion for each job
    schedule = env.getScheduleData()
    scheduled_ops = {(entry.job, entry.operation)
                     for machine_schedule in schedule
                     for entry in machine_schedule}

    for job_idx, job in enumerate(jobs):
        for op_idx in range(len(job.operations)):
            if (job_idx, op_idx) not in scheduled_ops:
                raise ValueError(f"Missing operation: Job {job_idx}, Operation {op_idx}")

    # Verify all scheduled operations completed
    for machine_schedule in schedule:
        for entry in machine_schedule:
            end_time = state.jobProgress(entry.job, entry.operation)
            if end_time == 0:
                raise ValueError(f"Operation not completed: Job {entry.job}, Op {entry.operation}")

    metrics = {
        'makespan': env.getTotalTime(),
        'operations_count': len(scheduled_ops),
        'expected_ops': sum(len(job.operations) for job in jobs),
        'schedule_data': [[{
            'job': e.job,
            'operation': e.operation,
            'machine': e.machine,
            'start': e.start,
            'duration': e.duration,
            'end': e.start + e.duration
        } for e in machine] for machine in schedule]
    }

    # Final sanity checks
    if metrics['operations_count'] != metrics['expected_ops']:
        raise ValueError(f"Operation count mismatch: {metrics['operations_count']} != {metrics['expected_ops']}")

    return metrics


class JobShopGymEnv(gym.Env):
    metadata: Dict[str, List[str]] = {'render.modes': ['human']}

    def __init__(
            self,
            jobshop_env: jobshop.JobShopEnvironment,
            max_steps: int = 200,
            observation_space: ObservationSpace = DefaultObservationSpace(),
            reward_function: RewardFunction = MakespanRewardFunction()
    ):
        super().__init__()
        self.env = jobshop_env
        self.num_jobs = len(self.env.getJobs())
        self.num_machines = self.env.getNumMachines()
        self.max_operations = max(len(job.operations) for job in self.env.getJobs())

        # Action space setup
        self.max_num_actions = self.num_jobs * self.num_machines * self.max_operations
        self.action_space = spaces.Discrete(self.max_num_actions)

        # Observation space setup
        self.observation_space_impl = observation_space
        self.observation_space = self.observation_space_impl.get_observation_space(self)

        # Environment setup
        self.reward_function = reward_function
        self.max_steps = max_steps
        self.action_map = {}
        self._action_mask = None

        # Tracking variables
        self.current_step = 0
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.episode_actions = []

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.reset()
        self.current_step = 0
        self._action_mask = None
        self.episode_actions = []
        obs = self.observation_space_impl.get_observation(self)
        self._update_action_mask()
        return obs, {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Handle invalid actions
        if self._action_mask[action_idx] == 0:
            return self._handle_invalid_action()

        # Execute action
        action = self.action_map[action_idx]
        self.episode_actions.append(action)
        state = self.env.step(action)

        # Get new observation and check termination
        obs = self.observation_space_impl.get_observation(self)
        done = self.env.isDone() or self.current_step >= self.max_steps
        info = self._get_step_info()

        # Calculate reward
        try:
            if done and self.env.isDone():
                # Validate schedule only when environment reports completion
                metrics = validate_schedule(self.env)
                reward = self.reward_function.calculate_reward(self, done)

                # Update best schedule if valid and better
                if metrics['makespan'] < self.best_makespan:
                    self.best_makespan = metrics['makespan']
                    self.best_schedule = self.episode_actions.copy()
            else:
                reward = self.reward_function.calculate_reward(self, done)

        except ValueError as e:
            # Invalid schedule should give large negative reward
            reward = -float('inf')
            info['error'] = str(e)
            done = True

        self._update_action_mask()
        return obs, reward, done, False, info

    def _handle_invalid_action(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = -1  # Penalty for invalid action
        done = self.current_step >= self.max_steps
        obs = self.observation_space_impl.get_observation(self)
        info = {
            'invalid_action': True,
            'makespan': self.env.getTotalTime(),
            'current_step': self.current_step,
            'max_steps': self.max_steps
        }
        return obs, reward, done, False, info

    def _get_step_info(self) -> Dict[str, Any]:
        info = {
            'makespan': self.env.getTotalTime(),
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'completed_jobs': sum(self.env.getState().completedJobs),
            'total_jobs': self.num_jobs
        }
        return info

    def _update_action_mask(self) -> None:
        possible_actions = self.env.getPossibleActions()
        self._action_mask = np.zeros(self.max_num_actions, dtype=np.int8)
        self.action_map.clear()

        for action in possible_actions:
            action_idx = action.job * self.num_machines * self.max_operations + \
                         action.machine * self.max_operations + \
                         action.operation
            self._action_mask[action_idx] = 1
            self.action_map[action_idx] = action

    def get_best_schedule(self) -> Tuple[List[jobshop.Action], float]:
        return self.best_schedule, self.best_makespan

    def action_masks(self) -> np.ndarray:
        return self._action_mask

