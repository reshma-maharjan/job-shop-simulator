from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

import jobshop
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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

        return spaces.Box(low=0, high=1, shape=(num_jobs * max_operations + num_jobs * 3 + num_machines,), dtype=np.float32)

    def get_observation(self, env: JobShopGymEnv) -> np.ndarray:
        state: jobshop.JobShopState = env.env.getState()
        total_time = env.env.getTotalTime()
        max_time = sum(op.duration for job in env.env.getJobs() for op in job.operations)

        job_progress = np.array(state.jobProgress, copy=False).flatten() / max_time
        completed_jobs = np.array(state.completedJobs, dtype=np.float32)
        job_start_times = np.array(state.jobStartTimes, dtype=np.float32) / max_time
        machine_availability = np.array(state.machineAvailability, dtype=np.float32) / max_time
        next_operation_for_job = np.array(state.nextOperationForJob, dtype=np.float32) / max(len(job.operations) for job in env.env.getJobs())

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

    def calculate_reward(self, env: JobShopGymEnv, done: bool) -> float:
        state = env.env.getState()
        current_progress = sum(state.nextOperationForJob) / sum(len(job.operations) for job in env.env.getJobs())
        progress_reward = (current_progress - self.last_progress) * 100
        self.last_progress = current_progress

        if done:
            return progress_reward + self.completion_bonus - env.env.getTotalTime()
        return progress_reward

class JobShopGymEnv(gym.Env):
    metadata: Dict[str, List[str]] = {'render.modes': ['human']}

    def __init__(self, jobshop_env: jobshop.JobShopEnvironment, max_steps: int = 200,
                 observation_space: ObservationSpace = DefaultObservationSpace(),
                 reward_function: RewardFunction = MakespanRewardFunction()):
        super().__init__()
        self.env: jobshop.JobShopEnvironment = jobshop_env
        self.num_jobs: int = len(self.env.getJobs())
        self.num_machines: int = self.env.getNumMachines()
        self.max_operations: int = max(len(job.operations) for job in self.env.getJobs())
        self.max_num_actions: int = self.num_jobs * self.num_machines * self.max_operations
        self.action_space: spaces.Discrete = spaces.Discrete(self.max_num_actions)

        self.observation_space_impl = observation_space
        self.observation_space = self.observation_space_impl.get_observation_space(self)
        self.reward_function = reward_function

        self.action_indices = np.array([
            [job * self.num_machines * self.max_operations + machine * self.max_operations
             for machine in range(self.num_machines)]
            for job in range(self.num_jobs)
        ])

        self.action_map: Dict[int, jobshop.Action] = {}
        self.use_masking: bool = True
        self._action_mask: Optional[np.ndarray] = None
        self.max_steps: int = max_steps
        self.current_step: int = 0

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.reset()
        self.current_step = 0
        obs: np.ndarray = self.observation_space_impl.get_observation(self)
        self._update_action_mask()
        return obs, {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        if self._action_mask[action_idx] == 0:
            reward: float = -1
            done: bool = self.current_step >= self.max_steps
            obs: np.ndarray = self.observation_space_impl.get_observation(self)
            info: Dict[str, Any] = {'invalid_action': True, 'makespan': self.env.getTotalTime()}

            if done:
                info["schedule_data"] = self.env.getScheduleData()
                info["isDone"] = True

            return obs, reward, done, False, info

        action: jobshop.Action = self.action_map[action_idx]
        self.env.step(action)
        done: bool = self.env.isDone() or self.current_step >= self.max_steps
        obs: np.ndarray = self.observation_space_impl.get_observation(self)
        makespan: int = self.env.getTotalTime()
        info: Dict[str, Any] = {'makespan': makespan}
        if done:
            info["schedule_data"] = self.env.getScheduleData()
            info["isDone"] = self.env.isDone()

        reward: float = self.reward_function.calculate_reward(self, done)
        self._update_action_mask()

        return obs, reward, done, False, info

    def _update_action_mask(self) -> None:
        possible_actions: List[jobshop.Action] = self.env.getPossibleActions()
        self._action_mask = np.zeros(self.max_num_actions, dtype=np.int8)
        self.action_map.clear()
        for action in possible_actions:
            action_idx: int = self._action_to_index(action)
            self._action_mask[action_idx] = 1
            self.action_map[action_idx] = action

    def action_masks(self) -> np.ndarray:
        return self._action_mask

    def _action_to_index(self, action: jobshop.Action) -> int:
        return self.action_indices[action.job, action.machine] + action.operation

    def get_jobshop_env(self) -> jobshop.JobShopEnvironment:
        return self.env

class MakespanCallback(BaseCallback):
    def __init__(self, verbose: int = 0, plotter: Optional[jobshop.LivePlotter] = None):
        super().__init__(verbose)
        self.best_makespan: float = float('inf')
        self.plotter = plotter
        self.best_schedule_data: Optional[List[jobshop.ScheduleEntry]] = None
        self.episode_count: int = 0
        self.episode_reward: float = 0
        self.episode_length: int = 0

    def _on_step(self) -> bool:
        self.episode_length += 1
        self.episode_reward += self.locals['rewards'][0]

        if self.locals['dones'][0]:
            self._on_episode_end()

        return True

    def _on_episode_end(self) -> None:
        self.episode_count += 1
        info = self.locals['infos'][0]
        current_makespan: float = info['makespan']
        isDone: bool = info.get('isDone', False)

        if current_makespan < self.best_makespan and isDone:
            self.best_makespan = current_makespan
            self.best_schedule_data = info['schedule_data']

            if self.plotter:
                self.plotter.updateSchedule(self.best_schedule_data, current_makespan)
                for _ in range(10):
                    self.plotter.render()

        self.logger.record("jobshop/best_makespan", self.best_makespan)
        self.logger.record("jobshop/episode_reward", self.episode_reward)
        self.logger.record("jobshop/episode_length", self.episode_length)
        self.logger.record("jobshop/episode_makespan", current_makespan)

        self.episode_reward = 0
        self.episode_length = 0

    def _on_training_end(self) -> None:
        print(f"\nTraining completed.")
        print(f"Total episodes: {self.episode_count}")
        print(f"Best makespan achieved: {self.best_makespan}")

        if self.best_schedule_data:
            print("\nBest Schedule:")
            jobshop_env = self.training_env.env_method("get_jobshop_env")[0]
            jobshop_env.printSchedule()
        else:
            print("No best schedule data available.")
      

def run_experiment(algorithm_name: str, taillard_instance: str, use_gui: bool, max_steps: int,
                   observation_space: str, reward_function: str) -> None:
    def make_env() -> Callable[[], gym.Env]:
    
        #instance: jobshop.TaillardInstance = getattr(jobshop.TaillardInstance, taillard_instance)
        jobs, ta_optimal = jobshop.ManualJobShopGenerator.generateFromFile("/workspaces/job-shop-simulator/jsp/environments/doris.csv")

        print(jobs, ta_optimal)
        #jobs, ta_optimal = jobshop.TaillardJobShopGenerator.loadProblem(instance, True)
        print(f"Optimal makespan for {taillard_instance}: {ta_optimal}")

        obs_space = DefaultObservationSpace() if observation_space == "default" else NormalizedObservationSpace()
        reward_func = MakespanRewardFunction() if reward_function == "makespan" else ProgressRewardFunction()

        return lambda: JobShopGymEnv(jobshop.JobShopEnvironment(jobs), max_steps, obs_space, reward_func)

    env: DummyVecEnv = DummyVecEnv([make_env()])
    model: MaskablePPO = MaskablePPO('MlpPolicy', env, verbose=1)
    total_timesteps: int = 100000

    plotter: Optional[jobshop.LivePlotter] = None
    if use_gui:
        jobshop_env = env.env_method("get_jobshop_env")[0]
        plotter = jobshop.LivePlotter(jobshop_env.getNumMachines())

    makespan_callback: MakespanCallback = MakespanCallback(plotter=plotter)
    model.learn(total_timesteps=total_timesteps, callback=makespan_callback)

    print(f"Best makespan achieved: {makespan_callback.best_makespan}")
    #print(f"Optimal makespan: {ta_optimal}")
    #print(f"Gap: {(makespan_callback.best_makespan - ta_optimal) / ta_optimal * 100:.2f}%")

if __name__ == "__main__":
    print(":D")
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Run Job Shop Scheduling experiment with PPO")
    parser.add_argument("--algorithm", choices=["PPO"],  default="PPO", help="Algorithm type")
    parser.add_argument("--taillard_instance", default="TA01", choices=[f"TA{i:02d}" for i in range(1, 81)], help="Taillard instance")
    parser.add_argument("--no-gui", action="store_false", help="Disable GUI")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--observation-space", choices=["default", "normalized"], default="normalized", help="Observation space type")
    parser.add_argument("--reward-function", choices=["makespan", "progress"], default="progress", help="Reward function type")
    args: argparse.Namespace = parser.parse_args()

    run_experiment(args.algorithm, args.taillard_instance, not args.no_gui, args.max_steps,
                   args.observation_space, args.reward_function)