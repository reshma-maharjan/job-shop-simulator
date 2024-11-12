from __future__ import annotations
import argparse
from typing import List, Optional, Callable
import jobshop
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from jsp.jsp_env import DefaultObservationSpace, MakespanRewardFunction, JobShopGymEnv, NormalizedObservationSpace, \
    ProgressRewardFunction


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
        jobs, ta_optimal = jobshop.ManualJobShopGenerator.generateFromFile("/home/per/jsp/jsp/environments/doris.json")
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
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Run Job Shop Scheduling experiment with PPO")
    parser.add_argument("--algorithm", choices=["PPO"], default="PPO", help="Algorithm type")
    parser.add_argument("--taillard_instance", default="TA01", choices=[f"TA{i:02d}" for i in range(1, 81)],
                        help="Taillard instance")
    parser.add_argument("--no-gui", action="store_false", help="Disable GUI")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--observation-space", choices=["default", "normalized"], default="normalized",
                        help="Observation space type")
    parser.add_argument("--reward-function", choices=["makespan", "progress"], default="progress",
                        help="Reward function type")
    args: argparse.Namespace = parser.parse_args()

    run_experiment(args.algorithm, args.taillard_instance, not args.no_gui, args.max_steps,
                   args.observation_space, args.reward_function)
