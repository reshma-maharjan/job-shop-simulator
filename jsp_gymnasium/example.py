import numpy as np
from typing import List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from job_shop_env import Job, Operation, JobShopGymEnv

class MakespanCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_makespan = float('inf')

    def _on_step(self) -> bool:
        result = super()._on_step()
        if len(self.evaluations_results):
            latest_makespan = self.evaluations_results["makespan"][-1]
            if latest_makespan < self.best_makespan:
                self.best_makespan = latest_makespan
                print(f"\nNew Best Makespan: {self.best_makespan}")
        return result

def read_taillard_instance(filepath: str) -> list[Job]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []
    current_line = 1
    
    for i in range(n_jobs):
        machines = list(map(int, lines[current_line].split()))
        times = list(map(int, lines[current_line + 1].split()))
        
        operations = []
        for machine, duration in zip(machines, times):
            operations.append(Operation(duration=duration, machine=machine))
        
        jobs.append(Job(operations=operations))
        current_line += 2
        
    return jobs

# Load instance
jobs = read_taillard_instance('/workspaces/job-shop-simulator/per_jsp/data/taillard_instances/ta01.txt')

# Create and train model
env = JobShopGymEnv(jobs)
eval_env = Monitor(JobShopGymEnv(jobs))

eval_callback = MakespanCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    learning_rate=1e-4,
    n_steps=1024,        # Reduced from 2048
    batch_size=64,       # Reduced from 128
    n_epochs=5,          # Reduced from 10
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

model.learn(
    print("Training model..."),
    total_timesteps=50000,  # Reduced from 2000000
    callback=eval_callback
)

print(f"\nFinal Best Makespan: {eval_callback.best_makespan}")
model.save("ta01_ppo")