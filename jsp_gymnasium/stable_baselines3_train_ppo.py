from stable_baselines3 import PPO, DQN, A2C, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from job_shop_env import JobShopGymEnv, Job, Operation
import argparse
import os
import json
import time
import logging

class TrainingMonitorCallback(EvalCallback):
    def __init__(self, eval_env, total_timesteps, output_dir, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.start_time = time.time()
        self.last_print = 0
        self.print_interval = 1000
        self.total_timesteps = total_timesteps
        self.best_makespan = float('inf')
        self.best_solution = None
        self.output_dir = output_dir

    def _on_step(self) -> bool:
        if self.n_calls - self.last_print >= self.print_interval:
            elapsed = time.time() - self.start_time
            current_makespan = self.training_env.get_attr('total_time')[0]
            print(f"\nIteration {self.n_calls}/{self.total_timesteps}")
            print(f"Current makespan: {current_makespan}")
            print(f"Best makespan so far: {self.best_makespan}")
            print(f"Elapsed time: {elapsed:.1f}s")
            
            if hasattr(self, 'best_mean_reward'):
                eval_makespan = -self.best_mean_reward
                if eval_makespan < self.best_makespan:
                    self.best_makespan = eval_makespan
                    self.best_solution = self.training_env.get_attr('action_history')[0]
                    self.save_best_solution()
            self.last_print = self.n_calls
        return True

    def save_best_solution(self):
        if self.best_solution:
            solution_data = {
                'makespan': float(self.best_makespan),
                'schedule': [(int(action.job), int(action.operation), int(action.machine)) 
                           for action in self.best_solution]
            }
            solution_path = os.path.join(self.output_dir, 'best_solution.json')
            with open(solution_path, 'w') as f:
                json.dump(solution_data, f, indent=2)

def get_model_class(model_name):
    models = {
        'PPO': PPO,
        'DQN': DQN,
        'A2C': A2C,
        'TD3': TD3,
        'SAC': SAC
    }
    return models.get(model_name.upper())

def train_model(env, model_name, total_timesteps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    env = Monitor(env)
    eval_env = Monitor(JobShopGymEnv(env.jobs))
    
    monitor_callback = TrainingMonitorCallback(
        eval_env,
        total_timesteps=total_timesteps,
        output_dir=output_dir,
        best_model_save_path=f"{output_dir}/logs/",
        log_path=f"{output_dir}/logs/",
        eval_freq=1000,
        deterministic=True
    )
    
    ModelClass = get_model_class(model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Model specific parameters
    model_params = {
        'PPO': {
            'learning_rate': 1e-4,
            'n_steps': 2048,
            'batch_size': 128,
            'gamma': 0.99,
        },
        'DQN': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'gamma': 0.99,
        },
        'A2C': {
            'learning_rate': 1e-4,
            'n_steps': 5,
            'gamma': 0.99,
        },
        'TD3': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'gamma': 0.99,
        },
        'SAC': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'gamma': 0.99,
        }
    }
    
    model = ModelClass(
        "MlpPolicy",
        env,
        verbose=1,
        **model_params[model_name.upper()]
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=monitor_callback,
        progress_bar=True
    )
    
    model_path = f"{output_dir}/final_model.zip"
    model.save(model_path)
    return model, monitor_callback.best_makespan, monitor_callback.best_solution

def main():
    parser = argparse.ArgumentParser(description='Train JSSP with different RL algorithms')
    parser.add_argument('--model', choices=['PPO', 'DQN', 'A2C', 'TD3', 'SAC'], required=True)
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='./training_output')
    args = parser.parse_args()

    try:
        jobs = read_taillard_instance(args.instance_path)
        env = JobShopGymEnv(jobs)
        
        model, best_makespan, best_solution = train_model(
            env,
            args.model,
            args.timesteps,
            f"{args.output_dir}_{args.model}"
        )
        
        print(f"\nTraining completed for {args.model}!")
        print(f"Best makespan: {best_makespan}")
        print(f"Results saved in: {args.output_dir}_{args.model}")

    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()