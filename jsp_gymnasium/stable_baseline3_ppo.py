from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from job_shop_env import JobShopGymEnv, Job, Operation
from job_shop_manual_generator import ManualGymGenerator
import json
import os
import argparse

def read_taillard_instance(filepath: str) -> list[Job]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []
    current_line = 1
    
    for i in range(n_jobs):
        machines = list(map(int, lines[current_line].split()))
        times = list(map(int, lines[current_line + 1].split()))
        operations = [Operation(duration=d, machine=m) for m, d in zip(machines, times)]
        jobs.append(Job(operations=operations))
        current_line += 2
    return jobs

class JSPCallback(EvalCallback):
    def __init__(self, eval_env, output_dir, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.best_makespan = float('inf')
        self.output_dir = output_dir
        self.solution_history = []
        self.last_makespan = float('inf')
        self.no_improvement_count = 0
        self.min_makespan_seen = float('inf')
        self.best_complete_solutions = []

    def _on_step(self):
        if len(self.evaluations_results) > 0:
            env = self.training_env.envs[0]
            current_makespan = env.total_time
            completed_jobs = np.sum(env.current_state.completed_jobs)
            total_jobs = len(env.jobs)
            
            print(f"\nTraining Progress:")
            print(f"Completed Jobs: {completed_jobs}/{total_jobs}")
            print(f"Current Makespan: {current_makespan}")
            print(f"Best Makespan: {self.best_makespan}")
            
            if completed_jobs == total_jobs:
                if current_makespan < self.best_makespan:
                    self.best_makespan = current_makespan
                    self.save_solution()
                    print(f"\nNew Best Complete Solution Found!")
                    print(f"Makespan: {self.best_makespan}")
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            
            if self.no_improvement_count >= 5:
                print("\nWarning: No improvement in last 5 evaluations")
        
        return True
    
    def save_solution(self):
        env = self.training_env.envs[0]
        schedule = []
        for action in env.action_history:
            op = env.jobs[action.job].operations[action.operation]
            end_time = env.current_state.job_progress[action.job, action.operation]
            start_time = end_time - op.duration
            schedule.append({
                'job': action.job,
                'operation': action.operation,
                'machine': action.machine,
                'start_time': int(start_time),
                'end_time': int(end_time),
                'duration': op.duration
            })

        solution = {
            'makespan': float(self.best_makespan),
            'schedule': schedule,
            'machine_utilization': env.get_machine_utilization()
        }

        with open(os.path.join(self.output_dir, 'best_solution.json'), 'w') as f:
            json.dump(solution, f, indent=2)

def train_jssp(env, total_timesteps=100000, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    
    env = Monitor(env)
    eval_env = Monitor(env)

    callback = JSPCallback(
        eval_env=eval_env,
        output_dir=output_dir,
        eval_freq=1000,
        best_model_save_path=output_dir,
        deterministic=True
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,    # Increased learning rate
        n_steps=1024,          # Reduced steps
        batch_size=128,        # Increased batch size
        n_epochs=5,            # Increased epochs
        gamma=0.99,
        ent_coef=0.01,        # Added entropy coefficient
        clip_range=0.2,
        verbose=1
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    return model, callback.best_makespan

def main():
    parser = argparse.ArgumentParser(description='Train JSSP with PPO')
    parser.add_argument('--instance_type', choices=['taillard', 'manual'], required=True)
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    if args.instance_type == 'taillard':
        jobs = read_taillard_instance(args.instance_path)
        env = JobShopGymEnv(jobs)
    else:
        env = ManualGymGenerator.create_env_from_file(args.instance_path)
    
    print(f"Environment created with {len(env.jobs)} jobs and {env.num_machines} machines")
    model, best_makespan = train_jssp(env, args.timesteps, args.output_dir)
    print(f"Training completed. Best makespan: {best_makespan}")

if __name__ == "__main__":
    main()