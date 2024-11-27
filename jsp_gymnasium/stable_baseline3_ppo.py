from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from job_shop_env import JobShopGymEnv, Job, Operation
from job_shop_manual_generator import ManualGymGenerator
import json
import os
import argparse
import numpy as np

def read_taillard_instance(file_path):
    """Read and parse a Taillard instance file."""
    with open(file_path, 'r') as f:
        # Read first line
        num_jobs, num_machines = map(int, f.readline().split())
        
        # Read processing times
        processing_times = []
        for _ in range(num_jobs):
            times = list(map(int, f.readline().split()))
            processing_times.append(times)
        
        # Read machine order
        machine_order = []
        for _ in range(num_jobs):
            machines = list(map(int, f.readline().split()))
            # Convert to 0-based indexing
            machines = [m-1 for m in machines]
            machine_order.append(machines)
        
        # Convert to Job and Operation objects
        jobs = []
        for job_id in range(num_jobs):
            operations = []
            for op_id in range(num_machines):
                operation = Operation(
                    duration=processing_times[job_id][op_id],
                    machine=machine_order[job_id][op_id]
                )
                operations.append(operation)
            job = Job(operations=operations)
            jobs.append(job)
            
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
        
        # Add step monitoring
        self.step_history = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Create directories for logs
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        self.log_file = os.path.join(output_dir, 'logs', 'training_log.csv')
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("step,reward,makespan,completed_jobs,total_jobs,done,machine_utilization\n")

    def _on_step(self):
        if len(self.evaluations_results) > 0:
            env = self.training_env.envs[0]
            current_makespan = env.total_time
            completed_jobs = np.sum(env.current_state.completed_jobs)
            total_jobs = len(env.jobs)
            
            # Get the last step information
            if hasattr(env, 'last_step_info'):
                next_state = env.last_step_info.get('next_state')
                reward = env.last_step_info.get('reward', 0)
                done = env.last_step_info.get('done', False)
                info = env.last_step_info.get('info', {})
                
                # Log step information
                self.log_step(
                    step=len(self.step_history),
                    reward=reward,
                    makespan=current_makespan,
                    completed_jobs=completed_jobs,
                    total_jobs=total_jobs,
                    done=done,
                    info=info
                )
                
                # Update episode reward
                self.current_episode_reward += reward
                if done:
                    self.episode_rewards.append(self.current_episode_reward)
                    self.current_episode_reward = 0
            
            print(f"\nTraining Progress:")
            print(f"Completed Jobs: {completed_jobs}/{total_jobs}")
            print(f"Current Makespan: {current_makespan}")
            print(f"Best Makespan: {self.best_makespan}")
            if hasattr(env, 'last_step_info'):
                print(f"Last Reward: {env.last_step_info.get('reward', 0):.2f}")
                print(f"Average Episode Reward: {np.mean(self.episode_rewards):.2f}")
            
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
                self.save_training_plots()
        
        return True
    
    def log_step(self, step, reward, makespan, completed_jobs, total_jobs, done, info):
        """Log step information to file."""
        machine_utilization = np.mean(info.get('machine_utilization', [0]))
        
        # Save to history
        self.step_history.append({
            'step': step,
            'reward': reward,
            'makespan': makespan,
            'completed_jobs': completed_jobs,
            'total_jobs': total_jobs,
            'done': done,
            'machine_utilization': machine_utilization
        })
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"{step},{reward},{makespan},{completed_jobs},{total_jobs},{done},{machine_utilization}\n")

    def save_training_plots(self):
        """Generate and save training visualization plots."""
        if not self.step_history:
            return
            
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot reward history
        steps = [step['step'] for step in self.step_history]
        rewards = [step['reward'] for step in self.step_history]
        ax1.plot(steps, rewards)
        ax1.set_title('Rewards over Time')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward')
        
        # Plot makespan history
        makespans = [step['makespan'] for step in self.step_history]
        ax2.plot(steps, makespans)
        ax2.set_title('Makespan over Time')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Makespan')
        
        # Plot completion rate
        completion_rates = [step['completed_jobs']/step['total_jobs'] 
                          for step in self.step_history]
        ax3.plot(steps, completion_rates)
        ax3.set_title('Job Completion Rate')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Completion Rate')
        
        # Plot machine utilization
        utilizations = [step['machine_utilization'] for step in self.step_history]
        ax4.plot(steps, utilizations)
        ax4.set_title('Machine Utilization')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Utilization')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_progress.png'))
        plt.close()

    def save_solution(self):
        """Save the best solution found."""
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
            'machine_utilization': env.get_machine_utilization(),
            'training_stats': {
                'total_steps': len(self.step_history),
                'average_reward': np.mean([step['reward'] for step in self.step_history]),
                'final_completion_rate': schedule[-1]['job'] / len(env.jobs),
                'average_machine_utilization': np.mean([step['machine_utilization'] 
                                                      for step in self.step_history])
            }
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
        learning_rate=1e-3,    
        n_steps=1024,          
        batch_size=128,        
        n_epochs=5,            
        gamma=0.99,
        ent_coef=0.01,        
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
        env = JobShopGymEnv(jobs=jobs)  # Pass jobs directly
    else:
        env = ManualGymGenerator.create_env_from_file(args.instance_path)
    
    num_jobs = len(env.jobs)
    num_machines = max(op.machine for job in env.jobs for op in job.operations) + 1
    print(f"Environment created with {num_jobs} jobs and {num_machines} machines")
    
    model, best_makespan = train_jssp(env, args.timesteps, args.output_dir)
    print(f"Training completed. Best makespan: {best_makespan}")

if __name__ == "__main__":
    main()