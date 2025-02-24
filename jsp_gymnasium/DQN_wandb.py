# job_shop_DQN.py
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import logging
import wandb
from Job_shop_taillard_generator import TaillardGymGenerator
from pathlib import Path
import json
import time
from job_shop_env import JobShopGymEnv


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Wandb initialization
wandb.login(key="22a8b69ab5255b120ca37b40c2f998f71db3c615")

def init_wandb(config, instance_name):
    wandb.init(
        project="jssp_DQN",
        entity="reshma-stha2016",  # Fixed: Using your actual username
        name=f"DQN_{instance_name}_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "DQN",
            "instance": instance_name,
            "problem_size": {
                "n_jobs": config.get('n_jobs'),
                "n_machines": config.get('n_machines')
            },
            "model_config": {
                "memory_size": config['memory_size'],
                "batch_size": config['batch_size'],
                "gamma": config['gamma'],
                "epsilon_start": config['epsilon_start'],
                "epsilon_min": config['epsilon_min'],
                "epsilon_decay": config['epsilon_decay'],
                "learning_rate": config['learning_rate']
            }
        },
        tags=["DQN", instance_name, "job-shop"]
    )

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

def setup_result_directory(instance_name):
    # Define the base directory to store results
    base_dir = Path('./results')
    
    # Create the directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for the instance
    instance_dir = base_dir / instance_name
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the result file path
    result_file = instance_dir / f'results_{time.strftime("%Y-%m-%d_%H-%M-%S")}.json'

    logger.info(f"Creating base directory: {base_dir}")
    logger.info(f"Creating instance directory: {instance_dir}")
    
    return result_file

class DQN:
    def __init__(self, env, policy_class, config):
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.policy = policy_class(self.obs_dim, self.action_dim).to(self.device)
        self.target_policy = policy_class(self.obs_dim, self.action_dim).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        self.best_makespan = float('inf')
        self.best_actions = []

    def normalize_observation(self, obs):
        """Normalize observation to be within the observation space."""
        return np.clip(obs.astype(np.float32), 
                      self.env.observation_space.low, 
                      self.env.observation_space.high)

    def remember(self, state, action, reward, next_state, done):
        # Ensure states are normalized and float32
        state = self.normalize_observation(state)
        next_state = self.normalize_observation(next_state)
        self.memory.append(Transition(state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state = self.normalize_observation(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy(state_tensor)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor([not done for done in batch.done], 
                                    dtype=torch.bool, device=self.device)
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # Convert list of next_states to numpy array first
        non_final_next_states = np.array([s for s, d in zip(batch.next_state, batch.done) if not d])
        non_final_next_states = torch.FloatTensor(non_final_next_states).to(self.device)
        
        current_q_values = self.policy(state_batch).gather(1, action_batch.unsqueeze(1))
        
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        if len(non_final_next_states) > 0:  # Only process if there are any non-final states
            with torch.no_grad():
                next_q_values[non_final_mask] = self.target_policy(non_final_next_states).max(1)[0]
            
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_policy.load_state_dict(self.policy.state_dict())

    def learn(self, nr_of_episodes=1):# before 250
        logger.info(f"Starting training for {nr_of_episodes} episodes...")
        logger.info(f"Problem size: {len(self.env_unwrapped.jobs)} jobs, {self.env_unwrapped.num_machines} machines")
        
        for episode in range(nr_of_episodes):
            state, _ = self.env.reset(seed=None, options=None) 
            
            state = self.normalize_observation(state)
            episode_reward = 0
            episode_actions = []
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = self.normalize_observation(next_state)
                
                episode_actions.append(action)
                episode_reward += reward
                
                self.remember(state, action, reward, next_state, done)
                self.replay()
                
                if done or truncated:
                    current_makespan = info.get('makespan', float('inf'))
                    if current_makespan < self.best_makespan:
                        self.best_makespan = current_makespan
                        self.best_actions = episode_actions.copy()
                        logger.info(f"New best makespan found: {self.best_makespan}")
                
                state = next_state
                
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if episode % 10 == 0:
                self.update_target_network()
                logger.info(f"Episode {episode}/{nr_of_episodes}, "
                          f"Reward: {episode_reward:.2f}, "
                          f"Epsilon: {self.epsilon:.3f}, "
                          f"Best Makespan: {self.best_makespan}")
        
        logger.info("Training completed!")
        logger.info(f"Best makespan achieved: {self.best_makespan}")
        return self.best_actions, self.best_makespan

def main():
    instance_name = 'ta31'
    base_env = TaillardGymGenerator.create_env_from_instance(f'{instance_name}.txt')
    
    # Register environment
    register(
        id='JobShop-Taillard-v0',
        entry_point='job_shop_env:JobShopGymEnv',
        kwargs={'jobs': base_env.jobs}
    )

    env = gym.make('JobShop-Taillard-v0')
    
    # DQN configuration
    config = {
        'memory_size': 1000,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 0.001,
        'n_jobs': len(base_env.jobs),
        'n_machines': base_env.num_machines
    }
    
    # Initialize wandb with your configuration
    init_wandb(config, instance_name)
    
    try:
        # Create and train agent
        agent = DQN(env, Policy, config)
        
        # Create artifact to store the model
        model_artifact = wandb.Artifact(
            f"model_{instance_name}", 
            type="model",
            description=f"DQN model for {instance_name}"
        )
        
        # Train the agent
        best_actions, best_makespan = agent.learn(nr_of_episodes=1)
        
        # Save final model
        model_path = f'model_{instance_name}.pt'
        torch.save({
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'best_makespan': best_makespan,
            'config': config
        }, model_path)
        
        # Log model as artifact
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
        
        # Execute best solution
        #env.reset(seed=None, options=None)
        env.reset(seed=None)
        schedule_data = []
        
        for action_id in best_actions:
            obs, reward, done, truncated, info = env.step(action_id)
            action = env.unwrapped._decode_action(action_id)
            
            schedule_data.append({
                'job': int(action.job),
                'machine': int(action.machine),
                'operation': int(action.operation),
                'start': int(info.get('start_time', 0)),
                'duration': int(env.unwrapped.jobs[action.job].operations[action.operation].duration),
                'end': int(info.get('end_time', 0))
            })
        
                # Create a wandb Table for machine utilization
        machine_utilization_table = wandb.Table(columns=["Machine", "Utilization"])
        machine_utilization = env.unwrapped.get_machine_utilization()

        # Populate the table with machine utilization data
        for i, u in enumerate(machine_utilization):
            machine_utilization_table.add_data(i, float(u))

        # Log the table along with other results
        wandb.log({
            "final_results": {
                "best_makespan": best_makespan,
                "number_of_actions": len(best_actions),
                # "schedule": schedule_data,
                # "machine_utilization": machine_utilization_table
            }
        })
        
        # Log a summary metric for machine_utilization (average or some other summary)
        average_utilization = sum(machine_utilization) / len(machine_utilization) if machine_utilization else 0
        wandb.log({"final_results.machine_utilization_avg": average_utilization})


        # Save schedule data as artifact
        schedule_artifact = wandb.Artifact(
            f"schedule_{instance_name}",
            type="schedule",
            description=f"Best schedule for {instance_name}"
        )
        
        schedule_path = f'schedule_{instance_name}.json'
        with open(schedule_path, 'w') as f:
            json.dump(schedule_data, f)
        schedule_artifact.add_file(schedule_path)
        wandb.log_artifact(schedule_artifact)
        
        # Log summary metrics
        wandb.run.summary["best_makespan"] = best_makespan
        wandb.run.summary["final_memory_size"] = len(agent.memory)
        wandb.run.summary["total_actions"] = len(best_actions)
        
        print(f"\nTraining completed. Results logged to W&B.")
        print(f"Best makespan: {best_makespan}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        wandb.run.summary["error"] = str(e)
        raise e
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()