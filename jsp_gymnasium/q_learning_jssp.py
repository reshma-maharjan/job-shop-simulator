import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from Job_shop_taillard_generator import TaillardGymGenerator
import wandb
import logging
from pathlib import Path
import json
import random
import time
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class QLearningAgent:
    """Q-learning agent for Job Shop Scheduling with improved stability"""
    
    def __init__(self, env, config):
        self.env = env
        self.env_unwrapped = env.unwrapped
        
        self.config = config
        
        # Initialize Q-table with small random values for better exploration
        self.q_table = np.random.uniform(0, 0.1, 
                                       (config['obs_space_size'], 
                                        config['action_space_size']))
        
        # Modified exploration parameters for better stability
        self.epsilon = config['epsilon_init']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        
        # Training variables
        self.total_timesteps = 0
        self.best_makespan = float('inf')
        self.best_actions = []
        self.episode_rewards = []
        self.episode_makespans = []
        self.current_episode = 0
        
        # Reference makespan for reward scaling
        self.reference_makespan = None
        
        # Track previous makespans for adaptive exploration
        self.previous_makespans = []
        self.makespan_window = 5  # Window size for tracking improvement

        logger.info("Improved Q-Learning Agent initialized")

    def get_next_action(self, state):
        """Select next action using epsilon-greedy with validity checking"""
        valid_actions = self.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # Adaptive exploration based on recent performance
        if len(self.previous_makespans) >= self.makespan_window:
            recent_trend = np.mean(np.diff(self.previous_makespans[-self.makespan_window:]))
            if recent_trend > 0:  # If makespan is increasing
                self.epsilon = min(self.epsilon * 1.1, 0.9)  # Increase exploration
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_idx = self.state_to_index(state)
        q_vals = self.q_table[state_idx]
        masked_q_vals = np.full_like(q_vals, -np.inf)
        masked_q_vals[valid_actions] = q_vals[valid_actions]
        
        return valid_actions[np.argmax(q_vals[valid_actions])]

    def state_to_index(self, state):
        """Improved state representation for Q-table indexing"""
        # Focus on key state features
        machine_times = state[:self.config['n_machines']]
        job_status = state[self.config['n_machines']:self.config['n_machines'] + self.config['n_jobs']]
        
        # Normalize and discretize state components
        normalized_times = machine_times / (np.max(machine_times) + 1e-8)
        normalized_status = job_status / (np.max(job_status) + 1e-8)
        
        # Combine state components into a single hash
        state_tuple = tuple(np.round(normalized_times, 2)) + tuple(np.round(normalized_status, 2))
        return hash(state_tuple) % self.config['obs_space_size']

    def get_valid_actions(self) -> list:
        """Get list of valid actions from environment"""
        valid_actions = []
        for action_id in range(self.config['action_space_size']):
            action = self.env_unwrapped._decode_action(action_id)
            if self.env_unwrapped._is_valid_action(action):
                valid_actions.append(action_id)
        return valid_actions

    def calculate_reward(self, info: dict, step_count: int) -> float:
        """Improved reward calculation with better scaling and incentives"""
        try:
            reward = 0.0
            
            # Get current makespan
            current_makespan = info.get('makespan', 0)
            if isinstance(current_makespan, (list, tuple)):
                current_makespan = max(current_makespan)
            
            # Initialize reference makespan if not set
            if self.reference_makespan is None and current_makespan > 0:
                self.reference_makespan = current_makespan
            
            # Penalty for invalid actions (severe)
            if not info.get('valid_action', True):
                return -50.0
            
            # Progressive reward for job completion
            completed_jobs = info.get('completed_jobs', 0)
            total_jobs = len(self.env.unwrapped.jobs)
            completion_ratio = completed_jobs / total_jobs
            reward += completion_ratio * 20.0
            
            # Makespan improvement reward
            if self.reference_makespan and current_makespan > 0:
                makespan_ratio = self.reference_makespan / current_makespan
                reward += (makespan_ratio - 1) * 30.0  # Reward improvement from reference
            
            # Early completion bonus
            if completed_jobs == total_jobs:
                time_factor = 1.0 - (step_count / 1000)  # Bonus for finishing early
                reward += 50.0 * time_factor
            
            # Machine utilization incentive
            if 'machine_utilization' in info:
                utilization = info['machine_utilization']
                if isinstance(utilization, (list, tuple)):
                    utilization = np.mean(utilization)
                reward += utilization * 10.0
            
            # Clip reward to prevent extreme values
            return float(np.clip(reward, -50.0, 50.0))
            
        except Exception as e:
            logging.error(f"Error in reward calculation: {e}")
            return 0.0

    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using standard Q-learning with stability measures"""
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        # Get current Q-value
        current_q = self.q_table[state_idx, action]
        
        # Calculate target Q-value with double Q-learning principle
        if not done:
            best_action = np.argmax(self.q_table[next_state_idx])
            next_q = self.q_table[next_state_idx, best_action]
        else:
            next_q = 0
        
        # Update Q-value with learning rate decay
        target = reward + (1 - done) * self.gamma * next_q
        new_q = current_q + self.learning_rate * (target - current_q)
        
        # Apply soft update
        self.q_table[state_idx, action] = new_q

    def update_epsilon(self):
        """Adaptive exploration rate update"""
        if len(self.previous_makespans) >= 2:
            if self.previous_makespans[-1] <= self.previous_makespans[-2]:
                # If improving, decay epsilon normally
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            else:
                # If getting worse, decay epsilon slower
                self.epsilon = max(self.epsilon_min, self.epsilon * np.sqrt(self.epsilon_decay))

    def rollout(self):
        """Execute one episode with improved stability measures"""
        try:
            cur_obs, _ = self.env.reset()
            episode_reward = 0
            episode_actions = []
            step_count = 0
            max_steps = 1000
            
            while step_count < max_steps:
                # Select and execute action
                action = self.get_next_action(cur_obs)
                if action is None:
                    break
                
                next_obs, _, terminated, truncated, info = self.env.step(action)
                
                # Calculate reward
                reward = self.calculate_reward(info, step_count)
                
                # Update Q-values
                self.update_q_value(cur_obs, action, reward, next_obs, 
                                  terminated or truncated)
                
                # Update statistics
                episode_reward += reward
                episode_actions.append(action)
                self.total_timesteps += 1
                step_count += 1
                
                # Update exploration rate
                self.update_epsilon()
                
                # Update current observation
                cur_obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Track episode results
            current_makespan = info.get('makespan', float('inf'))
            if isinstance(current_makespan, (list, tuple)):
                current_makespan = max(current_makespan)
            
            self.episode_makespans.append(current_makespan)
            self.previous_makespans.append(current_makespan)
            
            # Update best solution
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_actions = episode_actions.copy()
                # Update reference makespan for relative improvements
                self.reference_makespan = min(self.reference_makespan, current_makespan)
            
            return episode_reward, episode_actions, current_makespan
            
        except Exception as e:
            logging.error(f"Error in rollout: {str(e)}")
            raise

    def learn(self, nr_of_episodes: int) -> tuple:
        """Main training loop with improved logging and stability tracking"""
        logger.info(f"Starting training for {nr_of_episodes} episodes...")
        
        for episode in tqdm(range(nr_of_episodes)):
            self.current_episode = episode
            episode_reward, _, episode_makespan = self.rollout()
            self.episode_rewards.append(episode_reward)
            
            # Log detailed metrics
            if wandb.run is not None:
                wandb.log({
                    'episode_vs_makespan/current': episode_makespan,
                    'episode_vs_makespan/best_so_far': min(self.episode_makespans),
                    'episode_vs_makespan/average': np.mean(self.episode_makespans[-10:]),
                    'training/epsilon': self.epsilon,
                    'training/reward': episode_reward,
                    'episode': episode
                })
            
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_makespan = np.mean(self.episode_makespans[-10:])
                logger.info(
                    f"Episode {episode + 1}/{nr_of_episodes}, "
                    f"Mean Reward: {mean_reward:.2f}, "
                    f"Mean Makespan: {mean_makespan:.2f}, "
                    f"Best Makespan: {self.best_makespan}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )
        
        return self.best_actions, self.best_makespan

def run_job_shop_scheduling(
    instance_name,
    episodes=500,
    seed=42,
    enable_wandb=True,
    verbose=False
):
    """Run job shop scheduling with improved Q-learning"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Create environment
        logging.info(f"Creating environment for instance: {instance_name}")
        if instance_name.startswith('ta'):
            base_env = TaillardGymGenerator.create_env_from_instance(instance_name)
            n_jobs = len(base_env.jobs)
            n_machines = base_env.num_machines
            
            register(
                id='JobShop-Taillard-v0',
                entry_point='job_shop_env:JobShopGymEnv',
                kwargs={'jobs': base_env.jobs}
            )
            env = gym.make('JobShop-Taillard-v0')
        else:
            raise ValueError("Currently only Taillard instances are supported")
        
        # Create configuration with tuned parameters
        config = {
            'env_name': 'job-shop',
            'algorithm': 'Q-learning',
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'action_space_size': env.action_space.n,
            'obs_space_size': env.observation_space.shape[0],
            'gamma': 0.99,  # Future reward discount
            'learning_rate': 0.05,  # Reduced for stability
            'epsilon_init': 1.0,
            'epsilon_min': 0.1,  # Increased minimum exploration
            'epsilon_decay': 0.995,  # Slower decay
            'seed': seed
        }
        
        # Initialize wandb if enabled
        if enable_wandb:
            wandb.init(
                project="job-shop-qlearning",
                config=config,
                name=f"QL_{instance_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Create and train agent
        agent = QLearningAgent(env, config)
        best_actions, best_makespan = agent.learn(episodes)
        
        # Save results
        results = {
            'instance': instance_name,
            'makespan': float(best_makespan),
            'num_actions': len(best_actions),
            'actions': [int(a) for a in best_actions],
            'final_epsilon': float(agent.epsilon),
            'makespan_history': [float(m) for m in agent.episode_makespans]
        }
        
        if enable_wandb:
            wandb.log({"final_results": results})
            wandb.finish()
        
        return results
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        if enable_wandb:
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    try:
        wandb.login()
        
        results = run_job_shop_scheduling(
            instance_name="ta01.txt",
            episodes=20,
            seed=42,
            enable_wandb=True,
            verbose=True
        )
        
        print(f"\nExecution completed successfully!")
        print(f"Best makespan: {results['makespan']}")
        
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        raise