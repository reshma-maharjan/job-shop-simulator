import numpy as np
import gymnasium as gym
import logging
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DoubleQLearningCartPole:
    """
    Double Q-Learning with Experience Replay for CartPole
    """
    def __init__(self,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 episodes: int = 5000,
                 batch_size: int = 32,
                 memory_size: int = 50000):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.001
        self.exploration_decay = 0.997
        self.episodes = episodes
        self.batch_size = batch_size
        
        # Initialize two Q-tables for double Q-learning
        self.q_table_1 = {}
        self.q_table_2 = {}
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # State discretization parameters - different bins for different dimensions
        self.n_bins = {
            'position': 12,
            'velocity': 12,
            'angle': 20,    # More bins for angle (crucial for balancing)
            'angular_velocity': 16
        }
        
        # State space bounds
        self.state_bounds = {
            'position': (-4.8, 4.8),
            'velocity': (-10, 10),
            'angle': (-0.418, 0.418),
            'angular_velocity': (-10, 10)
        }
        
    def discretize_state(self, state: np.ndarray) -> tuple:
        """Convert continuous state to discrete state with variable bins"""
        bounds = list(self.state_bounds.values())
        n_bins_list = [self.n_bins[key] for key in ['position', 'velocity', 'angle', 'angular_velocity']]
        discretized = []
        
        for i, (val, (lower, upper), bins) in enumerate(zip(state, bounds, n_bins_list)):
            scaled = (val - lower) / (upper - lower)
            scaled = np.clip(scaled, 0, 1)
            bin_idx = int(scaled * (bins - 1))
            discretized.append(bin_idx)
            
        return tuple(discretized)
    
    def get_q_values(self, state_key: tuple, q_table: dict) -> np.ndarray:
        """Get Q-values for all actions in a given state"""
        return np.array([q_table.get((state_key, a), 0.0) for a in [0, 1]])
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy with combined Q-values"""
        if np.random.random() < self.exploration_rate:
            return np.random.choice([0, 1])
        
        state_key = self.discretize_state(state)
        # Combine Q-values from both tables
        combined_q = (self.get_q_values(state_key, self.q_table_1) + 
                     self.get_q_values(state_key, self.q_table_2)) / 2
        return np.argmax(combined_q)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def calculate_reward(self, state: np.ndarray, done: bool) -> float:
        """Calculate a more informative reward"""
        x, _, theta, _ = state
        
        # Position penalty
        position_penalty = abs(x) / 4.8  # Normalized position
        
        # Angle penalty
        angle_penalty = abs(theta) / 0.418  # Normalized angle
        
        if done:
            return -10.0  # Larger penalty for failure
        
        # Reward for staying alive and being centered
        return 1.0 - 0.5 * position_penalty - 0.5 * angle_penalty
    
    def replay(self, batch_size: int):
        """Learn from experience replay memory"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.discretize_state(state)
            next_state_key = self.discretize_state(next_state)
            
            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                main_q = self.q_table_1
                other_q = self.q_table_2
            else:
                main_q = self.q_table_2
                other_q = self.q_table_1
            
            # Get current Q-value
            current_q = main_q.get((state_key, action), 0.0)
            
            if done:
                target_q = reward
            else:
                # Use other Q-table for action selection
                next_action = np.argmax(self.get_q_values(next_state_key, main_q))
                # Use main Q-table for value estimation
                target_q = reward + self.discount_factor * \
                          other_q.get((next_state_key, next_action), 0.0)
            
            # Update Q-value
            new_q = current_q + self.learning_rate * (target_q - current_q)
            main_q[(state_key, action)] = new_q
    
    def evaluate_model(self, env_name: str = 'CartPole-v1', n_episodes: int = 100, render: bool = False):
        """Evaluate the model's performance over multiple episodes"""
        if render:
            env = gym.make(env_name, render_mode='human')
        else:
            env = gym.make(env_name)
            
        eval_rewards = []
        total_timesteps = 0
        success_episodes = 0  # Episodes with reward > 475 (solved threshold)
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(state)
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += 1
                total_timesteps += 1
                
                if render:
                    env.render()
            
            eval_rewards.append(episode_reward)
            if episode_reward > 475:  # CartPole-v1 solved threshold
                success_episodes += 1
                
            if render:
                print(f"Episode {episode + 1} Reward: {episode_reward}")
        
        env.close()
        
        # Calculate statistics
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        success_rate = (success_episodes / n_episodes) * 100
        
        print(f"\nEvaluation Results ({n_episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Max Reward: {max(eval_rewards)}")
        print(f"Min Reward: {min(eval_rewards)}")
        print(f"Average Steps per Second: {total_timesteps / n_episodes:.1f}")
        
        return {
            'rewards': eval_rewards,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'success_rate': success_rate
        }
    
    def save_model(self, filepath: str):
        """Save the Q-tables and training parameters"""
        model_data = {
            'q_table_1': self.q_table_1,
            'q_table_2': self.q_table_2,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'n_bins': self.n_bins,
            'state_bounds': self.state_bounds
        }
        np.save(filepath, model_data)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the Q-tables and training parameters"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.q_table_1 = model_data['q_table_1']
        self.q_table_2 = model_data['q_table_2']
        self.exploration_rate = model_data['exploration_rate']
        self.learning_rate = model_data['learning_rate']
        self.n_bins = model_data['n_bins']
        self.state_bounds = model_data['state_bounds']
        logger.info(f"Model loaded from {filepath}")
    
    def train(self, env_name: str = 'CartPole-v1') -> Dict[str, List[float]]:
        """Train the agent"""
        env = gym.make(env_name)
        rewards_history = []
        q_values_history = []
        best_reward = 0
        best_weights = None
        consecutive_good_episodes = 0
        
        for episode in range(self.episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_q_values = []
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(state)
                next_state, _, done, truncated, _ = env.step(action)
                
                # Calculate custom reward
                reward = self.calculate_reward(next_state, done)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Learn from replay memory
                self.replay(self.batch_size)
                
                total_reward += 1  # Count steps for performance tracking
                state = next_state
                
                # Store average Q-value for monitoring
                state_key = self.discretize_state(state)
                q_values = (self.get_q_values(state_key, self.q_table_1) + 
                          self.get_q_values(state_key, self.q_table_2)) / 2
                episode_q_values.append(np.mean(q_values))
            
            # Record metrics
            rewards_history.append(total_reward)
            avg_q = np.mean(episode_q_values) if episode_q_values else 0
            q_values_history.append(avg_q)
            
            # Track best performance
            if total_reward > best_reward:
                best_reward = total_reward
                # Save the current state of Q-tables
                best_weights = {
                    'q_table_1': self.q_table_1.copy(),
                    'q_table_2': self.q_table_2.copy()
                }
                
            # Track consecutive good episodes (reward > 400)
            if total_reward > 400:
                consecutive_good_episodes += 1
            else:
                consecutive_good_episodes = 0
                
            # Early stopping if we achieve consistently good performance
            if consecutive_good_episodes >= 10:
                logger.info("Early stopping: Achieved consistent good performance!")
                break
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                logger.info(f"Episode {episode + 1}/{self.episodes}")
                logger.info(f"  Average Reward: {avg_reward:.2f}")
                logger.info(f"  Best Reward: {best_reward}")
                logger.info(f"  Average Q-Value: {avg_q:.3f}")
                logger.info(f"  Exploration Rate: {self.exploration_rate:.3f}")
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
        
        env.close()
        return {
            'rewards': rewards_history,
            'q_values': q_values_history
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # Initialize and train the agent
    agent = DoubleQLearningCartPole(
        learning_rate=0.05,
        discount_factor=0.99,
        exploration_rate=1.0,
        episodes=1000
    )
    
    metrics = agent.train()
    
    # Create directory for plots
    os.makedirs('learning_curves', exist_ok=True)
    
    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training rewards with rolling average
    ax1.plot(metrics['rewards'], alpha=0.6, label='Raw Rewards')
    window_size = 100  # Increased window size for smoother curve
    rolling_mean = np.convolve(
        metrics['rewards'],
        np.ones(window_size)/window_size,
        mode='valid'
    )
    ax1.plot(
        range(window_size-1, len(metrics['rewards'])),
        rolling_mean,
        'r--',
        label='Rolling Average'
    )
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Q-values
    ax2.plot(metrics['q_values'])
    ax2.set_title('Average Q-Values')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Q-Value')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves/learning_progress_final_1.png')
    
    # Print final metrics
    print("\nTraining Results:")
    print("-" * 50)
    print(f"Final average reward: {np.mean(metrics['rewards'][-100:]):.2f}")
    print(f"Best reward achieved: {max(metrics['rewards'])}")
    print(f"Final average Q-value: {np.mean(metrics['q_values'][-100:]):.3f}")
    
    # Save the best model
    model_path = 'models/best_cartpole_model.npy'
    agent.save_model(model_path)
    print(f"\nBest model saved to: {model_path}")
    
    # Optional: Test the loaded model
    print("\nTesting loaded model...")
    test_agent = DoubleQLearningCartPole()
    test_agent.load_model(model_path)
    
    # Run a few test episodes
    env = gym.make('CartPole-v1', render_mode=None)
    n_test_episodes = 10
    test_rewards = []
    
    for episode in range(n_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = test_agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += 1
            
        test_rewards.append(episode_reward)
    
    env.close()
    
    print("\nRunning detailed evaluation...")
    # Evaluate with more episodes
    eval_results = test_agent.evaluate_model(n_episodes=100, render=False)
    
    print("\nRunning visual demonstration of best model...")
    # Show a few episodes with rendering
    demo_results = test_agent.evaluate_model(n_episodes=3, render=True)
    
    # Plot evaluation results
    plt.figure(figsize=(10, 5))
    plt.hist(eval_results['rewards'], bins=20, alpha=0.7)
    plt.axvline(eval_results['avg_reward'], color='r', linestyle='--', 
                label=f'Mean ({eval_results["avg_reward"]:.1f})')
    plt.axvline(475, color='g', linestyle='--', label='Solved Threshold (475)')
    plt.title('Distribution of Rewards in Evaluation')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curves/evaluation_distribution_final_1.png')
    
    # Save detailed evaluation metrics
    with open('learning_curves/evaluation_metrics_final_1.txt', 'w') as f:
        f.write("Evaluation Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average Reward: {eval_results['avg_reward']:.2f}\n")
        f.write(f"Standard Deviation: {eval_results['std_reward']:.2f}\n")
        f.write(f"Success Rate: {eval_results['success_rate']:.1f}%\n")
        f.write(f"Number of Episodes: {len(eval_results['rewards'])}\n")
    
    # Save metrics to file
    with open('learning_curves/final_metrics_final_1.txt', 'w') as f:
        f.write("Training Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final average reward: {np.mean(metrics['rewards'][-100:]):.2f}\n")
        f.write(f"Best reward achieved: {max(metrics['rewards'])}\n")
        f.write(f"Final average Q-value: {np.mean(metrics['q_values'][-100:]):.3f}\n")