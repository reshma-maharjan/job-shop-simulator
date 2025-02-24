import numpy as np
import gymnasium as gym
import logging
import matplotlib.pyplot as plt
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

class QLearningCartPole:
    """
    Q-Learning implementation for CartPole environment
    """
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 episodes: int = 1000):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        self.episodes = episodes
        
        # Initialize Q-table
        self.q_table = {}
        
        # Define state discretization parameters
        self.n_bins = 10  # Number of bins for each state dimension
        
        # Define state space bounds
        self.state_bounds = {
            'position': (-4.8, 4.8),
            'velocity': (-10, 10),
            'angle': (-0.418, 0.418),
            'angular_velocity': (-10, 10)
        }
        
    def discretize_state(self, state: np.ndarray) -> tuple:
        """Convert continuous state to discrete state"""
        bounds = list(self.state_bounds.values())
        discretized = []
        
        for i, (val, (lower, upper)) in enumerate(zip(state, bounds)):
            scaled = (val - lower) / (upper - lower)
            scaled = np.clip(scaled, 0, 1)
            bin_idx = int(scaled * (self.n_bins - 1))
            discretized.append(bin_idx)
            
        return tuple(discretized)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy"""
        if np.random.random() < self.exploration_rate:
            return np.random.choice([0, 1])
        
        state_key = self.discretize_state(state)
        q_values = [self.q_table.get((state_key, a), 0.0) for a in [0, 1]]
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> float:
        """Update Q-value for state-action pair"""
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        # Get current Q-value
        current_q = self.q_table.get((state_key, action), 0.0)
        
        if done:
            target_q = reward
        else:
            # Get max Q-value for next state
            next_q_values = [self.q_table.get((next_state_key, a), 0.0) for a in [0, 1]]
            target_q = reward + self.discount_factor * max(next_q_values)
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state_key, action)] = new_q
        
        return new_q
    
    def train(self, env_name: str = 'CartPole-v1') -> Dict[str, List[float]]:
        """Train the Q-learning agent"""
        env = gym.make(env_name)
        rewards_history = []
        q_values_history = []
        
        for episode in range(self.episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_q_values = []
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Update Q-table
                new_q = self.update(state, action, reward, next_state, done)
                episode_q_values.append(new_q)
                
                total_reward += reward
                state = next_state
            
            # Record metrics
            rewards_history.append(total_reward)
            q_values_history.append(np.mean(episode_q_values))
            
            # Log progress
            if (episode + 1) % 100 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}")
                logger.info(f"  Average Reward: {np.mean(rewards_history[-100:]):.2f}")
                logger.info(f"  Average Q-Value: {np.mean(q_values_history[-100:]):.3f}")
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
    
    # Initialize and train the Q-learning agent
    agent = QLearningCartPole(
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        episodes=5000
    )
    
    metrics = agent.train()
    
    # Create directory for plots
    os.makedirs('learning_curves', exist_ok=True)
    
    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training rewards with rolling average
    ax1.plot(metrics['rewards'], alpha=0.6, label='Raw Rewards')
    window_size = 20
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
    plt.savefig('learning_curves/learning_progress_only_q_learning.png')
    
    # Print final metrics
    print("\nTraining Results:")
    print("-" * 50)
    print(f"Final average reward: {np.mean(metrics['rewards'][-100:]):.2f}")
    print(f"Final average Q-value: {np.mean(metrics['q_values'][-100:]):.3f}")
    
    # Save metrics to file
    with open('learning_curves/final_metrics.txt', 'w') as f:
        f.write("Training Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final average reward: {np.mean(metrics['rewards'][-100:]):.2f}\n")
        f.write(f"Final average Q-value: {np.mean(metrics['q_values'][-100:]):.3f}\n")