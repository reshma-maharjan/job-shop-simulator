import numpy as np
import gymnasium as gym
import logging
from collections import deque
from typing import List, Tuple, Dict, Any
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import random
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

class EnhancedBinarizer:
    """Improved binarizer for CartPole state with overlap"""
    def __init__(self, bits_per_feature=8, overlap=2):
        self.bits_per_feature = bits_per_feature
        self.overlap = overlap
        self.ranges = {
            'position': (-4.8, 4.8),
            'velocity': (-10.0, 10.0),
            'angle': (-0.418, 0.418),
            'angular_velocity': (-10.0, 10.0)
        }
    
    def binarize_value(self, value: float, min_val: float, max_val: float) -> List[int]:
        """Convert continuous value to binary with overlapping thresholds"""
        normalized = (np.clip(value, min_val, max_val) - min_val) / (max_val - min_val)
        
        # Create overlapping thresholds
        thresholds = np.linspace(0, 1, self.bits_per_feature + self.overlap)
        binary = [1 if normalized > t else 0 for t in thresholds]
        
        return binary
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """Transform full state to binary features"""
        binary_features = []
        for value, (min_val, max_val) in zip(state, self.ranges.values()):
            binary = self.binarize_value(value, min_val, max_val)
            binary_features.extend(binary)
        return np.array(binary_features)

class HybridDoubleQLearningTsetlin:
    """
    Enhanced Double Q-Learning with Tsetlin Machine integration
    """
    def __init__(self,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 episodes: int = 5000,
                 batch_size: int = 64,
                 memory_size: int = 50000,
                 nr_clauses: int = 2000,
                 T: float = 50.0,
                 s: float = 5.0,
                 feature_bits: int = 10):
        
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        self.episodes = episodes
        self.batch_size = batch_size
        
        # Initialize double Q-tables
        self.q_table_1 = {}
        self.q_table_2 = {}
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Feature transformation
        self.binarizer = EnhancedBinarizer(bits_per_feature=feature_bits)
        
        # Initialize Tsetlin Machines (one for each action)
        self.tsetlin_machines = {
            0: MultiClassTsetlinMachine(nr_clauses, T, s, number_of_classes=2),
            1: MultiClassTsetlinMachine(nr_clauses, T, s, number_of_classes=2)
        }
        
        # Performance tracking
        self.best_reward = 0
        self.best_weights = None
        
    def get_state_features(self, state: np.ndarray) -> np.ndarray:
        """Get binary features for state"""
        return self.binarizer.transform(state)
    
    def calculate_reward(self, state: np.ndarray, done: bool) -> float:
        """Enhanced reward calculation"""
        x, x_dot, theta, theta_dot = state
        
        # Failure states
        if abs(x) > 2.4 or abs(theta) > 0.209:
            return -10.0
        
        if done:
            return 10.0 if abs(x) < 1.0 and abs(theta) < 0.1 else -5.0
            
        # Continuous reward based on cart position and pole angle
        position_reward = 1.0 - min(1.0, abs(x) / 2.4)
        angle_reward = 1.0 - min(1.0, abs(theta) / 0.209)
        velocity_penalty = -min(1.0, abs(x_dot) / 10.0)
        angular_velocity_penalty = -min(1.0, abs(theta_dot) / 10.0)
        
        return position_reward + 2*angle_reward + 0.5*velocity_penalty + 0.5*angular_velocity_penalty
    
    def get_q_values(self, state_features: np.ndarray, q_table: dict) -> np.ndarray:
        """Get Q-values for all actions in a given state"""
        state_key = tuple(state_features)
        return np.array([q_table.get((state_key, a), 0.0) for a in [0, 1]])
    
    def get_tm_prediction(self, state_features: np.ndarray, action: int) -> float:
        """Get TM prediction confidence for state-action pair"""
        tm = self.tsetlin_machines[action]
        prediction = tm.predict(state_features.reshape(1, -1))[0]
        return float(prediction)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy with TM guidance"""
        if np.random.random() < self.exploration_rate:
            return np.random.choice([0, 1])
        
        state_features = self.get_state_features(state)
        
        # Combine Q-values from both tables
        q_values_1 = self.get_q_values(state_features, self.q_table_1)
        q_values_2 = self.get_q_values(state_features, self.q_table_2)
        combined_q = (q_values_1 + q_values_2) / 2
        
        # Get TM predictions
        tm_values = np.array([self.get_tm_prediction(state_features, a) for a in [0, 1]])
        
        # Combine Q-values and TM predictions
        final_values = 0.7 * combined_q + 0.3 * tm_values
        return np.argmax(final_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        state_features = self.get_state_features(state)
        next_state_features = self.get_state_features(next_state)
        self.memory.append((state_features, action, reward, next_state_features, done))
    
    def replay(self, batch_size: int):
        """Learn from experience replay memory"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state_features, action, reward, next_state_features, done in batch:
            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                main_q = self.q_table_1
                other_q = self.q_table_2
            else:
                main_q = self.q_table_2
                other_q = self.q_table_1
            
            state_key = tuple(state_features)
            next_state_key = tuple(next_state_features)
            
            # Update Q-value
            current_q = main_q.get((state_key, action), 0.0)
            
            if done:
                target_q = reward
            else:
                next_action = np.argmax(self.get_q_values(next_state_features, main_q))
                target_q = reward + self.discount_factor * \
                          other_q.get((next_state_key, next_action), 0.0)
            
            new_q = current_q + self.learning_rate * (target_q - current_q)
            main_q[(state_key, action)] = new_q
            
            # Update TM
            tm = self.tsetlin_machines[action]
            tm_class = 1 if reward > 0 else 0
            tm.fit(state_features.reshape(1, -1), np.array([tm_class]), epochs=1)
    
    def train(self, env_name: str = 'CartPole-v1') -> Dict[str, List[float]]:
        """Train the hybrid agent"""
        env = gym.make(env_name)
        rewards_history = []
        q_values_history = []
        tm_accuracy_history = []
        
        for episode in range(self.episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_q_values = []
            episode_tm_predictions = []
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
                
                total_reward += 1  # Count steps
                state = next_state
                
                # Store metrics
                state_features = self.get_state_features(state)
                q_values = self.get_q_values(state_features, self.q_table_1)
                episode_q_values.append(np.mean(q_values))
                
                tm_pred = self.get_tm_prediction(state_features, action)
                episode_tm_predictions.append(tm_pred)
            
            # Record metrics
            rewards_history.append(total_reward)
            q_values_history.append(np.mean(episode_q_values))
            tm_accuracy_history.append(np.mean(episode_tm_predictions))
            
            # Track best performance
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_weights = {
                    'q_table_1': self.q_table_1.copy(),
                    'q_table_2': self.q_table_2.copy()
                }
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_q = np.mean(q_values_history[-100:])
                avg_tm = np.mean(tm_accuracy_history[-100:])
                
                logger.info(f"Episode {episode + 1}/{self.episodes}")
                logger.info(f"  Average Reward: {avg_reward:.2f}")
                logger.info(f"  Best Reward: {self.best_reward}")
                logger.info(f"  Average Q-Value: {avg_q:.3f}")
                logger.info(f"  TM Accuracy: {avg_tm:.3f}")
                logger.info(f"  Exploration Rate: {self.exploration_rate:.3f}")
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
        
        env.close()
        return {
            'rewards': rewards_history,
            'q_values': q_values_history,
            'tm_accuracy': tm_accuracy_history
        }

def plot_learning_curves(metrics: Dict[str, List[float]], output_dir: str = "learning_curves"):
    """Plot and save learning curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot rewards
    rewards = np.array(metrics['rewards'])
    ax1.plot(rewards, alpha=0.6, label='Raw Rewards')
    
    # Calculate rolling mean only if we have enough data
    if len(rewards) >= 100:
        window_size = 100
        rolling_mean = np.convolve(
            rewards,
            np.ones(window_size)/window_size,
            mode='valid'
        )
        ax1.plot(
            np.arange(window_size-1, len(rewards)),
            rolling_mean,
            'r--',
            label='Rolling Average'
        )
    
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Q-values
    ax2.plot(metrics['q_values'])
    ax2.set_title('Average Q-Values')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Q-Value')
    ax2.grid(True, alpha=0.3)
    
    # Plot TM accuracy
    ax3.plot(metrics['tm_accuracy'])
    ax3.set_title('TM Prediction Accuracy')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_progress_qtm.png')
    plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the hybrid agent
    agent = HybridDoubleQLearningTsetlin(
        learning_rate=0.05,
        discount_factor=0.99,
        exploration_rate=1.0,
        episodes=1000,
        batch_size=64,
        nr_clauses=2000,
        T=50.0,
        s=5.0
    )
    
    # Train the agent
    metrics = agent.train()
    
    # Plot learning curves
    plot_learning_curves(metrics)
    
    # Print final metrics
    print("\nTraining Results:")
    print("-" * 50)
    print(f"Final average reward: {np.mean(metrics['rewards'][-100:]):.2f}")
    print(f"Best reward achieved: {max(metrics['rewards'])}")
    print(f"Final average Q-value: {np.mean(metrics['q_values'][-100:]):.3f}")
    print(f"Final TM accuracy: {np.mean(metrics['tm_accuracy'][-100:]):.3f}")