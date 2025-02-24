import numpy as np
import gymnasium as gym
import logging
from typing import List, Tuple, Dict, Any
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

logger = logging.getLogger(__name__)

class CartPoleFeatureTransformer:
    """Feature transformer for CartPole state spaces"""
    def __init__(self, 
                 position_bits: int = 8,
                 velocity_bits: int = 8,
                 angle_bits: int = 8,
                 angular_velocity_bits: int = 8):
        self.position_bits = position_bits
        self.velocity_bits = velocity_bits
        self.angle_bits = angle_bits
        self.angular_velocity_bits = angular_velocity_bits
        
        self.total_features = (
            position_bits +
            velocity_bits +
            angle_bits +
            angular_velocity_bits
        )
        
    def transform(self, state) -> np.ndarray:
        """Transform CartPole state into binary features"""
        binary_features = np.zeros(self.total_features, dtype=np.int32)
        current_idx = 0
        
        # Position normalization and binarization
        pos_norm = (state[0] + 2.4) / 4.8  # Normalize to [0,1]
        pos_bin = format(int(pos_norm * (2**self.position_bits - 1)), 
                        f'0{self.position_bits}b')
        for bit in pos_bin:
            binary_features[current_idx] = int(bit)
            current_idx += 1
            
        # Velocity normalization and binarization
        vel_norm = (np.clip(state[1], -3, 3) + 3) / 6
        vel_bin = format(int(vel_norm * (2**self.velocity_bits - 1)),
                        f'0{self.velocity_bits}b')
        for bit in vel_bin:
            binary_features[current_idx] = int(bit)
            current_idx += 1
            
        # Angle normalization and binarization
        angle_norm = (state[2] + 0.418) / 0.836
        angle_bin = format(int(angle_norm * (2**self.angle_bits - 1)),
                          f'0{self.angle_bits}b')
        for bit in angle_bin:
            binary_features[current_idx] = int(bit)
            current_idx += 1
            
        # Angular velocity normalization and binarization
        ang_vel_norm = (np.clip(state[3], -3, 3) + 3) / 6
        ang_vel_bin = format(int(ang_vel_norm * (2**self.angular_velocity_bits - 1)),
                            f'0{self.angular_velocity_bits}b')
        for bit in ang_vel_bin:
            binary_features[current_idx] = int(bit)
            current_idx += 1
            
        return binary_features

class HybridTsetlinQLearningCartPole:
    """
    Adapted QTM algorithm for CartPole environment
    """
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 episodes: int = 1000,
                 nr_clauses: int = 2000,
                 T: float = 1500,
                 s: float = 1.5):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        
        # Tsetlin Machine parameters
        self.nr_clauses = nr_clauses
        self.T = T
        self.s = s
        
        # Initialize components
        self.q_table = {}
        self.tsetlin_machines = {}
        self.feature_transformer = CartPoleFeatureTransformer()
        self.rng = np.random.RandomState()
        
    def _get_or_create_tm(self, action: int) -> MultiClassTsetlinMachine:
        """Get existing TM or create new one for given action"""
        if action not in self.tsetlin_machines:
            tm = MultiClassTsetlinMachine(
                self.nr_clauses,
                self.T,
                self.s,
                number_of_classes=2
            )
            self.tsetlin_machines[action] = tm
        return self.tsetlin_machines[action]
    
    def _calculate_priority(self, state: np.ndarray, action: int) -> float:
        """Calculate priority score using TM prediction"""
        state_features = self.feature_transformer.transform(state)
        tm = self._get_or_create_tm(action)
        
        # Get TM prediction
        tm_prediction = tm.predict(state_features.reshape(1, -1))[0]
        
        # Get Q-value
        state_key = tuple(state_features)
        q_value = self.q_table.get((state_key, action), 0.0)
        
        # Combine TM prediction with Q-value
        final_priority = 0.7 * q_value + 0.3 * tm_prediction
        
        return final_priority
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy with TM guidance"""
        if self.rng.random() < self.exploration_rate:
            # Smart exploration using TM priorities
            priorities = []
            for action in [0, 1]:  # CartPole has 2 actions
                priority = self._calculate_priority(state, action)
                priorities.append(max(0.0, priority))
            
            # Normalize priorities for probability distribution
            total_priority = sum(priorities)
            if total_priority > 0:
                probabilities = [p/total_priority for p in priorities]
                return self.rng.choice([0, 1], p=probabilities)
            return self.rng.choice([0, 1])
        
        # Greedy selection
        priorities = [self._calculate_priority(state, a) for a in [0, 1]]
        return np.argmax(priorities)
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update both Q-table and Tsetlin Machine"""
        # Convert states to feature representation
        state_features = self.feature_transformer.transform(state)
        next_state_features = self.feature_transformer.transform(next_state)
        
        # Q-learning update
        state_key = tuple(state_features)
        next_state_key = tuple(next_state_features)
        
        current_q = self.q_table.get((state_key, action), 0.0)
        next_q_values = [self.q_table.get((next_state_key, a), 0.0) for a in [0, 1]]
        max_next_q = max(next_q_values)
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[(state_key, action)] = new_q
        
        # Update TM
        tm = self._get_or_create_tm(action)
        tm_class = 1 if new_q > current_q else 0
        tm.fit(state_features.reshape(1, -1), np.array([tm_class]), epochs=1)
        
    def train(self, env_name: str = 'CartPole-v1', render: bool = False) -> List[float]:
        """Train the hybrid algorithm on CartPole environment"""
        env = gym.make(env_name, render_mode='human' if render else None)
        rewards_history = []
        
        for episode in range(self.episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                self.update(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                
            rewards_history.append(total_reward)
            
            # Decay exploration rate
            self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                logger.info(f"Episode {episode + 1}/{self.episodes}, "
                          f"Average Reward: {avg_reward:.2f}, "
                          f"Exploration Rate: {self.exploration_rate:.3f}")
        
        env.close()
        return rewards_history

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and train the hybrid algorithm
    qtm = HybridTsetlinQLearningCartPole(
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        episodes=500,
        nr_clauses=2000,
        T=1500,
        s=1.5
    )
    
    rewards = qtm.train(render=True)
    print(f"Training completed!")
    print(f"Final average reward over last 100 episodes: {np.mean(rewards[-100:]):.2f}")