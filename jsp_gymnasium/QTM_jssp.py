import numpy as np
from typing import List, Dict, Tuple, Any
import torch
import logging
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from scipy.sparse import csr_matrix
import gymnasium as gym
from gymnasium.envs.registration import register
from Job_shop_taillard_generator import TaillardGymGenerator
import wandb
import argparse
import os
from tqdm.auto import tqdm  
import logging
from pathlib import Path
import json
import random
import time
import wandb



# This implementation includes:

# 1. `TsetlinAutomata`: Core Tsetlin Machine implementation adapted for JSSP:
#    - Handles clause generation and updating
#    - Manages TA states and feedback
#    - Includes JSSP-specific parameter tuning

# 2. `Policy`: Main policy class that:
#    - Manages multiple Tsetlin Machines (one per action)
#    - Handles state transformation and prediction
#    - Implements update logic for learning

# 3. `JSSPFeatureTransformer`: Specialized feature transformer that:
#    - Converts JSSP states into binary features
#    - Handles different aspects of the JSSP state (machine times, job status, operation status)
#    - Implements proper normalization and discretization

logger = logging.getLogger(__name__)

class TsetlinAutomata:
    """Memory-optimized Tsetlin Machine implementation"""
    
    def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1):
        self.number_of_clauses = number_of_clauses
        self.T = T
        self.s = s
        self.boost_true_positive_feedback = boost_true_positive_feedback
        
        # Initialize with None to save memory until needed
        self.ta_state = None
        self.clause_sign = None
        self.clause_output = None
        self.feedback_to_clauses = None
    
    def initialize(self, number_of_features):
        """Initialize using int8 instead of int32 to save memory"""
        self.number_of_features = number_of_features
        
        # Use int8 instead of int32
        self.ta_state = np.ones((self.number_of_clauses, 2, self.number_of_features), dtype=np.int8) * 16
        self.clause_sign = np.ones(self.number_of_clauses, dtype=np.int8)
        self.clause_output = np.zeros(self.number_of_clauses, dtype=np.int8)
        self.feedback_to_clauses = np.zeros(self.number_of_clauses, dtype=np.int8)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get current parameters of the Tsetlin Machine"""
        return self.ta_state, self.clause_sign, self.clause_output, self.feedback_to_clauses
        
    def set_params(self, ta_state: np.ndarray, clause_sign: np.ndarray, 
                  clause_output: np.ndarray, feedback_to_clauses: np.ndarray):
        """Set parameters of the Tsetlin Machine"""
        self.ta_state = ta_state
        self.clause_sign = clause_sign
        self.clause_output = clause_output
        self.feedback_to_clauses = feedback_to_clauses

class QTMAgent:
    """Q-learning agent using Tsetlin Machine for Job Shop Scheduling"""
    
    def __init__(self, env, config):
        self.env = env
        self.env_unwrapped = env.unwrapped
        
        # Reduce buffer sizes
        config['buffer_size'] = min(config['buffer_size'], 1000)
        config['sample_size'] = min(config['sample_size'], 32)
        
        self.config = config
        self.policy = Policy(env, config)
        
        # Initialize exploration parameters
        self.epsilon = config['epsilon_init']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        
        # Initialize training variables
        self.total_timesteps = 0
        self.best_makespan = float('inf')
        self.best_actions = []
        self.episode_rewards = []
        
        # Create smaller replay buffer
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'], 
            config['sample_size'],
            n=config.get('n_steps', 1)
        )
        
        logger.info("Memory-optimized QTM Agent initialized")

    def get_next_action(self, cur_obs):
        """Select next action using epsilon-greedy strategy with valid action masking"""
        if random.random() < self.epsilon:
            # Get valid actions from environment
            valid_actions = self.policy.get_valid_actions()  # Use Policy's method
            return random.choice(valid_actions) if valid_actions else 0
        else:
            q_vals = self.policy.predict(cur_obs)
            # Valid actions are already masked in predict()
            return np.argmax(q_vals)

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from environment"""
        valid_actions = []
        for action_id in range(self.config['action_space_size']):
            action = self.env_unwrapped._decode_action(action_id)
            if self.env_unwrapped._is_valid_action(action):
                valid_actions.append(action_id)
        return valid_actions

    def calculate_reward(self, info: Dict) -> float:
        """Calculate custom reward for JSSP"""
        reward = 0.0
        
        # Penalty for invalid actions
        if not info.get('valid_action', True):
            return -100.0
            
        # Reward for completing jobs
        if info.get('job_completed', False):
            reward += 50.0
            
        # Reward for operation completion
        if info.get('operation_completed', False):
            reward += 10.0
            
        # Penalize based on current makespan
        current_makespan = info.get('makespan', 0)
        if current_makespan > 0:
            reward -= current_makespan * 0.01
            
        # Bonus for improving best makespan
        if current_makespan < self.best_makespan:
            improvement = self.best_makespan - current_makespan
            reward += improvement * 0.5
            
        return reward

    def train(self):
        """Train the policy on a batch of experiences"""
        self.replay_buffer.sample()
        
        # Initialize TM inputs dictionary for each action
        tm_inputs = {}
     
        for i in range(self.config['sample_size']):
            next_obs = np.array(self.replay_buffer.sampled_next_obs[i])
            next_q_vals = self.policy.predict(next_obs)
            next_q_val = np.max(next_q_vals)
            
            reward = self.replay_buffer.sampled_rewards[i]
            terminated = self.replay_buffer.sampled_terminated[i]
            
            # Calculate target Q-value
            target_q_val = reward + (1 - terminated) * self.config['gamma'] * next_q_val
            
            # Get current observation and action
            cur_obs = self.replay_buffer.sampled_cur_obs[i]
            action = self.replay_buffer.sampled_actions[i]
            
            # Initialize action entry if not exists
            if action not in tm_inputs:
                tm_inputs[action] = {
                    'observations': [],
                    'target_q_vals': []
                }
            
            # Add to the appropriate action's data
            tm_inputs[action]['observations'].append(cur_obs)
            tm_inputs[action]['target_q_vals'].append(target_q_val)
        
        # Update policy with collected data
        self.policy.update(tm_inputs)

    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def rollout(self):
        """Execute one episode"""
        cur_obs, _ = self.env.reset()
        episode_reward = 0
        episode_actions = []
        
        while True:
            # Select and execute action
            action = self.get_next_action(cur_obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Calculate custom reward
            custom_reward = self.calculate_reward(info)
            
            # Store experience
            self.replay_buffer.save_experience(
                action, cur_obs, next_obs, custom_reward, 
                int(terminated), int(truncated)
            )
            
            # Update statistics
            episode_reward += custom_reward
            episode_actions.append(action)
            self.total_timesteps += 1
            
            # Train if buffer is ready
            if (self.total_timesteps >= self.config['sample_size'] and 
                self.total_timesteps % self.config['train_freq'] == 0):
                self.train()
            
            # Update exploration rate
            self.update_epsilon()
            
            # Update current observation
            cur_obs = next_obs
            
            # Check termination
            if terminated or truncated:
                # Update best solution if better
                current_makespan = info.get('makespan', float('inf'))
                if current_makespan < self.best_makespan:
                    self.best_makespan = current_makespan
                    self.best_actions = episode_actions.copy()
                    logger.info(f"New best makespan: {self.best_makespan}")
                break
            
        return episode_reward, episode_actions

    def learn(self, nr_of_episodes: int) -> Tuple[List[int], float]:
        """Main training loop"""
        logger.info(f"Starting training for {nr_of_episodes} episodes...")
        
        # Wrap range with tqdm for progress bar
        for episode in tqdm(range(nr_of_episodes), 
                          desc="Training", 
                          position=0, 
                          leave=True):
            episode_reward, _ = self.rollout()
            self.episode_rewards.append(episode_reward)
            
            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                logger.info(
                    f"Episode {episode + 1}/{nr_of_episodes}, "
                    f"Mean Reward: {mean_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}, "
                    f"Best Makespan: {self.best_makespan}"
                )
        
        return self.best_actions, self.best_makespan

class Policy:
    """Memory-optimized Policy class implementing Tsetlin Machine for JSSP"""
    
    def __init__(self, env, config):
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.config = config
        
        # Reduce number of TMs by using a sparse representation
        self.action_space_size = config['action_space_size']
        # Instead of creating TMs for all actions upfront, create them as needed
        self.tsetlin_machines = {}  # Will store TMs by action id
        
        # Initialize feature transformer
        self.feature_transformer = JSSPFeatureTransformer(config)
        
        # Initialize TM parameters
        self.nr_of_clauses = min(config['nr_of_clauses'], 1000)  # Reduce number of clauses
        self.T = config['T']
        self.s = config['s']
        
        logger.info(f"Policy initialized with sparse TM representation. Features: {self.feature_transformer.total_features}")

    def _get_or_create_tm(self, action_id):
        """Get existing TM or create new one for given action"""
        if action_id not in self.tsetlin_machines:
            tm = TsetlinAutomata(
                number_of_clauses=self.nr_of_clauses,
                T=self.T,
                s=self.s
            )
            tm.initialize(self.feature_transformer.total_features)
            self.tsetlin_machines[action_id] = tm
        return self.tsetlin_machines[action_id]

    def predict(self, state):
        """Memory-efficient prediction for valid actions only"""
        if len(state) != self.config['obs_space_size']:
            raise ValueError(
                f"State size mismatch. Expected {self.config['obs_space_size']}, "
                f"got {len(state)}. State shape: {state.shape}"
            )
        
        binary_features = self.feature_transformer.transform(state)
        valid_actions = self.get_valid_actions()
        q_values = np.zeros(self.action_space_size)
        q_values.fill(-np.inf)  # Invalid actions get -inf
        
        for action in valid_actions:
            tm = self._get_or_create_tm(action)
            clause_outputs = self.compute_clause_outputs(tm, binary_features)
            q_values[action] = self.calculate_q_value(clause_outputs, tm.clause_sign)
        
        return q_values

    def compute_clause_outputs(self, tm: TsetlinAutomata, features: np.ndarray) -> np.ndarray:
        """Compute clause outputs for given features"""
        clause_outputs = np.zeros(tm.number_of_clauses, dtype=np.int32)
        
        for j in range(tm.number_of_clauses):
            clause_outputs[j] = 1
            for k in range(tm.number_of_features):
                action_include = 1 if tm.ta_state[j, 0, k] > 16 else 0
                action_exclude = 1 if tm.ta_state[j, 1, k] > 16 else 0
                
                if (action_include == 1 and features[k] == 0) or \
                   (action_exclude == 1 and features[k] == 1):
                    clause_outputs[j] = 0
                    break
        
        return clause_outputs

    def calculate_q_value(self, clause_outputs: np.ndarray, clause_signs: np.ndarray) -> float:
        """Calculate Q-value from clause outputs"""
        sum_positive = np.sum(clause_outputs[clause_signs == 1])
        sum_negative = np.sum(clause_outputs[clause_signs == -1])
        return float(sum_positive - sum_negative)

    def update(self, tm_inputs: Dict[int, Dict[str, List]]):
        """Update Tsetlin Machines based on experience
        
        Args:
            tm_inputs: Dictionary mapping action_ids to their observations and target Q-values
                Format: {
                    action_id: {
                        'observations': List of observations,
                        'target_q_vals': List of target Q-values
                    }
                }
        """
        # Iterate through actions that have data
        for action_id, data in tm_inputs.items():
            if data['observations']:  # Check if we have any observations for this action
                tm = self._get_or_create_tm(action_id)
                
                # Update TM for each observation-target pair
                for obs, target in zip(data['observations'], data['target_q_vals']):
                    self.update_single_tm(tm, obs, target)

        logger.debug(f"Updated {len(tm_inputs)} Tsetlin Machines")

    def update_single_tm(self, tm: TsetlinAutomata, state: np.ndarray, target: float) -> None:
        """Update a single Tsetlin Machine"""
        # Transform state to binary features
        binary_features = self.feature_transformer.transform(state)
        
        # Compute current prediction
        clause_outputs = self.compute_clause_outputs(tm, binary_features)
        current_prediction = self.calculate_q_value(clause_outputs, tm.clause_sign)
        
        # Compute feedback
        for j in range(tm.number_of_clauses):
            if tm.clause_sign[j] == 1:
                tm.feedback_to_clauses[j] = 1 if target > current_prediction else 0
            else:
                tm.feedback_to_clauses[j] = 1 if target < current_prediction else 0
        
        # Update TA states
        self.update_ta_state(tm, binary_features)

    def update_ta_state(self, tm: TsetlinAutomata, features: np.ndarray) -> None:
        """Update TA states based on feedback"""
        for j in range(tm.number_of_clauses):
            if tm.feedback_to_clauses[j] == 1:
                # Type I feedback (strengthening)
                for k in range(tm.number_of_features):
                    if features[k] == 1:
                        if tm.ta_state[j, 0, k] > 1:
                            tm.ta_state[j, 0, k] -= 1
                        if random.random() < self.s:
                            if tm.ta_state[j, 1, k] < 32:
                                tm.ta_state[j, 1, k] += 1
                    else:
                        if tm.ta_state[j, 1, k] > 1:
                            tm.ta_state[j, 1, k] -= 1
                        if random.random() < self.s:
                            if tm.ta_state[j, 0, k] < 32:
                                tm.ta_state[j, 0, k] += 1
            else:
                # Type II feedback (weakening)
                for k in range(tm.number_of_features):
                    if features[k] == 1:
                        if random.random() < 1.0/self.T:
                            if tm.ta_state[j, 0, k] < 32:
                                tm.ta_state[j, 0, k] += 1
                        if tm.ta_state[j, 1, k] > 1:
                            tm.ta_state[j, 1, k] -= 1
                    else:
                        if random.random() < 1.0/self.T:
                            if tm.ta_state[j, 1, k] < 32:
                                tm.ta_state[j, 1, k] += 1
                        if tm.ta_state[j, 0, k] > 1:
                            tm.ta_state[j, 0, k] -= 1

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from environment"""
        valid_actions = []
        for action_id in range(self.action_space_size):
            action = self.env_unwrapped._decode_action(action_id)
            if self.env_unwrapped._is_valid_action(action):
                valid_actions.append(action_id)
        return valid_actions
    
class JSSPFeatureTransformer:
    """Memory-optimized feature transformer for JSSP state spaces"""
    
    def __init__(self, config):
        self.config = config
        
        # Basic dimensions
        self.machine_time_bits = min(config['bits_per_machine_time'], 4)
        self.job_status_bits = min(config['bits_per_job_status'], 2)
        self.n_machines = config['n_machines']
        self.n_jobs = config['n_jobs']
        
        # Directly use observation space size from config
        self.expected_state_size = config['obs_space_size']  # This should be 285
        
        # Calculate component sizes
        self.machine_times_size = self.n_machines                    # 15
        self.job_status_size = self.n_jobs                          # 15
        self.op_status_size = self.n_jobs * self.n_machines         # 225
        self.remaining_size = (self.expected_state_size - 
                             (self.machine_times_size + 
                              self.job_status_size + 
                              self.op_status_size))                  # 30 (additional state info)
        
        # Calculate total binary features
        self.total_features = (
            self.n_machines * self.machine_time_bits +  # Machine times binary encoding
            self.n_jobs * self.job_status_bits +       # Job status binary encoding
            self.op_status_size +                      # Operation status (already binary)
            self.remaining_size                        # Additional state information
        )
        
        logger.info(f"Feature transformer initialized:")
        logger.info(f"- Problem size: {self.n_jobs}x{self.n_machines}")
        logger.info(f"- Total features: {self.total_features}")
        logger.info(f"- State components:")
        logger.info(f"  * Machine times: {self.machine_times_size}")
        logger.info(f"  * Job status: {self.job_status_size}")
        logger.info(f"  * Operation status: {self.op_status_size}")
        logger.info(f"  * Additional state info: {self.remaining_size}")
        logger.info(f"- Total state size: {self.expected_state_size}")

    def transform(self, state: np.ndarray) -> np.ndarray:
        """Transform JSSP state into binary features"""
        if len(state) != self.expected_state_size:
            raise ValueError(
                f"State size mismatch. Expected {self.expected_state_size}, "
                f"got {len(state)}. State shape: {state.shape}"
            )
            
        binary_features = np.zeros(self.total_features, dtype=np.int32)
        idx = 0
        
        # 1. Process machine times (first n_machines elements)
        machine_times = state[:self.machine_times_size]
        max_time = np.max(machine_times) if np.max(machine_times) > 0 else 1
        normalized_times = (machine_times / max_time * (2**self.machine_time_bits - 1)).astype(int)
        
        for time in normalized_times:
            binary = format(min(time, 2**self.machine_time_bits - 1), 
                          f'0{self.machine_time_bits}b')
            for bit in binary:
                binary_features[idx] = int(bit)
                idx += 1
        
        # 2. Process job status (next n_jobs elements)
        job_status_start = self.machine_times_size
        job_status_end = job_status_start + self.job_status_size
        job_status = state[job_status_start:job_status_end]
        
        for job_stat in job_status:
            status = int(min(job_stat * (2**self.job_status_bits - 1), 
                           2**self.job_status_bits - 1))
            binary = format(status, f'0{self.job_status_bits}b')
            for bit in binary:
                binary_features[idx] = int(bit)
                idx += 1
        
        # 3. Process operation status matrix
        op_status_start = job_status_end
        op_status_end = op_status_start + self.op_status_size
        op_status = state[op_status_start:op_status_end]
        
        # Copy operation status directly (it's already binary)
        for val in op_status:
            binary_features[idx] = int(val)
            idx += 1
            
        # 4. Process remaining state information
        remaining_start = op_status_end
        remaining_state = state[remaining_start:]
        
        # Copy remaining state information directly
        for val in remaining_state:
            binary_features[idx] = int(val)
            idx += 1
        
        return binary_features

    def transform_batch(self, states: List[np.ndarray]) -> np.ndarray:
        """Transform a batch of states into binary features"""
        return np.array([self.transform(state) for state in states])

    def get_state_shape(self) -> str:
        """Get a string representation of the state shape"""
        return (f"State shape: machine_times({self.machine_times_size}) + "
                f"job_status({self.job_status_size}) + "
                f"operation_status({self.op_status_size}) + "
                f"additional_info({self.remaining_size}) = "
                f"{self.expected_state_size}")

class ReplayBuffer:
    """Memory-efficient experience replay buffer for reinforcement learning"""
    
    def __init__(self, buffer_size: int, sample_size: int, n: int = 1):
        """Initialize replay buffer
        
        Args:
            buffer_size: Maximum number of experiences to store
            sample_size: Number of experiences to sample each time
            n: Number of steps for n-step returns (default: 1)
        """
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.n = n
        
        # Initialize buffers
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.terminated = np.zeros(buffer_size, dtype=np.int8)
        self.truncated = np.zeros(buffer_size, dtype=np.int8)
        
        # Use lists for observations since they might be variable size
        self.cur_obs = []
        self.next_obs = []
        
        # Counter for number of experiences
        self.count = 0
        self.current_idx = 0
        
        # Sampled batch storage
        self.sampled_actions = None
        self.sampled_cur_obs = None
        self.sampled_next_obs = None
        self.sampled_rewards = None
        self.sampled_terminated = None
        self.sampled_truncated = None
        
        logger.info(f"Replay buffer initialized with size {buffer_size}, sample size {sample_size}")

    def save_experience(self, 
                       action: int, 
                       cur_obs: np.ndarray, 
                       next_obs: np.ndarray,
                       reward: float,
                       terminated: int,
                       truncated: int) -> None:
        """Save a single experience to the buffer"""
        idx = self.current_idx
        
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminated[idx] = terminated
        self.truncated[idx] = truncated
        
        # Handle observations
        if len(self.cur_obs) <= idx:
            self.cur_obs.append(cur_obs)
            self.next_obs.append(next_obs)
        else:
            self.cur_obs[idx] = cur_obs
            self.next_obs[idx] = next_obs
        
        # Update counters
        self.count = min(self.count + 1, self.buffer_size)
        self.current_idx = (self.current_idx + 1) % self.buffer_size

    def sample(self) -> None:
        """Sample a batch of experiences"""
        if self.count < self.sample_size:
            logger.warning(f"Not enough experiences in buffer. Have {self.count}, need {self.sample_size}")
            return
        
        indices = np.random.choice(self.count, self.sample_size, replace=False)
        
        # Store sampled experiences
        self.sampled_actions = self.actions[indices]
        self.sampled_rewards = self.rewards[indices]
        self.sampled_terminated = self.terminated[indices]
        self.sampled_truncated = self.truncated[indices]
        
        # Handle observations
        self.sampled_cur_obs = [self.cur_obs[i] for i in indices]
        self.sampled_next_obs = [self.next_obs[i] for i in indices]
        
        logger.debug(f"Sampled {self.sample_size} experiences from buffer")

    def compute_n_step_returns(self, gamma: float) -> np.ndarray:
        """Compute n-step returns for sampled experiences"""
        n_step_returns = np.zeros(self.sample_size)
        
        for i in range(self.sample_size):
            ret = 0.0
            for step in range(self.n):
                if i + step >= self.sample_size:
                    break
                if self.sampled_terminated[i + step] or self.sampled_truncated[i + step]:
                    ret += self.sampled_rewards[i + step] * (gamma ** step)
                    break
                ret += self.sampled_rewards[i + step] * (gamma ** step)
            n_step_returns[i] = ret
            
        return n_step_returns

    def clear(self) -> None:
        """Clear the replay buffer"""
        self.count = 0
        self.current_idx = 0
        self.cur_obs.clear()
        self.next_obs.clear()
        self.actions.fill(0)
        self.rewards.fill(0)
        self.terminated.fill(0)
        self.truncated.fill(0)
        
        self.sampled_actions = None
        self.sampled_cur_obs = None
        self.sampled_next_obs = None
        self.sampled_rewards = None
        self.sampled_terminated = None
        self.sampled_truncated = None
        
        logger.info("Replay buffer cleared")

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return self.count >= self.sample_size
    
def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def init_wandb(config: Dict[str, Any], instance_name: str) -> None:
    """Initialize Weights & Biases logging"""
    wandb.init(
        project="job-shop-qtm_server",
        entity="reshma-stha2016",  # Using your actual W&B username
        name=f"QTM_{instance_name}_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "QTM",
            "instance": instance_name,
            "problem_size": {
                "n_jobs": config.get('n_jobs'),
                "n_machines": config.get('n_machines')
            },
            "model_config": {
                "nr_of_clauses": config['nr_of_clauses'],
                "T": config['T'],
                "s": config.get('s', 1.5),
                "memory_size": config['buffer_size'],
                "batch_size": config['sample_size'],
                "gamma": config['gamma'],
                "epsilon_start": config['epsilon_init'],
                "epsilon_min": config['epsilon_min'],
                "epsilon_decay": config['epsilon_decay'],
            }
        },
        tags=["QTM", instance_name, "job-shop"]
    )

def get_default_config(env) -> Dict[str, Any]:
    """Get default configuration for the QTM agent"""
    return {
        'env_name': 'job-shop',
        'algorithm': 'QTM',
        'nr_of_clauses': 2000,           # Number of clauses per TM
        'T': 1500,                       # Threshold
        's': 1.5,                        # Specificity
        'bits_per_machine_time': 8,      # Bits for encoding machine times
        'bits_per_job_status': 4,        # Bits for encoding job status
        'n_jobs': len(env.unwrapped.jobs),         # Number of jobs
        'n_machines': env.unwarpped.num_machines,  # Number of machines
        'gamma': 0.99,                   # Discount factor
        'epsilon_init': 1.0,             # Initial exploration
        'epsilon_min': 0.01,             # Minimum exploration
        'epsilon_decay': 0.995,          # Exploration decay rate
        'buffer_size': 10000,            # Experience replay buffer size
        'sample_size': 64,               # Batch size
        'train_freq': 100,               # Training frequency
        'test_freq': 5,                  # Testing frequency
        'save': True,                    # Save results
        'seed': 42                       # Random seed
    }

def save_results(result_file: Path, results: Dict[str, Any]) -> None:
    """Save results to file"""
    try:
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logging.info(f"Results saved to: {result_file}")
        
        # Create a "latest" symlink
        latest_link = result_file.parent / f"{result_file.stem}_latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(result_file.name)
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling with QTM')
    
    # Add command line arguments
    parser.add_argument('--instance', type=str, required=True,
                      help='Instance name (e.g., ta01.txt)')
    parser.add_argument('--episodes', type=int, default=1, # Change from 1000 to 1
                      help='Number of training episodes')
    parser.add_argument('--clauses', type=int, default=2000,
                      help='Number of clauses per TM')
    parser.add_argument('--threshold', type=int, default=1500,
                      help='Threshold parameter T')
    parser.add_argument('--specificity', type=float, default=1.5,
                      help='Specificity parameter s')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Set random seeds
        set_seeds(args.seed)
        
        # Create environment
        logging.info(f"Creating environment for instance: {args.instance}")
        if args.instance.startswith('ta'):
            # Create the base environment first
            base_env = TaillardGymGenerator.create_env_from_instance(args.instance)
            
            # Get environment information before wrapping
            n_jobs = len(base_env.jobs)
            n_machines = base_env.num_machines
            
            # Create the gymnasium environment
            register(
                id='JobShop-Taillard-v0',
                entry_point='job_shop_env:JobShopGymEnv',
                kwargs={'jobs': base_env.jobs}
            )
            env = gym.make('JobShop-Taillard-v0')
        else:
            raise ValueError("Currently only Taillard instances are supported")
        
        # Create configuration dictionary
        config = {
            'env_name': 'job-shop',
            'algorithm': 'QTM',
            'nr_of_clauses': args.clauses,
            'T': args.threshold,
            's': args.specificity,
            'bits_per_machine_time': 8,
            'bits_per_job_status': 4,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'action_space_size': env.action_space.n,  # Add action space size
            'obs_space_size': env.observation_space.shape[0],  # Add observation space size
            'gamma': 0.99,
            'epsilon_init': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'sample_size': 64,
            'train_freq': 100,
            'test_freq': 5,
            'save': True,
            'seed': args.seed
        }
        
        # Print problem configuration
        logging.info("\nProblem Configuration:")
        logging.info(f"Instance: {args.instance}")
        logging.info(f"Jobs: {config['n_jobs']}")
        logging.info(f"Machines: {config['n_machines']}")
        logging.info(f"Action Space Size: {config['action_space_size']}")
        logging.info(f"Observation Space Size: {config['obs_space_size']}")
        logging.info(f"Clauses per TM: {config['nr_of_clauses']}")
        logging.info(f"Threshold T: {config['T']}")
        logging.info(f"Specificity s: {config['s']}")
        
        # Initialize W&B if enabled
        if args.wandb:
            init_wandb(config, args.instance)
        
        # Create and train agent
        logging.info("\nInitializing QTM agent...")
        agent = QTMAgent(env, config)
        
        logging.info("\nStarting training...")
        best_actions, best_makespan = agent.learn(nr_of_episodes=args.episodes)
        
        # Execute best solution and collect data
        logging.info("\nExecuting best solution...")
        env.reset()
        schedule_data = []
        
        for action_id in best_actions:
            action = env.unwrapped._decode_action(action_id)
            obs, reward, done, truncated, info = env.step(action_id)
            
            schedule_data.append({
                'job': int(action.job),
                'machine': int(action.machine),
                'operation': int(action.operation),
                'start': int(info.get('start_time', 0)),
                'duration': int(env.unwrapped.jobs[action.job].operations[action.operation].duration),
                'end': int(info.get('end_time', 0))
            })
        
        # Save results
        results = {
            'instance': args.instance,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'episodes': args.episodes,
                'clauses': config['nr_of_clauses'],
                'threshold': config['T'],
                'specificity': config['s'],
                'gamma': config['gamma'],
                'epsilon_init': config['epsilon_init'],
                'epsilon_min': config['epsilon_min'],
                'epsilon_decay': config['epsilon_decay']
            },
            'problem': {
                'num_jobs': config['n_jobs'],
                'num_machines': config['n_machines']
            },
            'solution': {
                'makespan': float(best_makespan),
                'num_actions': len(best_actions),
                'schedule': schedule_data,
                'actions': [int(a) for a in best_actions]
            },
            'metrics': {
                'machine_utilization': [float(u) for u in env.unwrapped.get_machine_utilization()]
            }
        }
        
        # Save results to file
        result_file =Path(f'results/{args.instance[:-4]}/qtm_{time.strftime("%Y%m%d_%H%M%S")}.json')
        result_file.parent.mkdir(parents=True, exist_ok=True)
        save_results(result_file, results)
        
        # Log to W&B if enabled
        if args.wandb:
            wandb.log({
                "final_results": {
                    "best_makespan": best_makespan,
                    "number_of_actions": len(best_actions)
                }
            })
            wandb.finish()
        
        # Display final results
        logging.info("\nFinal Results:")
        logging.info(f"Best Makespan: {best_makespan}")
        logging.info(f"Number of Actions: {len(best_actions)}")
        logging.info(f"Results saved to: {result_file}")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        if args.wandb and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
