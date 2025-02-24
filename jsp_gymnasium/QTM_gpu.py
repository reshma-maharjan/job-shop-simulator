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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

class TsetlinAutomata:
    """GPU-accelerated Tsetlin Machine implementation"""
    
    def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1):
        self.number_of_clauses = number_of_clauses
        self.T = T
        self.s = s
        self.boost_true_positive_feedback = boost_true_positive_feedback
        
        # Initialize with None
        self.ta_state = None
        self.clause_sign = None
        self.clause_output = None
        self.feedback_to_clauses = None
    
    def initialize(self, number_of_features):
        """Initialize using GPU tensors"""
        self.number_of_features = number_of_features
        
        # Move to GPU using PyTorch tensors
        self.ta_state = torch.ones((self.number_of_clauses, 2, self.number_of_features), 
                                 dtype=torch.int8, device=device) * 16
        self.clause_sign = torch.ones(self.number_of_clauses, 
                                    dtype=torch.int8, device=device)
        self.clause_output = torch.zeros(self.number_of_clauses, 
                                       dtype=torch.int8, device=device)
        self.feedback_to_clauses = torch.zeros(self.number_of_clauses, 
                                             dtype=torch.int8, device=device)

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current parameters"""
        return self.ta_state, self.clause_sign, self.clause_output, self.feedback_to_clauses
        
    def set_params(self, ta_state: torch.Tensor, clause_sign: torch.Tensor, 
                  clause_output: torch.Tensor, feedback_to_clauses: torch.Tensor):
        """Set parameters"""
        self.ta_state = ta_state.to(device)
        self.clause_sign = clause_sign.to(device)
        self.clause_output = clause_output.to(device)
        self.feedback_to_clauses = feedback_to_clauses.to(device)

class QTMAgent:
    """GPU-accelerated Q-learning agent"""
    
    def __init__(self, env, config):
        self.env = env
        self.env_unwrapped = env.unwrapped
        config['buffer_size'] = min(config['buffer_size'], 1000)
        config['sample_size'] = min(config['sample_size'], 32)
        self.config = config
        self.policy = Policy(env, config)
        
        # Move exploration parameters to GPU
        self.epsilon = torch.tensor(config['epsilon_init'], device=device)
        self.epsilon_min = torch.tensor(config['epsilon_min'], device=device)
        self.epsilon_decay = torch.tensor(config['epsilon_decay'], device=device)
        
        self.total_timesteps = 0
        self.best_makespan = float('inf')
        self.best_actions = []
        self.episode_rewards = []
        
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'], 
            config['sample_size'],
            n=config.get('n_steps', 1)
        )

        logger.info("QTM Agent initialized")

    def get_next_action(self, cur_obs):

        cur_obs=np.array(cur_obs, dtype=np.float32)

        """Select next action using epsilon-greedy strategy"""
        if torch.rand(1, device=device).item() < self.epsilon:
            valid_actions = self.policy.get_valid_actions()
            return random.choice(valid_actions) if valid_actions else 0
        else:
            q_vals = self.policy.predict(cur_obs)
            return torch.argmax(q_vals).item()
        
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
        """Train the policy on a batch of experiences with validation"""
        try:
            if not self.replay_buffer.is_ready:
                logger.warning("Training called but replay buffer not ready")
                return
            
            logger.debug("Sampling from replay buffer")
            self.replay_buffer.sample()
            
            if self.replay_buffer.sampled_cur_obs is None:
                logger.error("No observations sampled from buffer")
                return
                
            # Initialize TM inputs dictionary for each action
            tm_inputs = {}
            
            logger.debug(f"Processing {self.config['sample_size']} experiences")
            for i in range(self.config['sample_size']):
                next_obs = np.array(self.replay_buffer.sampled_next_obs[i])
                
                # Validate next_obs
                if next_obs.shape != (self.config['obs_space_size'],):
                    logger.error(f"Invalid next_obs shape: {next_obs.shape}")
                    continue
                    
                next_q_vals = self.policy.predict(next_obs)
                next_q_val = np.max(next_q_vals)
                
                reward = self.replay_buffer.sampled_rewards[i]
                terminated = self.replay_buffer.sampled_terminated[i]
                
                # Calculate target Q-value
                target_q_val = reward + (1 - terminated) * self.config['gamma'] * next_q_val
                
                # Get current observation and action
                cur_obs = self.replay_buffer.sampled_cur_obs[i]
                action = self.replay_buffer.sampled_actions[i].item()  # Convert from tensor
                
                # Validate current observation
                if not isinstance(cur_obs, np.ndarray):
                    logger.error(f"Invalid cur_obs type: {type(cur_obs)}")
                    continue
                    
                # Initialize action entry if not exists
                if action not in tm_inputs:
                    tm_inputs[action] = {
                        'observations': [],
                        'target_q_vals': []
                    }
                
                # Add to the appropriate action's data
                tm_inputs[action]['observations'].append(cur_obs)
                tm_inputs[action]['target_q_vals'].append(target_q_val)
            
            if not tm_inputs:
                logger.warning("No valid experiences to train on")
                return
                
            # Update policy with collected data
            logger.debug(f"Updating policy with {len(tm_inputs)} different actions")
            self.policy.update(tm_inputs)
            logger.debug("Policy update completed")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def rollout(self):
        """Execute one episode with enhanced debugging"""
        logger.info("Starting rollout")
        cur_obs, _ = self.env.reset()
        logger.debug(f"Initial observation: {cur_obs.shape}, dtype={cur_obs.dtype}")

        episode_reward = 0
        episode_actions = []
        step_count = 0
        
        try:
            while True:
                # Select and execute action
                logger.debug(f"Step {step_count}: Getting valid actions")
                valid_actions = self.get_valid_actions()
                logger.debug(f"Valid actions: {valid_actions}")
                
                logger.debug(f"Selecting action (epsilon: {self.epsilon.item():.3f})")
                action = self.get_next_action(cur_obs)
                logger.debug(f"Selected action: {action}")
                
                # Execute action
                logger.debug("Executing step in environment")
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                logger.debug(f"Raw env reward: {reward}")
                
                # Calculate custom reward
                custom_reward = self.calculate_reward(info)
                logger.debug(f"Custom reward: {custom_reward}")
                logger.debug(f"Current makespan: {info.get('makespan', 'N/A')}")
                
                # Store experience
                logger.debug("Saving experience to buffer")
                self.replay_buffer.save_experience(
                    action, cur_obs, next_obs, custom_reward, 
                    int(terminated), int(truncated)
                )
                
                # Update statistics
                episode_reward += custom_reward
                episode_actions.append(action)
                self.total_timesteps += 1
                step_count += 1
                
                # Train if buffer is ready
                if (self.total_timesteps >= self.config['sample_size'] and 
                    self.total_timesteps % self.config['train_freq'] == 0):
                    logger.debug("Training on buffered experiences")
                    try:
                        self.train()
                    except Exception as e:
                        logger.error(f"Error during training: {e}")
                        raise
                
                # Update exploration rate
                self.update_epsilon()
                
                # Update current observation
                cur_obs = next_obs
                
                # Check termination
                if terminated or truncated:
                    logger.debug(f"Episode ended after {step_count} steps")
                    current_makespan = info.get('makespan', float('inf'))
                    if current_makespan < self.best_makespan:
                        self.best_makespan = current_makespan
                        self.best_actions = episode_actions.copy()
                        logger.info(f"New best makespan: {self.best_makespan}")
                    break
                
                # Add safety check for maximum steps
                if step_count > 1000:  # Adjust this value based on your environment
                    logger.warning("Episode exceeded maximum steps, forcing termination")
                    break
                    
        except Exception as e:
            logger.error(f"Error during rollout: {str(e)}")
            raise
            
        logger.info(f"Rollout completed with reward: {episode_reward}")
        return episode_reward, episode_actions

    def learn(self, nr_of_episodes: int) -> Tuple[List[int], float]:
        """Main training loop with improved logging"""
        logger.info(f"Starting training for {nr_of_episodes} episodes...")
        
        for episode in tqdm(range(nr_of_episodes), 
                          desc="Training", 
                          position=0, 
                          leave=True):
            logger.info(f"Starting episode {episode + 1}/{nr_of_episodes}")
            episode_reward, actions = self.rollout()
            self.episode_rewards.append(episode_reward)
            
            logger.info(f"Episode {episode + 1} completed:")
            logger.info(f"- Reward: {episode_reward}")
            logger.info(f"- Actions taken: {len(actions)}")
            logger.info(f"- Current epsilon: {self.epsilon.item():.3f}")
            logger.info(f"- Best makespan: {self.best_makespan}")
            
        return self.best_actions, self.best_makespan

class Policy:
    """GPU-accelerated Policy class"""
    
    def __init__(self, env, config):
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.config = config
        self.action_space_size = config['action_space_size']
        self.tsetlin_machines = {}
        self.feature_transformer = JSSPFeatureTransformer(config)
        self.nr_of_clauses = min(config['nr_of_clauses'], 1000)
        self.T = config['T']
        self.s = config['s']

    def predict(self, state):
        """GPU-accelerated prediction"""
        if len(state) != self.config['obs_space_size']:
            raise ValueError(f"State size mismatch")
        
        binary_features = self.feature_transformer.transform(state)
        binary_features = torch.tensor(binary_features, device=device)
        valid_actions = self.get_valid_actions()
        q_values = torch.full((self.action_space_size,), float('-inf'), device=device)
        
        for action in valid_actions:
            tm = self._get_or_create_tm(action)
            clause_outputs = self.compute_clause_outputs(tm, binary_features)


            clause_outputs = clause_outputs.clone().detach().to(device)
            q_values[action] = self.calculate_q_value(clause_outputs, tm.clause_sign)
        
        return q_values
    
    def calculate_q_value(self, clause_outputs: torch.Tensor, clause_signs: torch.Tensor) -> float:
        """Calculate Q-value from clause outputs
        
        Args:
            clause_outputs: Tensor of clause outputs
            clause_signs: Tensor of clause signs
            
        Returns:
            float: Calculated Q-value
        """
        # Ensure inputs are on the correct device
        clause_outputs = clause_outputs.to(device)
        clause_signs = clause_signs.to(device)
        
        # Calculate sums using tensor operations
        sum_positive = torch.sum(clause_outputs[clause_signs == 1])
        sum_negative = torch.sum(clause_outputs[clause_signs == -1])
        
        # Convert result to CPU and float
        return float((sum_positive - sum_negative).cpu().item())
        
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
    
    def compute_clause_outputs(self, tm: TsetlinAutomata, features: torch.Tensor) -> torch.Tensor:
        """Compute clause outputs for given features
        
        Args:
            tm: TsetlinAutomata instance
            features: Input features tensor
            
        Returns:
            torch.Tensor: Clause outputs
        """
        # Initialize outputs tensor on GPU
        clause_outputs = torch.zeros(tm.number_of_clauses, dtype=torch.float32, device=device)
        
        # Ensure features is a tensor on the correct device
        features = features.to(device)
        
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
        state = np.array(state, dtype=np.float32)

        if len(state) != self.expected_state_size:
            raise ValueError(
                f"State size mismatch. Expected {self.expected_state_size}, "
                f"got {len(state)}. State shape: {state.shape}"
            )
            
        binary_features = np.zeros(self.total_features, dtype=np.float32)

        # Initialize index counter
        idx = 0

        # 1. Process machine times (first n_machines elements)
        machine_times = state[:self.machine_times_size]
        max_time = np.max(machine_times) if np.max(machine_times) > 0 else 1
        normalized_times = (machine_times / max_time * (2**self.machine_time_bits - 1)).astype(int)
        
        for time in normalized_times:
            binary = format(min(time, 2**self.machine_time_bits - 1), 
                        f'0{self.machine_time_bits}b')
            for bit in binary:
                binary_features[idx] = float(bit)
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
    """GPU-accelerated experience replay buffer"""
    
    def __init__(self, buffer_size: int, sample_size: int, n: int = 1):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.n = n
        
        # Initialize buffers on GPU
        self.actions = torch.zeros(buffer_size, dtype=torch.int32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.terminated = torch.zeros(buffer_size, dtype=torch.int8, device=device)
        self.truncated = torch.zeros(buffer_size, dtype=torch.int8, device=device)
        
        self.cur_obs = []
        self.next_obs = []
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
        project="jssp_QTM_gpu",
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

def setup_config():
    """Setup configuration without argument parsing"""
    config = {
        # Instance configuration
        'instance_name': 'ta01.txt',  # Change this to your instance name
        'episodes': 1,
        
        # Model parameters
        'nr_of_clauses': 2000,
        'T': 1500,  # Threshold parameter
        's': 1.5,   # Specificity parameter
        
        # Training parameters
        'bits_per_machine_time': 8,
        'bits_per_job_status': 4,
        'gamma': 0.99,
        'epsilon_init': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'sample_size': 64,
        'train_freq': 100,
        'test_freq': 5,
        
        # System settings
        'seed': 42,
        'save': True,
        'wandb_enabled': False,  # Set to True to enable W&B logging
        'verbose': True,
        
    }
    
    return config

def main():
    # Setup configuration
    config = setup_config()
    
    # Setup logging
    setup_logging(config['verbose'])
    
    try:
        # Set random seeds
        set_seeds(config['seed'])
        
        # Create environment
        logging.info(f"Creating environment for instance: {config['instance_name']}")
        if config['instance_name'].startswith('ta'):
            # Create the base environment first
            base_env = TaillardGymGenerator.create_env_from_instance(config['instance_name'])
            
            # Get environment information
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
        
        # Update config with environment info
        config.update({
            'env_name': 'job-shop',
            'algorithm': 'QTM',
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'action_space_size': env.action_space.n,
            'obs_space_size': env.observation_space.shape[0],
        })
        
        # Print problem configuration
        logging.info("\nProblem Configuration:")
        logging.info(f"Instance: {config['instance_name']}")
        logging.info(f"Jobs: {config['n_jobs']}")
        logging.info(f"Machines: {config['n_machines']}")
        logging.info(f"Action Space Size: {config['action_space_size']}")
        logging.info(f"Observation Space Size: {config['obs_space_size']}")
        logging.info(f"Clauses per TM: {config['nr_of_clauses']}")
        logging.info(f"Threshold T: {config['T']}")
        logging.info(f"Specificity s: {config['s']}")
        
        # Log GPU information
        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("GPU not available, using CPU")
        
        # Initialize W&B if enabled
        if config['wandb_enabled']:
            init_wandb(config, config['instance_name'])
        
        # Create and train agent
        logging.info("\nInitializing QTM agent...")
        agent = QTMAgent(env, config)
        
        logging.info("\nStarting training...")
        best_actions, best_makespan = agent.learn(nr_of_episodes=config['episodes'])
        
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
            'instance': config['instance_name'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'episodes': config['episodes'],
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
        result_file = Path(f'results/{config["instance_name"][:-4]}/qtm_{time.strftime("%Y%m%d_%H%M%S")}.json')
        result_file.parent.mkdir(parents=True, exist_ok=True)
        save_results(result_file, results)
        
        # Log to W&B if enabled
        if config['wandb_enabled']:
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
        if config['verbose']:
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        if config['wandb_enabled'] and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()