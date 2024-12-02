# Part 1: Imports and Initial Setup
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
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
from pathlib import Path
import json
import random
import time
from gymnasium.envs.registration import register
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import socket
import torch.nn as nn

DTYPE_FLOAT = torch.float32
DTYPE_INT = torch.int32

wandb.login(key="22a8b69ab5255b120ca37b40c2f998f71db3c615")

# Device configuration
logger = logging.getLogger(__name__)

def get_device(rank=None):
    if torch.cuda.is_available():
        if rank is not None:
            return torch.device(f'cuda:{rank}')
        else:
            return torch.device('cuda')
    return torch.device('cpu')

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup(rank, world_size):
    """
    Setup the distributed training environment
    """
    # Check if process group is already initialized
    if dist.is_initialized():
        return
    
    try:
        # Set static port for initial process group setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(29500)

        # Initialize the process group first
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            init_method="env://"
        )

        # Set GPU device
        torch.cuda.set_device(rank)

    except Exception as e:
        print(f"Error initializing process group on rank {rank}: {str(e)}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
    
def cleanup():

    if dist.is_initialized():
        dist.destroy_process_group()

# Add at the top of the file, after imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory

# Add GPU memory management function
def manage_gpu_memory():
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        # Set memory allocator settings
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory
        # Print memory stats
        print(f"GPU Memory Summary:")
        print(f"Allocated: {torch.cuda.memory_allocated(1)/1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(1)/1024**3:.2f} GB")

def setup_distributed():
    # Set environment variables if not already set
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    # Initialize the process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )

def binary_encode_gpu(value: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Convert a value to its binary representation using GPU operations"""
    value = value.to(DTYPE_INT)
    bits = torch.zeros(num_bits, dtype=DTYPE_INT, device=value.device)
    
    for i in range(num_bits):
        # Use modulo and division instead of bitwise operations
        bits[i] = value % 2
        value = value // 2
        
    return bits

def create_environment(instance_name):
    """Create and register the job shop environment"""
    try:
        # Create base environment from Taillard instance
        base_env = TaillardGymGenerator.create_env_from_instance(f'{instance_name}.txt')
        
        # Unregister if environment already exists
        try:
            gym.envs.registration.registry.pop('JobShop-Taillard-v0')
        except KeyError:
            pass
        
        # Register environment with proper parameters
        register(
            id='JobShop-Taillard-v0',
            entry_point='job_shop_env:JobShopGymEnv',
            kwargs={
                'jobs': base_env.jobs,
                'render_mode': None
            }
        )
        
        # Create environment instance
        env = gym.make('JobShop-Taillard-v0')
        
        # Apply the wrapper
        env = DTypeWrapper(env)
        
        # Verify observation space and types
        test_obs, _ = env.reset()
        assert test_obs.dtype == np.float32, f"Observation dtype is {test_obs.dtype}, expected float32"
        assert env.observation_space.contains(test_obs), "Initial observation out of bounds"
        
        return env, base_env
    
    except Exception as e:
        raise RuntimeError(f"Failed to create environment: {str(e)}") from e

def verify_environment(env):
    """Verify environment setup and observation space"""
    logger.info("Verifying environment setup...")
    
    # Check initial observation
    obs, _ = env.reset()
    logger.info(f"Observation dtype: {obs.dtype}")
    logger.info(f"Observation shape: {obs.shape}")
    logger.info(f"Observation space: {env.observation_space}")
    
    # Check observation bounds
    low = env.observation_space.low
    high = env.observation_space.high
    
    # Verify observation is within bounds
    within_bounds = np.all(obs >= low) and np.all(obs <= high)
    logger.info(f"Observation within bounds: {within_bounds}")
    
    # Take a test step
    action = env.action_space.sample()
    step_obs, _, _, _, _ = env.step(action)
    logger.info(f"Step observation dtype: {step_obs.dtype}")
    
    return {
        'initial_obs_dtype': obs.dtype,
        'step_obs_dtype': step_obs.dtype,
        'within_bounds': within_bounds
    }
class TsetlinAutomata:
    """GPU-optimized Tsetlin Machine implementation"""
    
    def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, device=None):
        self.number_of_clauses = number_of_clauses
        self.T = T
        self.s = s
        self.boost_true_positive_feedback = boost_true_positive_feedback

        # Set device (CPU or GPU)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with None
        self.ta_state = None
        self.clause_sign = None
        self.clause_output = None
        self.feedback_to_clauses = None
    
    def initialize(self, number_of_features):
        """Initialize using GPU tensors"""
        self.number_of_features = number_of_features
        
        # Initialize tensors on GPU
        self.ta_state = torch.ones((self.number_of_clauses, 2, self.number_of_features), 
                                 dtype=torch.int8, device=self.device) * 16
        self.clause_sign = torch.ones(self.number_of_clauses, 
                                    dtype=torch.int8, device=self.device)
        self.clause_output = torch.zeros(self.number_of_clauses, 
                                       dtype=torch.int8, device=self.device)
        self.feedback_to_clauses = torch.zeros(self.number_of_clauses, 
                                             dtype=torch.int8, device=self.device)
    
    def to(self, device):
        """Move the model to specified device"""
        self.device = device
        if self.ta_state is not None:
            self.ta_state = self.ta_state.to(device)
            self.clause_sign = self.clause_sign.to(device)
            self.clause_output = self.clause_output.to(device)
            self.feedback_to_clauses = self.feedback_to_clauses.to(device)
        return self
        
    
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current parameters of the Tsetlin Machine"""
        return self.ta_state, self.clause_sign, self.clause_output, self.feedback_to_clauses
        
    def set_params(self, ta_state: torch.Tensor, clause_sign: torch.Tensor, 
                  clause_output: torch.Tensor, feedback_to_clauses: torch.Tensor):
        """Set parameters of the Tsetlin Machine"""
        self.ta_state = ta_state.to(DEVICE)
        self.clause_sign = clause_sign.to(DEVICE)
        self.clause_output = clause_output.to(DEVICE)
        self.feedback_to_clauses = feedback_to_clauses.to(DEVICE)

class JSSPFeatureTransformer:
    def __init__(self, config):
        self.config = config
        
        # Basic dimensions
        self.machine_time_bits = min(config['bits_per_machine_time'], 4)
        self.job_status_bits = min(config['bits_per_job_status'], 2)
        self.n_machines = config['n_machines']
        self.n_jobs = config['n_jobs']
        
        self.expected_state_size = config['obs_space_size']
        
        # Calculate component sizes
        self.machine_times_size = self.n_machines
        self.job_status_size = self.n_jobs
        self.op_status_size = self.n_jobs * self.n_machines
        self.remaining_size = (self.expected_state_size - 
                             (self.machine_times_size + 
                              self.job_status_size + 
                              self.op_status_size))
        
        # Calculate total binary features
        self.total_features = (
            self.n_machines * self.machine_time_bits +
            self.n_jobs * self.job_status_bits +
            self.op_status_size +
            self.remaining_size
        )

        #store device reference
        self.device = None

    def to(self, device):
        """Set device for tensor operations"""
        self.device = device
        return self

    def transform(self, state: np.ndarray) -> torch.Tensor:
        """Transform JSSP state into binary features using GPU"""
        if len(state) != self.expected_state_size:
            raise ValueError(
                f"State size mismatch. Expected {self.expected_state_size}, "
                f"got {len(state)}."
            )

        if self.device is None:
            raise ValueError("Device not set. Call .to(device) first.")
        
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        
        # Convert to PyTorch tensor and move to device
        state_tensor = torch.as_tensor(state, dtype=DTYPE_FLOAT, device=self.device)

        binary_features = torch.zeros(self.total_features, dtype=DTYPE_INT, device=self.device)
        idx = 0

        
        # 1. Process machine times
        machine_times = state_tensor[:self.machine_times_size]
        max_time = torch.max(machine_times)
        if max_time > 0:
            normalized_times = ((machine_times / max_time) * 
                              (2**self.machine_time_bits - 1)).to(DTYPE_INT)
            
            for time in normalized_times:
                bits = binary_encode_gpu(time, self.machine_time_bits)
                binary_features[idx:idx + self.machine_time_bits] = bits
                idx += self.machine_time_bits
        else:
            idx += self.machine_time_bits * self.n_machines
        
        # 2. Process job status
        job_status = state_tensor[self.machine_times_size:
                                self.machine_times_size + self.job_status_size]
        
        for job_stat in job_status:
            status = torch.min(
                (job_stat * (2**self.job_status_bits - 1)).to(DTYPE_INT),
                torch.tensor(2**self.job_status_bits - 1, dtype=DTYPE_INT, device=self.device)
            )
            bits = binary_encode_gpu(status, self.job_status_bits)
            binary_features[idx:idx + self.job_status_bits] = bits
            idx += self.job_status_bits
        
        # 3. Process operation status matrix (already binary)
        op_status_start = self.machine_times_size + self.job_status_size
        op_status_end = op_status_start + self.op_status_size
        op_status = state_tensor[op_status_start:op_status_end].to(DTYPE_INT)
        
        binary_features[idx:idx + self.op_status_size] = op_status
        idx += self.op_status_size
        
        # 4. Process remaining state information
        remaining_state = state_tensor[op_status_end:].to(DTYPE_INT)
        binary_features[idx:] = remaining_state
        
        return binary_features

class Policy(nn.Module):
    """GPU-accelerated Policy class implementing Tsetlin Machine for JSSP"""
    
    def __init__(self, env, config):
        super(Policy, self).__init__()  # Initialize parent nn.Module
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.config = config
        
        self.action_space_size = config['action_space_size']
        self.tsetlin_machines = {}

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_transformer = JSSPFeatureTransformer(config).to(self.device)
        
        self.nr_of_clauses = min(config['nr_of_clauses'], 1000)
        self.T = config['T']
        self.s = config['s']
        
        # Register tensors as parameters so they're properly handled by DDP
        self.register_buffer('T_tensor', torch.tensor(self.T))
        self.register_buffer('s_tensor', torch.tensor(self.s))

    def forward(self, state):
        """Forward pass required by nn.Module - wrap predict method"""
        return self.predict(state)

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
        """GPU-accelerated prediction for valid actions only"""
        if len(state) != self.config['obs_space_size']:
            raise ValueError(
                f"State size mismatch. Expected {self.config['obs_space_size']}, "
                f"got {len(state)}."
            )
        
        binary_features = self.feature_transformer.transform(state).to(self.T_tensor.device).clone().detach()
        valid_actions = self.get_valid_actions()
        q_values = torch.full((self.action_space_size,), float('-inf'), device=self.T_tensor.device)
        
        for action in valid_actions:
            tm = self._get_or_create_tm(action)
            clause_outputs = self.compute_clause_outputs(tm, binary_features)
            q_values[action] = self.calculate_q_value(
                clause_outputs, 
                tm.clause_sign
            )
        
        return q_values

    def compute_clause_outputs(self, tm: TsetlinAutomata, 
                             features: torch.Tensor) -> torch.Tensor:
        """Compute clause outputs using GPU tensors"""
        # Move computation to GPU and use vectorized operations
        features_gpu = features.to(self.T_tensor.device)
        
        # Reshape for broadcasting
        features_expanded = features_gpu.unsqueeze(0).expand(tm.number_of_clauses, -1)
        
        # Calculate include and exclude conditions
        include_mask = tm.ta_state[:, 0, :] > 16
        exclude_mask = tm.ta_state[:, 1, :] > 16
        
        # Compute violations for include and exclude conditions
        include_violations = include_mask & (features_expanded == 0)
        exclude_violations = exclude_mask & (features_expanded == 1)
        
        # Combine violations
        any_violations = torch.any(include_violations | exclude_violations, dim=1)
        
        # Create output tensor
        clause_outputs = (~any_violations).to(torch.int32)
        
        return clause_outputs

    def calculate_q_value(self, clause_outputs: torch.Tensor, 
                         clause_signs: torch.Tensor) -> torch.Tensor:
        """Calculate Q-value using GPU tensor operations"""
        pos_contribution = clause_outputs[clause_signs == 1].sum()
        neg_contribution = clause_outputs[clause_signs == -1].sum()
        return pos_contribution - neg_contribution

    def update(self, tm_inputs: Dict[int, Dict[str, List]]):
        """Update Tsetlin Machines using GPU acceleration"""
        for action_id, data in tm_inputs.items():
            if not data['observations']:
                continue
                
            tm = self._get_or_create_tm(action_id)
            
            # Convert to tensors and move to GPU
            observations = [
                self.feature_transformer.transform(obs).to(self.device).clone().detach()
                for obs in data['observations']
            ]
            observations_tensor = torch.stack(observations)
            targets_tensor = torch.tensor(data['target_q_vals'], device=self.device).clone().detach()
            
            # Batch update
            self.batch_update_tm(tm, observations_tensor, targets_tensor)

    def batch_update_tm(self, tm: TsetlinAutomata, 
                       binary_features: torch.Tensor, 
                       targets: torch.Tensor):
        """Batch update TM using GPU acceleration"""
        batch_size = binary_features.size(0)
        
        # Compute current predictions for batch
        predictions = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            clause_outputs = self.compute_clause_outputs(tm, binary_features[i])
            predictions[i] = self.calculate_q_value(clause_outputs, tm.clause_sign)
        
        # Compute feedback for each clause
        feedback = torch.zeros((batch_size, tm.number_of_clauses), 
                             dtype=torch.int8, 
                             device=self.device)
        
        # Prepare masks for positive and negative clauses
        pos_mask = tm.clause_sign == 1
        neg_mask = tm.clause_sign == -1
        
        for i in range(batch_size):
            feedback[i, pos_mask] = (targets[i] > predictions[i]).to(torch.int8)
            feedback[i, neg_mask] = (targets[i] < predictions[i]).to(torch.int8)
        
        # Update TA states
        self.update_ta_states_batch(tm, binary_features, feedback)

    def update_ta_states_batch(self, tm: TsetlinAutomata, 
                             features: torch.Tensor, 
                             feedback: torch.Tensor):
        """Update TA states for batch using GPU operations"""
        batch_size = features.size(0)
        
        # Generate random matrices for probabilistic updates
        rand_s = torch.rand((batch_size, tm.number_of_clauses, tm.number_of_features), 
                          device=self.device)
        rand_T = torch.rand((batch_size, tm.number_of_clauses, tm.number_of_features), 
                          device=self.device)
        
        # Expand feedback for broadcasting
        feedback_expanded = feedback.unsqueeze(-1).expand(-1, -1, tm.number_of_features)
        
        # Type I feedback (strengthening)
        type_I_mask = feedback_expanded == 1
        
        # Features present (1)
        features_present = features.unsqueeze(1).expand(-1, tm.number_of_clauses, -1) == 1
        
        # Update include actions
        include_conditions = type_I_mask & features_present
        tm.ta_state[:, 0, :] = torch.where(
            include_conditions.any(dim=0),
            torch.clamp(tm.ta_state[:, 0, :] - 1, min=1),
            tm.ta_state[:, 0, :]
        )
        
        # Update exclude actions with probability s
        exclude_updates = type_I_mask & ~features_present & (rand_s < self.s_tensor)
        tm.ta_state[:, 1, :] = torch.where(
            exclude_updates.any(dim=0),
            torch.clamp(tm.ta_state[:, 1, :] + 1, max=32),
            tm.ta_state[:, 1, :]
        )
        
        # Type II feedback (weakening)
        type_II_mask = ~type_I_mask
        
        # Update with probability 1/T
        weaken_updates = type_II_mask & (rand_T < (1.0 / self.T_tensor))
        
        tm.ta_state[:, 0, :] = torch.where(
            (weaken_updates & features_present).any(dim=0),
            torch.clamp(tm.ta_state[:, 0, :] + 1, max=32),
            tm.ta_state[:, 0, :]
        )
        
        tm.ta_state[:, 1, :] = torch.where(
            (weaken_updates & ~features_present).any(dim=0),
            torch.clamp(tm.ta_state[:, 1, :] + 1, max=32),
            tm.ta_state[:, 1, :]
        )

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from environment"""
        valid_actions = []
        for action_id in range(self.action_space_size):
            action = self.env_unwrapped._decode_action(action_id)
            if self.env_unwrapped._is_valid_action(action):
                valid_actions.append(action_id)
        return valid_actions

class QTMAgent:
    """GPU-accelerated Q-learning agent for Job Shop Scheduling"""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any], rank):
        self.device = get_device(rank)
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.rank = rank
        self.config = config

        # Move policy to specific GPU
        self.policy = Policy(env, config).to(self.device)
        
        # Reduce buffer sizes for GPU memory
        config['buffer_size'] = min(config['buffer_size'], 1000)
        config['sample_size'] = min(config['sample_size'], 32)

        # Initialize exploration parameters on GPU
        self.epsilon = torch.tensor(config['epsilon_init'], device=self.device)
        self.epsilon_min = torch.tensor(config['epsilon_min'], device=self.device)
        self.epsilon_decay = torch.tensor(config['epsilon_decay'], device=self.device)
        
        # Initialize training variables
        self.total_timesteps = 0
        self.best_makespan = float('inf')
        self.best_actions = []
        self.episode_rewards = []
        self.last_training_loss = 0.0
        
        # Create replay buffer with specific device
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'], 
            config['sample_size'],
            n=config.get('n_steps', 1),
            device=self.device  # Pass the device to ReplayBuffer
        )
        
        
        # Logging
        logger.info(f"Initialized QTM Agent on rank {rank}")
        logger.info(f"Replay buffer size: {config['buffer_size']}")
        
        if torch.cuda.is_available():
            logger.info(f"Using GPU {rank}: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated(self.device)/1024**2:.2f}MB")

    def get_next_action(self, cur_obs):
        """Select next action using epsilon-greedy strategy with GPU tensors"""
        if torch.rand(1, device=self.device).item() < self.epsilon:  # Changed DEVICE to self.device
            valid_actions = self.get_valid_actions()
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
        reward = torch.tensor(0.0, device=self.device)  # Changed DEVICE to self.device
        
        if not info.get('valid_action', True):
            return torch.tensor(-100.0, device=self.device)  # Changed DEVICE to self.device
            
        if info.get('job_completed', False):
            reward += torch.tensor(50.0, device=self.device)  # Changed DEVICE to self.device
            
        if info.get('operation_completed', False):
            reward += torch.tensor(10.0, device=self.device)  # Changed DEVICE to self.device
            
        current_makespan = info.get('makespan', 0)
        if current_makespan > 0:
            reward -= torch.tensor(current_makespan * 0.01, device=self.device)  # Changed DEVICE to self.device
            
        if current_makespan < self.best_makespan:
            improvement = self.best_makespan - current_makespan
            reward += torch.tensor(improvement * 0.5, device=self.device)  # Changed DEVICE to self.device
            
        return reward.item()

    def train(self):
        """Train the policy on a batch of experiences"""
        if not self.replay_buffer.is_ready:
            return
            
        self.replay_buffer.sample()
        
        # Initialize TM inputs dictionary
        tm_inputs = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(self.config['sample_size']):
            # Use clone().detach() for tensor construction
            next_obs = torch.as_tensor(self.replay_buffer.sampled_next_obs[i], 
                                device=self.device).clone().detach()  # Changed DEVICE to self.device
            next_q_vals = self.policy.predict(next_obs)
            next_q_val = torch.max(next_q_vals)
            
            reward = torch.as_tensor(self.replay_buffer.sampled_rewards[i], 
                                device=self.device).clone().detach()  # Changed DEVICE to self.device
            terminated = torch.as_tensor(self.replay_buffer.sampled_terminated[i], 
                                    device=self.device).clone().detach() 
                
            # Calculate target Q-value
            target_q_val = reward + (1 - terminated.float()) * self.config['gamma'] * next_q_val
            
            # Get current observation and action
            cur_obs = self.replay_buffer.sampled_cur_obs[i]
            action = self.replay_buffer.sampled_actions[i]
            
            if action not in tm_inputs:
                tm_inputs[action] = {
                    'observations': [],
                    'target_q_vals': []
                }
            
            tm_inputs[action]['observations'].append(cur_obs)
            tm_inputs[action]['target_q_vals'].append(target_q_val.item())
            
            # Calculate loss for monitoring
            current_q = self.policy.predict(
                 torch.as_tensor(cur_obs, device=self.device).clone().detach() 
            )[action]
            loss = (target_q_val - current_q) ** 2
            total_loss += loss
        
        # Update policy
        self.policy.update(tm_inputs)
        
        # Store average loss
        self.last_training_loss = (total_loss / self.config['sample_size']).item()

    def update_epsilon(self):
        """Update exploration rate using GPU tensor operations"""
        self.epsilon = torch.max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def rollout(self):
        """Execute one episode"""
        cur_obs, _ = self.env.reset()
        episode_reward = torch.tensor(0.0, device=self.device)
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
                
                # Log training metrics
                if wandb.run is not None:
                    wandb.log({
                        "timestep": self.total_timesteps,
                        "training_loss": self.last_training_loss,
                        "buffer_size": self.replay_buffer.count
                    })
            
            # Update exploration rate
            self.update_epsilon()
            
            # Update current observation
            cur_obs = next_obs
            
            if terminated or truncated:
                current_makespan = info.get('makespan', float('inf'))
                if current_makespan < self.best_makespan:
                    self.best_makespan = current_makespan
                    self.best_actions = episode_actions.copy()
                    logger.info(f"New best makespan: {self.best_makespan}")
                    
                    # Log best solution
                    if wandb.run is not None:
                        wandb.log({
                            "best_makespan": self.best_makespan,
                            "best_solution_length": len(self.best_actions)
                        })
                break
            
        return episode_reward.item(), episode_actions

    def learn(self, nr_of_episodes: int) -> Tuple[List[int], float]:
        """Main training loop"""
        logger.info(f"Starting training for {nr_of_episodes} episodes...")
        
        for episode in tqdm(range(nr_of_episodes), 
                          desc="Training", 
                          position=0, 
                          leave=True):
            episode_reward, actions = self.rollout()
            self.episode_rewards.append(episode_reward)
            
            # Get current makespan
            _, _, _, _, info = self.env.step(actions[-1])
            current_makespan = info.get('makespan', float('inf'))
            
            # Log metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    "episode": episode,
                    "reward": episode_reward,
                    "makespan": current_makespan,
                    "epsilon": self.epsilon.item(),
                    "best_makespan": self.best_makespan
                })
            
            if (episode + 1) % 10 == 0:
                mean_reward = torch.tensor(self.episode_rewards[-10:], 
                                        device=self.device).mean()
                logger.info(
                    f"Episode {episode + 1}/{nr_of_episodes}, "
                    f"Mean Reward: {mean_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}, "
                    f"Best Makespan: {self.best_makespan}"
                )
        
        return self.best_actions, self.best_makespan

class ReplayBuffer:
    """GPU-accelerated experience replay buffer"""
    
    def __init__(self, buffer_size: int, sample_size: int, n: int = 1, device=None):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.n = n
        self.device = device if device is not None else get_device()
        
        # Initialize buffers on specified device with proper dtypes
        self.actions = torch.zeros(buffer_size, dtype=DTYPE_INT, device=self.device)
        self.rewards = torch.zeros(buffer_size, dtype=DTYPE_FLOAT, device=self.device)
        self.terminated = torch.zeros(buffer_size, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(buffer_size, dtype=torch.bool, device=self.device)
        
        self.cur_obs = []
        self.next_obs = []
        
        self.count = 0
        self.current_idx = 0
        
        # Initialize sample storage
        self.sampled_actions = None
        self.sampled_cur_obs = None
        self.sampled_next_obs = None
        self.sampled_rewards = None
        self.sampled_terminated = None
        self.sampled_truncated = None
        
        logger.info(f"Initialized replay buffer with size {buffer_size} on device {self.device}")

    def save_experience(self, action: int, cur_obs: np.ndarray, next_obs: np.ndarray,
                       reward: float, terminated: bool, truncated: bool) -> None:
        """Save a single experience to the buffer"""
        idx = self.current_idx
        
        # Convert numpy arrays to tensors with proper dtype
        cur_obs = torch.from_numpy(cur_obs.astype(np.float32)).to(self.device)
        next_obs = torch.from_numpy(next_obs.astype(np.float32)).to(self.device)
        
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
        if not self.is_ready:
            logger.warning(f"Not enough experiences in buffer. Have {self.count}, need {self.sample_size}")
            return
        
        indices = torch.randint(
            self.count, (self.sample_size,), 
            device=self.device
        )
        
        # Store sampled experiences
        self.sampled_actions = self.actions[indices]
        self.sampled_rewards = self.rewards[indices]
        self.sampled_terminated = self.terminated[indices]
        self.sampled_truncated = self.truncated[indices]
        
        # Handle observations
        self.sampled_cur_obs = [self.cur_obs[i.item()] for i in indices]
        self.sampled_next_obs = [self.next_obs[i.item()] for i in indices]
        
        logger.debug(f"Sampled {self.sample_size} experiences from buffer")

    def compute_n_step_returns(self, gamma: float) -> torch.Tensor:
        """Compute n-step returns for sampled experiences"""
        n_step_returns = torch.zeros(
            self.sample_size, dtype=DTYPE_FLOAT, device=self.device
        )
        
        for i in range(self.sample_size):
            ret = torch.tensor(0.0, dtype=DTYPE_FLOAT, device=self.device)
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
        
        self.actions.zero_()
        self.rewards.zero_()
        self.terminated.zero_()
        self.truncated.zero_()
        
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

    def get_samples(self) -> Dict[str, torch.Tensor]:
        """Get current samples as dictionary"""
        return {
            'actions': self.sampled_actions,
            'cur_obs': self.sampled_cur_obs,
            'next_obs': self.sampled_next_obs,
            'rewards': self.sampled_rewards,
            'terminated': self.sampled_terminated,
            'truncated': self.sampled_truncated
        }

    def set_samples(self, samples: Dict[str, torch.Tensor]) -> None:
        """Set samples from dictionary"""
        self.sampled_actions = samples['actions']
        self.sampled_cur_obs = samples['cur_obs']
        self.sampled_next_obs = samples['next_obs']
        self.sampled_rewards = samples['rewards']
        self.sampled_terminated = samples['terminated']
        self.sampled_truncated = samples['truncated']

    def combine_samples(self, samples_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine samples from multiple GPUs"""
        combined = {
            'actions': torch.cat([s['actions'] for s in samples_list]),
            'cur_obs': sum([s['cur_obs'] for s in samples_list], []),
            'next_obs': sum([s['next_obs'] for s in samples_list], []),
            'rewards': torch.cat([s['rewards'] for s in samples_list]),
            'terminated': torch.cat([s['terminated'] for s in samples_list]),
            'truncated': torch.cat([s['truncated'] for s in samples_list])
        }
        return combined


    def get_samples(self) -> Dict[str, torch.Tensor]:
        """Get current samples as dictionary"""
        return {
            'actions': self.sampled_actions,
            'cur_obs': self.sampled_cur_obs,
            'next_obs': self.sampled_next_obs,
            'rewards': self.sampled_rewards,
            'terminated': self.sampled_terminated,
            'truncated': self.sampled_truncated
        }

    def set_samples(self, samples: Dict[str, torch.Tensor]) -> None:
        """Set samples from dictionary"""
        self.sampled_actions = samples['actions']
        self.sampled_cur_obs = samples['cur_obs']
        self.sampled_next_obs = samples['next_obs']
        self.sampled_rewards = samples['rewards']
        self.sampled_terminated = samples['terminated']
        self.sampled_truncated = samples['truncated']

    def combine_samples(self, samples_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine samples from multiple GPUs"""
        combined = {
            'actions': torch.cat([s['actions'] for s in samples_list]),
            'cur_obs': sum([s['cur_obs'] for s in samples_list], []),
            'next_obs': sum([s['next_obs'] for s in samples_list], []),
            'rewards': torch.cat([s['rewards'] for s in samples_list]),
            'terminated': torch.cat([s['terminated'] for s in samples_list]),
            'truncated': torch.cat([s['truncated'] for s in samples_list])
        }
        return combined

class DTypeWrapper(gym.Wrapper):
    """Wrapper to ensure consistent data types"""
    
    def __init__(self, env):
        super().__init__(env)

         # Get the original observation space bounds
        orig_low = env.observation_space.low
        orig_high = env.observation_space.high

        # Convert bounds to float32 and create new observation space
        self.observation_space = gym.spaces.Box(
            low=orig_low.astype(np.float32),
            high=orig_high.astype(np.float32),
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Clip observation to be within bounds and ensure float32 type
        obs = np.clip(
            obs.astype(np.float32),
            self.observation_space.low,
            self.observation_space.high
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Clip observation to be within bounds and ensure float32 type
        obs = np.clip(
            obs.astype(np.float32),
            self.observation_space.low,
            self.observation_space.high
        )
        return obs, info
    
def train_on_gpu(rank, world_size, instance_name):
    setup(rank, world_size)
    device = get_device(rank)
    
    try:
        print(f"Starting training on GPU {rank}")
        
        # Create environment and agent setup
        env, base_env = create_environment(instance_name)
        config = create_config(env, base_env, obs_space_size)
        config['batch_size'] = config['batch_size'] // world_size  # Adjust batch size for each GPU

        #pass rank to QTMAgent
        agent = QTMAgent(env, config, rank)
        
        # Train
        best_actions, best_makespan = agent.learn(nr_of_episodes=1000)

        if world_size > 1:
            # Gather results from all GPUs
            makespans = [None] * world_size
            actions = [None] * world_size
            
            dist.all_gather_object(makespans, best_makespan)
            dist.all_gather_object(actions, best_actions)
            
            # Find best result
            best_idx = np.argmin(makespans)
            best_makespan = makespans[best_idx]
            best_actions = actions[best_idx]
        
        # Save results only from main process
        if rank == 0:
            save_results(agent, best_actions, best_makespan, instance_name)
            
    except Exception as e:
        print(f"Error on GPU {rank}: {str(e)}")
        torch.cuda.empty_cache()
        
    finally:
        cleanup()

# def print_problem_config(instance: str, config: Dict, env) -> None:
#     """Print detailed problem configuration with GPU info"""
#     logger.info("\n" + "="*50)
#     logger.info("PROBLEM CONFIGURATION")
#     logger.info("="*50)
    
#     # Instance details
#     logger.info("\nInstance Details:")
#     logger.info(f"- Name: {instance}")
#     logger.info(f"- Jobs: {config['n_jobs']}")
#     logger.info(f"- Machines: {config['n_machines']}")
#     logger.info(f"- Action Space Size: {config['action_space_size']}")
#     logger.info(f"- Observation Space Size: {config['obs_space_size']}")
    
#     # Model parameters
#     logger.info("\nModel Parameters:")
#     logger.info(f"- Clauses per TM: {config['nr_of_clauses']}")
#     logger.info(f"- Threshold T: {config['T']}")
#     logger.info(f"- Specificity s: {config['s']}")
#     logger.info(f"- Gamma: {config['gamma']}")
#     logger.info(f"- Epsilon: {config['epsilon_init']} → {config['epsilon_min']}")
    
#     # Memory configuration
#     logger.info("\nMemory Configuration:")
#     logger.info(f"- Buffer Size: {config['buffer_size']}")
#     logger.info(f"- Sample Size: {config['sample_size']}")
    
#     if torch.cuda.is_available():
#         # GPU details
#         logger.info("\nGPU Configuration:")
#         logger.info(f"- Device: {torch.cuda.get_device_name(0)}")
#         logger.info(f"- Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
#         logger.info(f"- Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
#         logger.info(f"- Memory Cached: {torch.cuda.memory_cached()/1024**2:.2f} MB")

# def log_training_metrics(episode: int, episode_reward: float, makespan: float, 
#                         epsilon: float, agent: 'QTMAgent'):
#     """Log training metrics to wandb"""
#     metrics = {
#         "episode": episode,
#         "reward": episode_reward,
#         "makespan": makespan,
#         "epsilon": epsilon,
#         "best_makespan": agent.best_makespan,
#     }

#     # Add GPU metrics if available
#     if torch.cuda.is_available():
#         metrics.update({
#             "gpu_utilization": torch.cuda.utilization(),
#             "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
#             "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2,    # MB
#         })

#     wandb.log(metrics)
    
def init_wandb(config: Dict[str, Any], instance_name: str, device=None):
    """Initialize Weights & Biases with GPU monitoring"""
    # Get device if not provided
    if device is None:
        device = get_device()

    wandb.init(
        project="gpu-jssp-qtm",
        name=f"GPU_QTM_{instance_name}_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "GPU-QTM",
            "instance": instance_name,
            "hardware": {
                "device": device.type,
                "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "None",
            },
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
        tags=["GPU-QTM", instance_name, "job-shop"]
    )

    # Initialize wandb GPU monitoring
    if torch.cuda.is_available():
        wandb.watch_called = False
        wandb.define_metric("gpu_utilization", summary="mean")
        wandb.define_metric("gpu_memory_allocated", summary="max")
        wandb.define_metric("gpu_memory_reserved", summary="max")
    
# def diagnose_jssp_environment(instance_name: str):
    """Diagnostic function to verify JSSP environment creation and GPU configuration"""
    logger.info("\n" + "="*50)
    logger.info("JSSP ENVIRONMENT DIAGNOSTICS")
    logger.info("="*50)
    
    # Check GPU availability
    logger.info("\n0. GPU Configuration:")
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA version: {torch.version.cuda}")
        logger.info(f"✓ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB")
    else:
        logger.warning("✗ No GPU available, using CPU")
    
    # Rest of the diagnostic checks...
    try:
        base_env = TaillardGymGenerator.create_env_from_instance(instance_name)
        env = gym.make('JobShop-Taillard-v0')
        
        # Additional GPU-specific checks
        if torch.cuda.is_available():
            # Test GPU tensor creation
            test_tensor = torch.zeros(100, device=self.device)
            logger.info("✓ Successfully created GPU tensor")
            
            # Test moving environment state to GPU
            initial_state, _ = env.reset()
            state_tensor = torch.tensor(initial_state, device=self.device)
            logger.info("✓ Successfully moved state to GPU")
            
            del test_tensor, state_tensor  # Clean up GPU memory
            torch.cuda.empty_cache()
    
        return base_env, env
        
    except Exception as e:
        logger.error(f"Environment diagnostics failed: {str(e)}")
        return None, None

# def inspect_taillard_instance(instance_name: str):
    """Inspects and validates a Taillard instance with GPU memory management"""
    logger.info("\n=== Taillard Instance Inspection ===")
    
    try:
        base_env = TaillardGymGenerator.create_env_from_instance(instance_name)
        
        n_jobs = len(base_env.jobs)
        n_machines = base_env.num_machines
        
        # Clean any GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return n_jobs, n_machines, base_env.jobs
        
    except Exception as e:
        logger.error(f"Error inspecting Taillard instance: {str(e)}")
        return None, None, None

# def validate_environment_creation(env):
#     """Validates the created environment and GPU configuration"""
#     logger.info("\n=== Environment Validation ===")
    
#     try:
#         # Check environment attributes
#         logger.info(f"\nEnvironment Type: {type(env)}")
#         logger.info(f"Action Space: {env.action_space}")
#         logger.info(f"Observation Space: {env.observation_space}")
        
#         # GPU-specific validation
#         if torch.cuda.is_available():
#             logger.info("\nGPU Validation:")
#             # Test state tensor creation
#             initial_state, _ = env.reset()
#             state_tensor = torch.tensor(initial_state, device=DEVICE)
#             logger.info(f"✓ State tensor shape on GPU: {state_tensor.shape}")
#             logger.info(f"✓ State tensor device: {state_tensor.device}")
            
#             # Clean up GPU memory
#             del state_tensor
#             torch.cuda.empty_cache()
        
#         return True
#     except Exception as e:
#         logger.error(f"Environment validation failed: {str(e)}")
#         return False
    
# def setup_logging(verbose: bool = False) -> None:
#     """Setup logging configuration"""
#     log_level = logging.DEBUG if verbose else logging.INFO
#     logging.basicConfig(
#         level=log_level,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )

# def set_seeds(seed: int = 42) -> None:
#     """Set random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # for multi-GPU
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

def save_results(agent, best_actions, best_makespan, instance_name: str, rank: int = 0) -> None:
    """
    Save results to file with proper JSON encoding
    
    Args:
        agent: The QTM agent
        best_actions: List of best actions found
        best_makespan: Best makespan achieved
        instance_name: Name of the problem instance
        rank: GPU rank (default 0 for main process)
    """
    try:
        # Only save results from the main process (rank 0)
        if rank != 0:
            return

        # Create results directory if it doesn't exist
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"{instance_name}_{timestamp}.json"

        # Collect schedule data
        schedule_data = []
        agent.env.reset()
        
        for action_id in best_actions:
            obs, reward, done, truncated, info = agent.env.step(action_id)
            action = agent.env.unwrapped._decode_action(action_id)
            
            schedule_data.append({
                'job': int(action.job),
                'machine': int(action.machine),
                'operation': int(action.operation),
                'start': int(info.get('start_time', 0)),
                'duration': int(agent.env.unwrapped.jobs[action.job].operations[action.operation].duration),
                'end': int(info.get('end_time', 0))
            })

        # Prepare results dictionary
        results = {
            'instance_name': instance_name,
            'timestamp': timestamp,
            'best_makespan': best_makespan,
            'num_actions': len(best_actions),
            'schedule_data': schedule_data,
            'config': agent.config,
            'gpu_count': torch.cuda.device_count(),
            'training_stats': {
                'final_memory_size': agent.replay_buffer.count if hasattr(agent, 'replay_buffer') else None,
                'total_actions': len(best_actions)
            }
        }

        # Save to file
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logging.info(f"Results saved to: {result_file}")
        
        # Create a "latest" symlink
        latest_link = result_file.parent / f"{instance_name}_latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(result_file.name)
        
        # Save model state if available
        if hasattr(agent, 'policy'):
            model_file = results_dir / f"model_{instance_name}_{timestamp}.pt"
            torch.save({
                'model_state_dict': agent.policy.state_dict(),
                'best_makespan': best_makespan,
                'config': agent.config
            }, model_file)
            logging.info(f"Model saved to: {model_file}")

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
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)
    
# def execute_best_solution(env, best_actions):
#     """Execute and verify the best solution"""
#     env.reset()
#     schedule_data = []
#     completed_operations = 0
#     total_operations = env.unwrapped.n_jobs * env.unwrapped.n_machines
    
#     logger.info(f"\nExecuting best solution ({len(best_actions)} actions):")
#     logger.info(f"Expected operations: {total_operations}")
    
#     for action_id in best_actions:
#         action = env.unwrapped._decode_action(action_id)
#         obs, reward, done, truncated, info = env.step(action_id)
        
#         if info.get('operation_completed', False):
#             completed_operations += 1
        
#         schedule_data.append({
#             'job': int(action.job),
#             'machine': int(action.machine),
#             'operation': int(action.operation),
#             'start': int(info.get('start_time', 0)),
#             'duration': int(env.unwrapped.jobs[action.job].operations[action.operation].duration),
#             'end': int(info.get('end_time', 0))
#         })
        
#         if completed_operations % 10 == 0:
#             logger.info(f"Completed {completed_operations}/{total_operations} operations")
    
#     logger.info(f"\nFinal completion: {completed_operations}/{total_operations} operations")
#     return schedule_data

def create_config(env, base_env, obs_space_size):
    """Create configuration dictionary"""
    return {
        # Buffer parameters
        'buffer_size': 500,  # Reduced size
        'sample_size': 32,   # Reduced size
        
        # Learning parameters
        'gamma': 0.99,
        'epsilon_init': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 0.001,
        
        # Environment parameters
        'n_jobs': len(base_env.jobs),
        'n_machines': base_env.num_machines,
        'action_space_size': env.action_space.n,
        'obs_space_size': obs_space_size,
        
        # Rest of your config parameters...
        'nr_of_clauses': 1000,
        'nr_of_state_bits': 10,
        'nr_of_tm_features': 10,
        'max_included_literals': 32,
        'T': 10, 
        's': 2.0,
        'Tmin': 1,
        'Tmax': 10,
        'T_decay': 0.9999,
        'threshold': 0.5,
        'bits_per_machine_time': 4,
        'bits_per_job_status': 2,
        'bits_per_job': 4,
        'bits_per_machine': 4,
        'bits_per_operation': 4,
        'bits_per_machine_status': 2,
        'bits_per_duration': 4,
        'bits_per_progress': 2,
        'tm_data_size': 32,
        'tm_num_classes': 32,
        'tm_clausesx2': 8,
        'tm_t': 8,
        'tm_s': 2.0,
        'focused_negative_sampling': True,
        'violation_penalty': 1.0,
        'max_conflict_rate': 0.1,
        'feature_window_size': 5,
        'tm_boost_true_positive': 1.0,
        'tm_boost_true_negative': 1.0,
        'tm_type_i_ii_ratio': 1.0,
        'weighted_clauses': True,
        'train_freq': 1,
        'n_steps': 1,
        'log_freq': 10,
        'eval_freq': 100
    }

def main():
    instance_name = 'ta11'
    world_size = 3

    # Setup distributed training
    local_rank = setup_distributed()

    print(f"Found {world_size} GPUs")
    
    try:
        
        if world_size > 1:
            # Multi-GPU setup
            mp.spawn(
                train_on_gpu,
                args=(world_size, instance_name),
                nprocs=world_size,
                join=True
            )

        # Create environment
        env, base_env = create_environment(instance_name)

        # Calculate observation space size first
        if hasattr(env.observation_space, 'shape'):
            obs_space_size = env.observation_space.shape[0]
        else:
            obs_space_size = env.observation_space.n
        
        # Create configuration 
        config= create_config(env, base_env, obs_space_size)
    
         # Verify environment setup
        env_status = verify_environment(env)
        logger.info("Environment verification results:")
        for key, value in env_status.items():
            logger.info(f"{key}: {value}")
        
        if not all(env_status.values()):
            raise RuntimeError("Environment verification failed")
        
        print(f"Observation space size: {obs_space_size}")
        print(f"Action space size: {config['action_space_size']}")
        
        # Initialize wandb
        init_wandb(config, instance_name)
        
        # Create and train agent
        agent = QTMAgent(env, config, rank=0)
        
        # Create model artifact
        model_artifact = wandb.Artifact(
            f"model_{instance_name}", 
            type="model",
            description=f"QTM model for {instance_name}"
        )
        
        # Train the agent
        best_actions, best_makespan = agent.learn(nr_of_episodes=2)
        
        # Save final model
        model_path = f'model_{instance_name}.pt'
        torch.save({
            #'model_state_dict': agent.policy.state_dict(),    # need to save model
            'best_makespan': best_makespan,
            'config': config
        }, model_path)
        
        # Log model artifact
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
        
        # Execute best solution
        env.reset(seed=None)
        schedule_data = []
        
        # Execute actions and collect schedule data
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
        
        # Log machine utilization
        machine_utilization_table = wandb.Table(columns=["Machine", "Utilization"])
        machine_utilization = env.unwrapped.get_machine_utilization()
        
        for i, u in enumerate(machine_utilization):
            machine_utilization_table.add_data(i, float(u))
        
        # Log final results
        wandb.log({
            "final_results": {
                "best_makespan": best_makespan,
                "number_of_actions": len(best_actions),
            }
        })
        
        # Log average machine utilization
        average_utilization = sum(machine_utilization) / len(machine_utilization) if machine_utilization else 0
        wandb.log({"final_results.machine_utilization_avg": average_utilization})
        
        # Save and log schedule data
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
        wandb.run.summary.update({
            "best_makespan": best_makespan,
            "final_memory_size": agent.replay_buffer.count,
            "total_actions": len(best_actions)
        })
        
        print(f"\nTraining completed. Results logged to W&B.")
        print(f"Best makespan: {best_makespan}")
        
    except Exception as e:
            # Basic error message
        error_msg = f"Error during training: {str(e)}"
        print(error_msg)
        
        # Log to wandb if running
        if wandb.run is not None:
            wandb.run.summary["error"] = error_msg
            
        # Basic GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise e

    finally:
        # Always close wandb
        if wandb.run is not None:
            wandb.finish()
if __name__ == "__main__":
    main()