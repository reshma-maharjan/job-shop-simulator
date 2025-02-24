import numpy as np
from typing import List, Dict, Tuple, Any
import torch
import logging
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from scipy.sparse import csr_matrix
import gymnasium as gym
from gymnasium.envs.registration import register
from Job_shop_taillard_generator import TaillardGymGenerator
from job_shop_manual_generator import ManualGymGenerator
import wandb
from tqdm.auto import tqdm  
import logging
from pathlib import Path
import json
import random
import time
import wandb

logger = logging.getLogger(__name__)

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
        
        # Track episode makespans
        self.episode_makespans = []
        self.current_episode = 0

        logger.info("Memory-optimized QTM Agent initialized")

    def get_next_action(self, cur_obs):
        """Select next action using epsilon-greedy strategy with strict valid action enforcement"""
        valid_actions = self.get_valid_actions()
        logging.info(f"Selecting from {len(valid_actions)} valid actions")
        
        if not valid_actions:
            logging.warning("No valid actions available")
            return None
        
        if random.random() < self.epsilon:
            selected_action = random.choice(valid_actions)
            logging.info(f"Random action selected: {selected_action}")
            return selected_action
        
        # Get Q-values for valid actions only
        q_vals = self.policy.predict(cur_obs)
        
        # Mask invalid actions
        masked_q_vals = np.full_like(q_vals, -np.inf)
        masked_q_vals[valid_actions] = q_vals[valid_actions]
        
        selected_action = valid_actions[np.argmax(q_vals[valid_actions])]
        logging.info(f"Greedy action selected: {selected_action}")
        
        return selected_action

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from environment"""
        valid_actions = []
        for action_id in range(self.config['action_space_size']):
            action = self.env_unwrapped._decode_action(action_id)
            if self.env_unwrapped._is_valid_action(action):
                valid_actions.append(action_id)
        return valid_actions
    
    def calculate_reward(self, info: Dict) -> float:
        """Calculate custom reward for JSSP with capped values"""
        try:
            # Debug logging at the start
            logging.debug("=== Debug Info ===")
            logging.debug(f"Full info dict: {info}")
            for key, value in info.items():
                logging.debug(f"{key}: {type(value)} = {value}")
            
            reward = 0.0
            
            # Penalty for invalid actions
            if not info.get('valid_action', True):
                logging.debug("Invalid action penalty applied")
                return -100.0
                
            # Base reward for completing jobs
            completed_jobs = info.get('completed_jobs', 0)
            logging.debug(f"Completed jobs: {completed_jobs} (type: {type(completed_jobs)})")
            
            total_jobs = self.env.unwrapped.jobs
            logging.debug(f"Total jobs: {total_jobs} (type: {type(total_jobs)})")

            total_jobs_count = len(total_jobs) if isinstance(total_jobs, list) else total_jobs

            completion_ratio = min(completed_jobs / max(total_jobs_count, 1), 1.0)
            logging.debug(f"Completion ratio: {completion_ratio}")
            reward += completion_ratio * 100.0
            
            # Penalize makespan (with bounds)
            current_makespan = info.get('makespan', 0)
            logging.debug(f"Raw makespan value: {current_makespan} (type: {type(current_makespan)})")
            
            if isinstance(current_makespan, (list, tuple, np.ndarray)):
                # If makespan is a list/array, log its contents
                logging.debug(f"Makespan is a sequence type, contents: {current_makespan}")
                # Take the maximum value if it's a sequence
                try:
                    current_makespan = float(max(current_makespan))
                    logging.debug(f"Converted makespan to max value: {current_makespan}")
                except (TypeError, ValueError) as e:
                    logging.error(f"Error processing makespan sequence: {e}")
                    current_makespan = 0
            
            if isinstance(current_makespan, (int, float)) and current_makespan > 0:
                makespan_penalty = min(current_makespan * 0.05, 200.0)
                logging.debug(f"Makespan penalty: {makespan_penalty}")
                reward -= makespan_penalty
            
            # Machine utilization bonus (with bounds)
            if 'machine_utilization' in info:
                utilization = info['machine_utilization']
                logging.debug(f"Raw utilization: {utilization} (type: {type(utilization)})")
                
                if isinstance(utilization, (list, tuple, np.ndarray)):
                    try:
                        utilization = float(np.mean(utilization))
                        logging.debug(f"Converted utilization to mean: {utilization}")
                    except (TypeError, ValueError) as e:
                        logging.error(f"Error processing utilization: {e}")
                        utilization = 0
                
                utilization_bonus = min(float(utilization) * 20.0, 20.0)
                logging.debug(f"Utilization bonus: {utilization_bonus}")
                reward += utilization_bonus
            
            # Completion bonus
            if completed_jobs == total_jobs:
                logging.debug("Adding completion bonus")
                reward += 200.0
                
            # Final reward calculation
            reward = float(np.clip(reward, -1000.0, 1000.0))
            logging.debug(f"Final calculated reward: {reward}")
            
            return reward
            
        except Exception as e:
            logging.error(f"Error in reward calculation: {e}")
            logging.exception("Full traceback:")  # This will print the full stack trace
            return 0.0

    def train(self):
        """Train the policy on a batch of experiences with detailed error tracking"""
        try:
            logging.info("Starting training process")
            
            # Verify buffer state
            logging.info(f"Replay buffer state - count: {self.replay_buffer.count}, sample_size: {self.config['sample_size']}")
            
            # Sample experiences
            logging.info("Sampling from replay buffer...")
            self.replay_buffer.sample()
            
            # Initialize TM inputs dictionary
            tm_inputs = {}
            
            # Process each sample
            for i in range(self.config['sample_size']):
                try:
                    next_obs = np.array(self.replay_buffer.sampled_next_obs[i])
                    next_q_vals = self.policy.predict(next_obs)
                    next_q_val = np.max(next_q_vals)
                    
                    reward = self.replay_buffer.sampled_rewards[i]
                    terminated = self.replay_buffer.sampled_terminated[i]
                    
                    target_q_val = reward + (1 - terminated) * self.config['gamma'] * next_q_val
                    
                    cur_obs = self.replay_buffer.sampled_cur_obs[i]
                    action = self.replay_buffer.sampled_actions[i]
                    
                    if action not in tm_inputs:
                        tm_inputs[action] = {
                            'observations': [],
                            'target_q_vals': []
                        }
                    
                    tm_inputs[action]['observations'].append(cur_obs)
                    tm_inputs[action]['target_q_vals'].append(target_q_val)
                    
                except Exception as e:
                    logging.error(f"Error processing sample {i}: {e}")
                    raise
            
            # Update policy
            logging.info(f"Updating policy with {len(tm_inputs)} unique actions")
            self.policy.update(tm_inputs)
            logging.info("Training completed successfully")     #..........................................training completed successfully
            
        except Exception as e:
            logging.error("Error in training process")
            logging.exception("Full training error traceback:")
            raise

    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def rollout(self):
        """Execute one episode with enhanced debugging"""
        try:
            # Reset environment and get initial observation
            logging.info("Starting rollout - resetting environment")
            cur_obs, _ = self.env.reset()
            logging.info(f"Initial observation shape: {cur_obs.shape}")
            
            episode_reward = 0
            episode_actions = []
            step_count = 0
            max_steps = 1000  # Maximum steps per episode to prevent infinite loops
            training_performed = False # Flag to track training status
            
            while step_count < max_steps:
                # Select action
                # logging.info(f"\nStep {step_count + 1}")
                valid_actions = self.get_valid_actions()
                # logging.info(f"Valid actions available: {len(valid_actions)}")
                
                if not valid_actions:
                    # logging.info("No valid actions available - terminating episode")
                    break
                    
                action = self.get_next_action(cur_obs)
                
                # Verify if selected action is valid
                if action not in valid_actions:
                    # logging.error(f"Invalid action selected: {action}")
                    logging.error(f"Valid actions were: {valid_actions}")
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                # logging.info(f"Step results: reward={reward}, terminated={terminated}, truncated={truncated}")
                
                
                # Calculate custom reward
                custom_reward = self.calculate_reward(info)
                self.monitor_rewards(custom_reward, step_count)
                # logging.info(f"Custom reward: {custom_reward}")
                
                # Store experience
                self.replay_buffer.save_experience(
                    action, cur_obs, next_obs, custom_reward, 
                    int(terminated), int(truncated)
                )
                
                # Update statistics
                episode_reward += custom_reward
                episode_actions.append(action)
                self.total_timesteps += 1
                step_count += 1
                
                # Perform training if conditions are met
                # Check training conditions
                if self.total_timesteps >= self.config['sample_size']:
                    if self.total_timesteps % self.config['train_freq'] == 0:
                        # logging.info("Starting agent training...")  #..............................training
                        try:
                            before_train = time.time()
                            self.train()
                            after_train = time.time()
                            # logging.info(f"Agent training completed in {after_train - before_train:.2f} seconds")
                        except Exception as e:
                            logging.error(f"Error during training: {e}")
                            logging.exception("Full training error traceback:")
                            raise
                
                # Update exploration rate
                self.update_epsilon()
                
                # Update current observation
                cur_obs = next_obs
                
                # Check termination
                if terminated or truncated:
                    break
                    
                   # Log episode summary
            current_makespan = info.get('makespan', float('inf'))
            logging.info(f"\nEpisode Summary:")                                     #..........................................episode summary after training..........
            logging.info(f"Steps completed: {step_count}")
            #logging.info(f"Training performed: {training_performed}")
            logging.info(f"Final makespan: {current_makespan}")
            logging.info(f"Total reward: {episode_reward}")

            # Add this line
            self.episode_makespans.append(current_makespan)
            
            # Track best solution
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_actions = episode_actions.copy()
            
            return episode_reward, episode_actions, current_makespan
            
        except Exception as e:
            logging.error(f"Error in rollout: {str(e)}")
            raise

    def learn(self, nr_of_episodes: int) -> Tuple[List[int], float]:
        """Main training loop with episode vs makespan tracking"""
        logger.info(f"Starting training for {nr_of_episodes} episodes...")
        
        for episode in tqdm(range(nr_of_episodes)):
            self.current_episode = episode
            episode_reward, _, episode_makespan = self.rollout()
            self.episode_rewards.append(episode_reward)
            
            # Log every episode's makespan to wandb
            if wandb.run is not None:
                wandb.log({
                    'episode_vs_makespan/current': episode_makespan,
                    'episode_vs_makespan/best_so_far': min(self.episode_makespans),
                    'episode_vs_makespan/average': np.mean(self.episode_makespans),
                    'episode': episode
                })
            
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_makespan = np.mean(self.episode_makespans[-10:])
                logger.info(
                    f"Episode {episode + 1}/{nr_of_episodes}, "
                    f"Mean Reward: {mean_reward:.2f}, "
                    f"Mean Makespan: {mean_makespan:.2f}, "
                    f"Best Makespan: {self.best_makespan}"
                )
        
        # At the end of training, log the final makespan curve
        if wandb.run is not None:
            # Create a table of episode vs makespan data
            makespan_table = wandb.Table(columns=["episode", "makespan"])
            for ep, makespan in enumerate(self.episode_makespans):
                makespan_table.add_data(ep, makespan)
            
            wandb.log({
                "final_makespan_curve": wandb.plot.line(
                    makespan_table, 
                    "episode", 
                    "makespan",
                    title="Episode vs Makespan"
                )
            })
        
        return self.best_actions, self.best_makespan

    def monitor_rewards(self, reward: float, step: int) -> None:
        """Monitor and log reward statistics"""
        if not hasattr(self, 'reward_stats'):
            self.reward_stats = {
                'min_reward': float('inf'),
                'max_reward': float('-inf'),
                'sum_reward': 0,
                'count': 0
            }
        
        self.reward_stats['min_reward'] = min(self.reward_stats['min_reward'], reward)
        self.reward_stats['max_reward'] = max(self.reward_stats['max_reward'], reward)
        self.reward_stats['sum_reward'] += reward
        self.reward_stats['count'] += 1
        
        if step % 10 == 0:                     #log every 100 steps
            avg_reward = self.reward_stats['sum_reward'] / max(self.reward_stats['count'], 1)
            logging.info(f"\nReward Statistics:")
            logging.info(f"Min Reward: {self.reward_stats['min_reward']:.2f}")
            logging.info(f"Max Reward: {self.reward_stats['max_reward']:.2f}")
            logging.info(f"Average Reward: {avg_reward:.2f}")
            
class Policy:
    """Memory-optimized Policy class implementing MultiClassTsetlinMachine for JSSP"""
    
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
        self.T = config.get('T', 1500)  # Use get() with default value
        self.s = config.get('s', 1.0)  # Use get() with default value
        
        logger.info(f"Policy initialized with sparse TM representation. Features: {self.feature_transformer.total_features}")

    def _get_or_create_tm(self, action_id):
        """Get existing TM or create new one for given action"""
        if action_id not in self.tsetlin_machines:
            tm = MultiClassTsetlinMachine(
                self.nr_of_clauses,
                self.T,
                self.s
            )
            # Initialize the TM with a dummy sample
            sample_features = np.zeros((1, self.feature_transformer.total_features), dtype=np.int32)
            sample_y = np.array([0], dtype=np.int32)
            tm.fit(sample_features, sample_y, epochs=1)
            self.tsetlin_machines[action_id] = tm
        return self.tsetlin_machines[action_id]

    def predict(self, state):
        """Memory-efficient prediction with valid action masking"""
        if len(state) != self.config['obs_space_size']:
            raise ValueError(
                f"State size mismatch. Expected {self.config['obs_space_size']}, "
                f"got {len(state)}. State shape: {state.shape}"
            )
        
        binary_features = self.feature_transformer.transform(state)
        binary_features = binary_features.reshape(1, -1)  # Reshape for TM input
        binary_features = binary_features.astype(np.int32)  # Ensure int32 type
        
        valid_actions = self.get_valid_actions()
        q_values = np.full(self.action_space_size, -np.inf, dtype=np.float32)
        
        for action in valid_actions:
            try:
                tm = self._get_or_create_tm(action)
                # Get prediction from TM
                votes = tm.predict(binary_features)
                q_values[action] = float(votes[0])
            except Exception as e:
                logger.error(f"Error predicting with TM for action {action}: {e}")
                q_values[action] = -np.inf
        
        return q_values


    def update(self, tm_inputs: Dict[int, Dict[str, List]]):
        """Update Tsetlin Machines based on experience with detailed TM error handling"""
        try:
            logging.info("Starting policy update...")             #..........................................updating policy..........
            logging.info(f"Received {len(tm_inputs)} actions to update")              #................received ... actions to update
            
            # First, validate all actions are in valid range
            valid_actions = self.get_valid_actions()
            logging.info(f"Valid actions available: {valid_actions}")      #..........valid actions available
            
            filtered_tm_inputs = {}
            for action_id, data in tm_inputs.items():
                if action_id in valid_actions:
                    filtered_tm_inputs[action_id] = data
                else:
                    logging.warning(f"Skipping invalid action {action_id}")
                    
            logging.info(f"Filtered to {len(filtered_tm_inputs)} valid actions")
            
            for action_id, data in filtered_tm_inputs.items():
                try:
                    logging.info(f"\nProcessing action {action_id}")
                    if not data['observations']:
                        logging.warning(f"No observations for action {action_id}")
                        continue
                    
                    # Get or create TM for this action
                    logging.info(f"Getting TM for action {action_id}")
                    tm = self._get_or_create_tm(action_id)
                    
                    # Convert observations to binary features
                    logging.info("Converting observations to binary features")
                    try:
                        binary_features = np.array([
                            self.feature_transformer.transform(obs) 
                            for obs in data['observations']
                        ], dtype=np.int32)
                        logging.info(f"Binary features shape: {binary_features.shape}")
                        
                        # Validate binary features
                        if not np.all(np.isin(binary_features, [0, 1])):
                            logging.error("Binary features contain values other than 0 and 1")
                            continue
                            
                    except Exception as e:
                        logging.error(f"Error in feature transformation: {e}")
                        continue
                    
                    # Convert Q-values to binary classes
                    logging.info("Converting Q-values to binary classes")
                    try:
                        target_classes = np.array([
                            1 if q_val > 0 else 0 
                            for q_val in data['target_q_vals']
                        ], dtype=np.int32)
                        logging.info(f"Target classes shape: {target_classes.shape}")
                        
                        # Validate shapes match
                        if binary_features.shape[0] != target_classes.shape[0]:
                            logging.error(f"Shape mismatch: features {binary_features.shape} vs targets {target_classes.shape}")
                            continue
                            
                    except Exception as e:
                        logging.error(f"Error in target class conversion: {e}")
                        continue
                    
                    # Train the TM with detailed error handling
                    logging.info(f"Training TM for action {action_id}")
                    try:
                        # Log TM parameters
                        logging.info(f"TM parameters - Clauses: {self.nr_of_clauses}, T: {self.T}, s: {self.s}")
                        
                        # Verify TM input shapes
                        logging.info(f"TM input shapes - X: {binary_features.shape}, y: {target_classes.shape}")
                        
                        # Attempt fitting with timeout
                        start_time = time.time()
                        tm.fit(binary_features, target_classes, epochs=1)
                        end_time = time.time()
                        
                        logging.info(f"Successfully trained TM for action {action_id} in {end_time - start_time:.2f} seconds")
                        
                    except Exception as e:
                        logging.error(f"Error in TM fitting: {str(e)}")
                        logging.error("TM fitting failed - skipping this action")
                        continue
                        
                except Exception as e:
                    logging.error(f"Error processing action {action_id}: {e}")
                    logging.exception("Full action processing traceback:")
                    continue
                    
            logging.info("Policy update completed successfully")       #..........................................policy update completed successfully
            
        except Exception as e:
            logging.error(f"Error in policy update: {e}")
            logging.exception("Full policy update traceback:")
            raise

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
        self.machine_time_bits = min(config['bits_per_machine_time'], 8)
        self.job_status_bits = min(config['bits_per_job_status'], 4)
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
        project="job-shop-qtm_MultiClassTsetlinMachine_episodes",
        entity="reshma-stha2016",  
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
    # Define custom x-axis for makespan plots
    wandb.define_metric("episode_vs_makespan/*", step_metric="episode")

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

def run_job_shop_scheduling(
    instance_name,
    episodes=2,
    num_clauses=2000,
    threshold=1500,
    specificity=1.5,
    seed=42,
    enable_wandb=True,
    verbose=False
):
    """
    Run job shop scheduling with QTM using direct parameter passing instead of command line arguments.
    """
    # Setup logging
    setup_logging(verbose)
    
    try:
        # Set random seeds
        set_seeds(seed)
        
        # Create environment
        logging.info(f"Creating environment for instance: {instance_name}")
        if instance_name.startswith('ta'):
            # Create the base environment
            base_env = TaillardGymGenerator.create_env_from_instance(instance_name)
            
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
        
        # Create configuration dictionary
        config = {
            'env_name': 'job-shop',
            'algorithm': 'QTM',
            'nr_of_clauses': num_clauses,
            'T': threshold,
            's': specificity,
            'bits_per_machine_time': 8,
            'bits_per_job_status': 4,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'action_space_size': env.action_space.n,
            'obs_space_size': env.observation_space.shape[0],
            'gamma': 0.99,
            'epsilon_init': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.998,
            'buffer_size': 5000,
            'sample_size': 128,
            'train_freq': 50,         #changed from 100 to 200
            'test_freq': 5,
            'save': True,
            'seed': seed
        }
        
        # Print problem configuration
        logging.info("\nProblem Configuration:")
        logging.info(f"Instance: {instance_name}")
        logging.info(f"Jobs: {config['n_jobs']}")
        logging.info(f"Machines: {config['n_machines']}")
        logging.info(f"Action Space Size: {config['action_space_size']}")
        logging.info(f"Observation Space Size: {config['obs_space_size']}")
        logging.info(f"Clauses per TM: {config['nr_of_clauses']}")
        logging.info(f"Threshold T: {config['T']}")
        logging.info(f"Specificity s: {config['s']}")
        
        # Initialize W&B if enabled
        if enable_wandb:
            init_wandb(config, instance_name)
        
        # Create and train agent
        logging.info("\nInitializing QTM agent...")
        agent = QTMAgent(env, config)
        
        logging.info("\nStarting training...")
        episode_rewards = []
        episode_makespans = []
        best_actions = []
        best_makespan = float('inf')

        for episode in range(episodes):
            logging.info(f"\nStarting Episode {episode + 1}/{episodes}")
            
            # Use the rollout method
            agent.current_episode = episode
            episode_reward, current_actions, current_makespan = agent.rollout()
            
            # Log episode metrics
            logging.info(f"\nEpisode {episode + 1} completed:")                      # after training finished ( after rollout)
            logging.info(f"Current makespan: {current_makespan}")
            logging.info(f"Episode reward: {episode_reward}")
            
            episode_rewards.append(episode_reward)
            episode_makespans.append(current_makespan)
            
            # Update best solution if current is better
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_actions = current_actions.copy()
            
            # Log to wandb if enabled
            if enable_wandb:
                wandb.log({
                    "episode": episode,
                    "episode_vs_makespan/current": current_makespan,
                    "episode_vs_makespan/best": best_makespan,
                    "episode_reward": episode_reward,
                    "epsilon": agent.epsilon,
                    "number_of_actions": len(current_actions)
                })

        # Execute best solution and collect data
        # logging.info("\nExecuting best solution...")
        env.reset()
        schedule_data = []
        
        for action_id in best_actions:
            action = env.unwrapped._decode_action(action_id)
            obs, reward, terminated, truncated, info = env.step(action_id)
            
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
            'instance': instance_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'episodes': episodes,
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
                'machine_utilization': [float(u) for u in env.unwrapped.get_machine_utilization()],
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_makespans': [float(m) for m in episode_makespans]
            }
        }
        
        # Log final metrics to wandb
        if enable_wandb:
            wandb.log({
                "final_metrics": {
                    "best_makespan": best_makespan,
                    "final_makespan": episode_makespans[-1],
                    "average_makespan": sum(episode_makespans) / len(episode_makespans),
                    "total_episodes": episodes,
                    "total_actions": len(best_actions)
                },
                "solution": results
            })
            
            # Log machine utilization
            machine_utilization = env.unwrapped.get_machine_utilization()
            for i, util in enumerate(machine_utilization):
                wandb.log({f"machine_utilization/machine_{i}": util})

        # Save results to file
        result_file = Path(f'results/{instance_name[:-4]}/qtm_{time.strftime("%Y%m%d_%H%M%S")}.json')
        result_file.parent.mkdir(parents=True, exist_ok=True)
        save_results(result_file, results)
        
        # Display final results
        logging.info("\nFinal Results:")   #..........................................final results also displayed
        logging.info(f"Best Makespan: {best_makespan}")
        logging.info(f"Number of Actions: {len(best_actions)}")
        logging.info(f"Results saved to: {result_file}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        if enable_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        if enable_wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    try:
        # Enable wandb logging (make sure you're logged in)
        wandb.login()
        
        # Run with default parameters
        results = run_job_shop_scheduling(
            instance_name="ta01.txt",
            episodes=500,
            num_clauses=2000,
            threshold=1500,
            specificity=1.5,
            seed=42,
            enable_wandb=True,
            verbose=True
        )
        
        print(f"\nExecution completed successfully!")
        
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        raise