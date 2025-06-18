import numpy as np
import torch
import os
import yaml
from tqdm import tqdm
import random
import sys

# Add the parent directory to Python path to find the 'algorithms' module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append('/workspaces/master_thesis')
print(os.getcwd())

from algorithms.misc.replay_buffer import ReplayBuffer

class QTM:
    """
    QTM implementation specifically for Job Shop Scheduling problems.
    This class adapts the CartPole QTM approach to work with the JobShopGym environment.
    """
    def __init__(self, env, Policy, config):
        # Create the results directory
        results_dir = f'./results/{config["env_name"]}/{config["algorithm"]}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        self.run_id = (
            len([i for i in os.listdir(results_dir)]) + 1
        )
        self.env = env
        self.action_space_size = env.action_space.n  # Number of jobs to choose from
        self.obs_space_size = self._get_flattened_obs_size()  # Calculate flattened observation size
        
        config['action_space_size'] = self.action_space_size
        config['obs_space_size'] = self.obs_space_size
        self.online_policy = Policy(config)

        # Set up epsilon-greedy parameters
        self.epsilon = config['epsilon_init']
        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 32)
        
        # Initialize replay buffer for job shop scheduling
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, 
            self.batch_size, 
            n_steps=config.get("n_steps", 1)
        )
        
        self.config = config
        self.save_path = ''

        # Set up saving configuration
        if config['save']:
            self.run_id = 'run_' + str(
                len([i for i in os.listdir(f'./results/{config["env_name"]}/{config["algorithm"]}')]) + 1)
            self.make_run_dir()
            self.save_config()
        else:
            print('Warning SAVING is OFF!')
            self.run_id = "unidentified_run"

        # Testing configuration
        self.nr_of_test_episodes = 10  # Fewer test episodes for job shop as it's more time-consuming
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435]
        
        # Episode tracking
        self.total_timesteps = 0
        self.cur_episode = 0
        self.cur_mean = 0
        self.total_score = []
        self.best_score = float('inf')  # For JSP, lower makespan is better
        self.announce()

    def _get_flattened_obs_size(self):
        """Calculate the size of the flattened observation space for JSP environment"""
        obs, _ = self.env.reset()
        
        # Get dimensions of each observation component
        total_size = 0
        
        # Job progress (jobs x max_operations)
        total_size += obs['job_progress'].size
        
        # Machine availability
        total_size += obs['machine_availability'].size
        
        # Next operation for job
        total_size += obs['next_operation_for_job'].size
        
        # Completed jobs
        total_size += obs['completed_jobs'].size
        
        # Job ready
        total_size += obs['job_ready'].size
        
        # Tool state if available
        if 'tool_state' in obs:
            total_size += obs['tool_state'].size
            
        return total_size

    def flatten_observation(self, obs):
        """Flatten the dictionary observation to a 1D numpy array"""
        flattened_parts = []
        
        # Add job progress
        flattened_parts.append(obs['job_progress'].flatten())
        
        # Add machine availability
        flattened_parts.append(obs['machine_availability'].flatten())
        
        # Add next operation for job
        flattened_parts.append(obs['next_operation_for_job'].flatten())
        
        # Add completed jobs
        flattened_parts.append(obs['completed_jobs'].flatten())
        
        # Add job ready
        flattened_parts.append(obs['job_ready'].flatten())
        
        # Add tool state if available
        if 'tool_state' in obs:
            flattened_parts.append(obs['tool_state'].flatten())
            
        # Concatenate all parts
        return np.concatenate(flattened_parts)

    def announce(self):
        print(f'JSP QTM {self.run_id} has been initialized!')

    def make_run_dir(self):
        base_dir = './results'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(os.path.join(base_dir, self.config['env_name'])):
            os.makedirs(os.path.join(base_dir, self.config['env_name']))
        if not os.path.exists(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'])):
            os.makedirs(os.path.join(base_dir, self.config['env_name'], self.config['algorithm']))
        if not os.path.exists(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id)):
            os.makedirs(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id))
        self.save_path = os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id)

    def save_config(self):
        if self.config["save"]:
            with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
                yaml.dump(self.config, yaml_file, default_flow_style=False)

    def get_next_action(self, cur_obs):
        """
        Get next action using epsilon-greedy policy
        For JSP, need to filter based on possible actions
        """
        flattened_obs = self.flatten_observation(cur_obs)
        
        # Get all valid jobs from current observation
        valid_jobs = np.where(cur_obs['job_ready'] == 1)[0]
        
        if len(valid_jobs) == 0:
            # No valid jobs, return a random action (will be handled by environment)
            return np.random.randint(0, self.action_space_size)
        
        if np.random.random() < self.epsilon:
            # Random exploration among valid jobs
            return np.random.choice(valid_jobs)
        else:
            # Exploit using Q-values but only consider valid jobs
            q_vals = self.online_policy.predict(flattened_obs)[0]
            
            # Mask invalid actions with large negative values
            masked_q_vals = np.ones(self.action_space_size) * float('-inf')
            masked_q_vals[valid_jobs] = q_vals[valid_jobs]
            
            return np.argmax(masked_q_vals)

    def temporal_difference(self, i, next_q_vals):
        """Regular 1-step TD learning"""
        return self.replay_buffer.sampled_rewards[i] + (
                1 - self.replay_buffer.sampled_terminated[i]) * self.config["gamma"] * next_q_vals

    def n_step_temporal_difference(self, i, next_q_vals):
        """N-step TD learning for JSP"""
        target_q_vals = []
        target_q_val = 0
        
        for j in range(len(self.replay_buffer.sampled_rewards[i])):
            target_q_val += (self.config["gamma"] ** j) * self.replay_buffer.sampled_rewards[i][j]
            if self.replay_buffer.sampled_terminated[i][j] or self.replay_buffer.sampled_trunc[i][j]:
                break
                
        if j < len(self.replay_buffer.sampled_rewards[i]) - 1:
            # If we broke before the end due to termination
            target_q_val += (1 - self.replay_buffer.sampled_terminated[i][j]) * (self.config["gamma"] ** j) * next_q_vals[0]
            
        target_q_vals.append(target_q_val)
        return target_q_vals

    def update_epsilon_greedy(self):
        """Update epsilon for exploration-exploitation balance"""
        epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon = epsilon_min + (self.config["epsilon_init"] - epsilon_min) * np.exp(
            -self.total_timesteps * self.config["epsilon_decay"])

    def get_q_val_and_obs_for_tm(self, action, target_q_vals, cur_obs):
        """Prepare inputs for TM update"""
        tm_inputs = [{'observations': [], 'target_q_vals': []} for _ in range(self.action_space_size)]
        tm_inputs[action]['observations'].append(cur_obs)
        tm_inputs[action]['target_q_vals'].append(target_q_vals)
        return tm_inputs

    def train_n_step(self):
        """Training method for n-step returns"""
        self.replay_buffer.sample_n_seq()
        for i in range(self.batch_size):
            # Get the last observation in the sequence
            sampled_next_obs = self.replay_buffer.sampled_next_obs[i][-1]
            
            # Predict Q-values for next state
            next_q_vals = self.online_policy.predict(sampled_next_obs)
            next_q_vals = np.max(next_q_vals, axis=1)

            # Calculate target Q-values using n-step returns
            target_q_vals = self.n_step_temporal_difference(i, next_q_vals)
            
            # Prepare inputs for TM update
            tm_inputs = self.get_q_val_and_obs_for_tm(
                self.replay_buffer.sampled_actions[i][0], 
                target_q_vals[0],
                self.replay_buffer.sampled_cur_obs[i][0]
            )
            
            # Update the TM policy
            _ = self.online_policy.update(tm_inputs)

    def train(self):
        """Training method for 1-step returns"""
        self.replay_buffer.sample()
        for i in range(self.batch_size):
            # Predict Q-values for next state
            next_q_vals = self.online_policy.predict(self.replay_buffer.sampled_next_obs[i])
            next_q_vals = np.max(next_q_vals, axis=1)
            
            # Calculate target Q-values using 1-step TD
            target_q_vals = self.temporal_difference(i, next_q_vals[0])

            # Prepare inputs for TM update
            tm_inputs = self.get_q_val_and_obs_for_tm(
                self.replay_buffer.sampled_actions[i], 
                target_q_vals,
                self.replay_buffer.sampled_cur_obs[i]
            )
            
            # Update the TM policy
            _ = self.online_policy.update(tm_inputs)

    def rollout(self):
        """Execute one episode of experience collection"""
        obs, _ = self.env.reset(seed=random.randint(1, 100))
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            # Flatten observation for policy
            flattened_obs = self.flatten_observation(obs)
            
            # Get action from policy
            action = self.get_next_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Flatten next observation for storage
            flattened_next_obs = self.flatten_observation(next_obs)
            
            # Store experience in replay buffer
            self.replay_buffer.save_experience(
                action, flattened_obs, flattened_next_obs, reward, int(terminated), truncated
            )
            
            # Update current observation
            obs = next_obs
            
            # Increment timestep counter
            self.total_timesteps += 1
            
            # Check if we should perform training
            n_steps = self.config.get("n_steps", 1)
            train_freq = self.config.get("train_freq", 1)
            
            if self.total_timesteps >= self.batch_size + n_steps and self.total_timesteps % train_freq == 0:
                if n_steps > 1:
                    self.train_n_step()
                else:
                    self.train()
                    
            # Update exploration-exploitation balance
            self.update_epsilon_greedy()

    def learn(self, nr_of_episodes):
        """Main learning loop"""
        for episode in tqdm(range(nr_of_episodes)):
            self.cur_episode = episode + 1
            
            # Periodically test performance
            if episode % self.config["test_freq"] == 0:
                self.test(self.total_timesteps)
                
            # Execute one episode
            self.rollout()
            
            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Best makespan: {self.best_score}, Epsilon: {self.epsilon:.4f}")

    def test(self, nr_of_steps):
        """Test the current policy"""
        # For JSP, we minimize makespan (not maximize score)
        episode_makespans = np.zeros(self.nr_of_test_episodes)
        
        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])
            terminated, truncated = False, False
            
            while not (terminated or truncated):
                # Flatten observation for policy
                flattened_obs = self.flatten_observation(obs)
                
                # Get best action using the policy (no exploration)
                q_vals = self.online_policy.predict(flattened_obs)[0]
                
                # Filter valid actions
                valid_jobs = np.where(obs['job_ready'] == 1)[0]
                
                if len(valid_jobs) == 0:
                    # No valid jobs, just pick the first one
                    action = 0
                else:
                    # Mask invalid actions with large negative values
                    masked_q_vals = np.ones(self.action_space_size) * float('-inf')
                    masked_q_vals[valid_jobs] = q_vals[valid_jobs]
                    action = np.argmax(masked_q_vals)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # If done, record the makespan
                if terminated or truncated:
                    episode_makespans[episode] = info['makespan']
            
        # Calculate statistics
        mean_makespan = np.mean(episode_makespans)
        std_makespan = np.std(episode_makespans)
        
        # For JSP, we want to minimize makespan
        self.total_score.append(mean_makespan)
        self.cur_mean = mean_makespan

        # Save results
        self.save_results(mean_makespan, std_makespan, nr_of_steps)
        
        # Update best score (lower is better for makespan)
        if mean_makespan < self.best_score:
            self.save_model()
            self.best_score = mean_makespan
            print(f'New best makespan after {nr_of_steps} steps: {mean_makespan}!')

    def save_model(self):
        """Save the best model"""
        if self.config["save"]:
            tms = self.online_policy.tms
            tms_save = []
            for tm in range(len(tms)):
                ta_state, clause_sign, clause_output, feedback_to_clauses = tms[tm].get_params()
                ta_state_save = np.zeros((len(ta_state), len(ta_state[0]), len(ta_state[0][0])), dtype=np.int32)
                clause_sign_save = np.zeros((len(clause_sign)), dtype=np.int32)
                clause_output_save = np.zeros((len(clause_output)), dtype=np.int32)
                feedback_to_clauses_save = np.zeros((len(feedback_to_clauses)), dtype=np.int32)

                for i in range(len(ta_state)):
                    for j in range(len(ta_state[i])):
                        for k in range(len(ta_state[i][j])):
                            ta_state_save[i][j][k] = int(ta_state[i][j][k])
                    clause_sign_save[i] = int(clause_sign[i])
                    clause_output_save[i] = int(clause_output[i])
                    feedback_to_clauses_save[i] = int(feedback_to_clauses[i])
                tms_save.append(
                    {'ta_state': ta_state_save, 'clause_sign': clause_sign_save, 'clause_output': clause_output_save,
                     'feedback_to_clauses': feedback_to_clauses_save})
            torch.save(tms_save, os.path.join(self.save_path, 'best'))

    def save_results(self, mean, std, nr_of_steps):
        """Save test results to CSV"""
        if self.config["save"]:
            file_name = 'test_results.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))

            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("mean,std,steps\n")
                file.write(f"{mean},{std},{nr_of_steps}\n")