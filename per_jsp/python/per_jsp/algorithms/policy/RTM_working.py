from per_jsp.python.per_jsp.algorithms.Q_Networks.JSPBinarizer import JSPBinarizer

import numpy as np
import pyximport;

pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)

import TM_lib.rtm as RTM

import numpy as np
import random
import torch
import os

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class Policy():
    """
    Policy implementation for Job Shop Scheduling using Tsetlin Machines

    """
    def __init__(self, env, config):
        # Initialize one TM for each job (action)
        self.env = env
        self.binarizer = config['binarizer']
        print("num_features:", config['num_features'])
        self.tms = [RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                      number_of_features=config['num_features'], 
                                      s=config['s'],
                                      number_of_states=config['number_of_state_bits_ta'],
                                      threshold=config['T'],
                                      max_target=config['y_max'], min_target=config['y_min'],
                                      max_update_p=config['max_update_p'],
                                      min_update_p=config['min_update_p'])
                    for _ in range(config['action_space_size'])]    #One TM is created for each possible action (job)

        # Dataset for binarizer initialization
        # Check if file exists, if not create one with random data
        #dataset_path = f'per_jsp/python/per_jsp/algorithms/misc/{config["dataset_file_name"]}.txt'

        # Create sample observations for initialization instead of loading from file
        # This avoids the issue with the dataset format
        self.config = config
        
        # Generate synthetic observations dictionary similar to what the environment returns
        # Get a sample observation from the environment to determine the structure
        sample_obs, _ = env.reset()
        
        # Use the sample observation to create synthetic data
        self.observations = self.generate_synthetic_observations(sample_obs, 20)
        
        # Initialize the binarizer and TMs
        self.init_binarizer()
        self.init_TMs()

    def generate_synthetic_observations(self, sample_obs, num_samples=20):
        """Generate synthetic observations based on the structure of a sample observation"""
        synthetic_observations = []
        
        for _ in range(num_samples):
            # Create a new observation dictionary with the same structure
            obs = {}
            for key, value in sample_obs.items():
                if isinstance(value, np.ndarray):
                    if key == 'job_progress':
                        # Job progress starts at 0
                        obs[key] = np.zeros_like(value)
                    elif key == 'machine_availability':
                        # Machine availability ranges from 0 to some positive value
                        obs[key] = np.random.uniform(0, 10, value.shape).astype(np.float32)
                    elif key == 'next_operation_for_job':
                        # Next operation is an integer index
                        max_op = value.shape[0] - 1 if value.shape[0] > 0 else 3
                        obs[key] = np.random.randint(0, max_op+1, value.shape).astype(np.int32)
                    elif key in ['completed_jobs', 'job_ready', 'tool_state']:
                        # Binary data
                        obs[key] = np.random.randint(0, 2, value.shape).astype(value.dtype)
                    else:
                        # Default case
                        obs[key] = np.random.random(value.shape).astype(value.dtype)
                else:
                    # For any non-array values, just copy them
                    obs[key] = value
            
            synthetic_observations.append(obs)
        
        return synthetic_observations

    def init_binarizer(self):
        """Initialize the binarizer with synthetic observations"""
        # The binarizer was already initialized in the config, no need to fit it again
        pass

    def init_TMs(self):
        """Initialize the Tsetlin Machines with some initial data"""
        # Transform the synthetic observations using the binarizer
        binary_observations = []
        for obs in self.observations:
            binary_obs = self.binarizer.transform(obs)
            # Ensure it's a 1D array of int32
            binary_obs = np.asarray(binary_obs, dtype=np.int32).flatten()
            binary_observations.append(binary_obs)
        
        # Stack the binary observations
        binary_observations = np.vstack(binary_observations)
        
        # Initialize each TM with random target values
        for tm in self.tms:
            random_targets = np.array([
                random.uniform(self.config['y_min'], self.config['y_max']) 
                for _ in range(len(binary_observations))
            ]).astype(dtype=np.float32)
            
            tm.fit(binary_observations, random_targets)

    def update(self, tms_input):
        """
        Update the Tsetlin Machines based on experiences
        
        Args:
            tms_input: Dictionary with observations and target Q-values for each action
            
        Returns:
            Dictionary with absolute errors for each TM
        """
        abs_errors = {f'actor{i}': [] for i in range(len(self.tms))}
        
        for idx, input_data in enumerate(tms_input):
            if len(tms_input[idx]['observations']) > 0:
                binary_obs_list = []
                
                # Transform each observation to binary features
                for obs in tms_input[idx]['observations']:
                    # Check if observation is already in binary format or is a dictionary
                    if isinstance(obs, dict):
                        binary_obs = self.binarizer.transform(obs)
                    else:
                        # Already in binary format
                        binary_obs = obs

                    # Ensure it's a 1D array of int32
                    binary_obs = np.asarray(binary_obs, dtype=np.int32).flatten()
                    binary_obs_list.append(binary_obs)
                
                # Stack the binary observations
                if len(binary_obs_list) > 0:
                    binary_obs_array = np.vstack(binary_obs_list)
                    
                    # Update the TM with the observations and target Q-values
                    targets = np.array(tms_input[idx]['target_q_vals'], dtype=np.float32)
                    abs_errors[f'actor{idx}'] = self.tms[idx].fit(binary_obs_array, targets)

        return abs_errors

    def predict(self, obs):
        """
        Predict Q-values for each action given the observation
        
        Args:
            obs: Observation dictionary or binary observation array
            
        Returns:
            Array of Q-values for each action
        """
        # If the input is a dictionary observation, transform it
        if isinstance(obs, dict):
            binary_obs = self.binarizer.transform(obs)
        else:
            # Input is already a binary observation array (from replay buffer)
            binary_obs = obs
            
        # Ensure binary_obs is a 1D array of int32
        #This ensures the binary patterns are represented as integers for the TM.
        binary_obs = np.asarray(binary_obs, dtype=np.int32).flatten()    
            
        # Get predictions from each TM
        tm_vals = []
        for tm in self.tms:
            tm_vals.append(tm.predict(binary_obs))
            
        return np.array([tm_vals])