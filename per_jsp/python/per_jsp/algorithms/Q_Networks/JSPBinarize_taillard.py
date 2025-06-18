import numpy as np
from typing import Dict, List


class JSPBinarizer:
    """
    Binary feature encoder for Job Shop Scheduling observations without tool changes.
    Converts dictionary observations into binary features suitable for Tsetlin Machines.
    """
    def __init__(self, env, feature_bits=8, max_threshold=1000, num_jobs=None, num_machines=None, max_operations=None):
        """
        Initialize the binarizer with environment parameters.
        
        Args:
            env: Job Shop Scheduling environment
            feature_bits: Number of bits for thermometer encoding
            max_threshold: Maximum time threshold for normalization
        """
        self.env = env
        self.feature_bits = feature_bits
        self.max_threshold = max_threshold
        
        # Use explicit parameters if provided
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_operations = max_operations
        
        # If not provided, try to extract from environment
        if self.num_jobs is None or self.num_machines is None or self.max_operations is None:
            if hasattr(env, 'env'):
                # For Gym wrapper
                self.num_jobs = self.num_jobs or (env.env.job_count if hasattr(env.env, 'job_count') else None)
                self.num_machines = self.num_machines or (env.env.machine_count if hasattr(env.env, 'machine_count') else None)
            else:
                # For direct JobShopEnvironment or JobShopGymEnv
                self.num_jobs = self.num_jobs or (env.job_count if hasattr(env, 'job_count') else None)
                self.num_machines = self.num_machines or (env.machine_count if hasattr(env, 'machine_count') else None)
            
        # If we couldn't get dimensions directly, try to get them from observation space
        if hasattr(env, 'observation_space') and self.num_jobs is None:
            try:
                # Try to infer from observation space
                space = env.observation_space
                if hasattr(space, 'spaces') and 'job_progress' in space.spaces:
                    self.num_jobs = space.spaces['job_progress'].shape[0]
                    self.max_operations = space.spaces['job_progress'].shape[1]
                if hasattr(space, 'spaces') and 'machine_availability' in space.spaces:
                    self.num_machines = space.spaces['machine_availability'].shape[0]
            except:
                pass
                
        # If we still don't have the dimensions, try to get them by calling reset
        if self.num_jobs is None or self.num_machines is None or not hasattr(self, 'max_operations'):
            # We need to call reset and check observation
            env.reset()
            # Access observation through environment attributes if possible
            if hasattr(env, 'job_progress'):
                self.num_jobs = env.job_progress.shape[0]
                self.max_operations = env.job_progress.shape[1]
            if hasattr(env, 'machine_availability'):
                self.num_machines = env.machine_availability.shape[0]
                
            # As a last resort, try to extract from observation dictionary
            if not hasattr(self, 'num_jobs') or not hasattr(self, 'num_machines') or not hasattr(self, 'max_operations'):
                try:
                    # Get observation manually
                    obs = env.get_observation() if hasattr(env, 'get_observation') else None
                    if obs is None and hasattr(env, 'reset'):
                        obs = env.reset()
                    
                    if obs is not None and isinstance(obs, dict):
                        if 'job_progress' in obs:
                            self.num_jobs = obs['job_progress'].shape[0]
                            self.max_operations = obs['job_progress'].shape[1]
                        if 'machine_availability' in obs:
                            self.num_machines = obs['machine_availability'].shape[0]
                except:
                    # If all else fails, we need explicit values
                    raise ValueError("Could not determine environment dimensions. Please pass explicit num_jobs, num_machines, and max_operations to the constructor.")
        
        # Calculate total number of binary features
        self.feature_count = self._calculate_feature_dimensions()
    
    def _calculate_feature_dimensions(self) -> int:
        """Calculate the total number of binary features"""
        features_count = 0
        
        # 1. Job progress thermometer encoding
        features_count += self.num_jobs * self.max_operations * self.feature_bits
        
        # 2. Machine availability thermometer encoding
        features_count += self.num_machines * self.feature_bits
        
        # 3. Next operation for job - one-hot
        features_count += self.num_jobs * (self.max_operations + 1)  # +1 for completed
        
        # 4. Completed jobs - binary
        features_count += self.num_jobs
        
        # 5. Job ready - binary
        features_count += self.num_jobs
        
        # 6. Job dependencies - binary matrix
        features_count += self.num_jobs * self.num_jobs
        
        # 7. Operation machine assignments - binary matrix
        features_count += self.num_jobs * self.max_operations * self.num_machines
        
        print(f"Total binary features: {features_count}")
        return features_count
    
    def _thermometer_encode(self, value: float, max_val: float, bits: int) -> List[int]:
        """
        Convert a continuous value to thermometer encoding
        
        Args:
            value: Value to encode
            max_val: Maximum value for scaling
            bits: Number of bits to use
        
        Returns:
            List of binary values representing the thermometer encoding
        """
        if np.isnan(value) or value <= 0:
            return [0] * bits
        
        # Scale value to range [0, bits]
        scaled = min(max_val, value) / max_val * bits
        # Create thermometer encoding (e.g. 2.7 -> [1, 1, 0, 0, 0, ...])
        binary = [1 if i < scaled else 0 for i in range(bits)]
        return binary
    
    def transform(self, observation):
        """
        Convert the observation to binary features for Tsetlin Machine.
        Handles various observation formats including NumPy arrays.
        
        Args:
            observation: The observation from the environment (dict, tuple, or ndarray)
        
        Returns:
            Binary features array
        """
        binary_features = []
        
        # Handle different observation formats
        if isinstance(observation, tuple):
            # For tuple observations, check length and handle accordingly
            if len(observation) == 0:
                raise ValueError("Empty observation tuple")
            elif len(observation) == 1:
                # Tuple with single element, which might be the actual observation
                if isinstance(observation[0], dict):
                    # This is a dict observation wrapped in a tuple
                    obs_dict = observation[0]
                else:
                    # This is a non-dict observation, need to examine it
                    raise ValueError(f"Unsupported single-element tuple observation: {type(observation[0])}")
            else:
                # If we're getting here, you need to examine what's in your tuple
                # and construct the obs_dict appropriately
                print(f"DEBUG: Tuple observation with {len(observation)} elements")
                for i, item in enumerate(observation):
                    print(f"  Element {i}: Type {type(item)}, Shape: {getattr(item, 'shape', 'No shape')}")
                
                # Try a few common formats - you may need to adjust this
                if len(observation) >= 2 and isinstance(observation[0], dict):
                    # First element is the observation dict
                    obs_dict = observation[0]
                else:
                    # Let's create a dummy observation for now - you'll need to replace this
                    # with the correct mapping based on your environment's observation format
                    print("WARNING: Creating dummy observation - update code with correct mapping!")
                    obs_dict = {
                        'job_progress': np.zeros((self.num_jobs, self.max_operations)),
                        'machine_availability': np.zeros(self.num_machines),
                        'next_operation_for_job': np.zeros(self.num_jobs),
                        'completed_jobs': np.zeros(self.num_jobs, dtype=np.uint8),
                        'job_ready': np.ones(self.num_jobs, dtype=np.uint8)
                    }
        elif isinstance(observation, dict):
            # If it's already a dictionary, use as is
            obs_dict = observation
        elif isinstance(observation, np.ndarray):
            # If it's a NumPy array, create a standardized observation dictionary
            print(f"Converting NumPy array observation with shape {observation.shape} to dictionary")
            
            # Determine structure based on array shape
            array_size = observation.size
            
            # Create a suitable structure based on environment dimensions
            # This is a heuristic approach - you might need to adjust based on your actual array structure
            
            # Estimate how many elements might be for job progress, machine availability, etc.
            # based on dimensions we have
            job_progress_size = self.num_jobs * self.max_operations
            machine_avail_size = self.num_machines
            next_op_size = self.num_jobs
            
            # Create a dummy observation with zeros
            obs_dict = {
                'job_progress': np.zeros((self.num_jobs, self.max_operations)),
                'machine_availability': np.zeros(self.num_machines),
                'next_operation_for_job': np.zeros(self.num_jobs),
                'completed_jobs': np.zeros(self.num_jobs, dtype=np.uint8),
                'job_ready': np.ones(self.num_jobs, dtype=np.uint8)
            }
            
            # Try to determine if this array has a known structure we can interpret
            try:
                # If the array has clear sections that we can identify, extract them
                if array_size >= job_progress_size + machine_avail_size + next_op_size:
                    # This is just a guess at how the array might be structured
                    # You may need to adjust based on your actual array layout
                    reshaped = observation.flatten()
                    
                    # Extract slices of the array for different features
                    # Note: These indices are estimates and might need adjusting
                    offset = 0
                    
                    # Try to extract job progress data
                    job_progress_flat = reshaped[offset:offset + job_progress_size]
                    offset += job_progress_size
                    obs_dict['job_progress'] = job_progress_flat.reshape(self.num_jobs, self.max_operations)
                    
                    # Try to extract machine availability
                    machine_avail = reshaped[offset:offset + machine_avail_size]
                    offset += machine_avail_size
                    obs_dict['machine_availability'] = machine_avail
                    
                    # Try to extract next operation for job
                    next_op = reshaped[offset:offset + next_op_size]
                    offset += next_op_size
                    obs_dict['next_operation_for_job'] = next_op
                    
                    # The rest could be completed jobs and job ready flags
                    remaining_size = array_size - offset
                    if remaining_size >= self.num_jobs:
                        completed_jobs = reshaped[offset:offset + self.num_jobs]
                        obs_dict['completed_jobs'] = (completed_jobs > 0).astype(np.uint8)
                        offset += self.num_jobs
                    
                    if remaining_size - self.num_jobs >= self.num_jobs:
                        job_ready = reshaped[offset:offset + self.num_jobs]
                        obs_dict['job_ready'] = (job_ready > 0).astype(np.uint8)
                else:
                    # If the array is too small, use it to fill as many features as possible
                    # This ensures we still generate a valid binary representation
                    print(f"Array size {array_size} is too small for full mapping. Using partial mapping.")
            except Exception as e:
                print(f"Error parsing NumPy array: {e}, using default structure")
        else:
            # If it's a single numpy array or other object, try to interpret it
            print(f"WARNING: Observation is not a tuple, dict, or numpy array but {type(observation)}")
            # You'll need to create a suitable mapping based on your environment
            obs_dict = {
                'job_progress': np.zeros((self.num_jobs, self.max_operations)),
                'machine_availability': np.zeros(self.num_machines),
                'next_operation_for_job': np.zeros(self.num_jobs),
                'completed_jobs': np.zeros(self.num_jobs, dtype=np.uint8),
                'job_ready': np.ones(self.num_jobs, dtype=np.uint8)
            }
        
        # Get current time for relative calculations
        current_time = np.max(obs_dict['machine_availability'])
        max_time_observed = max(self.max_threshold, current_time * 1.5)
        
        # 1. Job Progress - Thermometer encoding
        job_progress = obs_dict['job_progress']
        for job_idx in range(job_progress.shape[0]):
            for op_idx in range(job_progress.shape[1]):
                # If operation has an end time, encode it relative to current time
                if job_progress[job_idx, op_idx] > 0:
                    relative_time = job_progress[job_idx, op_idx] - current_time
                    binary_features.extend(
                        self._thermometer_encode(relative_time, max_time_observed, self.feature_bits)
                    )
                else:
                    # Operation not started or completed
                    binary_features.extend([0] * self.feature_bits)
        
        # 2. Machine Availability - Thermometer encoding
        machine_avail = obs_dict['machine_availability']
        for m_idx in range(machine_avail.shape[0]):
            relative_time = machine_avail[m_idx] - current_time
            binary_features.extend(
                self._thermometer_encode(relative_time, max_time_observed, self.feature_bits)
            )
        
        # 3. Next Operation for Job - One-hot encoding
        next_op = obs_dict['next_operation_for_job']
        for job_idx in range(next_op.shape[0]):
            one_hot = [0] * (self.max_operations + 1)  # +1 for completed state
            op_idx = min(int(next_op[job_idx]), self.max_operations)
            one_hot[op_idx] = 1
            binary_features.extend(one_hot)
        
        # 4. Completed Jobs - Already binary
        completed = obs_dict['completed_jobs']
        binary_features.extend(completed)
        
        # 5. Job Ready - Already binary
        ready = obs_dict['job_ready']
        binary_features.extend(ready)
        
        # 6. Job dependencies - binary matrix (if available)
        # Skip this if we don't have access to job dependencies
        try:
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'jobs'):
                jobs = self.env.env.jobs
            elif hasattr(self.env, 'jobs'):
                jobs = self.env.jobs
            else:
                # Skip this feature if jobs not accessible
                jobs = []
            
            # Create job dependency matrix if we have job info
            if jobs:
                dependency_matrix = np.zeros((self.num_jobs, self.num_jobs), dtype=np.uint8)
                for i, job in enumerate(jobs):
                    if hasattr(job, 'dependent_jobs'):
                        for dep_job_idx in job.dependent_jobs:
                            dependency_matrix[i, dep_job_idx] = 1
                
                # Flatten and add to features
                binary_features.extend(dependency_matrix.flatten().tolist())
            else:
                # Placeholder for job dependencies if not available
                binary_features.extend([0] * (self.num_jobs * self.num_jobs))
        except:
            # Placeholder if we encounter any errors
            binary_features.extend([0] * (self.num_jobs * self.num_jobs))
        
        # 7. Operation machine assignments - binary matrix (if available)
        try:
            if jobs:
                machine_assignments = np.zeros((self.num_jobs, self.max_operations, self.num_machines), dtype=np.uint8)
                
                for job_idx, job in enumerate(jobs):
                    if job_idx < len(jobs):
                        for op_idx, operation in enumerate(job.operations):
                            if op_idx < len(job.operations):
                                machine_idx = operation.machine if hasattr(operation, 'machine') else -1
                                if 0 <= machine_idx < self.num_machines:
                                    machine_assignments[job_idx, op_idx, machine_idx] = 1
                
                # Flatten and add to features
                binary_features.extend(machine_assignments.flatten().tolist())
            else:
                # Placeholder if not available
                binary_features.extend([0] * (self.num_jobs * self.max_operations * self.num_machines))
        except:
            # Placeholder if we encounter any errors
            binary_features.extend([0] * (self.num_jobs * self.max_operations * self.num_machines))
        
        return np.array(binary_features, dtype=np.uint8)
    
    def fit(self, observations=None):
        """
        Placeholder method to maintain compatibility with sklearn-style interfaces.
        
        Args:
            observations: Not used, included for interface compatibility
        
        Returns:
            self
        """
        # This binarizer doesn't need fitting, transformation is deterministic
        return self
    
    def fit_transform(self, observation):
        """
        Fit to observation and return transformed binary features (just calls transform).
        
        Args:
            observation: The observation to transform
            
        Returns:
            Binary features array
        """
        self.fit()
        return self.transform(observation)


# Example usage
if __name__ == "__main__":
    # This code would be used to test the binarizer with a real environment
    
    import gymnasium as gym
    import sys
    import os
    import numpy as np
    sys.path.append(os.path.abspath("/workspaces/job-shop-simulator"))

    # Import your JSP environment classes here
    from per_jsp.python.per_jsp.environment.job_shop_environment_origianl_gym import Job, Operation
    # Import JobShopGymEnv from the correct module
    from per_jsp.python.per_jsp.environment.job_shop_environment_origianl_gym import JobShopGymEnv
    from per_jsp.python.per_jsp.environment.job_shop_taillard_generator  import TaillardJobShopGenerator
    from per_jsp.python.per_jsp.algorithms.Q_Networks.JSPBinarizer_without_tool_change import JSPBinarizer
    
    # load the taillard data
    taillard_file_path = "/workspaces/job-shop-simulator/per_jsp/data/taillard_instances/ta01.txt"
    jobs, makespan_estimate = TaillardJobShopGenerator.load_problem(taillard_file_path)
    
    # Create the environment (using JobShopGymEnv)
    # Check the correct initialization parameters for your environment
    env = JobShopGymEnv(
        jobs,
    )
    
    # Create the binarizer with explicit dimensions
    num_jobs = len(jobs)
    num_machines = max(op.machine for job in jobs for op in job.operations) + 1
    max_operations = max(len(job.operations) for job in jobs)
    
    # Create binarizer with explicit dimensions
    binarizer = JSPBinarizer(
        env, 
        feature_bits=8, 
        num_jobs=num_jobs, 
        num_machines=num_machines, 
        max_operations=max_operations
    )
    
    # Reset the environment
    obs = env.reset()
    
    # Inspect the observation in detail
    print("\nDetailed observation inspection:")
    print(f"Observation type: {type(obs)}")
    
    if isinstance(obs, tuple):
        print(f"Tuple with {len(obs)} elements")
        for i, elem in enumerate(obs):
            if hasattr(elem, 'shape'):
                print(f"  Element {i}: {type(elem)}, shape: {elem.shape}")
            elif isinstance(elem, dict):
                print(f"  Element {i}: dict with keys: {list(elem.keys())}")
            else:
                print(f"  Element {i}: {type(elem)}")
    elif isinstance(obs, dict):
        print(f"Dictionary with keys: {list(obs.keys())}")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"Unexpected observation type: {type(obs)}")
    
    try:
        # Try to transform the observation
        binary_features = binarizer.transform(obs)
        print(f"Binary features shape: {binary_features.shape}")
        print(f"First 20 binary features: {binary_features[:20]}")
    except Exception as e:
        print(f"Error transforming observation: {e}")
        print("Try creating a custom mapping for your observation format.")
    
    # Print environment information
    print(f"Number of jobs: {binarizer.num_jobs}")
    print(f"Number of machines: {binarizer.num_machines}")
    print(f"Max operations: {binarizer.max_operations}")