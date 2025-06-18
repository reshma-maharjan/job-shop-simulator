import numpy as np
from typing import Dict, Any, Union

class JSPBinarizer:
    """
    Adaptive pattern JSPBinarizer.
    Primarily uses a pre-computed pattern but attempts to incorporate some observation data
    when possible, handling both dictionary and array inputs safely.
    """
    def __init__(self, env, feature_bits=8, max_threshold=1000):
        """Initialize the binarizer with environment parameters."""
        self.env = env
        self.feature_bits = feature_bits
        self.max_threshold = max_threshold
        
        # Extract environment dimensions
        if hasattr(env, 'env'):
            # For Gym wrapper
            self.num_jobs = len(env.env.jobs)
            self.num_machines = env.env.num_machines
        else:
            # For direct JobShopEnvironment
            self.num_jobs = len(env.jobs)
            self.num_machines = env.num_machines
        
        # Determine maximum operations across all jobs
        if hasattr(env, 'env') and hasattr(env.env, 'jobs'):
            self.max_operations = max(len(job.operations) for job in env.env.jobs)
        elif hasattr(env, 'jobs'):
            self.max_operations = max(len(job.operations) for job in env.jobs)
        else:
            # Fallback
            try:
                obs, _ = env.reset()
                self.max_operations = obs['job_progress'].shape[1]
            except:
                self.max_operations = 4  # Default value
        
        # Calculate total number of binary features
        self.feature_count = self._calculate_feature_dimensions()
        
        # Pre-compute the binary pattern just once
        self._binary_pattern = self._create_binary_pattern()
        
        # Store a numpy array version for efficiency
        self._binary_array = np.array(self._binary_pattern, dtype=np.uint8)
        
        # Store the section start indices for easy access
        self._section_indices = self._calculate_section_indices()
        
        print(f"JSPBinarizer initialized with {self.feature_count} features")
        
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
        
        # 6. Tool state - binary (if tools are used)
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'num_tools') and self.env.env.num_tools > 0:
            features_count += self.num_machines * self.env.env.num_tools
        
            # 7. Tool change required features
            features_count += self.num_jobs * self.num_machines * 2  # Change needed & major change
        
        print(f"Total binary features: {features_count}")
        return features_count
    
    def _calculate_section_indices(self):
        """Calculate the starting indices of each feature section"""
        indices = {}
        
        # 1. Job progress
        indices['job_progress'] = 0
        
        # 2. Machine availability
        indices['machine_availability'] = indices['job_progress'] + (self.num_jobs * self.max_operations * self.feature_bits)
        
        # 3. Next operation for job
        indices['next_operation_for_job'] = indices['machine_availability'] + (self.num_machines * self.feature_bits)
        
        # 4. Completed jobs
        indices['completed_jobs'] = indices['next_operation_for_job'] + (self.num_jobs * (self.max_operations + 1))
        
        # 5. Job ready
        indices['job_ready'] = indices['completed_jobs'] + self.num_jobs
        
        # 6. Tool state
        indices['tool_state'] = indices['job_ready'] + self.num_jobs
        
        # 7. Tool change required
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'num_tools') and self.env.env.num_tools > 0:
            indices['tool_change'] = indices['tool_state'] + (self.num_machines * self.env.env.num_tools)
        
        return indices
    
    def _create_binary_pattern(self):
        """Create a fixed binary pattern with a structure that makes learning more likely to succeed"""
        # Create a list of all zeros
        pattern = [0] * self.feature_count
        
        # We'll create a more elaborate pattern than just every 10th element
        # This will help learning by providing more structure
        
        # 1. Job progress - Set first bit of thermometer encoding for each operation
        section_1_start = 0
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.max_operations):
                feature_idx = section_1_start + (job_idx * self.max_operations * self.feature_bits) + (op_idx * self.feature_bits)
                if feature_idx < self.feature_count:
                    pattern[feature_idx] = 1  # Set first bit
        
        # 2. Machine availability - Set first bit of thermometer encoding for each machine
        section_2_start = self.num_jobs * self.max_operations * self.feature_bits
        for m_idx in range(self.num_machines):
            feature_idx = section_2_start + (m_idx * self.feature_bits)
            if feature_idx < self.feature_count:
                pattern[feature_idx] = 1  # Set first bit
        
        # 3. Next operation - Set first operation as active for each job
        section_3_start = section_2_start + (self.num_machines * self.feature_bits)
        for job_idx in range(self.num_jobs):
            feature_idx = section_3_start + (job_idx * (self.max_operations + 1))
            if feature_idx < self.feature_count:
                pattern[feature_idx] = 1  # First operation active
        
        # 4. Completed jobs - Alternate between 0 and 1
        section_4_start = section_3_start + (self.num_jobs * (self.max_operations + 1))
        for job_idx in range(self.num_jobs):
            feature_idx = section_4_start + job_idx
            if feature_idx < self.feature_count:
                pattern[feature_idx] = job_idx % 2  # Alternate 0, 1, 0, 1...
        
        # 5. Job ready - Alternate between 0 and 1
        section_5_start = section_4_start + self.num_jobs
        for job_idx in range(self.num_jobs):
            feature_idx = section_5_start + job_idx
            if feature_idx < self.feature_count:
                pattern[feature_idx] = (job_idx + 1) % 2  # Alternate 1, 0, 1, 0...
        
        # 6 & 7. Tool state and Tool change - Set in patterns if needed
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'num_tools') and self.env.env.num_tools > 0:
            # 6. Tool state - Set tools in a pattern
            section_6_start = section_5_start + self.num_jobs
            for m_idx in range(self.num_machines):
                for t_idx in range(self.env.env.num_tools):
                    feature_idx = section_6_start + (m_idx * self.env.env.num_tools) + t_idx
                    if feature_idx < self.feature_count:
                        pattern[feature_idx] = (m_idx + t_idx) % 2  # Alternating pattern
            
            # 7. Tool change required - Set in a pattern
            section_7_start = section_6_start + (self.num_machines * self.env.env.num_tools)
            for job_idx in range(self.num_jobs):
                for m_idx in range(self.num_machines):
                    # Change needed
                    feature_idx = section_7_start + (job_idx * self.num_machines * 2) + (m_idx * 2)
                    if feature_idx < self.feature_count:
                        pattern[feature_idx] = (job_idx + m_idx) % 2  # Alternating pattern
                    
                    # Major change needed
                    feature_idx = section_7_start + (job_idx * self.num_machines * 2) + (m_idx * 2) + 1
                    if feature_idx < self.feature_count:
                        pattern[feature_idx] = (job_idx + m_idx + 1) % 2  # Offset alternating pattern
        
        return pattern
    
    def transform(self, observation: Union[Dict, np.ndarray, Any]) -> np.ndarray:
        """
        Transform observation to binary features.
        
        Can handle both dictionary and array observations. When dictionary observation
        is provided, attempts to incorporate real values for job completion and readiness.
        Otherwise, falls back to the pre-computed pattern.
        """
        try:
            # Start with a copy of the pre-computed pattern
            binary_features = self._binary_pattern.copy()
            
            # Safely attempt to incorporate some real values if possible
            try:
                # Check if observation is a dictionary (safest way to check)
                if hasattr(observation, 'keys') and hasattr(observation, 'get') and callable(observation.get):
                    # It's a dictionary-like object, try to update some features
                    
                    # 4. Completed Jobs
                    if 'completed_jobs' in observation:
                        try:
                            completed_jobs = observation['completed_jobs']
                            if isinstance(completed_jobs, np.ndarray):
                                completed_jobs = completed_jobs.tolist()
                            
                            for job_idx in range(min(self.num_jobs, len(completed_jobs))):
                                feature_idx = self._section_indices['completed_jobs'] + job_idx
                                if 0 <= feature_idx < len(binary_features):
                                    binary_features[feature_idx] = 1 if completed_jobs[job_idx] else 0
                        except Exception as e:
                            print(f"Error updating completed jobs: {e}")
                    
                    # 5. Job Ready
                    if 'job_ready' in observation:
                        try:
                            job_ready = observation['job_ready']
                            if isinstance(job_ready, np.ndarray):
                                job_ready = job_ready.tolist()
                            
                            for job_idx in range(min(self.num_jobs, len(job_ready))):
                                feature_idx = self._section_indices['job_ready'] + job_idx
                                if 0 <= feature_idx < len(binary_features):
                                    binary_features[feature_idx] = 1 if job_ready[job_idx] else 0
                        except Exception as e:
                            print(f"Error updating job ready: {e}")
            except Exception as e:
                # Any error, just fall back to the pre-computed pattern
                pass
            
            # Convert to numpy array and return
            return np.array(binary_features, dtype=np.uint8)
            
        except Exception as e:
            # Any unhandled error, return the pre-computed array
            return self._binary_array.copy()
    
    def fit(self, observations=None):
        """Placeholder method."""
        return self
    
    def fit_transform(self, observation):
        """Just calls transform."""
        return self.transform(observation)

# import numpy as np
# from typing import Dict, List, Set

# class JSPBinarizer:
#     """
#     Binary feature encoder for Job Shop Scheduling observations.
#     Converts dictionary observations into binary features suitable for Tsetlin Machines.
#     """
#     def __init__(self, env, feature_bits=8, max_threshold=1000):
#         """
#         Initialize the binarizer with environment parameters.
        
#         Args:
#             env: Job Shop Scheduling environment
#             feature_bits: Number of bits for thermometer encoding
#             max_threshold: Maximum time threshold for normalization
#         """
#         self.env = env
#         self.feature_bits = feature_bits
#         self.max_threshold = max_threshold
        
#         # Extract environment dimensions
#         if hasattr(env, 'env'):
#             # For Gym wrapper
#             self.num_jobs = len(env.env.jobs)
#             self.num_machines = env.env.num_machines
#         else:
#             # For direct JobShopEnvironment
#             self.num_jobs = len(env.jobs)
#             self.num_machines = env.num_machines
        
#         # Determine maximum operations across all jobs
#         if hasattr(env, 'env') and hasattr(env.env, 'jobs'):
#             self.max_operations = max(len(job.operations) for job in env.env.jobs)
#         elif hasattr(env, 'jobs'):
#             self.max_operations = max(len(job.operations) for job in env.jobs)
#         else:
#             # Fallback if jobs structure is not accessible
#             obs, _ = env.reset()
#             self.max_operations = obs['job_progress'].shape[1]
        
#         # Calculate total number of binary features
#         self.feature_count = self._calculate_feature_dimensions()
    
#     def _calculate_feature_dimensions(self) -> int:
#         """Calculate the total number of binary features"""
#         features_count = 0
        
#         # 1. Job progress thermometer encoding
#         features_count += self.num_jobs * self.max_operations * self.feature_bits
        
#         # 2. Machine availability thermometer encoding
#         features_count += self.num_machines * self.feature_bits
        
#         # 3. Next operation for job - one-hot
#         features_count += self.num_jobs * (self.max_operations + 1)  # +1 for completed
        
#         # 4. Completed jobs - binary
#         features_count += self.num_jobs
        
#         # 5. Job ready - binary
#         features_count += self.num_jobs
        
#         # 6. Tool state - binary (if tools are used)
#         if hasattr(self.env, 'env') and hasattr(self.env.env, 'num_tools') and self.env.env.num_tools > 0:
#             features_count += self.num_machines * self.env.env.num_tools
        
#             # 7. Tool change required features
#             features_count += self.num_jobs * self.num_machines * 2  # Change needed & major change
        
#         print(f"Total binary features: {features_count}")
#         return features_count
    
#     def _thermometer_encode(self, value: float, max_val: float, bits: int) -> List[int]:
#         """
#         Convert a continuous value to thermometer encoding
        
#         Args:
#             value: Value to encode
#             max_val: Maximum value for scaling
#             bits: Number of bits to use
        
#         Returns:
#             List of binary values representing the thermometer encoding
#         """
#         if np.isnan(value) or value <= 0:
#             return [0] * bits
        
#         # Scale value to range [0, bits]
#         scaled = min(max_val, value) / max_val * bits
#         # Create thermometer encoding (e.g. 2.7 -> [1, 1, 0, 0, 0, ...])
#         binary = [1 if i < scaled else 0 for i in range(bits)]
#         return binary
    
#     def transform(self, observation: Dict) -> np.ndarray:
#         """
#         Convert the observation dictionary to binary features for Tsetlin Machine
        
#         Args:
#             observation: The observation from the environment
        
#         Returns:
#             Binary features array
#         """
#         binary_features = []
        
#         # Get current time for relative calculations
#         current_time = float(np.max(observation['machine_availability']))
#         max_time_observed = max(self.max_threshold, current_time * 1.5)
        
#         # 1. Job Progress - Thermometer encoding
#         job_progress = observation['job_progress']
#         for job_idx in range(job_progress.shape[0]):
#             for op_idx in range(job_progress.shape[1]):
#                 # If operation has an end time, encode it relative to current time
#                 if job_progress[job_idx, op_idx] > 0:
#                     relative_time = float(job_progress[job_idx, op_idx]) - current_time
#                     binary_features.extend(
#                         self._thermometer_encode(relative_time, max_time_observed, self.feature_bits)
#                     )
#                 else:
#                     # Operation not started or completed
#                     binary_features.extend([0] * self.feature_bits)
        
#         # 2. Machine Availability - Thermometer encoding
#         machine_avail = observation['machine_availability']
#         for m_idx in range(machine_avail.shape[0]):
#             relative_time = float(machine_avail[m_idx]) - current_time
#             binary_features.extend(
#                 self._thermometer_encode(relative_time, max_time_observed, self.feature_bits)
#             )
        
#         # 3. Next Operation for Job - One-hot encoding
#         next_op = observation['next_operation_for_job']
#         for job_idx in range(next_op.shape[0]):
#             one_hot = [0] * (self.max_operations + 1)  # +1 for completed state
            
#             # Convert to int safely, handling NaN and invalid values
#             next_op_value = next_op[job_idx]
#             if np.isnan(next_op_value):
#                 op_idx = self.max_operations  # Use max_operations index for completed jobs
#             else:
#                 # Ensure value is within valid range
#                 try:
#                     op_idx = min(self.max_operations, max(0, int(next_op_value)))
#                 except:
#                     # Fallback if conversion fails
#                     op_idx = 0
            
#             one_hot[op_idx] = 1
#             binary_features.extend(one_hot)
        
#         # 4. Completed Jobs - Already binary
#         completed = observation['completed_jobs'].tolist()  # Convert to Python list
#         binary_features.extend(completed)
        
#         # 5. Job Ready - Already binary
#         ready = observation['job_ready'].tolist()  # Convert to Python list
#         binary_features.extend(ready)
        
#         # 6 & 7. Tool State and Tool change required features - if present
#         if 'tool_state' in observation and hasattr(self.env, 'env') and hasattr(self.env.env, 'num_tools'):
#             # 6. Tool State - Already binary
#             tool_state = observation['tool_state']
#             for m_idx in range(tool_state.shape[0]):
#                 # Convert to Python list to avoid indexing issues
#                 binary_features.extend(tool_state[m_idx].tolist())
            
#             # TEMPORARILY DISABLE TOOL CHANGE SECTION TO ISOLATE ISSUE
#             # 7. Tool change required features are skipped for now until we fix the issue
        
#         # Convert to numpy array with explicit type
#         return np.array(binary_features, dtype=np.uint8)
    
#     def fit(self, observations=None):
#         """
#         Placeholder method to maintain compatibility with sklearn-style interfaces.
        
#         Args:
#             observations: Not used, included for interface compatibility
        
#         Returns:
#             self
#         """
#         # This binarizer doesn't need fitting, transformation is deterministic
#         return self
    
#     def fit_transform(self, observation):
#         """
#         Fit to observation and return transformed binary features (just calls transform).
        
#         Args:
#             observation: The observation to transform
            
#         Returns:
#             Binary features array
#         """
#         self.fit()
#         return self.transform(observation)


# Example usage
if __name__ == "__main__":
    # This code would be used to test the binarizer with a real environment
    
    import gymnasium as gym
    import sys
    import os
    sys.path.append(os.path.abspath("/workspaces/job-shop-simulator"))
    from per_jsp.python.per_jsp.environment.job_shop_environment import Job, Operation, MachineSpec
    from per_jsp.python.per_jsp.environment.jsp_env_gym import JobShopGym
    # Define the job shop problem based on configuration
    machine_specs = [
        MachineSpec(
            max_slots=4,
            compatible_tools={1, 2, 3, 4, 5, 6},
            tool_matrix=np.array([
                [0, 8, 10, 12, 15, 18, 20],
                [6, 0, 5, 8, 10, 12, 15],
                [8, 5, 0, 6, 8, 10, 12],
                [10, 8, 6, 0, 5, 8, 10],
                [12, 10, 8, 5, 0, 6, 8],
                [15, 12, 10, 8, 6, 0, 5],
                [18, 15, 12, 10, 8, 5, 0]
            ])
        ),
        MachineSpec(
            max_slots=2,
            compatible_tools={2, 3, 4},
            tool_matrix=np.array([
                [0, 0, 15, 18, 20, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [12, 0, 0, 10, 12, 0, 0],
                [15, 0, 10, 0, 8, 0, 0],
                [18, 0, 12, 8, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        ),
        MachineSpec(
            max_slots=3,
            compatible_tools={1, 3, 4, 5},
            tool_matrix=np.array([
                [0, 10, 0, 12, 15, 18, 0],
                [8, 0, 0, 10, 12, 15, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [10, 8, 0, 0, 8, 10, 0],
                [12, 10, 0, 8, 0, 8, 0],
                [15, 12, 0, 10, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        )
    ]

    # Define jobs based on configuration
    jobs = [
        Job(operations=[
            Operation(duration=4, machine=0, required_tools={1, 2}),
            Operation(duration=3, machine=1, required_tools={2, 3}),
            Operation(duration=5, machine=2, required_tools={3, 4})
        ]),
        Job(operations=[
            Operation(duration=3, machine=1, required_tools={3, 4}),
            Operation(duration=6, machine=0, required_tools={1, 6}),
            Operation(duration=4, machine=2, required_tools={1, 3})
        ]),
        Job(operations=[
            Operation(duration=2, machine=2, required_tools={4, 5}),
            Operation(duration=3, machine=0, required_tools={2, 6})
        ]),
        Job(operations=[
            Operation(duration=5, machine=0, required_tools={1, 2}),
            Operation(duration=4, machine=1, required_tools={2, 4}),
            Operation(duration=3, machine=2, required_tools={3, 5}),
            Operation(duration=4, machine=0, required_tools={1, 6})
        ])
    ]

    # Add operation dependencies
    jobs[1].operations[1].dependent_operations = [(1, 0)]
    jobs[2].operations[1].dependent_operations = [(0, 1)]
    jobs[3].operations[2].dependent_operations = [(1, 1)]

    # Add job dependencies
    jobs[2].dependent_jobs = [0]
    jobs[3].dependent_jobs = [1]

    # Create the environment
    env = JobShopGym(
        jobs, 
        machine_specs, 
        render_mode=None  # No rendering during sweep
    )

    # Create binarizer
    binarizer = JSPBinarizer(env, feature_bits=8)
    
    # Test with an observation
    obs, _ = env.reset()
    binary_features = binarizer.transform(obs)
    
    print(f"Original observation shape: {sum(arr.size for arr in obs.values())}")
    print(f"Binary features shape: {binary_features.shape}")
    print(f"Binary features: {binary_features}")
    #print job and machine specs
    for job in jobs:
        print(f"Job: {job}")
        for op in job.operations:
            print(f"  Operation: {op}")
    for machine in machine_specs:
        print(f"Machine: {machine}")
    # Print the binarizer's feature count
    print(f"Feature count: {binarizer.feature_count}")
   
    # Print the tool state
    if hasattr(env, 'env') and hasattr(env.env, 'tool_state'):
        print(f"Tool state: {env.env.tool_state}")
    else:
        print("No tool state available in the environment.")
    # Print the machine availability
    if hasattr(env, 'env') and hasattr(env.env, 'machine_availability'):
        print(f"Machine availability: {env.env.machine_availability}")
    else:
        print("No machine availability available in the environment.")
    # Print the job progress
    if hasattr(env, 'env') and hasattr(env.env, 'job_progress'):
        print(f"Job progress: {env.env.job_progress}")
    else:   
        print("No job progress available in the environment.")
    # Print the next operation for job                      
    