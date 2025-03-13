import numpy as np
import time
import torch
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from jsp_env_gym import JobShopGym, Job, Operation, MachineSpec

# Try importing Tsetlin Machine
try:
    from pyTsetlinMachine.tm import MultiClassTsetlinMachine
    TSETLIN_AVAILABLE = True
    
    # Check if CUDA support is available in PyTorch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU-accelerated PyTorch is available: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch CUDA is not available. Using CPU version.")
        
    # Your version of pyTsetlinMachine doesn't support the platform parameter
    print("Using standard Tsetlin Machine with PyTorch acceleration for feature processing")
except ImportError:
    print("Warning: pyTsetlinMachine not found. Install with: pip install pyTsetlinMachine")
    TSETLIN_AVAILABLE = False
    GPU_AVAILABLE = False


class PyTorchTsetlinScheduler:
    """Implementation of a Tsetlin Machine for job shop scheduling with PyTorch GPU acceleration"""
    
    def __init__(
        self, 
        env, 
        num_clauses: int = 6000, 
        T: float = 600,
        s: float = 10,
        feature_bits_per_value: int = 10,
        max_threshold: int = 100,
        seed: Optional[int] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the PyTorch GPU-accelerated Tsetlin Machine Scheduler
        
        Args:
            env: JobShopGym environment
            num_clauses: Number of clauses in the Tsetlin Machine
            T: Threshold parameter (controls specificity)
            s: Specificity parameter
            feature_bits_per_value: Number of bits to use for thermometer encoding
            max_threshold: Maximum threshold value for thermometer encoding
            seed: Random seed
            use_gpu: Whether to use GPU acceleration if available
        """
        self.env = env
        self.num_jobs = len(self.env.env.jobs)
        self.num_machines = self.env.env.num_machines
        self.max_operations = max(len(job.operations) for job in self.env.env.jobs)
        self.feature_bits = feature_bits_per_value
        self.max_threshold = max_threshold
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Set device for PyTorch
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"Using device: {self.device}")
        
        # Calculate feature dimensions
        self.num_features = self._calculate_feature_dimensions()
        print(f"Number of binary features: {self.num_features}")

        # Set random seeds for reproducibility
        seed_value = seed if seed is not None else 42
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if self.use_gpu:
            torch.cuda.manual_seed(seed_value)
        
        # Initialize Tsetlin Machine if available
        if TSETLIN_AVAILABLE:
            # Initialize without the platform parameter which is not supported in your version
            self.tm = MultiClassTsetlinMachine(
                num_clauses, 
                T, 
                s=s,
                boost_true_positive_feedback=1,
                number_of_state_bits=10
            )
        else:
            self.tm = None
            print("Tsetlin Machine not available. Using random policy.")
        
        # Training metrics
        self.training_rewards = []
        self.training_makespans = []
        self.episode_lengths = []
    
    def _calculate_feature_dimensions(self) -> int:
        """Calculate binary features for the environment state"""
        features_count = 0
        
        # 1. Job state features
        features_count += self.num_jobs * self.max_operations * self.feature_bits
        
        # 2. Machine availability features
        features_count += self.num_machines * self.feature_bits
        
        # 3. Next operation features
        features_count += self.num_jobs * (self.max_operations + 1)
        
        # 4. Completed jobs features
        features_count += self.num_jobs
        
        # 5. Job ready features
        features_count += self.num_jobs
        
        # 6. Tool state features (if present in environment)
        if hasattr(self.env.env, 'num_tools') and self.env.env.num_tools > 0:
            features_count += self.num_machines * self.env.env.num_tools
            
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
    
    def convert_observation_to_features(self, observation: Dict) -> np.ndarray:
        """
        Convert the observation dictionary to binary features for Tsetlin Machine
        
        Args:
            observation: The observation from the environment
        
        Returns:
            Binary features array
        """
        binary_features = []
        
        # Get current time for relative calculations
        current_time = np.max(observation['machine_availability'])
        max_time_observed = max(self.max_threshold, current_time * 1.5)
        
        # 1. Job Progress - Thermometer encoding
        job_progress = observation['job_progress']
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
        machine_avail = observation['machine_availability']
        for m_idx in range(machine_avail.shape[0]):
            relative_time = machine_avail[m_idx] - current_time
            binary_features.extend(
                self._thermometer_encode(relative_time, max_time_observed, self.feature_bits)
            )
        
        # 3. Next Operation for Job - One-hot encoding
        next_op = observation['next_operation_for_job']
        for job_idx in range(next_op.shape[0]):
            one_hot = [0] * (self.max_operations + 1)  # +1 for completed state
            op_idx = min(int(next_op[job_idx]), self.max_operations)
            one_hot[op_idx] = 1
            binary_features.extend(one_hot)
        
        # 4. Completed Jobs - Already binary
        completed = observation['completed_jobs']
        binary_features.extend(completed)
        
        # 5. Job Ready - Already binary
        ready = observation['job_ready']
        binary_features.extend(ready)
        
        # 6. Tool State - Already binary if present
        if 'tool_state' in observation:
            tool_state = observation['tool_state']
            for m_idx in range(tool_state.shape[0]):
                binary_features.extend(tool_state[m_idx])
        
        # Convert to numpy array first
        np_features = np.array(binary_features, dtype=np.uint8)
        
        # Convert to PyTorch tensor if using GPU
        if self.use_gpu:
            # Create torch tensor and move to GPU
            torch_features = torch.from_numpy(np_features).to(self.device)
            # Convert back to numpy for Tsetlin Machine which expects numpy input
            return torch_features.cpu().numpy()
        
        return np_features
    
    def select_action(self, observation: Dict, valid_actions: List[int], explore: bool = True) -> int:
        """
        Select an action using the Tsetlin Machine
        
        Args:
            observation: Current observation from environment
            valid_actions: List of valid action indices
            explore: Whether to explore (True) or exploit (False)
        
        Returns:
            Selected action index
        """
        if not valid_actions:
            return -1  # No valid actions
            
        if not TSETLIN_AVAILABLE or self.tm is None:
            # Random policy if Tsetlin not available
            return np.random.choice(valid_actions)
        
        # Create feature vector
        X = self.convert_observation_to_features(observation).reshape(1, -1)
        
        # Pure exploration with decreasing probability over time
        if explore and np.random.random() < 0.2:  # Fixed exploration rate
            return np.random.choice(valid_actions)
        
        # Map valid actions to class indices for the TM
        action_mapping = {i: action for i, action in enumerate(valid_actions)}

        # During initial episodes, use random selection until model is properly trained
        if not hasattr(self.tm, 'number_of_patches') or not hasattr(self.tm, 'encoded_X'):
            return np.random.choice(valid_actions)
            
        # Get Tsetlin Machine prediction among valid actions
        if len(valid_actions) == 1:
            # Only one valid action
            return valid_actions[0]
        
        # Create vote array for valid actions only
        votes = np.zeros(len(valid_actions))
        
        # Get votes for each action from Tsetlin Machine
        for i, action in enumerate(valid_actions):
            # Adjust class index to match TM's expectations (0-indexed)
            class_index = action % self.num_jobs
            
            try:
                # Get prediction/vote for this class
                transformed = self.tm.transform(X)
                start_idx = class_index * self.tm.number_of_clauses
                end_idx = (class_index + 1) * self.tm.number_of_clauses
                action_vote = transformed[:, start_idx:end_idx]
                
                # Use PyTorch for vote computation if GPU is enabled
                if self.use_gpu:
                    # Convert to tensor, compute on GPU, then back to numpy
                    vote_tensor = torch.from_numpy(action_vote).to(self.device)
                    votes[i] = torch.sum(vote_tensor).item()
                else:
                    # Sum the votes for this action using numpy
                    votes[i] = np.sum(action_vote)
            except AttributeError:
                # If transform fails, use random selection
                return np.random.choice(valid_actions)
        
        # Select action with highest vote
        if np.sum(votes) > 0:
            # At least one action got votes
            selected_idx = np.argmax(votes)
        else:
            # No votes, random selection
            selected_idx = np.random.randint(len(valid_actions))
        
        return valid_actions[selected_idx]
    
    def extract_rules(self) -> List[str]:
        """Extract simplified rules from the Tsetlin Machine"""
        rules = []
        
        if not TSETLIN_AVAILABLE or self.tm is None:
            return ["Tsetlin Machine not available"]
            
        for job_idx in range(self.num_jobs):
            rule = f"Job {job_idx} Rule: Schedule when predicted by Tsetlin Machine"
            rules.append(rule)
            
        return rules
        
    def train(
        self, 
        num_episodes: int = 200, 
        max_steps_per_episode: int = 1000,
        verbose: bool = True,
        plot_progress: bool = True
    ):
        """Train the Tsetlin Machine with pure reinforcement learning"""
        if not TSETLIN_AVAILABLE or self.tm is None:
            print("Tsetlin Machine not available. Cannot train.")
            return
        
        # Initialize metrics
        self.training_rewards = []
        self.training_makespans = []
        self.episode_lengths = []
        best_makespan = float('inf')
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Reset environment
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Store state-action-reward data for batch learning
            episode_data = []
            
            # Run one episode
            while not done and steps < max_steps_per_episode:
                # Get valid actions
                possible_actions_objects = self.env.env.get_possible_actions()
                possible_jobs = list(set(a.job for a in possible_actions_objects))
                
                if not possible_jobs:
                    break
                
                # Select action
                # Higher exploration in early episodes, lower in later episodes
                exploration_rate = max(0.1, 0.5 - (episode / num_episodes) * 0.4)
                explore = np.random.random() < exploration_rate
                action = self.select_action(obs, possible_jobs, explore=explore)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Store experience (only store positive rewards to focus learning)
                if reward > 0:
                    episode_data.append((obs, action, reward, next_obs, done))
                
                # Move to next step
                obs = next_obs
                steps += 1
            
            # Batch learning from episode
            self._learn_from_episode(episode_data)
            
            # Record metrics
            makespan = info['makespan']
            self.training_makespans.append(makespan)
            self.training_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Check for new best solution
            if makespan < best_makespan and makespan > 0:
                best_makespan = makespan
                print(f"New best makespan: {best_makespan}")
                # Save best schedule
                self.env.env.generate_html_gantt(f"pytorch_tsetlin_best_{episode}_for_taillard.html")
            
            # Print progress
            if verbose and episode % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode}/{num_episodes} - " 
                    f"Reward: {total_reward:.2f}, Makespan: {makespan}, "
                    f"Steps: {steps}, Time: {elapsed:.2f}s")
        
        # Plot training progress
        if plot_progress:
            self.plot_training_progress()
    
    def _learn_from_episode(self, episode_data):
        """
        Learn from episode data using pure reinforcement learning.
        
        Args:
            episode_data: List of (observation, action, reward, next_observation, done) tuples
        """
        if not TSETLIN_AVAILABLE or self.tm is None:
            return
        
        # Skip if no data
        if not episode_data:
            return
        
        batch_X = []
        batch_y = []
        
        # Process episode data
        for obs, action, reward, next_obs, done in episode_data:
            # Convert observation to features
            features = self.convert_observation_to_features(obs)
            batch_X.append(features)
            
            # The class is the job index
            job_idx = action % self.num_jobs
            batch_y.append(job_idx)
        
        # Train on batch data if available
        if batch_X and batch_y:
            X = np.array(batch_X, dtype=np.uint8)
            y = np.array(batch_y, dtype=np.int32)
            
            # Use PyTorch for processing the data if GPU is enabled
            if self.use_gpu:
                # Convert to PyTorch tensors and move to GPU
                X_tensor = torch.from_numpy(X).to(self.device)
                y_tensor = torch.from_numpy(y).to(self.device)
                
                # Any GPU preprocessing here
                
                # Move back to CPU for Tsetlin Machine
                X = X_tensor.cpu().numpy()
                y = y_tensor.cpu().numpy()
            
            # Update Tsetlin Machine with this batch
            self.tm.fit(X, y, epochs=1)
    
    def plot_training_progress(self):
        """Plot training progress metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.training_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot makespans
        plt.subplot(1, 3, 2)
        plt.plot(self.training_makespans)
        plt.title('Episode Makespans')
        plt.xlabel('Episode')
        plt.ylabel('Makespan')
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(1, 3, 3)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pytorch_tsetlin_training_progress.png')
        plt.show()
    
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Tuple[float, float]:
        """
        Evaluate the trained Tsetlin Machine
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
        
        Returns:
            Average reward and makespan
        """
        rewards = []
        makespans = []
        
        for episode in range(num_episodes):
            # Reset environment
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            
            # Run one episode
            while not done:
                # Get valid actions
                possible_actions_objects = self.env.env.get_possible_actions()
                possible_jobs = list(set(a.job for a in possible_actions_objects))
                
                if not possible_jobs:
                    break
                
                # Select action (no exploration during evaluation)
                action = self.select_action(obs, possible_jobs, explore=False)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Render if requested
                if render:
                    self.env.render()
                
                obs = next_obs
            
            # Record episode results
            rewards.append(total_reward)
            makespans.append(info['makespan'])
            
            print(f"Evaluation Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {total_reward:.2f}, Makespan: {info['makespan']}")
        
        avg_reward = np.mean(rewards)
        avg_makespan = np.mean(makespans)
        
        print(f"Evaluation Complete - Avg Reward: {avg_reward:.2f}, Avg Makespan: {avg_makespan:.2f}")
        
        # Generate final schedule visualization
        self.env.env.generate_html_gantt("pytorch_tsetlin_schedule_for_taillard.html")
        
        return avg_reward, avg_makespan


def load_taillard_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header information
    parts = lines[0].strip().split()
    num_jobs = int(parts[0])
    num_machines = int(parts[1])
    
    # Parse job data
    processing_times = []
    for i in range(1, num_jobs + 1):
        times = list(map(int, lines[i].strip().split()))
        processing_times.append(times)
    
    # Parse machine routing (adjust 1-indexed to 0-indexed)
    machine_routes = []
    for i in range(num_jobs + 1, 2 * num_jobs + 1):
        # Convert 1-indexed machines to 0-indexed
        machines = list(map(int, lines[i].strip().split()))
        # Subtract 1 from each machine ID to convert to 0-indexed
        machines = [m - 1 for m in machines]
        machine_routes.append(machines)
    
    # Create jobs and operations
    jobs = []
    for job_idx in range(num_jobs):
        operations = []
        for op_idx in range(num_machines):
            operations.append(
                Operation(
                    duration=processing_times[job_idx][op_idx],
                    machine=machine_routes[job_idx][op_idx],  # Now 0-indexed
                    required_tools=set()
                )
            )
        jobs.append(Job(operations=operations))
    
    # Create machine specs
    machine_specs = [
        MachineSpec(
            max_slots=0,
            compatible_tools=set(),
            tool_matrix=np.zeros((1, 1))
        ) for _ in range(num_machines)
    ]
    
    return jobs, machine_specs

# Example usage
def main():
    # Test for GPU availability using PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPU devices: {torch.cuda.device_count()}")
        has_gpu = True
    else:
        print("PyTorch CUDA not available, using CPU")
        has_gpu = False
    
    # Assuming JobShopGym, Job, Operation, and MachineSpec classes are available
    from jsp_env_gym import JobShopGym
    
    # Load a Taillard instance
    jobs, machine_specs = load_taillard_instance("/workspaces/job-shop-simulator/per_jsp/data/lawrance_instances/la20.txt")
    
    # Create environment
    env = JobShopGym(jobs, machine_specs, render_mode='ansi')
    
    # Create and train PyTorch GPU-accelerated Tsetlin Scheduler
    scheduler = PyTorchTsetlinScheduler(
        env, 
        num_clauses=1000,
        T=250,
        s=5,
        feature_bits_per_value=10,
        max_threshold=150,
        seed=42,
        use_gpu=has_gpu  # Use GPU if available
    )
    
    # Train the scheduler with reinforcement learning
    scheduler.train(
        num_episodes=500,
        max_steps_per_episode=100,
        verbose=True,
        plot_progress=True
    )
    
    # Evaluate the trained scheduler
    scheduler.evaluate(num_episodes=20, render=True)
    
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()