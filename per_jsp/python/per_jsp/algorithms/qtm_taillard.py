import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

logger = logging.getLogger(__name__)

class HybridTsetlinQLearningScheduler:
    """
    A hybrid scheduler that combines Q-learning with Tsetlin Machine for job shop scheduling.
    Uses Q-learning for high-level policy learning and Tsetlin Machine for feature-based action selection.
    """
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 episodes: int = 1000,
                 nr_clauses: int = 2000,
                 T: float = 1500,
                 s: float = 1.5):
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        
        # Tsetlin Machine parameters
        self.nr_clauses = nr_clauses
        self.T = T
        self.s = s
        
        # Initialize state trackers
        self.q_table = None
        self.tsetlin_machines = {}  # Dict to store TM for each action
        self.best_time = float('inf')
        self.best_schedule = []
        self.rng = np.random.RandomState()
        
        # Feature transformer for Tsetlin Machine
        self.feature_transformer = None

    def _initialize_q_table(self, env) -> None:
        """Initialize Q-table with proper dimensions."""
        max_operations = max(len(job.operations) for job in env.jobs)
        self.q_table = np.zeros((
            len(env.jobs),          # Number of jobs
            env.num_machines,       # Number of machines
            max_operations         # Max operations per job
        ))

    def _initialize_feature_transformer(self, env) -> None:
        """Initialize the feature transformer for the Tsetlin Machine."""
        self.feature_transformer = JSSPFeatureTransformer(
            n_jobs=len(env.jobs),
            n_machines=env.num_machines,
            machine_time_bits=8,
            job_status_bits=4
        )

    def _get_or_create_tm(self, action_id: int) -> MultiClassTsetlinMachine:
        """Get existing TM or create new one for given action."""
        try:
            if action_id not in self.tsetlin_machines:
                tm = MultiClassTsetlinMachine(
                    self.nr_clauses,
                    self.T,
                    self.s
                )
                # Initialize with dummy data
                sample_features = np.zeros((1, self.feature_transformer.total_features), dtype=np.int32)
                #assert sample_features.shape[1] == 405, f"Feature dimension mismatch: {sample_features.shape[1]} != 405"
                sample_y = np.array([0], dtype=np.int32)

                tm.fit(sample_features, sample_y, epochs=1)
                self.tsetlin_machines[action_id] = tm
            return self.tsetlin_machines[action_id]
        
        except Exception as e:
            print(f"TM creation error: {e}")
            print(f"Total features: {self.feature_transformer.total_features}")
            raise

    def _calculate_priority(self, env, action) -> float:
        """Calculate priority score combining Q-value and TM prediction."""
        # Get Q-value
        q_value = self.q_table[action.job, action.machine, action.operation]
        
        # Get TM prediction
        state_features = self.feature_transformer.transform(env.current_state)
        tm = self._get_or_create_tm(action.job * env.num_machines + action.machine)
        tm_prediction = tm.predict(state_features.reshape(1, -1))[0]
        
        # Combine Q-value and TM prediction
        priority = 0.7 * q_value + 0.3 * tm_prediction
        
        return priority

    def _select_action(self, env) -> Any:
        """Select action using hybrid strategy."""
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            return None
            
        if self.rng.random() < self.exploration_rate:
            # Use priorities for smarter exploration
            priorities = [self._calculate_priority(env, action) for action in possible_actions]
            total_priority = sum(priorities)
            if total_priority > 0:
                priorities = [p/total_priority for p in priorities]
                return possible_actions[self.rng.choice(len(possible_actions), p=priorities)]
            return self.rng.choice(possible_actions)

        # Greedy selection based on combined Q-values and TM predictions
        best_priority = float('-inf')
        best_action = possible_actions[0]
        
        for action in possible_actions:
            priority = self._calculate_priority(env, action)
            if priority > best_priority:
                best_priority = priority
                best_action = action
                
        return best_action

    def _update_models(self, env, action, prev_time: int) -> None:
        """Update both Q-table and Tsetlin Machine based on action results."""
        try:
            # Calculate rewards
            time_reward = -(env.total_time - prev_time)
            utils = env.get_machine_utilization()
            util_reward = np.mean(utils) * 100
            combined_reward = time_reward + util_reward
        
            # Update Q-value
            possible_actions = env.get_possible_actions()
            max_future_q = 0.0
            if possible_actions:
                max_future_q = max(
                    self.q_table[a.job, a.machine, a.operation]
                    for a in possible_actions
                )
                
            current_q = self.q_table[action.job, action.machine, action.operation]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                    combined_reward + self.discount_factor * max_future_q
            )
            self.q_table[action.job, action.machine, action.operation] = new_q
        
            # Debug prints
            print(f"Rewards - Time: {time_reward}, Util: {util_reward}, Combined: {combined_reward}")
            
            state_features = self.feature_transformer.transform(env.current_state)
            print(f"Transformed features shape: {state_features.shape}")
            
            action_tm = self._get_or_create_tm(action.job * env.num_machines + action.machine)
            print(f"TM input shape: {state_features.reshape(1, -1).shape}")

            # Update Tsetlin Machine
            state_features = self.feature_transformer.transform(env.current_state)
            print(f"State features: {state_features}")

            action_tm = self._get_or_create_tm(action.job * env.num_machines + action.machine)
            print(f"TM input shape: {state_features.reshape(1, -1).shape}")

            # Convert reward to binary class (1 for positive reward, 0 for negative)
            tm_class = 1 if combined_reward > 0 else 0
            action_tm.fit(state_features.reshape(1, -1), 
                        np.array([tm_class]), 
                        epochs=1)
        except Exception as e:
            print(f"Update models error: {e}")
            print(f"Action: {action}")
            print(f"Environment state: {env.current_state}")
            raise

    def _run_episode(self, env, max_steps: int = 1000) -> List[Any]:
        """Run a single episode using hybrid approach."""
        env.reset()
        episode_actions = []
        
        while not env.is_done() and len(episode_actions) < max_steps:
            prev_time = env.total_time
            
            action = self._select_action(env)
            if action is None:
                print("No valid actions left, checking environment state...")
                print(f"Completed jobs: {len(env.completed_jobs)}/{len(env.jobs)}")
                print(f"Current time: {env.total_time}")
                continue
                
            env.step(action)
            episode_actions.append(action)
            print(f"Action: {episode_actions}")
            print(f"Time: {env.total_time}")

            try:
                self._update_models(env, action, prev_time)
            except Exception as e:
                print(f"Error updating models: {e}")
                print(f"Current action: {action}")
                print(f"State features: ")
                print(self.feature_transformer.transform(env.current_state))
                raise
            
        return episode_actions

    def solve(self, env, max_steps: int = 1000) -> Tuple[List[Any], int]:
        """Solve using hybrid Q-learning and Tsetlin Machine approach."""
        start_time = time.time()
        
        # Initialize models
        if self.q_table is None:
            self._initialize_q_table(env)
        if self.feature_transformer is None:
            self._initialize_feature_transformer(env)
            
        logger.info("Starting hybrid training...")
        
        for episode in range(self.episodes):
            # Run episode
            episode_actions = self._run_episode(env, max_steps)
            
            # Evaluate episode
            env.reset()
            for action in episode_actions:
                env.step(action)
                
            # Track best solution
            if env.total_time < self.best_time:
                self.best_time = env.total_time
                self.best_schedule = episode_actions.copy()
                logger.info(f"New best makespan: {self.best_time}")
                
            # Decay exploration rate
            self.exploration_rate *= 0.9999
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}, "
                          f"Best makespan: {self.best_time}")
                
        # Final run with best actions
        env.reset()
        for action in self.best_schedule:
            env.step(action)
            
        solve_time = time.time() - start_time
        logger.info(f"Hybrid solver completed in {solve_time:.2f} seconds")
        logger.info(f"Final makespan: {env.total_time}")
        
        return self.best_schedule, env.total_time


class JSSPFeatureTransformer:
    """Memory-optimized feature transformer for JSSP state spaces"""
    
    def __init__(self, 
                 n_jobs: int = None,
                 n_machines: int = None,
                 machine_time_bits: int = 8,
                 job_status_bits: int = 4,
                 obs_space_size: int = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the feature transformer.
        """
        # Handle both config dict and direct arguments
        if config is not None:
            self.n_jobs = config['n_jobs']
            self.n_machines = config['n_machines']
            self.machine_time_bits = min(config.get('machine_time_bits', 8), 8)
            self.job_status_bits = min(config.get('job_status_bits', 4), 4)
        else:
            if n_jobs is None or n_machines is None:
                raise ValueError("Must provide either config dict or n_jobs and n_machines")
            self.n_jobs = n_jobs
            self.n_machines = n_machines
            self.machine_time_bits = min(machine_time_bits, 8)
            self.job_status_bits = min(job_status_bits, 4)
        
        # Calculate feature sizes for each component
        self.machine_times_size = self.n_machines * self.machine_time_bits
        self.job_status_size = self.n_jobs * self.job_status_bits
        self.op_status_size = self.n_jobs * self.n_machines
        
        
        # Calculate total binary features
        self.total_features = (
            self.machine_times_size +  # Machine times binary encoding
            self.job_status_size +     # Job status binary encoding
            self.op_status_size        # Operation status matrix
        )
        
        self._log_initialization()

    def _log_initialization(self):
        """Log initialization details"""
        print(f"Feature transformer initialized:")
        print(f"- Problem size: {self.n_jobs}x{self.n_machines}")
        print(f"- Total features: {self.total_features}")
        print(f"- Feature composition:")
        print(f"  ****Machine times: {self.n_machines} machines × {self.machine_time_bits} bits = {self.machine_times_size}")
        print(f"  ***Job status: {self.n_jobs} jobs × {self.job_status_bits} bits = {self.job_status_size}")
        print(f"  ***Operation status: {self.n_jobs} jobs × {self.n_machines} = {self.op_status_size}")
        

    def transform(self, state) -> np.ndarray:
        """
        Transform JSSP state into binary features.
        """
        try:
           # print(f"Incoming state attributes: {dir(state)}")  # Debug state structure
            print(f"Machine availability: {state.machine_availability}")
            print(f"Job progress length: {len(state.job_progress)}")
            
            binary_features = np.zeros(self.total_features, dtype=np.int32)
            current_idx = 0

            # Debug machine times processing
            machine_times = state.machine_availability
            print(f"Machine times: {machine_times}")
            max_time = np.max(machine_times) if np.max(machine_times) > 0 else 1
            normalized_times = (machine_times / max_time * (2**self.machine_time_bits - 1)).astype(int)
            print(f"Normalized times: {normalized_times}")
       
            
        
            for machine_idx in range(len(normalized_times)):
                time = normalized_times[machine_idx]
                binary = format(min(time, 2**self.machine_time_bits - 1), 
                            f'0{self.machine_time_bits}b')
                for bit in binary:
                    if current_idx < self.machine_times_size:
                        binary_features[current_idx] = int(bit)
                        current_idx += 1

            
            print(f"Processing machines: {len(state.machine_availability)} machines")
            
            # 2. Process job progress
            job_progress = state.job_progress
            for job_idx in range(self.n_jobs):
                # Calculate job progress as fraction of completed operations
                if job_idx < len(job_progress):
                    completed_ops = sum(1 for op_progress in job_progress[job_idx] if op_progress > 0)
                    total_ops = len(job_progress[job_idx])
                    progress = completed_ops / max(1, total_ops)
                else:
                    progress = 0

                # Convert progress to binary representation
                norm_progress = int(min(progress, 1.0) * (2**self.job_status_bits - 1))
                binary = format(norm_progress, f'0{self.job_status_bits}b')
                for bit in binary:
                    if current_idx < self.machine_times_size + self.job_status_size:
                        binary_features[current_idx] = int(bit)
                        current_idx += 1

            print(f"Processing jobs: {len(state.job_progress)} jobs")

            # 3. Process operation status matrix
            for job_idx in range(self.n_jobs):
                if job_idx < len(job_progress):
                    # Get the operations for this job
                    job_ops = job_progress[job_idx]
                    for machine_idx in range(self.n_machines):
                        if current_idx < self.total_features:
                            # Check if any operation on this machine is completed
                            has_completed_op = False
                            for op_idx, op_progress in enumerate(job_ops):
                                if hasattr(state, 'jobs') and job_idx < len(state.jobs):
                                    job = state.jobs[job_idx]
                                    if (op_idx < len(job.operations) and 
                                        job.operations[op_idx].machine == machine_idx and 
                                        op_progress > 0):
                                        has_completed_op = True
                                        break
                            binary_features[current_idx] = int(has_completed_op)
                            current_idx += 1
                else:
                    # Fill remaining features with zeros
                    for _ in range(self.n_machines):
                        if current_idx < self.total_features:
                            binary_features[current_idx] = 0
                            current_idx += 1

            print(f"Final feature count: {current_idx} / {self.total_features}")
            
            return binary_features
        
        except Exception as e:
            print(f"Transform error: {e}")
        print(f"Current index: {current_idx}")
        print(f"Binary features shape: {binary_features.shape}")
        raise ValueError("Error during feature transformation")


    def transform_batch(self, states: List) -> np.ndarray:
        """Transform a batch of states into binary features"""
        return np.array([self.transform(state) for state in states])
        
    def get_state_shape(self) -> str:
        """Get a string representation of the state shape"""
        return (f"State shape: machine_times({self.machine_times_size}) + "
                f"job_status({self.job_status_size}) + "
                f"operation_status({self.op_status_size})")