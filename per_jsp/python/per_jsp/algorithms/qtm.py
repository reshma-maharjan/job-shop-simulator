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
                 nr_clauses: int = 2000, #2000
                 T: float = 1500, # 1500
                 s: float = 1.5, # 1.5
                 optimal_makespan: int = None):
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        self.optimal_makespan = optimal_makespan
        
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
                    self.s,
                    number_of_classes=2,
                )
                
                self.tsetlin_machines[action_id] = tm

            return self.tsetlin_machines[action_id]
        
        except Exception as e:
            print(f"TM creation error: {e}")
            print(f"Total features: {self.feature_transformer.total_features}")
            raise

    def _calculate_priority(self, env, action) -> float:

        """Calculate priority score using Tsetlin Machine prediction and scheduling factors."""
        # Get state features for TM
        state_features = self.feature_transformer.transform(env.current_state)
        
        # Get or create TM for this job-machine combination
        tm_index = action.job * env.num_machines + action.machine
        tm = self._get_or_create_tm(tm_index)
        
        # Get TM prediction
        tm_prediction = tm.predict(state_features.reshape(1, -1))[0]
        
        # Calculate remaining time
        remaining_time = sum(
            op.duration
            for op in env.jobs[action.job].operations[action.operation:]
        )
        
        # Calculate machine utilization
        machine_time = env.current_state.machine_availability[action.machine]
        total_time = max(1, env.total_time)
        machine_utilization = machine_time / total_time
        
        # Basic scheduling priority
        scheduling_priority = remaining_time * (1 - machine_utilization)
        
        # Combine TM prediction with scheduling priority (weighted sum)
        final_priority = 0.7 * scheduling_priority + 0.3 * tm_prediction
        
        return final_priority

    def _select_action(self, env) -> Any:
        """Select action with improved validation and debugging."""
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            return None
            
        try:
            # Log available actions for debugging
            logging.debug(f"Available actions: {len(possible_actions)}")
            for i, action in enumerate(possible_actions):
                logging.debug(f"Action {i}: Job {action.job}, Machine {action.machine}")
            
            # Enhanced exploration vs exploitation
            if self.rng.random() < self.exploration_rate:
                # Use smart exploration
                priorities = []
                valid_actions = []
                
                for action in possible_actions:
                    try:
                        # Validate action before calculating priority
                        if (0 <= action.job < len(env.jobs) and 
                            0 <= action.machine < env.num_machines):
                            priority = self._calculate_priority(env, action)
                            priorities.append(max(0.0, priority))
                            valid_actions.append(action)
                    except Exception as e:
                        logging.warning(f"Error calculating priority: {e}")
                        continue
                
                if valid_actions:
                    total_priority = sum(priorities)
                    if total_priority > 0:
                        probabilities = [p/total_priority for p in priorities]
                        return valid_actions[self.rng.choice(len(valid_actions), p=probabilities)]
                    return self.rng.choice(valid_actions)
            
            # Greedy selection
            best_action = None
            best_priority = float('-inf')

            for action in possible_actions:
                try:
                    priority = self._calculate_priority(env, action)
                    if priority > best_priority:
                        best_priority = priority
                        best_action = action
                except Exception as e:
                    logging.warning(f"Error in greedy selection: {e}")
                    continue
            
            if best_action is None:
                logging.warning("No valid action found in greedy selection")
                return possible_actions[0]
                
            return best_action
            
        except Exception as e:
            logging.error(f"Error in action selection: {e}")
            if possible_actions:
                return possible_actions[0]
            return None
    
    def _update_models(self, env, action, prev_time):
        try:
            # Calculate rewards based on time and utilization
            time_reward = -(env.total_time - prev_time)
            utils = env.get_machine_utilization()
            util_reward = np.mean(utils) * 10

            # Combined reward focusing on minimizing time 
            combined_reward = 0.8 * time_reward + 0.2 * util_reward
            scaled_reward = combined_reward * 100

            # Get future Q-values and update Q-table
            possible_actions = env.get_possible_actions()
            future_q_values = []

            for a in possible_actions:
                if self._is_valid_action(a):
                    future_q_values.append(
                        self.q_table[a.job, a.machine, a.operation]
                    )

            max_future_q = max(future_q_values) if future_q_values else 0.0
            current_q = self.q_table[action.job, action.machine, action.operation]
            
            new_q = current_q + self.learning_rate * (
                scaled_reward + self.discount_factor * max_future_q - current_q
            )
            
            self.q_table[action.job, action.machine, action.operation] = new_q

            # Update TM
            state_features = self.feature_transformer.transform(env.current_state)
            action_tm = self._get_or_create_tm(action.job * env.num_machines + action.machine)
            tm_class = 1 if new_q > current_q else 0
            
            features_reshaped = state_features.reshape(1, -1)
            action_tm.fit(features_reshaped, np.array([tm_class]), epochs=100)

        except Exception as e:
            logging.error(f"Error in model updates: {e}")
            raise

    def _is_valid_action(self, action):
        return (action.job < self.q_table.shape[0] and
                action.machine < self.q_table.shape[1] and 
                action.operation < self.q_table.shape[2])


    def _run_episode(self, env, max_steps: int = 1000) -> List[Any]:
        """Run a single episode with enhanced state tracking."""
        env.reset()
        episode_actions = []

        while not env.is_done() and len(episode_actions) < max_steps:
            prev_time = env.total_time

            action = self._select_action(env)
            if action is None:
                break

            env.step(action)
            episode_actions.append(action)

            self._update_models(env, action, prev_time)

        return episode_actions

    def solve(self, env, max_steps: int = 1000) -> Tuple[List[Any], int]:
        """Enhanced solve method with job completion verification."""
        if self.q_table is None:
            self._initialize_q_table(env)
        if self.feature_transformer is None:
            self._initialize_feature_transformer(env)
            
        logging.info(f"Starting solution for {len(env.jobs)} jobs on {env.num_machines} machines")
        
        best_episode_actions = None
        best_makespan = float('inf')
        
        for episode in range(self.episodes):
            
            # Run episode
            episode_actions = self._run_episode(env, max_steps)
            
            # Verify solution
            env.reset()
            
            for action in episode_actions:
                env.step(action)
                
            # Check completion
            completed_jobs = sum(1 for job in range(len(env.jobs)) 
                            if all(env.current_state.job_progress[job]))
            
            # Update best solution if better
            if completed_jobs == len(env.jobs) and env.total_time < best_makespan:
                best_makespan = env.total_time
                best_episode_actions = episode_actions.copy()
                logging.info(f"New best solution found! Makespan: {best_makespan}\n")
                
            # Decay exploration rate
            self.exploration_rate = max(0.01, self.exploration_rate * 0.999)

            if (episode + 1) % 10 == 0:
                logging.info(f"Episode {episode + 1}/{self.episodes}, Best makespan: {best_makespan}")
            
        if best_episode_actions is None:
            logging.error("No complete solution found!")
            return [], 0
            
        # Final verification
        env.reset()
        for action in best_episode_actions:
            env.step(action)
            
        return best_episode_actions, env.total_time


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
        #self.op_status_size = self.n_jobs * self.n_machines
        self.op_status_size = self.n_jobs
        
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
        print(f"  ****Machine times: {self.n_machines} machines × {self.machine_time_bits} bits = {self.machine_times_size}")
        print(f"  ***Job status: {self.n_jobs} jobs × {self.job_status_bits} bits = {self.job_status_size}")
        #print(f"  ***Operation status: {self.n_jobs} jobs × {self.n_machines} = {self.op_status_size}")
        print(f"  ***Operation status: {self.n_jobs} jobs  = {self.op_status_size}")


    def transform(self, state) -> np.ndarray:
        """
        Transform JSSP state into binary features.
        """
        try:
            
            binary_features = np.zeros(self.total_features, dtype=np.int32)
            current_idx = 0

            # Debug machine times processing
            machine_times = state.machine_availability
            max_time = np.max(machine_times) if np.max(machine_times) > 0 else 1
            normalized_times = (machine_times / max_time * (2**self.machine_time_bits - 1)).astype(int)
        
            for machine_idx in range(len(normalized_times)):
                time = normalized_times[machine_idx]
                binary = format(min(time, 2**self.machine_time_bits - 1), 
                            f'0{self.machine_time_bits}b')
                for bit in binary:
                    if current_idx < self.machine_times_size:
                        binary_features[current_idx] = int(bit)
                        current_idx += 1
            
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

            #print(f"Final feature count: {current_idx} / {self.total_features}")
            
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