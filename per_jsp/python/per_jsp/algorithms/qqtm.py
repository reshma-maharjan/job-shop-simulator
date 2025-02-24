import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from collections import defaultdict

logger = logging.getLogger(__name__)

class HybridTsetlinQLearningScheduler:
    """
    An optimized hybrid scheduler combining Q-learning with Tsetlin Machine for job shop scheduling.
    Uses Q-learning for high-level policy learning and Tsetlin Machine for feature-based action selection.
    """
    def __init__(self,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 episodes: int = 2000,
                 nr_clauses: int = 4000,
                 T: float = 2000,
                 s: float = 2.0,
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
        
        # Action tracking
        self.action_visits = defaultdict(int)
        self.total_steps = 0
        
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
            machine_time_bits=12,
            job_status_bits=6
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
            logger.error(f"TM creation error: {e}")
            raise

    def _encode_slack_times(self, state, env) -> np.ndarray:
        """Encode slack times using job_progress information."""
        features = np.zeros(self.slack_time_size, dtype=np.int32)
        max_time = max(state.machine_availability) if max(state.machine_availability) > 0 else 1
        
        for machine_idx in range(self.n_machines):
            current_time = state.machine_availability[machine_idx]
            
            # Calculate future operations duration using job_progress
            future_ops_duration = 0
            for job_idx, job in enumerate(env.jobs):
                for op_idx, op in enumerate(job.operations):
                    # Check if operation is not completed using job_progress
                    if op.machine == machine_idx and not state.job_progress[job_idx][op_idx]:
                        future_ops_duration += op.duration
            
            slack = max(0, max_time - (current_time + future_ops_duration))
            
            # Encode slack time
            norm_slack = int((slack / max_time) * (2**self.machine_time_bits - 1))
            binary = format(min(norm_slack, 2**self.machine_time_bits - 1),
                        f'0{self.machine_time_bits}b')
            
            start_idx = machine_idx * self.machine_time_bits
            for bit_idx, bit in enumerate(binary):
                features[start_idx + bit_idx] = int(bit)
                
        return features

    def _calculate_critical_path_length(self, env, action) -> float:
        """Calculate the critical path length if this action is chosen."""
        remaining_ops = len(env.jobs[action.job].operations) - action.operation - 1
        return remaining_ops * env.jobs[action.job].operations[action.operation].duration

    def _calculate_urgency(self, env, action) -> float:
        """Calculate operation urgency based on waiting time and remaining work."""
        waiting_time = env.current_state.machine_availability[action.machine]
        total_remaining = sum(op.duration for op in env.jobs[action.job].operations[action.operation:])
        return waiting_time / (total_remaining + 1)

    def _calculate_workload_balance(self, env, action) -> float:
        """Calculate workload balance factor using job_progress."""
        machine_loads = [0] * env.num_machines
        
        # Calculate load for each machine based on remaining operations
        for job_idx, job in enumerate(env.jobs):
            job_progress = env.current_state.job_progress[job_idx]
            for op_idx, op in enumerate(job.operations):
                # Check if operation is not completed using job_progress
                if not job_progress[op_idx]:
                    machine_loads[op.machine] += 1
                
        current_load = machine_loads[action.machine]
        avg_load = sum(machine_loads) / len(machine_loads)
        return 1 / (1 + abs(current_load - avg_load))

    def _calculate_slack(self, env, action) -> float:
        """Calculate slack time for a given action.
        
        Args:
            env: The job shop environment
            action: The action to calculate slack for
            
        Returns:
            float: The calculated slack time
        """
        # Get current time for the machine
        current_time = env.current_state.machine_availability[action.machine]
        
        # Calculate total remaining work on this machine
        remaining_work = 0
        for job_idx, job in enumerate(env.jobs):
            for op_idx, op in enumerate(job.operations):
                # Check if operation is not completed and uses this machine
                if op.machine == action.machine and not env.current_state.job_progress[job_idx][op_idx]:
                    remaining_work += op.duration
        
        # Calculate maximum completion time across all machines
        max_completion = max(env.current_state.machine_availability)
        
        # Slack is the difference between max completion time and 
        # (current time + remaining work)
        slack = max(0, max_completion - (current_time + remaining_work))
        
        return slack
    
    def _calculate_priority(self, env, action) -> float:
        """Calculate priority score using multiple factors."""
        # Get state features for TM with environment
        state_features = self.feature_transformer.transform(env.current_state, env)
        
        # Get TM prediction
        tm_index = action.job * env.num_machines + action.machine
        tm = self._get_or_create_tm(tm_index)
        tm_prediction = tm.predict(state_features.reshape(1, -1))[0]
        
        # Calculate various priority factors
        critical_path = self._calculate_critical_path_length(env, action)
        slack = self._calculate_slack(env, action)
        urgency = self._calculate_urgency(env, action)
        workload_balance = self._calculate_workload_balance(env, action)
        
        # Normalize factors
        max_critical_path = max(1, env.total_time)
        normalized_critical_path = critical_path / max_critical_path
        normalized_slack = slack / max_critical_path
        
        # Weighted combination of all factors
        priority = (
            0.3 * tm_prediction +
            0.25 * normalized_critical_path +
            0.2 * urgency +
            0.15 * (1 - normalized_slack) +  # Invert slack so higher is better
            0.1 * workload_balance
        )
        
        return priority

    def _select_action(self, env) -> Any:
        """Select action using UCB exploration strategy."""
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            return None
            
        # Calculate completion percentage from job progress
        total_operations = sum(len(progress) for progress in env.current_state.job_progress)
        completed_operations = sum(sum(1 for op in progress if op > 0) 
                                for progress in env.current_state.job_progress)
        completion_percentage = completed_operations / max(1, total_operations)
            
        # Adaptive exploration rate based on completion
        local_exploration_rate = self.exploration_rate * (1 - completion_percentage)
        
        if self.rng.random() < local_exploration_rate:
            # Smart exploration using UCB
            ucb_scores = []
            for action in possible_actions:
                priority = self._calculate_priority(env, action)
                visits = self.action_visits[(action.job, action.machine)]
                ucb = priority + np.sqrt(2 * np.log(self.total_steps) / (visits + 1))
                ucb_scores.append(ucb)
                
            return possible_actions[np.argmax(ucb_scores)]
        else:
            # Greedy selection with tie-breaking
            best_actions = []
            best_priority = float('-inf')
            
            for action in possible_actions:
                priority = self._calculate_priority(env, action)
                if abs(priority - best_priority) < 1e-6:  # Within epsilon
                    best_actions.append(action)
                elif priority > best_priority:
                    best_priority = priority
                    best_actions = [action]
            
            return self.rng.choice(best_actions)
    
    def _update_models(self, env, action, prev_time):
        try:
            # Calculate rewards
            time_reward = -(env.total_time - prev_time)
            utils = env.get_machine_utilization()
            util_reward = np.mean(utils) * 10
            
            # Combined reward
            combined_reward = 0.8 * time_reward + 0.2 * util_reward
            scaled_reward = combined_reward * 100
            
            # Update action visits count
            self.action_visits[(action.job, action.machine)] += 1
            self.total_steps += 1
            
            # Get future Q-values
            possible_actions = env.get_possible_actions()
            future_q_values = []
            
            for a in possible_actions:
                if self._is_valid_action(a):
                    future_q_values.append(
                        self.q_table[a.job, a.machine, a.operation]
                    )
            
            max_future_q = max(future_q_values) if future_q_values else 0.0
            current_q = self.q_table[action.job, action.machine, action.operation]
            
            # Update Q-value
            new_q = current_q + self.learning_rate * (
                scaled_reward + self.discount_factor * max_future_q - current_q
            )
            
            self.q_table[action.job, action.machine, action.operation] = new_q
            
            # Update TM with environment
            state_features = self.feature_transformer.transform(env.current_state, env)
            action_tm = self._get_or_create_tm(action.job * env.num_machines + action.machine)
            tm_class = 1 if new_q > current_q else 0
            
            features_reshaped = state_features.reshape(1, -1)
            action_tm.fit(features_reshaped, np.array([tm_class]), epochs=100)
            
        except Exception as e:
            logger.error(f"Error in model updates: {e}")
            raise

    def _is_valid_action(self, action):
        """Check if action is valid within Q-table dimensions."""
        return (action.job < self.q_table.shape[0] and
                action.machine < self.q_table.shape[1] and 
                action.operation < self.q_table.shape[2])

    def _run_episode(self, env, max_steps: int = 1000) -> List[Any]:
        """Run a single episode."""
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
        """Solve the job shop scheduling problem."""
        if self.q_table is None:
            self._initialize_q_table(env)
        if self.feature_transformer is None:
            self._initialize_feature_transformer(env)
            
        logger.info(f"Starting solution for {len(env.jobs)} jobs on {env.num_machines} machines")
        
        best_episode_actions = None
        best_makespan = float('inf')
        no_improvement_count = 0
        prev_best_makespan = float('inf')
        
        for episode in range(self.episodes):
            # Run episode
            episode_actions = self._run_episode(env, max_steps)
            
            # Verify solution
            env.reset()
            for action in episode_actions:
                env.step(action)
                
            # Check completion and update best solution
            completed_jobs = sum(1 for job in range(len(env.jobs)) 
                               if all(env.current_state.job_progress[job]))
                               
            if completed_jobs == len(env.jobs) and env.total_time < best_makespan:
                best_makespan = env.total_time
                best_episode_actions = episode_actions.copy()
                no_improvement_count = 0
                logger.info(f"New best solution found! Makespan: {best_makespan}\n")
            else:
                no_improvement_count += 1
            
            # Early stopping check
            if no_improvement_count > 200:
                if abs(best_makespan - self.optimal_makespan) < 1e-6:
                    logger.info("Reached optimal makespan. Stopping early.")
                    break
                elif best_makespan < prev_best_makespan:
                    logger.info("No further improvement possible. Stopping early.")
                    break
            
            # Decay exploration rate
            self.exploration_rate = max(0.01, self.exploration_rate * 0.999)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}, Best makespan: {best_makespan}")
            
            prev_best_makespan = best_makespan
        
        if best_episode_actions is None:
            logger.error("No complete solution found!")
            return [], 0
            
        # Final verification
        env.reset()
        for action in best_episode_actions:
            env.step(action)
            
        return best_episode_actions, env.total_time


class JSSPFeatureTransformer:
    """Enhanced feature transformer for JSSP state spaces."""
    
    def __init__(self, 
                 n_jobs: int = None,
                 n_machines: int = None,
                 machine_time_bits: int = 12,
                 job_status_bits: int = 6,
                 obs_space_size: int = None,
                 config: Dict[str, Any] = None):
        
        if config is not None:
            self.n_jobs = config['n_jobs']
            self.n_machines = config['n_machines']
            self.machine_time_bits = min(config.get('machine_time_bits', 12), 12)
            self.job_status_bits = min(config.get('job_status_bits', 6), 6)
        else:
            if n_jobs is None or n_machines is None:
                raise ValueError("Must provide either config dict or n_jobs and n_machines")
            self.n_jobs = n_jobs
            self.n_machines = n_machines
            self.machine_time_bits = min(machine_time_bits, 12)
            self.job_status_bits = min(job_status_bits, 6)
        
        # Calculate feature sizes
        self.machine_times_size = self.n_machines * self.machine_time_bits
        self.job_status_size = self.n_jobs * self.job_status_bits
        self.op_status_size = self.n_jobs
        self.slack_time_size = self.n_machines * self.machine_time_bits
        self.workload_size = self.n_machines * self.machine_time_bits
        
        # Calculate total features
        self.total_features = (
            self.machine_times_size +
            self.job_status_size +
            self.op_status_size +
            self.slack_time_size +
            self.workload_size
        )
        
        self._log_initialization()

    def _log_initialization(self):
        """Log initialization details"""
        print(f"Enhanced Feature transformer initialized:")
        print(f"  Machine times: {self.n_machines} machines × {self.machine_time_bits} bits = {self.machine_times_size}")
        print(f"  Job status: {self.n_jobs} jobs × {self.job_status_bits} bits = {self.job_status_size}")
        print(f"  Operation status: {self.n_jobs} jobs = {self.op_status_size}")
        print(f"  Slack time features: {self.slack_time_size}")
        print(f"  Workload features: {self.workload_size}")
        print(f"  Total features: {self.total_features}")

    def _encode_machine_times(self, machine_times, max_time) -> np.ndarray:
        features = np.zeros(self.machine_times_size, dtype=np.int32)
        normalized_times = (machine_times / max_time * (2**self.machine_time_bits - 1)).astype(int)
        
        for machine_idx, time in enumerate(normalized_times):
            binary = format(min(time, 2**self.machine_time_bits - 1), 
                          f'0{self.machine_time_bits}b')
            start_idx = machine_idx * self.machine_time_bits
            for bit_idx, bit in enumerate(binary):
                features[start_idx + bit_idx] = int(bit)
                
        return features

    def _encode_job_status(self, job_progress) -> np.ndarray:
        features = np.zeros(self.job_status_size, dtype=np.int32)
        
        for job_idx in range(self.n_jobs):
            if job_idx < len(job_progress):
                completed_ops = sum(1 for op_progress in job_progress[job_idx] if op_progress > 0)
                total_ops = len(job_progress[job_idx])
                progress = completed_ops / max(1, total_ops)
            else:
                progress = 0
                
            norm_progress = int(min(progress, 1.0) * (2**self.job_status_bits - 1))
            binary = format(norm_progress, f'0{self.job_status_bits}b')
            
            start_idx = job_idx * self.job_status_bits
            for bit_idx, bit in enumerate(binary):
                features[start_idx + bit_idx] = int(bit)
                
        return features

    def _encode_operation_status(self, state) -> np.ndarray:
        features = np.zeros(self.op_status_size, dtype=np.int32)
        job_progress = state.job_progress
        
        for job_idx in range(self.n_jobs):
            if job_idx < len(job_progress):
                any_completed = any(op_progress > 0 for op_progress in job_progress[job_idx])
                features[job_idx] = int(any_completed)
                
        return features

    def _encode_slack_times(self, state, env) -> np.ndarray:
        features = np.zeros(self.slack_time_size, dtype=np.int32)
        max_time = max(state.machine_availability) if max(state.machine_availability) > 0 else 1
        
        for machine_idx in range(self.n_machines):
            current_time = state.machine_availability[machine_idx]
            
            # Get future operations for this machine from env.jobs and job_progress
            future_ops_duration = 0
            for job_idx, job in enumerate(env.jobs):
                for op_idx, op in enumerate(job.operations):
                    if op.machine == machine_idx and not state.job_progress[job_idx][op_idx]:
                        future_ops_duration += op.duration
            
            slack = max(0, max_time - (current_time + future_ops_duration))
            
            # Encode slack time
            norm_slack = int((slack / max_time) * (2**self.machine_time_bits - 1))
            binary = format(min(norm_slack, 2**self.machine_time_bits - 1),
                          f'0{self.machine_time_bits}b')
            
            start_idx = machine_idx * self.machine_time_bits
            for bit_idx, bit in enumerate(binary):
                features[start_idx + bit_idx] = int(bit)
                
        return features

    def _encode_workload(self, state, env) -> np.ndarray:
        """Encode workload using job_progress information."""
        features = np.zeros(self.workload_size, dtype=np.int32)
        
        # Calculate workload for each machine using job_progress
        machine_loads = np.zeros(self.n_machines)
        for job_idx, job in enumerate(env.jobs):
            for op_idx, op in enumerate(job.operations):
                # Check if operation is not completed using job_progress
                if not state.job_progress[job_idx][op_idx]:
                    machine_loads[op.machine] += op.duration
        
        # Normalize and encode workload
        max_load = max(machine_loads) if max(machine_loads) > 0 else 1
        normalized_loads = (machine_loads / max_load * (2**self.machine_time_bits - 1)).astype(int)
        
        for machine_idx, load in enumerate(normalized_loads):
            binary = format(min(load, 2**self.machine_time_bits - 1),
                        f'0{self.machine_time_bits}b')
            
            start_idx = machine_idx * self.machine_time_bits
            for bit_idx, bit in enumerate(binary):
                features[start_idx + bit_idx] = int(bit)
                
        return features

    def transform(self, state, env) -> np.ndarray:
        """Transform state into binary features using environment information."""
        try:
            features = np.zeros(self.total_features, dtype=np.int32)
            current_idx = 0
            
            # 1. Encode machine times
            max_time = max(state.machine_availability) if max(state.machine_availability) > 0 else 1
            machine_time_features = self._encode_machine_times(state.machine_availability, max_time)
            features[current_idx:current_idx + self.machine_times_size] = machine_time_features
            current_idx += self.machine_times_size
            
            # 2. Encode job progress
            job_status_features = self._encode_job_status(state.job_progress)
            features[current_idx:current_idx + self.job_status_size] = job_status_features
            current_idx += self.job_status_size
            
            # 3. Encode operation status
            op_status_features = self._encode_operation_status(state)
            features[current_idx:current_idx + self.op_status_size] = op_status_features
            current_idx += self.op_status_size
            
            # 4. Encode slack times using environment information
            slack_features = self._encode_slack_times(state, env)
            features[current_idx:current_idx + self.slack_time_size] = slack_features
            current_idx += self.slack_time_size
            
            # 5. Encode workload using environment information
            workload_features = self._encode_workload(state, env)
            features[current_idx:current_idx + self.workload_size] = workload_features
            
            return features
            
        except Exception as e:
            print(f"Transform error: {e}")
            print(f"Current index: {current_idx}")
            print(f"Binary features shape: {features.shape}")
            raise ValueError("Error during feature transformation")

    def transform_batch(self, states: List, env) -> np.ndarray:
        """Transform a batch of states into binary features."""
        return np.array([self.transform(state, env) for state in states])