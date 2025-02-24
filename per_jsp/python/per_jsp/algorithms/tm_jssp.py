import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

logger = logging.getLogger(__name__)

class TsetlinScheduler:
    def __init__(self,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 episodes: int = 1000,
                 nr_clauses: int = 2000,
                 T: float = 1500,
                 s: float = 1.5,
                 optimal_makespan: int = None):
        
        # Learning parameters
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        self.optimal_makespan = optimal_makespan
        
        # Tsetlin Machine parameters
        self.nr_clauses = nr_clauses
        self.T = T
        self.s = s
        
        # Initialize state trackers
        self.tsetlin_machines = {}  # Dict to store TM for each state-action pair
        self.best_time = float('inf')
        self.best_schedule = []
        self.rng = np.random.RandomState()
        self.feature_transformer = None

    def _initialize_feature_transformer(self, env) -> None:
        """Initialize the feature transformer for the Tsetlin Machine."""
        self.feature_transformer = JSSPFeatureTransformer(
            n_jobs=len(env.jobs),
            n_machines=env.num_machines,
            machine_time_bits=8,
            job_status_bits=4
        )

    def _get_or_create_tm(self, state_action_id: str) -> MultiClassTsetlinMachine:
        """Create TM for state-action pair"""
        try:
            if state_action_id not in self.tsetlin_machines:
                tm = MultiClassTsetlinMachine(
                    self.nr_clauses,
                    self.T,
                    self.s,
                    number_of_classes=2  # Increased classes for finer reward granularity
                )
                self.tsetlin_machines[state_action_id] = tm
            return self.tsetlin_machines[state_action_id]
        except Exception as e:
            print(f"TM creation error: {e}")
            raise

    def _calculate_priority(self, env, action) -> float:
        """Calculate priority using TM prediction and scheduling factors."""
        state_features = self.feature_transformer.transform(env.current_state)
        state_action_id = f"{hash(str(state_features))}-{action.job}-{action.machine}"
        tm = self._get_or_create_tm(state_action_id)
        
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
        
        # Combined priority score with multiple factors
        priority = (0.5 * tm_prediction +  # Normalized TM prediction
                   0.3 * (1 - machine_utilization) +  # Machine availability
                   0.2 * (1/max(1, remaining_time)))  # Job urgency
        
        return priority

    def _select_action(self, env) -> Any:
        """Select action with exploration vs exploitation."""
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            return None
            
        try:
            # Enhanced exploration vs exploitation
            if self.rng.random() < self.exploration_rate:
                priorities = []
                valid_actions = []
                
                for action in possible_actions:
                    try:
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

    def _update_tsetlin(self, env, action, prev_time):
        """Update TM based on action outcome"""
        try:
            state_features = self.feature_transformer.transform(env.current_state)
            state_action_id = f"{hash(str(state_features))}-{action.job}-{action.machine}"
            tm = self._get_or_create_tm(state_action_id)

            # Calculate rewards based on multiple factors
            time_diff = env.total_time - prev_time
            utils = env.get_machine_utilization()
            utilization = np.mean(utils)
            
            # Normalize time difference relative to total time
            normalized_time_diff = time_diff / max(1, env.total_time)
            
            # Calculate combined reward focusing on both time and utilization
            combined_reward = (0.7 * (1 - normalized_time_diff) + 
                             0.3 * utilization)
            
            # Convert to binary class (0 or 1)
            reward_class = 1 if combined_reward > 0.5 else 0
            
            # Update TM with reward class
            features_reshaped = state_features.reshape(1, -1)
            tm.fit(features_reshaped, np.array([reward_class]), epochs=50)

        except Exception as e:
            logging.error(f"Error in Tsetlin update: {e}")
            raise

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
            self._update_tsetlin(env, action, prev_time)

        return episode_actions

    def solve(self, env, max_steps: int = 1000) -> Tuple[List[Any], int]:
        """Main solving method with solution verification."""
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
                logging.info(f"New best solution found! Makespan: {best_makespan}")
                
            # Decay exploration
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
        """Initialize the feature transformer."""
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
        
        # Calculate feature sizes
        self.machine_times_size = self.n_machines * self.machine_time_bits
        self.job_status_size = self.n_jobs * self.job_status_bits
        self.op_status_size = self.n_jobs
        
        # Total binary features
        self.total_features = (
            self.machine_times_size +
            self.job_status_size +
            self.op_status_size
        )

    def transform(self, state) -> np.ndarray:
        """Transform JSSP state into binary features."""
        try:
            binary_features = np.zeros(self.total_features, dtype=np.int32)
            current_idx = 0

            # Process machine times
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

            # Process job progress
            job_progress = state.job_progress
            for job_idx in range(self.n_jobs):
                if job_idx < len(job_progress):
                    completed_ops = sum(1 for op_progress in job_progress[job_idx] if op_progress > 0)
                    total_ops = len(job_progress[job_idx])
                    progress = completed_ops / max(1, total_ops)
                else:
                    progress = 0

                norm_progress = int(min(progress, 1.0) * (2**self.job_status_bits - 1))
                binary = format(norm_progress, f'0{self.job_status_bits}b')
                for bit in binary:
                    if current_idx < self.machine_times_size + self.job_status_size:
                        binary_features[current_idx] = int(bit)
                        current_idx += 1

            # Process operation status
            for job_idx in range(self.n_jobs):
                if job_idx < len(job_progress):
                    job_ops = job_progress[job_idx]
                    for machine_idx in range(self.n_machines):
                        if current_idx < self.total_features:
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
                    for _ in range(self.n_machines):
                        if current_idx < self.total_features:
                            binary_features[current_idx] = 0
                            current_idx += 1
            
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