import numpy as np
import time
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action

logger = logging.getLogger(__name__)

class QLearningScheduler(BaseScheduler):
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,  # Increased discount factor
                 exploration_rate: float = 1.0,
                 episodes: int = 10000,  # Increased episodes
                 min_exploration_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.episodes = episodes
        
        self.q_table = None
        self.best_time = float('inf')
        self.best_schedule = []
        self.rng = np.random.RandomState()
        
        # New features for improved learning
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def _initialize_q_table(self, env: JobShopEnvironment) -> None:
        """Initialize Q-table with enhanced state representation."""
        max_operations = max(len(job.operations) for job in env.jobs)
        num_state_features = 5  # Increased state features
        
        self.q_table = np.zeros((
            len(env.jobs),
            env.num_machines,
            max_operations,
            num_state_features
        ))

    def _get_state_features(self, env: JobShopEnvironment, action: Action) -> np.ndarray:
        """Get enhanced state features for better decision making."""
        features = np.zeros(5)
        
        # 1. Job progress
        total_ops = len(env.jobs[action.job].operations)
        current_op = action.operation
        features[0] = current_op / total_ops
        
        # 2. Machine utilization
        machine_util = env.get_machine_utilization()[action.machine]
        features[1] = machine_util
        
        # 3. Job urgency (remaining operations / total time)
        remaining_ops = total_ops - current_op
        features[2] = remaining_ops / (env.total_time + 1)
        
        # 4. Machine load (number of pending operations)
        pending_ops = sum(1 for a in env.get_possible_actions() 
                         if a.machine == action.machine)
        features[3] = pending_ops / len(env.jobs)
        
        # 5. Operation criticality
        critical_path = env.get_critical_path()
        features[4] = 1.0 if (action.job, action.operation) in critical_path else 0.0
        
        return features

    def _calculate_priority(self, env: JobShopEnvironment, action: Action) -> float:
        """Enhanced priority calculation with multiple factors."""
        # Get operation duration
        op_duration = env.jobs[action.job].operations[action.operation].duration
        
        # Calculate remaining processing time
        remaining_time = sum(
            op.duration
            for op in env.jobs[action.job].operations[action.operation:]
        )
        
        # Calculate machine load
        machine_time = env.current_state.machine_availability[action.machine]
        total_time = max(1, env.total_time)
        machine_load = machine_time / total_time
        
        # Calculate job progress
        progress = action.operation / len(env.jobs[action.job].operations)
        
        # Normalize each component to be between 0 and 1
        max_duration = max(op.duration for job in env.jobs for op in job.operations)
        norm_duration = op_duration / max_duration
        
        max_remaining = max(sum(op.duration for op in job.operations) for job in env.jobs)
        norm_remaining = remaining_time / max_remaining
        
        # Combine factors with weights - result will be between 0 and 1
        priority = (
            0.3 * norm_remaining +
            0.3 * (1 - machine_load) +
            0.2 * norm_duration +
            0.2 * (1 - progress)
        )
        
        return priority

    def _calculate_reward(self, env: JobShopEnvironment, action: Action, 
                         prev_time: int, prev_util: float) -> float:
        """Enhanced reward calculation considering multiple objectives."""
        # Time-based reward
        time_diff = env.total_time - prev_time
        time_reward = -time_diff / max(1, env.total_time)
        
        # Utilization improvement reward
        current_util = np.mean(env.get_machine_utilization())
        util_reward = (current_util - prev_util) * 100
        
        # Progress reward
        progress = env.current_state.next_operation_for_job[action.job] / len(env.jobs[action.job].operations)
        progress_reward = progress * 10
        
        # Load balancing reward
        utils = env.get_machine_utilization()
        balance_reward = -np.std(utils) * 50
        
        # Critical path reward
        critical_path = env.get_critical_path()
        critical_reward = 20 if (action.job, action.operation) in critical_path else 0
        
        # Combine rewards
        total_reward = (
            0.4 * time_reward +
            0.2 * util_reward +
            0.2 * progress_reward +
            0.1 * balance_reward +
            0.1 * critical_reward
        )
        
        return total_reward

    def _select_action(self, env: JobShopEnvironment) -> Action:
        """Enhanced action selection with sophisticated exploration."""
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            return None
            
        if self.rng.random() < self.exploration_rate:
            # Smart exploration using priorities
            priorities = []
            
            # Debug logging for raw priorities
            logger.debug("Calculating priorities for possible actions:")
            
            for action in possible_actions:
                # Calculate sophisticated priority score
                priority = self._calculate_priority(env, action)
                logger.debug(f"Raw priority for action {action}: {priority}")
                
                # Add Q-value influence
                q_value = float(self.q_table[action.job, action.machine, action.operation].mean())
                logger.debug(f"Q-value for action {action}: {q_value}")
                
                # Ensure priority is positive using softmax-style transformation
                try:
                    priority = np.exp(min(50, priority * (1 + 0.5 * q_value)))  # Clip to prevent overflow
                    logger.debug(f"Transformed priority: {priority}")
                except OverflowError:
                    priority = float('inf')
                    logger.warning(f"Priority overflow, setting to inf for action {action}")
                
                priorities.append(float(priority))  # Ensure float type
            
            # Debug logging for priorities array
            logger.debug(f"All priorities before normalization: {priorities}")
            
            # Handle the case where all priorities are infinite
            if all(np.isinf(p) for p in priorities):
                logger.warning("All priorities are infinite, using uniform distribution")
                return possible_actions[self.rng.choice(len(possible_actions))]
            
            # Normalize priorities with additional safety checks
            total_priority = sum(p for p in priorities if not np.isinf(p))
            if total_priority > 0:
                # Convert infinite values to max of finite values
                max_finite = max(p for p in priorities if not np.isinf(p))
                priorities = [max_finite if np.isinf(p) else p for p in priorities]
                
                # Calculate probabilities
                probs = [max(0.0, float(p/total_priority)) for p in priorities]
                probs_sum = sum(probs)
                
                logger.debug(f"Probabilities before final normalization: {probs}")
                logger.debug(f"Sum of probabilities: {probs_sum}")
                
                if probs_sum > 0:
                    # Renormalize to ensure sum is 1
                    probs = [p/probs_sum for p in probs]
                    logger.debug(f"Final normalized probabilities: {probs}")
                    
                    # Verify probabilities
                    if any(p < 0 for p in probs):
                        logger.error(f"Negative probability detected: {probs}")
                        return possible_actions[self.rng.choice(len(possible_actions))]
                        
                    if abs(sum(probs) - 1.0) > 1e-9:
                        logger.error(f"Probabilities don't sum to 1: {sum(probs)}")
                        return possible_actions[self.rng.choice(len(possible_actions))]
                    
                    try:
                        return possible_actions[self.rng.choice(len(possible_actions), p=probs)]
                    except ValueError as e:
                        logger.error(f"Error in numpy choice: {e}")
                        logger.error(f"Probabilities causing error: {probs}")
                        return possible_actions[self.rng.choice(len(possible_actions))]
                else:
                    logger.warning("Zero total probability, using uniform distribution")
                    return possible_actions[self.rng.choice(len(possible_actions))]
            else:
                logger.warning("Zero total priority, using uniform distribution")
                return possible_actions[self.rng.choice(len(possible_actions))]
        
        # Greedy selection with state features
        best_value = float('-inf')
        best_action = possible_actions[0]
        
        for action in possible_actions:
            features = self._get_state_features(env, action)
            q_values = self.q_table[action.job, action.machine, action.operation]
            value = float(np.dot(q_values, features))  # Ensure float type
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

    def _update_q_value(self, env: JobShopEnvironment, action: Action, 
                       prev_time: int, prev_util: float) -> None:
        """Enhanced Q-value update with experience replay."""
        # Calculate enhanced reward
        reward = self._calculate_reward(env, action, prev_time, prev_util)
        
        # Get state features
        features = self._get_state_features(env, action)
        
        # Store experience
        self.state_history.append((action.job, action.machine, action.operation))
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Get future value estimate
        possible_actions = env.get_possible_actions()
        max_future_q = 0.0
        if possible_actions:
            future_values = []
            for future_action in possible_actions:
                future_features = self._get_state_features(env, future_action)
                q_values = self.q_table[future_action.job, 
                                      future_action.machine, 
                                      future_action.operation]
                future_values.append(np.dot(q_values, future_features))
            max_future_q = max(future_values)
        
        # Update Q-values with features
        current_q = self.q_table[action.job, action.machine, action.operation]
        new_q = current_q + self.learning_rate * (
            reward + 
            self.discount_factor * max_future_q - 
            np.dot(current_q, features)
        )
        self.q_table[action.job, action.machine, action.operation] = new_q
        
        # Experience replay (replay last N experiences)
        if len(self.state_history) >= 10:
            self._experience_replay(env, 5)

    def _experience_replay(self, env: JobShopEnvironment, batch_size: int) -> None:
        """Implement experience replay for better learning."""
        if len(self.state_history) < batch_size:
            return
            
        # Sample random experiences
        indices = self.rng.choice(len(self.state_history), batch_size, replace=False)
        
        for idx in indices:
            state = self.state_history[idx]
            action = self.action_history[idx]
            reward = self.reward_history[idx]
            
            # Update Q-values for sampled experiences
            features = self._get_state_features(env, action)
            current_q = self.q_table[state[0], state[1], state[2]]
            new_q = current_q + self.learning_rate * 0.5 * (
                reward - np.dot(current_q, features)
            )
            self.q_table[state[0], state[1], state[2]] = new_q

    def _run_episode(self, env: JobShopEnvironment, max_steps: int = 1000) -> List[Action]:
        """Enhanced episode running with adaptive exploration."""
        env.reset()
        episode_actions = []
        
        # Clear experience history for new episode
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        while not env.is_done() and len(episode_actions) < max_steps:
            prev_time = env.total_time
            prev_util = np.mean(env.get_machine_utilization())
            
            action = self._select_action(env)
            if action is None:
                break
                
            env.step(action)
            episode_actions.append(action)
            
            self._update_q_value(env, action, prev_time, prev_util)
        
        return episode_actions

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """Enhanced solving process with adaptive parameters."""
        start_time = time.time()
        
        if self.q_table is None:
            self._initialize_q_table(env)
        
        logger.info(f"Starting improved Q-learning training for {self.episodes} episodes...")
        
        # Learning rate decay
        lr_decay = (0.01 / self.learning_rate) ** (1 / self.episodes)
        
        for episode in range(self.episodes):
            # Run episode
            episode_actions = self._run_episode(env, max_steps)
            
            # Evaluate episode
            env.reset()
            for action in episode_actions:
                env.step(action)
            
            # Update best solution
            if env.total_time < self.best_time:
                self.best_time = env.total_time
                self.best_schedule = episode_actions.copy()
                logger.info(f"New best makespan: {self.best_time}")
            
            # Adaptive parameter updates
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * 0.995
            )
            self.learning_rate *= lr_decay
            
            if (episode + 1) % 100 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}, "
                          f"Best makespan: {self.best_time}, "
                          f"Exploration rate: {self.exploration_rate:.3f}")
        
        # Final run with best actions
        env.reset()
        for action in self.best_schedule:
            env.step(action)
        
        solve_time = time.time() - start_time
        logger.info(f"Improved Q-learning solved in {solve_time:.2f} seconds")
        logger.info(f"Final makespan: {env.total_time}")
        
        return self.best_schedule, env.total_time