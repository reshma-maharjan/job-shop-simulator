import numpy as np
import time
import logging
from typing import List, Tuple, Dict
from per_jsp.algorithms.base import BaseScheduler
from per_jsp.environment.job_shop_environment import JobShopEnvironment, Action



logger = logging.getLogger(__name__)

class QLearningScheduler(BaseScheduler):
    """Q-learning algorithm for job shop scheduling."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 episodes: int = 100):
        """
        Initialize Q-learning scheduler.

        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Probability of random exploration
            episodes: Number of training episodes
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table: Dict[int, Dict[Tuple[int, int, int], float]] = {}

    def _get_state_key(self, env: JobShopEnvironment) -> int:
        """Create a unique key for the current state."""
        state_tuple = (
            tuple(env.current_state.machine_availability),
            tuple(env.current_state.next_operation_for_job),
            tuple(env.current_state.completed_jobs)
        )
        return hash(state_tuple)

    def _get_q_value(self, state_key: int, action: Action) -> float:
        """Get Q-value for state-action pair."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        action_key = (action.job, action.machine, action.operation)
        return self.q_table[state_key].get(action_key, 0.0)

    def _update_q_value(self, state_key: int, action: Action, value: float):
        """Update Q-value for state-action pair."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        action_key = (action.job, action.machine, action.operation)
        self.q_table[state_key][action_key] = value

    def _calculate_reward(self, env: JobShopEnvironment, action: Action) -> float:
        """Calculate reward for taking an action."""
        job = env.jobs[action.job]
        operation = job.operations[action.operation]

        # Combine multiple factors for reward
        completion_bonus = 1.0 if env.current_state.completed_jobs[action.job] else 0.0
        duration_penalty = -operation.duration / 100.0  # Penalize longer operations
        machine_utilization = -env.current_state.machine_availability[action.machine] / 1000.0

        return completion_bonus + duration_penalty + machine_utilization

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """Solve using Q-learning algorithm."""
        start_time = time.time()
        best_actions = []
        best_makespan = float('inf')

        # Training phase
        logger.info(f"Starting Q-learning training for {self.episodes} episodes...")

        for episode in range(self.episodes):
            env.reset()
            episode_actions = []
            step_count = 0

            while not env.is_done() and step_count < max_steps:
                state_key = self._get_state_key(env)
                possible_actions = env.get_possible_actions()

                if not possible_actions:
                    break

                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.choice(possible_actions)
                else:
                    action = max(
                        possible_actions,
                        key=lambda a: self._get_q_value(state_key, a)
                    )

                # Take action and observe result
                env.step(action)
                episode_actions.append(action)
                reward = self._calculate_reward(env, action)

                # Update Q-value
                new_state_key = self._get_state_key(env)
                new_possible_actions = env.get_possible_actions()

                if new_possible_actions:
                    max_future_q = max(
                        self._get_q_value(new_state_key, a)
                        for a in new_possible_actions
                    )
                else:
                    max_future_q = 0

                old_q = self._get_q_value(state_key, action)
                new_q = (1 - self.learning_rate) * old_q + self.learning_rate * (
                        reward + self.discount_factor * max_future_q
                )
                self._update_q_value(state_key, action, new_q)

                step_count += 1

            # Check if this episode found a better solution
            if env.total_time < best_makespan:
                best_makespan = env.total_time
                best_actions = episode_actions.copy()

            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}, "
                            f"Best makespan: {best_makespan}")

        # Final run with best policy
        env.reset()
        for action in best_actions:
            env.step(action)

        solve_time = time.time() - start_time
        logger.info(f"Q-learning solved in {solve_time:.2f} seconds")
        logger.info(f"Final makespan: {env.total_time}")

        return best_actions, env.total_time