import numpy as np
import random
from typing import List, Tuple, Callable
import logging
from dataclasses import dataclass
import time

from per_jsp.algorithms.base import BaseScheduler
from per_jsp.environment.job_shop_environment import JobShopEnvironment, Action

logger = logging.getLogger(__name__)

class QLearningScheduler(BaseScheduler):
    """Direct Python implementation of the C++ Q-learning scheduler."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 episodes: int = 1000):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes

        self.q_table = None
        self.best_time = float('inf')
        self.best_schedule = []
        self.rng = np.random.RandomState()

    def _initialize_q_table(self, env: JobShopEnvironment) -> None:
        """Initialize Q-table with proper dimensions."""
        max_operations = max(len(job.operations) for job in env.jobs)
        self.q_table = np.zeros((
            len(env.jobs),          # Number of jobs
            env.num_machines,       # Number of machines
            max_operations         # Max operations per job
        ))

    def _calculate_priority(self, env: JobShopEnvironment, action: Action) -> float:
        """Calculate priority score for an action."""
        # Calculate remaining processing time for the job
        remaining_time = sum(
            op.duration
            for op in env.jobs[action.job].operations[action.operation:]
        )

        # Calculate machine utilization
        machine_time = env.current_state.machine_availability[action.machine]
        total_time = max(1, env.total_time)
        machine_utilization = machine_time / total_time

        # Priority combines remaining time and machine availability
        return remaining_time * (1 - machine_utilization)

    def _select_action(self, env: JobShopEnvironment) -> Action:
        """Select action using epsilon-greedy strategy with priority-based exploration."""
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

        # Greedy selection based on Q-values
        best_q = float('-inf')
        best_action = possible_actions[0]

        for action in possible_actions:
            q_value = self.q_table[action.job, action.machine, action.operation]
            if q_value > best_q:
                best_q = q_value
                best_action = action

        return best_action

    def _update_q_value(self, env: JobShopEnvironment, action: Action, prev_time: int) -> None:
        """Update Q-value for the given action."""
        # Calculate time-based reward
        time_reward = -(env.total_time - prev_time)

        # Calculate utilization-based reward
        utils = env.get_machine_utilization()
        util_reward = np.mean(utils) * 100

        # Combined reward
        reward = time_reward + util_reward

        # Get maximum future Q-value
        possible_actions = env.get_possible_actions()
        max_future_q = 0.0
        if possible_actions:
            max_future_q = max(
                self.q_table[a.job, a.machine, a.operation]
                for a in possible_actions
            )

        # Update Q-value
        current_q = self.q_table[action.job, action.machine, action.operation]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q
        )
        self.q_table[action.job, action.machine, action.operation] = new_q

    def _run_episode(self, env: JobShopEnvironment, max_steps: int = 1000) -> List[Action]:
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

            self._update_q_value(env, action, prev_time)

        return episode_actions

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """Solve using Q-learning."""
        start_time = time.time()

        # Initialize Q-table
        if self.q_table is None:
            self._initialize_q_table(env)

        logger.info(f"Starting Q-learning training for {self.episodes} episodes...")

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
        logger.info(f"Q-learning solved in {solve_time:.2f} seconds")
        logger.info(f"Final makespan: {env.total_time}")

        return self.best_schedule, env.total_time