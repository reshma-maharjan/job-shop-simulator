import logging
import time
from typing import List, Tuple
from per_jsp.environment.job_shop_environment import JobShopEnvironment, Action
from per_jsp.algorithms.base import BaseScheduler

logger = logging.getLogger(__name__)

class GreedyScheduler(BaseScheduler):
    """Greedy algorithm for job shop scheduling."""

    def __init__(self, use_longest=False):
        """
        Initialize greedy scheduler.

        Args:
            use_longest: If True, prioritize longest operations, otherwise shortest
        """
        self.use_longest = use_longest

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """
        Solve using greedy heuristic - choosing shortest/longest duration operations.
        """
        start_time = time.time()
        actions_taken = []
        step_count = 0

        while not env.is_done() and step_count < max_steps:
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break

            # Choose action based on duration
            chosen_action = max(
                possible_actions,
                key=lambda a: env.jobs[a.job].operations[a.operation].duration
            ) if self.use_longest else min(
                possible_actions,
                key=lambda a: env.jobs[a.job].operations[a.operation].duration
            )

            env.step(chosen_action)
            actions_taken.append(chosen_action)
            step_count += 1

        solve_time = time.time() - start_time
        logger.info(f"Greedy solved in {solve_time:.2f} seconds, {step_count} steps")
        logger.info(f"Makespan: {env.total_time}")

        return actions_taken, env.total_time