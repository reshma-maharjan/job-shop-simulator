# per_jsp/python/per_jsp/algorithms/spt.py

from typing import List, Tuple, Optional
import logging
from .base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import Action

logger = logging.getLogger(__name__)

class SPTScheduler(BaseScheduler):
    """
    Shortest Processing Time (SPT) scheduler for job shop scheduling.
    Prioritizes operations with shorter processing times.
    """
    def __init__(self):
        super().__init__()
        self.name = "SPT"

    def solve(self, env, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """
        Solve the job shop scheduling problem using SPT rule.
        
        Args:
            env: JobShopEnvironment instance
            max_steps: Maximum number of steps to take
            
        Returns:
            Tuple of (action_list, makespan)
        """
        env.reset()
        logger.info(f"Starting SPT solution for {len(env.jobs)} jobs on {env.num_machines} machines")
        
        actions = []
        steps = 0
        
        while not env.is_done() and steps < max_steps:
            action = self._select_next_action(env)
            if action is None:
                break
                
            env.step(action)
            actions.append(action)
            steps += 1
        
        return actions, env.total_time

    def _select_next_action(self, env) -> Optional[Action]:
        """
        Select next action using SPT rule.
        
        Args:
            env: JobShopEnvironment instance
            
        Returns:
            Selected action or None if no action is possible
        """
        possible_actions = env.get_possible_actions()
        if not possible_actions:
            return None
            
        # Calculate processing time for each possible action
        action_times = []
        for action in possible_actions:
            processing_time = env.jobs[action.job].operations[action.operation].duration
            action_times.append((action, processing_time))
            
        # Sort by processing time and select shortest
        return min(action_times, key=lambda x: x[1])[0]