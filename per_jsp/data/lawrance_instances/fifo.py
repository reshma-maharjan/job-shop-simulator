from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FIFOScheduler(BaseScheduler):
    """
    FIFO (First In First Out) scheduler implementation.
    Selects jobs based on remaining processing time.
    """
    
    def __init__(self):
        """Initialize the FIFO scheduler."""
        super().__init__()
        
    def _calculate_remaining_time(self, env: JobShopEnvironment, action: Action) -> float:
        """
        Calculate remaining processing time for a job given an action.
        """
        job = env.jobs[action.job]
        
        # Calculate remaining time for operations from current action onwards
        remaining_time = sum(
            op.duration 
            for op in job.operations[action.operation:]
        )
        return remaining_time
        
    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> tuple[list[Action], int]:
        """
        Solve the scheduling problem using FIFO strategy.
        
        Args:
            env: JobShopEnvironment instance
            max_steps: Maximum number of steps
            
        Returns:
            Tuple of (actions taken, makespan)
        """
        logger.info("Starting FIFO scheduling...")
        env.reset()
        actions = []
        
        while not env.is_done() and len(actions) < max_steps:
            # Get possible actions
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break
                
            # Calculate remaining time for each possible action
            remaining_times = []
            
            for action in possible_actions:
                remaining_time = self._calculate_remaining_time(env, action)
                remaining_times.append((action, remaining_time))
            
            if not remaining_times:
                break
                
            # Sort by remaining time (highest first)
            remaining_times.sort(key=lambda x: x[1], reverse=True)
            action = remaining_times[0][0]
            
            # Take the action
            env.step(action)
            actions.append(action)
            
            if (len(actions) % 10) == 0:
                logger.info(f"Completed {len(actions)} actions")
        
        logger.info(f"FIFO scheduling completed with {len(actions)} actions")
        makespan = env.total_time
        return actions, makespan