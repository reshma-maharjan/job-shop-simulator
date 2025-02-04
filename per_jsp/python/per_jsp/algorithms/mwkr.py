from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MWKRScheduler(BaseScheduler):
    """
    Most Work Remaining (MWKR) scheduler implementation.
    Selects jobs based on the highest remaining processing time for prioritization.
    """
    
    def __init__(self):
        """Initialize the MWKR scheduler."""
        super().__init__()
        
    def _calculate_remaining_times(self, env: JobShopEnvironment, possible_actions: list[Action]) -> np.ndarray:
        remaining_times = np.zeros(len(possible_actions))
        
        for i, action in enumerate(possible_actions):
            job = env.jobs[action.job]
            # Calculate remaining time for all operations from current operation onwards
            remaining_times[i] = sum(op.duration for op in job.operations[action.operation:])
        
    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> tuple[list[Action], int]:
        logger.info("Starting MWKR scheduling...")
        env.reset()
        actions = []
        
        while not env.is_done() and len(actions) < max_steps:
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break
                
            # Calculate and log remaining times for debugging
            remaining_times = []
            for action in possible_actions:
                job = env.jobs[action.job]
                remaining_time = sum(op.duration for op in job.operations[action.operation:])
                remaining_times.append((action, remaining_time))
                logger.debug(f"Job {action.job} Op {action.operation}: Remaining time = {remaining_time}")
            
            # Sort by remaining time and select highest
            remaining_times.sort(key=lambda x: x[1], reverse=True)
            selected_action = remaining_times[0][0]
            
            logger.debug(f"Selected Job {selected_action.job} with remaining time {remaining_times[0][1]}")
            
            env.step(selected_action)
            actions.append(selected_action)

        logger.info(f"MWKR scheduling completed with {len(actions)} actions")
        makespan = env.total_time
        return actions, makespan