# greedy_solver.py

import time
import logging
from typing import List, Tuple, Optional
import numpy as np
from job_shop_env import JobShopGymEnv, Action

logger = logging.getLogger(__name__)

class GreedyScheduler:
    """Greedy algorithm for job shop scheduling using Gymnasium environment."""
    
    def __init__(self, use_longest: bool = False):
        """
        Initialize greedy scheduler.
        
        Args:
            use_longest: If True, prioritize longest operations, otherwise shortest
        """
        self.use_longest = use_longest
        
    def solve(self, env: JobShopGymEnv, max_steps: int = 1000) -> Tuple[List[int], float]:
        """
        Solve using greedy heuristic - choosing shortest/longest duration operations.
        
        Args:
            env: Gymnasium environment
            max_steps: Maximum steps to take
            
        Returns:
            List of actions (integers) and final makespan
        """
        start_time = time.time()
        actions_taken = []
        step_count = 0
        
        # Reset environment
        env.reset()
        
        while step_count < max_steps:
            # Get valid actions in Gymnasium format
            valid_actions = []
            for a in range(env.action_space.n):
                action = env._decode_action(a)
                if action.job < len(env.jobs) and \
                   action.operation < len(env.jobs[action.job].operations) and \
                   env._is_valid_action(action):
                    valid_actions.append(a)
            
            if not valid_actions or env.is_done():
                break
                
            # Choose action based on duration
            if self.use_longest:
                chosen_action = max(
                    valid_actions,
                    key=lambda a: self._get_operation_duration(env, env._decode_action(a))
                )
            else:
                chosen_action = min(
                    valid_actions,
                    key=lambda a: self._get_operation_duration(env, env._decode_action(a))
                )
            
            # Execute action
            obs, reward, done, truncated, info = env.step(chosen_action)
            actions_taken.append(chosen_action)
            step_count += 1
            
            if done:
                break
                
        solve_time = time.time() - start_time
        makespan = env.total_time
        
        logger.info(f"Greedy ({'longest' if self.use_longest else 'shortest'}) "
                   f"solved in {solve_time:.2f} seconds")
        logger.info(f"Steps taken: {step_count}")
        logger.info(f"Makespan: {makespan}")
        
        return actions_taken, makespan
    
    def _get_operation_duration(self, env: JobShopGymEnv, action: Action) -> int:
        """Get the duration of an operation."""
        if action.job >= len(env.jobs) or \
           action.operation >= len(env.jobs[action.job].operations):
            return float('inf') if not self.use_longest else float('-inf')
        return env.jobs[action.job].operations[action.operation].duration

def main():
    # Example usage
    from Job_shop_taillard_generator import TaillardGymGenerator
    from manual_generator import ManualGymGenerator
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve JSSP using Greedy Algorithm')
    parser.add_argument('--instance_type', choices=['taillard', 'manual'], required=True,
                       help='Instance type (taillard or manual)')
    parser.add_argument('--instance', type=str, required=True,
                       help='Instance name')
    parser.add_argument('--use_longest', action='store_true',
                       help='Use longest operation first (default: shortest first)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps to take')
    parser.add_argument('--output', type=str,
                       help='Output file for solution')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create environment
        if args.instance_type == 'taillard':
            env = TaillardGymGenerator.create_env_from_instance(args.instance)
        else:
            env = ManualGymGenerator.create_env_from_file(args.instance)
            
        # Create and run greedy solver
        solver = GreedyScheduler(use_longest=args.use_longest)
        
        print(f"\nSolving {args.instance_type} instance: {args.instance}")
        print(f"Strategy: {'Longest' if args.use_longest else 'Shortest'} operation first")
        
        actions, makespan = solver.solve(env, max_steps=args.max_steps)
        
        print("\nBest solution found:")
        print(f"Makespan: {makespan}")
        
        print("\nFinal schedule:")
        env.reset()
        for action in actions:
            obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        # Save results if requested
        if args.output:
            import json
            results = {
                'instance_type': args.instance_type,
                'instance': args.instance,
                'strategy': 'longest' if args.use_longest else 'shortest',
                'makespan': float(makespan),
                'num_steps': len(actions),
                'actions': [int(a) for a in actions]
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()