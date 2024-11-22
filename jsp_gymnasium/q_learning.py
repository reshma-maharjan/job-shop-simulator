import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict
import time
import logging
import argparse
import json  # Add this import
from pathlib import Path  # Add this import
from collections import defaultdict

logger = logging.getLogger(__name__)

class QLearningAgent:
    """Q-learning agent for Gymnasium Job Shop Scheduling environment."""
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        episodes: int = 1000
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.best_makespan = float('inf')
        self.best_actions = []
        
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions in current state."""
        valid_actions = []
        for action_id in range(self.env.action_space.n):
            action = self.env._decode_action(action_id)
            if self.env._is_valid_action(action):
                valid_actions.append(action_id)
        return valid_actions
    
    def _get_state_key(self, observation: np.ndarray) -> int:
        """Create a unique key for the current state."""
        # Discretize machine availability times to reduce state space
        machine_availability = tuple(
            int(t/10) for t in self.env.current_state.machine_availability
        )
        next_operations = tuple(self.env.current_state.next_operation_for_job)
        completed_jobs = tuple(self.env.current_state.completed_jobs)
        
        return hash((machine_availability, next_operations, completed_jobs))
    
    def _calculate_reward(self, info: dict, done: bool) -> float:
        """Calculate reward based on environment feedback."""
        reward = 0.0
    
        # Penalty for invalid actions
        if not info.get('valid_action', False):
            return -100.0
    
        # Reward for completing jobs (increased)
        completed_jobs = info.get('completed_jobs', 0)
        reward += completed_jobs * 500.0  # Increased from 200 to 500
    
        # Reward for makespan improvement (increased)
        current_makespan = info.get('makespan', float('inf'))
        if current_makespan < self.best_makespan:
            improvement = self.best_makespan - current_makespan
            reward += improvement * 20.0  # Increased from 10 to 20
    
        # Penalty based on current makespan (adjusted)
        reward -= current_makespan * 0.05  # Reduced from 0.1 to 0.05
    
        # Bonus for completing the schedule (increased)
        if done:
         reward += 2000.0  # Increased from 1000 to 2000
    
        return reward

    def train(self, max_steps: int = 1000) -> Tuple[List[int], float]:
        """Train the Q-learning agent."""
        start_time = time.time()
        episodes_without_improvement = 0
        
        logger.info(f"Starting Q-learning training for {self.episodes} episodes...")
        logger.info(f"Problem size: {len(self.env.jobs)} jobs, {self.env.num_machines} machines")
        
        for episode in range(self.episodes):
            observation, _ = self.env.reset()
            state_key = self._get_state_key(observation)
            episode_actions = []
            
            for step in range(max_steps):
                # Get valid actions
                valid_actions = self._get_valid_actions()
                if not valid_actions:
                    break
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.choice(valid_actions)
                else:
                    action = max(
                        valid_actions,
                        key=lambda a: self.q_table[state_key][a]
                    )
                
                # Take action
                next_observation, reward, done, truncated, info = self.env.step(action)
                episode_actions.append(action)
                
                # Calculate reward
                custom_reward = self._calculate_reward(info, done)
                
                # Update Q-value
                next_state_key = self._get_state_key(next_observation)
                next_valid_actions = self._get_valid_actions()
                
                if next_valid_actions:
                    max_future_q = max(
                        self.q_table[next_state_key][a]
                        for a in next_valid_actions
                    )
                else:
                    max_future_q = 0
                
                current_q = self.q_table[state_key][action]
                new_q = (1 - self.learning_rate) * current_q + \
                        self.learning_rate * (custom_reward + self.discount_factor * max_future_q)
                self.q_table[state_key][action] = new_q
                
                state_key = next_state_key
                
                if done or truncated:
                    # Update best solution if episode completed successfully
                    current_makespan = info.get('makespan', float('inf'))
                    if done and current_makespan < self.best_makespan:
                        self.best_makespan = current_makespan
                        self.best_actions = episode_actions.copy()
                        episodes_without_improvement = 0
                        logger.info(f"New best makespan found: {self.best_makespan}")
                    else:
                        episodes_without_improvement += 1
                    break
            
            # Decay epsilon
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
            
            # Logging
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{self.episodes}, "
                    f"Best makespan: {self.best_makespan}, "
                    f"Epsilon: {self.epsilon:.3f}, "
                    f"Steps: {len(episode_actions)}"
                )
            
            # Early stopping
            if episodes_without_improvement >= 50:  # Adjust as needed
                logger.info(f"Early stopping after {episode + 1} episodes - No improvement for 50 episodes")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best makespan: {self.best_makespan}")
        
        return self.best_actions, self.best_makespan

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def setup_result_directory(instance_name: str) -> Path:
    """Setup result directory and generate result file path."""
    # Get current directory (jsp_gymnasium)
    current_dir = Path(__file__).parent
    
    # Create results directory if it doesn't exist
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create instance-specific directory
    instance_base = Path(instance_name).stem  # Gets filename without extension
    instance_dir = results_dir / instance_base
    instance_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create filename: instance_timestamp.json
    result_file = instance_dir / f"{instance_base}_{timestamp}.json"
    
    return result_file

# Then update the main function:

def main():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling with Q-Learning (Gymnasium)')
    
    # Add instance name argument
    parser.add_argument('--instance', type=str, required=True,
                      help='Instance name (e.g., ta01.txt)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                      help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0,
                      help='Starting epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.01,
                      help='Final epsilon for exploration')
    parser.add_argument('--eps_decay', type=float, default=0.995,
                      help='Epsilon decay rate')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum steps per episode')
    parser.add_argument('--verbose', action='store_true',
                      help='Show detailed output')

    args = parser.parse_args()
 

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Setup result file path
        result_file = setup_result_directory(args.instance)
        
        # Create environment based on instance type
        if args.instance.startswith('ta'):
            from Job_shop_taillard_generator import TaillardGymGenerator
            env = TaillardGymGenerator.create_env_from_instance(args.instance)
        else:
            from manual_generator import ManualGymGenerator
            env = ManualGymGenerator.create_env_from_file(args.instance)
        
        print("\nProblem Configuration:")
        print(f"Instance: {args.instance}")
        print(f"Jobs: {len(env.jobs)}")
        print(f"Machines: {env.num_machines}")
        print(f"Learning rate: {args.lr}")
        print(f"Discount factor: {args.gamma}")
        print(f"Episodes: {args.episodes}")
        print(f"Max steps per episode: {args.max_steps}")
        
        # Create and train agent
        agent = QLearningAgent(
            env=env,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay=args.eps_decay,
            episodes=args.episodes
        )
        
        print("\nStarting training...")
        best_actions, best_makespan = agent.train(max_steps=args.max_steps)
        
        # Execute best solution and collect data
        env.reset()
        schedule_data = []
        
        for action_id in best_actions:
            action = env._decode_action(action_id)
            obs, reward, done, truncated, info = env.step(action_id)
            
            schedule_data.append({
                'job': int(action.job),
                'machine': int(action.machine),
                'operation': int(action.operation),
                'start': int(info.get('start_time', 0)),
                'duration': int(env.jobs[action.job].operations[action.operation].duration),
                'end': int(info.get('end_time', 0))
            })
        
        # Save results
        results = {
            'instance': args.instance,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'episodes': int(args.episodes),
                'learning_rate': float(args.lr),
                'discount_factor': float(args.gamma),
                'epsilon_start': float(args.eps_start),
                'epsilon_end': float(args.eps_end),
                'epsilon_decay': float(args.eps_decay),
                'max_steps': int(args.max_steps)
            },
            'problem': {
                'num_jobs': len(env.jobs),
                'num_machines': env.num_machines
            },
            'solution': {
                'makespan': float(best_makespan),
                'num_actions': len(best_actions),
                'schedule': schedule_data,
                'actions': [int(a) for a in best_actions]
            },
            'metrics': {
                'machine_utilization': [float(u) for u in env.get_machine_utilization()]
            }
        }
        
        try:
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            print(f"\nResults saved to: {result_file}")
            
            # Also create a "latest" symlink
            latest_link = result_file.parent / f"{Path(args.instance).stem}_latest.json"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(result_file.name)
            
        except Exception as e:
            print(f"Error saving results: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        # Display final schedule
        print("\nFinal schedule:")
        env.render()
        print(f"\nMakespan: {best_makespan}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()