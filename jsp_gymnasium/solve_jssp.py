# solve_jssp.py

import argparse
import logging
from pathlib import Path
import numpy as np
import time
from typing import List, Dict, Tuple
import json

# Import our Gymnasium environment and components
from job_shop_env import JobShopGymEnv, Job, Operation
from Job_shop_taillard_generator import TaillardGymGenerator
from job_shop_manual_generator import ManualGymGenerator
from q_learning import QLearningAgent

class JSSPGymSolver:
    """Solver for Job Shop Scheduling using Gymnasium environment."""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 episodes: int = 1000,
                 max_steps: int = 1000):
        
        self.learning_params = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'episodes': episodes,
            'max_steps': max_steps
        }
        
        self.logger = logging.getLogger(__name__)
        self.env = None
        self.agent = None
        self.instance_type = None
        self.instance_name = None

    def initialize_environment(self, instance_type: str, instance_path: str):
        """Initialize environment based on instance type."""
        self.instance_type = instance_type
        self.instance_name = Path(instance_path).name
        
        if instance_type == 'taillard':
            self.logger.info(f"Initializing Taillard instance: {instance_path}")
            self.env = TaillardGymGenerator.create_env_from_instance(instance_path)
        
        elif instance_type == 'manual':
            self.logger.info(f"Initializing manual instance: {instance_path}")
            self.env = ManualGymGenerator.create_env_from_file(instance_path)
            
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")
            
        self._setup_agent()

    def _setup_agent(self):
        """Setup Q-learning agent."""
        if self.env is None:
            raise ValueError("Environment must be initialized before setting up agent")
            
        self.agent = QLearningAgent(
            env=self.env,
            learning_rate=self.learning_params['learning_rate'],
            discount_factor=self.learning_params['discount_factor'],
            epsilon_start=self.learning_params['epsilon_start'],
            epsilon_end=self.learning_params['epsilon_end'],
            epsilon_decay=self.learning_params['epsilon_decay'],
            episodes=self.learning_params['episodes']
        )

    def solve(self) -> Tuple[List[int], float, Dict]:
        """Train agent and return best solution with metrics."""
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent must be initialized before solving")
            
        start_time = time.time()
        self.logger.info(f"Starting Q-learning training for {self.instance_type} instance: {self.instance_name}")
        
        # Train agent
        best_actions, best_makespan = self.agent.train(
            max_steps=self.learning_params['max_steps']
        )
        
        training_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            'training_time': training_time,
            'best_makespan': best_makespan,
            'num_steps': len(best_actions),
            'instance_type': self.instance_type,
            'instance_name': self.instance_name
        }
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best makespan: {best_makespan}")
        
        return best_actions, best_makespan, metrics
    
    def visualize_solution(self, actions: List[int]):
        """Visualize the solution."""
        if self.env is None:
            raise ValueError("Environment must be initialized before visualization")
            
        print(f"\nVisualizing solution for {self.instance_type} instance: {self.instance_name}")
        self.env.reset()
        
        for step, action in enumerate(actions, 1):
            decoded = self.env._decode_action(action)
            print(f"\nStep {step}:")
            print(f"Job: {decoded.job}, Operation: {decoded.operation}, "
                  f"Machine: {decoded.machine}")
            
            obs, reward, done, truncated, info = self.env.step(action)
            self.env.render()
            
            if done:
                break
                
        print(f"\nFinal makespan: {self.env.total_time}")

def list_available_instances():
    """List all available instances (both Taillard and manual)."""
    print("\nAvailable instances:")
    
    # List Taillard instances
    taillard_instances = TaillardGymGenerator.list_available_instances()
    if taillard_instances:
        print("\nTaillard instances:")
        for instance in taillard_instances:
            print(f"  {instance}")
    else:
        print("No Taillard instances found")
    
    # List manual instances
    project_root = Path(__file__).parent.parent
    manual_dir = project_root / "per_jsp" / "data" / "problem_instances"
    
    if manual_dir.exists():
        manual_instances = list(manual_dir.glob("*.json"))
        if manual_instances:
            print("\nManual instances:")
            for instance in manual_instances:
                print(f"  {instance.name}")
        else:
            print("No manual instances found")
    else:
        print("Manual instances directory not found")

def main():
    parser = argparse.ArgumentParser(description='Solve JSSP using Gymnasium and Q-Learning')
    
    parser.add_argument('--instance_type', choices=['taillard', 'manual'], required=True,
                       help='Instance type (taillard or manual)')
    parser.add_argument('--instance', type=str, required=True,
                       help='Instance name (e.g., ta01.txt or problem1.json)')
    
    # Learning parameters
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.995)
    parser.add_argument('--max_steps', type=int, default=1000)
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for solution')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--list', action='store_true',
                       help='List available instances')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.list:
        list_available_instances()
        return
    
    try:
        # Create solver
        solver = JSSPGymSolver(
            learning_rate=args.lr,
            discount_factor=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay=args.eps_decay,
            episodes=args.episodes,
            max_steps=args.max_steps
        )
        
        # Initialize environment
        solver.initialize_environment(args.instance_type, args.instance)
        
        # Solve problem
        print("\nStarting optimization...")
        best_actions, best_makespan, metrics = solver.solve()
        
        # Visualize solution
        solver.visualize_solution(best_actions)
        
        # Save results
        if args.output:
            results = {
                'instance_type': args.instance_type,
                'instance_name': args.instance,
                'parameters': solver.learning_params,
                'metrics': {k: float(v) if isinstance(v, np.number) else v 
                          for k, v in metrics.items()},
                'solution': {
                    'makespan': float(best_makespan),
                    'actions': [int(a) for a in best_actions]
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()