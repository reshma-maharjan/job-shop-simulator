import numpy as np
import time
import logging
import random
import wandb
from typing import List, Tuple, Callable
from dataclasses import dataclass
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action
import argparse
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator

logger = logging.getLogger(__name__)

class QLearningScheduler(BaseScheduler):
    """Q-learning scheduler with wandb logging."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 episodes: int = 2000):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        
        # Exploration parameters
        self.min_exploration_rate = 0.05
        self.exploration_decay = 0.997
        
        self.q_table = None
        self.best_time = float('inf')
        self.best_schedule = []
        self.rng = np.random.RandomState()

        # Initialize wandb with configuration
        wandb.init(
            project="job-shop-scheduling",
            config={
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "initial_exploration_rate": exploration_rate,
                "episodes": episodes,
                "min_exploration_rate": self.min_exploration_rate,
                "exploration_decay": self.exploration_decay,
                "algorithm": "q_learning"
            }
        )

    def _initialize_q_table(self, env: JobShopEnvironment) -> None:
        """Initialize Q-table with proper dimensions."""
        max_operations = max(len(job.operations) for job in env.jobs)
        self.q_table = np.zeros((
            len(env.jobs),          
            env.num_machines,       
            max_operations         
        ))

    def _calculate_priority(self, env: JobShopEnvironment, action: Action) -> float:
        """Calculate priority score for an action."""
        job = env.jobs[action.job]
        current_op = job.operations[action.operation]
        
        # Calculate immediate operation priority
        duration_priority = 1.0 / max(1, current_op.duration)
        
        # Machine availability priority
        machine_time = env.current_state.machine_availability[action.machine]
        machine_priority = 1.0 / max(1, machine_time)
        
        # Remaining work priority
        remaining_time = sum(op.duration for op in job.operations[action.operation:])
        remaining_priority = 1.0 / max(1, remaining_time)
        
        return duration_priority * 0.4 + machine_priority * 0.4 + remaining_priority * 0.2

    def _select_action(self, env: JobShopEnvironment) -> Action:
        """Select action using epsilon-greedy strategy."""
        possible_actions = env.get_possible_actions()

        if not possible_actions:
            return None

        if self.rng.random() < self.exploration_rate:
            # Priority-based exploration
            priorities = [self._calculate_priority(env, action) for action in possible_actions]
            total_priority = sum(priorities)
            if total_priority > 0:
                probs = [p/total_priority for p in priorities]
                return possible_actions[self.rng.choice(len(possible_actions), p=probs)]
            return self.rng.choice(possible_actions)

        # Q-value based exploitation
        return max(possible_actions, 
                  key=lambda a: self.q_table[a.job, a.machine, a.operation])

    def _update_q_value(self, env: JobShopEnvironment, action: Action, prev_time: int) -> None:
        """Update Q-value with focus on makespan reduction."""
        # Calculate time-based reward
        time_diff = env.total_time - prev_time
        reward = -time_diff  # Linear penalty
        
        # Penalty for exceeding best time
        if env.total_time > self.best_time:
            reward -= (env.total_time - self.best_time) * 0.5
        
        # Update Q-value
        possible_actions = env.get_possible_actions()
        max_future_q = max((self.q_table[a.job, a.machine, a.operation] 
                          for a in possible_actions), default=0.0)

        current_q = self.q_table[action.job, action.machine, action.operation]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q
        )
        self.q_table[action.job, action.machine, action.operation] = new_q

    def _run_episode(self, env: JobShopEnvironment, max_steps: int = 1000) -> List[Action]:
        """Run a single episode."""
        env.reset()
        episode_actions = []
        episode_rewards = []

        while not env.is_done() and len(episode_actions) < max_steps:
            prev_time = env.total_time
            action = self._select_action(env)
            
            if action is None:
                break

            env.step(action)
            episode_actions.append(action)
            
            # Track rewards for this episode
            time_diff = env.total_time - prev_time
            episode_rewards.append(-time_diff)
            
            self._update_q_value(env, action, prev_time)

        return episode_actions, np.mean(episode_rewards) if episode_rewards else 0

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """Solve using Q-learning with wandb logging."""
        start_time = time.time()

        # Initialize Q-table
        if self.q_table is None:
            self._initialize_q_table(env)

        logger.info(f"Starting Q-learning training for {self.episodes} episodes...")
        
        consecutive_no_improvement = 0
        episode_times = []

        for episode in range(self.episodes):
            episode_start = time.time()
            
            # Run episode and get actions and average reward
            episode_actions, avg_reward = self._run_episode(env, max_steps)

            # Evaluate episode
            env.reset()
            for action in episode_actions:
                env.step(action)

            current_makespan = env.total_time
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)

            # Track best solution
            if current_makespan < self.best_time:
                self.best_time = current_makespan
                self.best_schedule = episode_actions.copy()
                consecutive_no_improvement = 0
                logger.info(f"New best makespan: {self.best_time}")
            else:
                consecutive_no_improvement += 1

            # Adaptive exploration decay
            if consecutive_no_improvement > 50:
                self.exploration_rate = min(1.0, self.exploration_rate * 1.5)
                consecutive_no_improvement = 0
            else:
                self.exploration_rate = max(
                    self.min_exploration_rate,
                    self.exploration_rate * self.exploration_decay
                )

            # Calculate additional metrics
            machine_utilization = np.mean(env.get_machine_utilization()) * 100
            makespan_improvement = ((self.best_time - current_makespan) / self.best_time 
                                  if self.best_time != float('inf') else 0)

            # Log metrics to wandb
            wandb.log({
                "episode": episode,
                "current_makespan": current_makespan,
                "best_makespan": self.best_time,
                "exploration_rate": self.exploration_rate,
                "episode_time": episode_time,
                "average_episode_time": np.mean(episode_times),
                "machine_utilization": machine_utilization,
                "makespan_improvement": makespan_improvement,
                "average_reward": avg_reward,
                "consecutive_no_improvement": consecutive_no_improvement,
                "q_value_mean": np.mean(self.q_table),
                "q_value_std": np.std(self.q_table)
            })

            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}, "
                          f"Best makespan: {self.best_time}, "
                          f"Current exploration rate: {self.exploration_rate:.3f}")

        # Final evaluation
        env.reset()
        for action in self.best_schedule:
            env.step(action)

        solve_time = time.time() - start_time
        
        # Log final metrics
        wandb.log({
            "final_makespan": env.total_time,
            "total_solve_time": solve_time,
            "best_makespan_overall": self.best_time,
            "average_episode_time": np.mean(episode_times),
            "total_episodes": self.episodes,
            "final_machine_utilization": np.mean(env.get_machine_utilization()) * 100
        })

        # Create makespan vs episodes plot
        wandb.log({"makespan_history": wandb.plot.line_series(
            xs=range(self.episodes),
            ys=[self.best_time],
            keys=["Best Makespan"],
            title="Makespan vs Episodes",
            xname="Episode"
        )})

        logger.info(f"Q-learning solved in {solve_time:.2f} seconds")
        logger.info(f"Final makespan: {env.total_time}")

        return self.best_schedule, env.total_time