import numpy as np
import per_jspp as jobshop
import wandb
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List
from pathlib import Path


@dataclass
class QConfig:
    """Configuration for Q-learning"""
    num_episodes: int = 5000
    project: str = "jobshop_10x10"
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.999
    reward_scaling: float = 0.01
    problem_type: str = "json"  # 'manual', 'taillard', or 'json'
    problem_instance: str = "TA01"
    problem_path: str = None
    optimal_makespan: float = None

    def __post_init__(self):
        problem_identifier = self.problem_instance if self.problem_type == 'taillard' else Path(
            self.problem_path).stem if self.problem_path else 'manual'
        self.run_name = f"qlearn_lr{self.learning_rate}_eps{self.epsilon_decay}_{self.problem_type}_{problem_identifier}"


def load_problem(config: QConfig) -> Tuple[jobshop.JobShopEnvironment, float]:
    """
    Load problem instance based on configuration
    Returns: (environment, optimal_makespan)
    """
    if config.problem_type == "taillard":
        jobs, optimal_makespan = jobshop.TaillardJobShopGenerator.loadProblem("/home/per/jsp/jsp/environments/taillard/ta01.txt")
        return jobshop.JobShopEnvironment(jobs), optimal_makespan

    elif config.problem_type == "json":
        if not config.problem_path:
            raise ValueError("problem_path must be specified for JSON problems")
        jobs, optimal_makespan = jobshop.ManualJobShopGenerator.generateFromFile(config.problem_path)
        return jobshop.JobShopEnvironment(jobs), optimal_makespan

    elif config.problem_type == "manual":
        return create_manual_problem()

    else:
        raise ValueError(f"Unknown problem type: {config.problem_type}")


def create_manual_problem() -> Tuple[jobshop.JobShopEnvironment, None]:
    """Create manual 10x10 jobshop problem"""
    jobs = []
    # Operations format: (duration, machine)
    job_configs = [
        [(54, 0), (34, 1), (61, 2), (2, 3), (46, 4), (34, 5), (61, 6), (2, 7), (45, 8), (69, 9)],
        [(83, 1), (77, 2), (87, 3), (38, 4), (60, 5), (98, 6), (93, 7), (17, 8), (41, 9), (44, 0)],
        [(78, 2), (90, 3), (75, 4), (96, 5), (44, 6), (10, 7), (89, 8), (14, 9), (72, 0), (32, 1)],
        [(45, 3), (40, 4), (91, 5), (35, 6), (78, 7), (89, 8), (95, 9), (9, 0), (10, 1), (85, 2)],
        [(60, 4), (27, 5), (95, 6), (82, 7), (56, 8), (45, 9), (35, 0), (73, 1), (23, 2), (86, 3)],
        [(85, 5), (77, 6), (81, 7), (87, 8), (39, 9), (76, 0), (24, 1), (38, 2), (19, 3), (48, 4)],
        [(46, 6), (39, 7), (54, 8), (62, 9), (37, 0), (85, 1), (94, 2), (53, 3), (65, 4), (86, 5)],
        [(74, 7), (42, 8), (69, 9), (29, 0), (13, 1), (93, 2), (21, 3), (73, 4), (54, 5), (95, 6)],
        [(78, 8), (18, 9), (79, 0), (71, 1), (78, 2), (70, 3), (77, 4), (98, 5), (88, 6), (68, 7)],
        [(92, 9), (87, 0), (94, 1), (84, 2), (62, 3), (74, 4), (81, 5), (27, 6), (69, 7), (45, 8)]
    ]

    for ops in job_configs:
        job = jobshop.Job()
        job.operations = [jobshop.Operation(d, m) for d, m in ops]
        for op, (_, m) in zip(job.operations, ops):
            op.setEligible(m, True)
        jobs.append(job)
    return jobshop.JobShopEnvironment(jobs), None


def state_to_tuple(env: jobshop.JobShopEnvironment) -> Tuple:
    """
    Simplified but informative state representation focusing on:
    - Machine availability
    - Next operation for each job
    - Job completion status
    """
    state = env.getState()
    machine_load = [0] * env.getNumMachines()

    # Calculate immediate machine load
    for job_idx, job in enumerate(env.getJobs()):
        next_op_idx = state.nextOperationForJob[job_idx]
        if next_op_idx < len(job.operations):
            op = job.operations[next_op_idx]
            machine_load[op.machine] += op.duration

    return (
        tuple(state.machineAvailability),
        tuple(state.nextOperationForJob),
        tuple(state.completedJobs),
        tuple(machine_load)  # Add immediate machine load
    )



def action_to_tuple(action: jobshop.Action) -> Tuple[int, int, int]:
    """Convert action to hashable tuple"""
    return action.job, action.machine, action.operation


def calculate_reward(env: jobshop.JobShopEnvironment, prev_state, current_state) -> float:
    """
    Simplified reward function focusing primarily on makespan minimization
    """
    # Get completion status
    prev_progress_arr = np.array(prev_state.jobProgress.__array__(copy=False))
    current_progress_arr = np.array(current_state.jobProgress.__array__(copy=False))
    prev_completed = np.count_nonzero(prev_progress_arr)
    current_completed = np.count_nonzero(current_progress_arr)

    # Basic reward for completing operations
    progress_reward = (current_completed - prev_completed) * 10

    # Current makespan
    current_makespan = env.getTotalTime()

    if env.isDone():
        total_ops = sum(len(job.operations) for job in env.getJobs())
        if current_completed == total_ops:
            # Calculate minimal theoretical makespan
            min_makespan = max(
                sum(op.duration for op in job.operations)
                for job in env.getJobs()
            )

            # Reward inversely proportional to makespan
            return progress_reward + 1000 * (min_makespan / current_makespan)
        else:
            return -1000  # Penalty for invalid termination

    # During episode, give small reward for keeping makespan low
    time_efficiency = -0.1 * current_makespan / env.getNumMachines()

    return progress_reward + time_efficiency




class QLearningAgent:
    def __init__(self, config: QConfig):
        self.config = config
        self.q_table: Dict[Tuple, Dict[Tuple[int, int, int], float]] = defaultdict(lambda: defaultdict(float))
        self.epsilon = config.epsilon_start
        self.env, self.optimal_makespan = load_problem(config)
        self.best_makespan = float('inf')
        self.best_actions = []

    def get_action(self, state: Tuple, valid_actions: List[jobshop.Action]) -> jobshop.Action:
        """
        Action selection with simple epsilon-greedy policy but smarter exploitation
        """
        if np.random.random() < self.epsilon:
            # Pure exploration
            return np.random.choice(valid_actions)

        # Get Q-values for all valid actions
        state_q = self.q_table[state]
        action_values = {action_to_tuple(action): state_q[action_to_tuple(action)]
                         for action in valid_actions}

        if not action_values:
            return np.random.choice(valid_actions)

        # Get all actions with the maximum Q-value
        max_q_value = max(action_values.values())
        best_actions = [
            action for action in valid_actions
            if state_q[action_to_tuple(action)] == max_q_value
        ]

        # Break ties by choosing operation with minimum duration
        if len(best_actions) > 1:
            min_duration = float('inf')
            shortest_actions = []

            for action in best_actions:
                duration = self.env.getJobs()[action.job].operations[action.operation].duration
                if duration < min_duration:
                    min_duration = duration
                    shortest_actions = [action]
                elif duration == min_duration:
                    shortest_actions.append(action)

            return np.random.choice(shortest_actions)

        return best_actions[0]

    def update_q_value(self, state: Tuple, action: jobshop.Action, reward: float, next_state: Tuple):
        """Q-value update with double Q-learning approach to prevent overestimation"""
        action_tuple = action_to_tuple(action)

        # Get maximum Q-value for next state
        next_actions = self.env.getPossibleActions()
        if next_actions:
            next_q_values = [self.q_table[next_state][action_to_tuple(a)] for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
        else:
            max_next_q = 0

        # Q-learning update rule
        current_q = self.q_table[state][action_tuple]
        self.q_table[state][action_tuple] = current_q + \
                                            self.config.learning_rate * (
                                                    reward * self.config.reward_scaling +
                                                    self.config.discount_factor * max_next_q -
                                                    current_q
                                            )

    def train(self):
        """Train the Q-learning agent"""
        with wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                config=self.config.__dict__
        ) as run:
            if self.optimal_makespan:
                wandb.config.update(
                    {"optimal_makespan": self.optimal_makespan},
                    allow_val_change=True  # Allow updating the value
                )

            table_size = 0

            for episode in range(self.config.num_episodes):
                self.env.reset()
                state = state_to_tuple(self.env)
                episode_actions = []
                step_count = 0
                total_reward = 0

                prev_env_state = self.env.getState()

                while not self.env.isDone():
                    valid_actions = self.env.getPossibleActions()
                    if not valid_actions:
                        break

                    action = self.get_action(state, valid_actions)
                    episode_actions.append(action)

                    self.env.step(action)
                    next_state = state_to_tuple(self.env)
                    current_env_state = self.env.getState()

                    reward = calculate_reward(self.env, prev_env_state, current_env_state)
                    total_reward += reward

                    self.update_q_value(state, action, reward, next_state)

                    state = next_state
                    prev_env_state = current_env_state
                    step_count += 1

                self.epsilon = max(
                    self.config.epsilon_end,
                    self.epsilon * self.config.epsilon_decay
                )

                table_size = sum(len(actions) for actions in self.q_table.values())

                if self.env.isDone():
                    makespan = self.env.getTotalTime()
                    current_progress_arr = np.array(current_env_state.jobProgress.__array__(copy=False))
                    completed_ops = np.count_nonzero(current_progress_arr)
                    total_ops = sum(len(job.operations) for job in self.env.getJobs())

                    # Calculate optimality gap if optimal makespan is known
                    optimality_gap = None
                    if self.optimal_makespan:
                        optimality_gap = (makespan - self.optimal_makespan) / self.optimal_makespan * 100

                    metrics = {
                        'episode': episode,
                        'makespan': makespan,
                        'steps': step_count,
                        'epsilon': self.epsilon,
                        'q_table_size': table_size,
                        'total_reward': total_reward,
                        'completion_ratio': completed_ops / total_ops,
                    }

                    if optimality_gap is not None:
                        metrics['optimality_gap'] = optimality_gap

                    wandb.log(metrics)

                    if makespan < self.best_makespan and completed_ops == total_ops:
                        self.best_makespan = makespan
                        self.best_actions = episode_actions.copy()

                        schedule = self.env.getScheduleData()
                        schedule_data = [
                            [m_idx, entry.job, entry.operation, entry.start, entry.duration]
                            for m_idx, m_schedule in enumerate(schedule)
                            for entry in m_schedule
                        ]

                        summary = {
                            'best_makespan': makespan,
                            'best_solution_episode': episode,
                            'q_table_final_size': table_size
                        }

                        if self.optimal_makespan:
                            summary['optimal_makespan'] = self.optimal_makespan
                            summary['final_optimality_gap'] = (makespan - self.optimal_makespan) / self.optimal_makespan * 100

                        wandb.run.summary.update(summary)

                        wandb.log({
                            'best_schedule': wandb.Table(
                                columns=['Machine', 'Job', 'Operation', 'Start', 'Duration'],
                                data=schedule_data
                            )
                        })

                if episode % 100 == 0:
                    status = f"Episode {episode}: Best makespan = {self.best_makespan}"
                    if self.optimal_makespan:
                        gap = (self.best_makespan - self.optimal_makespan) / self.optimal_makespan * 100
                        status += f", Gap = {gap:.2f}%"
                    status += f", Completion ratio = {completed_ops/total_ops:.2f}"
                    print(status)


def parse_args():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling with Q-Learning')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Starting epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.01,
                        help='Final epsilon for exploration')
    parser.add_argument('--eps_decay', type=float, default=0.999,
                        help='Epsilon decay rate')
    parser.add_argument('--project', type=str, default="jobshop_10x10",
                        help='WandB project name')
    parser.add_argument('--problem_type', choices=['manual', 'taillard', 'json'],
                        default='json', help='Type of problem to solve')
    parser.add_argument('--taillard_instance',
                        choices=[f"TA{i:02d}" for i in range(1, 81)],
                        default="TA01", help="Taillard instance")
    parser.add_argument('--problem_path', type=str, default="/home/per/jsp/jsp/environments/doris.json",
                        help='Path to JSON problem file')
    parser.add_argument('--reward_scaling', type=float, default=0.01,
                        help='Scaling factor for rewards')

    return parser.parse_args()


def main():
    args = parse_args()

    config = QConfig(
        num_episodes=args.episodes,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay=args.eps_decay,
        problem_type=args.problem_type,
        problem_instance=args.taillard_instance,
        problem_path=args.problem_path,
        project=args.project,
        reward_scaling=args.reward_scaling
    )

    print(f"Starting training with configuration:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")

    agent = QLearningAgent(config)

    if config.problem_type != 'manual' and agent.optimal_makespan:
        print(f"Optimal makespan for this instance: {agent.optimal_makespan}")

    agent.train()

    print("\nTraining completed!")
    print(f"Best makespan achieved: {agent.best_makespan}")
    if agent.optimal_makespan:
        gap = (agent.best_makespan - agent.optimal_makespan) / agent.optimal_makespan * 100
        print(f"Gap to optimal: {gap:.2f}%")


if __name__ == "__main__":
    main()
