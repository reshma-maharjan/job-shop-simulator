# main.py
import argparse
import logging
import time
import wandb
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.python.per_jsp.algorithms.q_learning import QLearningScheduler

from per_jsp.python.per_jsp.environment.job_shop_environment import (
    JobShopEnvironment,
    Action,
    ScheduleEntry,
    Job,
)
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator
from per_jsp.python.per_jsp.environment.job_shop_manual_generator import ManualJobShopGenerator
from per_jsp.python.per_jsp.environment.job_shop_automatic_generator import AutomaticJobShopGenerator
from per_jsp.python.per_jsp.environment.structs import GenerationParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling')
    parser.add_argument('--taillard', action='store_true',
                      help='Use Taillard instance')
    parser.add_argument('--algorithm', type=str, default='q-learning',
                      help='Algorithm to use')
    parser.add_argument('--taillard-file', type=str,
                      help='Path to Taillard instance file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize wandb for the run
    run_name = f"ta80-qlearning-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="job-shop-scheduling_q-learning",
        name=run_name,
        config={
            "algorithm": args.algorithm,
            "taillard_file": args.taillard_file,
        }
    )

    # Load Taillard instance
    generator = TaillardJobShopGenerator()
    jobs, estimated_makespan = generator.load_problem(args.taillard_file)
    logger.info(f"Loaded Taillard instance with estimated makespan: {estimated_makespan}")

    # Update wandb config with problem details
    wandb.config.update({
        "num_jobs": len(jobs),
        "num_machines": len(jobs[0].operations),
        "estimated_makespan": estimated_makespan
    })

    # Create environment
    env = JobShopEnvironment(jobs)
    logger.info(f"Created environment with {len(jobs)} jobs and {env.num_machines} machines")

    # Create and run scheduler
    logger.info(f"Using algorithm: {args.algorithm}")
    scheduler = QLearningScheduler(
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        episodes=5000
    )

    # Solve and get results
    best_schedule, makespan = scheduler.solve(env)

    # Log final results to wandb
    wandb.log({
        "final_makespan": makespan,
        "makespan_ratio": makespan / estimated_makespan if estimated_makespan > 0 else 0,
        "schedule_length": len(best_schedule)
    })

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()