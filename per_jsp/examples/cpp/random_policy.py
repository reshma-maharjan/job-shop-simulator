import numpy as np
import per_jspp as jobshop
from dataclasses import dataclass
import wandb


@dataclass
class Config:
    num_episodes: int = 10
    project: str = "jobshop"


def create_jobs() -> jobshop.JobShopEnvironment:
    """Create sample 3x3 jobshop problem"""
    jobs = []
    # Operations: (duration, machine)
    job_configs = [
        [(3, 0), (2, 1), (4, 2)],  # Job 1
        [(2, 1), (4, 0), (3, 2)],  # Job 2
        [(3, 2), (3, 1), (2, 0)]  # Job 3
    ]

    for ops in job_configs:
        job = jobshop.Job()
        job.operations = [jobshop.Operation(d, m) for d, m in ops]
        for op, (_, m) in zip(job.operations, ops):
            op.setEligible(m, True)
        jobs.append(job)

    return jobshop.JobShopEnvironment(jobs)


def run_experiment(config: Config):
    """Run random policy experiment"""
    env = create_jobs()
    best_makespan = float('inf')

    with wandb.init(project=config.project) as run:
        for episode in range(config.num_episodes):
            env.reset()
            steps = 0

            while not env.isDone():
                actions = env.getPossibleActions()
                if not actions:
                    break

                env.step(np.random.choice(actions))
                steps += 1

            if env.isDone():  # Only log successful episodes
                makespan = env.getTotalTime()
                wandb.log({
                    'episode': episode,
                    'makespan': makespan,
                    'steps': steps,
                })

                if makespan < best_makespan:
                    best_makespan = makespan
                    wandb.run.summary['best_makespan'] = makespan


if __name__ == "__main__":
    run_experiment(Config())
