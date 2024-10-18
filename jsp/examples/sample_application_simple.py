import argparse
import jobshop
from tqdm import tqdm

def run_experiment(algorithm_name: str, taillard_instance: str):
    algorithm_class = getattr(jobshop, f"JobShop{algorithm_name}")
    instance = getattr(jobshop.TaillardInstance, taillard_instance)
    jobs, ta_optimal = jobshop.TaillardJobShopGenerator.loadProblem(instance, True)
    print(f"Optimal makespan for {taillard_instance}: {ta_optimal}")

    env = jobshop.JobShopEnvironment(jobs)
    algorithm = algorithm_class(env, 0.1, 0.9, 0.3)

    print("Initial Schedule:")
    algorithm.printBestSchedule()

    total_episodes = 10000
    best_makespan = float('inf')

    with tqdm(total=total_episodes, desc="Training") as pbar:
        def callback(make_span: int):
            nonlocal best_makespan
            if make_span < best_makespan:
                best_makespan = make_span
                pbar.set_postfix_str(f"Best makespan: {best_makespan}")
            pbar.update(1)

        algorithm.train(total_episodes, callback)

    print("\nFinal Best Schedule:")
    algorithm.printBestSchedule()
    print(f"Best makespan achieved: {best_makespan}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Job Shop Scheduling experiment")
    parser.add_argument("algorithm", choices=["QLearning", "ActorCritic"], help="Algorithm type")
    parser.add_argument("taillard_instance", choices=[f"TA{i:02d}" for i in range(1, 81)], help="Taillard instance")
    args = parser.parse_args()

    run_experiment(args.algorithm, args.taillard_instance)