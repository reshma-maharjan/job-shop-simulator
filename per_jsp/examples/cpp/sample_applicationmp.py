import argparse
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import List

import jobshop
from tqdm import tqdm

@dataclass
class UpdateData:
    schedule_data: List[jobshop.ScheduleEntry]
    makespan: int
    process_id: int

def run_training(process_id: int, algorithm_class, env: jobshop.JobShopEnvironment,
                 total_episodes: int, update_queue: mp.Queue, error_queue: mp.Queue,
                 episode_counter: mp.Value, best_makespan: mp.Value):
    try:
        algorithm = algorithm_class(env, 0.1, 0.9, 0.3)
        print(f"Initial Schedule for process {process_id}:")
        algorithm.printBestSchedule()
        print(f"\nTraining process {process_id}...")

        def callback(make_span: int):
            with episode_counter.get_lock():
                episode_counter.value += 1
                current_episode = episode_counter.value

            with best_makespan.get_lock():
                if make_span < best_makespan.value:
                    best_makespan.value = make_span
                    print(f"Episode {current_episode}, New best time: {make_span}")
                    schedule_data = env.getScheduleData()
                    update_queue.put(UpdateData(schedule_data, env.getTotalTime(), process_id))

        algorithm.train(total_episodes, callback)

        print(f"\nFinal Best Schedule for process {process_id}:")
        algorithm.printBestSchedule()
        algorithm.saveBestScheduleToFile(f"schedule_data_process_{process_id}.txt")

    except Exception as e:
        error_queue.put((process_id, str(e)))

def run_experiments(algorithm_name: str, taillard_instance: str, n_processes: int, use_gui: bool):
    algorithm_class = getattr(jobshop, f"JobShop{algorithm_name}")
    instance = getattr(jobshop.TaillardInstance, taillard_instance)
    jobs, ta_optimal = jobshop.TaillardJobShopGenerator.loadProblem(instance, True)
    print(f"Optimal makespan for {taillard_instance}: {ta_optimal}")

    environments = [jobshop.JobShopEnvironment(jobs) for _ in range(n_processes)]
    plotter = jobshop.LivePlotter(environments[0].getNumMachines()) if use_gui else None

    update_queue = mp.Queue()
    error_queue = mp.Queue()
    episode_counter = mp.Value('i', 0)
    best_makespan = mp.Value('i', 1000000)

    total_episodes = 100000
    episodes_per_process = total_episodes // n_processes

    with tqdm(total=total_episodes, desc="Episodes", position=0) as episode_bar, \
            tqdm(total=100, desc="Makespan", position=1) as makespan_bar:

        processes = [
            mp.Process(target=run_training,
                       args=(i, algorithm_class, environments[i], episodes_per_process,
                             update_queue, error_queue, episode_counter, best_makespan))
            for i in range(n_processes)
        ]

        for process in processes:
            process.start()

        lowest_makespan = 1000000
        try:
            while any(p.is_alive() for p in processes):
                if use_gui:
                    plotter.render()

                try:
                    while True:
                        update = update_queue.get_nowait()
                        if update.makespan < lowest_makespan:
                            if use_gui:
                                plotter.updateSchedule(update.schedule_data, update.makespan)
                            lowest_makespan = update.makespan
                            print(f"New lowest makespan: {lowest_makespan}")

                        episode_bar.n = episode_counter.value
                        episode_bar.refresh()
                        makespan_progress = min(100.0, 100.0 * ta_optimal / lowest_makespan)
                        makespan_bar.n = int(makespan_progress)
                        makespan_bar.set_postfix_str(f"Makespan: {lowest_makespan}")
                        makespan_bar.refresh()

                except queue.Empty:
                    pass

                try:
                    while True:
                        process_id, error_msg = error_queue.get_nowait()
                        print(f"Error in process {process_id}: {error_msg}")
                except queue.Empty:
                    pass

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Interrupted by user. Stopping processes...")

        finally:
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    print(f"Process {process.pid} did not terminate gracefully. Terminating...")
                    process.terminate()

    print(f"Best makespan achieved: {lowest_makespan}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Job Shop Scheduling experiments")
    parser.add_argument("algorithm", choices=["QLearning", "ActorCritic"], help="Algorithm type")
    parser.add_argument("taillard_instance", choices=[f"TA{i:02d}" for i in range(1, 81)], help="Taillard instance")
    parser.add_argument("--processes", type=int, default=24, help="Number of processes to use")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    args = parser.parse_args()

    run_experiments(args.algorithm, args.taillard_instance, args.processes, not args.no_gui)