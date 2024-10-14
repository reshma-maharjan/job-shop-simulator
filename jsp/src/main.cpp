#include "job_shop_environment.h"
#include "job_shop_qlearning.h"
#include "job_shop_actor_critic.h"
#include "job_shop_plotter.h"
#include "job_shop_taillard_generator.h"
#include "job_shop_manual_generator.h"
#include <iostream>
#include <thread>
#include <indicators/progress_bar.hpp>
#include <indicators/multi_progress.hpp>

struct UpdateData {
    std::vector<std::vector<ScheduleEntry>> scheduleData;
    int makespan;
    int thread_id;
};

template <typename Algorithm>
void runExperiments(int n_threads, bool use_gui) {
    // Load problem
    auto [jobs, ta01Optimal] = TaillardJobShopGenerator::loadProblem(TaillardInstance::TA42);
    //std::cout << "Optimal makespan for TA42: " << ta01Optimal << std::endl;

    //auto [jobs, ta01Optimal] = ManualJobShopGenerator::generateFromFile("/home/per/jsp/jsp/environments/doris.csv");

    // Create environments and agents
    std::vector<std::unique_ptr<JobShopEnvironment>> environments;
    std::vector<std::unique_ptr<Algorithm>> agents;

    for (int i = 0; i < n_threads; ++i) {
        environments.push_back(std::make_unique<JobShopEnvironment>(jobs));
        agents.push_back(std::make_unique<Algorithm>(*environments.back(), 0.1, 0.9, 0.3));
    }

    // Generate operation graph (only once)
    environments[0]->generateOperationGraph("operation_graph.dot");
    std::cout << "Operation graph generated: operation_graph.dot\n";
    std::cout << "Use a DOT viewer to visualize the graph.\n";


    std::unique_ptr<LivePlotter> plotter = nullptr;
    if (use_gui) {
        plotter = std::make_unique<LivePlotter>(environments[0]->getNumMachines());
    }

    // Progress bars
    auto episode_bar = indicators::ProgressBar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"|"},
            indicators::option::Fill{"█"},
            indicators::option::Lead{"█"},
            indicators::option::Remainder{"-"},
            indicators::option::End{"|"},
            indicators::option::PostfixText{"Episodes"},
            indicators::option::ForegroundColor{indicators::Color::cyan},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
    };

    auto makespan_bar = indicators::ProgressBar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"|"},
            indicators::option::Fill{"█"},
            indicators::option::Lead{"█"},
            indicators::option::Remainder{"-"},
            indicators::option::End{"|"},
            indicators::option::PostfixText{"Makespan"},
            indicators::option::ForegroundColor{indicators::Color::green},
    };

    indicators::MultiProgress<indicators::ProgressBar, 2> multi_progress(episode_bar, makespan_bar);

    std::queue<UpdateData> updateQueue;
    std::mutex queueMutex;
    std::atomic<bool> training_complete(false);
    std::atomic<int> active_threads(n_threads);

    auto training_function = [&](int thread_id) {
        Algorithm& algorithm = *agents[thread_id];
        JobShopEnvironment& env = *environments[thread_id];

        std::cout << "Initial Schedule for thread " << thread_id << ":" << std::endl;
        algorithm.printBestSchedule();
        std::cout << "\nTraining thread " << thread_id << "..." << std::endl;

        int bestMakespan = std::numeric_limits<int>::max();
        int totalEpisodes = 100000;

        algorithm.train(totalEpisodes, [&](int makeSpan) {
            static thread_local int episode = 0;
            episode++;

            // Update episode progress
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                episode_bar.set_progress(100.0 * episode / (totalEpisodes * n_threads));
            }

            if (makeSpan < bestMakespan) {
                bestMakespan = makeSpan;

                // Update the schedule data in the plotter
                auto scheduleData = env.getScheduleData();
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    updateQueue.push({scheduleData, env.getTotalTime(), thread_id});
                }

                // Update makespan progress
                double makespanProgress = std::min(100.0, 100.0 * static_cast<double>(ta01Optimal) / bestMakespan);
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    makespan_bar.set_progress(makespanProgress);
                    makespan_bar.set_option(indicators::option::PostfixText{"Makespan: " + std::to_string(bestMakespan)});
                }
            }
        });

        std::cout << "\nFinal Best Schedule for thread " << thread_id << ":" << std::endl;
        algorithm.printBestSchedule();
        algorithm.saveBestScheduleToFile("schedule_data_thread_" + std::to_string(thread_id) + ".txt");

        if (--active_threads == 0) {
            training_complete.store(true);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(training_function, i);
    }

    int lowestMakespan = std::numeric_limits<int>::max();
    while (!training_complete.load()) {
        if (use_gui) {
            plotter->render();
        }

        // Process update queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!updateQueue.empty()) {
                auto update = updateQueue.front();
                updateQueue.pop();
                if (update.makespan < lowestMakespan) {
                    if (use_gui) {
                        plotter->updateSchedule(update.scheduleData, update.makespan);
                    }
                    lowestMakespan = update.makespan;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

int main(int argc, char* argv[]) {
    argc = 3;
    argv[1] = "qlearning";
    argv[2] = "--gui";

    if (argc > 2) {
        std::string algorithmType = argv[1];
        std::string guiOption = argv[2];


        bool use_gui = (guiOption != "--no-gui");

        if (algorithmType == "qlearning") {
            runExperiments<JobShopQLearning>(24, use_gui);
        } else if (algorithmType == "actorcritic") {
            runExperiments<JobShopActorCritic>(24, use_gui);
        } else {
            std::cerr << "Unknown algorithm type: " << algorithmType << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " <algorithm> <gui_option>" << std::endl;
        return 1;
    }

    return 0;
}
