#include <spdlog/spdlog.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <indicators/progress_bar.hpp>
#include <indicators/multi_progress.hpp>
#include "environment/job_shop_environment.h"
#include "algorithms/job_shop_qlearning.h"
#include "algorithms/job_shop_ppo.h"
#include "algorithms/job_shop_actor_critic.h"
#include "gui/job_shop_plotter.h"
#include "environment/job_shop_environment_generator.h"
#include "environment/job_shop_taillard_generator.h"
#include "environment/job_shop_manual_generator.h"

// Algorithm name resolution
template <typename Algorithm>
std::string getAlgorithmName() {
    if constexpr (std::is_same_v<Algorithm, JobShopQLearning>) {
        return "Q-Learning";
    } else if constexpr (std::is_same_v<Algorithm, JobShopActorCritic>) {
        return "Actor-Critic";
    } else if constexpr (std::is_same_v<Algorithm, JobShopPPO>) {
        return "Proximal Policy Optimization (PPO)";
    }
    return "Unknown Algorithm";
}

// Data structures
struct UpdateData {
    std::vector<std::vector<ScheduleEntry>> scheduleData;
    int makespan;
    int thread_id;
};

struct ProgressBars {
    bool enabled;
    indicators::ProgressBar episode_bar;
    indicators::ProgressBar makespan_bar;

    void updateEpisodeProgress(double progress) {
        if (enabled) {
            episode_bar.set_progress(progress);
        }
    }

    void updateMakespanProgress(double progress, int makespan) {
        if (enabled) {
            makespan_bar.set_progress(progress);
            makespan_bar.set_option(indicators::option::PostfixText{
                    "Makespan: " + std::to_string(makespan)});
        }
    }
};

// Progress bar creation
ProgressBars createProgressBars(bool enabled) {

    return ProgressBars{
            .enabled = enabled,
            .episode_bar = indicators::ProgressBar(
                    indicators::option::BarWidth{50},
                    indicators::option::Start{"|"},
                    indicators::option::Fill{"█"},
                    indicators::option::Lead{"█"},
                    indicators::option::Remainder{"-"},
                    indicators::option::End{"|"},
                    indicators::option::PostfixText{"Episodes"},
                    indicators::option::ForegroundColor{indicators::Color::cyan},
                    indicators::option::ShowElapsedTime{true},
                    indicators::option::ShowRemainingTime{true}
            ),
            .makespan_bar = indicators::ProgressBar(
                    indicators::option::BarWidth{50},
                    indicators::option::Start{"|"},
                    indicators::option::Fill{"█"},
                    indicators::option::Lead{"█"},
                    indicators::option::Remainder{"-"},
                    indicators::option::End{"|"},
                    indicators::option::PostfixText{"Makespan"},
                    indicators::option::ForegroundColor{indicators::Color::green}
            )
    };
}

// Initialize environments and agents
template <typename Algorithm>
std::pair<std::vector<std::unique_ptr<JobShopEnvironment>>, std::vector<std::unique_ptr<Algorithm>>>
initializeEnvironmentsAndAgents(const std::vector<Job>& jobs, int numThreads) {
    std::vector<std::unique_ptr<JobShopEnvironment>> environments;
    std::vector<std::unique_ptr<Algorithm>> agents;

    for (int i = 0; i < numThreads; ++i) {
        environments.push_back(std::make_unique<JobShopEnvironment>(jobs));
        if constexpr (std::is_same_v<Algorithm, JobShopPPO>) {
            agents.push_back(std::make_unique<Algorithm>(*environments.back(), 0.1, 0.9, 0.3, 0.5));
        } else {
            agents.push_back(std::make_unique<Algorithm>(*environments.back(), 0.1, 0.9, 0.3));
        }
    }

    return {std::move(environments), std::move(agents)};
}

// Problem type and configuration
enum class ProblemType {
    TAILLARD,
    MANUAL,
    AUTO_DEFAULT,
    AUTO_DIFFICULT
};

struct ProblemConfig {
    ProblemType type;
    std::string filePath;
    struct AutoGenParams {
        int numJobs;
        int numMachines;
        int minDuration;
        int maxDuration;
        double dependencyDensity;
        int maxDependenciesPerJob;
        double longJobRate;
        double longJobFactor;
        std::optional<std::string> outputFile;
    };
    std::optional<AutoGenParams> autoParams;
};

struct TrainingConfig {
    int numThreads;
    bool useGUI;
    bool showProgress;
    int totalEpisodes;
    std::string outputPrefix;
    ProblemConfig problemConfig;
};

// Problem generation
std::pair<std::vector<Job>, int> generateProblem(const ProblemConfig& config) {
    switch (config.type) {
        case ProblemType::TAILLARD: {
            SPDLOG_INFO("Loading Taillard instance from: {}", config.filePath);
            return TaillardJobShopGenerator::loadProblem(config.filePath);
        }
        case ProblemType::MANUAL: {
            SPDLOG_INFO("Loading manual problem from: {}", config.filePath);
            return ManualJobShopGenerator::generateFromFile(config.filePath);
        }
        case ProblemType::AUTO_DEFAULT: {
            if (!config.autoParams) {
                throw std::runtime_error("Auto generation parameters required but not provided");
            }
            SPDLOG_INFO("Generating default automatic problem with {} jobs and {} machines",
                        config.autoParams->numJobs, config.autoParams->numMachines);
            return AutomaticJobShopGenerator::generateDefault(
                    config.autoParams->numJobs,
                    config.autoParams->numMachines,
                    config.autoParams->outputFile);
        }
        case ProblemType::AUTO_DIFFICULT: {
            if (!config.autoParams) {
                throw std::runtime_error("Auto generation parameters required but not provided");
            }
            SPDLOG_INFO("Generating difficult automatic problem");
            AutomaticJobShopGenerator::GenerationParams params{
                    .numJobs = config.autoParams->numJobs,
                    .numMachines = config.autoParams->numMachines,
                    .minDuration = config.autoParams->minDuration,
                    .maxDuration = config.autoParams->maxDuration,
                    .dependencyDensity = config.autoParams->dependencyDensity,
                    .maxDependenciesPerJob = config.autoParams->maxDependenciesPerJob,
                    .longJobRate = config.autoParams->longJobRate,
                    .longJobFactor = config.autoParams->longJobFactor,
                    .outputFile = config.autoParams->outputFile
            };
            return AutomaticJobShopGenerator::generate(params);
        }
        default:
            throw std::runtime_error("Unknown problem type");
    }
}

// Main experiment running function
template <typename Algorithm>
void runExperiments(const TrainingConfig& config) {
    SPDLOG_INFO("Running Algorithm: {}", getAlgorithmName<Algorithm>());

    auto [jobs, optimalMakespan] = generateProblem(config.problemConfig);
    auto [environments, agents] = initializeEnvironmentsAndAgents<Algorithm>(jobs, config.numThreads);

    environments[0]->generateOperationGraph("operation_graph.dot");
    SPDLOG_INFO("Operation graph generated: operation_graph.dot");

    std::unique_ptr<LivePlotter> plotter = config.useGUI ?
                                           std::make_unique<LivePlotter>(
                                                   environments[0]->getNumMachines(),
                                                    environments[0]->getJobs()
                                                   ) : nullptr;

    auto progress_bars = createProgressBars(config.showProgress);
    std::queue<UpdateData> updateQueue;
    std::mutex queueMutex;
    std::atomic<bool> training_complete(false);
    std::atomic<int> active_threads(config.numThreads);

    auto training_function = [&](int thread_id) {
        Algorithm& algorithm = *agents[thread_id];
        JobShopEnvironment& env = *environments[thread_id];

        SPDLOG_INFO("Initial Schedule for thread {}", thread_id);
        algorithm.printBestSchedule();
        SPDLOG_INFO("Training thread {}...", thread_id);

        int bestMakespan = std::numeric_limits<int>::max();

        algorithm.train(config.totalEpisodes, [&](int makeSpan) {
            static thread_local int episode = 0;
            episode++;

            {
                std::lock_guard<std::mutex> lock(queueMutex);
                progress_bars.updateEpisodeProgress(
                        100.0 * episode / (config.totalEpisodes * config.numThreads));
            }

            if (makeSpan < bestMakespan) {
                bestMakespan = makeSpan;
                auto scheduleData = env.getScheduleData();
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    updateQueue.push({scheduleData, env.getTotalTime(), thread_id});
                }

                double makespanProgress = std::min(100.0,
                                                   100.0 * static_cast<double>(optimalMakespan) / bestMakespan);
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    progress_bars.updateMakespanProgress(makespanProgress, bestMakespan);
                }
            }
        });

        SPDLOG_INFO("Final Best Schedule for thread {}", thread_id);
        algorithm.printBestSchedule();
        algorithm.saveBestScheduleToFile(fmt::format("{}_thread_{}.txt",
                                                     config.outputPrefix, thread_id));

        if (--active_threads == 0) {
            training_complete.store(true);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < config.numThreads; ++i) {
        threads.emplace_back(training_function, i);
    }

    int lowestMakespan = std::numeric_limits<int>::max();
    while (!training_complete.load()) {
        if (config.useGUI) {
            plotter->render();
        }

        std::lock_guard<std::mutex> lock(queueMutex);
        while (!updateQueue.empty()) {
            auto update = updateQueue.front();
            updateQueue.pop();
            if (update.makespan < lowestMakespan) {
                if (config.useGUI) {
                    plotter->updateSchedule(update.scheduleData, update.makespan);
                }
                lowestMakespan = update.makespan;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    SPDLOG_INFO("Training completed. Best makespan achieved: {}", lowestMakespan);
}
int main(int argc, char* argv[]) {
    try {
        spdlog::set_level(spdlog::level::debug);

        const int MAX_ARGS = 6;  // Increased to accommodate --no-progress
        std::vector<std::unique_ptr<char[]>> default_args;
        char** argv_to_use = argv;

        // Set default values if no arguments provided
        if (argc == 1) {
            default_args.resize(MAX_ARGS);
            for (int i = 0; i < MAX_ARGS; i++) {
                default_args[i] = std::make_unique<char[]>(100);
            }

            strcpy(default_args[0].get(), "program");
            strcpy(default_args[1].get(), "qlearning");
            strcpy(default_args[2].get(), "--gui");
            strcpy(default_args[3].get(), "manual");
            strcpy(default_args[4].get(), "/home/per/jsp/jsp/environments/doris.json");
            strcpy(default_args[5].get(), "--no-progress");

            argv_to_use = new char*[MAX_ARGS];
            for (int i = 0; i < MAX_ARGS; i++) {
                argv_to_use[i] = default_args[i].get();
            }
            argc = MAX_ARGS;
        }

        if (argc < 4) {
            SPDLOG_ERROR("Usage: {} <algorithm> <gui_option> <problem_type> [problem_params...] [--no-progress]", argv_to_use[0]);
            SPDLOG_ERROR("Algorithms: qlearning, actorcritic, ppo");
            SPDLOG_ERROR("GUI options: --gui, --no-gui");
            SPDLOG_ERROR("Problem types: taillard, manual, auto_default, auto_difficult");
            return 1;
        }

        std::string algorithmType = argv_to_use[1];
        std::string guiOption = argv_to_use[2];
        std::string problemType = argv_to_use[3];

        // Check for --no-progress option
        bool showProgress = true;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv_to_use[i]) == "--no-progress") {
                showProgress = false;
                break;
            }
        }

        // Rest of the function remains the same...
        ProblemConfig problemConfig;
        if (problemType == "taillard") {
            if (argc < 5) {
                SPDLOG_ERROR("Taillard problem requires file path");
                return 1;
            }
            problemConfig.type = ProblemType::TAILLARD;
            problemConfig.filePath = argv_to_use[4];
        }
        else if (problemType == "manual") {
            if (argc < 5) {
                SPDLOG_ERROR("Manual problem requires file path");
                return 1;
            }
            problemConfig.type = ProblemType::MANUAL;
            problemConfig.filePath = argv_to_use[4];
        }
        else if (problemType == "auto_default") {
            problemConfig.type = ProblemType::AUTO_DEFAULT;
            problemConfig.autoParams = ProblemConfig::AutoGenParams{
                    .numJobs = 36,
                    .numMachines = 3,
                    .minDuration = 1,
                    .maxDuration = 100,
                    .dependencyDensity = 0.3,
                    .maxDependenciesPerJob = 3,
                    .longJobRate = 0.1,
                    .longJobFactor = 2.0,
                    .outputFile = "default_problem.json"
            };
        }
        else if (problemType == "auto_difficult") {
            problemConfig.type = ProblemType::AUTO_DIFFICULT;
            problemConfig.autoParams = ProblemConfig::AutoGenParams{
                    .numJobs = 180,
                    .numMachines = 3,
                    .minDuration = 10,
                    .maxDuration = 210,
                    .dependencyDensity = 0.95,
                    .maxDependenciesPerJob = 12,
                    .longJobRate = 0.35,
                    .longJobFactor = 3.0,
                    .outputFile = "difficult_problem.json"
            };
        }
        else {
            SPDLOG_ERROR("Unknown problem type: {}", problemType);
            return 1;
        }

        TrainingConfig config{
                .numThreads = 24,
                .useGUI = (guiOption == "--gui"),
                .showProgress = showProgress,
                .totalEpisodes = 10000,
                .outputPrefix = fmt::format("schedule_data_{}", problemType),
                .problemConfig = problemConfig
        };

        if (algorithmType == "qlearning") {
            runExperiments<JobShopQLearning>(config);
        } else if (algorithmType == "actorcritic") {
            runExperiments<JobShopActorCritic>(config);
        } else if (algorithmType == "ppo") {
            runExperiments<JobShopPPO>(config);
        } else {
            SPDLOG_ERROR("Unknown algorithm type: {}", algorithmType);
            return 1;
        }

        // Cleanup if we used default args
        if (argv_to_use != argv) {
            delete[] argv_to_use;
        }

        return 0;
    }
    catch (const std::exception& e) {
        SPDLOG_ERROR("Program terminated with error: {}", e.what());
        return 1;
    }
}