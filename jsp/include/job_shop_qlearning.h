#ifndef JOB_SHOP_QLEARNING_H
#define JOB_SHOP_QLEARNING_H

#include "job_shop_environment.h"
#include <random>
#include <unordered_map>
#include <functional>
#include <array>
#include <limits>
#include <iostream>
#include <fstream>

class JobShopQLearning {
private:
    JobShopEnvironment& env;
    MultiDimensionalArray<double, 3> qTable;
    double learningRate;
    double discountFactor;
    double explorationRate;
    std::mt19937 rng;
    int bestTime;
    std::vector<Action> bestSchedule;
    mutable std::vector<Action> cachedActions;
    mutable std::vector<double> cachedQValues;

    std::array<size_t, 3> initializeQTableDimensions() {
        const auto& jobs = env.getJobs();
        size_t maxOperations = std::max_element(jobs.begin(), jobs.end(),
                                                [](const Job& a, const Job& b) { return a.operations.size() < b.operations.size(); }
        )->operations.size();

        return {jobs.size(), static_cast<size_t>(env.getNumMachines()), maxOperations};
    }

    double calculatePriority(const Action& action, const State& state) const {
        const auto& jobs = env.getJobs();
        double remainingTime = 0;
        for (size_t i = action.operation; i < jobs[action.job].operations.size(); ++i) {
            remainingTime += jobs[action.job].operations[i].duration;
        }
        double machineUtilization = state.machineAvailability[action.machine] / static_cast<double>(env.getTotalTime());
        return remainingTime * (1 - machineUtilization);
    }

    Action getAction(const State& state) {
        const std::vector<Action>& possibleActions = env.getPossibleActions();

        if (possibleActions.empty()) {
            return Action(-1, -1, -1);  // No-op action
        }

        if (std::uniform_real_distribution<>(0, 1)(rng) < explorationRate) {
            std::vector<double> priorities;
            for (const auto& action : possibleActions) {
                priorities.push_back(calculatePriority(action, state));
            }
            std::discrete_distribution<> dist(priorities.begin(), priorities.end());
            return possibleActions[dist(rng)];
        } else {
            double maxQ = -std::numeric_limits<double>::max();
            Action bestAction = possibleActions[0];
            for (const auto& action : possibleActions) {
                double q = qTable(action.job, action.machine, action.operation);
                if (q > maxQ) {
                    maxQ = q;
                    bestAction = action;
                }
            }
            return bestAction;
        }
    }

    void updateQValue(const State& state, const Action& action, double reward, const State& nextState) {
        auto currentQ = qTable(action.job, action.machine, action.operation);
        const std::vector<Action>& nextPossibleActions = env.getPossibleActions();
        double maxNextQ = 0.0;
        for (const auto& nextAction : nextPossibleActions) {
            maxNextQ = std::max(maxNextQ, qTable(nextAction.job, nextAction.machine, nextAction.operation));
        }

        double timeReward = -static_cast<double>(env.getTotalTime());
        double utilizationReward = 0;

        for (int machineAvailability : state.machineAvailability) {
            utilizationReward += static_cast<double>(machineAvailability) / env.getTotalTime();
        }

        double combinedReward = timeReward + utilizationReward;

        qTable(action.job, action.machine, action.operation) = (1 - learningRate) * currentQ + learningRate * (combinedReward + discountFactor * maxNextQ);
    }

    int evaluateEpisode(const std::vector<Action>& actions) {
        env.reset();
        for (const Action& action : actions) {
            env.step(action);
        }
        return env.getTotalTime();
    }

public:
    JobShopQLearning(JobShopEnvironment& environment, double alpha, double gamma, double epsilon)
            : env(environment),
              learningRate(alpha),
              qTable(initializeQTableDimensions()),
              discountFactor(gamma),
              explorationRate(epsilon),
              rng(std::random_device{}()),
              bestTime(std::numeric_limits<int>::max())
    {
        qTable.fill(0.0);
        cachedActions.reserve(env.getNumMachines() * env.getJobs().size());
        cachedQValues.reserve(env.getNumMachines() * env.getJobs().size());
    }

    std::vector<Action> runEpisode() {
        env.reset();
        std::vector<Action> schedule;
        schedule.reserve(env.getJobs().size() * env.getNumMachines());

        while (!env.isDone()) {
            State currentState = env.getState();
            Action action = getAction(currentState);

            if (action.job == -1 && action.machine == -1 && action.operation == -1) {
                break;  // No more actions available
            }

            schedule.push_back(action);
            State nextState = env.step(action);

            updateQValue(currentState, action, 0, nextState);
        }

        return schedule;
    }

    void train(int numEpisodes, const std::function<void(int)>& episodeCallback = [](int){}) {
        for (int i = 0; i < numEpisodes; ++i) {
            std::vector<Action> episode = runEpisode();
            int episodeTime = evaluateEpisode(episode);

            if (episodeTime < bestTime) {
                bestTime = episodeTime;
                bestSchedule = std::move(episode);
                std::cout << "Episode " << i << ", New best time: " << bestTime << std::endl;
            }

            explorationRate *= 0.9999;  // Slower decay
            episodeCallback(episodeTime);
        }

        std::cout << "Training completed. Best time: " << bestTime << std::endl;
        applyAndPrintSchedule(bestSchedule);
    }

    void applyAndPrintSchedule(const std::vector<Action>& schedule) {
        env.reset();
        for (const Action& action : schedule) {
            env.step(action);
        }
        env.printSchedule();
    }

    void printBestSchedule() {
        std::cout << "Best Schedule (Total time: " << bestTime << "):" << std::endl;
        applyAndPrintSchedule(bestSchedule);
    }

    void saveBestScheduleToFile(const std::string& filename) {
        env.reset();
        for (const Action& action : bestSchedule) {
            env.step(action);
        }
        auto scheduleData = env.getScheduleData();

        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return;
        }

        for (size_t machine = 0; machine < scheduleData.size(); ++machine) {
            for (const auto& entry : scheduleData[machine]) {
                outFile << machine << " " << entry.job << " " << entry.start << " " << entry.duration << "\n";
                std::cout << machine << " " << entry.job << " " << entry.start << " " << entry.duration << "\n";
            }
        }
        outFile.close();

        std::cout << "Best schedule data saved to " << filename << std::endl;
    }
};

#endif // JOB_SHOP_QLEARNING_H