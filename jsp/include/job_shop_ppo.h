// job_shop_ppo.cpp

#ifndef JOB_SHOP_PPO_H
#define JOB_SHOP_PPO_H

#include "job_shop_environment.h"
#include "job_shop_algorithms.h"
#include <random>
#include <vector>
#include <functional>
#include <array>
#include <algorithm>
#include <iostream>
#include <limits>
#include <fstream>
#include <cmath>
#include <numeric>


// Structure to store experiences
struct Experience {
    std::vector<double> state;
    Action action;
    double reward;
    std::vector<double> next_state;
    bool done;
    double old_log_prob;
};

// JobShopPPO class definition and implementation
class JobShopPPO : public JobShopAlgorithm {
private:
    JobShopEnvironment& env;
    std::vector<Experience> buffer;
    double learningRate;
    double discountFactor;
    double epsilon;
    double clipRange;
    std::mt19937 rng;
    int bestTime;  // Holds the best time found
    std::vector<Action> bestSchedule;

    // Neural Network placeholders
    MultiDimensionalArray<double, 2> actorWeights;
    MultiDimensionalArray<double, 2> criticWeights;

    // Input and output sizes
    size_t inputSize;
    size_t outputSize;

    std::pair<size_t, size_t> calculateInputOutputSizes() {
        const auto& jobs = env.getJobs();
        size_t maxOperations = std::max_element(jobs.begin(), jobs.end(),
            [](const Job& a, const Job& b) { return a.operations.size() < b.operations.size(); }
        )->operations.size();

        size_t inputSize = jobs.size() * env.getNumMachines() * maxOperations;
        size_t outputSize = env.getNumMachines(); // Assuming the output is the choice of machine

        return {inputSize, outputSize};
    }

    void initializeWeights(std::mt19937& rng, double minVal, double maxVal) {
        std::uniform_real_distribution<double> distribution(minVal, maxVal);

        for (size_t i = 0; i < actorWeights.getDimensions()[0]; ++i) {
            for (size_t j = 0; j < actorWeights.getDimensions()[1]; ++j) {
                actorWeights(i, j) = distribution(rng);
            }
        }

        for (size_t i = 0; i < criticWeights.getDimensions()[0]; ++i) {
            for (size_t j = 0; j < criticWeights.getDimensions()[1]; ++j) {
                criticWeights(i, j) = distribution(rng);
            }
        }
    }

    Action selectAction(const State& state){
        std::vector<double> stateVec = stateToVector(state);
        std::vector<double> actionProbs(env.getNumMachines());

        for (size_t i = 0; i < env.getNumMachines(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < stateVec.size(); ++j) {
                sum += stateVec[j] * actorWeights(j, i);
            }
            actionProbs[i] = std::exp(sum);
        }

        double sumProbs = std::accumulate(actionProbs.begin(), actionProbs.end(), 0.0);
        for (auto& prob : actionProbs) {
            prob /= sumProbs;
        }

        std::vector<Action> possibleActions = env.getPossibleActions();
        std::vector<double> filteredProbs;
        for (const auto& action : possibleActions) {
            filteredProbs.push_back(actionProbs[action.machine]);
        }
        
        std::discrete_distribution<> dist(filteredProbs.begin(), filteredProbs.end());
        int actionIndex = dist(rng);
        
        return possibleActions[actionIndex];

    }
    
    void update(const std::vector<Experience>& buffer){
        std::vector<double> advantages = computeAdvantages(buffer);
        for (int epoch = 0; epoch < 10; ++epoch) {  // Number of PPO epochs
            for (size_t i = 0; i < buffer.size(); ++i) {
                const auto& experience = buffer[i];
                std::vector<double> actionProbs(env.getNumMachines());

                // Forward pass for actor
                for (size_t j = 0; j < env.getNumMachines(); ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < experience.state.size(); ++k) {
                        sum += experience.state[k] * actorWeights(k, j);
                    }
                    actionProbs[j] = std::exp(sum);
                }
            
                // Normalize probabilities
                double sumProbs = std::accumulate(actionProbs.begin(), actionProbs.end(), 0.0);
                for (auto& prob : actionProbs) {
                    prob /= sumProbs;
                }

                double newLogProb = std::log(actionProbs[experience.action.machine]);
                double ratio = std::exp(newLogProb - experience.old_log_prob);
            
                double surrogate1 = ratio * advantages[i];
                double surrogate2 = std::clamp(ratio, 1.0 - clipRange, 1.0 + clipRange) * advantages[i];
            
                double actorLoss = -std::min(surrogate1, surrogate2);
            
                // Forward pass for critic
                double value = 0.0;
                for (size_t j = 0; j < experience.state.size(); ++j) {
                    value += experience.state[j] * criticWeights(j, 0);
                }
            
                double criticLoss = std::pow(value - (experience.reward + discountFactor * value), 2);
            
                // Update actor weights (gradient ascent)
                for (size_t j = 0; j < actorWeights.getDimensions()[1]; ++j) {
                    actorWeights(j, experience.action.machine) -= learningRate * actorLoss * experience.state[j];
                }
            
                // Update critic weights (gradient descent)
                for (size_t j = 0; j < criticWeights.getDimensions()[1]; ++j) {
                    criticWeights(j, 0) -= learningRate * criticLoss * experience.state[j];
                }
            }
        }   
    }

    std::vector<double> stateToVector(const State& state) const{
        std::vector<double> stateVector;
        const auto& jobs = env.getJobs();
        size_t maxOperations = std::max_element(jobs.begin(), jobs.end(),
            [](const Job& a, const Job& b) { return a.operations.size() < b.operations.size(); }
        )->operations.size();

        for (size_t i = 0; i < jobs.size(); ++i) {
            for (size_t j = 0; j < env.getNumMachines(); ++j) {
                for (size_t k = 0; k < maxOperations; ++k) {
                    if (k < state.jobProgress.getDimensions()[1]) {
                        stateVector.push_back(static_cast<double>(state.jobProgress(i, k)));
                    } else {
                        stateVector.push_back(0.0);  // Padding for jobs with fewer operations
                    }
                }
            }
        }

        return stateVector;
    }

    std::vector<double> actionToVector(const Action& action) const{
        std::vector<double> actionVector(env.getNumMachines(), 0.0);
        actionVector[action.machine] = 1.0;
        return actionVector;
    }
    
    std::vector<double> computeAdvantages(const std::vector<Experience>& experiences) const{
        std::vector<double> advantages(experiences.size());
        double nextValue = 0;
    
        for (int i = experiences.size() - 1; i >= 0; --i) {
            double value = 0.0;
            for (size_t j = 0; j < experiences[i].state.size(); ++j) {
                value += experiences[i].state[j] * criticWeights(j, 0);
            }
            double tdError = experiences[i].reward + discountFactor * nextValue - value;
            advantages[i] = tdError;
            nextValue = value;
        }
    
        return advantages;
    }

    int evaluateEpisode(const std::vector<Action>& actions){
        env.reset();
        for (const Action& action : actions) {
            env.step(action);
        }
        return env.getTotalTime();
    }
    
    // Simple feed-forward for neural network
    std::vector<double> forwardPass(const std::vector<double>& input, const MultiDimensionalArray<double, 2>& weights) const{
        std::vector<double> output(weights.getDimensions()[1]);
        for (size_t i = 0; i < weights.getDimensions()[1]; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < input.size(); ++j) {
                sum += input[j] * weights(j, i);
            }
            output[i] = sum;
        }
        return output;
    }

public:
    JobShopPPO(JobShopEnvironment& environment, double alpha, double gamma, double epsilon, double clipRange)
        : env(environment),
          learningRate(alpha),
          discountFactor(gamma),
          epsilon(epsilon),
          clipRange(clipRange),
          rng(std::random_device{}()),
          bestTime(std::numeric_limits<int>::max()),
          actorWeights(std::array<std::size_t, 2>{calculateInputOutputSizes().first, calculateInputOutputSizes().second}),
          criticWeights(std::array<std::size_t, 2>{calculateInputOutputSizes().first, 1})
          {
            // Initialize weights
            initializeWeights(rng, -0.1, 0.1);
          }

    std::vector<Action> runEpisode(){
        env.reset();
        std::vector<Action> schedule;
        buffer.clear();

        while (!env.isDone()) {
            State currentState = env.getState();
            Action action = selectAction(currentState);

            if (action.job == -1 && action.machine == -1 && action.operation == -1) {
                break;  // No more actions available
            }

            schedule.push_back(action);
            State nextState = env.step(action);
            double reward = -static_cast<double>(env.getTotalTime());

            std::vector<double> stateVec = stateToVector(currentState);
            std::vector<double> nextStateVec = stateToVector(nextState);

            std::vector<double> actionProbs(env.getNumMachines());
            for (size_t i = 0; i < env.getNumMachines(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < stateVec.size(); ++j) {
                    sum += stateVec[j] * actorWeights(j, i);
                }
                actionProbs[i] = std::exp(sum);
            }
            
            double sumProbs = std::accumulate(actionProbs.begin(), actionProbs.end(), 0.0);
            for (auto& prob : actionProbs) {
                prob /= sumProbs;
            }

            double oldLogProb = std::log(actionProbs[action.machine]);

            buffer.push_back({stateVec, action, reward, nextStateVec, env.isDone(), oldLogProb});
        }

        update(buffer);
        return schedule;
    }

    void train(int numEpisodes, const std::function<void(int)>& episodeCallback = [](int){}){
        for (int i = 0; i < numEpisodes; ++i) {
            std::vector<Action> episode = runEpisode();
            int episodeTime = evaluateEpisode(episode);

            if (episodeTime < bestTime) {
                bestTime = episodeTime;
                bestSchedule = std::move(episode);
                std::cout << "Episode " << i << ", New best time: " << bestTime << std::endl;
            }

            epsilon *= 0.9999;  // Decay exploration rate
            episodeCallback(episodeTime);
        }

        std::cout << "Training completed. Best time: " << bestTime << std::endl;
        applyAndPrintSchedule(bestSchedule);

    }

    void printBestSchedule(){
        std::cout << "Best Schedule (Total time: " << bestTime << "):" << std::endl;
        applyAndPrintSchedule(bestSchedule);
        
    }
    void saveBestScheduleToFile(const std::string& filename){
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

    void applyAndPrintSchedule(const std::vector<Action>& schedule){
        env.reset();
        for (const Action& action : schedule) {
            env.step(action);
        }
        env.printSchedule();
    }
};

#endif // JOB_SHOP_PPO_H

