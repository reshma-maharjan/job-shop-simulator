#ifndef JOB_SHOP_ACTOR_CRITIC_H
#define JOB_SHOP_ACTOR_CRITIC_H

#include "job_shop_environment.h"
#include "job_shop_algorithms.h"
#include <random>
#include <unordered_map>
#include <functional>
#include <array>

class JobShopActorCritic : public JobShopAlgorithm{
private:
    JobShopEnvironment& env;
    MultiDimensionalArray<double, 3> valueTable;  // Critic's value function table
    MultiDimensionalArray<double, 3> policyTable; // Actor's policy function table
    double learningRate;
    double discountFactor;
    double explorationRate;
    std::mt19937 rng;
    int bestTime;
    std::vector<Action> bestSchedule;

    Action getAction(const State& state);
    void updateValueFunction(const State& state, const Action& action, double reward, const State& nextState);
    void updatePolicy(const State& state, const Action& action, double tdError);
    int evaluateEpisode(const std::vector<Action>& actions);
    double calculatePriority(const Action& action, const State& state) const;

public:
    JobShopActorCritic(JobShopEnvironment& environment, double alpha, double gamma, double epsilon);
    std::vector<Action> runEpisode();
    void train(int numEpisodes, const std::function<void(int)>& episodeCallback = [](int){});
    void printBestSchedule();
    void saveBestScheduleToFile(const std::string &filename);
    void applyAndPrintSchedule(const std::vector<Action> &schedule);

    std::array<size_t, 3> initializeTableDimensions();
};

#endif // JOB_SHOP_ACTOR_CRITIC_H
