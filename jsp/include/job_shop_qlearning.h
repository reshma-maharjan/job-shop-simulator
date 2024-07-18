#ifndef JOB_SHOP_QLEARNING_H
#define JOB_SHOP_QLEARNING_H

#include "job_shop_environment.h"
#include "job_shop_algorithms.h"
#include <random>
#include <unordered_map>

class JobShopQLearning: public JobShopAlgorithm{
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


    Action getAction(const State& state);
    void updateQValue(const State& state, const Action& action, double reward, const State& nextState);
    int evaluateEpisode(const std::vector<Action>& actions);
    double calculatePriority(const Action& action, const State& state) const;

public:
    JobShopQLearning(JobShopEnvironment& environment, double alpha, double gamma, double epsilon);
    std::vector<Action> runEpisode();
    void train(int numEpisodes, const std::function<void(int)>& episodeCallback = [](int){});
    void printBestSchedule();
    void saveBestScheduleToFile(const std::string &filename);
    void applyAndPrintSchedule(const std::vector<Action> &schedule);

    std::array<size_t, 3> initializeQTableDimensions();
};

#endif // JOB_SHOP_QLEARNING_H