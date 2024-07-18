#ifndef JOB_SHOP_ENVIRONMENT_H
#define JOB_SHOP_ENVIRONMENT_H

#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <bitset>
#include "multidimensional_array.hpp"


constexpr size_t MAX_JOBS = 1000;
constexpr size_t MAX_MACHINES = 100;
constexpr size_t MAX_OPERATIONS = 100;

struct Job {
    std::vector<int> operations;
    std::vector<int> machines;
    std::vector<std::bitset<MAX_MACHINES>> eligibleMachines;
};

struct Action {
    int job;
    int machine;
    int operation;
    Action(){}

    Action(int j, int m, int o) : job(j), machine(m), operation(o) {}
};

struct State {
    MultiDimensionalArray<int, 2> jobProgress;
    std::vector<int> machineAvailability;
    std::vector<int> nextOperationForJob;

    State(int numJobs, int numMachines, int maxOperations);
};

struct ScheduleEntry {
    int job;
    int start;
    int duration;
};

class JobShopEnvironment {
private:
    std::vector<Job> jobs;
    int numMachines;
    std::unique_ptr<State> currentState;
    int totalTime;
    std::vector<Action> actionHistory;
    std::array<std::array<std::vector<Action>, MAX_OPERATIONS>, MAX_JOBS> actionLookup;
    std::vector<Action> currentPossibleActions;


    void precomputeActions() {
        for (size_t i = 0; i < jobs.size(); ++i) {
            for (size_t j = 0; j < jobs[i].operations.size(); ++j) {
                const auto& eligibleMachines = jobs[i].eligibleMachines[j];
                for (int m = 0; m < numMachines; ++m) {
                    if (eligibleMachines[m]) {
                        actionLookup[i][j].emplace_back(i, m, j);
                    }
                }
            }
        }
    }

public:
    explicit JobShopEnvironment(std::vector<Job> j);

    State& step(const Action& action);
    void reset();
    [[nodiscard]] const State& getState() const { return *currentState; }
    std::vector<Action>& getPossibleActions();

    [[nodiscard]] bool isDone() const;

    [[nodiscard]] int getTotalTime() const { return totalTime; }
    [[nodiscard]] const std::vector<Job>& getJobs() const { return jobs; }
    [[nodiscard]] int getNumMachines() const { return numMachines; }
    [[nodiscard]] std::vector<std::vector<ScheduleEntry>> getScheduleData() const;
    void printSchedule() const;
};

#endif // JOB_SHOP_ENVIRONMENT_H