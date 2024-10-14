#ifndef JOB_SHOP_ENVIRONMENT_H
#define JOB_SHOP_ENVIRONMENT_H

#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <bitset>
#include <algorithm>
#include "multidimensional_array.hpp"

constexpr size_t MAX_JOBS = 1000;
constexpr size_t MAX_MACHINES = 100;
constexpr size_t MAX_OPERATIONS = 100;

struct Operation {
    int duration;
    int machine;
    std::bitset<MAX_MACHINES> eligibleMachines;
    std::vector<int> dependentOperations;
};

struct Job {
    std::vector<Operation> operations;
    std::vector<int> dependentJobs;
};

struct Action {
    int job;
    int machine;
    int operation;
    Action() {}
    Action(int j, int m, int o) : job(j), machine(m), operation(o) {}
};

struct State {
    MultiDimensionalArray<int, 2> jobProgress;
    std::vector<int> machineAvailability;
    std::vector<int> nextOperationForJob;
    std::vector<std::vector<bool>> completedOperations;

    State(int numJobs, int numMachines, int maxOperations);
};

struct ScheduleEntry {
    int job;
    int start;
    int duration;
};

class JobShopEnvironment {
private:
    int numMachines;
    int totalTime;
    std::vector<Job> jobs;
    std::unique_ptr<State> currentState;
    std::vector<Action> actionHistory;
    std::vector<std::vector<std::vector<Action>>> actionLookup;
    std::vector<Action> currentPossibleActions;

    void precomputeActions();
    bool isOperationReady(int job, int operation) const;

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