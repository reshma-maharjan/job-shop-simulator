#ifndef JOB_SHOP_ENVIRONMENT_H
#define JOB_SHOP_ENVIRONMENT_H

#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <bitset>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <span>
#include "utilities/multidimensional_array.hpp"

constexpr size_t MAX_MACHINES = 100;

// Forward declarations
class JobShopEnvironment;

struct Operation {
    int duration;
    int machine;
    std::bitset<MAX_MACHINES> eligibleMachines;
    // Use small_vector or reserve space for common case to avoid allocations
    std::vector<std::pair<int, int>> dependentOperations; // (jobIndex, opIndex)

    Operation() = default;
    Operation(int d, int m) noexcept : duration(d), machine(m) {
        dependentOperations.reserve(4); // Reserve space for typical use case
    }
};

struct Job {
    std::vector<Operation> operations;
    std::vector<int> dependentJobs;

    Job() {
        operations.reserve(8);    // Reserve typical size
        dependentJobs.reserve(4); // Reserve typical size
    }

    Job(std::vector<Operation> ops, std::vector<int> deps)
            : operations(std::move(ops)), dependentJobs(std::move(deps)) {}
};

struct Action {
    int job{};
    int machine{};
    int operation{};

    constexpr Action() noexcept = default;
    constexpr Action(int j, int m, int o) noexcept : job(j), machine(m), operation(o) {}

    // Add comparison operators for std::algorithm usage
    constexpr bool operator==(const Action& other) const noexcept {
        return job == other.job && machine == other.machine && operation == other.operation;
    }
};

class State {


public:
    std::vector<int> nextOperationForJob;
    std::vector<bool> completedJobs;
    std::vector<int> jobStartTimes;
    explicit State(int numJobs, int numMachines, int maxOperations)
            : jobProgress({static_cast<size_t>(numJobs), static_cast<size_t>(maxOperations)})
            , machineAvailability(numMachines, 0)
            , nextOperationForJob(numJobs, 0)
            , completedJobs(numJobs, false)
            , jobStartTimes(numJobs, -1) {
        jobProgress.fill(0);
    }

    // Add const accessors
    [[nodiscard]] const auto& getJobProgress() const noexcept { return jobProgress; }
    [[nodiscard]] const auto& getMachineAvailability() const noexcept { return machineAvailability; }
    [[nodiscard]] const auto& getNextOperationForJob() const noexcept { return nextOperationForJob; }
    [[nodiscard]] const auto& getCompletedJobs() const noexcept { return completedJobs; }
    [[nodiscard]] const auto& getJobStartTimes() const noexcept { return jobStartTimes; }

    // Add non-const accessors for JobShopEnvironment
    friend class JobShopEnvironment;

    std::vector<int> machineAvailability;
    MultiDimensionalArray<int, 2> jobProgress;
};

struct ScheduleEntry {
    int job;
    int operation;
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

    void precomputeActions() {
        for (size_t i = 0; i < jobs.size(); ++i) {
            for (size_t j = 0; j < jobs[i].operations.size(); ++j) {
                const auto& eligibleMachines = jobs[i].operations[j].eligibleMachines;
                actionLookup[i][j].reserve(numMachines); // Preallocate for worst case
                for (int m = 0; m < numMachines; ++m) {
                    if (eligibleMachines[m]) {
                        actionLookup[i][j].emplace_back(i, m, j);
                    }
                }
            }
        }
    }

    [[nodiscard]] bool isJobReady(int job) const noexcept {
        const auto& deps = jobs[job].dependentJobs;
        return std::all_of(deps.begin(), deps.end(),
                           [this](int depJob) { return currentState->completedJobs[depJob]; });
    }

    [[nodiscard]] bool isOperationReady(int jobIndex, int opIndex) const noexcept {
        if (!isJobReady(jobIndex)) {
            return false;
        }

        const Operation& op = jobs[jobIndex].operations[opIndex];
        return std::all_of(op.dependentOperations.begin(), op.dependentOperations.end(),
                           [this](const auto& dep) {
                               return currentState->jobProgress(dep.first, dep.second) != 0;
                           });
    }

public:
    explicit JobShopEnvironment(std::vector<Job> j)
            : jobs(std::move(j))
            , totalTime(0) {
        if (jobs.empty()) {
            throw std::invalid_argument("Jobs vector cannot be empty");
        }

        numMachines = 0;
        for (const auto& job : jobs) {
            for (const auto& op : job.operations) {
                numMachines = std::max(numMachines, op.machine + 1);
            }
        }

        if (numMachines == 0 || numMachines > MAX_MACHINES) {
            throw std::invalid_argument("Invalid number of machines");
        }

        auto maxOperations = std::max_element(jobs.begin(), jobs.end(),
                                              [](const Job& a, const Job& b) {
                                                  return a.operations.size() < b.operations.size();
                                              })->operations.size();

        currentState = std::make_unique<State>(jobs.size(), numMachines, maxOperations);

        // Preallocate vectors
        actionHistory.reserve(jobs.size() * maxOperations);
        currentPossibleActions.reserve(numMachines * 2);

        actionLookup.resize(jobs.size());
        for (size_t i = 0; i < jobs.size(); ++i) {
            actionLookup[i].resize(jobs[i].operations.size());
        }

        precomputeActions();
    }

    State& step(const Action& action) {
        const Operation& op = jobs[action.job].operations[action.operation];
        int startTime = currentState->machineAvailability[action.machine];

        // Consider job dependencies
        for (int depJob : jobs[action.job].dependentJobs) {
            startTime = std::max(startTime,
                                 currentState->jobProgress(depJob, jobs[depJob].operations.size() - 1));
        }

        // Check if previous operation in the job is completed
        if (action.operation > 0) {
            startTime = std::max(startTime,
                                 currentState->jobProgress(action.job, action.operation - 1));
        }

        // Consider dependent operations within the job
        for (const auto& [depJob, depOp] : op.dependentOperations) {
            startTime = std::max(startTime,
                                 currentState->jobProgress(depJob, depOp));
        }

        int endTime = startTime + op.duration;

        if (currentState->jobStartTimes[action.job] == -1) {
            currentState->jobStartTimes[action.job] = startTime;
        }

        currentState->jobProgress(action.job, action.operation) = endTime;
        currentState->machineAvailability[action.machine] = endTime;

        if (currentState->nextOperationForJob[action.job] == action.operation) {
            currentState->nextOperationForJob[action.job]++;

            // Check if job is completed
            if (currentState->nextOperationForJob[action.job] ==
                static_cast<int>(jobs[action.job].operations.size())) {
                currentState->completedJobs[action.job] = true;
            }
        }

        totalTime = std::max(totalTime, endTime);
        actionHistory.push_back(action);

        return *currentState;
    }

    void reset() {
        currentState->jobProgress.fill(0);
        std::fill(currentState->machineAvailability.begin(), currentState->machineAvailability.end(), 0);
        std::fill(currentState->nextOperationForJob.begin(), currentState->nextOperationForJob.end(), 0);
        std::fill(currentState->completedJobs.begin(), currentState->completedJobs.end(), false);
        std::fill(currentState->jobStartTimes.begin(), currentState->jobStartTimes.end(), -1);
        totalTime = 0;
        actionHistory.clear();
    }

    [[nodiscard]] const State& getState() const noexcept { return *currentState; }

    std::vector<Action>& getPossibleActions() {
        currentPossibleActions.clear();

        for (size_t i = 0; i < jobs.size(); ++i) {
            if (currentState->completedJobs[i]) continue;

            const size_t opIndex = currentState->nextOperationForJob[i];
            if (opIndex >= jobs[i].operations.size()) continue;

            if (currentState->jobProgress(i, opIndex) == 0 && isOperationReady(i, opIndex)) {
                const auto& actions = actionLookup[i][opIndex];
                currentPossibleActions.insert(
                        currentPossibleActions.end(),
                        actions.begin(),
                        actions.end()
                );
            }
        }

        return currentPossibleActions;
    }

    [[nodiscard]] bool isDone() const noexcept {
        return std::all_of(
                currentState->completedJobs.begin(),
                currentState->completedJobs.end(),
                [](bool completed) { return completed; }
        );
    }

    [[nodiscard]] constexpr int getTotalTime() const noexcept { return totalTime; }
    [[nodiscard]] const std::vector<Job>& getJobs() const noexcept { return jobs; }
    [[nodiscard]] constexpr int getNumMachines() const noexcept { return numMachines; }

    [[nodiscard]] std::vector<std::vector<ScheduleEntry>> getScheduleData() const {
        std::vector<std::vector<ScheduleEntry>> scheduleData(numMachines);
        for (size_t i = 0; i < numMachines; ++i) {
            scheduleData[i].reserve(actionHistory.size() / numMachines);
        }

        for (const Action& action : actionHistory) {
            const Operation& op = jobs[action.job].operations[action.operation];
            int end = currentState->jobProgress(action.job, action.operation);
            scheduleData[action.machine].push_back({
                                                           action.job,
                                                           action.operation,
                                                           end - op.duration,
                                                           op.duration
                                                   });
        }
        return scheduleData;
    }

    void printSchedule() const {
        const auto scheduleData = getScheduleData();
        std::cout << "Schedule:\n";
        for (size_t i = 0; i < scheduleData.size(); ++i) {
            std::cout << "Machine " << i << ": ";
            for (const auto& entry : scheduleData[i]) {
                std::cout << "(Job " << entry.job
                          << ", Op " << entry.operation
                          << ", Start: " << entry.start
                          << ", Duration: " << entry.duration << ") ";
            }
            std::cout << '\n';
        }
        std::cout << "Total time: " << totalTime << '\n';
    }

    void generateOperationGraph(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Unable to open file for writing graph: " + filename);
        }

        file << "digraph JobShop {\n"
             << "    rankdir=LR;\n"
             << "    node [shape=rectangle, style=filled, fillcolor=lightgrey];\n";

        // Generate nodes and intra-job edges
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];

            for (size_t opIndex = 0; opIndex < job.operations.size(); ++opIndex) {
                const Operation& op = job.operations[opIndex];
                const std::string nodeName = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex);

                file << "    " << nodeName << " [label=\"Job " << jobIndex
                     << "\\nOp " << opIndex << "\\nDur: " << op.duration << "\"];\n";

                // Edge to next operation in the same job
                if (opIndex + 1 < job.operations.size()) {
                    file << "    " << nodeName << " -> J" << jobIndex << "_O"
                         << (opIndex + 1) << " [style=bold, color=blue];\n";
                }

                // Add operation dependencies
                for (const auto& [depJobIndex, depOpIndex] : op.dependentOperations) {
                    file << "    J" << depJobIndex << "_O" << depOpIndex
                         << " -> " << nodeName << " [color=green, label=\"Op Dep\"];\n";
                }
            }

            // Add job dependencies
            for (int depJobIndex : job.dependentJobs) {
                file << "    J" << depJobIndex << "_O"
                     << (jobs[depJobIndex].operations.size() - 1)
                     << " -> J" << jobIndex
                     << "_O0 [style=dashed, color=red, label=\"Job Dep\"];\n";
            }
        }

        file << "}\n";
    }
};

#endif // JOB_SHOP_ENVIRONMENT_H