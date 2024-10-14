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
#include "multidimensional_array.hpp"

constexpr size_t MAX_JOBS = 1000;
constexpr size_t MAX_MACHINES = 100;
constexpr size_t MAX_OPERATIONS = 1000;

struct Operation {
    int duration;
    int machine;
    std::bitset<MAX_MACHINES> eligibleMachines;
    std::vector<std::pair<int, int>> dependentOperations; // (jobIndex, opIndex)
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
    std::vector<bool> completedJobs;
    std::vector<int> jobStartTimes;

    State(int numJobs, int numMachines, int maxOperations)
            : jobProgress({static_cast<size_t>(numJobs), static_cast<size_t>(maxOperations)})
            , machineAvailability(numMachines, 0)
            , nextOperationForJob(numJobs, 0)
            , completedJobs(numJobs, false)
            , jobStartTimes(numJobs, -1) {
        jobProgress.fill(0);
    }
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
                for (int m = 0; m < numMachines; ++m) {
                    if (eligibleMachines[m]) {
                        actionLookup[i][j].emplace_back(i, m, j);
                    }
                }
            }
        }
    }

    bool isJobReady(int job) const {
        for (int depJob : jobs[job].dependentJobs) {
            if (!currentState->completedJobs[depJob]) {
                return false;
            }
        }
        return true;
    }

    bool isOperationReady(int jobIndex, int opIndex) const {
        const Operation& op = jobs[jobIndex].operations[opIndex];

        // Check if the job is ready
        if (!isJobReady(jobIndex)) {
            return false;
        }

        // Check if dependent operations are completed
        for (const auto& [depJob, depOp] : op.dependentOperations) {
            int depOpEndTime = currentState->jobProgress(depJob, depOp);
            if (depOpEndTime == 0) {
                return false;
            }
        }

        return true;
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
                                              [](const Job& a, const Job& b) { return a.operations.size() < b.operations.size(); })->operations.size();
        currentState = std::make_unique<State>(jobs.size(), numMachines, maxOperations);

        actionLookup.resize(jobs.size());
        for (size_t i = 0; i < jobs.size(); ++i) {
            actionLookup[i].resize(jobs[i].operations.size());
        }

        precomputeActions();
    }

    State& step(const Action& action) {
        const Operation& op = jobs[action.job].operations[action.operation];
        int operationTime = op.duration;
        int startTime = 0;

        // Determine the earliest possible start time based on machine availability
        startTime = std::max(startTime, currentState->machineAvailability[action.machine]);

        // Consider job dependencies
        for (int depJob : jobs[action.job].dependentJobs) {
            int depJobCompletionTime = currentState->jobProgress(depJob, jobs[depJob].operations.size() - 1);
            startTime = std::max(startTime, depJobCompletionTime);
        }

        // Check if previous operation in the job is completed
        if (action.operation > 0) {
            int prevOpEndTime = currentState->jobProgress(action.job, action.operation - 1);
            startTime = std::max(startTime, prevOpEndTime);
        }

        // Consider dependent operations within the job
        for (const auto& [depJob, depOp] : op.dependentOperations) {
            int depOpEndTime = currentState->jobProgress(depJob, depOp);
            startTime = std::max(startTime, depOpEndTime);
        }

        int endTime = startTime + operationTime;

        if (currentState->jobStartTimes[action.job] == -1) {
            currentState->jobStartTimes[action.job] = startTime;
        }

        currentState->jobProgress(action.job, action.operation) = endTime;
        currentState->machineAvailability[action.machine] = endTime;

        // Update next operation index if this operation is the next expected
        if (currentState->nextOperationForJob[action.job] == action.operation) {
            currentState->nextOperationForJob[action.job]++;
        }

        // Check if job is completed
        if (currentState->nextOperationForJob[action.job] == static_cast<int>(jobs[action.job].operations.size())) {
            currentState->completedJobs[action.job] = true;
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

    [[nodiscard]] const State& getState() const { return *currentState; }

    std::vector<Action>& getPossibleActions() {
        currentPossibleActions.clear();
        for (size_t i = 0; i < jobs.size(); ++i) {
            for (size_t opIndex = 0; opIndex < jobs[i].operations.size(); ++opIndex) {
                // Check if the operation is ready and not yet scheduled
                if (currentState->jobProgress(i, opIndex) == 0 && isOperationReady(i, opIndex)) {
                    const auto& actions = actionLookup[i][opIndex];
                    currentPossibleActions.insert(currentPossibleActions.end(), actions.begin(), actions.end());
                }
            }
        }
        return currentPossibleActions;
    }

    [[nodiscard]] bool isDone() const {
        return std::all_of(currentState->completedJobs.begin(), currentState->completedJobs.end(), [](bool completed) { return completed; });
    }

    [[nodiscard]] int getTotalTime() const { return totalTime; }
    [[nodiscard]] const std::vector<Job>& getJobs() const { return jobs; }
    [[nodiscard]] int getNumMachines() const { return numMachines; }

    [[nodiscard]] std::vector<std::vector<ScheduleEntry>> getScheduleData() const {
        std::vector<std::vector<ScheduleEntry>> scheduleData(numMachines);
        for (const Action& action : actionHistory) {
            int job = action.job;
            int operation = action.operation;
            int machine = action.machine;
            int duration = jobs[job].operations[operation].duration;
            int start = currentState->jobProgress(job, operation) - duration;
            scheduleData[machine].push_back({job, operation, start, duration});
        }
        return scheduleData;
    }

    void printSchedule() const {
        auto scheduleData = getScheduleData();
        std::cout << "Schedule:\n";
        for (size_t i = 0; i < scheduleData.size(); ++i) {
            std::cout << "Machine " << i << ": ";
            for (const auto& entry : scheduleData[i]) {
                std::cout << "(Job " << entry.job << ", Op " << entry.operation << ", Start: " << entry.start
                          << ", Duration: " << entry.duration << ") ";
            }
            std::cout << '\n';
        }
        std::cout << "Total time: " << totalTime << '\n';
    }

    void generateOperationGraph(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file for writing graph: " + filename);
        }

        file << "digraph JobShop {\n";
        file << "    rankdir=LR;\n"; // Left to right layout

        // Define node styles
        file << "    node [shape=rectangle, style=filled, fillcolor=lightgrey];\n";

        // Generate nodes and intra-job edges
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];

            for (size_t opIndex = 0; opIndex < job.operations.size(); ++opIndex) {
                const Operation& op = job.operations[opIndex];
                std::string nodeName = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex);
                file << "    " << nodeName << " [label=\"Job " << jobIndex << "\\nOp " << opIndex << "\\nDur: " << op.duration << "\"];\n";

                // Edge to next operation in the same job
                if (opIndex + 1 < job.operations.size()) {
                    std::string nextNodeName = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex + 1);
                    file << "    " << nodeName << " -> " << nextNodeName << " [style=bold, color=blue];\n";
                }
            }
        }

        // Add job dependencies
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            for (int depJobIndex : job.dependentJobs) {
                std::string fromNode = "J" + std::to_string(depJobIndex) + "_O" + std::to_string(jobs[depJobIndex].operations.size() - 1);
                std::string toNode = "J" + std::to_string(jobIndex) + "_O0";
                file << "    " << fromNode << " -> " << toNode << " [style=dashed, color=red, label=\"Job Dep\"];\n";
            }
        }

        // Add operation dependencies
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            for (size_t opIndex = 0; opIndex < job.operations.size(); ++opIndex) {
                const Operation& op = job.operations[opIndex];
                std::string fromNode = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex);
                for (const auto& [depJobIndex, depOpIndex] : op.dependentOperations) {
                    std::string toNode = "J" + std::to_string(depJobIndex) + "_O" + std::to_string(depOpIndex);
                    file << "    " << toNode << " -> " << fromNode << " [color=green, label=\"Op Dep\"];\n";
                }
            }
        }

        file << "}\n";
        file.close();
    }
};

#endif // JOB_SHOP_ENVIRONMENT_H
