#include "job_shop_environment.h"
#include <iostream>

State::State(int numJobs, int numMachines, int maxOperations)
        : jobProgress({static_cast<size_t>(numJobs), static_cast<size_t>(maxOperations)})
        , machineAvailability(numMachines, 0)
        , nextOperationForJob(numJobs, 0)
        , completedOperations(numJobs, std::vector<bool>(maxOperations, false)) {
    jobProgress.fill(0);
}

JobShopEnvironment::JobShopEnvironment(std::vector<Job> j)
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

void JobShopEnvironment::precomputeActions() {
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

bool JobShopEnvironment::isOperationReady(int job, int operation) const {
    // Check if all dependent operations are completed
    for (int depOp : jobs[job].operations[operation].dependentOperations) {
        if (!currentState->completedOperations[job][depOp]) {
            return false;
        }
    }

    // Check if all dependent jobs are completed
    for (int depJob : jobs[job].dependentJobs) {
        if (currentState->nextOperationForJob[depJob] < static_cast<int>(jobs[depJob].operations.size())) {
            return false;
        }
    }

    return true;
}

State& JobShopEnvironment::step(const Action& action) {
    const Operation& op = jobs[action.job].operations[action.operation];
    int operationTime = op.duration;
    int startTime = std::max(currentState->machineAvailability[action.machine],
                             (action.operation > 0) ? currentState->jobProgress(action.job, action.operation - 1) : 0);
    int endTime = startTime + operationTime;

    currentState->jobProgress(action.job, action.operation) = endTime;
    currentState->machineAvailability[action.machine] = endTime;
    currentState->nextOperationForJob[action.job]++;
    currentState->completedOperations[action.job][action.operation] = true;
    totalTime = std::max(totalTime, endTime);

    actionHistory.push_back(action);
    return *currentState;
}

void JobShopEnvironment::reset() {
    currentState->jobProgress.fill(0);
    std::fill(currentState->machineAvailability.begin(), currentState->machineAvailability.end(), 0);
    std::fill(currentState->nextOperationForJob.begin(), currentState->nextOperationForJob.end(), 0);
    for (auto& jobOps : currentState->completedOperations) {
        std::fill(jobOps.begin(), jobOps.end(), false);
    }
    totalTime = 0;
    actionHistory.clear();
}

std::vector<Action>& JobShopEnvironment::getPossibleActions() {
    currentPossibleActions.clear();
    for (size_t i = 0; i < jobs.size(); ++i) {
        int nextOp = currentState->nextOperationForJob[i];
        if (nextOp < static_cast<int>(jobs[i].operations.size()) && isOperationReady(i, nextOp)) {
            const auto& actions = actionLookup[i][nextOp];
            currentPossibleActions.insert(currentPossibleActions.end(), actions.begin(), actions.end());
        }
    }
    return currentPossibleActions;
}

bool JobShopEnvironment::isDone() const {
    return std::all_of(currentState->nextOperationForJob.begin(), currentState->nextOperationForJob.end(),
                       [this](int nextOp) { return nextOp == static_cast<int>(jobs[nextOp].operations.size()); });
}

std::vector<std::vector<ScheduleEntry>> JobShopEnvironment::getScheduleData() const {
    std::vector<std::vector<ScheduleEntry>> scheduleData(numMachines);
    for (const Action& action : actionHistory) {
        int job = action.job;
        int machine = action.machine;
        int operation = action.operation;
        int duration = jobs[job].operations[operation].duration;
        int start = currentState->jobProgress(job, operation) - duration;
        scheduleData[machine].push_back({job, start, duration});
    }
    return scheduleData;
}

void JobShopEnvironment::printSchedule() const {
    auto scheduleData = getScheduleData();
    std::cout << "Schedule:\n";
    for (size_t i = 0; i < scheduleData.size(); ++i) {
        std::cout << "Machine " << i << ": ";
        for (const auto& entry : scheduleData[i]) {
            std::cout << "(Job " << entry.job << ", Start: " << entry.start
                      << ", Duration: " << entry.duration << ") ";
        }
        std::cout << '\n';
    }
    std::cout << "Total time: " << totalTime << '\n';
}