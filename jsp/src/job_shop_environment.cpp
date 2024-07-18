#include "job_shop_environment.h"
#include <algorithm>
#include <iostream>
State::State(int numJobs, int numMachines, int maxOperations)
        : jobProgress({static_cast<size_t>(numJobs), static_cast<size_t>(maxOperations)})
        , machineAvailability(numMachines, 0)
        , nextOperationForJob(numJobs, 0) {
    jobProgress.fill(0);
}

JobShopEnvironment::JobShopEnvironment(std::vector<Job> j)
        : jobs(std::move(j))
        , totalTime(0) {
    if (jobs.empty()) {
        throw std::invalid_argument("Jobs vector cannot be empty");
    }
    numMachines = jobs[0].machines.size();
    if (numMachines == 0 || numMachines > MAX_MACHINES) {
        throw std::invalid_argument("Invalid number of machines");
    }
    auto maxOperations = std::max_element(jobs.begin(), jobs.end(),
                                          [](const Job& a, const Job& b) { return a.operations.size() < b.operations.size(); })->operations.size();
    currentState = std::make_unique<State>(jobs.size(), numMachines, maxOperations);

    currentPossibleActions.reserve(10000);
    currentPossibleActions.resize(10000);

    precomputeActions();
}



State& JobShopEnvironment::step(const Action& action) {
    int operationTime = jobs[action.job].operations[action.operation];
    int startTime = std::max(currentState->machineAvailability[action.machine],
                             (action.operation > 0) ? currentState->jobProgress(action.job, action.operation - 1) : 0);
    int endTime = startTime + operationTime;

    currentState->jobProgress(action.job, action.operation) = endTime;
    currentState->machineAvailability[action.machine] = endTime;
    currentState->nextOperationForJob[action.job]++;
    totalTime = std::max(totalTime, endTime);

    actionHistory.push_back(action);
    return *currentState;
}

void JobShopEnvironment::reset() {
    currentState->jobProgress.fill(0);
    std::fill(currentState->machineAvailability.begin(), currentState->machineAvailability.end(), 0);
    std::fill(currentState->nextOperationForJob.begin(), currentState->nextOperationForJob.end(), 0);
    totalTime = 0;
    actionHistory.clear();

}

std::vector<std::vector<ScheduleEntry>> JobShopEnvironment::getScheduleData() const {
    std::vector<std::vector<ScheduleEntry>> scheduleData(numMachines);
    for (const Action& action : actionHistory) {
        int job = action.job;
        int machine = action.machine;
        int operation = action.operation;
        int duration = jobs[job].operations[operation];
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

bool JobShopEnvironment::isDone() const {
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (currentState->nextOperationForJob[i] < static_cast<int>(jobs[i].operations.size())) {
            return false;  // If any job has remaining operations, we're not done
        }
    }
    return true;  // All jobs have completed all operations
}

std::vector<Action>&JobShopEnvironment::getPossibleActions() {

    int n = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        int nextOp = currentState->nextOperationForJob[i];
        if (nextOp < static_cast<int>(jobs[i].operations.size())) {
            const auto& actions = actionLookup[i][nextOp];

            for(auto& ac : actions){
                currentPossibleActions[n].job = ac.job;
                currentPossibleActions[n].machine = ac.machine;
                currentPossibleActions[n].operation = ac.operation;
                n++;
            }
        }
    }
    currentPossibleActions.resize(n);
    return currentPossibleActions;
}
