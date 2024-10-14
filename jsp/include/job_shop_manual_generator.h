#pragma once

#include <fstream>

class ManualJobShopGenerator {
private:
    static std::pair<std::vector<std::vector<int>>, std::vector<int>> loadMatrixFromFile(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<std::vector<int>> dependencyMatrix;
        std::vector<int> timeMatrix;
        std::string line;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<int> row;
            std::string value;

            while (std::getline(iss, value, ',')) {
                row.push_back(std::stoi(value));
            }

            if (file.peek() == EOF) {
                timeMatrix = row;
            } else {
                dependencyMatrix.push_back(row);
            }
        }

        return {dependencyMatrix, timeMatrix};
    }

    static int calculateOptimalMakespan(const std::vector<Job>& jobs) {
        int numJobs = jobs.size();
        int numMachines = *std::max_element(jobs[0].machines.begin(), jobs[0].machines.end()) + 1;
        std::vector<int> jobCompletionTimes(numJobs, 0);
        std::vector<int> machineAvailability(numMachines, 0);

        for (int i = 0; i < numJobs; ++i) {
            for (size_t j = 0; j < jobs[i].operations.size(); ++j) {
                int machine = jobs[i].machines[j];
                int duration = jobs[i].operations[j];
                int startTime = std::max(jobCompletionTimes[i], machineAvailability[machine]);
                int endTime = startTime + duration;

                jobCompletionTimes[i] = endTime;
                machineAvailability[machine] = endTime;
            }
        }

        return *std::max_element(jobCompletionTimes.begin(), jobCompletionTimes.end());
    }

public:
    static std::pair<std::vector<Job>, int> generateFromFile(const std::string& filename) {
        auto [dependencyMatrix, timeMatrix] = loadMatrixFromFile(filename);

        int numJobs = dependencyMatrix.size();
        int numOperations = dependencyMatrix[0].size();

        std::vector<Job> jobs(numJobs);

        for (int i = 0; i < numJobs; ++i) {
            for (int j = 0; j < numOperations; ++j) {
                int duration = timeMatrix[j];
                if (duration > 0) {
                    jobs[i].operations.push_back(duration);
                    jobs[i].machines.push_back(j);

                    std::bitset<MAX_MACHINES> eligibleMachines;
                    eligibleMachines.set(j);
                    jobs[i].eligibleMachines.push_back(eligibleMachines);
                }
            }
        }

        // Process job dependencies
        for (int i = 0; i < numJobs; ++i) {
            for (int j = 0; j < numOperations; ++j) {
                if (dependencyMatrix[i][j] == 1) {
                    jobs[i].jobDependencies.push_back(&jobs[j]);
                }
            }
        }

        int optimalMakespan = calculateOptimalMakespan(jobs);
        return {jobs, optimalMakespan};
    }
};