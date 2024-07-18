#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <effolkronium/random.hpp>
#include "job_shop_environment.h"

using Random = effolkronium::random_static;

class JobShopGenerator {
private:

    static int calculateOptimalMakespan(const std::vector<Job>& jobs, int numMachines) {
        std::vector<int> machineWorkload(numMachines, 0);
        std::vector<int> jobCompletionTimes(jobs.size(), 0);
        int maxJobWorkload = 0;

        for (size_t i = 0; i < jobs.size(); ++i) {
            int jobWorkload = 0;
            for (size_t j = 0; j < jobs[i].operations.size(); ++j) {
                int duration = jobs[i].operations[j];
                int machine = jobs[i].machines[j];

                jobWorkload += duration;
                machineWorkload[machine] += duration;

                // For jobs with machine limitations
                if (!jobs[i].eligibleMachines.empty()) {
                    jobCompletionTimes[i] = std::max(jobCompletionTimes[i], machineWorkload[machine]);
                }
            }
            maxJobWorkload = std::max(maxJobWorkload, jobWorkload);

            // For jobs without machine limitations
            if (jobs[i].eligibleMachines.empty()) {
                jobCompletionTimes[i] = jobWorkload;
            }
        }

        int maxMachineWorkload = *std::max_element(machineWorkload.begin(), machineWorkload.end());
        int maxJobCompletionTime = *std::max_element(jobCompletionTimes.begin(), jobCompletionTimes.end());

        return std::max({maxJobWorkload, maxMachineWorkload, maxJobCompletionTime});
    }


    JobShopGenerator() = default;
public:


    static std::pair<std::vector<Job>, int> generate(int numJobs, int numMachines, int minOperations, int maxOperations,
                                              int minDuration, int maxDuration) {
        std::vector<Job> jobs;
        jobs.reserve(numJobs);


        for (int i = 0; i < numJobs; ++i) {
            int numOperations = Random::get<int>(minOperations, maxOperations);
            std::vector<int> operations(numOperations);
            std::vector<int> machines(numOperations);

            for (int j = 0; j < numOperations; ++j) {
                operations[j] = Random::get<int>(minDuration, maxDuration);
                machines[j] = Random::get<int>(0, numMachines - 1);
            }

            jobs.push_back({std::move(operations), std::move(machines)});
        }

        int optimalMakespan = calculateOptimalMakespan(jobs, numMachines);
        return {jobs, optimalMakespan};
    }

    static std::pair<std::vector<Job>, int> generateWithMachineLimitations(int numJobs, int numMachines, int minOperations, int maxOperations,
                                                                           int minDuration, int maxDuration, double machineEligibilityRate) {
        std::vector<Job> jobs;
        jobs.reserve(numJobs);

        for (int i = 0; i < numJobs; ++i) {
            int numOperations = Random::get<int>(minOperations, maxOperations);
            std::vector<int> operations(numOperations);
            std::vector<int> machines(numOperations);
            std::vector<std::bitset<MAX_MACHINES>> eligibleMachines(numOperations);

            for (int j = 0; j < numOperations; ++j) {
                operations[j] = Random::get<int>(minDuration, maxDuration);

                // Generate eligible machines for this operation
                for (int m = 0; m < numMachines; ++m) {
                    if (Random::get<double>(0.0, 1.0) < machineEligibilityRate) {
                        eligibleMachines[j].set(m);
                    }
                }

                // Ensure at least one machine is eligible
                if (eligibleMachines[j].none()) {
                    int randomMachine = Random::get<int>(0, numMachines - 1);
                    eligibleMachines[j].set(randomMachine);
                }

                // Assign a random eligible machine
                int randomEligibleMachine;
                do {
                    randomEligibleMachine = Random::get<int>(0, numMachines - 1);
                } while (!eligibleMachines[j].test(randomEligibleMachine));

                machines[j] = randomEligibleMachine;
            }

            jobs.push_back(Job{std::move(operations), std::move(machines), std::move(eligibleMachines)});
        }

        int optimalMakespan = calculateOptimalMakespan(jobs, numMachines);
        return {jobs, optimalMakespan};
    }


};