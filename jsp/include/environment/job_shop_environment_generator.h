#pragma once

#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <optional>
#include <effolkronium/random.hpp>
#include "spdlog/spdlog.h"
#include "environment/job_shop_environment.h"
#include "job_shop_json_handler.h"
#include "job_shop_manual_generator.h"

using Random = effolkronium::random_static;

class AutomaticJobShopGenerator {
private:
    static std::vector<std::vector<int>> generateDependencies(
            int numJobs,
            double dependencyDensity,
            int maxDependenciesPerJob) {
        std::vector<std::vector<int>> dependencies(numJobs);

        for (int i = 0; i < numJobs; i++) {
            int maxPossibleDeps = std::min(i, maxDependenciesPerJob);
            if (maxPossibleDeps == 0) continue;

            int numDeps = Random::get<int>(0, maxPossibleDeps);
            if (Random::get<double>(0.0, 1.0) > dependencyDensity) {
                numDeps = 0;
            }

            if (numDeps > 0) {
                std::vector<int> possibleDeps(i);
                std::iota(possibleDeps.begin(), possibleDeps.end(), 0);

                Random::shuffle(possibleDeps);
                dependencies[i].assign(possibleDeps.begin(),
                                       possibleDeps.begin() + numDeps);
                std::sort(dependencies[i].begin(), dependencies[i].end());
            }
        }

        return dependencies;
    }

    static std::vector<int> generateDurations(
            int numJobs,
            int minDuration,
            int maxDuration,
            double longJobRate = 0.1,
            double longJobFactor = 3.0) {
        std::vector<int> durations(numJobs);

        for (int i = 0; i < numJobs; i++) {
            if (Random::get<double>(0.0, 1.0) < longJobRate) {
                durations[i] = Random::get<int>(
                        static_cast<int>(minDuration * longJobFactor),
                        static_cast<int>(maxDuration * longJobFactor)
                );
            } else {
                durations[i] = Random::get<int>(minDuration, maxDuration);
            }
        }

        return durations;
    }

    static std::vector<int> assignMachines(int numJobs, int numMachines) {
        std::vector<int> machineAssignments(numJobs);
        for (int i = 0; i < numJobs; i++) {
            machineAssignments[i] = Random::get<int>(0, numMachines - 1);
        }
        return machineAssignments;
    }

    /**
     * @brief Creates JobShopJsonHandler data structure from generated components
     */
    static JobShopJsonHandler::JobShopData createJobShopData(
            int numJobs,
            int numMachines,
            const std::vector<int>& durations,
            const std::vector<std::vector<int>>& dependencies) {
        JobShopJsonHandler::JobShopData jobShopData;
        jobShopData.metadata.numJobs = numJobs;
        jobShopData.metadata.numMachines = numMachines;

        // Generate random machine assignments
        auto machineAssignments = assignMachines(numJobs, numMachines);

        for (int i = 0; i < numJobs; i++) {
            JobShopJsonHandler::JobShopData::JobData jobData;
            jobData.id = i;
            jobData.duration = durations[i];
            jobData.machine = machineAssignments[i];
            jobData.dependencies = dependencies[i];
            jobShopData.jobs.push_back(jobData);
        }

        return jobShopData;
    }

    /**
     * @brief Converts JobShopData directly to Job structures without file I/O
     */
    static std::pair<std::vector<Job>, int> convertToJobs(
            const JobShopJsonHandler::JobShopData& jobShopData) {
        std::vector<Job> jobs(jobShopData.metadata.numJobs);

        // Create jobs
        for (const auto& jobData : jobShopData.jobs) {
            Job& job = jobs[jobData.id];

            Operation op;
            op.duration = jobData.duration;
            op.machine = jobData.machine;  // Use the assigned machine
            op.eligibleMachines.set(op.machine);

            job.operations.push_back(op);
            job.dependentJobs = jobData.dependencies;
        }

        // Calculate makespan
        int optimalMakespan = 0;
        std::vector<int> machineWorkload(jobShopData.metadata.numMachines, 0);

        for (const auto& job : jobs) {
            for (const auto& op : job.operations) {
                machineWorkload[op.machine] += op.duration;
                optimalMakespan = std::max(optimalMakespan, op.duration);
            }
        }

        // Consider both maximum machine workload and longest chain
        optimalMakespan = std::max(optimalMakespan,
                                   *std::max_element(machineWorkload.begin(), machineWorkload.end()));

        return {jobs, optimalMakespan};
    }

public:
    struct GenerationParams {
        int numJobs;
        int numMachines;
        int minDuration;
        int maxDuration;
        double dependencyDensity;
        int maxDependenciesPerJob;
        double longJobRate;
        double longJobFactor;
        std::optional<std::string> outputFile;
    };

    static std::pair<std::vector<Job>, int> generate(const GenerationParams& params) {
        SPDLOG_INFO("Generating random job shop problem with {} jobs and {} machines",
                    params.numJobs, params.numMachines);

        // Input validation
        if (params.numJobs <= 0 || params.numMachines <= 0) {
            throw std::invalid_argument("Number of jobs and machines must be positive");
        }
        if (params.minDuration <= 0 || params.maxDuration <= 0 ||
            params.minDuration > params.maxDuration) {
            throw std::invalid_argument("Invalid duration range");
        }
        if (params.dependencyDensity < 0.0 || params.dependencyDensity > 1.0) {
            throw std::invalid_argument("Dependency density must be between 0.0 and 1.0");
        }

        // Generate problem components
        auto dependencies = generateDependencies(
                params.numJobs,
                params.dependencyDensity,
                params.maxDependenciesPerJob
        );

        auto durations = generateDurations(
                params.numJobs,
                params.minDuration,
                params.maxDuration,
                params.longJobRate,
                params.longJobFactor
        );

        // Create data structure
        auto jobShopData = createJobShopData(
                params.numJobs,
                params.numMachines,
                durations,
                dependencies
        );

        // If output file is specified, write to file
        if (params.outputFile) {
            JobShopJsonHandler::writeToJson(*params.outputFile, jobShopData);
        }

        // Generate jobs using ManualJobShopGenerator
        return ManualJobShopGenerator::generateFromData(jobShopData);
    }

    static std::pair<std::vector<Job>, int> generateDefault(
            int numJobs,
            int numMachines,
            std::optional<std::string> outputFile = std::nullopt) {
        GenerationParams params{
                .numJobs = numJobs,
                .numMachines = numMachines,
                .minDuration = 1,
                .maxDuration = 100,
                .dependencyDensity = 0.3,
                .maxDependenciesPerJob = 3,
                .longJobRate = 0.1,
                .longJobFactor = 2.0,
                .outputFile = outputFile
        };

        return generate(params);
    }
};