#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <curl/curl.h>
#include "environment/job_shop_environment.h"
#include "fmt/format.h"
#include "utilities/util.h"
#include "environment/job_shop_manual_generator.h"


class TaillardJobShopGenerator {
private:


    static JobShopJsonHandler::JobShopData convertToJobShopData(const std::string& problemData) {
        std::istringstream dataStream(problemData);

        int numJobs, numMachines;
        dataStream >> numJobs >> numMachines;

        SPDLOG_INFO("Converting Taillard data: {} jobs, {} machines", numJobs, numMachines);

        JobShopJsonHandler::JobShopData jobShopData;
        jobShopData.metadata.numJobs = numJobs * numMachines; // Each operation becomes a job
        jobShopData.metadata.numMachines = numMachines;

        // Read processing times
        std::vector<std::vector<int>> processingTimes(numJobs, std::vector<int>(numMachines));
        for (int i = 0; i < numJobs; ++i) {
            for (int j = 0; j < numMachines; ++j) {
                dataStream >> processingTimes[i][j];
            }
        }

        // Read machine orders (1-based in input)
        std::vector<std::vector<int>> machineOrders(numJobs, std::vector<int>(numMachines));
        for (int i = 0; i < numJobs; ++i) {
            for (int j = 0; j < numMachines; ++j) {
                dataStream >> machineOrders[i][j];
                machineOrders[i][j]--; // Convert to 0-based indexing
            }
        }

        // Convert each operation to a job
        int jobCounter = 0;
        for (int i = 0; i < numJobs; ++i) {
            for (int j = 0; j < numMachines; ++j) {
                JobShopJsonHandler::JobShopData::JobData jobData;
                jobData.id = jobCounter++;
                jobData.duration = processingTimes[i][j];
                jobData.machine = machineOrders[i][j];

                // Dependencies: operation depends on the previous operation in the same original job
                if (j > 0) {
                    jobData.dependencies.push_back(i * numMachines + (j - 1));
                } else {
                    jobData.dependencies = {};
                }

                SPDLOG_DEBUG("Created job {} (original job {}, op {}) with duration {} on machine {}",
                             jobData.id, i, j, jobData.duration, jobData.machine);

                jobShopData.jobs.push_back(jobData);
            }
        }

        // Validate the conversion
        SPDLOG_INFO("Created {} jobs from {} original jobs with {} machines each",
                    jobShopData.jobs.size(), numJobs, numMachines);

        //JobShopJsonHandler::writeToJson("taillard_data.json", jobShopData);
        return jobShopData;
    }

    static std::string loadFile(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

public:
    static std::pair<std::vector<Job>, int> loadProblem(const std::string& filePath) {
        SPDLOG_INFO("Loading Taillard instance from {}", filePath);

        // Load and parse the file
        std::string problemData = loadFile(filePath);
        auto jobShopData = convertToJobShopData(problemData);

        // Generate job shop using ManualJobShopGenerator
        auto [jobs, makespanEstimate] = ManualJobShopGenerator::generateFromData(jobShopData);

        SPDLOG_INFO("Successfully loaded Taillard instance with {} jobs and {} machines",
                    jobShopData.metadata.numJobs, jobShopData.metadata.numMachines);

        // Verify the generated data
        verifyJobsData(jobs);

        return {jobs, makespanEstimate};
    }

    static void verifyJobsData(const std::vector<Job>& jobs) {
        if (jobs.empty()) {
            throw std::runtime_error("Jobs vector is empty");
        }

        for (size_t i = 0; i < jobs.size(); ++i) {
            const auto& job = jobs[i];

            if (job.operations.empty()) {
                throw std::runtime_error(fmt::format("Job {} has no operations", i));
            }

            // Verify each operation
            for (size_t j = 0; j < job.operations.size(); ++j) {
                const auto& op = job.operations[j];

                if (op.duration <= 0) {
                    throw std::runtime_error(fmt::format("Job {} operation {} has invalid duration: {}",
                                                         i, j, op.duration));
                }

                if (op.machine < 0) {
                    throw std::runtime_error(fmt::format("Job {} operation {} has invalid machine: {}",
                                                         i, j, op.machine));
                }

                if (op.eligibleMachines.count() != 1) {
                    throw std::runtime_error(fmt::format("Job {} operation {} has invalid number of eligible machines: {}",
                                                         i, j, op.eligibleMachines.count()));
                }

                // Log operation details at debug level
                SPDLOG_DEBUG("Job {} Op {} - Duration: {}, Machine: {}",
                             i, j, op.duration, op.machine);
            }

            // Verify operation sequence
            for (size_t j = 1; j < job.operations.size(); ++j) {
                bool hasSequentialDependency = false;
                for (const auto& [depJob, depOp] : job.operations[j].dependentOperations) {
                    if (depJob == i && depOp == j - 1) {
                        hasSequentialDependency = true;
                        break;
                    }
                }

                if (!hasSequentialDependency) {
                    SPDLOG_WARN("Job {} Op {} may not properly depend on previous operation", i, j);
                }
            }
        }

        SPDLOG_INFO("Job data verification completed successfully");
    }
};