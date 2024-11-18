#pragma once

#include <vector>
#include <functional>
#include <set>
#include "spdlog/spdlog.h"
#include "environment/job_shop_environment.h"
#include "environment/job_shop_json_handler.h"

class ManualJobShopGenerator {
private:
    static bool detectCircularDependencies(const std::vector<std::vector<int>>& jobDependencies) {
        std::vector<int> visited(jobDependencies.size(), 0);
        std::vector<int> recursionStack(jobDependencies.size(), 0);

        std::function<bool(int)> hasCycle = [&](int job) -> bool {
            if (visited[job] == 0) {
                visited[job] = 1;
                recursionStack[job] = 1;

                for (int depJob : jobDependencies[job]) {
                    if (depJob >= jobDependencies.size()) {
                        SPDLOG_ERROR("Invalid job dependency: Job {} depends on non-existent job {}", job, depJob);
                        throw std::runtime_error("Invalid job dependency reference");
                    }

                    if (visited[depJob] == 0 && hasCycle(depJob)) {
                        return true;
                    }
                    else if (recursionStack[depJob] == 1) {
                        SPDLOG_ERROR("Circular dependency detected involving job {}", depJob);
                        return true;
                    }
                }
            }
            recursionStack[job] = 0;
            return false;
        };

        for (size_t i = 0; i < jobDependencies.size(); i++) {
            if (visited[i] == 0 && hasCycle(i)) {
                return true;
            }
        }
        return false;
    }

    static void logJobDependencies(const std::vector<std::vector<int>>& jobDependencies) {
        SPDLOG_INFO("Job Dependencies Graph:");
        for (size_t i = 0; i < jobDependencies.size(); i++) {
            if (!jobDependencies[i].empty()) {
                SPDLOG_INFO("Job {} depends on: {}", i, fmt::join(jobDependencies[i], ", "));
            }
        }
    }

    static void validateJobShop(const std::vector<Job>& jobs, int numMachines) {
        SPDLOG_INFO("Performing final job shop validation");

        std::vector<std::set<int>> machineOperations(numMachines);

        for (size_t jobId = 0; jobId < jobs.size(); jobId++) {
            const auto& job = jobs[jobId];

            for (size_t opId = 0; opId < job.operations.size(); opId++) {
                const auto& op = job.operations[opId];

                if (op.machine >= numMachines) {
                    SPDLOG_ERROR("Job {} Operation {} assigned to invalid machine {}",
                                 jobId, opId, op.machine);
                    throw std::runtime_error(
                            fmt::format("Invalid machine assignment for Job {} Operation {}",
                                        jobId, opId));
                }

                machineOperations[op.machine].insert(jobId);

                if (op.duration <= 0) {
                    SPDLOG_ERROR("Job {} Operation {} has invalid duration {}",
                                 jobId, opId, op.duration);
                    throw std::runtime_error(
                            fmt::format("Invalid duration for Job {} Operation {}",
                                        jobId, opId));
                }

                for (const auto& [depJobId, depOpId] : op.dependentOperations) {
                    if (depJobId >= static_cast<int>(jobs.size())) {
                        SPDLOG_ERROR("Invalid operation dependency: Job {} -> Job {}",
                                     jobId, depJobId);
                        throw std::runtime_error("Invalid operation dependency");
                    }

                    const auto& depJob = jobs[depJobId];
                    if (depOpId >= static_cast<int>(depJob.operations.size())) {
                        SPDLOG_ERROR("Invalid operation dependency: Job {} Op {} -> Job {} Op {}",
                                     jobId, opId, depJobId, depOpId);
                        throw std::runtime_error("Invalid operation dependency");
                    }
                }
            }
        }

        for (size_t machineId = 0; machineId < machineOperations.size(); machineId++) {
            if (machineOperations[machineId].empty()) {
                SPDLOG_WARN("Machine {} has no assigned operations", machineId);
            } else {
                SPDLOG_DEBUG("Machine {} is used by {} jobs",
                             machineId, machineOperations[machineId].size());
            }
        }

        SPDLOG_INFO("Job shop validation complete - configuration is valid");
    }




    /**
     * @brief Identifies dependency chains and their machine requirements
     * @param jobs Vector of all jobs
     * @param jobDependencies Dependency graph
     * @return Vector of dependency chains with their total duration
     */
    static std::vector<std::vector<int>> identifyDependencyChains(
            const std::vector<Job>& jobs,
            const std::vector<std::vector<int>>& jobDependencies) {
        const int numJobs = jobs.size();
        std::vector<std::vector<int>> chains;
        std::vector<bool> visited(numJobs, false);

        // Find root jobs (those with no incoming dependencies)
        std::vector<bool> hasIncomingDeps(numJobs, false);
        for (const auto& deps : jobDependencies) {
            for (int dep : deps) {
                hasIncomingDeps[dep] = true;
            }
        }

        std::function<void(int, std::vector<int>&)> buildChain =
                [&](int jobId, std::vector<int>& currentChain) {
                    if (visited[jobId]) return;

                    visited[jobId] = true;
                    currentChain.push_back(jobId);

                    // Check if this is a leaf node (no outgoing dependencies)
                    bool isLeaf = true;
                    for (size_t i = 0; i < jobDependencies.size(); i++) {
                        if (std::find(jobDependencies[i].begin(), jobDependencies[i].end(), jobId)
                            != jobDependencies[i].end()) {
                            isLeaf = false;
                            buildChain(i, currentChain);
                        }
                    }

                    if (isLeaf) {
                        chains.push_back(currentChain);
                    }
                };

        // Start from each root job
        for (int i = 0; i < numJobs; i++) {
            if (!hasIncomingDeps[i] && !visited[i]) {
                std::vector<int> chain;
                buildChain(i, chain);
            }
        }

        return chains;
    }

    /**
     * @brief Calculates chain conflict score based on machine usage
     * @param jobs Vector of all jobs
     * @param chain1 First chain of jobs
     * @param chain2 Second chain of jobs
     * @return Conflict score between the chains
     */
    static int calculateChainConflict(
            const std::vector<Job>& jobs,
            const std::vector<int>& chain1,
            const std::vector<int>& chain2) {
        int conflictScore = 0;
        std::set<int> machines1, machines2;

        // Collect machine usage for each chain
        for (int jobId : chain1) {
            for (const auto& op : jobs[jobId].operations) {
                machines1.insert(op.machine);
            }
        }
        for (int jobId : chain2) {
            for (const auto& op : jobs[jobId].operations) {
                machines2.insert(op.machine);
            }
        }

        // Count shared machines
        for (int machine : machines1) {
            if (machines2.count(machine) > 0) {
                conflictScore++;
            }
        }

        return conflictScore;
    }

    /**
     * @brief Calculates chain duration including machine conflicts
     * @param jobs Vector of all jobs
     * @param chain Vector of job indices in the chain
     * @return Total duration of the chain
     */
    static int calculateChainDuration(
            const std::vector<Job>& jobs,
            const std::vector<int>& chain) {
        int duration = 0;
        std::set<int> usedMachines;

        for (int jobId : chain) {
            int jobDuration = 0;
            for (const auto& op : jobs[jobId].operations) {
                jobDuration += op.duration;
                usedMachines.insert(op.machine);
            }
            duration += jobDuration;
        }

        // Add overhead for machine switches within the chain
        duration += usedMachines.size() * 10;  // Arbitrary overhead for machine switching

        return duration;
    }

    static int calculateOptimalMakespan(
            const std::vector<Job>& jobs,
            const std::vector<std::vector<int>>& jobDependencies,
            int numMachines) {
        // Identify all dependency chains
        auto chains = identifyDependencyChains(jobs, jobDependencies);

        // Calculate base workload
        int totalWorkload = 0;
        std::vector<int> machineWorkload(numMachines, 0);
        for (const auto &job: jobs) {
            for (const auto &op: job.operations) {
                totalWorkload += op.duration;
                machineWorkload[op.machine] += op.duration;
            }
        }

        // Calculate chain durations with conflicts
        int maxChainDuration = 0;
        for (const auto &chain: chains) {
            int chainDuration = calculateChainDuration(jobs, chain);
            maxChainDuration = std::max(maxChainDuration, chainDuration);

            // Look for conflicts with other chains
            for (const auto &otherChain: chains) {
                if (&chain != &otherChain) {
                    int conflictScore = calculateChainConflict(jobs, chain, otherChain);
                    if (conflictScore > 0) {
                        // Add conflict-based overhead
                        chainDuration += conflictScore * 15;  // Arbitrary overhead for chain conflicts
                    }
                }
            }
            maxChainDuration = std::max(maxChainDuration, chainDuration);
        }

        // Get the maximum machine workload
        int maxMachineWorkload = *std::max_element(machineWorkload.begin(), machineWorkload.end());

        // Calculate lower bound considering all factors
        int lowerBound = std::max({
                                          maxChainDuration,
                                          maxMachineWorkload,
                                          (totalWorkload + numMachines - 1) / numMachines
                                  });

        SPDLOG_INFO("Max chain duration (with conflicts): {}", maxChainDuration);
        SPDLOG_INFO("Max machine workload: {}", maxMachineWorkload);
        SPDLOG_INFO("Average workload per machine: {}", totalWorkload / numMachines);

        return lowerBound;
    }



public:

    static std::pair<std::vector<Job>, int> generateFromData(const JobShopJsonHandler::JobShopData& jobShopData) {
        SPDLOG_INFO("Starting to generate job shop from data structure");

        int numJobs = jobShopData.metadata.numJobs;
        int numMachines = jobShopData.metadata.numMachines;

        SPDLOG_INFO("Data loaded: {} jobs, {} machines", numJobs, numMachines);

        // Convert to job dependencies format
        std::vector<std::vector<int>> jobDependencies(numJobs);
        for (const auto& jobData : jobShopData.jobs) {
            jobDependencies[jobData.id] = jobData.dependencies;
        }

        // Check for circular dependencies
        if (detectCircularDependencies(jobDependencies)) {
            SPDLOG_ERROR("Circular dependencies detected in job dependencies");
            throw std::runtime_error("Circular dependencies detected");
        }

        // Create jobs
        std::vector<Job> jobs(numJobs);
        for (const auto& jobData : jobShopData.jobs) {
            Job& job = jobs[jobData.id];

            // Create a single operation for this job
            Operation op;
            op.duration = jobData.duration;
            op.machine = jobData.machine;
            op.eligibleMachines.set(op.machine);

            // Only add operation dependencies if they exist in the JSON
            // and if they're properly defined
            if (!jobData.operationDependencies.empty()) {
                for (const auto& opDep : jobData.operationDependencies) {
                    for (int depJobId : opDep.dependencies) {
                        op.dependentOperations.emplace_back(depJobId, 0); // Use 0 as operation index since we have single operations
                    }
                }
            }

            job.operations.push_back(op);
            job.dependentJobs = jobDependencies[jobData.id];

            SPDLOG_DEBUG("Job {} Operation 0: Duration {}, Machine {}",
                         jobData.id, jobData.duration, op.machine);
        }

        validateJobShop(jobs, numMachines);

        // Calculate optimal makespan
        int optimalMakespan = calculateOptimalMakespan(jobs, jobDependencies, numMachines);

        SPDLOG_INFO("Job shop generation complete");
        SPDLOG_INFO("Total jobs: {}", jobs.size());
        SPDLOG_INFO("Total machines: {}", numMachines);
        SPDLOG_INFO("Optimal makespan: {}", optimalMakespan);

        return {jobs, optimalMakespan};
    }

    static std::pair<std::vector<Job>, int> generateFromFile(const std::string& filename) {
        SPDLOG_INFO("Starting to generate job shop from file: {}", filename);
        auto jobShopData = JobShopJsonHandler::readFromJson(filename);
        std::cout << "Job shop data: " << jobShopData << std::endl;
        return generateFromData(jobShopData);
    }


};