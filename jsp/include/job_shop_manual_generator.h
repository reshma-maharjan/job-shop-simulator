#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <set>
#include <queue>
#include <functional>
#include <numeric>
#include "spdlog/spdlog.h"
#include "job_shop_environment.h"

/**
 * @class ManualJobShopGenerator
 * @brief Generates job shop problems from formatted input files
 *
 * This class handles parsing and validation of job shop problem specifications
 * from text files. It includes comprehensive error checking and logging.
 */
class ManualJobShopGenerator {
private:
    /**
     * @brief Detects circular dependencies in the job dependency graph
     * @param jobDependencies Vector of dependencies for each job
     * @return true if circular dependencies are found, false otherwise
     */
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

    /**
     * @brief Verifies that all operation durations are valid
     * @param durations Vector of operation durations for each job
     * @throws runtime_error if invalid durations are found
     */
    static void verifyOperationDurations(const std::vector<std::pair<int, int>>& durations) {
        for (const auto& [op, duration] : durations) {
            if (duration < 0) {
                throw std::runtime_error(fmt::format("Invalid negative duration: {}", duration));
            }
        }
    }

    /**
     * @brief Reads metadata section from input file
     * @param file Input file stream
     * @return Map of metadata key-value pairs
     */
    static std::unordered_map<std::string, int> readMetadata(std::ifstream& file) {
        std::unordered_map<std::string, int> metadata;
        std::string line;
        while (std::getline(file, line)) {
            if (line == "[METADATA END]") break;
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string key;
            int value;
            if (std::getline(iss, key, ':') && iss >> value) {
                metadata[trim(key)] = value;
                SPDLOG_DEBUG("Metadata: {} = {}", trim(key), value);
            }
        }
        return metadata;
    }

    /**
     * @brief Reads operation durations in new format (operation:duration)
     * @param file Input file stream
     * @param numJobs Number of jobs expected
     * @param endMarker String marking end of section
     * @return Vector of operation durations for each job
     */
    static std::vector<std::pair<int, int>> readOperationDurations(
            std::ifstream& file, int numJobs, const std::string& endMarker) {
        std::vector<std::pair<int, int>> durations;
        std::string line;
        std::set<int> seenJobs;  // Track which jobs we've seen

        SPDLOG_INFO("Reading operation durations for {} jobs", numJobs);

        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue;

            // Parse job:duration format
            size_t colonPos = line.find(':');
            if (colonPos == std::string::npos) {
                SPDLOG_ERROR("Invalid duration format, missing colon: {}", line);
                throw std::runtime_error("Invalid duration format: " + line);
            }

            try {
                int jobIndex = std::stoi(trim(line.substr(0, colonPos)));
                int duration = std::stoi(trim(line.substr(colonPos + 1)));

                if (jobIndex >= numJobs) {
                    SPDLOG_ERROR("Job index {} exceeds number of jobs {}", jobIndex, numJobs);
                    throw std::runtime_error(fmt::format("Invalid job index: {}", jobIndex));
                }

                if (duration < 0) {
                    SPDLOG_ERROR("Negative duration {} for job {}", duration, jobIndex);
                    throw std::runtime_error(fmt::format("Negative duration for job {}: {}", jobIndex, duration));
                }

                if (seenJobs.count(jobIndex) > 0) {
                    SPDLOG_ERROR("Duplicate duration entry for job {}", jobIndex);
                    throw std::runtime_error(fmt::format("Duplicate duration entry for job {}", jobIndex));
                }

                seenJobs.insert(jobIndex);
                durations.emplace_back(jobIndex, duration);
                SPDLOG_DEBUG("Job {} Duration: {}", jobIndex, duration);

            } catch (const std::exception& e) {
                SPDLOG_ERROR("Failed to parse duration line: {}", line);
                throw std::runtime_error("Failed to parse duration: " + line);
            }
        }

        // Verify all jobs have durations
        if (seenJobs.size() != numJobs) {
            std::vector<int> missingJobs;
            for (int i = 0; i < numJobs; i++) {
                if (seenJobs.count(i) == 0) {
                    missingJobs.push_back(i);
                }
            }
            if (!missingJobs.empty()) {
                SPDLOG_ERROR("Missing durations for jobs: {}", fmt::join(missingJobs, ", "));
                throw std::runtime_error("Missing durations for some jobs");
            }
        }

        // Sort durations by job index for consistency
        std::sort(durations.begin(), durations.end());

        return durations;
    }

    /**
     * @brief Reads job dependencies in sparse format
     * @param file Input file stream
     * @param numJobs Number of jobs
     * @param endMarker String marking end of section
     * @return Vector of job dependencies
     */
    static std::vector<std::vector<int>> readJobDependenciesSparse(
            std::ifstream& file, int numJobs, const std::string& endMarker) {
        std::vector<std::vector<int>> jobDependencies(numJobs);
        std::string line;
        std::set<std::pair<int, int>> dependencyPairs;  // For detecting duplicates

        SPDLOG_INFO("Reading job dependencies for {} jobs", numJobs);

        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue;

            size_t colonPos = line.find(':');
            if (colonPos == std::string::npos) {
                SPDLOG_ERROR("Invalid job dependency format: {}", line);
                throw std::runtime_error("Invalid job dependency format, missing colon");
            }

            int jobIndex = std::stoi(trim(line.substr(0, colonPos)));
            if (jobIndex >= numJobs) {
                SPDLOG_ERROR("Invalid job index in dependencies: {}", jobIndex);
                throw std::runtime_error(fmt::format("Job index {} exceeds number of jobs {}", jobIndex, numJobs));
            }

            std::string dependenciesStr = line.substr(colonPos + 1);
            std::vector<int> deps = parseDependencyList(dependenciesStr);

            // Validate dependencies and check for duplicates
            for (int depJob : deps) {
                if (depJob >= numJobs) {
                    SPDLOG_ERROR("Invalid dependency job index: {} for job {}", depJob, jobIndex);
                    throw std::runtime_error(fmt::format("Invalid dependency job index: {}", depJob));
                }

                auto [it, inserted] = dependencyPairs.insert({jobIndex, depJob});
                if (!inserted) {
                    SPDLOG_WARN("Duplicate dependency: Job {} -> Job {}", jobIndex, depJob);
                }
            }

            jobDependencies[jobIndex] = deps;
            SPDLOG_DEBUG("Job {} depends on: {}", jobIndex, fmt::join(deps, ", "));
        }

        return jobDependencies;
    }

    /**
     * @brief Reads operation dependencies
     * @param file Input file stream
     * @param numJobs Number of jobs
     * @param endMarker String marking end of section
     * @return Vector of operation dependencies for each job
     */
    static std::vector<std::unordered_map<int, std::vector<int>>> readOperationDependencies(
            std::ifstream& file, int numJobs, const std::string& endMarker) {
        std::vector<std::unordered_map<int, std::vector<int>>> dependencies(numJobs);
        std::string line;

        SPDLOG_INFO("Reading operation dependencies");

        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue;

            size_t colonPos = line.find(':');
            if (colonPos == std::string::npos) continue;  // Skip malformed lines

            try {
                int jobIndex = std::stoi(line.substr(0, colonPos));
                if (jobIndex >= numJobs) {
                    SPDLOG_ERROR("Invalid job index in operation dependencies: {}", jobIndex);
                    continue;
                }

                std::string depsStr = line.substr(colonPos + 1);
                std::istringstream iss(depsStr);
                std::string token;

                while (std::getline(iss, token, ';')) {
                    token = trim(token);
                    if (token.empty()) continue;

                    size_t opColonPos = token.find(':');
                    if (opColonPos == std::string::npos) continue;

                    int opIndex = std::stoi(token.substr(0, opColonPos));
                    std::vector<int> depOps = parseDependencyList(token.substr(opColonPos + 1));
                    dependencies[jobIndex][opIndex] = depOps;

                    SPDLOG_DEBUG("Job {} Operation {} depends on operations: {}",
                                 jobIndex, opIndex, fmt::join(depOps, ", "));
                }
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Error parsing operation dependency line: {}", line);
                continue;
            }
        }

        return dependencies;
    }

    /**
     * @brief Parses a comma-separated list of dependencies
     * @param depsStr String containing dependencies
     * @return Vector of dependency indices
     */
    static std::vector<int> parseDependencyList(const std::string& depsStr) {
        std::vector<int> deps;
        std::istringstream iss(depsStr);
        std::string token;

        while (std::getline(iss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
                try {
                    deps.push_back(std::stoi(token));
                } catch (const std::exception& e) {
                    SPDLOG_ERROR("Invalid dependency value: {}", token);
                    throw std::runtime_error(fmt::format("Invalid dependency value: {}", token));
                }
            }
        }
        return deps;
    }

    /**
     * @brief Trims whitespace from string
     * @param str Input string
     * @return Trimmed string
     */
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t");
        if (std::string::npos == first) return str;
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, (last - first + 1));
    }

    /**
     * @brief Logs the complete job dependency graph
     * @param jobDependencies Vector of job dependencies
     */
    static void logJobDependencies(const std::vector<std::vector<int>>& jobDependencies) {
        SPDLOG_INFO("Job Dependencies Graph:");
        for (size_t i = 0; i < jobDependencies.size(); i++) {
            if (!jobDependencies[i].empty()) {
                SPDLOG_INFO("Job {} depends on: {}", i, fmt::join(jobDependencies[i], ", "));
            }
        }
    }

public:
    /**
     * @brief Generates job shop problem from input file
     * @param filename Path to input file
     * @return Pair of job vector and optimal makespan
     */
    static std::pair<std::vector<Job>, int> generateFromFile(const std::string& filename) {
        SPDLOG_INFO("Starting to generate job shop from file: {}", filename);

        std::ifstream file(filename);
        if (!file.is_open()) {
            SPDLOG_ERROR("Failed to open file: {}", filename);
            throw std::runtime_error("Unable to open file: " + filename);
        }

        std::string line;

        // Read metadata
        while (std::getline(file, line) && line != "[METADATA]");
        auto metadata = readMetadata(file);

        if (!metadata.count("num_jobs") || !metadata.count("num_machines")) {
            SPDLOG_ERROR("Missing required metadata fields");
            throw std::runtime_error("Missing required metadata fields");
        }

        int numJobs = metadata["num_jobs"];
        int numMachines = metadata["num_machines"];

        SPDLOG_INFO("Metadata loaded: {} jobs, {} machines", numJobs, numMachines);

        // Read job dependencies
        while (std::getline(file, line) && line != "[JOB DEPENDENCIES]");
        auto jobDependencies = readJobDependenciesSparse(file, numJobs, "[JOB DEPENDENCIES END]");

        if (detectCircularDependencies(jobDependencies)) {
            SPDLOG_ERROR("Circular dependencies detected in job dependencies");
            throw std::runtime_error("Circular dependencies detected");
        }

        // Read operation durations with new format
        while (std::getline(file, line) && line != "[OPERATION DURATIONS]");
        auto durations = readOperationDurations(file, numJobs, "[OPERATION DURATIONS END]");

        std::vector<Job> jobs(numJobs);

        // Create jobs with their durations
        for (const auto& [jobIndex, duration] : durations) {
            Job& job = jobs[jobIndex];

            Operation op;
            op.duration = duration;
            op.machine = jobIndex % numMachines;  // Simple round-robin assignment
            op.eligibleMachines.set(op.machine);

            SPDLOG_DEBUG("Job {} Operation 0: Duration {}, Machine {}",
                         jobIndex, duration, op.machine);

            job.operations.push_back(op);
            job.dependentJobs = jobDependencies[jobIndex];
        }

        // Calculate optimal makespan
        int optimalMakespan = 0;
        for (const auto& job : jobs) {
            int jobMakespan = 0;
            for (const auto& op : job.operations) {
                jobMakespan += op.duration;
            }
            optimalMakespan = std::max(optimalMakespan, jobMakespan);
        }

        SPDLOG_INFO("Job shop generation complete");
        SPDLOG_INFO("Total jobs: {}", jobs.size());
        SPDLOG_INFO("Total machines: {}", numMachines);
        SPDLOG_INFO("Optimal makespan: {}", optimalMakespan);

        return {jobs, optimalMakespan};
    }

private:
    /**
     * @brief Validates the complete job shop configuration
     * @param jobs Vector of all jobs
     * @param numMachines Number of machines
     * @throws runtime_error if validation fails
     */
    static void validateJobShop(const std::vector<Job>& jobs, int numMachines) {
        SPDLOG_INFO("Performing final job shop validation");

        // Check machine assignments
        std::vector<std::set<int>> machineOperations(numMachines);

        for (size_t jobId = 0; jobId < jobs.size(); jobId++) {
            const auto& job = jobs[jobId];

            // Verify operation sequence
            for (size_t opId = 0; opId < job.operations.size(); opId++) {
                const auto& op = job.operations[opId];

                // Verify machine assignment
                if (op.machine >= numMachines) {
                    SPDLOG_ERROR("Job {} Operation {} assigned to invalid machine {}",
                                 jobId, opId, op.machine);
                    throw std::runtime_error(
                            fmt::format("Invalid machine assignment for Job {} Operation {}",
                                        jobId, opId));
                }

                // Track machine usage
                machineOperations[op.machine].insert(jobId);

                // Verify duration
                if (op.duration <= 0) {
                    SPDLOG_ERROR("Job {} Operation {} has invalid duration {}",
                                 jobId, opId, op.duration);
                    throw std::runtime_error(
                            fmt::format("Invalid duration for Job {} Operation {}",
                                        jobId, opId));
                }

                // Verify operation dependencies
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

        // Verify machine utilization
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
};