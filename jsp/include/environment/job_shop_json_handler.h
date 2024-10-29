#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "spdlog/spdlog.h"

using json = nlohmann::json;

/**
 * @class JobShopJsonHandler
 * @brief Handles reading and writing job shop problem specifications in JSON format
 */
class JobShopJsonHandler {
public:
    struct JobShopData {
        struct Metadata {
            int numJobs;
            int numMachines;
        } metadata;

        struct JobData {
            int id;
            int duration;
            int machine;  // Machine assignment for this job
            std::vector<int> dependencies;
            struct OperationData {
                int operationId;
                std::vector<int> dependencies;
            };
            std::vector<OperationData> operationDependencies;
        };
        std::vector<JobData> jobs;
    };

    /**
     * @brief Reads job shop data from a JSON file
     * @param filename Path to input JSON file
     * @return JobShopData structure containing all problem data
     */
    static JobShopData readFromJson(const std::string& filename) {
        SPDLOG_INFO("Reading job shop from JSON file: {}", filename);

        std::ifstream file(filename);
        if (!file.is_open()) {
            SPDLOG_ERROR("Failed to open file: {}", filename);
            throw std::runtime_error("Unable to open file: " + filename);
        }

        json jobShopJson;
        try {
            jobShopJson = json::parse(file);
        } catch (const json::parse_error& e) {
            SPDLOG_ERROR("JSON parse error: {}", e.what());
            throw;
        }

        JobShopData data;
        validateJsonStructure(jobShopJson);

        // Parse metadata
        data.metadata.numJobs = jobShopJson["metadata"]["numJobs"];
        data.metadata.numMachines = jobShopJson["metadata"]["numMachines"];

        // Parse jobs
        for (const auto& jobJson : jobShopJson["jobs"]) {
            JobShopData::JobData job;
            job.id = jobJson["id"];
            job.duration = jobJson["duration"];
            job.machine = jobJson["machine"];  // Read machine assignment
            job.dependencies = jobJson["dependencies"].get<std::vector<int>>();

            // Validate machine assignment
            if (job.machine < 0 || job.machine >= data.metadata.numMachines) {
                SPDLOG_ERROR("Invalid machine assignment for job {}: {}", job.id, job.machine);
                throw std::runtime_error(
                        fmt::format("Invalid machine assignment for job {}: machine {} is outside range [0, {}]",
                                    job.id, job.machine, data.metadata.numMachines - 1));
            }

            // Parse operation dependencies if they exist
            if (jobJson.contains("operationDependencies")) {
                for (const auto& opDepJson : jobJson["operationDependencies"]) {
                    JobShopData::JobData::OperationData opDep;
                    opDep.operationId = opDepJson["operationId"];
                    opDep.dependencies = opDepJson["dependencies"].get<std::vector<int>>();
                    job.operationDependencies.push_back(opDep);
                }
            }

            data.jobs.push_back(job);
        }

        // Additional validation
        validateJobData(data);

        return data;
    }

    /**
     * @brief Writes job shop data to a JSON file
     * @param filename Output filename
     * @param data JobShopData structure containing problem specification
     */
    static void writeToJson(const std::string& filename, const JobShopData& data) {
        // Validate data before writing
        validateJobData(data);

        json jobShopJson;

        // Write metadata
        jobShopJson["metadata"] = {
                {"numJobs", data.metadata.numJobs},
                {"numMachines", data.metadata.numMachines}
        };

        // Write jobs
        jobShopJson["jobs"] = json::array();
        for (const auto& job : data.jobs) {
            json jobJson = {
                    {"id", job.id},
                    {"duration", job.duration},
                    {"machine", job.machine},
                    {"dependencies", job.dependencies}
            };

            // Write operation dependencies if they exist
            if (!job.operationDependencies.empty()) {
                jobJson["operationDependencies"] = json::array();
                for (const auto& opDep : job.operationDependencies) {
                    jobJson["operationDependencies"].push_back({
                                                                       {"operationId", opDep.operationId},
                                                                       {"dependencies", opDep.dependencies}
                                                               });
                }
            }

            jobShopJson["jobs"].push_back(jobJson);
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            SPDLOG_ERROR("Failed to create output file: {}", filename);
            throw std::runtime_error("Failed to create output file: " + filename);
        }

        file << jobShopJson.dump(2);  // Pretty print with 2-space indent
        SPDLOG_INFO("Successfully wrote job shop data to {}", filename);
    }

private:
    /**
     * @brief Validates the JSON structure has all required fields
     * @param jobShopJson JSON object to validate
     */
    static void validateJsonStructure(const json& jobShopJson) {
        if (!jobShopJson.contains("metadata")) {
            throw std::runtime_error("Missing metadata section");
        }
        if (!jobShopJson["metadata"].contains("numJobs")) {
            throw std::runtime_error("Missing numJobs in metadata");
        }
        if (!jobShopJson["metadata"].contains("numMachines")) {
            throw std::runtime_error("Missing numMachines in metadata");
        }
        if (!jobShopJson.contains("jobs")) {
            throw std::runtime_error("Missing jobs section");
        }

        for (const auto& jobJson : jobShopJson["jobs"]) {
            if (!jobJson.contains("id")) {
                throw std::runtime_error("Job missing id");
            }
            if (!jobJson.contains("duration")) {
                throw std::runtime_error("Job missing duration");
            }
            if (!jobJson.contains("machine")) {
                throw std::runtime_error("Job missing machine assignment");
            }
            if (!jobJson.contains("dependencies")) {
                throw std::runtime_error("Job missing dependencies");
            }
        }
    }

    /**
     * @brief Validates the job shop data for consistency
     * @param data JobShopData structure to validate
     */
    static void validateJobData(const JobShopData& data) {
        if (data.metadata.numJobs <= 0) {
            throw std::runtime_error("Number of jobs must be positive");
        }
        if (data.metadata.numMachines <= 0) {
            throw std::runtime_error("Number of machines must be positive");
        }
        if (data.jobs.size() != static_cast<size_t>(data.metadata.numJobs)) {
            throw std::runtime_error(fmt::format("Number of jobs ({}) doesn't match metadata ({})",
                                                 data.jobs.size(), data.metadata.numJobs));
        }

        // Validate job IDs and machine assignments
        std::vector<bool> usedIds(data.metadata.numJobs, false);
        for (const auto& job : data.jobs) {
            if (job.id < 0 || job.id >= data.metadata.numJobs) {
                throw std::runtime_error(fmt::format("Invalid job ID: {}", job.id));
            }
            if (usedIds[job.id]) {
                throw std::runtime_error(fmt::format("Duplicate job ID: {}", job.id));
            }
            usedIds[job.id] = true;

            if (job.machine < 0 || job.machine >= data.metadata.numMachines) {
                throw std::runtime_error(fmt::format("Invalid machine assignment for job {}: {}",
                                                     job.id, job.machine));
            }
            if (job.duration <= 0) {
                throw std::runtime_error(fmt::format("Invalid duration for job {}: {}",
                                                     job.id, job.duration));
            }

            // Validate dependencies
            for (int depId : job.dependencies) {
                if (depId < 0 || depId >= data.metadata.numJobs) {
                    throw std::runtime_error(fmt::format("Invalid dependency {} for job {}",
                                                         depId, job.id));
                }
            }

            // Validate operation dependencies
            for (const auto& opDep : job.operationDependencies) {
                if (opDep.operationId < 0) {
                    throw std::runtime_error(fmt::format("Invalid operation ID for job {}: {}",
                                                         job.id, opDep.operationId));
                }
                for (int depId : opDep.dependencies) {
                    if (depId < 0) {
                        throw std::runtime_error(fmt::format("Invalid operation dependency {} for job {} operation {}",
                                                             depId, job.id, opDep.operationId));
                    }
                }
            }
        }
    }
};