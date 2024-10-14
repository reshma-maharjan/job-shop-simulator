#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include "job_shop_environment.h"

class ManualJobShopGenerator {
private:
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
            }
        }
        return metadata;
    }

    static std::vector<std::vector<int>> readMatrix(std::ifstream& file, int rows, int cols, const std::string& endMarker) {
        std::vector<std::vector<int>> matrix;
        std::string line;
        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<int> row = parseRow(line);
            if (row.size() != cols) {
                throw std::runtime_error("Incorrect number of columns in matrix. Expected: " + std::to_string(cols) + ", Got: " + std::to_string(row.size()));
            }
            matrix.push_back(row);
        }
        if (matrix.size() != rows) {
            throw std::runtime_error("Incorrect number of rows in matrix. Expected: " + std::to_string(rows) + ", Got: " + std::to_string(matrix.size()));
        }
        return matrix;
    }

    static std::vector<std::vector<int>> readOperationDurations(std::ifstream& file, int numJobs, const std::string& endMarker) {
        std::vector<std::vector<int>> durations;
        std::string line;
        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue;
            durations.push_back(parseRow(line));
        }
        if (durations.size() != numJobs) {
            throw std::runtime_error("Incorrect number of jobs in operation durations.");
        }
        return durations;
    }

    static std::vector<std::unordered_map<int, std::vector<int>>> readOperationDependencies(std::ifstream& file, int numJobs, const std::string& endMarker) {
        // Create a vector to hold dependencies for each job. Each job will have a map of operation dependencies.
        std::vector<std::unordered_map<int, std::vector<int>>> dependencies(numJobs);
        std::string line;

        // Read each line until we reach the end marker
        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue; // Skip comments or empty lines

            size_t colonPos = line.find(':');
            if (colonPos == std::string::npos) {
                throw std::runtime_error("Invalid operation dependency format, missing job index.");
            }

            // Extract the job index at the start of the line
            int jobIndex = std::stoi(line.substr(0, colonPos));

            // The remaining part contains the operation dependencies for that job
            std::string dependenciesStr = line.substr(colonPos + 1);
            std::unordered_map<int, std::vector<int>> opDeps;
            std::istringstream iss(dependenciesStr);
            std::string token;

            // Parse each operation dependency pair
            while (std::getline(iss, token, ';')) {
                token = trim(token);
                if (token.empty()) continue;

                size_t opColonPos = token.find(':');
                if (opColonPos == std::string::npos) {
                    throw std::runtime_error("Invalid operation dependency format in job " + std::to_string(jobIndex));
                }

                int opIndex = std::stoi(token.substr(0, opColonPos));
                std::string depsStr = token.substr(opColonPos + 1);

                // Parse the list of dependent operation indices
                std::vector<int> depOps = parseDependencyList(depsStr);
                opDeps[opIndex] = depOps;  // Store the operation dependencies for this operation
            }

            // Store the parsed dependencies for this job
            dependencies[jobIndex] = opDeps;
        }

        return dependencies;
    }

    static std::vector<int> parseRow(const std::string& line) {
        std::vector<int> row;
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ',')) {
            value = trim(value);
            if (!value.empty()) {
                try {
                    row.push_back(std::stoi(value));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid data: " + value);
                }
            }
        }
        return row;
    }

    static std::vector<int> parseDependencyList(const std::string& depsStr) {
        std::vector<int> depOps;
        std::istringstream depIss(depsStr);
        std::string dep;
        while (std::getline(depIss, dep, ',')) {
            dep = trim(dep);
            if (!dep.empty()) {
                depOps.push_back(std::stoi(dep));
            }
        }
        return depOps;
    }

    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t");
        if (std::string::npos == first) return str;
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, (last - first + 1));
    }

    static std::vector<std::vector<int>> readJobDependenciesSparse(std::ifstream& file, int numJobs, const std::string& endMarker) {
        std::vector<std::vector<int>> jobDependencies(numJobs);
        std::string line;

        // Read each line until we reach the end marker
        while (std::getline(file, line) && line != endMarker) {
            if (line.empty() || line[0] == '#') continue; // Skip comments or empty lines

            size_t colonPos = line.find(':');
            if (colonPos == std::string::npos) {
                throw std::runtime_error("Invalid job dependency format, missing job index.");
            }

            // Extract the job index
            int jobIndex = std::stoi(trim(line.substr(0, colonPos)));

            // The remaining part contains the dependencies for that job
            std::string dependenciesStr = line.substr(colonPos + 1);
            std::vector<int> depJobs;

            if (!dependenciesStr.empty()) {
                depJobs = parseDependencyList(dependenciesStr);
            }

            // Store the dependencies for this job
            jobDependencies[jobIndex] = depJobs;
        }

        return jobDependencies;
    }

public:
    static std::pair<std::vector<Job>, int> generateFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        std::string line;

        // Read metadata
        while (std::getline(file, line) && line != "[METADATA]");
        auto metadata = readMetadata(file);
        int numJobs = metadata["num_jobs"];
        int numMachines = metadata["num_machines"];

        // Read job dependencies (sparse format)
        while (std::getline(file, line) && line != "[JOB DEPENDENCIES]");
        auto jobDependencies = readJobDependenciesSparse(file, numJobs, "[JOB DEPENDENCIES END]");

        // Read operation durations
        while (std::getline(file, line) && line != "[OPERATION DURATIONS]");
        auto operationDurations = readOperationDurations(file, numJobs, "[OPERATION DURATIONS END]");

        // Read operation dependencies
        while (std::getline(file, line) && line != "[OPERATION DEPENDENCIES]");
        auto operationDependencies = readOperationDependencies(file, numJobs, "[OPERATION DEPENDENCIES END]");

        std::vector<Job> jobs;
        jobs.reserve(numJobs);

        int numOperations = 0;

        // Process each job
        for (int i = 0; i < numJobs; ++i) {
            Job job;
            int numOps = operationDurations[i].size();
            numOperations = std::max(numOperations, numOps);

            // Process each operation in the job
            for (int opIndex = 0; opIndex < numOps; ++opIndex) {
                Operation op;
                op.duration = operationDurations[i][opIndex];
                op.machine = (i + opIndex) % numMachines; // Assign machines as needed
                op.eligibleMachines.set(op.machine);

                // Set operation dependencies within the job
                if (!operationDependencies[i].empty()) {
                    auto& opDeps = operationDependencies[i]; // Access the unordered_map
                    if (opDeps.find(opIndex) != opDeps.end()) {
                        for (int depOpIndex : opDeps[opIndex]) {
                            op.dependentOperations.emplace_back(i, depOpIndex);
                        }
                    }
                }

                job.operations.push_back(op);
            }

            // Set job dependencies (across different jobs)
            job.dependentJobs = jobDependencies[i];

            jobs.push_back(job);
        }

        // Calculate the optimal makespan (the total duration across all operations)
        int optimalMakespan = 0;
        for (const auto& durations : operationDurations) {
            optimalMakespan += std::accumulate(durations.begin(), durations.end(), 0);
        }

        return {jobs, optimalMakespan};
    }

};
