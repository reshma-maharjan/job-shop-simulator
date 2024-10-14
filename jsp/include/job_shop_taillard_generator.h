#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <curl/curl.h>
#include "job_shop_environment.h"
#include "fmt/format.h"
#include "util.h"

enum class TaillardInstance {
    TA01 = 84, TA02 = 85, TA03 = 86, TA04 = 87, TA05 = 88,
    TA06 = 89, TA07 = 90, TA08 = 91, TA09 = 92, TA10 = 93,
    TA11 = 94, TA12 = 95, TA13 = 96, TA14 = 97, TA15 = 98,
    TA16 = 99, TA17 = 100, TA18 = 101, TA19 = 102, TA20 = 103,
    TA21 = 104, TA22 = 105, TA23 = 106, TA24 = 107, TA25 = 108,
    TA26 = 109, TA27 = 110, TA28 = 111, TA29 = 112, TA30 = 113,
    TA31 = 114, TA32 = 115, TA33 = 116, TA34 = 117, TA35 = 118,
    TA36 = 119, TA37 = 120, TA38 = 121, TA39 = 122, TA40 = 123,
    TA41 = 124, TA42 = 125, TA43 = 126, TA44 = 127, TA45 = 128,
    TA46 = 129, TA47 = 130, TA48 = 131, TA49 = 132, TA50 = 133,
    TA51 = 134, TA52 = 135, TA53 = 136, TA54 = 137, TA55 = 138,
    TA56 = 139, TA57 = 140, TA58 = 141, TA59 = 142, TA60 = 143,
    TA61 = 144, TA62 = 145, TA63 = 146, TA64 = 147, TA65 = 148,
    TA66 = 149, TA67 = 150, TA68 = 151, TA69 = 152, TA70 = 153,
    TA71 = 154, TA72 = 155, TA73 = 156, TA74 = 157, TA75 = 158,
    TA76 = 159, TA77 = 160, TA78 = 161, TA79 = 162, TA80 = 163
};

inline std::string getTaillardInstanceName(TaillardInstance instance) {
    return "ta" + std::to_string(static_cast<int>(instance) - 83);
}

inline int getTaillardInstanceId(TaillardInstance instance) {
    return static_cast<int>(instance);
}

class TaillardJobShopGenerator {
private:
    TaillardJobShopGenerator() = default;

public:
    static std::pair<std::vector<Job>, int> loadProblem(TaillardInstance instance, bool showProgress = true) {
        int instanceId = getTaillardInstanceId(instance);
        std::string url = "http://jobshop.jjvh.nl/specification_file.php?instance_id=" + std::to_string(instanceId) + "&specification=taillard";
        std::string problemData = CurlUtility::FetchUrl(url, showProgress);

        std::istringstream dataStream(problemData);

        int numJobs, numMachines;
        dataStream >> numJobs >> numMachines;

        std::vector<Job> jobs(numJobs);

        // Read processing times
        for (int i = 0; i < numJobs; ++i) {
            jobs[i].operations.resize(numMachines);
            for (int j = 0; j < numMachines; ++j) {
                dataStream >> jobs[i].operations[j].duration;
            }
        }

        // Read machine orders
        for (int i = 0; i < numJobs; ++i) {
            for (int j = 0; j < numMachines; ++j) {
                int machine;
                dataStream >> machine;
                jobs[i].operations[j].machine = machine - 1;  // Adjust for 0-based indexing

                // Set only the assigned machine as eligible
                jobs[i].operations[j].eligibleMachines.reset();
                jobs[i].operations[j].eligibleMachines.set(jobs[i].operations[j].machine);
            }
        }

        int optimalMakespan = loadOptimalMakespan(instance);
        return {jobs, optimalMakespan};
    }

    static void verifyJobsData(const std::vector<Job>& jobs) {
        if (jobs.empty()) {
            throw std::runtime_error("Jobs vector is empty");
        }

        int numJobs = jobs.size();
        int numMachines = jobs[0].operations.size();

        for (int i = 0; i < numJobs; ++i) {
            if (jobs[i].operations.size() != numMachines) {
                throw std::runtime_error(fmt::format("Job {} has incorrect number of operations", i));
            }

            for (int j = 0; j < numMachines; ++j) {
                if (jobs[i].operations[j].duration <= 0) {
                    throw std::runtime_error(fmt::format("Job {} operation {} has invalid duration", i, j));
                }
                if (jobs[i].operations[j].machine < 0 || jobs[i].operations[j].machine >= numMachines) {
                    throw std::runtime_error(fmt::format("Job {} operation {} has invalid machine index", i, j));
                }
                if (jobs[i].operations[j].eligibleMachines.count() != 1) {
                    throw std::runtime_error(fmt::format("Job {} operation {} has incorrect number of eligible machines", i, j));
                }
            }
        }

        std::cout << "Jobs data verification passed successfully." << std::endl;
    }

    static void verifyOptimalSolution(const std::vector<Job>& jobs, const std::vector<int>& optimalSolution) {
        int numJobs = jobs.size();
        int numMachines = jobs[0].operations.size();
        int totalOperations = numJobs * numMachines;

        if (optimalSolution.size() != totalOperations) {
            throw std::runtime_error(fmt::format("Optimal solution has incorrect number of operations. Expected {}, got {}",
                                                 totalOperations, optimalSolution.size()));
        }

        std::vector<int> operationCounts(numJobs, 0);
        std::vector<std::vector<bool>> machineUsed(numJobs, std::vector<bool>(numMachines, false));

        for (int op : optimalSolution) {
            if (op < 0 || op >= totalOperations) {
                throw std::runtime_error(fmt::format("Invalid operation index in optimal solution: {}", op));
            }

            int job = op % numJobs;
            int machineIndex = operationCounts[job];

            if (machineIndex >= numMachines) {
                throw std::runtime_error(fmt::format("Too many operations for job {} in optimal solution", job));
            }

            int machine = jobs[job].operations[machineIndex].machine;
            if (machineUsed[job][machine]) {
                throw std::runtime_error(fmt::format("Machine {} used twice for job {} in optimal solution", machine, job));
            }

            machineUsed[job][machine] = true;
            operationCounts[job]++;
        }

        for (int i = 0; i < numJobs; ++i) {
            if (operationCounts[i] != numMachines) {
                throw std::runtime_error(fmt::format("Job {} has incorrect number of operations in optimal solution", i));
            }
        }

        std::cout << "Optimal solution verification passed successfully." << std::endl;
    }

    static void runAllVerifications(const std::vector<Job>& jobs, const std::vector<int>& optimalSolution) {
        verifyJobsData(jobs);
        verifyOptimalSolution(jobs, optimalSolution);
        std::cout << "All verifications passed successfully." << std::endl;
    }

private:


    static int loadOptimalMakespan(TaillardInstance instance, bool showProgress = true) {
        auto iD = getTaillardInstanceId(instance);
        std::string url = fmt::format("http://jobshop.jjvh.nl/solutions_file.php?instance_id={}", iD);
        std::string solutionData = CurlUtility::FetchUrl(url, showProgress);

        std::istringstream dataStream(solutionData);
        int optimalMakespan;
        dataStream >> optimalMakespan;

        return optimalMakespan;
    }



    static int calculateOptimalMakespan(const std::vector<Job>& jobs, const std::vector<int>& optimalSolution) {
        int numJobs = jobs.size();
        int numMachines = jobs[0].operations.size();

        std::vector<int> jobCompletionTimes(numJobs, 0);
        std::vector<int> machineAvailability(numMachines, 0);
        int optimalMakespan = 0;

        for (int op : optimalSolution) {
            int job = op % numJobs;
            int opIndex = op / numJobs;
            int machine = jobs[job].operations[opIndex].machine;
            int duration = jobs[job].operations[opIndex].duration;

            int startTime = std::max(jobCompletionTimes[job], machineAvailability[machine]);
            int endTime = startTime + duration;

            jobCompletionTimes[job] = endTime;
            machineAvailability[machine] = endTime;
            optimalMakespan = std::max(optimalMakespan, endTime);
        }

        return optimalMakespan;
    }
};