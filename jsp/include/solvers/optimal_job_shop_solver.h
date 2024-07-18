#pragma once

#include <ortools/sat/cp_model.h>
#include "job_shop_environment.h"

class OptimalJobShopSolver {
private:
    const JobShopEnvironment& env;

public:
    OptimalJobShopSolver(const JobShopEnvironment& environment) : env(environment) {}

    int solve() {
        operations_research::sat::CpModel model;

        const std::vector<Job>& jobs = env.getJobs();
        int num_machines = env.getNumMachines();

        // Create variables
        std::vector<std::vector<operations_research::sat::IntVar>> start_vars;
        std::vector<std::vector<operations_research::sat::IntVar>> end_vars;
        operations_research::sat::IntVar obj_var = model.NewIntVar({0, INT_MAX}, "makespan");

        for (const auto& job : jobs) {
            std::vector<operations_research::sat::IntVar> job_start_vars;
            std::vector<operations_research::sat::IntVar> job_end_vars;
            for (size_t i = 0; i < job.operations.size(); ++i) {
                auto start = model.NewIntVar({0, INT_MAX}, "");
                auto end = model.NewIntVar({0, INT_MAX}, "");
                job_start_vars.push_back(start);
                job_end_vars.push_back(end);
                model.AddLessOrEqual(end, obj_var);
            }
            start_vars.push_back(job_start_vars);
            end_vars.push_back(job_end_vars);
        }

        // Add constraints
        for (size_t j = 0; j < jobs.size(); ++j) {
            for (size_t i = 0; i < jobs[j].operations.size(); ++i) {
                int duration = jobs[j].operations[i];
                model.AddEquality(end_vars[j][i], start_vars[j][i] + duration);
                if (i > 0) {
                    model.AddLessOrEqual(end_vars[j][i-1], start_vars[j][i]);
                }
            }
        }

        std::vector<std::vector<operations_research::sat::IntervalVar>> machine_intervals(num_machines);
        for (size_t j = 0; j < jobs.size(); ++j) {
            for (size_t i = 0; i < jobs[j].operations.size(); ++i) {
                int machine = jobs[j].machines[i];
                int duration = jobs[j].operations[i];
                auto interval = model.NewIntervalVar(start_vars[j][i], duration, end_vars[j][i], "");
                machine_intervals[machine].push_back(interval);
            }
        }

        for (const auto& intervals : machine_intervals) {
            model.AddNoOverlap(intervals);
        }

        // Set objective
        model.Minimize(obj_var);

        // Solve
        operations_research::sat::CpSolver solver;
        operations_research::sat::CpSolverResponse response = solver.Solve(model);

        if (response.status() == operations_research::sat::CpSolverStatus::OPTIMAL ||
            response.status() == operations_research::sat::CpSolverStatus::FEASIBLE) {
            return solver.ObjectiveValue();
        } else {
            return -1;  // No solution found
        }
    }
};