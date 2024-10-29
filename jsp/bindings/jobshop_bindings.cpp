
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <optional>
#include "environment/job_shop_environment.h"
#include "algorithms/job_shop_qlearning.h"
#include "algorithms/job_shop_actor_critic.h"
#include "environment/job_shop_taillard_generator.h"
#include "gui/job_shop_plotter.h"
#include "algorithms/job_shop_algorithms.h"
#include "environment/job_shop_environment_generator.h"
#include "utilities/multidimensional_array.hpp"
#include "environment/job_shop_manual_generator.h"

namespace nb = nanobind;

template<typename T, std::size_t NDim>
void bind_multi_dim_array(nb::module_ &m, const char* name) {
    nb::class_<MultiDimensionalArray<T, NDim>>(m, name)
            .def(nb::init<const std::array<std::size_t, NDim>&>())
            .def("__array__", [](MultiDimensionalArray<T, NDim>& arr, nb::kwargs kwargs) {
                bool copy = kwargs.contains("copy") && nb::cast<bool>(kwargs["copy"]);

                // throw if copy is requested
                if (copy) {
                    throw std::runtime_error("copy is not supported");
                }

                std::vector<size_t> shape(arr.shape().begin(), arr.shape().end());

                nb::handle owner = nb::cast(&arr, nb::rv_policy::reference_internal);

                return nb::ndarray<T, nb::numpy, nb::shape<>>(
                        arr.data_ptr(),
                        shape.size(),
                        shape.data(),
                        owner,
                        nullptr, // strides.data(),
                        nb::dtype<T>(),
                        0,  // device_type (CPU)
                        0   // device_id
                );
            }, nb::rv_policy::reference_internal)
            .def("shape", &MultiDimensionalArray<T, NDim>::shape)
            .def("size", &MultiDimensionalArray<T, NDim>::size)
            .def("fill", &MultiDimensionalArray<T, NDim>::fill);
}

NB_MODULE(jobshop, m) {

    bind_multi_dim_array<float, 1>(m, "MultiDimArray1f");
    bind_multi_dim_array<float, 2>(m, "MultiDimArray2f");
    bind_multi_dim_array<float, 3>(m, "MultiDimArray3f");
    bind_multi_dim_array<double, 1>(m, "MultiDimArray1d");
    bind_multi_dim_array<double, 2>(m, "MultiDimArray2d");
    bind_multi_dim_array<double, 3>(m, "MultiDimArray3d");
    bind_multi_dim_array<int, 1>(m, "MultiDimArray1i");
    bind_multi_dim_array<int, 2>(m, "MultiDimArray2i");
    bind_multi_dim_array<int, 3>(m, "MultiDimArray3i");

    // Bind Operation struct
    nb::class_<Operation>(m, "Operation")
            .def(nb::init<>())
            .def(nb::init<int, int>(), nb::arg("duration"), nb::arg("machine"))
            .def_rw("duration", &Operation::duration)
            .def_rw("machine", &Operation::machine)
            .def_rw("eligibleMachines", &Operation::eligibleMachines)
            .def_rw("dependentOperations", &Operation::dependentOperations)
            .def("__getstate__", [](const Operation &op) {
                return nb::make_tuple(
                        op.duration,
                        op.machine,
                        op.eligibleMachines.to_string(),  // Convert bitset to string
                        op.dependentOperations
                );
            })
            .def("__setstate__", [](Operation &op, nb::tuple state) {
                if (state.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }
                op.duration = nb::cast<int>(state[0]);
                op.machine = nb::cast<int>(state[1]);
                op.eligibleMachines = std::bitset<MAX_MACHINES>(nb::cast<std::string>(state[2]));  // Convert string back to bitset
                op.dependentOperations = nb::cast<std::vector<std::pair<int, int>>>(state[3]);
            });

    // Bind Job struct
    nb::class_<Job>(m, "Job")
            .def(nb::init<>())
            .def_rw("operations", &Job::operations)
            .def_rw("dependentJobs", &Job::dependentJobs)
            .def("__getstate__", [](const Job &job) {
                return std::make_tuple(job.operations, job.dependentJobs);
            })
            .def("__setstate__", [](Job &job, const std::tuple<std::vector<Operation>, std::vector<int>> &state) {
                new (&job) Job{
                        std::get<0>(state),
                        std::get<1>(state)
                };
            });

    // Bind Action struct
    nb::class_<Action>(m, "Action")
            .def(nb::init<>())
            .def(nb::init<int, int, int>())
            .def_rw("job", &Action::job)
            .def_rw("machine", &Action::machine)
            .def_rw("operation", &Action::operation)
            .def("__eq__", [](const Action &a1, const Action &a2) {
                return a1 == a2;
            });

    // Bind State class
    nb::class_<State>(m, "State")
            .def(nb::init<int, int, int>())
            .def_rw("jobProgress", &State::jobProgress)
            .def_rw("machineAvailability", &State::machineAvailability)
            .def_rw("nextOperationForJob", &State::nextOperationForJob)
            .def_rw("completedJobs", &State::completedJobs)
            .def_rw("jobStartTimes", &State::jobStartTimes);

    // Bind ScheduleEntry struct
    nb::class_<ScheduleEntry>(m, "ScheduleEntry")
            .def(nb::init<>())
            .def(nb::init<int, int, int, int>(), nb::arg("job"), nb::arg("operation"), nb::arg("start"), nb::arg("duration"))
            .def_rw("job", &ScheduleEntry::job)
            .def_rw("operation", &ScheduleEntry::operation)
            .def_rw("start", &ScheduleEntry::start)
            .def_rw("duration", &ScheduleEntry::duration)
            .def("__getstate__", [](const ScheduleEntry &se) {
                return std::make_tuple(se.job, se.operation, se.start, se.duration);
            })
            .def("__setstate__", [](ScheduleEntry &se, const std::tuple<int, int, int, int> &state) {
                new (&se) ScheduleEntry{
                        std::get<0>(state),
                        std::get<1>(state),
                        std::get<2>(state),
                        std::get<3>(state)
                };
            });

    // Bind JobShopEnvironment class
    nb::class_<JobShopEnvironment>(m, "JobShopEnvironment")
            .def(nb::init<std::vector<Job>>(), nb::arg("jobs"))
            .def("step", &JobShopEnvironment::step, nb::arg("action"))
            .def("reset", [](JobShopEnvironment &env, nb::kwargs kwargs) {
                std::optional<unsigned int> seed;
                if (kwargs.contains("seed")) {
                    seed = nb::cast<unsigned int>(kwargs["seed"]);
                }
                return env.reset();
            }, nb::arg("seed") = nb::none(), "Reset the environment. Optionally provide a seed for randomization.")
            .def("getState", &JobShopEnvironment::getState, nb::rv_policy::reference_internal)
            .def("getPossibleActions", &JobShopEnvironment::getPossibleActions, nb::rv_policy::reference_internal)
            .def("isDone", &JobShopEnvironment::isDone)
            .def("getTotalTime", &JobShopEnvironment::getTotalTime)
            .def("getJobs", &JobShopEnvironment::getJobs, nb::rv_policy::reference_internal)
            .def("getNumMachines", &JobShopEnvironment::getNumMachines)
            .def("getScheduleData", &JobShopEnvironment::getScheduleData)
            .def("printSchedule", &JobShopEnvironment::printSchedule)
            .def("generateOperationGraph", &JobShopEnvironment::generateOperationGraph, nb::arg("filename"));

    // Bind JobShopAlgorithm abstract class
    nb::class_<JobShopAlgorithm>(m, "JobShopAlgorithm")
            .def("train", &JobShopAlgorithm::train)
            .def("printBestSchedule", &JobShopAlgorithm::printBestSchedule)
            .def("saveBestScheduleToFile", &JobShopAlgorithm::saveBestScheduleToFile);

    // Bind JobShopQLearning class
    nb::class_<JobShopQLearning>(m, "JobShopQLearning")
            .def(nb::init<JobShopEnvironment&, double, double, double>())
            .def("runEpisode", &JobShopQLearning::runEpisode)
            .def("train", &JobShopQLearning::train)
            .def("printBestSchedule", &JobShopQLearning::printBestSchedule)
            .def("saveBestScheduleToFile", &JobShopQLearning::saveBestScheduleToFile)
            .def("applyAndPrintSchedule", &JobShopQLearning::applyAndPrintSchedule);

    // Bind JobShopActorCritic class
    nb::class_<JobShopActorCritic>(m, "JobShopActorCritic")
            .def(nb::init<JobShopEnvironment&, double, double, double>())
            .def("runEpisode", &JobShopActorCritic::runEpisode)
            .def("train", &JobShopActorCritic::train)
            .def("printBestSchedule", &JobShopActorCritic::printBestSchedule)
            .def("saveBestScheduleToFile", &JobShopActorCritic::saveBestScheduleToFile)
            .def("applyAndPrintSchedule", &JobShopActorCritic::applyAndPrintSchedule);

    // Bind TaillardJobShopGenerator class
    nb::class_<TaillardJobShopGenerator>(m, "TaillardJobShopGenerator")
            .def_static("loadProblem", &TaillardJobShopGenerator::loadProblem, nb::arg("filePath"))
            .def_static("verifyJobsData", &TaillardJobShopGenerator::verifyJobsData, nb::arg("jobs"));

    // Bind ManualJobShopGenerator class
    nb::class_<ManualJobShopGenerator>(m, "ManualJobShopGenerator")
            .def_static("generateFromFile", &ManualJobShopGenerator::generateFromFile, nb::arg("filename"))
            .def_static("generateFromData", &ManualJobShopGenerator::generateFromData, nb::arg("jobShopData"));

    // Bind AutomaticJobShopGenerator class
    nb::class_<AutomaticJobShopGenerator>(m, "AutomaticJobShopGenerator")
            .def_static("generate", &AutomaticJobShopGenerator::generate, nb::arg("params"))
            .def_static("generateDefault", &AutomaticJobShopGenerator::generateDefault,
                        nb::arg("numJobs"), nb::arg("numMachines"), nb::arg("outputFile") = std::nullopt);

    // Bind GenerationParams struct
    nb::class_<AutomaticJobShopGenerator::GenerationParams>(m, "GenerationParams")
            .def(nb::init<>())
            .def_rw("numJobs", &AutomaticJobShopGenerator::GenerationParams::numJobs)
            .def_rw("numMachines", &AutomaticJobShopGenerator::GenerationParams::numMachines)
            .def_rw("minDuration", &AutomaticJobShopGenerator::GenerationParams::minDuration)
            .def_rw("maxDuration", &AutomaticJobShopGenerator::GenerationParams::maxDuration)
            .def_rw("dependencyDensity", &AutomaticJobShopGenerator::GenerationParams::dependencyDensity)
            .def_rw("maxDependenciesPerJob", &AutomaticJobShopGenerator::GenerationParams::maxDependenciesPerJob)
            .def_rw("longJobRate", &AutomaticJobShopGenerator::GenerationParams::longJobRate)
            .def_rw("longJobFactor", &AutomaticJobShopGenerator::GenerationParams::longJobFactor)
            .def_rw("outputFile", &AutomaticJobShopGenerator::GenerationParams::outputFile);

    // Bind LivePlotter class
    nb::class_<LivePlotter>(m, "LivePlotter")
            .def(nb::init<int, const std::vector<Job>&>(), nb::arg("machines"), nb::arg("jobs"))
            .def("render", &LivePlotter::render)
            .def("updateSchedule", &LivePlotter::updateSchedule, nb::arg("newSchedule"), nb::arg("newTotalTime"))
            .def("shouldClose", &LivePlotter::shouldClose);
}
