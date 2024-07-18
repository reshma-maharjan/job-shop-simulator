#ifndef JSP_DOTFILE_H
#define JSP_DOTFILE_H

#include <fstream>
#include <string>
#include <vector>
#include "job.h"

class DotFile {
public:
    static void writeJobsAndOperations(const std::vector<std::shared_ptr<Job>>& jobs, std::ofstream& dotFile) {
        for (const auto& job : jobs) {
            dotFile << "    subgraph cluster_job" << job->jobID << " {" << std::endl;
            dotFile << "        label=\"Job " << job->jobID << "\";" << std::endl;
            dotFile << "        color=blue;" << std::endl;
            dotFile << "        style=filled;" << std::endl;
            dotFile << "        fillcolor=lightblue;" << std::endl;

            // We still define a dummy node for aesthetic reasons or future use
            std::string dummyNodeID = "dummyJob" + std::to_string(job->jobID);
            dotFile << "        " << dummyNodeID << " [style=invis];" << std::endl;

            for (const auto& operation : job->operations) {
                std::string operationNodeID = "Job" + std::to_string(job->jobID) + "Op" + std::to_string(operation->operationID);
                dotFile << "        " << operationNodeID << " [label=\"Op " << operation->operationID
                        << "\", shape=ellipse, color=black, style=filled, fillcolor=lightgray];" << std::endl;
            }

            dotFile << "    }" << std::endl; // Close subgraph
        }
    }

    static void writeOperationDependencies(const std::vector<std::shared_ptr<Job>>& jobs, std::ofstream& dotFile) {
        for (const auto& job : jobs) {
            for (const auto& operation : job->operations) {
                for (const auto& dependency : operation->dependencies) {
                    // Ensure both operation and dependency IDs are prefixed with jobID
                    std::string operationNodeID = "Job" + std::to_string(job->jobID) + "Op" + std::to_string(operation->operationID);
                    std::string dependencyNodeID = "Job" + std::to_string(job->jobID) + "Op" + std::to_string(dependency->operationID);
                    dotFile << "    " << dependencyNodeID << " -> " << operationNodeID
                            << " [color=black];" << std::endl;
                }
            }
        }
    }

    static void writeJobDependencies(const std::vector<std::shared_ptr<Job>>& jobs, std::ofstream& dotFile) {
        // For each job, create an outside node representing the job itself
        for (const auto& job : jobs) {
            dotFile << "    JobNode" << job->jobID << " [label=\"Job " << job->jobID
                    << "\", shape=box, style=filled, fillcolor=lightblue, color=blue];" << std::endl;
        }

        // Now, draw dependencies from job nodes to subgraphs
        for (const auto& job : jobs) {
            for (const auto& depJob : job->jobDependencies) {
                // Since we cannot directly link to a subgraph, use the lhead attribute
                // to visually indicate the edge is entering the subgraph of the dependent job.
                dotFile << "    JobNode" << depJob->jobID << " -> JobNode" << job->jobID
                        << " [ltail=cluster_job" << depJob->jobID << ", lhead=cluster_job" << job->jobID
                        << ", style=dotted, color=red];" << std::endl;
            }
        }
    }

    static void generateDotFile(const std::vector<std::shared_ptr<Job>>& jobs, const std::string& filename) {
        std::ofstream dotFile(filename);
        dotFile << "digraph G {" << std::endl;
        dotFile << "    graph [nodesep=1, ranksep=\"1 equally\"];" << std::endl;
        dotFile << "    node [style=filled];" << std::endl;

        writeJobsAndOperations(jobs, dotFile);
        writeOperationDependencies(jobs, dotFile);
        writeJobDependencies(jobs, dotFile);

        dotFile << "}" << std::endl;
        dotFile.close();

#ifdef _WIN32
        std::string command = R"("C:\Program Files\Graphviz\bin\dot.exe" -Tpng ")" + filename + "\" -o \"" + filename + ".png\"";
        system(("start \"\" " + command).c_str());
#else
        std::string command = "dot -Tpng \"" + filename + "\" -o \"" + filename + ".png\"";
        system(command.c_str());
#endif
    }
};

#endif //JSP_DOTFILE_H
