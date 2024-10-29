#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <vector>
#include <algorithm>
#include <string>
#include <thread>
#include <queue>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <fstream>
#include "environment/job_shop_environment.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

class LivePlotter {
private:
    GLFWwindow* window;
    std::vector<std::vector<ScheduleEntry>> scheduleData;
    int numMachines;
    int totalTime;
    std::vector<ImVec4> jobColors;
    float transitionProgress;
    std::vector<std::vector<ScheduleEntry>> oldScheduleData;
    std::queue<std::pair<std::vector<std::vector<ScheduleEntry>>, int>> updateQueue;
    int currentVisualizationMakespan;
    const std::vector<Job>& jobs; // Store jobs data
    float zoomLevel;
    ImVec2 panOffset;
    bool captureScreenshot;

    void initializeGLFW() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(1280, 720, "Job Shop Schedule Live Plotter", NULL, NULL);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        if (glewInit() != GLEW_OK) {
            throw std::runtime_error("Failed to initialize GLEW");
        }
    }

    void initializeImGui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        ImGui::StyleColorsDark();
    }

    void generateJobColors(int numJobs) {
        jobColors.clear();
        jobColors.reserve(numJobs);
        for (int i = 0; i < numJobs; ++i) {
            float hue = i * (1.0f / numJobs);
            ImVec4 color = ImColor::HSV(hue, 0.6f, 0.6f);
            jobColors.push_back(color);
        }
    }

    ImVec4 lerpColor(const ImVec4& a, const ImVec4& b, float t) {
        return ImVec4(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t
        );
    }

    // Helper function to compute levels of each operation based on dependencies
    void computeOperationLevels(std::vector<std::vector<int>>& opLevels, int& maxLevel) {
        maxLevel = 0;
        opLevels.resize(jobs.size());
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            opLevels[jobIndex].resize(job.operations.size(), -1);
        }

        // Helper lambda to recursively compute levels
        std::function<int(int, int)> computeLevel = [&](int jobIndex, int opIndex) -> int {
            if (opLevels[jobIndex][opIndex] != -1) {
                return opLevels[jobIndex][opIndex];
            }

            int level = 0;

            // Dependencies from previous operation in the same job
            if (opIndex > 0) {
                level = std::max(level, computeLevel(jobIndex, opIndex - 1) + 1);
            }

            // Job dependencies
            for (int depJobIndex : jobs[jobIndex].dependentJobs) {
                int depJobLastOpIndex = jobs[depJobIndex].operations.size() - 1;
                level = std::max(level, computeLevel(depJobIndex, depJobLastOpIndex) + 1);
            }

            // Operation dependencies
            const Operation& op = jobs[jobIndex].operations[opIndex];
            for (const auto& dep : op.dependentOperations) {
                int depJobIndex = dep.first;
                int depOpIndex = dep.second;
                level = std::max(level, computeLevel(depJobIndex, depOpIndex) + 1);
            }

            opLevels[jobIndex][opIndex] = level;
            maxLevel = std::max(maxLevel, level);
            return level;
        };

        // Compute levels for all operations
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            for (size_t opIndex = 0; opIndex < jobs[jobIndex].operations.size(); ++opIndex) {
                computeLevel(jobIndex, opIndex);
            }
        }
    }

    // Function to draw operation graph with improved layout and zooming/panning
    void drawOperationGraph() {
        ImGui::Text("Operation Graph");

        // Add zoom and pan controls
        ImGui::SliderFloat("Zoom", &zoomLevel, 0.1f, 2.0f, "%.1f");
        ImGui::Text("Pan: X=%.1f, Y=%.1f", panOffset.x, panOffset.y);

        // Create a child window
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();
        ImGui::BeginChild("OperationGraphRegion", canvas_size, false, ImGuiWindowFlags_NoMove);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

        // Handle panning with mouse dragging
        ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            panOffset.x += io.MouseDelta.x;
            panOffset.y += io.MouseDelta.y;
        }

        // Compute operation levels
        std::vector<std::vector<int>> opLevels;
        int maxLevel = 0;
        computeOperationLevels(opLevels, maxLevel);

        // Calculate layout parameters
        float node_width = 100.0f * zoomLevel;
        float node_height = 40.0f * zoomLevel;
        float level_spacing = 150.0f * zoomLevel;
        float job_spacing = 80.0f * zoomLevel;

        // Generate positions for nodes
        std::unordered_map<std::string, ImVec2> node_positions;
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            for (size_t opIndex = 0; opIndex < job.operations.size(); ++opIndex) {
                int level = opLevels[jobIndex][opIndex];
                float x = canvas_pos.x + level * level_spacing + panOffset.x + 50;
                float y = canvas_pos.y + jobIndex * job_spacing + panOffset.y + 50;
                std::string nodeKey = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex);
                node_positions[nodeKey] = ImVec2(x, y);
            }
        }

        // Draw edges (dependencies)
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            for (size_t opIndex = 0; opIndex < job.operations.size(); ++opIndex) {
                const Operation& op = job.operations[opIndex];
                std::string fromNodeKey = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex);
                ImVec2 from_pos = node_positions[fromNodeKey];

                // Edge to next operation in the same job
                if (opIndex + 1 < job.operations.size()) {
                    std::string toNodeKey = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex + 1);
                    ImVec2 to_pos = node_positions[toNodeKey];
                    draw_list->AddBezierCubic(
                            ImVec2(from_pos.x + node_width, from_pos.y + node_height / 2),
                            ImVec2(from_pos.x + node_width + 50 * zoomLevel, from_pos.y + node_height / 2),
                            ImVec2(to_pos.x - 50 * zoomLevel, to_pos.y + node_height / 2),
                            ImVec2(to_pos.x, to_pos.y + node_height / 2),
                            IM_COL32(0, 0, 255, 255), 2.0f);
                }

                // Draw operation dependencies
                for (const auto& dep : op.dependentOperations) {
                    int depJobIndex = dep.first;
                    int depOpIndex = dep.second;
                    std::string depNodeKey = "J" + std::to_string(depJobIndex) + "_O" + std::to_string(depOpIndex);
                    ImVec2 dep_pos = node_positions[depNodeKey];
                    draw_list->AddBezierCubic(
                            ImVec2(dep_pos.x + node_width, dep_pos.y + node_height / 2),
                            ImVec2(dep_pos.x + node_width + 50 * zoomLevel, dep_pos.y + node_height / 2),
                            ImVec2(from_pos.x - 50 * zoomLevel, from_pos.y + node_height / 2),
                            ImVec2(from_pos.x, from_pos.y + node_height / 2),
                            IM_COL32(0, 255, 0, 255), 1.5f);
                }
            }
        }

        // Draw job dependencies
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            std::string firstNodeKey = "J" + std::to_string(jobIndex) + "_O0";
            ImVec2 first_op_pos = node_positions[firstNodeKey];
            for (int depJobIndex : job.dependentJobs) {
                const Job& depJob = jobs[depJobIndex];
                size_t depOpIndex = depJob.operations.size() - 1;
                std::string depNodeKey = "J" + std::to_string(depJobIndex) + "_O" + std::to_string(depOpIndex);
                ImVec2 dep_op_pos = node_positions[depNodeKey];
                draw_list->AddBezierCubic(
                        ImVec2(dep_op_pos.x + node_width, dep_op_pos.y + node_height / 2),
                        ImVec2(dep_op_pos.x + node_width + 50 * zoomLevel, dep_op_pos.y + node_height / 2),
                        ImVec2(first_op_pos.x - 50 * zoomLevel, first_op_pos.y + node_height / 2),
                        ImVec2(first_op_pos.x, first_op_pos.y + node_height / 2),
                        IM_COL32(255, 0, 0, 255), 1.5f);
            }
        }

        // Draw nodes
        for (size_t jobIndex = 0; jobIndex < jobs.size(); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            ImVec4 color = jobColors[jobIndex % jobColors.size()];
            for (size_t opIndex = 0; opIndex < job.operations.size(); ++opIndex) {
                const Operation& op = job.operations[opIndex];
                std::string nodeKey = "J" + std::to_string(jobIndex) + "_O" + std::to_string(opIndex);
                ImVec2 pos = node_positions[nodeKey];
                ImVec2 rect_min(pos.x, pos.y);
                ImVec2 rect_max(pos.x + node_width, pos.y + node_height);
                draw_list->AddRectFilled(rect_min, rect_max, ImColor(color));
                draw_list->AddRect(rect_min, rect_max, IM_COL32(0, 0, 0, 255));

                // Add text
                std::string label = "J" + std::to_string(jobIndex) + " O" + std::to_string(opIndex);
                ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
                ImVec2 text_pos(rect_min.x + (node_width - text_size.x) / 2, rect_min.y + (node_height - text_size.y) / 2);
                draw_list->AddText(text_pos, IM_COL32(0, 0, 0, 255), label.c_str());
            }
        }

        ImGui::EndChild();

        // Add a button to capture the screenshot
        if (ImGui::Button("Capture Screenshot")) {
            captureScreenshot = true;
        }
    }

    // Function to capture the screenshot of the current window
    void captureWindow() {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        std::vector<unsigned char> pixels(width * height * 4); // RGBA

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        // Flip the image vertically
        int rowSize = width * 4;
        std::vector<unsigned char> flippedPixels(pixels.size());
        for (int y = 0; y < height; y++) {
            memcpy(&flippedPixels[y * rowSize], &pixels[(height - y - 1) * rowSize], rowSize);
        }

        // Save the image using stb_image_write
        std::string filename = "screenshot.png";
        stbi_write_png(filename.c_str(), width, height, 4, flippedPixels.data(), width * 4);
        std::cout << "Screenshot saved to " << filename << std::endl;
        captureScreenshot = false;
    }

public:
    LivePlotter(int machines, const std::vector<Job>& jobs)
            : numMachines(machines), jobs(jobs), totalTime(0), transitionProgress(1.0f), currentVisualizationMakespan(0), zoomLevel(1.0f), panOffset(0, 0), captureScreenshot(false) {
        initializeGLFW();
        initializeImGui();
        scheduleData.resize(numMachines);
        oldScheduleData.resize(numMachines);

        generateJobColors(static_cast<int>(jobs.size()));
    }

    ~LivePlotter() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void updateSchedule(const std::vector<std::vector<ScheduleEntry>>& newSchedule, int newTotalTime) {
        updateQueue.push({newSchedule, newTotalTime});
    }

    void render() {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create main window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Job Shop Schedule", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        // Create tab bar
        if (ImGui::BeginTabBar("MainTabBar")) {

            // Schedule Visualization Tab
            if (ImGui::BeginTabItem("Schedule Visualization")) {
                // Add title for current visualization
                ImGui::SetCursorPosY(10);
                ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("Current Makespan: 0").x) / 2);
                ImGui::Text("Current Makespan: %d", currentVisualizationMakespan);

                // Add zoom slider
                ImGui::SliderFloat("Zoom", &zoomLevel, 0.5f, 5.0f, "%.1f");

                // Add panning information
                ImGui::Text("Pan: X=%.1f, Y=%.1f", panOffset.x, panOffset.y);

                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                ImVec2 canvas_size = ImGui::GetContentRegionAvail();

                // Handle panning with mouse dragging
                ImGuiIO& io = ImGui::GetIO();
                if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                    panOffset.x += io.MouseDelta.x;
                    panOffset.y += io.MouseDelta.y;
                }

                // Adjust canvas position and size further for child window
                canvas_pos.y += 30;
                canvas_size.y -= 30;

                // Create a child window for scrolling
                ImGui::BeginChild("ScrollingRegion", canvas_size, false);

                ImDrawList* child_draw_list = ImGui::GetWindowDrawList();
                ImVec2 child_canvas_pos = ImGui::GetCursorScreenPos();

                // Draw machines and jobs
                float machineHeight = 100.0f * zoomLevel; // Adjusted for zoom
                float machineSpacing = 20.0f * zoomLevel;
                float jobHeight = machineHeight - machineSpacing;

                // Calculate timeScale to fit the schedule in the window when zoomLevel is 1.0
                float timeScale = (canvas_size.x - 50) / totalTime * zoomLevel;

                // Calculate content size for scrolling
                float content_height = numMachines * machineHeight + 50;
                float content_width = totalTime * timeScale + 50;

                ImGui::InvisibleButton("ScheduleCanvas", ImVec2(content_width, content_height));
                ImVec2 origin = ImGui::GetItemRectMin();

                // Apply pan offset
                origin.x += panOffset.x;
                origin.y += panOffset.y;

                // Draw grid lines and time labels
                int timeInterval = std::max(1, totalTime / 10);
                for (int t = 0; t <= totalTime; t += timeInterval) {
                    float x = origin.x + t * timeScale;
                    child_draw_list->AddLine(ImVec2(x, origin.y), ImVec2(x, origin.y + content_height), IM_COL32(150, 150, 150, 100));
                    child_draw_list->AddText(ImVec2(x - 10, origin.y - 20), IM_COL32(255, 255, 255, 255), std::to_string(t).c_str());
                }

                // Draw horizontal grid lines for machines
                for (int m = 0; m <= numMachines; ++m) {
                    float y = origin.y + m * machineHeight;
                    child_draw_list->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + content_width, y), IM_COL32(150, 150, 150, 100));
                }

                for (int m = 0; m < numMachines; ++m) {
                    float y = origin.y + m * machineHeight + machineSpacing / 2 + 10; // Added 10 pixels to avoid overlap
                    child_draw_list->AddText(ImVec2(origin.x + 5, y - 25), IM_COL32(200, 200, 200, 255), ("Machine " + std::to_string(m)).c_str());

                    for (size_t i = 0; i < std::max(scheduleData[m].size(), oldScheduleData[m].size()); ++i) {
                        const ScheduleEntry& entry = i < scheduleData[m].size() ? scheduleData[m][i] : ScheduleEntry{-1, 0, 0};
                        const ScheduleEntry& oldEntry = i < oldScheduleData[m].size() ? oldScheduleData[m][i] : ScheduleEntry{-1, 0, 0};

                        float x1 = origin.x + std::lerp(oldEntry.start, entry.start, transitionProgress) * timeScale;
                        float x2 = x1 + std::lerp(oldEntry.duration, entry.duration, transitionProgress) * timeScale;
                        ImVec2 rect_min(x1, y + machineSpacing / 2);
                        ImVec2 rect_max(x2, y + jobHeight);

                        ImVec4 color = entry.job >= 0 ? jobColors[entry.job % jobColors.size()] : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                        ImVec4 oldColor = oldEntry.job >= 0 ? jobColors[oldEntry.job % jobColors.size()] : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                        ImVec4 lerpedColor = lerpColor(oldColor, color, transitionProgress);

                        child_draw_list->AddRectFilled(rect_min, rect_max, ImColor(lerpedColor));
                        child_draw_list->AddRect(rect_min, rect_max, IM_COL32(0, 0, 0, 255));

                        if (entry.job >= 0) {
                            std::string job_text = "J" + std::to_string(entry.job);
                            ImVec2 text_size = ImGui::CalcTextSize(job_text.c_str());
                            ImVec2 text_pos((x1 + x2 - text_size.x) / 2, ((rect_min.y + rect_max.y - text_size.y) / 2));
                            child_draw_list->AddText(text_pos, IM_COL32(0, 0, 0, 255), job_text.c_str());
                        }
                    }
                }

                ImGui::EndChild();

                // Add a button to capture the screenshot
                if (ImGui::Button("Capture Screenshot")) {
                    captureScreenshot = true;
                }

                ImGui::EndTabItem();
            }

            // Operation Graph Tab
            if (ImGui::BeginTabItem("Operation Graph")) {
                drawOperationGraph();
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);

        // Update transition progress
        transitionProgress = std::min(transitionProgress + 0.1f, 1.0f); // Slower transition for smoother animation

        // Check if current transition is complete and there are updates in the queue
        if (transitionProgress >= 1.0f && !updateQueue.empty()) {
            oldScheduleData = scheduleData;
            auto [newSchedule, newTotalTime] = updateQueue.front();
            updateQueue.pop();
            scheduleData = newSchedule;
            totalTime = newTotalTime;
            currentVisualizationMakespan = newTotalTime;

            generateJobColors(static_cast<int>(jobs.size()));
            transitionProgress = 0.0f;
        }

        // Capture screenshot if requested
        if (captureScreenshot) {
            captureWindow();
        }
    }

    bool shouldClose() {
        return glfwWindowShouldClose(window);
    }
};