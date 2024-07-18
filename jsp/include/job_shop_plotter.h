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
#include "job_shop_environment.h"

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
        jobColors.reserve(numJobs);
        for (int i = 0; i < numJobs; ++i) {
            float hue = i * (1.0f / numJobs);
            ImVec4 color = ImColor::HSV(hue, 0.7f, 0.7f);
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

public:
    LivePlotter(int machines) : numMachines(machines), totalTime(0), transitionProgress(1.0f), currentVisualizationMakespan(0) {
        initializeGLFW();
        initializeImGui();
        scheduleData.resize(numMachines);
        oldScheduleData.resize(numMachines);
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

        // Add title for current visualization
        ImGui::SetCursorPosY(10);
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("Current Makespan: 0").x) / 2);
        ImGui::Text("Current Makespan: %d", currentVisualizationMakespan);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();

        // Adjust canvas position and size to avoid overlap
        canvas_pos.y += 20;
        canvas_size.y -= 40;


        // Draw X-axis with timeline labels
        float timeScale = canvas_size.x / static_cast<float>(totalTime);
        for (int t = 0; t <= totalTime; t += std::max(1, totalTime / 10)) {
            float x = canvas_pos.x + t * timeScale;
            draw_list->AddLine(ImVec2(x, canvas_pos.y), ImVec2(x, canvas_pos.y + canvas_size.y), IM_COL32(100, 100, 100, 255));
            draw_list->AddText(ImVec2(x - 10, canvas_pos.y - 20), IM_COL32(200, 200, 200, 255), std::to_string(t).c_str());
        }

        // Adjust canvas position and size further for child window
        canvas_pos.y += 30;
        canvas_size.y -= 30;

        // Create a child window for scrolling
        ImGui::BeginChild("ScrollingRegion", ImVec2(canvas_size.x, canvas_size.y), false, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar);

        ImVec2 child_canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 child_canvas_size = ImGui::GetContentRegionAvail();

        // Draw machines and jobs
        float machineHeight = 100.0f; // Fixed height for each machine
        float machineSpacing = 20.0f;
        float jobHeight = machineHeight - machineSpacing;

        // Set the size of the child canvas to be large enough to accommodate all machines
        ImGui::Dummy(ImVec2(child_canvas_size.x, numMachines * machineHeight));
        ImGui::SetCursorPos(child_canvas_pos);

        float totalHeight = numMachines * machineHeight;
        draw_list->AddRect(child_canvas_pos, ImVec2(child_canvas_pos.x + child_canvas_size.x, child_canvas_pos.y + totalHeight), IM_COL32(50, 50, 50, 255));

        for (int m = 0; m < numMachines; ++m) {
            float y = child_canvas_pos.y + m * machineHeight + machineSpacing / 2 + 10; // Added 10 pixels to avoid overlap
            draw_list->AddText(ImVec2(child_canvas_pos.x, y - 5), IM_COL32(200, 200, 200, 255), ("Machine " + std::to_string(m)).c_str());

            for (size_t i = 0; i < std::max(scheduleData[m].size(), oldScheduleData[m].size()); ++i) {
                const ScheduleEntry& entry = i < scheduleData[m].size() ? scheduleData[m][i] : ScheduleEntry{-1, 0, 0};
                const ScheduleEntry& oldEntry = i < oldScheduleData[m].size() ? oldScheduleData[m][i] : ScheduleEntry{-1, 0, 0};

                float x1 = child_canvas_pos.x + std::lerp(oldEntry.start, entry.start, transitionProgress) * timeScale;
                float x2 = x1 + std::lerp(oldEntry.duration, entry.duration, transitionProgress) * timeScale;
                ImVec2 rect_min(x1, y + machineSpacing / 2);
                ImVec2 rect_max(x2, y + jobHeight);

                ImVec4 color = entry.job >= 0 ? jobColors[entry.job] : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                ImVec4 oldColor = oldEntry.job >= 0 ? jobColors[oldEntry.job] : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                ImVec4 lerpedColor = lerpColor(oldColor, color, transitionProgress);

                draw_list->AddRectFilled(rect_min, rect_max, ImColor(lerpedColor));
                draw_list->AddRect(rect_min, rect_max, IM_COL32(0, 0, 0, 255));

                if (entry.job >= 0) {
                    std::string job_text = "J" + std::to_string(entry.job);
                    ImVec2 text_size = ImGui::CalcTextSize(job_text.c_str());
                    ImVec2 text_pos((x1 + x2 - text_size.x) / 2, ((rect_min.y + rect_max.y - text_size.y) / 2));
                    draw_list->AddText(text_pos, IM_COL32(0, 0, 0, 255), job_text.c_str());
                }
            }
        }

        ImGui::EndChild();
        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);

        // Update transition progress
        transitionProgress = std::min(transitionProgress + 0.2f, 1.0f); // Slower transition for smoother animation

        // Check if current transition is complete and there are updates in the queue
        if (transitionProgress >= 1.0f && !updateQueue.empty()) {
            oldScheduleData = scheduleData;
            auto [newSchedule, newTotalTime] = updateQueue.front();
            updateQueue.pop();
            scheduleData = newSchedule;
            totalTime = newTotalTime;
            currentVisualizationMakespan = newTotalTime;

            int maxJob = 0;
            for (const auto& machine : scheduleData) {
                for (const auto& entry : machine) {
                    maxJob = std::max(maxJob, entry.job);
                }
            }
            generateJobColors(maxJob + 1);
            transitionProgress = 0.0f;
        }
    }




    bool shouldClose() {
        return glfwWindowShouldClose(window);
    }
};