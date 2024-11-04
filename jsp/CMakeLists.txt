cmake_minimum_required(VERSION 3.21)
project(jobshop VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CURL REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)





find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Check if nanobind_DIR is set
if(DEFINED nanobind_DIR)
    list(APPEND CMAKE_MODULE_PATH "${nanobind_DIR}")
    message(STATUS "nanobind_DIR is set to: ${nanobind_DIR}")
else()
    message(WARNING "nanobind_DIR is not set")
    # site-packages/nanobind/cmake
    list(APPEND CMAKE_MODULE_PATH "${Python_SITEARCH}/nanobind/cmake")


    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message( "${Python_SITEARCH}/nanobind/cmake")
endif()

# Try to find nanobind
find_package(nanobind CONFIG REQUIRED)
if(NOT nanobind_FOUND)
    message(FATAL_ERROR "nanobind not found. Please set nanobind_DIR to the directory containing nanobind-config.cmake")
endif()

find_package(imgui CONFIG REQUIRED)

# Add source files
set(SOURCES
        src/dummy.cpp
)

# Add header files
set(HEADERS
        include/environment/job_shop_environment.h
        include/algorithms/job_shop_qlearning.h
        include/algorithms/job_shop_actor_critic.h
        include/algorithms/job_shop_algorithms.h
        include/environment/job_shop_environment_generator.h
        include/gui/job_shop_plotter.h
        include/environment/job_shop_taillard_generator.h
        include/utilities/multidimensional_array.hpp
        include/utilities/util.h
)

# Create the main executable if not building for pip installation
if(NOT DEFINED BUILD_EXECUTABLE OR BUILD_EXECUTABLE)
    add_executable(jobshop-exe ${SOURCES} ${HEADERS} src/main.cpp)
    target_include_directories(jobshop-exe PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OPENGL_INCLUDE_DIR}
            ${GLEW_INCLUDE_DIRS}
            ${CURL_INCLUDE_DIRS}
            include
    )
    target_link_libraries(jobshop-exe PRIVATE
            ${OPENGL_LIBRARIES}
            GLEW::GLEW
            glfw
            ${CURL_LIBRARIES}
            fmt::fmt
            imgui::imgui
    )
endif()

# Create the Python module
nanobind_add_module(jobshop bindings/jobshop_bindings.cpp  ${SOURCES} ${HEADERS})
target_include_directories(jobshop PRIVATE include ${Python_INCLUDE_DIRS})
target_link_libraries(jobshop PRIVATE
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        ${CURL_LIBRARIES}
        fmt::fmt
        imgui::imgui
        ${Python_LIBRARIES}
        nlohmann_json::nlohmann_json
)

# Installation
if(NOT DEFINED BUILD_EXECUTABLE OR BUILD_EXECUTABLE)
    install(TARGETS jobshop-exe
            EXPORT jobshopTargets
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
    )
endif()

install(TARGETS jobshop
        EXPORT jobshopTargets
        LIBRARY DESTINATION "${Python_SITEARCH}"
        RUNTIME DESTINATION "${Python_SITEARCH}"
)

install(FILES ${HEADERS} DESTINATION include/jobshop)

# Generate and install CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/jobshopConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(EXPORT jobshopTargets
        FILE jobshopTargets.cmake
        NAMESPACE jobshop::
        DESTINATION lib/cmake/jobshop
)

configure_file(Config.cmake.in jobshopConfig.cmake @ONLY)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/jobshopConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/jobshopConfigVersion.cmake"
        DESTINATION lib/cmake/jobshop
)