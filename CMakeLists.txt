# minimum required version of CMake
cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0148 NEW)

# Specify g++ as the C++ compiler
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE STRING "C++ compiler")

# Project name.
project(Matrix)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization and warnings (for G++/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -Werror)
endif()

# Header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# clean-build target
add_custom_target(clean-build
    COMMAND rm -rf *
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)

# Executable
add_executable(main ${SOURCES})

# Find pybind11 package
find_package(pybind11 REQUIRED)
# Include bindings.cpp as a Python module
pybind11_add_module(matrix MODULE ${PROJECT_SOURCE_DIR}/bindings.cpp ${SOURCES})
target_include_directories(matrix PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Set the output directory for the Python module
set_target_properties(matrix PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)