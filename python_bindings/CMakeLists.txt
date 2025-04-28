cmake_minimum_required(VERSION 3.12)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Check for Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Find pybind11
# If you don't have pybind11 installed, you can add it as a submodule
# or set PYBIND11_PYTHON_VERSION to match your Python version
find_package(pybind11 REQUIRED)

# Set compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

# Library object targets
pybind11_add_module(rodespmm rode_spmm.cpp)

target_link_libraries(rodespmm PRIVATE)

# Include parent directory so we can include "../utils/matrix_utils.h" etc.
target_include_directories(rodespmm PRIVATE ..)

# Link with parent library and direct dependencies
target_link_libraries(rodespmm PRIVATE rodespmm_lib myutils RoDe_spmm) 