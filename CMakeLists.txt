cmake_minimum_required(VERSION 3.13)

project(SPC CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_definitions(-w)

find_package(CUDA)
include(FindCUDA/select_compute_arch)

set(CMAKE_CXX_STANDARD 17)

# If no architectures specified, set defaults
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    SET(CMAKE_CUDA_ARCHITECTURES "80")
endif()

set(UTILS_LIBS ${PROJECT_SOURCE_DIR}/utils)

set(RoDe_SPMM_LIBS ${PROJECT_SOURCE_DIR}/RoDe_SpMM)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})


include_directories("${PROJECT_SOURCE_DIR}")
MESSAGE( STATUS "this var key = ${CMAKE_CUDA_FLAGS}.")

# Create the library target for RoDeSpMM
add_library(rodespmm_lib INTERFACE)
target_link_libraries(rodespmm_lib INTERFACE myutils RoDe_spmm)

add_subdirectory(utils)
add_subdirectory(RoDe_SpMM)
add_subdirectory(eval)

# Add python bindings subdirectory
add_subdirectory(python_bindings)
