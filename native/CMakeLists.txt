cmake_minimum_required(VERSION 3.18)
project(pead_engine LANGUAGES CXX)

option(PEAD_USE_CUDA "Build the CUDA compute backend" OFF)
option(PEAD_BUILD_PYBIND "Build the pybind11 module" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

set(PEAD_SOURCES
    src/arrow_io.cpp
    src/compute.cpp
    src/compute_cpu.cpp)

if(PEAD_USE_CUDA)
  enable_language(CUDA)
  list(APPEND PEAD_SOURCES src/compute_cuda.cu)
endif()

add_library(pead_core STATIC ${PEAD_SOURCES})
target_include_directories(pead_core PUBLIC include)
target_link_libraries(pead_core PUBLIC Arrow::arrow_shared Parquet::parquet_shared)

if(PEAD_USE_CUDA)
  target_compile_definitions(pead_core PUBLIC PEAD_USE_CUDA)
  set_target_properties(pead_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(pead_engine src/main.cpp)
target_link_libraries(pead_engine PRIVATE pead_core)

if(PEAD_BUILD_PYBIND)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(pead_native bindings/pybind.cpp)
  target_link_libraries(pead_native PRIVATE pead_core)
endif()
