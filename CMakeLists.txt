cmake_minimum_required(VERSION 3.2)
project(raytracer LANGUAGES C CXX CUDA)

include_directories(include)
include_directories(/usr/local/cuda-10.2/include)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS  "${CMAKE_CUDA_FLAGS} -g -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_53 -g")

add_executable(raytracer 
    ./src/main.cpp
    ./src/kernel.cu
)

target_link_libraries(raytracer ${PNG_LIBRARY})
