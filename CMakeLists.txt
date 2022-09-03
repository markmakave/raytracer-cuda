cmake_minimum_required(VERSION 3.2)
project(raytracer LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
    include
    /usr/local/cuda-10.2/include
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

cuda_add_executable(raytracer 
    ./src/main.cpp
    ./src/render.cu

    ./src/hitable/sphere.cu
    ./src/hitable/mesh.cu
    ./src/hitable/triangle.cu
)

target_link_libraries(raytracer ${PNG_LIBRARY})

