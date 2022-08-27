cmake_minimum_required(VERSION 3.2)
project(raytracer LANGUAGES C CXX CUDA)

include_directories(include)
include_directories(/usr/local/cuda-10.2/include)

add_executable(raytracer 
    ./src/main.cpp
)