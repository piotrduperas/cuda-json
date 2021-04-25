cmake_minimum_required(VERSION 3.19)
project(cuda-json CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda-json main.cu)

set_target_properties(
        cuda-json
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)