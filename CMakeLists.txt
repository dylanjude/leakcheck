cmake_minimum_required(VERSION 3.18)

# set the project name
project(mpi_gpu_buffer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED true)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_70,code=sm_70 -lineinfo")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CUDA_LIBRARY CUDA::cudart)

add_executable(bw_and_leak_check main.cpp)
set_property(TARGET bw_and_leak_check PROPERTY CUDA_ARCHITECTURES 70 )
target_include_directories(bw_and_leak_check PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(bw_and_leak_check
        ${CUDA_LIBRARY}
        ${MPI_CXX_LIBRARIES}
        MPI::MPI_CXX)
