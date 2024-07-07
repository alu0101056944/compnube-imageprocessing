cmake_minimum_required(VERSION 3.9.0)

project(main VERSION 0.4.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE Debug)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED core imgproc highgui cudaarithm)
set(CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
set(CMAKE_CUDA_ARCHITECTURES 80)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(OpenCV REQUIRED)

find_package(MPI)

add_executable(main_sequential main_sequential.cpp
    src/image_process_sequential.cpp)
target_link_libraries(main_sequential ${OpenCV_LIBS})

add_executable(main_parallel main_parallel.cpp src/image_process_parallel.cpp)
target_compile_options(main_parallel PRIVATE "-fopenmp")
target_link_libraries(main_parallel ${OpenCV_LIBS} -fopenmp)

add_executable(main_mpi main_mpi.cpp src/image_process_mpi.cpp)
target_link_libraries(main_mpi ${OpenCV_LIBS})
include_directories(${MPI_C_INCLUDE_PATH})
target_link_libraries(main_mpi ${MPI_C_LIBRARIES})

add_executable(main_cuda main_cuda.cu)
add_library(main_cuda_lib src/image_process_cuda.cu)
set_target_properties(main_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main_cuda main_cuda_lib ${OpenCV_LIBS} ${CUDA_LIBRARIES})
target_link_libraries(main_cuda_lib ${OpenCV_LIBS} ${CUDA_LIBRARIES})

add_executable(main_all main_all.cpp
src/image_process_sequential.cpp src/image_process_parallel.cpp)
target_compile_options(main_all PRIVATE "-fopenmp")
target_link_libraries(main_all ${OpenCV_LIBS} -fopenmp)
