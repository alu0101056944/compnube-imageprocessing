cmake_minimum_required(VERSION 3.9.0)
# project(main VERSION 0.4.0 LANGUAGES CXX CUDA)
project(main VERSION 0.4.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CUDA_NVCC_FLAGS "--gpu-architecture=sm_86")
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6")
# set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
# set(CUDA_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
# set(CUDA_RT_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib")
# set(CUDA_HOST_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin/nvcc.exe")
# set(CUDA_HOST_LINKER "${CUDA_HOST_COMPILER}")
# set(CUDA_STANDARD 11)

# find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories($ENV{ICONV_PATH})

# add_library(iconv STATIC IMPORTED)
# set_target_properties(iconv PROPERTIES
#                       IMPORTED_LOCATION "C:\\opencv\\libiconv-for-Windows\\output\\x64\\Debug\\libiconvD.lib")

add_executable(main_sequential main_sequential.cpp
    src/image_process_sequential.cpp)
target_link_libraries(main_sequential ${OpenCV_LIBS})
target_link_libraries(main_sequential iconv)

add_executable(main_parallel main_parallel.cpp src/image_process_parallel.cpp)
target_compile_options(main_parallel PRIVATE "-fopenmp")
target_link_libraries(main_parallel ${OpenCV_LIBS} -fopenmp)
target_link_libraries(main_parallel iconv)

find_package(MPI)
add_executable(main_mpi main_mpi.cpp src/image_process_mpi.cpp)
target_link_libraries(main_mpi ${OpenCV_LIBS})
include_directories(${MPI_C_INCLUDE_PATH})
target_link_libraries(main_mpi ${MPI_C_LIBRARIES})
target_link_libraries(main_mpi iconv)

find_package(CUDA)
add_executable(main_cuda main_cuda.cu)
add_library(main_cuda_lib src/image_process_cuda.cu)
target_link_libraries(main_cuda main_cuda_lib)
target_compile_options(main_cuda PRIVATE "${CUDA_NVCC_FLAGS}")
target_link_libraries(main_cuda ${OpenCV_LIBS})
target_link_libraries(main_cuda_lib ${OpenCV_LIBS})
# target_link_libraries(main_cuda iconv)

add_executable(main_all main_all.cpp
src/image_process_sequential.cpp src/image_process_parallel.cpp)
target_compile_options(main_all PRIVATE "-fopenmp")
target_link_libraries(main_all ${OpenCV_LIBS} -fopenmp)
target_link_libraries(main_all iconv)
  