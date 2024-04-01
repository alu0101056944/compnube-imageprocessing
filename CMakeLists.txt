cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
find_package(OpenCV REQUIRED)

add_executable(main_sequential main_sequential.cpp
    src/image_process_sequential.cpp)
target_compile_features(main_sequential PRIVATE cxx_std_17)
target_link_libraries(main_sequential ${OpenCV_LIBS})

add_executable(main_parallel main_parallel.cpp
    src/image_process_parallel.cpp)
target_compile_features(main_parallel PRIVATE cxx_std_17)
target_compile_options(main_parallel PRIVATE "-fopenmp")
target_link_libraries(main_parallel ${OpenCV_LIBS})

add_executable(main_mpi main_mpi.cpp
    src/image_process_mpi.cpp)
target_compile_features(main_mpi PRIVATE cxx_std_17)
target_link_libraries(main_mpi ${OpenCV_LIBS})

add_executable(main_all main_all.cpp
    src/image_process_sequential.cpp src/image_process_parallel.cpp)
target_compile_features(main_all PRIVATE cxx_std_17)
target_link_libraries(main_all ${OpenCV_LIBS})
