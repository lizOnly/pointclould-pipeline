cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(pcd_pipeline)

find_package(PCL 1.3 REQUIRED COMPONENTS)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

add_executable(
    pcd_pipeline 
    src/main.cpp 
    src/reconstruction.cpp headers/reconstruction.h
    src/property.cpp headers/property.h 
    src/evaluation.cpp headers/evaluation.h
    src/occlusion.cpp headers/occlusion.h
    src/scanner.cpp headers/scanner.h
    )

target_link_libraries(pcd_pipeline ${PCL_LIBRARIES})
