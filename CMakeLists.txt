cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(pcd_pipeline)

find_package(PCL 1.3 REQUIRED COMPONENTS)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(
    pcd_pipeline 
    src/main.cpp 
    src/reconstruction.cpp headers/reconstruction.h
    src/property.cpp headers/property.h 
    src/evaluation.cpp headers/evaluation.h
    src/helper.cpp headers/helper.h
    src/visualizer.cpp headers/visualizer.h
    )

target_link_libraries(pcd_pipeline ${PCL_LIBRARIES})
