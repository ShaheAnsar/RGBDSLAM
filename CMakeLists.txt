cmake_minimum_required(VERSION 3.10)

project(ceres_slam)
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED TRUE)
set(SOURCE_FILES ceres_test.cpp)
add_executable(ceres_slam.out ${SOURCE_FILES})
target_link_libraries(ceres_slam.out -lceres)
