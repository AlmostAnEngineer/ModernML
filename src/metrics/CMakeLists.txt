cmake_minimum_required(VERSION 3.26)
project(MODERNML_METRICS)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_library(${PROJECT_NAME} SHARED
        R2Score.hpp)

link_directories(../utils)
link_libraries(${PROJECT_NAME} MODERNML_UTILS)