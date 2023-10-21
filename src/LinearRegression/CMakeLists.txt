cmake_minimum_required(VERSION 3.26)
project(ModernML)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(ModernML main.cpp LinearRegression.hpp)