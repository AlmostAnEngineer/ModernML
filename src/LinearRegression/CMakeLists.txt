cmake_minimum_required(VERSION 3.26)
project(LinearRegression)

set(LIB_SOURCES
        LinearRegression.cpp
        LinearRegression.hpp
)

add_library(LinearRegressionLib ${LIB_SOURCES})

add_executable(LinearRegression main.cpp)

target_link_libraries(LinearRegression PRIVATE LinearRegressionLib)