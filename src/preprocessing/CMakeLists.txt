cmake_minimum_required(VERSION 3.26)
project(MODERNML_PREPROCESSING)

add_library(${PROJECT_NAME} MinMaxScaler.hpp)

link_directories(../utils)
link_libraries(${PROJECT_NAME} MODERNML_UTILS)