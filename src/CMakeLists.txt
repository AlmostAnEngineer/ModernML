cmake_minimum_required(VERSION 3.26)
project(MODERNML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(${PROJECT_NAME}
        main.cpp
        preprocessing/Scaler.hpp)

link_directories(utils LinearModels metrics preprocessing)
link_libraries(${PROJECT_NAME} MODERNML_LINEARMODELS
        MODERNML_UTILS
        MODERNML_METRICS
        MODERNML_PREPROCESSING)