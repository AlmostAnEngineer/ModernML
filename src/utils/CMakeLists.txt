cmake_minimum_required(VERSION 3.26)

project(MODERNML_UTILS)
set(CMAKE_CXX_STANDARD 20)

add_library(${PROJECT_NAME} SHARED
        Concepts.hpp
        MatrixMathOperations.hpp
        Outputs.hpp
        PrettyPrinter.hpp
        RandomGenerator.hpp
)