cmake_minimum_required(VERSION 3.20)

project(HBDIAFormat LANGUAGES CXX CUDA)

# Create the HBDIA library
add_library(HBDIAFormat STATIC
    HBDIA.cpp
    HBDIAVector.cpp
)

# Set include directories for this library
target_include_directories(HBDIAFormat PUBLIC
    ${CMAKE_SOURCE_DIR}/include/Format
    ${CMAKE_SOURCE_DIR}/include
)

message(STATUS "HBDIAFormat library configured")
