cmake_minimum_required(VERSION 3.20)

project(HBDIADataExchange LANGUAGES CXX)

# Create a dummy source file for the library
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy_dataexchange.cpp 
     "// Placeholder file for DataExchange library\n")

# Create the DataExchange library (placeholder for future development)
add_library(HBDIADataExchange STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/dummy_dataexchange.cpp
)

# Set include directories for this library
target_include_directories(HBDIADataExchange PUBLIC
    ${CMAKE_SOURCE_DIR}/include/DataExchange
    ${CMAKE_SOURCE_DIR}/include
)

message(STATUS "HBDIADataExchange library configured (placeholder)")
