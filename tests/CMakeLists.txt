cmake_minimum_required(VERSION 3.20)

project(HBDIATesting LANGUAGES CXX)

# Create the test executable
add_executable(test_print_HBDIA
    run_HBDIA_print_test.cpp
)

# Link with the HBDIA library
target_link_libraries(test_print_HBDIA
    HBDIA
    CUDA::cudart
)

# Set include directories for tests
target_include_directories(test_print_HBDIA PRIVATE
    ${CMAKE_SOURCE_DIR}/include/Format
    ${CMAKE_SOURCE_DIR}/include
)

# Enable testing
enable_testing()

# Add test cases
add_test(NAME HBDIA_conversion_test COMMAND test_print_HBDIA)

message(STATUS "HBDIA tests configured")
