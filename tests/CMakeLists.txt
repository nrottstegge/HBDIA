cmake_minimum_required(VERSION 3.20)

project(HBDIATesting LANGUAGES CXX)

# Create the test executables
add_executable(test_print_HBDIA
    run_HBDIA_print_test.cpp
)

add_executable(test_HBDIA_cusparse_comparison
    run_HBDIA_cusparse_comparison.cpp
)

# Link with the HBDIA library
target_link_libraries(test_print_HBDIA
    HBDIA
    CUDA::cudart
)

target_link_libraries(test_HBDIA_cusparse_comparison
    HBDIA
    CUDA::cudart
    CUDA::cusparse
    MPI::MPI_CXX
)

# Set include directories for tests
target_include_directories(test_print_HBDIA PRIVATE
    ${CMAKE_SOURCE_DIR}/include/Format
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_HBDIA_cusparse_comparison PRIVATE
    ${CMAKE_SOURCE_DIR}/include/Format
    ${CMAKE_SOURCE_DIR}/include
)

# Enable testing
enable_testing()

# Add test cases
add_test(NAME HBDIA_conversion_test COMMAND test_print_HBDIA)
add_test(NAME HBDIA_cusparse_comparison_test COMMAND mpirun -n 2 test_HBDIA_cusparse_comparison)

message(STATUS "HBDIA tests configured")
