cmake_minimum_required(VERSION 3.20)
project(HBDIA VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -lineinfo")

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Find OpenMP for CPU optimization
find_package(OpenMP REQUIRED)

# Find NVPL sparse from NVIDIA HPC SDK
set(NVPL_ROOT "/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/nvpl")
set(NVPL_INCLUDE_DIR "${NVPL_ROOT}/include")
set(NVPL_LIBRARY_DIR "${NVPL_ROOT}/lib")

# Set MPI paths based on the working HighPerformanceHPCG_Thesis configuration
#set(MPI_CXX_INCLUDE_PATH "/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-mpich-8.1.30-wb5peugemrg2ebx7psp2iz2abmqy3rgz/include")
#set(MPI_CXX_LIBRARIES "/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-mpich-8.1.30-wb5peugemrg2ebx7psp2iz2abmqy3rgz/lib/libmpi_gnu_123.so;/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-gtl-8.1.30-yjzx5mja37woqidkb7lj6lpbu3omqrsr/lib/libmpi_gtl_cuda.so")

# Find MPI package
find_package(MPI REQUIRED)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native -ffp-contract=fast -fopenmp")

# Add MPI definitions (following HighPerformanceHPCG_Thesis pattern)
add_definitions(-DMPI_GPU_AWARE)

# Include directories
include_directories(include/Format)
include_directories(include)
include_directories(${MPI_CXX_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${NVPL_INCLUDE_DIR})

# Create the main HBDIA library directly
add_library(HBDIA STATIC
    src/Format/HBDIA.cpp
    src/Format/HBDIAPrinter.cpp
    src/Format/HBDIAVector.cpp
    src/DataExchange/BasicExtractor.cpp
    src/DataExchange/BasicDistributor.cpp
    src/DataExchange/MPICommunicator.cpp
    src/Operations/HBDIASpMV.cu
)

# Set include directories for the library
target_include_directories(HBDIA PUBLIC
    ${CMAKE_SOURCE_DIR}/include/Format
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVPL_INCLUDE_DIR}
)

# Add the test executable
add_executable(test_HBDIA
    tests/run_HBDIA_test.cpp
)

# Link the test with the HBDIA library
target_link_libraries(test_HBDIA HBDIA MPI::MPI_CXX CUDA::cudart)

# Add the benchmark executable
add_executable(benchmark
    benchmarking/benchmark.cpp
    benchmarking/benchCusparse.cpp
    benchmarking/benchHBDIA.cpp
)

# Link the benchmark with the HBDIA library
target_link_libraries(benchmark HBDIA MPI::MPI_CXX CUDA::cudart CUDA::cusparse)

# Link HBDIA library with necessary dependencies including NVPL sparse
target_link_libraries(HBDIA PUBLIC 
    MPI::MPI_CXX 
    CUDA::cudart 
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    ${NVPL_SPARSE_LIB}
)

# Add tests subdirectory
add_subdirectory(tests)

# Enable testing
enable_testing()
add_test(NAME HBDIA_basic_test COMMAND test_HBDIA)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "HBDIA project configured successfully")
