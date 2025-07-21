// types.hpp
#ifndef TYPES_HPP
#define TYPES_HPP

using DataType = double;

// HBDIA Default Parameters
#define DEFAULT_BLOCK_WIDTH 32
#define DEFAULT_THRESHOLD 16
#define MAX_CPU_ENTRIES 10000
#define NUMBER_THREADS_SPMV 1  // CPU threads for GH200 hybrid SpMV


// MPI Error Checking Macro - simple version without exceptions
#define CHECK_MPI(call) \
    do { \
        int mpi_result = (call); \
        if (mpi_result != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int length; \
            MPI_Error_string(mpi_result, error_string, &length); \
            std::cerr << "MPI Error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << #call << std::endl \
                      << "Error: " << error_string << std::endl; \
        } \
    } while(0)

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#endif // TYPES_HPP