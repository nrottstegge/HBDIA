#include "benchCusparse.hpp"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

void benchCusparse(const std::vector<int>& rowIndices, 
                   const std::vector<int>& colIndices,
                   const std::vector<DataType>& values,
                   const std::vector<DataType>& inputVector,
                   std::vector<DataType>& outputVector,
                   int numRows, int numCols,
                   std::vector<double>& measurements) {
    
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Device pointers
    int *d_rowIndices, *d_colIndices;
    DataType *d_values, *d_x, *d_y;
    int nnz = values.size();
    
    // Allocate device memory
    cudaMalloc((void**)&d_rowIndices, nnz * sizeof(int));
    cudaMalloc((void**)&d_colIndices, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(DataType));
    cudaMalloc((void**)&d_x, numCols * sizeof(DataType));
    cudaMalloc((void**)&d_y, numRows * sizeof(DataType));
    
    // Copy data to device
    cudaMemcpy(d_rowIndices, rowIndices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndices, colIndices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, inputVector.data(), numCols * sizeof(DataType), cudaMemcpyHostToDevice);
    
    // Create matrix and vector descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCoo(&matA, numRows, numCols, nnz, d_rowIndices, d_colIndices, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, numCols, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, numRows, d_y, CUDA_R_64F);
    
    // Get buffer size and allocate workspace
    DataType alpha = 1.0, beta = 0.0;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    void* d_buffer = nullptr;
    if (bufferSize > 0) {
        cudaMalloc(&d_buffer, bufferSize);
    }
    
    measurements.clear();
    measurements.reserve(NUM_BENCH_ITER);
    
    // Benchmark iterations
    for(int i = 0; i < NUM_BENCH_ITER; i++) {
        cudaMemset(d_y, 0, numRows * sizeof(DataType));
        cudaDeviceSynchronize();
        
        auto start = std::chrono::high_resolution_clock::now();
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        measurements.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    // Copy result back
    outputVector.resize(numRows);
    cudaMemcpy(outputVector.data(), d_y, numRows * sizeof(DataType), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cusparseDestroy(handle);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(d_rowIndices);
    cudaFree(d_colIndices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    if (d_buffer) cudaFree(d_buffer);
}
