// run_HBDIA_cusparse_comparison.cpp
// Compare HBDIA distributed SpMV results with cuSPARSE reference implementation

#include "../include/Format/HBDIA.hpp"
#include "../include/Format/HBDIAVector.hpp"
#include "../include/DataExchange/BasicDistributor.hpp"
#include "../include/DataExchange/BasicExtractor.hpp"
#include "../include/DataExchange/MPICommunicator.hpp"
#include "../include/Operations/HBDIASpMV.cuh"

#include <cusparse.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <stdio.h>

using DataType = double;

// cuSPARSE reference SpMV implementation with separate timing
bool cusparseSpMV(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, 
                  const std::vector<DataType>& values, const std::vector<DataType>& inputVector,
                  std::vector<DataType>& outputVector, int numRows, int numCols,
                  double& setupTime, double& executionTime) {
    
    auto setupStart = std::chrono::high_resolution_clock::now();
    
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
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
    
    // Create sparse matrix in COO format
    cusparseCreateCoo(&matA, numRows, numCols, nnz, d_rowIndices, d_colIndices, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    // Create dense vectors
    cusparseCreateDnVec(&vecX, numCols, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, numRows, d_y, CUDA_R_64F);
    
    // Get buffer size (part of setup)
    DataType alpha = 1.0, beta = 0.0;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    
    void* d_buffer = nullptr;
    if (bufferSize > 0) {
        cudaMalloc(&d_buffer, bufferSize);
    }
    
    // Synchronize before timing actual execution
    cudaDeviceSynchronize();
    auto setupEnd = std::chrono::high_resolution_clock::now();

    auto execStart = std::chrono::high_resolution_clock::now();
    auto execEnd = std::chrono::high_resolution_clock::now();
    
    // === ACTUAL SPMV EXECUTION - TIMED SEPARATELY ===
    for(int i = 0; i < 20; i++) {
        // Reset output vector to ensure computation actually happens
        cudaMemset(d_y, 0, numRows * sizeof(DataType));
        
        // Force the input to be different each iteration to prevent optimization
        DataType alpha = 1.0;  // Small variation
        cudaDeviceSynchronize();
        auto execStart = std::chrono::high_resolution_clock::now();
        
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        
        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();
        auto execEnd = std::chrono::high_resolution_clock::now();
        
        std::cout << "cuSPARSE SpMV iteration " << i + 1 << " took \t"
                  << std::chrono::duration<double, std::milli>(execEnd - execStart).count() 
                  << " ms" << std::endl;

    }
    
    // Copy result back (this could also be timed separately if needed)
    outputVector.resize(numRows);
    cudaMemcpy(outputVector.data(), d_y, numRows * sizeof(DataType), cudaMemcpyDeviceToHost);
    
    // Calculate timing
    setupTime = std::chrono::duration<double, std::milli>(setupEnd - setupStart).count();
    executionTime = std::chrono::duration<double, std::milli>(execEnd - execStart).count();
    
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
    
    return true;
}

int main(int argc, char *argv[]) {
    auto hbdiaStart = std::chrono::high_resolution_clock::now();
    std::string filename = "/users/nrottstegge/SuiteSparseMatrixCollection/channel-500x100x100-b050/channel-500x100x100-b050.mtx";
    
    // Initialize MPI
    auto communicator = std::make_unique<MPICommunicator<DataType>>();
    communicator->initialize(argc, argv);
    
    auto extractor = std::make_unique<BasicExtractor<DataType>>();
    BasicDistributor<DataType> distributor(std::move(communicator), std::move(extractor), 0);
    distributor.setPartitioningStrategy(HBDIAExtractor<DataType>::PartitioningStrategy::ROW_WISE);
    
    int myRank = distributor.getRank();
    
    // 1. Load matrix and create test vector
    HBDIA<DataType> globalMatrix, localMatrix;
    std::vector<DataType> globalVector, localVector;
    
    if (myRank == 0) {
        std::cout << "=== HBDIA vs cuSPARSE Comparison Test ===" << std::endl;
        
        // Load matrix
        //globalMatrix.loadMTX(filename);
        globalMatrix.create3DStencil27Point(128, 128, 128);
        globalMatrix.print(); // Print matrix info for debugging
        hbdiaStart = std::chrono::high_resolution_clock::now();
        
        // Create test vector (0, 1, 2, ..., n-1)
        int numRows = globalMatrix.getNumRows();
        globalVector.resize(numRows);
        for (int i = 0; i < numRows; i++) {
            globalVector[i] = static_cast<DataType>(i);
        }
        
        std::cout << "Matrix size: " << numRows << "x" << globalMatrix.getNumCols() << std::endl;
        std::cout << "Non-zeros: " << globalMatrix.getNumNonZeros() << std::endl;
    }
    
    // 2. Distribute matrix and vector
    if (myRank == 0) {
        distributor.scatterMatrix(globalMatrix, localMatrix);
        distributor.scatterVector(globalVector, localVector);
    } else {
        distributor.receiveMatrix(localMatrix);
        distributor.receiveVector(localVector);
    }
    
    // Create HBDIA vector and exchange data
    HBDIAVector<DataType> hbdiaVector(localVector, localMatrix, myRank, distributor.getSize());
    distributor.exchangeData(localMatrix, hbdiaVector);
    
    // 3. HBDIA SpMV
    HBDIAVector<DataType> hbdiaResult(std::vector<DataType>(localMatrix.getNumRows(), 0.0));

    bool hbdiaSuccess = false;
    auto hbdiaEnd = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 20; i++) {
        // Reset both unified memory output and CPU results buffer to ensure computation happens
        CHECK_CUDA(cudaMemset(hbdiaResult.getDeviceLocalPtr(), 0, localMatrix.getNumRows() * sizeof(DataType)));
        CHECK_CUDA(cudaMemset(hbdiaResult.getDeviceCpuResultsPtr(), 0, localMatrix.getNumRows() * sizeof(DataType)));
        std::fill(hbdiaResult.getDeviceLocalPtr(), hbdiaResult.getDeviceLocalPtr() + localMatrix.getNumRows(), 0.0);
        std::fill(hbdiaResult.getDeviceCpuResultsPtr(), hbdiaResult.getDeviceCpuResultsPtr() + localMatrix.getNumRows(), 0.0);
        
        cudaDeviceSynchronize();
        hbdiaStart = std::chrono::high_resolution_clock::now();
        hbdiaSuccess = hbdiaSpMV(localMatrix, hbdiaVector, hbdiaResult);
        hbdiaEnd = std::chrono::high_resolution_clock::now();
        std::cout << "HBDIA SpMV iteration " << i + 1 << " took \t"
                  << std::chrono::duration<double, std::milli>(hbdiaEnd - hbdiaStart).count() 
                  << " ms" << std::endl;
    }
    
    if (!hbdiaSuccess) {
        std::cerr << "HBDIA SpMV failed!" << std::endl;
        distributor.getCommunicator().finalize();
        return 1;
    }
    
    // Gather HBDIA results
    std::vector<DataType> hbdiaGlobalResult;
    distributor.gatherVector(hbdiaResult, hbdiaGlobalResult);
    
    // 4. cuSPARSE reference (only on rank 0)
    std::vector<DataType> cusparseResult;
    double cusparseSetupTime = 0.0, cusparseExecTime = 0.0;
    
    if (myRank == 0) {
        bool cusparseSuccess = cusparseSpMV(localMatrix.getRowIndices(), 
                                           localMatrix.getColIndices(),
                                           localMatrix.getValues(),
                                           localVector,
                                           cusparseResult,
                                           localMatrix.getNumRows(),
                                           localMatrix.getNumCols(),
                                           cusparseSetupTime,
                                           cusparseExecTime);
        
        if (!cusparseSuccess) {
            std::cerr << "cuSPARSE SpMV failed!" << std::endl;
            distributor.getCommunicator().finalize();
            return 1;
        }
    }
    
    // 5. Compare results (rank 0 only)
    if (myRank == 0) {
        auto hbdiaTime = std::chrono::duration<double, std::milli>(hbdiaEnd - hbdiaStart).count();
        
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "HBDIA time:            " << hbdiaTime << " ms" << std::endl;
        std::cout << "cuSPARSE setup time:   " << cusparseSetupTime << " ms" << std::endl;
        std::cout << "cuSPARSE exec time:    " << cusparseExecTime << " ms" << std::endl;
        std::cout << "cuSPARSE total time:   " << (cusparseSetupTime + cusparseExecTime) << " ms" << std::endl;
        std::cout << "Speedup vs cuSPARSE exec: " << cusparseExecTime / hbdiaTime << "x" << std::endl;
        std::cout << "Speedup vs cuSPARSE total: " << (cusparseSetupTime + cusparseExecTime) / hbdiaTime << "x" << std::endl;
        
        // Accuracy comparison
        DataType maxError = 0.0;
        DataType hbdiaSum = 0.0, cusparseSum = 0.0;
        int errorCount = 0;
        
        for (size_t i = 0; i < hbdiaGlobalResult.size(); i++) {
            DataType error = std::abs(hbdiaGlobalResult[i] - cusparseResult[i]);
            maxError = std::max(maxError, error);
            if (error > 1e-6) errorCount++;
            
            hbdiaSum += std::abs(hbdiaGlobalResult[i]);
            cusparseSum += std::abs(cusparseResult[i]);
        }
        
        std::cout << "\n=== Accuracy Results ===" << std::endl;
        std::cout << "Max error:     " << maxError << std::endl;
        std::cout << "Error count:   " << errorCount << "/" << hbdiaGlobalResult.size() << std::endl;
        std::cout << "HBDIA sum:     " << hbdiaSum << std::endl;
        std::cout << "cuSPARSE sum:  " << cusparseSum << std::endl;
        std::cout << "Relative error: " << std::abs(hbdiaSum - cusparseSum) / cusparseSum << std::endl;
        
        if (maxError < 1e-6) {
            std::cout << "✅ ACCURACY TEST PASSED!" << std::endl;
        } else {
            std::cout << "❌ ACCURACY TEST FAILED!" << std::endl;
            
            // Show first few mismatches
            std::cout << "\nFirst 5 mismatches:" << std::endl;
            int shown = 0;
            for (size_t i = 0; i < hbdiaGlobalResult.size() && shown < 5; i++) {
                DataType error = std::abs(hbdiaGlobalResult[i] - cusparseResult[i]);
                if (error > 1e-10) {
                    std::cout << "Row " << i << ": HBDIA=" << hbdiaGlobalResult[i] 
                              << ", cuSPARSE=" << cusparseResult[i] 
                              << ", error=" << error << std::endl;
                    shown++;
                }
            }
        }
    }
    
    distributor.getCommunicator().finalize();
    return 0;
}
