#include "benchHBDIA.hpp"
#include "../include/Operations/HBDIASpMV.cuh"
#include "../include/types.hpp"
#include <chrono>
#include <iostream>



void benchHBDIA(HBDIA<DataType>& matrix,
                const std::vector<DataType>& inputVector,
                std::vector<DataType>& outputVector,
                bool execCOOCPU, bool execCOOGPU,
                std::vector<double>& measurements) {
    
    // Initialize cuSPARSE if needed
    if (execCOOGPU) {
        std::cout << "Initializing cuSPARSE for HBDIA SpMV..." << std::endl;
        matrix.initializeCuSparse();
    }
    
    // Create HBDIA vectors
    HBDIAVector<DataType> hbdiaVecX(inputVector);
    HBDIAVector<DataType> hbdiaVecY(std::vector<DataType>(matrix.getNumRows(), 0.0));
    
    measurements.clear();
    measurements.reserve(NUM_BENCH_ITER);
    
    // Benchmark iterations
    for(int i = 0; i < NUM_BENCH_ITER; i++) {
        // Reset output vectors
        CHECK_CUDA(cudaMemset(hbdiaVecY.getDeviceLocalPtr(), 0, matrix.getNumRows() * sizeof(DataType)));
        CHECK_CUDA(cudaMemset(hbdiaVecY.getDeviceCOOResultsPtr(), 0, matrix.getNumRows() * sizeof(DataType)));
        
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = hbdiaSpMV(matrix, hbdiaVecX, hbdiaVecY, execCOOCPU, execCOOGPU);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!success) {
            std::cerr << "HBDIA SpMV failed at iteration " << i << std::endl;
            return;
        }

        measurements.push_back(std::chrono::duration<double, std::milli>(end - start).count());

    }
    
    // Copy result back
    outputVector.resize(matrix.getNumRows());
    CHECK_CUDA(cudaMemcpy(outputVector.data(), hbdiaVecY.getDeviceLocalPtr(), 
                          matrix.getNumRows() * sizeof(DataType), cudaMemcpyDeviceToHost));
}
