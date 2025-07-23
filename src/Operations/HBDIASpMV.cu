#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include "../../include/Format/HBDIA.hpp"
#include "../../include/Format/HBDIAVector.hpp"

// GPU kernel for HBDIA SpMV - processes block-diagonal elements
// Supports both single GPU (non-partial) and distributed (partial) matrices
template<typename T>
__global__ void hbdia_spmv_kernel(
    const T* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
    const int* __restrict__ blockSizes,
    const int* __restrict__ blockStartIndices,
    const T* __restrict__ inputVector,
    int numBlocks,
    int blockWidth,
    int numRows,
    T* __restrict__ outputVector,
    bool isPartialMatrix,
    const int* __restrict__ flattenedVectorOffsets
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= numRows) return;
    
    T sum = T(0);
    
    // Determine which block this row belongs to
    int currentBlock = row / blockWidth;
    if (currentBlock >= numBlocks) {
        // Note: We don't write zeros here since CPU might contribute to this row
        return;
    }
    
    // Get block information
    int blockStart = blockStartIndices[currentBlock];
    int blockSize = blockSizes[currentBlock];
    
    if (blockSize == 0) {
        // Note: We don't write zeros here since CPU might contribute to this row
        return;
    }
    
    // Calculate lane within block
    int lane = row % blockWidth;

    
    // Process all offsets in this block
    for (int offsetIdx = 0; offsetIdx < blockSize; offsetIdx++) {
        // Access vector value using appropriate indexing
        T vectorValue;
        if (isPartialMatrix) {
            // For partial matrices, use flattenedVectorOffsets to find the correct location in unified memory
            int vectorOffsetIdx = blockStart + offsetIdx;
            int memoryOffset = flattenedVectorOffsets[vectorOffsetIdx] + lane;

            if (memoryOffset >= 0) {
                vectorValue = inputVector[memoryOffset];
            } else {
                // Invalid offset, skip this entry
                continue;
            }
        } else {
            // For single GPU, get the diagonal offset and calculate column index
            int offset = flattenedOffsets[blockStart + offsetIdx];
            int col = row + offset;
            
            if (col >= 0 && col < numRows) {
                vectorValue = inputVector[col];
            } else {
                // Out of bounds, skip this entry
                continue;
            }
        }
        
        // Calculate matrix data index - use blockStart which accounts for variable block sizes
        int matrixDataIdx = blockStart * blockWidth + offsetIdx * blockWidth + lane;
        
        // Access matrix value
        T matrixValue = hbdiaData[matrixDataIdx];
        
        // Accumulate result
        sum += matrixValue * vectorValue;
    }
    
    if (sum != T(0)) {
        outputVector[row] = sum;
    }
}


template<typename T>
void hbdia_cpu_coo_spmv(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    std::vector<T>& cpuResults  // Separate CPU results buffer
) {
    // Initialize CPU results to zero
    cpuResults.assign(numRows, T(0));
    
    // Process each CPU fallback entry - single threaded for optimal performance
    for (size_t i = 0; i < cpuRowIndices.size(); ++i) {
        int row = cpuRowIndices[i];
        int col = cpuColIndices[i];
        T value = cpuValues[i];
        
        T result = value * inputVector[col];
        cpuResults[row] += result;  // Simple addition in CPU memory
    }
}

template<typename T>
void hbdia_cpu_coo_spmv_partial(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    std::vector<T>& cpuResults,  // Separate CPU results buffer
    const HBDIA<T>& matrix      // Matrix object to use findMemoryOffsetForGlobalIndex
) {
    // Initialize CPU results to zero
    cpuResults.assign(numRows, T(0));
    
    // Get metadata needed for memory offset calculation
    const auto& metadata = matrix.getPartialMatrixMetadata();
    int leftBufferSize = 0;
    int rightBufferSize = 0;
    int rank = matrix.getRank();
    int size = matrix.getSize();
    
    // Calculate buffer sizes from processDataRanges
    for (int procId = 0; procId < static_cast<int>(metadata.processDataRanges.size()); procId++) {
        if (procId < rank) {
            // Count elements needed from processes with lower rank (left buffer)
            for (const auto& range : metadata.processDataRanges[procId]) {
                leftBufferSize += std::get<1>(range) - std::get<0>(range);
            }
        } else if (procId > rank) {
            // Count elements needed from processes with higher rank (right buffer)
            for (const auto& range : metadata.processDataRanges[procId]) {
                rightBufferSize += std::get<1>(range) - std::get<0>(range);
            }
        }
    }
    
    // Calculate global data range for this process
    int rowsPerProcess = matrix.getNumGlobalRows() / size;
    int globalStart = rank * rowsPerProcess;
    int globalEnd = globalStart + rowsPerProcess + ((rank == size - 1) ? matrix.getNumGlobalRows() % rowsPerProcess : 0);
    int localVectorSize = matrix.getNumRows();
    
    // Process each CPU fallback entry
    for (size_t i = 0; i < cpuRowIndices.size(); ++i) {
        int row = cpuRowIndices[i];
        int globalCol = cpuColIndices[i];  // This is a global column index
        T value = cpuValues[i];
        
        // Find the memory offset for this global column index in unified memory
        int memoryOffset = matrix.findMemoryOffsetForGlobalIndex(
            globalCol, 
            leftBufferSize, 
            localVectorSize, 
            globalStart, 
            globalEnd, 
            rank
        );
        
        // Skip invalid offsets
        if (memoryOffset == INT_MIN || memoryOffset < 0) {
            std::cerr << "CPU COO: Skipping invalid memory offset for global column " << globalCol << std::endl;
            continue;
        }
        
        // Access the vector value at the calculated memory offset
        T vectorValue = inputVector[memoryOffset];
        
        // Compute and accumulate result
        T result = value * vectorValue;
        cpuResults[row] += result;
    }
}

// Host function for hybrid GPU+CPU HBDIA SpMV
template<typename T>
bool hbdiaSpMV(const HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, HBDIAVector<T>& outputVector) {
    // Get matrix dimensions and block information
    int numBlocks = static_cast<int>(matrix.getOffsetsPerBlock().size());
    int numRows = matrix.getNumRows();
    int blockWidth = matrix.getBlockWidth();
    bool isPartialMatrix = matrix.isPartialMatrix();

    // Get pointer to the local section of the output vector in unified memory
    T* outputPtr = outputVector.getLocalDataPtr();
    
    // Get CPU fallback data
    const auto& cpuRowIndices = matrix.getCpuRowIndices();
    const auto& cpuColIndices = matrix.getCpuColIndices();
    const auto& cpuValues = matrix.getCpuValues();
    
    // Launch GPU kernel configuration
    int threadsPerBlock = THREADS_PER_BLOCK_SPMV;
    int numBlocks_grid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    
    // GPU thread - launches kernel and measures its execution time
    // Launch GPU kernel (blocking in this thread)
    hbdia_spmv_kernel<<<numBlocks_grid, threadsPerBlock>>>(
        matrix.getHBDIADataDevice(),           // Matrix data on GPU
        matrix.getFlattenedOffsetsDevice(),    // Flattened offsets
        matrix.getBlockSizesDevice(),          // Block sizes
        matrix.getBlockStartIndicesDevice(),   // Block start indices
        inputVector.getUnifiedDataPtr(),       // Input vector (unified memory)
        numBlocks,                             // Number of blocks
        blockWidth,                            // Block width
        numRows,                               // Number of rows
        outputPtr,                             // Output vector (unified memory)
        isPartialMatrix,                       // Flag for partial matrix
        matrix.getFlattenedVectorOffsetsDevice() // Vector offsets (null for non-partial)
    );
    
    // CPU processing with separate accumulation buffer
    std::vector<T> cpuResults;
    std::thread cpuThread;
    bool hasCpuWork = !cpuValues.empty();
            
    // For distributed SpMV, CPU also needs to use the unified memory for column access
    const T* cpuInputPtr = inputVector.getUnifiedDataPtr();
    
    if (isPartialMatrix) {
        // For partial matrices, use the specialized function that handles global column indices
        hbdia_cpu_coo_spmv_partial<T>(cpuRowIndices, cpuColIndices, cpuValues, cpuInputPtr, numRows, cpuResults, matrix);
    } else {
        // For non-partial matrices, use the original function
        hbdia_cpu_coo_spmv<T>(cpuRowIndices, cpuColIndices, cpuValues, cpuInputPtr, numRows, cpuResults);
    }
    
    // Wait for GPU to complete
    cudaDeviceSynchronize();
    
    // Combine CPU results with GPU results if CPU processing occurred
    if (hasCpuWork) {
        // CPU results are in CPU memory, output is in unified memory
        for (int i = 0; i < numRows; ++i) {
            if (cpuResults[i] != T(0)) {
                outputPtr[i] += cpuResults[i];
            }
        }
    }

    return true;
}

// Explicit template instantiations
template __global__ void hbdia_spmv_kernel<float>(
    const float* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
    const int* __restrict__ blockSizes,
    const int* __restrict__ blockStartIndices,
    const float* __restrict__ inputVector,
    int numBlocks,
    int blockWidth,
    int numRows,
    float* __restrict__ outputVector,
    bool isPartialMatrix,
    const int* __restrict__ flattenedVectorOffsets
);

template __global__ void hbdia_spmv_kernel<double>(
    const double* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
    const int* __restrict__ blockSizes,
    const int* __restrict__ blockStartIndices,
    const double* __restrict__ inputVector,
    int numBlocks,
    int blockWidth,
    int numRows,
    double* __restrict__ outputVector,
    bool isPartialMatrix,
    const int* __restrict__ flattenedVectorOffsets
);

// Explicit template instantiations for CPU function
template void hbdia_cpu_coo_spmv<float>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<float>& cpuValues,
    const float* inputVector,
    int numRows,
    std::vector<float>& cpuResults
);

template void hbdia_cpu_coo_spmv<double>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<double>& cpuValues,
    const double* inputVector,
    int numRows,
    std::vector<double>& cpuResults
);

// Explicit template instantiations for partial CPU function
template void hbdia_cpu_coo_spmv_partial<float>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<float>& cpuValues,
    const float* inputVector,
    int numRows,
    std::vector<float>& cpuResults,
    const HBDIA<float>& matrix
);

template void hbdia_cpu_coo_spmv_partial<double>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<double>& cpuValues,
    const double* inputVector,
    int numRows,
    std::vector<double>& cpuResults,
    const HBDIA<double>& matrix
);

// Explicit template instantiations for host function
template bool hbdiaSpMV<float>(const HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, HBDIAVector<float>& outputVector);
template bool hbdiaSpMV<double>(const HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, HBDIAVector<double>& outputVector);