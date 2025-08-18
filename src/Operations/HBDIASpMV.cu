#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <omp.h>
#include <cusparse.h>
#include "../../include/types.hpp"
#include "../../include/Format/HBDIA.hpp"
#include "../../include/Format/HBDIAVector.hpp"

// GPU kernel for HBDIA SpMV - processes block-diagonal elements
// Supports both single GPU (non-partial) and distributed (partial) matrices
template<typename T>
__global__ void hbdia_spmv_kernel(
    const T* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
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
    int blockSize = blockStartIndices[currentBlock+1] - blockStart;
    
    if (blockSize == 0) {
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
    
    outputVector[row] = sum;  // Directly accumulate into output vector
}

// GPU kernel for vector addition - aggregates CPU results with GPU results
template<typename T>
__global__ void vector_add_kernel(
    T* __restrict__ dest,           // Destination vector (GPU results in unified memory)
    const T* __restrict__ src,      // Source vector (CPU results buffer)
    int numElements                 // Number of elements to add
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) {
        // Only add if source element is non-zero to avoid unnecessary operations
        dest[idx] += src[idx];
    }
}


template<typename T>
void hbdia_cpu_coo_spmv(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    T* cpuResults  // CPU results buffer from HBDIAVector
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t nnz = cpuRowIndices.size();
    if (nnz == 0) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < nnz; ++i) {
        int row = cpuRowIndices[i];
        int col = cpuColIndices[i];
        T coeff = cpuValues[i];
        T value = inputVector[col];
        T result = coeff * value;
        #pragma omp atomic
        cpuResults[row] += result;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    //std::cout << "CPU COO SpMV processing time (OpenMP): " << elapsed.count() * 1000 << " ms" << std::endl;
}

template<typename T>
void hbdia_cpu_coo_spmv_partial(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    T* cpuResults,  // CPU results buffer from HBDIAVector
    const HBDIA<T>& matrix      // Matrix object to use findMemoryOffsetForGlobalIndex
) {
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

    int numRows = matrix.getNumRows();

    bool isPartialMatrix = matrix.isPartialMatrix();

    // Get pointer to the local section of the output vector in unified memory
    T* outputPtr = outputVector.getLocalDataPtr();

    cudaStream_t stream_compute;
    CHECK_CUDA(cudaStreamCreate(&stream_compute));
    
    // Use OpenMP to parallelize GPU and CPU execution
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            int numBlocks = matrix.getNumBlocks();
            int blockWidth = matrix.getBlockWidth();
            // Launch GPU kernel configuration
            int threadsPerBlock = THREADS_PER_BLOCK_SPMV;
            int numBlocks_grid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
            // GPU Section - Launch kernel and wait for completion
            hbdia_spmv_kernel<<<numBlocks_grid, threadsPerBlock>>>(
                matrix.getHBDIADataDevice(),           // Matrix data on GPU
                matrix.getFlattenedOffsetsDevice(),    // Flattened offsets
                matrix.getBlockStartIndicesDevice(),   // Block start indices
                inputVector.getDeviceDataPtr(),       // Input vector
                numBlocks,                             // Number of blocks
                blockWidth,                            // Block width
                numRows,                               // Number of rows
                outputVector.getDeviceLocalPtr(),      // Output vector - FIXED: use device pointer
                false,//isPartialMatrix,                       // Flag for partial matrix
                matrix.getFlattenedVectorOffsetsDevice() // Vector offsets (null for non-partial)
            );
        }
        
        #pragma omp section
        {
            // Get CPU fallback data
            const auto& cpuRowIndices = matrix.getCpuRowIndices();
            const auto& cpuColIndices = matrix.getCpuColIndices();
            const auto& cpuValues = matrix.getCpuValues();
            // Check if we have CPU work to do
            bool hasCPUWork = !cpuValues.empty();

            if(hasCPUWork){
                //create cuda stream for CPU operations
                // cudaStream_t stream_memcpy;
                // CHECK_CUDA(cudaStreamCreate(&stream_memcpy));
                // 1. copy input vector to host memory asynchronously
                //CHECK_CUDA(cudaMemcpyAsync(inputVector.getHostDataPtr(), inputVector.getDeviceDataPtr(), inputVector.getTotalSize() * sizeof(T), cudaMemcpyDeviceToDevice, stream_memcpy));
                // 2. Prepare CPU results buffer in output vector
                //std::fill(outputVector.getHostCpuResultsPtr(), outputVector.getHostCpuResultsPtr() + numRows, T(0));
                // 3 compute CPU results
                //CHECK_CUDA(cudaStreamSynchronize(stream_memcpy));
                hbdia_cpu_coo_spmv<T>(cpuRowIndices, cpuColIndices, cpuValues, inputVector.getDeviceLocalPtr(), numRows, outputVector.getDeviceCpuResultsPtr());
                // 4. Copy CPU results back to device asynchronously
                //CHECK_CUDA(cudaMemcpyAsync(outputVector.getDeviceCpuResultsPtr(), outputVector.getHostCpuResultsPtr(), numRows * sizeof(T), cudaMemcpyDeviceToDevice, stream_memcpy));
                // 5. Add CPU results to GPU results
                int threadsPerBlock_add = THREADS_PER_BLOCK_VECTOR_ADD;
                int numBlocks_add = (numRows + threadsPerBlock_add - 1) / threadsPerBlock_add;
                //CHECK_CUDA(cudaStreamSynchronize(stream_memcpy));
                vector_add_kernel<<<numBlocks_add, threadsPerBlock_add>>>(
                    outputVector.getDeviceLocalPtr(),  // Destination vector (GPU results) - FIXED
                    outputVector.getDeviceCpuResultsPtr(), // Source vector (CPU results buffer)
                    numRows  // Number of elements to add
                );
            }
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    return true;
}

// Explicit template instantiations
template __global__ void hbdia_spmv_kernel<float>(
    const float* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
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
    const int* __restrict__ blockStartIndices,
    const double* __restrict__ inputVector,
    int numBlocks,
    int blockWidth,
    int numRows,
    double* __restrict__ outputVector,
    bool isPartialMatrix,
    const int* __restrict__ flattenedVectorOffsets
);

// Explicit template instantiations for vector addition kernel
template __global__ void vector_add_kernel<float>(
    float* __restrict__ dest,
    const float* __restrict__ src,
    int numElements
);

template __global__ void vector_add_kernel<double>(
    double* __restrict__ dest,
    const double* __restrict__ src,
    int numElements
);

// Explicit template instantiations for CPU function
template void hbdia_cpu_coo_spmv<float>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<float>& cpuValues,
    const float* inputVector,
    int numRows,
    float* cpuResults  // CPU results buffer from HBDIAVector
);

template void hbdia_cpu_coo_spmv<double>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<double>& cpuValues,
    const double* inputVector,
    int numRows,
    double* cpuResults  // CPU results buffer from HBDIAVector
);

// Explicit template instantiations for partial CPU function
template void hbdia_cpu_coo_spmv_partial<float>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<float>& cpuValues,
    const float* inputVector,
    int numRows,
    float* cpuResults,  // CPU results buffer from HBDIAVector
    const HBDIA<float>& matrix
);

template void hbdia_cpu_coo_spmv_partial<double>(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<double>& cpuValues,
    const double* inputVector,
    int numRows,
    double* cpuResults,  // CPU results buffer from HBDIAVector
    const HBDIA<double>& matrix
);

// Explicit template instantiations for host function
template bool hbdiaSpMV<float>(const HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, HBDIAVector<float>& outputVector);
template bool hbdiaSpMV<double>(const HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, HBDIAVector<double>& outputVector);