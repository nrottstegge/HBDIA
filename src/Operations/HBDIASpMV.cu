#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "../../include/types.hpp"

// GPU kernel for blocked-DIA part of HBDIA SpMV
template<typename T>
__global__ void bdia_spmv_kernel(
    const T* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
    const int* __restrict__ blockStartIndices,
    const T* __restrict__ inputVector,
    T* __restrict__ outputVector,
    int numRows,
    int numBlocks,
    int blockWidth
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= numRows) return;
    
    T sum = T(0);
    
    // Determine which block this row belongs to
    int currentBlock = row / blockWidth;
    if (currentBlock >= numBlocks) { //gate
        return;
    }
    
    // Get block information
    int blockStart = blockStartIndices[currentBlock];
    int blockSize = blockStartIndices[currentBlock+1] - blockStart;
    
    if (blockSize == 0) { // If block size is zero, skip this row
        return;
    }
    
    // Calculate lane within block
    int lane = row % blockWidth;
    
    // Process all offsets in this block
    for (int offsetIdx = 0; offsetIdx < blockSize; offsetIdx++) {

        // Calculate matrix data index - use blockStart which accounts for variable block sizes
        int matrixDataIdx = blockStart * blockWidth + offsetIdx * blockWidth + lane;
        
        // Access matrix value
        T matrixValue = hbdiaData[matrixDataIdx];

        if(matrixValue == T(0)) {
            // If matrix value is zero, skip this entry
            continue;
        }

        // Access vector value
        T vectorValue;

        // Get the diagonal offset and calculate column index
        int offset = flattenedOffsets[blockStart + offsetIdx];
        int col = row + offset;
        
        if (col >= 0 && col < numRows) {
            vectorValue = inputVector[col];
        } else {
            // Out of bounds, skip this entry
            continue;
        }
        
        // Accumulate result
        sum += matrixValue * vectorValue;
    }

    outputVector[row] = sum;  // Directly accumulate into output vector
}

// GPU kernel for vector addition - aggregates CPU results with GPU results
template<typename T>
__global__ void vector_add_kernel(
    T* __restrict__ dest,           // Destination vector (GPU results computed on device by bdia_spmv_kernel)
    const T* __restrict__ src,      // Source vector (CPU results buffer)
    int numElements                 // Number of elements to add
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) { //gate
        dest[idx] += src[idx];
    }
}

// CUDA wrapper function for BDIA kernel launch
template<typename T>
void launch_bdia_spmv_kernel(
    const T* hbdiaData,
    const int* flattenedOffsets,
    const int* blockStartIndices,
    const T* inputVector,
    T* outputVector,
    int numRows,
    int numBlocks,
    int blockWidth,
    cudaStream_t stream
) {
    // Launch GPU kernel configuration
    int threadsPerBlock = THREADS_PER_BLOCK_SPMV;
    int numBlocks_grid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    bdia_spmv_kernel<<<numBlocks_grid, threadsPerBlock, 0, stream>>>(
        hbdiaData,           // Matrix data on GPU
        flattenedOffsets,    // Flattened offsets
        blockStartIndices,   // Block start indices
        inputVector,         // Input vector
        outputVector,        // Output vector
        numRows,             // Number of rows
        numBlocks,           // Number of blocks
        blockWidth           // Block width
    );
}

// CUDA wrapper function for vector add kernel launch
template<typename T>
void launch_vector_add_kernel(
    T* dest,
    const T* src,
    int numElements,
    cudaStream_t stream
) {
    int threadsPerBlock_add = THREADS_PER_BLOCK_VECTOR_ADD;
    int numBlocks_add = (numElements + threadsPerBlock_add - 1) / threadsPerBlock_add;
    
    vector_add_kernel<<<numBlocks_add, threadsPerBlock_add, 0, stream>>>(
        dest,        // Destination vector (BDIA results)
        src,         // Source vector (COO results buffer)
        numElements  // Number of elements to add
    );
}

// Explicit template instantiations for kernel wrappers
template void launch_bdia_spmv_kernel<float>(
    const float* hbdiaData,
    const int* flattenedOffsets,
    const int* blockStartIndices,
    const float* inputVector,
    float* outputVector,
    int numRows,
    int numBlocks,
    int blockWidth,
    cudaStream_t stream
);

template void launch_bdia_spmv_kernel<double>(
    const double* hbdiaData,
    const int* flattenedOffsets,
    const int* blockStartIndices,
    const double* inputVector,
    double* outputVector,
    int numRows,
    int numBlocks,
    int blockWidth,
    cudaStream_t stream
);

template void launch_vector_add_kernel<float>(
    float* dest,
    const float* src,
    int numElements,
    cudaStream_t stream
);

template void launch_vector_add_kernel<double>(
    double* dest,
    const double* src,
    int numElements,
    cudaStream_t stream
);

// Explicit template instantiations for kernels (for completeness)
template __global__ void bdia_spmv_kernel<float>(
    const float* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
    const int* __restrict__ blockStartIndices,
    const float* __restrict__ inputVector,
    float* __restrict__ outputVector,
    int numRows,
    int numBlocks,
    int blockWidth
);

template __global__ void bdia_spmv_kernel<double>(
    const double* __restrict__ hbdiaData,
    const int* __restrict__ flattenedOffsets,
    const int* __restrict__ blockStartIndices,
    const double* __restrict__ inputVector,
    double* __restrict__ outputVector,
    int numRows,
    int numBlocks,
    int blockWidth
);