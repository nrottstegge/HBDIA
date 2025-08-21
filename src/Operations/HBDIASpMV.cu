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

    std::fill(cpuResults, cpuResults + numRows, T(0)); // Initialize results to zero
    
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

// Separate method for CPU COO execution
template<typename T>
void executeCOOOnCPU(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, 
                     HBDIAVector<T>& outputVector, int numRows, 
                     cudaStream_t sD2H, cudaStream_t sH2D, cudaStream_t sCOO, cudaEvent_t cooEvent) {
    // Get CPU fallback data
    const auto& cpuRowIndices = matrix.getCpuRowIndices();
    const auto& cpuColIndices = matrix.getCpuColIndices();
    const auto& cpuValues = matrix.getCpuValues();
    
    // 1. Copy input vector to host memory asynchronously
    CHECK_CUDA(cudaMemcpyAsync(inputVector.getHostDataPtr(), inputVector.getDeviceDataPtr(), 
                               inputVector.getTotalSize() * sizeof(T), cudaMemcpyDeviceToHost, sD2H));
    CHECK_CUDA(cudaStreamSynchronize(sD2H));
    
    // 2. Compute CPU results
    if (matrix.isPartialMatrix()) {
        hbdia_cpu_coo_spmv_partial<T>(cpuRowIndices, cpuColIndices, cpuValues, 
                                      inputVector.getHostDataPtr(), numRows, 
                                      outputVector.getHostCOOResultsPtr(), matrix);
    } else {
        hbdia_cpu_coo_spmv<T>(cpuRowIndices, cpuColIndices, cpuValues, 
                              inputVector.getHostLocalPtr(), numRows, 
                              outputVector.getHostCOOResultsPtr());
    }
    
    // 3. Copy CPU results back to device asynchronously
    CHECK_CUDA(cudaMemcpyAsync(outputVector.getDeviceCOOResultsPtr(), outputVector.getHostCOOResultsPtr(), 
                               numRows * sizeof(T), cudaMemcpyHostToDevice, sH2D));
    
    // Record event after COO work completion
    cudaEventRecord(cooEvent, sH2D);
}

// Method for GPU COO execution using cuSPARSE
template<typename T>
void executeCOOOnGPU(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, 
                     HBDIAVector<T>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent) {
    
    // Get CPU fallback data
    const auto& cpuRowIndices = matrix.getCpuRowIndices();
    if (cpuRowIndices.empty()) return;
    
    // Ensure cuSPARSE is initialized
    if (!matrix.isCuSparseInitialized()) {
        matrix.initializeCuSparse();
    }

    
    // Get cuSPARSE objects from matrix
    cusparseHandle_t handle = matrix.getCuSparseHandle();
    cusparseSpMatDescr_t matA = matrix.getCOOMatDescr();
    void* d_buffer = matrix.getCOOBuffer();
    
    // Get output destination
    T* d_cooResult = outputVector.getDeviceCOOResultsPtr();
    
    // Create vector descriptors for this specific SpMV call
    cusparseDnVecDescr_t vecX, vecY;
    int numCols = matrix.getNumCols();
    
    if constexpr (std::is_same_v<T, float>) {
        cusparseCreateDnVec(&vecX, numCols, (void*)inputVector.getDeviceDataPtr(), CUDA_R_32F);
        cusparseCreateDnVec(&vecY, numRows, d_cooResult, CUDA_R_32F);
    } else {
        cusparseCreateDnVec(&vecX, numCols, (void*)inputVector.getDeviceDataPtr(), CUDA_R_64F);
        cusparseCreateDnVec(&vecY, numRows, d_cooResult, CUDA_R_64F);
    }
    
    // Execute SpMV
    T alpha = T(1.0), beta = T(0.0);
    if constexpr (std::is_same_v<T, float>) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, 
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    } else {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, 
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    }

    // Record event after COO work completion
    cudaEventRecord(cooEvent, sCOO);
    
    // Cleanup vector descriptors (matrix descriptor and buffer are reused)
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
}

// Host function for hybrid GPU+CPU HBDIA SpMV
template<typename T>
bool hbdiaSpMV(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, HBDIAVector<T>& outputVector, 
               bool execCOOCPU, bool execCOOGPU) {
    // Get matrix dimensions and block information
    int numRows = matrix.getNumRows();

    // Ensure streams are initialized
    if (!matrix.areStreamsInitialized()) {
        std::cerr << "CUDA streams not initialized. Call matrix.initializeStreams() first." << std::endl;
        return false;
    }
    
    // Get streams and events from the matrix
    cudaStream_t sBDIA = matrix.getBDIAStream();
    cudaStream_t sD2H = matrix.getD2HStream();
    cudaStream_t sH2D = matrix.getH2DStream();
    cudaStream_t sCOO = matrix.getCOOStream();
    cudaStream_t sADD = matrix.getADDStream();
    cudaEvent_t bdiaEvent = matrix.getBDIAEvent();
    cudaEvent_t cooEvent = matrix.getCOOEvent();

    // Get CPU fallback data to check if we have COO work
    const auto& cpuValues = matrix.getCpuValues();
    bool hasCOOWork = !cpuValues.empty();

    // Use OpenMP to parallelize GPU and CPU/COO execution
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            int numBlocks = matrix.getNumBlocks();
            int blockWidth = matrix.getBlockWidth();
            // Launch GPU kernel configuration
            int threadsPerBlock = THREADS_PER_BLOCK_SPMV;
            int numBlocks_grid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

            bdia_spmv_kernel<<<numBlocks_grid, threadsPerBlock, 0, sBDIA>>>(
                matrix.getHBDIADataDevice(),           // Matrix data on GPU
                matrix.getFlattenedOffsetsDevice(),    // Flattened offsets
                matrix.getBlockStartIndicesDevice(),   // Block start indices
                inputVector.getDeviceDataPtr(),       // Input vector
                outputVector.getDeviceLocalPtr(),      // Output vector - FIXED: use device pointer
                numRows,                               // Number of rows
                numBlocks,                             // Number of blocks
                blockWidth                            // Block width
            );
            // Record event after BDIA kernel completion
            cudaEventRecord(bdiaEvent, sBDIA);
        }
        
        #pragma omp section
        {
            // Execute COO part based on flags
            if (hasCOOWork) {
                if (execCOOCPU) {
                    executeCOOOnCPU<T>(matrix, inputVector, outputVector, numRows, sD2H, sH2D, sCOO, cooEvent);
                } else if (execCOOGPU) {
                    executeCOOOnGPU<T>(matrix, inputVector, outputVector, numRows, sCOO, cooEvent);
                }
            }
        }
    }

    if (hasCOOWork) {
        // Wait for both BDIA kernel and COO work to complete before vector addition
        cudaStreamWaitEvent(sADD, bdiaEvent, 0);  // Wait for BDIA kernel
        cudaStreamWaitEvent(sADD, cooEvent, 0);    // Wait for COO work

        // Add COO results to BDIA results on dedicated ADD stream
        int threadsPerBlock_add = THREADS_PER_BLOCK_VECTOR_ADD;
        int numBlocks_add = (numRows + threadsPerBlock_add - 1) / threadsPerBlock_add;
        vector_add_kernel<<<numBlocks_add, threadsPerBlock_add, 0, sADD>>>(
            outputVector.getDeviceLocalPtr(),      // Destination vector (BDIA results)
            outputVector.getDeviceCOOResultsPtr(), // Source vector (COO results buffer)
            numRows  // Number of elements to add
        );
        
        // Wait for vector addition to complete
        cudaStreamSynchronize(sADD);
    } else {
        // If no COO work, just wait for BDIA kernel
        cudaEventSynchronize(bdiaEvent);
    }

    return true;
}

// Explicit template instantiations
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

template void executeCOOOnGPU<float>(HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, 
                                      HBDIAVector<float>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent);

template void executeCOOOnGPU<double>(HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, 
                                       HBDIAVector<double>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent);


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

// Explicit template instantiations for COO execution methods
template void executeCOOOnCPU<float>(HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, 
                                      HBDIAVector<float>& outputVector, int numRows, 
                                      cudaStream_t sD2H, cudaStream_t sH2D, cudaStream_t sCOO, cudaEvent_t cooEvent);

template void executeCOOOnCPU<double>(HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, 
                                       HBDIAVector<double>& outputVector, int numRows, 
                                       cudaStream_t sD2H, cudaStream_t sH2D, cudaStream_t sCOO, cudaEvent_t cooEvent);

// Explicit template instantiations for host function
template bool hbdiaSpMV<float>(HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, HBDIAVector<float>& outputVector, bool execCOOCPU, bool execCOOGPU);
template bool hbdiaSpMV<double>(HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, HBDIAVector<double>& outputVector, bool execCOOCPU, bool execCOOGPU);