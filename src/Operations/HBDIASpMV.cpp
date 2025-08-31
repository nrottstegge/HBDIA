#include <iostream>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "../../include/types.hpp"
#include "../../include/Format/HBDIA.hpp"
#include "../../include/Format/HBDIAVector.hpp"

// Forward declarations for CUDA functions
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
);

template<typename T>
void launch_vector_add_kernel(
    T* dest,
    const T* src,
    int numElements,
    cudaStream_t stream
);

// CPU COO SpMV kernel with OpenMP
template<typename T>
void cpu_coo_spmv_kernel(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    T* cpuResults
) {
    const size_t nnz = cpuRowIndices.size();
    if (nnz == 0) return;

    #pragma omp parallel
    {
        const int num_threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        
        // Divide rows among threads
        const int rows_per_thread = (numRows + num_threads - 1) / num_threads;
        const int start_row = tid * rows_per_thread;
        const int end_row = std::min(start_row + rows_per_thread, numRows);
        
        // search for start position
        size_t start_idx = 0;
        if (start_row > 0) {
            auto it = std::lower_bound(cpuRowIndices.begin(), cpuRowIndices.end(), start_row);
            start_idx = it - cpuRowIndices.begin();
        }
        
        // search for end position
        size_t end_idx = nnz;
        if (end_row < numRows) {
            auto it = std::lower_bound(cpuRowIndices.begin(), cpuRowIndices.end(), end_row);
            end_idx = it - cpuRowIndices.begin();
        }
        
        // Process elements in range
        T row_sum = T(0);
        int current_row = -1;
        
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            const int row = cpuRowIndices[idx];
            
            //on new row store previous sum and reset
            if (row != current_row) {
                if (current_row >= start_row && current_row < end_row) {
                    cpuResults[current_row] = row_sum;
                }
                current_row = row;
                row_sum = T(0);
            }
            
            // Accumulate for current row
            const int col = cpuColIndices[idx];
            const T val = cpuValues[idx];
            row_sum += val * inputVector[col];
        }
        
        // Store final row sum
        if (current_row >= start_row && current_row < end_row) {
            cpuResults[current_row] = row_sum;
        }
    }
}

// CPU COO execution method
template<typename T>
void executeCOOOnCPU(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, 
                     HBDIAVector<T>& outputVector, int numRows, 
                     cudaStream_t sD2H, cudaStream_t sH2D, cudaStream_t sCOO, cudaEvent_t cooEvent) {
    
    // Get CPU fallback data
    const auto& cpuRowIndices = matrix.getCpuRowIndices();
    const auto& cpuColIndices = matrix.getCpuColIndices();
    const auto& cpuValues = matrix.getCpuValues();

    if(outputVector.isUnifiedMemory()) {
        // No transfers needed, directly use unified memory
        cpu_coo_spmv_kernel<T>(cpuRowIndices, cpuColIndices, cpuValues, inputVector.getDeviceLocalPtr(), numRows, outputVector.getDeviceCOOResultsPtr());

    } else { //explicit transfers GPU - CPU, CPU - GPU
        
        // 1. Copy input vector to host memory asynchronously
        CHECK_CUDA(cudaMemcpyAsync(inputVector.getHostDataPtr(), inputVector.getDeviceDataPtr(), inputVector.getTotalSize() * sizeof(T), cudaMemcpyDeviceToHost, sD2H));
        CHECK_CUDA(cudaStreamSynchronize(sD2H));
        
        // 2. Compute CPU results
        cpu_coo_spmv_kernel<T>(cpuRowIndices, cpuColIndices, cpuValues, inputVector.getHostLocalPtr(), numRows, outputVector.getHostCOOResultsPtr());
        
        // 3. Copy CPU results back to device asynchronously
        CHECK_CUDA(cudaMemcpyAsync(outputVector.getDeviceCOOResultsPtr(), outputVector.getHostCOOResultsPtr(), numRows * sizeof(T), cudaMemcpyHostToDevice, sH2D));
        
        // Record event after COO work completion
        cudaEventRecord(cooEvent, sH2D);
    }
}

// GPU COO execution using cuSPARSE
template<typename T>
void executeCOOOnGPU(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, 
                     HBDIAVector<T>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent) {
    
    // Get CPU fallback data
    const auto& cpuRowIndices = matrix.getCpuRowIndices();
    if (cpuRowIndices.empty()) return;
    
    // Ensure cuSPARSE is initialized
    if (!matrix.isCuSparseInitialized()) {
        //matrix.initializeCuSparse();
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

// Main HBDIA SpMV function
template<typename T>
bool hbdiaSpMV(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, HBDIAVector<T>& outputVector, 
               bool execCOOCPU, bool execCOOGPU) {

    // Get data from the matrix
    int numRows = matrix.getNumRows();
    cudaStream_t sBDIA = matrix.getBDIAStream();
    
    cudaStream_t sD2H = matrix.getD2HStream();
    cudaStream_t sH2D = matrix.getH2DStream();
    cudaStream_t sCOO = matrix.getCOOStream();
    cudaStream_t sADD = matrix.getADDStream();
    cudaEvent_t bdiaEvent = matrix.getBDIAEvent();
    cudaEvent_t cooEvent = matrix.getCOOEvent();               
    
    // check if we have COO work
    bool hasCOOWork = !matrix.getCpuValues().empty();

    int numBlocks = matrix.getNumBlocks();
    int blockWidth = matrix.getBlockWidth();
    
    // Launch GPU kernel via CUDA wrapper
    launch_bdia_spmv_kernel<T>(
        matrix.getHBDIADataDevice(),           // Matrix data on GPU
        matrix.getFlattenedOffsetsDevice(),    // Flattened offsets
        matrix.getBlockStartIndicesDevice(),   // Block start indices
        inputVector.getDeviceDataPtr(),       // Input vector
        outputVector.getDeviceLocalPtr(),      // Output vector
        numRows,                               // Number of rows
        numBlocks,                             // Number of blocks
        blockWidth,                           // Block width
        sBDIA                                 // CUDA stream
    );

    // Record event after BDIA kernel completion
    cudaEventRecord(bdiaEvent, sBDIA);

    // Execute COO part based on flags
    if (hasCOOWork) {
        if (execCOOCPU) {
            executeCOOOnCPU<T>(matrix, inputVector, outputVector, numRows, sD2H, sH2D, sCOO, cooEvent);
        } else if (execCOOGPU) {
            executeCOOOnGPU<T>(matrix, inputVector, outputVector, numRows, sCOO, cooEvent);
        } else {
            std::cerr << "COO data is available but execCOOCPU and execCOOGPU are both false. No COO execution will be performed." << std::endl;
            throw std::runtime_error("Invalid COO execution flags");
        }
    }

    // combine results if COO work was done
    if (hasCOOWork) {
        // Wait for both BDIA kernel and COO work to complete before vector addition
        cudaStreamWaitEvent(sADD, bdiaEvent, 0);  // Wait for BDIA kernel
        if(!outputVector.isUnifiedMemory()) cudaStreamWaitEvent(sADD, cooEvent, 0);    // Wait for COO work

        // Add COO results to BDIA via CUDA wrapper
        launch_vector_add_kernel<T>(
            outputVector.getDeviceLocalPtr(),      // Destination vector (BDIA results)
            outputVector.getDeviceCOOResultsPtr(), // Source vector (COO results buffer)
            numRows,                               // Number of elements to add
            sADD                                   // CUDA stream
        );
    }

    return true;
}

// Explicit template instantiations for functions called from other compilation units
template bool hbdiaSpMV<float>(HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, HBDIAVector<float>& outputVector, bool execCOOCPU, bool execCOOGPU);
template bool hbdiaSpMV<double>(HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, HBDIAVector<double>& outputVector, bool execCOOCPU, bool execCOOGPU);

template void executeCOOOnGPU<float>(HBDIA<float>& matrix, const HBDIAVector<float>& inputVector, HBDIAVector<float>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent);
template void executeCOOOnGPU<double>(HBDIA<double>& matrix, const HBDIAVector<double>& inputVector, HBDIAVector<double>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent);
