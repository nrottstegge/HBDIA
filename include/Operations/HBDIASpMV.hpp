#ifndef HBDIASPMV_HPP
#define HBDIASPMV_HPP

#include <cuda_runtime.h>
#include "../Format/HBDIA.hpp"
#include "../Format/HBDIAVector.hpp"

template<typename T>
bool hbdiaSpMV(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, HBDIAVector<T>& outputVector, 
               bool execCooCPU, bool execCooGPU);

// Separate method for CPU COO execution
template<typename T>
void executeCOOOnCPU(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, 
                     HBDIAVector<T>& outputVector, int numRows, 
                     cudaStream_t sD2H, cudaStream_t sH2D, cudaStream_t sCOO, cudaEvent_t cooEvent);

// Method for GPU COO execution using cuSPARSE
template<typename T>
void executeCOOOnGPU(HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, 
                     HBDIAVector<T>& outputVector, int numRows, cudaStream_t sCOO, cudaEvent_t cooEvent);

// CPU helper function for processing COO fallback entries with separate accumulation buffer
template<typename T>
void hbdia_cpu_coo_spmv(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    T* cpuResults  // CPU results buffer from HBDIAVector
);

// CPU helper function for processing COO fallback entries in distributed matrices
template<typename T>
void hbdia_cpu_coo_spmv_partial(
    const std::vector<int>& cpuRowIndices,
    const std::vector<int>& cpuColIndices,
    const std::vector<T>& cpuValues,
    const T* inputVector,
    int numRows,
    T* cpuResults,  // CPU results buffer from HBDIAVector
    const HBDIA<T>& matrix
);

#endif // HBDIASPMV_HPP
