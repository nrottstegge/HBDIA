#ifndef HBDIASPMV_CUH
#define HBDIASPMV_CUH

#include "../Format/HBDIA.hpp"
#include "../Format/HBDIAVector.hpp"

// Hybrid GPU+CPU HBDIA SpMV function
// Processes HBDIA block-diagonal elements on GPU and COO fallback elements on CPU concurrently
// Supports both single GPU and distributed (partial matrix) scenarios
// Uses separate CPU accumulation buffer to avoid GPU memory contention
template<typename T>
bool hbdiaSpMV(const HBDIA<T>& matrix, const HBDIAVector<T>& inputVector, HBDIAVector<T>& outputVector);

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

#endif // HBDIASPMV_CUH
