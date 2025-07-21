// HBDIAVector.cpp

#include "../../include/Format/HBDIAVector.hpp"
#include "../../include/Format/HBDIAPrinter.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
HBDIAVector<T>::HBDIAVector() 
    : unified_data_ptr_(nullptr), unified_left_ptr_(nullptr), 
      unified_local_ptr_(nullptr), unified_right_ptr_(nullptr) {
}

template <typename T>
HBDIAVector<T>::HBDIAVector(const std::vector<T>& localData) //singleGPU version
    : localVector_(localData), unified_data_ptr_(nullptr), unified_left_ptr_(nullptr),
      unified_local_ptr_(nullptr), unified_right_ptr_(nullptr) {
        size_recv_left_ = 0;
        size_recv_right_ = 0;
        size_local = localData.size();
        setupUnifiedMemory();
}

template <typename T>
HBDIAVector<T>::HBDIAVector(const std::vector<T>& localData, const HBDIA<T>& matrix, int rank, int size) //multiGPU version
    : localVector_(localData), unified_data_ptr_(nullptr), unified_left_ptr_(nullptr),
      unified_local_ptr_(nullptr), unified_right_ptr_(nullptr) {
    
    // Store local size before we clear the vector
    size_local = localData.size();
    
    // Calculate buffer sizes from matrix metadata if it's a partial matrix
    if (matrix.isPartialMatrix() && matrix.hasPartialMatrixMetadata()) {
        const auto& metadata = matrix.getPartialMatrixMetadata();
        
        // Calculate left and right buffer sizes based on data ranges
        size_recv_left_ = 0;
        size_recv_right_ = 0;
        
        for (int procId = 0; procId < size; procId++) {
            if (procId == rank) continue; // Skip our own process
            
            for (const auto& range : metadata.processDataRanges[procId]) {
                int rangeSize = std::get<1>(range) - std::get<0>(range);
                
                if (procId < rank) {
                    size_recv_left_ += rangeSize;
                } else {
                    size_recv_right_ += rangeSize;
                }
            }
        }
    }
    
    // Setup unified memory layout
    setupUnifiedMemory();
}

template <typename T>
HBDIAVector<T>::~HBDIAVector() {
    cleanupGPUData();
    cleanupUnifiedMemory();
}

template <typename T>
void HBDIAVector<T>::setupUnifiedMemory() {
    cleanupUnifiedMemory(); // Clean up any existing memory
    
    size_t totalSize = size_recv_left_ + size_local + size_recv_right_;
    if (totalSize == 0) return;
    
    // Allocate unified memory
    if (cudaMallocManaged(&unified_data_ptr_, totalSize * sizeof(T)) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA managed memory for vector");
    }
    
    // Set up memory layout: [left_buffer | local_data | right_buffer]
    // Initialize the three section pointers
    unified_left_ptr_ = unified_data_ptr_;
    unified_local_ptr_ = unified_data_ptr_ + size_recv_left_;
    unified_right_ptr_ = unified_data_ptr_ + size_recv_left_ + size_local;
    
    // Initialize left buffer section to zero (will be filled via MPI communication)
    if (size_recv_left_ > 0) {
        std::fill(unified_left_ptr_, unified_left_ptr_ + size_recv_left_, T{});
    }
    
    // Copy local data
    std::copy(localVector_.begin(), localVector_.end(), unified_local_ptr_);
    
    // Clear the host vector to save memory - data is now in unified memory
    localVector_.clear();
    localVector_.shrink_to_fit();
    
    // Initialize right buffer section to zero (will be filled via MPI communication)
    if (size_recv_right_ > 0) {
        std::fill(unified_right_ptr_, unified_right_ptr_ + size_recv_right_, T{});
    }
}

template <typename T>
void HBDIAVector<T>::cleanupUnifiedMemory() {
    if (unified_data_ptr_) {
        cudaFree(unified_data_ptr_);
        unified_data_ptr_ = nullptr;
        unified_left_ptr_ = nullptr;
        unified_local_ptr_ = nullptr;
        unified_right_ptr_ = nullptr;
    }
}


template <typename T>
void HBDIAVector<T>::cleanupGPUData() {
    if (d_unified_data_ptr_) {
        cudaFree(d_unified_data_ptr_);
        d_unified_data_ptr_ = nullptr;
    }
    gpuDataPrepared_ = false;
}

template <typename T>
void HBDIAVector<T>::print(const std::string& vectorName) const {
    // Forward to HBDIAPrinter for the actual implementation
    HBDIAPrinter<T>::printVector(*this, vectorName);
}

// Explicit template instantiations
template class HBDIAVector<double>;
template class HBDIAVector<float>;
template class HBDIAVector<int>;
