// HBDIAVector.cpp

#include "../../include/Format/HBDIAVector.hpp"
#include "../../include/Format/HBDIAPrinter.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
HBDIAVector<T>::HBDIAVector(){
}

template <typename T>
HBDIAVector<T>::HBDIAVector(const std::vector<T>& localData)
    : localVector_(localData){
        size_recv_left_ = 0;
        size_recv_right_ = 0;
        size_local_ = localData.size();
        setupUnifiedMemory();
}

template <typename T>
HBDIAVector<T>::HBDIAVector(const std::vector<T>& localData, const HBDIA<T>& matrix, int rank, int size)
    : localVector_(localData) {

    // Store local size before we clear the vector
    size_local_ = localData.size();
    
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
    cleanupUnifiedMemory();
}

template <typename T>
void HBDIAVector<T>::setupUnifiedMemory() {
    size_t totalSize = size_recv_left_ + size_local_ + size_recv_right_;
    int deviceId;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    
    CHECK_CUDA(cudaMallocManaged(&data_ptr_d_, totalSize * sizeof(T)));

    CHECK_CUDA(cudaMemset(data_ptr_d_, 0, totalSize * sizeof(T)));
    left_ptr_d_ = data_ptr_d_;
    local_ptr_d_ = data_ptr_d_ + size_recv_left_;
    right_ptr_d_ = data_ptr_d_ + size_recv_left_ + size_local_;
    CHECK_CUDA(cudaMemcpy(local_ptr_d_, localVector_.data(), size_local_ * sizeof(T), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMallocManaged(&cpu_result_ptr_d_, size_local_ * sizeof(T)));
    CHECK_CUDA(cudaMemset(cpu_result_ptr_d_, 0, size_local_ * sizeof(T)));

    //data_ptr_h_ = (T*)malloc(totalSize * sizeof(T));
    CHECK_CUDA(cudaMallocManaged(&data_ptr_h_, totalSize * sizeof(T)));
    //cudaMemAdvise(data_ptr_h_, totalSize * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
    std::fill(data_ptr_h_, data_ptr_h_ + totalSize, T{});
    left_ptr_h_ = data_ptr_h_;
    local_ptr_h_ = data_ptr_h_ + size_recv_left_;
    right_ptr_h_ = data_ptr_h_ + size_recv_left_ + size_local_;
    
    //cpu_result_ptr_h_ = (T*)malloc(size_local_ * sizeof(T));
    CHECK_CUDA(cudaMallocManaged(&cpu_result_ptr_h_, size_local_ * sizeof(T)));
    //cudaMemAdvise(cpu_result_ptr_h_, size_local_ * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
    std::fill(cpu_result_ptr_h_, cpu_result_ptr_h_ + size_local_, T{});

    // Set up unified pointers to point to the same allocated memory
    unified_left_ptr_ = nullptr;
    unified_local_ptr_ = nullptr;
    unified_right_ptr_ = nullptr;

    // cleanupUnifiedMemory(); // Clean up any existing memory
    
    // size_t totalSize = size_recv_left_ + size_local + size_recv_right_;
    // if (totalSize == 0) return;
    
    // // Allocate regular malloc memory to enable ATS on GH200
    // // Note: Using malloc instead of cudaMallocManaged to enable ATS service
    // //unified_data_ptr_ = static_cast<T*>(malloc(totalSize * sizeof(T)));
    // CHECK_CUDA(cudaMallocManaged(&unified_data_ptr_, totalSize * sizeof(T)));
    
    // if (!unified_data_ptr_) {
    //     throw std::runtime_error("Failed to allocate memory for vector");
    // }
    // cpu_results_ptr_ = static_cast<T*>(malloc(size_local * sizeof(T)));
    // if (!cpu_results_ptr_) {
    //     throw std::runtime_error("Failed to allocate CPU results buffer");
    // }
    
    // // Set up memory layout: [left_buffer | local_data | right_buffer]
    // // Initialize the three section pointers
    // unified_left_ptr_ = unified_data_ptr_;
    // unified_local_ptr_ = unified_data_ptr_ + size_recv_left_;
    // unified_right_ptr_ = unified_data_ptr_ + size_recv_left_ + size_local;
    
    // // Apply Brook University cudaMemAdvise optimizations
    // size_t totalBytes = totalSize * sizeof(T);
    // int deviceId;
    // if (cudaGetDevice(&deviceId) == cudaSuccess) {
        
    //     // Set preferred location based on usage pattern
    //     if (usage_ == VectorUsage::INPUT) {
    //         // Input vectors: prefer GPU location for faster kernel access
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetPreferredLocation, deviceId);
            
    //         // Mark as read-mostly for input vectors
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetReadMostly, deviceId);
            
    //         // Indicate which device will access this memory
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetAccessedBy, deviceId);
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
            
    //     } else if (usage_ == VectorUsage::OUTPUT) {
    //         // Output vectors: prefer GPU location since GPU writes results
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetPreferredLocation, deviceId);
            
    //         // Indicate which devices will access this memory
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetAccessedBy, deviceId);
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    //         // cpu result should be on GPU, cpu will write, gpu only reads
    //         cudaMemAdvise(cpu_results_ptr_, size_local * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
    //         cudaMemAdvise(cpu_results_ptr_, size_local * sizeof(T), cudaMemAdviseSetReadMostly, deviceId);
    //         cudaMemAdvise(cpu_results_ptr_, size_local * sizeof(T), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
            
    //     } else { // VectorUsage::INOUT
    //         // For input/output vectors: balanced approach
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetPreferredLocation, deviceId);
            
    //         // Indicate both CPU and GPU will access
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetAccessedBy, deviceId);
    //         cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    //     }
    // }
    // cudaMemAdvise(unified_data_ptr_, totalBytes, cudaMemAdviseSetPreferredLocation, deviceId);

    // // Copy local data to unified memory
    // std::copy(localVector_.begin(), localVector_.end(), unified_local_ptr_);
    
    // // Clear the host vector to save memory - data is now in unified memory
    // localVector_.clear();
    // localVector_.shrink_to_fit();
    
    // // Initialize right buffer section to zero (will be filled via MPI communication)
    // if (size_recv_right_ > 0) {
    //     std::fill(unified_right_ptr_, unified_right_ptr_ + size_recv_right_, T{});
    // }

    // // Initialize left buffer section to zero (will be filled via MPI communication)
    // if (size_recv_left_ > 0) {
    //     std::fill(unified_left_ptr_, unified_left_ptr_ + size_recv_left_, T{});
    // }

    // // Initialize to zero for accumulation
    // std::fill(cpu_results_ptr_, cpu_results_ptr_ + size_local, T{});

    // //cudaMemPrefetchAsync(unified_data_ptr_, totalBytes, deviceId);
    // cudaMemPrefetchAsync(cpu_results_ptr_, size_local * sizeof(T), deviceId);
    // cudaMemPrefetchAsync(unified_data_ptr_, totalBytes, deviceId);
    
}

template <typename T>
void HBDIAVector<T>::cleanupUnifiedMemory() {
    if(data_ptr_d_) {
        cudaFree(data_ptr_d_);
        data_ptr_d_ = nullptr;
    }
    if(cpu_result_ptr_d_) {
        cudaFree(cpu_result_ptr_d_);
        cpu_result_ptr_d_ = nullptr;
    }
    if(data_ptr_h_) {
        cudaFree(data_ptr_h_);
        data_ptr_h_ = nullptr;
    }
    if(cpu_result_ptr_h_) {
        cudaFree(cpu_result_ptr_h_);
        cpu_result_ptr_h_ = nullptr;
    }
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
