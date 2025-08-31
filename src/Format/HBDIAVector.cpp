// HBDIAVector.cpp

#include "../../include/Format/HBDIAVector.hpp"
#include "../../include/Format/HBDIAPrinter.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <numa.h>

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
HBDIAVector<T>::HBDIAVector(const std::vector<T>& localData, bool unifiedMemory, bool unifiedMemoryMalloc, bool unifiedMemoryManagedMalloc, bool unifiedNumaAllocOnNode)
    : localVector_(localData){
        size_recv_left_ = 0;
        size_recv_right_ = 0;
        unifiedMemory_ = unifiedMemory;
        unifiedMemoryMalloc_ = unifiedMemoryMalloc;
        unifiedMemoryManagedMalloc_ = unifiedMemoryManagedMalloc;
        unifiedNumaAllocOnNode_ = unifiedNumaAllocOnNode;
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


    if(unifiedMemory_) { // Use unified memory allocation
        if (unifiedMemoryMalloc_) {
            // Use simple malloc for unified memory
            data_ptr_d_ = static_cast<T*>(malloc(totalSize * sizeof(T)));
            left_ptr_d_ = data_ptr_d_;
            local_ptr_d_ = data_ptr_d_ + size_recv_left_;
            right_ptr_d_ = data_ptr_d_ + size_recv_left_ + size_local_;
            CHECK_CUDA(cudaMemset(data_ptr_d_, 0, totalSize * sizeof(T)));
            cudaMemAdvise(data_ptr_d_, totalSize * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
            CHECK_CUDA(cudaMemcpy(local_ptr_d_, localVector_.data(), size_local_ * sizeof(T), cudaMemcpyHostToDevice));

            coo_result_ptr_d_ = static_cast<T*>(malloc(size_local_ * sizeof(T)));
            CHECK_CUDA(cudaMemset(coo_result_ptr_d_, 0, size_local_ * sizeof(T)));
            cudaMemAdvise(coo_result_ptr_d_, size_local_ * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
            printf("Unified memory allocated with malloc: %zu bytes\n", totalSize * sizeof(T) + size_local_ * sizeof(T));
        } else if (unifiedMemoryManagedMalloc_) {
            // Use cudaMallocManaged for managed memory
            CHECK_CUDA(cudaMallocManaged(&data_ptr_d_, totalSize * sizeof(T)));
            left_ptr_d_ = data_ptr_d_;
            local_ptr_d_ = data_ptr_d_ + size_recv_left_;
            right_ptr_d_ = data_ptr_d_ + size_recv_left_ + size_local_;
            CHECK_CUDA(cudaMemset(data_ptr_d_, 0, totalSize * sizeof(T)));
            cudaMemAdvise(data_ptr_d_, totalSize * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
            CHECK_CUDA(cudaMemcpy(local_ptr_d_, localVector_.data(), size_local_ * sizeof(T), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMallocManaged(&coo_result_ptr_d_, size_local_ * sizeof(T)));
            CHECK_CUDA(cudaMemset(coo_result_ptr_d_, 0, size_local_ * sizeof(T)));
            cudaMemAdvise(coo_result_ptr_d_, size_local_ * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
            printf("Unified memory allocated with cudaMallocManaged: %zu bytes\n", totalSize * sizeof(T) + size_local_ * sizeof(T));
        } else if (unifiedNumaAllocOnNode_) {
            // Use numa_alloc_onnode for NUMA-aware allocation
            data_ptr_d_ = static_cast<T*>(numa_alloc_onnode(totalSize * sizeof(T), 4 + 8 * deviceId));
            left_ptr_d_ = data_ptr_d_;
            local_ptr_d_ = data_ptr_d_ + size_recv_left_;
            right_ptr_d_ = data_ptr_d_ + size_recv_left_ + size_local_;
            CHECK_CUDA(cudaMemset(data_ptr_d_, 0, totalSize * sizeof(T)));
            cudaMemAdvise(data_ptr_d_, totalSize * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
            CHECK_CUDA(cudaMemcpy(local_ptr_d_, localVector_.data(), size_local_ * sizeof(T), cudaMemcpyHostToDevice));

            coo_result_ptr_d_ = static_cast<T*>(numa_alloc_onnode(size_local_ * sizeof(T), 4 + 8 * deviceId));
            CHECK_CUDA(cudaMemset(coo_result_ptr_d_, 0, size_local_ * sizeof(T)));
            cudaMemAdvise(coo_result_ptr_d_, size_local_ * sizeof(T), cudaMemAdviseSetPreferredLocation, deviceId);
            printf("Unified memory allocated with numa_alloc_onnode: %zu bytes\n", totalSize * sizeof(T) + size_local_ * sizeof(T));
        } else {
            std::cerr << "Error: No valid unified memory allocation method specified." << std::endl;
            throw std::runtime_error("Failed to setup unified memory for HBDIAVector");
        }
    } else { // Seperate allocation without unified memory
        //GPU
        CHECK_CUDA(cudaMalloc(&data_ptr_d_, totalSize * sizeof(T)));
        CHECK_CUDA(cudaMemset(data_ptr_d_, 0, totalSize * sizeof(T)));
        left_ptr_d_ = data_ptr_d_;
        local_ptr_d_ = data_ptr_d_ + size_recv_left_;
        right_ptr_d_ = data_ptr_d_ + size_recv_left_ + size_local_;
        CHECK_CUDA(cudaMemcpy(local_ptr_d_, localVector_.data(), size_local_ * sizeof(T), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&coo_result_ptr_d_, size_local_ * sizeof(T)));
        CHECK_CUDA(cudaMemset(coo_result_ptr_d_, 0, size_local_ * sizeof(T)));

        //CPU
        CHECK_CUDA(cudaMallocHost(&data_ptr_h_, totalSize * sizeof(T)));
        std::fill(data_ptr_h_, data_ptr_h_ + totalSize, T{});
        left_ptr_h_ = data_ptr_h_;
        local_ptr_h_ = data_ptr_h_ + size_recv_left_;
        right_ptr_h_ = data_ptr_h_ + size_recv_left_ + size_local_;

        CHECK_CUDA(cudaMallocHost(&coo_result_ptr_h_, size_local_ * sizeof(T)));
        std::fill(coo_result_ptr_h_, coo_result_ptr_h_ + size_local_, T{});
        printf("Separate CPU/GPU memory allocated: %zu bytes\n", totalSize * sizeof(T) + size_local_ * sizeof(T));
    }
}

template <typename T>
void HBDIAVector<T>::cleanupUnifiedMemory() {
    if (unifiedMemory_) {
        // Free managed memory if allocated
        if (data_ptr_d_) {
            if (unifiedMemoryMalloc_ || unifiedMemoryManagedMalloc_) {
                cudaFree(data_ptr_d_);
            } else if (unifiedNumaAllocOnNode_) {
                //numa_free(data_ptr_d_, (size_recv_left_ + size_local_ + size_recv_right_) * sizeof(T));
            }
            data_ptr_d_ = nullptr;
        }
        if (coo_result_ptr_d_) {
            if (unifiedMemoryMalloc_ || unifiedMemoryManagedMalloc_) {
                cudaFree(coo_result_ptr_d_);
            } else if (unifiedNumaAllocOnNode_) {
                //numa_free(coo_result_ptr_d_, size_local_ * sizeof(T));
            }
            coo_result_ptr_d_ = nullptr;
        }
    } else {
        // Free separate GPU memory
        CHECK_CUDA(cudaFree(data_ptr_d_));
        CHECK_CUDA(cudaFree(coo_result_ptr_d_));
        
        // Free CPU memory
        CHECK_CUDA(cudaFreeHost(data_ptr_h_));
        CHECK_CUDA(cudaFreeHost(coo_result_ptr_h_));
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
