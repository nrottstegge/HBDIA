// HBDIAVector.cpp
#include "../../include/Format/HBDIAVector.hpp"
#include "../../include/DataExchange/HBDIAExtractor.hpp"
#include "../../include/DataExchange/HBDIADistributor.hpp"
#include "../../include/types.hpp"
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>

template <typename T>
HBDIAVector<T>::HBDIAVector() {
    // Default constructor
}

template <typename T>
HBDIAVector<T>::HBDIAVector(const std::vector<T>& localData) 
    : localVector_(localData) {
}

template <typename T>
HBDIAVector<T>::~HBDIAVector() {
    // Clean up unified memory
    if (unified_data_ptr_) {
        CHECK_CUDA(cudaFree(unified_data_ptr_));
        unified_data_ptr_ = nullptr;
    }
}

template <typename T>
void HBDIAVector<T>::addApplicableMatrix(std::shared_ptr<HBDIA<T>> matrix) {
    if (!matrix) {
        std::cerr << "Error: Cannot add null matrix to HBDIAVector" << std::endl;
        return;
    }
    
    // Check if matrix already exists
    auto it = std::find_if(applicableMatrices_.begin(), applicableMatrices_.end(),
        [matrix](const ApplicableMatrix& am) {
            return am.matrix == matrix;
        });
    
    if (it != applicableMatrices_.end()) {
        return; // Matrix already exists, silently skip
    }
    
    // Add the matrix
    applicableMatrices_.emplace_back(matrix);
}

template <typename T>
void HBDIAVector<T>::removeApplicableMatrix(std::shared_ptr<HBDIA<T>> matrix) {
    auto it = std::find_if(applicableMatrices_.begin(), applicableMatrices_.end(),
        [matrix](const ApplicableMatrix& am) {
            return am.matrix == matrix;
        });
    
    if (it != applicableMatrices_.end()) {
        applicableMatrices_.erase(it);
    }
}

template <typename T>
void HBDIAVector<T>::addApplicableMatrix(HBDIA<T>& matrix, HBDIADistributor<T>& distributor) {
    // Create a shared_ptr to the matrix for storage in ApplicableMatrix
    auto matrixPtr = std::shared_ptr<HBDIA<T>>(&matrix, [](HBDIA<T>*){});
    
    // First add the matrix using the basic method
    addApplicableMatrix(matrixPtr);
    
    // If it's a partial matrix, process metadata using the distributor's extractor
    if (matrix.isPartialMatrix()) {
        // Get the extractor from the distributor
        auto& extractor = distributor.getExtractor();
        
        // Process metadata directly into the matrix
        extractor.extractPartialMatrixMetadata(matrix, distributor.getSize());
        
        // Get the processed metadata from the matrix
        if (!matrix.hasPartialMatrixMetadata()) {
            std::cerr << "Failed to extract partial matrix metadata" << std::endl;
            return;
        }
        
        const auto& metadata = matrix.getPartialMatrixMetadata();
        
        // Find the applicable matrix we just added
        auto it = std::find_if(applicableMatrices_.begin(), applicableMatrices_.end(),
            [matrixPtr](const ApplicableMatrix& am) {
                return am.matrix == matrixPtr;
            });
        
        if (it != applicableMatrices_.end()) {
            
            // Store rank for later use
            rank_ = distributor.getRank();
            
            // Calculate buffer sizes for left and right processes
            for(int i = 0; i < distributor.getSize(); i++){
                if(i == distributor.getRank()) continue;
                for(const auto& range : metadata.processDataRanges[i]) {
                    int rangeStart = std::get<0>(range);
                    int rangeEnd = std::get<1>(range);
                    int bufferSize = rangeEnd - rangeStart;
                    if(i < distributor.getRank()) {
                        size_recv_left_ += bufferSize;
                    } else {
                        size_recv_right_ += bufferSize;
                    }
                }
            }
            
            // Allocate unified memory (GPU+CPU accessible)
            size_t total_size = size_recv_left_ + localVector_.size() + size_recv_right_;

            CHECK_CUDA(cudaMallocManaged(&unified_data_ptr_, total_size * sizeof(T)));
            
            // Set up memory layout: [left_recv][local_data][right_recv]
            T* left_start = unified_data_ptr_;
            local_data_start_ = unified_data_ptr_ + size_recv_left_;
            T* right_start = local_data_start_ + localVector_.size();
            
            // Copy local vector data
            std::copy(localVector_.begin(), localVector_.end(), local_data_start_);
            
            // Initialize receive buffers
            recvBuffer_left_.resize(size_recv_left_);
            recvBuffer_right_.resize(size_recv_right_);
            
            // Set up block row to data pointer mapping for fast GPU access
            setupBlockRowPointersRowWise(matrix, distributor.getRank(), distributor.getSize());
        }

    }
}

template <typename T>
void HBDIAVector<T>::(const HBDIA<T>& matrix, int rank, int size) {
    // Check if matrix is in HBDIA format
    if (!matrix.isHBDIAFormat()) {
        std::cerr << "Error: Matrix must be in HBDIA format for block row pointer setup" << std::endl;
        return;
    }
    
    // Check if matrix has metadata (for partial matrices)
    if (matrix.isPartialMatrix() && !matrix.hasPartialMatrixMetadata()) {
        std::cerr << "Error: Partial matrix missing metadata for block row pointer setup" << std::endl;
        return;
    }
    
    const std::vector<std::vector<int>>& offsetsPerBlock = matrix.getOffsetsPerBlock();
    int blockWidth = matrix.getBlockWidth();
    int numBlocks = offsetsPerBlock.size();
    int rowsPerProcess = matrix.getNumGlobalRows() / size;
    globalStart_ = rank * rowsPerProcess;
    globalEnd_ = globalStart_ + rowsPerProcess + ((rank == size - 1) ? matrix.getNumGlobalRows() % rowsPerProcess : 0);
    
    // Initialize block row pointers
    block_row_to_data_ptr_.resize(numBlocks);
    
    // For each block
    for (int blockId = 0; blockId < numBlocks; blockId++) {
        block_row_to_data_ptr_[blockId].clear();
        
        // Calculate global data range for this block: [rank * rowsPerProcess + blockId * blockWidth, rank * rowsPerProcess + (blockId+1) * blockWidth)
        int global_block_start0 = rank * rowsPerProcess + blockId * blockWidth;
        
        // Go through each offset in this block
        const auto& offsets = offsetsPerBlock[blockId];
        for (int offset : offsets) {
            // The offset describes which column index within this block's row range is needed
            int global_data_index = global_block_start0 + offset;
            
            // Find where this data is located using dataRange tuples
            T* data_ptr = findDataPointerForGlobalIndexRowWise(global_data_index, matrix);
            block_row_to_data_ptr_[blockId].push_back(data_ptr);
        }
    }
}

template <typename T>
T* HBDIAVector<T>::findDataPointerForGlobalIndexRowWise(int globalIndex, const HBDIA<T>& matrix) {
    // Check if it's in local data range
    if (globalIndex >= globalStart_ && globalIndex < globalEnd_) {
        int localIndex = globalIndex - globalStart_;
        return local_data_start_ + localIndex;
    }
    
    // For partial matrices, search in receive buffer ranges using matrix metadata
    if (!matrix.isPartialMatrix() || !matrix.hasPartialMatrixMetadata()) {
        return nullptr; // Non-partial matrix or no metadata available
    }
    
    const auto& metadata = matrix.getPartialMatrixMetadata();
    
    // Search in receive buffer ranges
    int procId = 0;
    int offset = 0;
    
    if (globalIndex < globalStart_) {
        // Search in left buffer (lower process IDs)
        offset = 0;
        procId = 0;
    } else {
        // Search in right buffer (higher process IDs)
        offset = size_recv_left_ + localVector_.size();
        procId = rank_ + 1;
    }

    for (; procId < metadata.processDataRanges.size(); procId++) {
        // Skip our own process
        if (procId == rank_) continue;
        
        for (const auto& range : metadata.processDataRanges[procId]) {
            int rangeStart = std::get<0>(range);
            int rangeEnd = std::get<1>(range);
            
            if (globalIndex >= rangeStart && globalIndex < rangeEnd) {
                int offsetInRange = globalIndex - rangeStart;
                return unified_data_ptr_ + offset + offsetInRange;
            }
            
            // Only increment offset for the buffer we're searching in
            if ((globalIndex < globalStart_ && procId < rank_) || 
                (globalIndex >= globalEnd_ && procId > rank_)) {
                offset += rangeEnd - rangeStart;
            }
        }
    }
    
    return nullptr;  // Not found
}

template <typename T>
void HBDIAVector<T>::print() const {
    std::cout << "=== HBDIA VECTOR ===" << std::endl;
    std::cout << "Local vector size: " << localVector_.size() << std::endl;
    std::cout << "Number of applicable matrices: " << applicableMatrices_.size() << std::endl;
    std::cout << "Unified memory size: " << (size_recv_left_ + localVector_.size() + size_recv_right_) << std::endl;
    std::cout << "Block rows: " << block_row_to_data_ptr_.size() << std::endl;
    std::cout << "CPU fallback pointers: " << cpu_fallback_ptrs_.size() << std::endl;
    std::cout << "===================" << std::endl;
}

// Explicit template instantiations
template class HBDIAVector<double>;
template class HBDIAVector<float>;
template class HBDIAVector<int>;