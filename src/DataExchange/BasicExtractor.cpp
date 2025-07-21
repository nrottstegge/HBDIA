// BasicExtractor.cpp

#include "../../include/DataExchange/BasicExtractor.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

template <typename T>
BasicExtractor<T>::BasicExtractor() {
    this->strategy_ = HBDIAExtractor<T>::PartitioningStrategy::ROW_WISE;
}

template <typename T>
BasicExtractor<T>::~BasicExtractor() = default;

template <typename T>
bool BasicExtractor<T>::createMatrixPartitions(
    const HBDIA<T>& matrix, 
    int numProcesses
) {
    if (numProcesses <= 0) {
        std::cerr << "Invalid number of processes: " << numProcesses << std::endl;
        return false;
    }

    if (!matrix.isCOOFormat()) {
        std::cerr << "Matrix must be in COO format for partitioning" << std::endl;
        return false;
    }
    
    // Clear existing data
    this->matrixData_.clear();
    this->partitions_.clear();
    this->matrixData_.resize(numProcesses);
    this->partitions_.resize(numProcesses);
    
    // Create partitions based on strategy
    switch (this->strategy_) {
        case HBDIAExtractor<T>::PartitioningStrategy::ROW_WISE:
            return partitionRowWise(matrix, numProcesses);
        default:
            std::cerr << "Unknown partitioning strategy" << std::endl;
            return false;
    }
}

template <typename T>
void BasicExtractor<T>::setPartitioningStrategy(typename HBDIAExtractor<T>::PartitioningStrategy strategy) {
    this->strategy_ = strategy;
}

template <typename T>
const std::vector<MatrixData>& BasicExtractor<T>::getMatrixData() const {
    return this->matrixData_;
}

template <typename T>
const std::vector<Partition<T>>& BasicExtractor<T>::getPartitions() const {
    return this->partitions_;
}

template <typename T>
const std::vector<VectorPartition<T>>& BasicExtractor<T>::getVectorPartitions() const {
    return this->vectorPartitions_;
}

template <typename T>
bool BasicExtractor<T>::createVectorPartitions(
    const std::vector<T>& globalVector,
    int numProcesses
) {
    if (numProcesses <= 0) {
        std::cerr << "Invalid number of processes: " << numProcesses << std::endl;
        return false;
    }
    
    if (globalVector.empty()) {
        std::cerr << "Global vector is empty" << std::endl;
        return false;
    }
    
    // Clear existing data
    this->vectorPartitions_.clear();
    this->vectorPartitions_.resize(numProcesses);
    
    // Create partitions based on strategy
    switch (this->strategy_) {
        case HBDIAExtractor<T>::PartitioningStrategy::ROW_WISE:
            return partitionVectorRowWise(globalVector, numProcesses);
        default:
            std::cerr << "Unknown partitioning strategy" << std::endl;
            return false;
    }
}

template <typename T>
bool BasicExtractor<T>::partitionVectorRowWise(const std::vector<T>& globalVector, int num_partitions) {
    int numElements = globalVector.size();
    
    // Calculate elements per partition (should match matrix row partitioning)
    int elementsPerPartition = numElements / num_partitions;
    int remainingElements = numElements % num_partitions;
    
    // Distribute vector elements to partitions using contiguous memory operations
    int currentIndex = 0;
    for (int p = 0; p < num_partitions; p++) {
        int localElements = elementsPerPartition;
        if (p == num_partitions - 1) {
            localElements += remainingElements; // Last partition gets the remainder
        }
        
        // Resize to exact size and copy contiguous block
        this->vectorPartitions_[p].localValues.resize(localElements);
        std::copy(globalVector.begin() + currentIndex, 
                  globalVector.begin() + currentIndex + localElements,
                  this->vectorPartitions_[p].localValues.begin());
        
        currentIndex += localElements;
    }
    
    return true;
}

template <typename T>
bool BasicExtractor<T>::partitionRowWise(const HBDIA<T>& matrix, int num_partitions) {
    // Access matrix properties through getters
    int numRows = matrix.getNumRows();
    int numCols = matrix.getNumCols();
    int numNonZeros = matrix.getNumNonZeros();
    
    if (numRows == 0 || numCols == 0 || numNonZeros == 0) {
        std::cerr << "Invalid matrix dimensions" << std::endl;
        return false;
    }
    
    // Calculate rows per partition
    int rowsPerPartition = numRows / num_partitions;
    int remainingRows = numRows % num_partitions;
    
    // Initialize MatrixData for each partition
    for (int i = 0; i < num_partitions; i++) {
        this->matrixData_[i].numGlobalRows = numRows;
        this->matrixData_[i].numGlobalCols = numCols;
        this->matrixData_[i].numGlobalNonZeros = numNonZeros;
        this->matrixData_[i].numLocalRows = rowsPerPartition;
        if (i == num_partitions - 1) {
            this->matrixData_[i].numLocalRows += remainingRows; // Last partition gets the remainder
        }
        this->matrixData_[i].numLocalCols = numCols; // All partitions have the same number of columns
        this->matrixData_[i].numLocalNonZeros = 0; // Will be calculated while distributing elements
    }
    
    // Initialize partitions
    for (int i = 0; i < num_partitions; i++) {
        this->partitions_[i].localRowIndices.clear();
        this->partitions_[i].localColIndices.clear();
        this->partitions_[i].localValues.clear();
        this->partitions_[i].globalRowMapping.clear();
    }
    
    // Calculate row ranges for each partition and create global row mapping
    std::vector<int> rowRangeStart(num_partitions);
    std::vector<int> rowRangeEnd(num_partitions);
    
    int currentRow = 0;
    for (int p = 0; p < num_partitions; p++) {
        rowRangeStart[p] = currentRow;
        int localRows = rowsPerPartition + (p == num_partitions - 1 ? remainingRows : 0);
        currentRow += localRows;
        rowRangeEnd[p] = currentRow - 1; // Inclusive end
        
        // Create global row mapping for this partition
        this->partitions_[p].globalRowMapping.reserve(localRows);
        for (int localRow = 0; localRow < localRows; localRow++) {
            int globalRow = rowRangeStart[p] + localRow;
            this->partitions_[p].globalRowMapping.push_back(globalRow);
        }
    }
    
    // Go through COO part of HBDIA matrix and distribute elements based on row ownership
    if (!matrix.getHasCOO()) {
        std::cerr << "Matrix does not have COO format available" << std::endl;
        return false;
    }
    
    // Get COO data through getters
    const auto& rowIndices = matrix.getRowIndices();
    const auto& colIndices = matrix.getColIndices();
    const auto& values = matrix.getValues();
    
    // Distribute COO elements to partitions based on row ownership
    for (int i = 0; i < numNonZeros; i++) {
        int row = rowIndices[i];
        int col = colIndices[i];
        T value = values[i];
        
        // Find which partition owns this row
        int ownerPartition = -1;
        for (int p = 0; p < num_partitions; p++) {
            if (row >= rowRangeStart[p] && row <= rowRangeEnd[p]) {
                ownerPartition = p;
                break;
            }
        }
        
        if (ownerPartition == -1) {
            std::cerr << "Error: Row " << row << " not assigned to any partition" << std::endl;
            return false;
        }
        
        // Add this element to the owner partition (convert to local row index)
        int localRow = row - rowRangeStart[ownerPartition];
        this->partitions_[ownerPartition].localRowIndices.push_back(localRow);
        this->partitions_[ownerPartition].localColIndices.push_back(col);
        this->partitions_[ownerPartition].localValues.push_back(value);
        
        // Update local non-zeros count
        this->matrixData_[ownerPartition].numLocalNonZeros++;
    }
    
    // Validate partitioning
    int totalDistributedNonZeros = 0;
    for (int p = 0; p < num_partitions; p++) {
        totalDistributedNonZeros += this->matrixData_[p].numLocalNonZeros;
    }
    
    if (totalDistributedNonZeros != numNonZeros) {
        std::cerr << "Error: Distributed " << totalDistributedNonZeros 
                  << " non-zeros but matrix has " << numNonZeros << std::endl;
        return false;
    }
    
    return true;
}

template <typename T>
void BasicExtractor<T>::clearMatrixPartitions() {
    this->matrixData_.clear();
    this->partitions_.clear();
}

template <typename T>
void BasicExtractor<T>::clearVectorPartitions() {
    this->vectorPartitions_.clear();
}

template <typename T>
void BasicExtractor<T>::extractPartialMatrixMetadata(
    HBDIA<T>& matrix,
    int numProcesses
) {
    // Use the current partitioning strategy
    switch (this->strategy_) {
        case HBDIAExtractor<T>::PartitioningStrategy::ROW_WISE:
            extractPartialMatrixMetadataRowWise(matrix, numProcesses);
            break;
        default:
            std::cerr << "Unknown partitioning strategy" << std::endl;
            // Initialize empty metadata for unknown strategy
            if (matrix.isPartialMatrix()) {
                PartialMatrixMetadata<T> emptyMetadata;
                emptyMetadata.totalBufferSize = 0;
                matrix.setPartialMatrixMetadata(emptyMetadata);
            }
            break;
    }
}

template <typename T>
void BasicExtractor<T>::extractPartialMatrixMetadataRowWise(
    HBDIA<T>& matrix,
    int numProcesses
) {
    // Only process if it's a partial matrix
    if (!matrix.isPartialMatrix()) {
        return;
    }
    
    // Get the data ranges that this matrix needs
    const auto& dataRanges = matrix.getDataRanges();
    
    // Initialize new metadata
    PartialMatrixMetadata<T> metadata;
    metadata.dataRanges = dataRanges;
    metadata.totalBufferSize = 0;
    
    // Calculate row distribution across processes (same as in partitionRowWise)
    int numGlobalRows = matrix.getNumGlobalRows();
    int rowsPerProcess = numGlobalRows / numProcesses;
    
    // Initialize process data ranges
    metadata.processDataRanges.resize(numProcesses);
    
    // Process each data range using simplified division-based approach
    for (const auto& range : dataRanges) {
        int rangeStart = std::get<0>(range);
        int rangeEnd = std::get<1>(range); // Exclusive end
        
        // Add to total buffer size
        metadata.totalBufferSize += (rangeEnd - rangeStart);
        
        // Calculate which processes are involved using division
        int startProcess = rangeStart / rowsPerProcess;
        int endProcess = (rangeEnd - 1) / rowsPerProcess; // -1 because rangeEnd is exclusive
        
        // Clamp to valid process range
        startProcess = std::max(0, std::min(startProcess, numProcesses - 1));
        endProcess = std::max(0, std::min(endProcess, numProcesses - 1));
        
        // Split the range across process boundaries
        int currentPos = rangeStart;
        
        for (int p = startProcess; p <= endProcess && currentPos < rangeEnd; ++p) {
            // Calculate this process's boundary
            int processBoundary = (p + 1) * rowsPerProcess;
            if(p == numProcesses - 1) {
                processBoundary = rangeEnd; // Last process takes the remainder
            }
            
            // Find the end position for this process
            int segmentEnd = std::min(rangeEnd, processBoundary);
            
            // Add the segment if it's valid
            if (currentPos < segmentEnd) {
                metadata.processDataRanges[p].emplace_back(std::make_tuple(currentPos, segmentEnd));
                currentPos = segmentEnd;
            }
        }
    }
    
    // Store the metadata in the matrix
    matrix.setPartialMatrixMetadata(metadata);
}

template <typename T>
void BasicExtractor<T>::printProcessedDataRanges(const HBDIA<T>& matrix) {
    std::cout << "=== PROCESSED DATA RANGES ===" << std::endl;
    
    if (!matrix.isPartialMatrix()) {
        std::cout << "Matrix is not a partial matrix - no processed data ranges to display" << std::endl;
        std::cout << "=================================" << std::endl;
        return;
    }
    
    // Check if the matrix has processed metadata
    if (!matrix.hasPartialMatrixMetadata()) {
        std::cout << "No processed metadata available - call extractPartialMatrixMetadata first" << std::endl;
        std::cout << "=================================" << std::endl;
        return;
    }
    
    const auto& metadata = matrix.getPartialMatrixMetadata();
    
    // Print original data ranges
    std::cout << "Original data ranges needed: " << metadata.dataRanges.size() << std::endl;
    for (size_t i = 0; i < metadata.dataRanges.size(); ++i) {
        auto [start, end] = metadata.dataRanges[i];
        int rangeSize = end - start;
        std::cout << "  Range " << (i + 1) << ": [" << start << ", " << end << ") -> " 
                  << rangeSize << " elements" << std::endl;
    }
    
    // Print process ownership summary
    std::cout << "\nProcess ownership mapping:" << std::endl;
    for (size_t p = 0; p < metadata.processDataRanges.size(); ++p) {
        if (!metadata.processDataRanges[p].empty()) {
            std::cout << "  Process " << p << " will provide " 
                      << metadata.processDataRanges[p].size() << " range(s):";
            for (const auto& range : metadata.processDataRanges[p]) {
                std::cout << " [" << std::get<0>(range) << ", " << std::get<1>(range) << ")";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "\nTotal buffer size needed: " << metadata.totalBufferSize << " elements" << std::endl;
    std::cout << "=================================" << std::endl;
}

// Explicit template instantiations
template class BasicExtractor<int>;
template class BasicExtractor<float>;
template class BasicExtractor<double>;