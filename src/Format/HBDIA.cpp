// HBDIA.cpp
#include "HBDIA.hpp"
#include "HBDIAPrinter.hpp"
#include "types.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <unordered_set>
#include <map>
#include <cuda_runtime.h>
#include <limits.h>

template <typename T>
HBDIA<T>::HBDIA() : numRows(0), numCols(0), numNonZeros(0), partialMatrix(false), hasCOO(false), hasDIA(false), hasHBDIA(false) {}

template <typename T>
HBDIA<T>::HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values) 
    : values(values), rowIndices(rowIndices), colIndices(colIndices), partialMatrix(false), hasCOO(true), hasDIA(false), hasHBDIA(false) {
    
    // Validate that all vectors have the same size
    if (rowIndices.size() != colIndices.size() || rowIndices.size() != values.size()) {
        throw std::invalid_argument("Row indices, column indices, and values vectors must have the same size");
    }
    
    numNonZeros = static_cast<int>(values.size());
    
    // Calculate matrix dimensions by finding max indices
    numRows = 0;
    numCols = 0;
    
    for (int row : rowIndices) {
        if (row >= numRows) {
            numRows = row + 1;  // +1 because indices are 0-based
        }
    }
    
    for (int col : colIndices) {
        if (col >= numCols) {
            numCols = col + 1;  // +1 because indices are 0-based
        }
    }
}

template <typename T>
HBDIA<T>::HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values, 
                int numRows, int numCols) 
    : values(values), rowIndices(rowIndices), colIndices(colIndices), numRows(numRows), numCols(numCols), 
      partialMatrix(false), hasCOO(true), hasDIA(false), hasHBDIA(false) {
    
    // Validate that all vectors have the same size
    if (rowIndices.size() != colIndices.size() || rowIndices.size() != values.size()) {
        throw std::invalid_argument("Row indices, column indices, and values vectors must have the same size");
    }
    
    // Validate dimensions
    if (numRows <= 0 || numCols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    numNonZeros = static_cast<int>(values.size());
    
    // Validate that all indices are within bounds
    for (size_t i = 0; i < rowIndices.size(); ++i) {
        if (rowIndices[i] < 0 || rowIndices[i] >= numRows) {
            throw std::out_of_range("Row index " + std::to_string(rowIndices[i]) + 
                                  " is out of range [0, " + std::to_string(numRows-1) + "]");
        }
        if (colIndices[i] < 0 || colIndices[i] >= numCols) {
            throw std::out_of_range("Column index " + std::to_string(colIndices[i]) + 
                                  " is out of range [0, " + std::to_string(numCols-1) + "]");
        }
    }
}

template <typename T>
HBDIA<T>::HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values, 
                int numRows, int numCols, const std::vector<int>& globalRowMapping) 
    : values(values), rowIndices(rowIndices), colIndices(colIndices), numRows(numRows), numCols(numCols), 
      globalRowMapping(globalRowMapping), partialMatrix(false), hasCOO(true), hasDIA(false), hasHBDIA(false) {
    
    // Validate that all vectors have the same size
    if (rowIndices.size() != colIndices.size() || rowIndices.size() != values.size()) {
        throw std::invalid_argument("Row indices, column indices, and values vectors must have the same size");
    }
    
    // Validate dimensions
    if (numRows <= 0 || numCols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    // Validate global row mapping if provided
    if (!globalRowMapping.empty() && static_cast<int>(globalRowMapping.size()) != numRows) {
        throw std::invalid_argument("Global row mapping size must match number of rows");
    }
    
    numNonZeros = static_cast<int>(values.size());
    
    // Validate that all indices are within bounds
    for (size_t i = 0; i < rowIndices.size(); ++i) {
        if (rowIndices[i] < 0 || rowIndices[i] >= numRows) {
            throw std::out_of_range("Row index " + std::to_string(rowIndices[i]) + 
                                  " is out of range [0, " + std::to_string(numRows-1) + "]");
        }
        if (colIndices[i] < 0 || colIndices[i] >= numCols) {
            throw std::out_of_range("Column index " + std::to_string(colIndices[i]) + 
                                  " is out of range [0, " + std::to_string(numCols-1) + "]");
        }
    }
}

template <typename T>
HBDIA<T>::HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values, 
                int numRows, int numCols, const std::vector<int>& globalRowMapping, bool partialMatrix, int numGlobalRows, int numGlobalCols, int numGlobalNonZeros,
                int rank, int size) 
    : values(values), rowIndices(rowIndices), colIndices(colIndices), numRows(numRows), numCols(numCols), 
      globalRowMapping(globalRowMapping), partialMatrix(partialMatrix), numGlobalRows(numGlobalRows), numGlobalCols(numGlobalCols), numGlobalNonZeros(numGlobalNonZeros), 
      rank_(rank), size_(size), hasCOO(true), hasDIA(false), hasHBDIA(false) {
    
    // Validate that all vectors have the same size
    if (rowIndices.size() != colIndices.size() || rowIndices.size() != values.size()) {
        throw std::invalid_argument("Row indices, column indices, and values vectors must have the same size");
    }
    
    // Validate dimensions
    if (numRows <= 0 || numCols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    // Validate global row mapping if provided
    if (!globalRowMapping.empty() && static_cast<int>(globalRowMapping.size()) != numRows) {
        throw std::invalid_argument("Global row mapping size must match number of rows");
    }
    
    numNonZeros = static_cast<int>(values.size());
    
    // Validate that all indices are within bounds
    for (size_t i = 0; i < rowIndices.size(); ++i) {
        if (rowIndices[i] < 0 || rowIndices[i] >= numRows) {
            throw std::out_of_range("Row index " + std::to_string(rowIndices[i]) + 
                                  " is out of range [0, " + std::to_string(numRows-1) + "]");
        }
        if (colIndices[i] < 0 || colIndices[i] >= numCols) {
            throw std::out_of_range("Column index " + std::to_string(colIndices[i]) + 
                                  " is out of range [0, " + std::to_string(numCols-1) + "]");
        }
    }
}

template <typename T>
HBDIA<T>::~HBDIA() {
    cleanupGPUData();
}

template <typename T>
bool HBDIA<T>::loadMTX(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    bool isSymmetric = false;
    
    // Parse header and check for symmetry
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) {
                isSymmetric = true;
                std::cout << "Detected symmetric matrix - will expand to full matrix" << std::endl;
            }
        } else {
            break; // First non-comment line contains dimensions
        }
    }
    
    // Parse header line (rows, cols, non-zeros)
    std::istringstream headerStream(line);
    headerStream >> numRows >> numCols >> numNonZeros;
    
    // Reserve space for vectors (more space if symmetric)
    int expectedEntries = isSymmetric ? numNonZeros * 2 : numNonZeros;
    values.reserve(expectedEntries);
    rowIndices.reserve(expectedEntries);
    colIndices.reserve(expectedEntries);
    
    // Read matrix entries
    int row, col;
    T value;
    int originalEntries = 0;
    
    while (std::getline(file, line) && originalEntries < numNonZeros) {
        std::istringstream entryStream(line);
        
        // For pattern matrices, there might be no value column
        if (entryStream >> row >> col) {
            if (!(entryStream >> value)) {
                value = T(1); // Default value for pattern matrices
            }
            
            // Convert to 0-based indexing (MTX format is 1-based)
            row -= 1;
            col -= 1;
            
            // Add the original entry
            rowIndices.push_back(row);
            colIndices.push_back(col);
            values.push_back(value);
            originalEntries++;
            
            // If symmetric and not on diagonal, add mirrored entry
            if (isSymmetric && row != col) {
                rowIndices.push_back(col);
                colIndices.push_back(row);
                values.push_back(value);
            }
        }
    }
    
    file.close();
    
    // Update numNonZeros to reflect actual entries stored
    numNonZeros = values.size();
    hasCOO = true;  // Set COO format flag
    
    std::cout << "Loaded " << originalEntries << " entries from file";
    if (isSymmetric) {
        std::cout << ", expanded to " << numNonZeros << " total entries";
    }
    std::cout << std::endl;
    
    return originalEntries > 0;
}

template <typename T>
bool HBDIA<T>::loadMTX(const std::string& filename, int numRows, int numCols) {
    // Validate dimensions first
    if (numRows <= 0 || numCols <= 0) {
        std::cerr << "Error: Matrix dimensions must be positive" << std::endl;
        return false;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    bool isSymmetric = false;
    
    // Parse header and check for symmetry
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) {
                isSymmetric = true;
                std::cout << "Detected symmetric matrix - will expand to full matrix" << std::endl;
            }
        } else {
            break; // First non-comment line contains dimensions
        }
    }
    
    // Parse header line (rows, cols, non-zeros) from file
    int fileRows, fileCols, fileNonZeros;
    std::istringstream headerStream(line);
    headerStream >> fileRows >> fileCols >> fileNonZeros;
    
    // Set the dimensions to the provided values (may be larger than file dimensions)
    this->numRows = numRows;
    this->numCols = numCols;
    
    std::cout << "File dimensions: " << fileRows << "x" << fileCols << ", target dimensions: " << numRows << "x" << numCols << std::endl;
    
    // Reserve space for vectors
    int expectedEntries = isSymmetric ? fileNonZeros * 2 : fileNonZeros;
    values.reserve(expectedEntries);
    rowIndices.reserve(expectedEntries);
    colIndices.reserve(expectedEntries);
    
    // Read matrix entries
    int row, col;
    T value;
    int originalEntries = 0;
    
    while (std::getline(file, line) && originalEntries < fileNonZeros) {
        std::istringstream entryStream(line);
        
        // For pattern matrices, there might be no value column
        if (entryStream >> row >> col) {
            if (!(entryStream >> value)) {
                value = T(1); // Default value for pattern matrices
            }
            
            // Convert to 0-based indexing (MTX format is 1-based)
            row -= 1;
            col -= 1;
            
            // Validate indices are within target dimensions
            if (row < 0 || row >= numRows) {
                throw std::out_of_range("Row index " + std::to_string(row) + 
                                      " from file is out of range [0, " + std::to_string(numRows-1) + "]");
            }
            if (col < 0 || col >= numCols) {
                throw std::out_of_range("Column index " + std::to_string(col) + 
                                      " from file is out of range [0, " + std::to_string(numCols-1) + "]");
            }
            
            // Add the original entry
            rowIndices.push_back(row);
            colIndices.push_back(col);
            values.push_back(value);
            originalEntries++;
            
            // If symmetric and not on diagonal, add mirrored entry
            if (isSymmetric && row != col) {
                rowIndices.push_back(col);
                colIndices.push_back(row);
                values.push_back(value);
            }
        }
    }
    
    file.close();
    
    // Update numNonZeros to reflect actual entries stored
    numNonZeros = values.size();
    hasCOO = true;  // Set COO format flag
    
    std::cout << "Loaded " << originalEntries << " entries from file";
    if (isSymmetric) {
        std::cout << ", expanded to " << numNonZeros << " total entries";
    }
    std::cout << " into " << numRows << "x" << numCols << " matrix" << std::endl;

    this->convertToDIAFormat();
    
    return originalEntries > 0;
}

template <typename T>
void HBDIA<T>::removeCOODuplicates() {
    if (!hasCOO) {
        std::cout << "No COO format available for duplicate removal" << std::endl;
        return;
    }
    
    if (values.empty()) {
        return; // Empty matrix
    }
    
    // Step 1: Create map to consolidate duplicates by summing them up
    std::map<std::pair<int, int>, T> uniqueEntries;
    int originalEntries = values.size();
    
    for (size_t i = 0; i < values.size(); ++i) {
        std::pair<int, int> position(rowIndices[i], colIndices[i]);
        uniqueEntries[position] += values[i];
    }
    
    // Step 2: Replace COO vectors with deduplicated data
    values.clear();
    rowIndices.clear();
    colIndices.clear();
    
    values.reserve(uniqueEntries.size());
    rowIndices.reserve(uniqueEntries.size());
    colIndices.reserve(uniqueEntries.size());
    
    for (const auto& entry : uniqueEntries) {
        rowIndices.push_back(entry.first.first);
        colIndices.push_back(entry.first.second);
        values.push_back(entry.second);
    }
    
    // Update numNonZeros to reflect deduplicated count
    numNonZeros = values.size();
    
    std::cout << "Consolidated " << originalEntries << " entries to " << uniqueEntries.size() 
              << " unique entries (removed " << (originalEntries - uniqueEntries.size()) << " duplicates)" << std::endl;
}

template <typename T>
void HBDIA<T>::convertToDIAFormat(bool COOisUnique) {
    if (hasDIA) {
        std::cout << "Matrix is already in DIA format" << std::endl;
        return;
    }
    
    if (!hasCOO) {
        std::cout << "No COO format available for conversion to DIA" << std::endl;
        return;
    }
    
    if (numNonZeros == 0) {
        hasDIA = true;
        return; // Empty matrix
    }
    
    std::cout << "Converting matrix from coordinate format to DIA format..." << std::endl;
    
    // Step 1: Remove duplicates from COO format if not already unique
    if (!COOisUnique) {
        removeCOODuplicates();
    }
    
    // Step 2: Find all unique diagonal offsets using global coordinates when available
    std::set<int> uniqueOffsets;
    for (size_t i = 0; i < values.size(); ++i) {
        int localRow = rowIndices[i];
        int col = colIndices[i];
        
        // Use global row coordinates if this is a partial matrix with global row mapping
        int globalRow;
        if (partialMatrix && !globalRowMapping.empty()) {
            globalRow = globalRowMapping[localRow];
        } else {
            globalRow = localRow;
        }
        
        int offset = col - globalRow; // col - global_row
        uniqueOffsets.insert(offset);
    }
    
    // Step 3: Convert set to sorted vector of offsets
    offsets.assign(uniqueOffsets.begin(), uniqueOffsets.end());
    int numDiagonals = offsets.size();
    
    // Step 4: Initialize diagonals matrix
    diagonals.resize(numDiagonals);
    
    // For partial matrices, all diagonals should have the same length: min(numRows, numCols)
    int diagLength = std::min(numRows, numCols);
    
    for (int d = 0; d < numDiagonals; ++d) {
        // Initialize diagonal with zeros
        diagonals[d].assign(diagLength, T(0));
    }
    
    // Step 5: Fill in the diagonal values from COO data
    for (size_t i = 0; i < values.size(); ++i) {
        int localRow = rowIndices[i];
        int col = colIndices[i];
        
        // Use global row coordinates if this is a partial matrix with global row mapping
        int globalRow;
        if (partialMatrix && !globalRowMapping.empty()) {
            globalRow = globalRowMapping[localRow];
        } else {
            globalRow = localRow;
        }
        
        int offset = col - globalRow; // col - global_row
        T value = values[i];
        
        // Find which diagonal this belongs to
        auto it = std::lower_bound(offsets.begin(), offsets.end(), offset);
        int diagIndex = std::distance(offsets.begin(), it);
        
        // Calculate position in the diagonal - for partial matrices, always use local row
        int diagPos = localRow;
        
        // Store the value
        if (diagPos >= 0 && diagPos < static_cast<int>(diagonals[diagIndex].size())) {
            diagonals[diagIndex][diagPos] = value;
        }
    }
    
    // Step 6: Set flag to indicate DIA format is available
    hasDIA = true;

    // load to GPU
    prepareForGPU();
}

template <typename T>
void HBDIA<T>::convertToHBDIAFormat(int blockWidth, int threshold, bool COOisUnique) {
    if (hasHBDIA) {
        std::cout << "Matrix is already in HBDIA format" << std::endl;
        return;
    }
    
    if (!hasCOO) {
        std::cout << "No COO format available for conversion to HBDIA" << std::endl;
        return;
    }
    
    if (values.empty()) {
        std::cout << "Cannot convert empty matrix to HBDIA format" << std::endl;
        return;
    }
    
    this->blockWidth = blockWidth;
    this->threshold = threshold;
    
    // Step 1: Remove duplicates from COO format if not already unique
    if (!COOisUnique) {
        removeCOODuplicates();
    }
    
    // Step 2: Calculate maximum number of blocks (use minimum of rows and columns for partial matrices)
    int maxNumBlocks = (std::min(numRows, numCols) + blockWidth - 1) / blockWidth;
    
    // Step 3: Count offsets per block using COO data
    std::vector<std::map<int, int>> offsetCountPerBlock(maxNumBlocks);
    
    for (size_t i = 0; i < values.size(); ++i) {
        int localRow = rowIndices[i];
        int c = colIndices[i];
        
        // Use global row coordinates for offset calculation if this is a partial matrix
        int globalRow;
        if (partialMatrix && !globalRowMapping.empty()) {
            globalRow = globalRowMapping[localRow];
        } else {
            globalRow = localRow;
        }
        
        int block = localRow / blockWidth;  // Block is determined by local row for HBDIA storage
        int offset = c - globalRow;  // col - global_row: correct offset calculation for diagonal consistency
        
        if (block < maxNumBlocks) {
            offsetCountPerBlock[block][offset]++;
        }
    }
    
    // Step 4: Calculate storage requirements and filter offsets
    int storageRequired = 0;
    std::vector<std::vector<int>> validOffsetsPerBlock(maxNumBlocks);

    int cpu_entries = 0;
    
    for (int b = 0; b < maxNumBlocks; ++b) {
        for (const auto& pair : offsetCountPerBlock[b]) {
            if (pair.second >= threshold || cpu_entries >= MAX_CPU_ENTRIES) {
                validOffsetsPerBlock[b].push_back(pair.first);
                storageRequired += blockWidth;
            }else{
                cpu_entries += pair.second; // Count entries that will go to CPU fallback
            }
        }
        // Sort offsets for consistent ordering
        std::sort(validOffsetsPerBlock[b].begin(), validOffsetsPerBlock[b].end());
    }
    
    
    // Step 5: Allocate contiguous memory and set up pointers
    hbdiaData.resize(storageRequired);
    ptrToBlock.resize(maxNumBlocks, nullptr);
    offsetsPerBlock = std::move(validOffsetsPerBlock);
    
    T* dataPtr = hbdiaData.data();
    
    for (int b = 0; b < maxNumBlocks; ++b) {
        if (!offsetsPerBlock[b].empty()) {
            ptrToBlock[b] = dataPtr;
            dataPtr += offsetsPerBlock[b].size() * blockWidth;
        }
    }
    
    // Step 6: Initialize data to zero
    std::fill(hbdiaData.begin(), hbdiaData.end(), T(0));
    
    // Step 7: Fill data and separate CPU fallback entries
    cpuRowIndices.clear();
    cpuColIndices.clear();
    cpuValues.clear();
    
    for (size_t i = 0; i < values.size(); ++i) {
        int localRow = rowIndices[i];
        int c = colIndices[i];
        T v = values[i];
        
        // Use global row coordinates for offset calculation if this is a partial matrix
        int globalRow;
        if (partialMatrix && !globalRowMapping.empty()) {
            globalRow = globalRowMapping[localRow];
        } else {
            globalRow = localRow;
        }
        
        // Block and lane calculations use LOCAL coordinates for HBDIA storage
        int block = localRow / blockWidth;  // Block is determined by local row
        int offset = c - globalRow;  // col - global_row: correct offset calculation for diagonal consistency
        int lane = localRow % blockWidth;   // Lane is the position within the block (local row position in block)
        
        if (block < maxNumBlocks && !offsetsPerBlock[block].empty()) {
            // Find offset index in this block
            auto it = std::find(offsetsPerBlock[block].begin(), offsetsPerBlock[block].end(), offset);
            
            if (it != offsetsPerBlock[block].end()) {
                int offsetIndex = std::distance(offsetsPerBlock[block].begin(), it);
                int dataIndex = offsetIndex * blockWidth + lane;
                ptrToBlock[block][dataIndex] = v;
            } else {
                // Offset not stored in GPU format, use CPU fallback
                cpuRowIndices.push_back(localRow);
                cpuColIndices.push_back(c);
                cpuValues.push_back(v);
            }
        } else {
            // Block doesn't exist or is empty, use CPU fallback
            cpuRowIndices.push_back(localRow);
            cpuColIndices.push_back(c);
            cpuValues.push_back(v);
        }
    }
    
    std::cout << "HBDIA conversion complete:" << std::endl;
    std::cout << "  GPU storage: " << (values.size() - cpuValues.size()) << " entries" << std::endl;
    std::cout << "  CPU fallback: " << cpuValues.size() << " entries" << std::endl;
    std::cout << "  Active blocks: " << std::count_if(ptrToBlock.begin(), ptrToBlock.end(), 
                                                      [](T* ptr) { return ptr != nullptr; }) << "/" << maxNumBlocks << std::endl;
    
    // Set flag to indicate HBDIA format is available
    hasHBDIA = true;
}

template <typename T>
void HBDIA<T>::prepareForGPU() {
    // Only prepare for HBDIA format matrices
    if (!hasHBDIA) {
        return;
    }
    
    // Clean up any existing data first
    cleanupGPUData();
    
    // Create flattened arrays and copy to managed memory
    std::vector<int> flattenedOffsets;
    std::vector<int> blockStartIndices;
    std::vector<int> blockSizes;
    std::vector<int> flattenedVectorOffsets;
    
    // Flatten data structures
    for (size_t blockId = 0; blockId < offsetsPerBlock.size(); blockId++) {
        const auto& matrixBlockOffsets = offsetsPerBlock[blockId];
        
        blockStartIndices.push_back(flattenedOffsets.size());
        blockSizes.push_back(matrixBlockOffsets.size());
        
        for (int matrixOffset : matrixBlockOffsets) {
            flattenedOffsets.push_back(matrixOffset);
        }
        
        // Add vector offsets if this is a partial matrix and vector offsets are available
        if (partialMatrix && blockId < vectorOffsets.size()) {
            const auto& vectorBlockOffsets = vectorOffsets[blockId];
            for (int vectorOffset : vectorBlockOffsets) {
                flattenedVectorOffsets.push_back(vectorOffset);
            }
        } else if (partialMatrix) {
            // Fill with -1 for partial matrices without calculated offsets
            for (size_t i = 0; i < matrixBlockOffsets.size(); i++) {
                flattenedVectorOffsets.push_back(INT_MIN);
            }
        }
        // For non-partial matrices, don't add vector offsets at all
    }
    
    // Allocate GPU device memory and copy data
    if (!hbdiaData.empty()) {
        CHECK_CUDA(cudaMalloc(&hbdiaData_d_, hbdiaData.size() * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(hbdiaData_d_, hbdiaData.data(), hbdiaData.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    if (!flattenedOffsets.empty()) {
        CHECK_CUDA(cudaMalloc(&flattenedOffsets_d_, flattenedOffsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(flattenedOffsets_d_, flattenedOffsets.data(), flattenedOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    if (!blockStartIndices.empty()) {
        CHECK_CUDA(cudaMalloc(&blockStartIndices_d_, blockStartIndices.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(blockStartIndices_d_, blockStartIndices.data(), blockStartIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    if (!blockSizes.empty()) {
        CHECK_CUDA(cudaMalloc(&blockSizes_d_, blockSizes.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(blockSizes_d_, blockSizes.data(), blockSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    if (!flattenedVectorOffsets.empty() && partialMatrix) {
        CHECK_CUDA(cudaMalloc(&flattenedVectorOffsets_d_, flattenedVectorOffsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(flattenedVectorOffsets_d_, flattenedVectorOffsets.data(), flattenedVectorOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
}

template <typename T>
void HBDIA<T>::cleanupGPUData() {
    if (hbdiaData_d_) {
        cudaFree(hbdiaData_d_);
        hbdiaData_d_ = nullptr;
    }
    if (flattenedOffsets_d_) {
        cudaFree(flattenedOffsets_d_);
        flattenedOffsets_d_ = nullptr;
    }
    if (blockStartIndices_d_) {
        cudaFree(blockStartIndices_d_);
        blockStartIndices_d_ = nullptr;
    }
    if (blockSizes_d_) {
        cudaFree(blockSizes_d_);
        blockSizes_d_ = nullptr;
    }
    if (flattenedVectorOffsets_d_) {
        cudaFree(flattenedVectorOffsets_d_);
        flattenedVectorOffsets_d_ = nullptr;
    }
}

template <typename T>
void HBDIA<T>::calculateVectorOffsets(int rank, int size) {
    int leftBufferSize = -1;  // Default to -1 to calculate from processDataRanges
    int rightBufferSize = -1; // Default to -1 to calculate from processData
    int localVectorSize = numRows; // Local vector size is the number of rows in the local partition

    if(!partialMatrix) {
        std::cerr << "Error: Cannot calculate vector offsets - this is not a partial matrix" << std::endl;
        return;
    }
    if (!hasHBDIA) {
        std::cerr << "Error: Cannot calculate vector offsets - HBDIA format not available" << std::endl;
        return;
    }
    if (!hasPartialMatrixMetadata()) {
        std::cerr << "Error: Cannot calculate vector offsets - no partial matrix metadata available" << std::endl;
        return;
    }
    
    // Clear any existing offsets
    vectorOffsets.clear();
    
    // Get HBDIA data
    const auto& offsetsPerBlockRef = getOffsetsPerBlock();
    const auto& metadata = getPartialMatrixMetadata();
    int numBlocks = offsetsPerBlockRef.size();
    
    // Calculate buffer sizes from processDataRanges if not provided
    if (leftBufferSize == -1 || rightBufferSize == -1) {
        leftBufferSize = 0;
        rightBufferSize = 0;
        
        // Calculate buffer sizes by counting elements in processDataRanges
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
    }
    
    // Initialize vectorOffsets with same structure as offsetsPerBlock
    vectorOffsets.resize(numBlocks);
    
    // Memory layout: [recv_left | local_data | recv_right]
    // We need to calculate the offset from unified_data_ptr for each data access
    
    // Calculate global data ranges (similar to setupBlockRowPointersRowWise logic)
    int rowsPerProcess = numGlobalRows / size;
    int globalStart = rank * rowsPerProcess;
    int globalEnd = globalStart + rowsPerProcess + ((rank == size - 1) ? numGlobalRows % rowsPerProcess : 0);
    
    // For each block
    for (int blockId = 0; blockId < numBlocks; blockId++) {
        const auto& blockOffsets = offsetsPerBlockRef[blockId];
        vectorOffsets[blockId].resize(blockOffsets.size());
        
        int global_block_start = rank * rowsPerProcess + blockId * blockWidth;
        
        // Go through each offset in this block
        for (size_t i = 0; i < blockOffsets.size(); i++) {
            int offset = blockOffsets[i];
            int global_data_index = global_block_start + offset;
            
            // Find the offset for this global index in unified memory
            int memoryOffset = findMemoryOffsetForGlobalIndex(global_data_index, 
                                                              leftBufferSize, 
                                                              localVectorSize, 
                                                              globalStart, 
                                                              globalEnd, 
                                                              rank);
            
            vectorOffsets[blockId][i] = memoryOffset;
        }
    }
}

template <typename T>
int HBDIA<T>::findMemoryOffsetForGlobalIndex(int globalIndex, int leftBufferSize, 
                                           int localVectorSize, int globalStart, 
                                           int globalEnd, int rank) const {
    // Check if it's in local data range
    if (globalIndex >= globalStart && globalIndex < globalEnd) {
        int localIndex = globalIndex - globalStart;
        return leftBufferSize + localIndex;  // offset into unified memory
    }
    if(globalIndex < 0){
        return globalIndex;
    }
    
    // For partial matrices, search in receive buffer ranges using matrix metadata
    if (!hasPartialMatrixMetadata()) {
        std::cerr << "Error: No partial matrix metadata available for global index search" << std::endl;
        return -1;  // Not found
    }
    
    const auto& metadata = getPartialMatrixMetadata();
    
    // Search in receive buffer ranges
    int offset = 0;
    
    if (globalIndex < globalStart) {
        // Search in left buffer (lower process IDs)
        for (int procId = 0; procId < rank; procId++) {
            for (const auto& range : metadata.processDataRanges[procId]) {
                int rangeStart = std::get<0>(range);
                int rangeEnd = std::get<1>(range);
                
                if (globalIndex >= rangeStart && globalIndex < rangeEnd) {
                    int offsetInRange = globalIndex - rangeStart;
                    return offset + offsetInRange;
                }
                
                offset += rangeEnd - rangeStart;
            }
        }
    } else {
        // Search in right buffer (higher process IDs)
        offset = leftBufferSize + localVectorSize;
        
        for (int procId = rank + 1; procId < metadata.processDataRanges.size(); procId++) {
            for (const auto& range : metadata.processDataRanges[procId]) {
                int rangeStart = std::get<0>(range);
                int rangeEnd = std::get<1>(range);
                
                if (globalIndex >= rangeStart && globalIndex < rangeEnd) {
                    int offsetInRange = globalIndex - rangeStart;
                    return offset + offsetInRange;
                }
                
                offset += rangeEnd - rangeStart;
            }
        }
    }
    
    std::cerr << rank << "\t Error: Global index " << globalIndex << " not found in any buffer ranges" << std::endl;
    return INT_MIN;  // Not found
}

template <typename T>
bool HBDIA<T>::isDIAFormat() const {
    return hasDIA;
}

template <typename T>
bool HBDIA<T>::isCOOFormat() const {
    return hasCOO;
}

template <typename T>
bool HBDIA<T>::isHBDIAFormat() const {
    return hasHBDIA;
}

template <typename T>
void HBDIA<T>::print() const {
    HBDIAPrinter<T>::print(*this);
}

template <typename T>
void HBDIA<T>::printCOO() const {
    HBDIAPrinter<T>::printCOO(*this);
}

template <typename T>
void HBDIA<T>::printDIA(int block_width) const {
    HBDIAPrinter<T>::printDIA(*this, block_width);
}

template <typename T>
void HBDIA<T>::printHBDIA() const {
    HBDIAPrinter<T>::printHBDIA(*this);
}

template <typename T>
void HBDIA<T>::printDense() const {
    HBDIAPrinter<T>::printDense(*this);
}

template <typename T>
void HBDIA<T>::printDataRanges() const {
    HBDIAPrinter<T>::printDataRanges(*this);
}

template <typename T>
void HBDIA<T>::deleteMatrix() {
    // Clear all format data
    values.clear();
    rowIndices.clear();
    colIndices.clear();
    
    diagonals.clear();
    offsets.clear();
    
    hbdiaData.clear();
    ptrToBlock.clear();
    offsetsPerBlock.clear();
    cpuRowIndices.clear();
    cpuColIndices.clear();
    cpuValues.clear();
    
    // Reset metadata
    numRows = 0;
    numCols = 0;
    numNonZeros = 0;
    blockWidth = 0;
    threshold = 0;
    
    // Reset all flags
    hasCOO = false;
    hasDIA = false;
    hasHBDIA = false;
    
    std::cout << "All matrix data deleted. Object reset to empty state." << std::endl;
}

template <typename T>
void HBDIA<T>::deleteCOOFormat() {
    if (!hasCOO) {
        std::cout << "COO format is not available" << std::endl;
        return;
    }
    
    // Check if this is the last format
    int availableFormats = (hasCOO ? 1 : 0) + (hasDIA ? 1 : 0) + (hasHBDIA ? 1 : 0);
    if (availableFormats <= 1) {
        std::cout << "Cannot delete COO format: it's the last remaining format. Call deleteMatrix() instead." << std::endl;
        return;
    }
    
    // Clear COO data
    values.clear();
    rowIndices.clear();
    colIndices.clear();
    hasCOO = false;
    
    std::cout << "COO format deleted. Memory freed." << std::endl;
}

template <typename T>
void HBDIA<T>::deleteDIAFormat() {
    if (!hasDIA) {
        std::cout << "DIA format is not available" << std::endl;
        return;
    }
    
    // Check if this is the last format
    int availableFormats = (hasCOO ? 1 : 0) + (hasDIA ? 1 : 0) + (hasHBDIA ? 1 : 0);
    if (availableFormats <= 1) {
        std::cout << "Cannot delete DIA format: it's the last remaining format. Call deleteMatrix() instead." << std::endl;
        return;
    }
    
    // Clear DIA data
    diagonals.clear();
    offsets.clear();
    hasDIA = false;
    
    std::cout << "DIA format deleted. Memory freed." << std::endl;
}

template <typename T>
void HBDIA<T>::deleteHBDIAFormat() {
    if (!hasHBDIA) {
        std::cout << "HBDIA format is not available" << std::endl;
        return;
    }
    
    // Check if this is the last format
    int availableFormats = (hasCOO ? 1 : 0) + (hasDIA ? 1 : 0) + (hasHBDIA ? 1 : 0);
    if (availableFormats <= 1) {
        std::cout << "Cannot delete HBDIA format: it's the last remaining format. Call deleteMatrix() instead." << std::endl;
        return;
    }
    
    // Clear HBDIA data
    hbdiaData.clear();
    ptrToBlock.clear();
    offsetsPerBlock.clear();
    cpuRowIndices.clear();
    cpuColIndices.clear();
    cpuValues.clear();
    blockWidth = 0;
    threshold = 0;
    hasHBDIA = false;
    
    std::cout << "HBDIA format deleted. Memory freed." << std::endl;
}

template <typename T>
void HBDIA<T>::analyzeDataRanges() {
    if (!partialMatrix) {
        std::cout << "Cannot analyze data ranges: not a partial matrix" << std::endl;
        return;
    }

    if(!hasHBDIA) {
        std::cerr << "Cannot analyze data ranges: HBDIA format not available" << std::endl;
        return;
    }
    
    // Use unordered_set for O(1) average insertion and fast duplicate removal
    std::unordered_set<int> uniqueColsSet;
    
    // Naive size estimation: start with COO indices + HBDIA block estimates
    size_t estimatedSize = colIndices.size();
    if (hasHBDIA) {
        // Add estimated size for HBDIA blocks: blocks * offsets * blockWidth
        for (const auto& blockOffsets : offsetsPerBlock) {
            estimatedSize += blockOffsets.size() * blockWidth;
        }
    }
    uniqueColsSet.reserve(estimatedSize);
    
    // Part 1: Add all column indices from COO format (original matrix data)
    for (int col : colIndices) {
        uniqueColsSet.insert(col);
    }
    
    std::vector<int> globalMapping = getGlobalRowMapping();

    // Part 2: Add all column indices that HBDIA blocks will access
    if (hasHBDIA && !offsetsPerBlock.empty()) {
        for (size_t blockId = 0; blockId < offsetsPerBlock.size(); blockId++) {
            const auto& blockOffsets = offsetsPerBlock[blockId];
            
            // For each offset in this block
            for (int offset : blockOffsets) {
                // For each row (lane) in this block, calculate the column it will access
                int blockStart = blockId * blockWidth;
                
                for (int lane = 0; lane < blockWidth; lane++) {
                    int col = globalMapping[blockStart + lane] + offset; // Calculate global column index
                    
                    // Add valid column indices (must be within global matrix bounds)
                    if (col >= 0 && col < numGlobalCols) {
                        uniqueColsSet.insert(col);
                    }
                }
            }
        }
    }
    
    dataRanges.clear();
    
    if (uniqueColsSet.empty()) {
        return;
    }
    
    // Convert to sorted vector for range merging
    std::vector<int> sortedCols(uniqueColsSet.begin(), uniqueColsSet.end());
    std::sort(sortedCols.begin(), sortedCols.end());
    
    // Merge consecutive indices into ranges
    int rangeStart = sortedCols[0];
    int rangeEnd = sortedCols[0] + 1;
    
    for (size_t i = 1; i < sortedCols.size(); ++i) {
        if (sortedCols[i] == rangeEnd) {
            // Consecutive index, extend current range
            rangeEnd = sortedCols[i] + 1;
        } else {
            // Gap found, close current range and start new one
            dataRanges.emplace_back(rangeStart, rangeEnd);
            rangeStart = sortedCols[i];
            rangeEnd = sortedCols[i] + 1;
        }
    }
    
    // Add the last range
    dataRanges.emplace_back(rangeStart, rangeEnd);
}

// Explicit template instantiations for common types
template class HBDIA<double>;
template class HBDIA<float>;
template class HBDIA<int>;
