#ifndef HBDIA_HPP
#define HBDIA_HPP

#include <string>
#include <vector>
#include <tuple>
#include "types.hpp"

template <typename T>
class HBDIAPrinter;

template <typename T>
class HBDIAVector;

template <typename T>
struct PartialMatrixMetadata {
    std::vector<std::tuple<int, int>> dataRanges;
    int totalBufferSize = 0;
    std::vector<std::vector<std::tuple<int, int>>> processDataRanges;
};

template <typename T>
class HBDIA {
    // Friend classes for access to private members
    friend class HBDIAPrinter<T>;
    friend class HBDIAVector<T>;
    
    public:
        HBDIA();
        HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values);
        HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values, 
              int numRows, int numCols);
        HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values, 
              int numRows, int numCols, const std::vector<int>& globalRowMapping);
        HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values, 
              int numRows, int numCols, const std::vector<int>& globalRowMapping, bool partialMatrix, int numGlobalRows, int numGlobalCols, int numGlobalNonZeros,
              int rank = -1, int size = -1);
        ~HBDIA();
        
        bool loadMTX(const std::string& filename);
        bool loadMTX(const std::string& filename, int numRows, int numCols);
        
        void convertToDIAFormat(bool COOisUnique = false);
        bool isDIAFormat() const;
        bool isCOOFormat() const;
        bool isHBDIAFormat() const;
        void convertToHBDIAFormat(int blockWidth = DEFAULT_BLOCK_WIDTH, int threshold = DEFAULT_THRESHOLD, bool COOisUnique = false);
        
        // Convenience wrapper methods that delegate to HBDIAPrinter
        void print() const;
        void printCOO() const;
        void printDIA(int block_width = 0) const;
        void printHBDIA() const;
        void printDense() const;
        void printDataRanges() const;
        
        // Static factory method for creating 3D stencil matrices
        // Creates a 27-point stencil matrix for a 3D grid of size nx x ny x nz
        // Each grid point connects to all 26 neighbors + itself in a 3x3x3 cube
        // Uses realistic stencil weights: center=26, faces=-1, edges=-0.1, corners=-0.01
        void create3DStencil27Point(int nx, int ny, int nz);
        
        // Format deletion methods
        void deleteCOOFormat();
        void deleteDIAFormat();
        void deleteHBDIAFormat();
        void deleteMatrix();
        
        // Helper method to remove duplicates from COO format
        void removeCOODuplicates();
        
        // Getters for matrix dimensions and data
        int getNumRows() const { return numRows; }
        int getNumCols() const { return numCols; }
        int getNumNonZeros() const { return numNonZeros; }
        int getNumGlobalRows() const { return partialMatrix ? numGlobalRows : numRows; }
        int getNumGlobalCols() const { return partialMatrix ? numGlobalCols : numCols; }
        int getNumGlobalNonZeros() const { return partialMatrix ? numGlobalNonZeros : numNonZeros; }
        
        
        // Getters for COO data
        const std::vector<T>& getValues() const { return values; }
        const std::vector<int>& getRowIndices() const { return rowIndices; }
        const std::vector<int>& getColIndices() const { return colIndices; }
        
        // Getters for DIA data
        const std::vector<std::vector<T>>& getDiagonals() const { return diagonals; }
        const std::vector<int>& getOffsets() const { return offsets; }
        
        // Getters for HBDIA data
        const std::vector<T>& getHBDIAData() const { return hbdiaData; }
        const std::vector<T*>& getPtrToBlock() const { return ptrToBlock; }
        const std::vector<std::vector<int>>& getOffsetsPerBlock() const { return offsetsPerBlock; }
        const std::vector<int>& getCpuRowIndices() const { return cpuRowIndices; }
        const std::vector<int>& getCpuColIndices() const { return cpuColIndices; }
        const std::vector<T>& getCpuValues() const { return cpuValues; }
        int getBlockWidth() const { return blockWidth; }
        int getThreshold() const { return threshold; }
        int getNumBlocks() const { return static_cast<int>(offsetsPerBlock.size()); }
        
        // Getters for GPU-friendly flattened data structures (removed - use managed memory getters instead)

        // Non-const getters for modification (use carefully)
        std::vector<T>& getValuesRef() { return values; }
        std::vector<int>& getRowIndicesRef() { return rowIndices; }
        std::vector<int>& getColIndicesRef() { return colIndices; }
        
        // Global row mapping getters
        const std::vector<int>& getGlobalRowMapping() const { return globalRowMapping; }
        std::vector<int>& getGlobalRowMappingRef() { return globalRowMapping; }
        bool hasGlobalRowMapping() const { return !globalRowMapping.empty(); }
        int getGlobalRowIndex(int localRowIndex) const { 
            return (hasGlobalRowMapping() && localRowIndex < globalRowMapping.size()) ? 
                   globalRowMapping[localRowIndex] : localRowIndex; 
        }
        
        // Partial matrix functionality
        bool isPartialMatrix() const { return partialMatrix; }
        const std::vector<std::tuple<int, int>>& getDataRanges() const { return dataRanges; }
        void analyzeDataRanges();
        int getRank() const { return rank_; }
        int getSize() const { return size_; }
        
        // Partial matrix metadata methods
        const PartialMatrixMetadata<T>& getPartialMatrixMetadata() const { return partialMatrixMetadata_; }
        PartialMatrixMetadata<T>& getPartialMatrixMetadataRef() { return partialMatrixMetadata_; }
        void setPartialMatrixMetadata(const PartialMatrixMetadata<T>& metadata) { partialMatrixMetadata_ = metadata; }
        bool hasPartialMatrixMetadata() const { return partialMatrix && partialMatrixMetadata_.totalBufferSize > 0; }
        
        // Format flag getters
        bool getHasCOO() const { return hasCOO; }
        bool getHasDIA() const { return hasDIA; }
        bool getHasHBDIA() const { return hasHBDIA; }
        
        // Format flag setters
        void setHasCOO(bool flag) { hasCOO = flag; }
        void setHasDIA(bool flag) { hasDIA = flag; }
        void setHasHBDIA(bool flag) { hasHBDIA = flag; }
        
        // Declare HBDIAPrinter as friend to access private members
        friend class HBDIAPrinter<T>;
        
        // Declare HBDIAVector as friend to access private methods
        friend class HBDIAVector<T>;
        
        // GPU device memory getters 
        const T* getHBDIADataDevice() const { return hbdiaData_d_; }
        const int* getFlattenedOffsetsDevice() const { return flattenedOffsets_d_; }
        const int* getBlockStartIndicesDevice() const { return blockStartIndices_d_; }
        const int* getBlockSizesDevice() const { return blockSizes_d_; }
        const int* getFlattenedVectorOffsetsDevice() const { return flattenedVectorOffsets_d_; }
        
        void prepareForGPU(); // Convert to GPU device memory after convertToHBDIAFormat()
        void cleanupGPUData(); // Clean up GPU device memory
        
        void calculateVectorOffsets(int rank, int size);
        
        // Public method for finding memory offset for global index (needed for CPU COO partial SpMV)
        int findMemoryOffsetForGlobalIndex(int globalIndex, int leftBufferSize, 
                                         int localVectorSize, int globalStart, 
                                         int globalEnd, int rank) const;
        
    private:
        // Coordinate format storage
        std::vector<T> values;
        std::vector<int> rowIndices;
        std::vector<int> colIndices;
        
        // DIA format storage
        std::vector<std::vector<T>> diagonals;
        std::vector<int> offsets;
        
        // HBDIA format storage
        std::vector<T> hbdiaData;                    // Contiguous memory for all blocks
        std::vector<T*> ptrToBlock;                  // Pointers to each block's data
        std::vector<std::vector<int>> offsetsPerBlock; // Offsets for each block
        std::vector<int> cpuRowIndices;              // CPU fallback rows
        std::vector<int> cpuColIndices;              // CPU fallback cols
        std::vector<T> cpuValues;                    // CPU fallback values
        int blockWidth;                              // Block width for HBDIA
        int threshold;                               // Threshold for CPU fallback
        
        // Matrix metadata
        int numRows;
        int numCols;
        int numNonZeros;
        
        // Global row mapping for distributed matrices
        std::vector<int> globalRowMapping; // Maps local row indices to global row indices
        
        // Partial matrix functionality
        bool partialMatrix;                             // Flag indicating this is a partial matrix
        int numGlobalRows;
        int numGlobalCols;
        int numGlobalNonZeros;                         // Global non-zero count for partial matrices
        std::vector<std::tuple<int, int>> dataRanges;  // Data ranges (x,y) this process needs
        PartialMatrixMetadata<T> partialMatrixMetadata_; // Metadata for partial matrix communication
        
        // Distributed computing context (only for partial matrices)
        int rank_;    // MPI rank (-1 if not set)
        int size_;    // MPI size (-1 if not set)

        // Format flags
        bool hasCOO = true;       // Flag indicating COO format is available
        bool hasDIA = false;      // Flag indicating DIA format is available
        bool hasHBDIA = false;    // Flag indicating HBDIA format is available

        // Vector memory layout offsets for distributed SpMV (only for partial matrices)
        std::vector<std::vector<int>> vectorOffsets; // vectorOffsets[block][row] = offset in unified memory

        // GPU device memory pointers for flattened data structures
        T* hbdiaData_d_ = nullptr;              // values of the matrix stored row-wise per block
        int* flattenedOffsets_d_ = nullptr;     // actual offsets per block
        int* blockStartIndices_d_ = nullptr;    // offset per block where this blocks offsets are stored in flattenedOffsets. so if blockStartIndices[blockId] = 5, then flattenedOffsets[5:5+blockSizes[blockId]] contains the offsets for this block
        int* blockSizes_d_ = nullptr;           // number of offsets per block
        int* flattenedVectorOffsets_d_ = nullptr; // this gives the offset per block where the values for a row are in the unified memory layout. so if flattenedVectorOffsets[blockId][row] = 10, then the values for this row start at unified_data_ptr[10] in the unified memory layout
};

#endif // HBDIA_HPP
