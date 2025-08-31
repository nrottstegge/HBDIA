#ifndef HBDIA_HPP
#define HBDIA_HPP

#include <string>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include <cusparse.h>
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
    //friend classes for access to private members
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
        
        //format loading
        bool loadMTX(const std::string& filename);
        bool loadMTX(const std::string& filename, int numRows, int numCols);
        void create3DStencil27Point(int nx, int ny, int nz, double noise = 0.0, int iteration = 0);
        
        //format conversion
        void convertToDIAFormat(bool COOisUnique = false);
        bool isDIAFormat() const;
        bool isCOOFormat() const;
        bool isHBDIAFormat() const;
        void convertToHBDIAFormat(int blockWidth = DEFAULT_BLOCK_WIDTH, int threshold = DEFAULT_THRESHOLD, 
                                 int max_coo_entries = MAX_COO_ENTRIES, bool COOisUnique = false, 
                                 ExecutionMode execMode = ExecutionMode::GPU_COO);
        
        //print wrappers
        void print() const;
        void printCOO() const;
        void printDIA(int block_width = 0) const;
        void printHBDIA() const;
        void printDense() const;
        void printDataRanges() const;
        
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
        const std::vector<int>& getNnzPerRow() const { return nnzPerRow; }
        
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
        const std::vector<int>& getCpuRowPtr() const { return cpuRowPtr; }
        int getBlockWidth() const { return blockWidth; }
        int getThreshold() const { return threshold; }
        int getMaxCooEntries() const { return max_coo_entries; }
        int getNumBlocks() const { return static_cast<int>(offsetsPerBlock.size()); }
        ExecutionMode getExecutionMode() const { return executionMode; }
        
        // HBDIA statistics getters
        int getNumberDiagonals() const { return numberDiagonals; }
        const std::vector<int>& getHistogramBlocks() const { return histogramBlocks; }
        const std::vector<int>& getHistogramNnz() const { return histogramNnz; }
        
        // Non-const getters for modification
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
        
        // GPU device memory getters for COO fallback data
        const int* getCpuRowIndicesDevice() const { return cpuRowIndices_d_; }
        const int* getCpuColIndicesDevice() const { return cpuColIndices_d_; }
        const T* getCpuValuesDevice() const { return cpuValues_d_; }
        const int* getCpuRowPtrDevice() const { return cpuRowPtr_d_; }
        
        void prepareForGPU(); // Convert to GPU device memory after convertToHBDIAFormat()
        void cleanupGPUData(); // Clean up GPU device memory
        
        // CUDA stream and event management for SpMV operations
        void initializeStreams(); // Initialize CUDA streams and events
        void cleanupStreams(); // Cleanup CUDA streams and events
        bool areStreamsInitialized() const { return streamsInitialized_; }
        
        // cuSPARSE management for GPU COO execution
        void initializeCuSparse(); // Initialize cuSPARSE handle and descriptors
        void cleanupCuSparse(); // Cleanup cuSPARSE objects
        bool isCuSparseInitialized() const { return cusparseInitialized_; }
        
        // Stream getters for SpMV operations
        cudaStream_t getBDIAStream() const { return sBDIA_; }
        cudaStream_t getD2HStream() const { return sD2H_; }
        cudaStream_t getH2DStream() const { return sH2D_; }
        cudaStream_t getCOOStream() const { return sCOO_; }
        cudaStream_t getADDStream() const { return sADD_; }
        cudaEvent_t getBDIAEvent() const { return bdiaEvent_; }
        cudaEvent_t getCOOEvent() const { return cooEvent_; }
        
        // cuSPARSE getters for GPU COO execution
        cusparseHandle_t getCuSparseHandle() const { return cusparseHandle_; }
        cusparseSpMatDescr_t getCOOMatDescr() const { return cooMatDescr_; }
        cusparseDnVecDescr_t getCOOVecX() const { return cooVecX_; }
        cusparseDnVecDescr_t getCOOVecY() const { return cooVecY_; }
        void* getCOOBuffer() const { return cooBuffer_; }
        
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
        std::vector<int> cpuRowPtr;                  // CPU fallback row pointers (CSR format)
        int blockWidth;                              // Block width for HBDIA
        int threshold;                               // Threshold for CPU fallback
        int max_coo_entries;
        ExecutionMode executionMode;                 // Execution mode for COO fallback
        
        // HBDIA statistics
        int numberDiagonals;                         // Total number of unique diagonals
        std::vector<int> histogramBlocks;            // Histogram: blocks with x diagonals
        std::vector<int> histogramNnz;               // Histogram: nnz in blocks with x diagonals
        
        // Matrix metadata
        int numRows;
        int numCols;
        int numNonZeros;
        std::vector<int> nnzPerRow;                     // Non-zeros count per row
        
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
        
        // GPU device memory pointers for COO fallback data (for cuSPARSE execution)
        int* cpuRowIndices_d_ = nullptr;        // COO row indices on device
        int* cpuColIndices_d_ = nullptr;        // COO column indices on device  
        T* cpuValues_d_ = nullptr;              // COO values on device
        int* cpuRowPtr_d_ = nullptr;        // COO row indices on device
        
        // CUDA streams and events for SpMV operations
        cudaStream_t sBDIA_ = nullptr;          // BDIA kernel stream
        cudaStream_t sD2H_ = nullptr;           // Device to Host transfers
        cudaStream_t sH2D_ = nullptr;           // Host to Device transfers  
        cudaStream_t sCOO_ = nullptr;           // cuSPARSE COO execution
        cudaStream_t sADD_ = nullptr;           // Vector addition stream
        cudaEvent_t bdiaEvent_ = nullptr;       // Event for BDIA kernel completion
        cudaEvent_t cooEvent_ = nullptr;        // Event for COO work completion
        bool streamsInitialized_ = false;      // Flag to track if streams are initialized
        
        // cuSPARSE objects for GPU COO execution
        cusparseHandle_t cusparseHandle_ = nullptr;     // cuSPARSE handle
        cusparseSpMatDescr_t cooMatDescr_ = nullptr;    // COO matrix descriptor
        cusparseDnVecDescr_t cooVecX_ = nullptr;        // Input vector descriptor
        cusparseDnVecDescr_t cooVecY_ = nullptr;        // Output vector descriptor
        void* cooBuffer_ = nullptr;                     // cuSPARSE workspace buffer
        size_t cooBufferSize_ = 0;                      // Buffer size
        bool cusparseInitialized_ = false;             // Flag for cuSPARSE initialization
};

#endif // HBDIA_HPP
