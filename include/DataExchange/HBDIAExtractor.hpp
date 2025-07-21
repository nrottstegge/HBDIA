// HBDIAExtractor.hpp
#ifndef HBDIA_EXTRACTOR_HPP
#define HBDIA_EXTRACTOR_HPP

#include "../Format/HBDIA.hpp"
#include <vector>
#include <tuple>

struct MatrixData {
    int numGlobalRows;
    int numGlobalCols;
    int numGlobalNonZeros;
    int numLocalRows;
    int numLocalCols;
    int numLocalNonZeros;
};

template <typename T>
struct Partition {
    std::vector<int> localRowIndices; // Local row indices for this partition
    std::vector<int> localColIndices; // Local column indices for this partition
    std::vector<T> localValues;        // Local values for this partition
    std::vector<int> globalRowMapping; // Maps local row indices to global row indices
};

template <typename T>
struct VectorPartition {
    std::vector<T> localValues;        // Local vector elements for this partition
};

template <typename T>
class HBDIAExtractor {
    public:
        // Define partitioning strategies
        enum class PartitioningStrategy {
            ROW_WISE
        };

        HBDIAExtractor() = default;
        virtual ~HBDIAExtractor() = default;

        // Create partitions for distribution (pure virtual)
        virtual bool createMatrixPartitions(
            const HBDIA<T>& matrix, 
            int numProcesses
        ) = 0;
        
        // Create vector partitions for distribution (pure virtual)
        virtual bool createVectorPartitions(
            const std::vector<T>& globalVector,
            int numProcesses
        ) = 0;
        
        // Partial matrix extraction methods - now works directly with matrix metadata
        virtual void extractPartialMatrixMetadata(
            HBDIA<T>& matrix,
            int numProcesses
        ) = 0;
        
        // Print methods for debugging
        virtual void printProcessedDataRanges(const HBDIA<T>& matrix) = 0;
        
        // Partitioning methods (pure virtual)
        virtual void setPartitioningStrategy(PartitioningStrategy strategy) = 0;

        // Getters for accessing partitioned data
        virtual const std::vector<MatrixData>& getMatrixData() const = 0;
        virtual const std::vector<Partition<T>>& getPartitions() const = 0;
        virtual const std::vector<VectorPartition<T>>& getVectorPartitions() const = 0;
        
        // Cleanup methods
        virtual void clearMatrixPartitions() = 0;
        virtual void clearVectorPartitions() = 0;

    protected:
        PartitioningStrategy strategy_ = PartitioningStrategy::ROW_WISE; // Default strategy

        std::vector<MatrixData> matrixData_; // Matrix data per rank
        std::vector<Partition<T>> partitions_; // Partitions for each rank
        std::vector<VectorPartition<T>> vectorPartitions_; // Vector partitions for each rank
};

#endif // HBDIA_EXTRACTOR_HPP