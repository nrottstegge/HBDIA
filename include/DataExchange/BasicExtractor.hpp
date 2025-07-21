// BasicExtractor.hpp
#ifndef BASIC_EXTRACTOR_HPP
#define BASIC_EXTRACTOR_HPP

#include "HBDIAExtractor.hpp"

template <typename T>
class BasicExtractor : public HBDIAExtractor<T> {
    public:
        BasicExtractor();
        ~BasicExtractor() override;

        // Override methods from HBDIAExtractor
        bool createMatrixPartitions(
            const HBDIA<T>& matrix, 
            int numProcesses
        ) override;
        
        bool createVectorPartitions(
            const std::vector<T>& globalVector,
            int numProcesses
        ) override;
        
        void setPartitioningStrategy(typename HBDIAExtractor<T>::PartitioningStrategy strategy) override;

        // Partial matrix extraction methods - now works directly with matrix metadata
        void extractPartialMatrixMetadata(
            HBDIA<T>& matrix,
            int numProcesses
        ) override;
        
        // Print methods for debugging
        void printProcessedDataRanges(const HBDIA<T>& matrix) override;
        
        // Partitioning methods (pure virtual)
        const std::vector<MatrixData>& getMatrixData() const override;
        const std::vector<Partition<T>>& getPartitions() const override;
        const std::vector<VectorPartition<T>>& getVectorPartitions() const override;
        
        // Cleanup methods
        void clearMatrixPartitions() override;
        void clearVectorPartitions() override;

    private:
        // Helper method for row-wise partitioning
        bool partitionRowWise(const HBDIA<T>& matrix, int num_partitions);
        bool partitionVectorRowWise(const std::vector<T>& globalVector, int num_partitions);
        
        // Helper method for row-wise partial matrix metadata extraction
        void extractPartialMatrixMetadataRowWise(HBDIA<T>& matrix, int numProcesses);
};

#endif // BASIC_EXTRACTOR_HPP
        