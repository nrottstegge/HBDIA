// BasicExtractor.hpp
#ifndef BASIC_EXTRACTOR_HPP
#define BASIC_EXTRACTOR_HPP

#include "HBDIAExtractor.hpp"

template <typename T>
class BasicExtractor : public HBDIAExtractor<T> {
    public:
        BasicExtractor();
        ~BasicExtractor() override;

        bool createMatrixPartitions(
            const HBDIA<T>& matrix, 
            int numProcesses
        ) override;
        
        bool createVectorPartitions(
            const std::vector<T>& globalVector,
            int numProcesses
        ) override;
        
        void setPartitioningStrategy(typename HBDIAExtractor<T>::PartitioningStrategy strategy) override;

        void extractPartialMatrixMetadata(
            HBDIA<T>& matrix,
            int numProcesses
        ) override;
        
        void printProcessedDataRanges(const HBDIA<T>& matrix) override;
        
        // Partitioning methods (pure virtual)
        const std::vector<MatrixData>& getMatrixData() const override;
        const std::vector<Partition<T>>& getPartitions() const override;
        const std::vector<VectorPartition<T>>& getVectorPartitions() const override;
        
        void clearMatrixPartitions() override;
        void clearVectorPartitions() override;

    private:
        bool partitionRowWise(const HBDIA<T>& matrix, int num_partitions);
        bool partitionVectorRowWise(const std::vector<T>& globalVector, int num_partitions);
        
        void extractPartialMatrixMetadataRowWise(HBDIA<T>& matrix, int numProcesses);
};

#endif // BASIC_EXTRACTOR_HPP
        