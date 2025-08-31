// MPICommunicator.cpp
#ifndef MPI_COMMUNICATOR_HPP
#define MPI_COMMUNICATOR_HPP

#include "HBDIACommunicator.hpp"
#include <mpi.h>
#include <vector>

template <typename T>
class MPICommunicator : public HBDIACommunicator<T> {
    public:
        MPICommunicator() : HBDIACommunicator<T>() {}
        ~MPICommunicator() override {}
        bool initialize(int argc, char* argv[]) override;
        bool finalize() override;
        int getRank() const override { return this->rank_; }
        int getSize() const override { return this->size_; }
        bool isInitialized() const override { return this->initialized_; }
        void barrier() override;
        bool sendPartition(const std::vector<MatrixData>& matrixDataVec, const std::vector<Partition<T>>& partitions, int senderRank);
        bool receivePartition(MatrixData& matrixData, Partition<T>& partition, int senderRank);
        bool sendVectorPartition(const std::vector<VectorPartition<T>>& vectorPartitions, int senderRank);
        bool receiveVectorPartition(VectorPartition<T>& vectorPartition, int senderRank);
        bool exchangeData(const HBDIA<T>& matrix, HBDIAVector<T>& vector) override;
        bool verifyDataExchange(const HBDIA<T>& matrix, const HBDIAVector<T>& vector) override;
        bool gatherVectorData(const HBDIAVector<T>& localVector, std::vector<T>& globalVector, int rootRank) override;
    
    private:
        MPI_Datatype mpi_datatype;
        MPI_Datatype getMPIDataType();
};

#endif // MPI_COMMUNICATOR_HPP