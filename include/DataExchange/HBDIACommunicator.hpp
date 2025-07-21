// HBDIACommunicator.hpp
#ifndef HBDIA_COMMUNICATOR_HPP
#define HBDIA_COMMUNICATOR_HPP

#include <memory>
#include <vector>
#include "../Format/HBDIA.hpp"
#include "../Format/HBDIAVector.hpp"
#include "HBDIAExtractor.hpp"

template <typename T>
class HBDIACommunicator {
    public:
        HBDIACommunicator() : rank_(-1), size_(-1), initialized_(false) {}
        virtual ~HBDIACommunicator() = default;
    
        // Initialize communication backend
        virtual bool initialize(int argc, char* argv[]) = 0;
        virtual bool finalize() = 0;
        virtual int getRank() const = 0;
        virtual int getSize() const = 0;
        virtual bool isInitialized() const = 0;
        
        // Communication methods
        virtual void barrier() = 0;
        virtual bool sendPartition(const std::vector<MatrixData>& matrixDataVec, const std::vector<Partition<T>>& partitions, int senderRank) = 0;
        virtual bool receivePartition(MatrixData& matrixData, Partition<T>& partition, int senderRank) = 0;
        virtual bool sendVectorPartition(const std::vector<VectorPartition<T>>& vectorPartitions, int senderRank) = 0;
        virtual bool receiveVectorPartition(VectorPartition<T>& vectorPartition, int senderRank) = 0;
        // Gather vector data from all ranks to root. Only root fills globalVector.
        virtual bool gatherVectorData(const HBDIAVector<T>& localVector, std::vector<T>& globalVector, int rootRank) = 0;
        virtual bool exchangeData(const HBDIA<T>& matrix, HBDIAVector<T>& vector) = 0;
        virtual bool verifyDataExchange(const HBDIA<T>& matrix, const HBDIAVector<T>& vector) = 0;

    protected:
        int rank_;
        int size_;
        bool initialized_;
};

#endif // HBDIA_COMMUNICATOR_HPP