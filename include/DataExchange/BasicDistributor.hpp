// BasicDistributor.hpp
#ifndef BASIC_DISTRIBUTOR_HPP
#define BASIC_DISTRIBUTOR_HPP

#include "HBDIADistributor.hpp"
#include "BasicExtractor.hpp"

template <typename T>
class BasicDistributor : public HBDIADistributor<T> {
public:
    BasicDistributor(std::unique_ptr<HBDIACommunicator<T>> communicator, 
                     std::unique_ptr<HBDIAExtractor<T>> extractor,
                     int rootProcess = 0);
    ~BasicDistributor();

    bool scatterMatrix(
        HBDIA<T>& globalMatrix,
        HBDIA<T>& localMatrix
    ) override;
    
    bool receiveMatrix(
        HBDIA<T>& matrix
    ) override;
    
    bool scatterVector(
        const std::vector<T>& globalVector,
        std::vector<T>& localVector
    ) override;
    
    bool receiveVector(
        std::vector<T>& vector
    ) override;
    
    // Gather vector from all ranks to root. Fills globalVector on root, leaves it empty on others.
    bool gatherVector(const HBDIAVector<T>& localVector, std::vector<T>& globalVector);
    
    bool exchangeData(
        const HBDIA<T>& matrix,
        HBDIAVector<T>& vector
    ) override;
    
    void setPartitioningStrategy(typename HBDIAExtractor<T>::PartitioningStrategy strategy) override;
    int getRank() const override { return this->communicator_->getRank(); }
    int getSize() const override { return this->communicator_->getSize(); }
    int getRootProcess() const { return this->rootProcess_; }
    HBDIACommunicator<T>& getCommunicator() const override { return *this->communicator_; }

};

#endif // BASIC_DISTRIBUTOR_HPP