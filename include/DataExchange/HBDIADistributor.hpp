// HBDIADistributor.hpp
#ifndef HBDIA_DISTRIBUTOR_HPP
#define HBDIA_DISTRIBUTOR_HPP

#include "../Format/HBDIA.hpp"
#include "../Format/HBDIAVector.hpp"
#include "HBDIACommunicator.hpp"
#include "HBDIAExtractor.hpp"
#include <memory>
#include <stdexcept>

template <typename T>
class HBDIADistributor {
public:
    HBDIADistributor(std::unique_ptr<HBDIACommunicator<T>> communicator, 
                     std::unique_ptr<HBDIAExtractor<T>> extractor,
                     int rootProcess = 0)
        : communicator_(std::move(communicator))
        , extractor_(std::move(extractor))
        , strategy_(HBDIAExtractor<T>::PartitioningStrategy::ROW_WISE)
        , rootProcess_(rootProcess)
    {
        if (!communicator_) {
            throw std::invalid_argument("Communicator cannot be null");
        }
        if (!extractor_) {
            throw std::invalid_argument("Extractor cannot be null");
        }
    }
    
    virtual ~HBDIADistributor() = default;
    
    virtual bool scatterMatrix(
        HBDIA<T>& globalMatrix,
        HBDIA<T>& localMatrix
    ) = 0;
    
    virtual bool receiveMatrix(
        HBDIA<T>& matrix
    ) = 0;
    
    virtual bool scatterVector(
        const std::vector<T>& globalVector,
        std::vector<T>& localVector
    ) = 0;
    
    virtual bool receiveVector(
        std::vector<T>& vector
    ) = 0;
    
    virtual bool exchangeData(
        const HBDIA<T>& matrix,
        HBDIAVector<T>& vector
    ) = 0;
    
    virtual void setPartitioningStrategy(typename HBDIAExtractor<T>::PartitioningStrategy strategy) = 0;
    virtual int getRank() const = 0;
    virtual int getSize() const = 0;
    virtual HBDIACommunicator<T>& getCommunicator() const = 0;
    virtual HBDIAExtractor<T>& getExtractor() const { return *extractor_; }
    virtual bool isRootProcess() const { return getCommunicator().getRank() == rootProcess_; }
    virtual int getRootProcess() const { return rootProcess_; }

protected:
    int rootProcess_;

    std::unique_ptr<HBDIACommunicator<T>> communicator_;
    std::unique_ptr<HBDIAExtractor<T>> extractor_;
    typename HBDIAExtractor<T>::PartitioningStrategy strategy_;
};

#endif // HBDIA_DISTRIBUTOR_HPP