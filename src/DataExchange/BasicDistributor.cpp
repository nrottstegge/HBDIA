// BasicDistributor.cpp

#include "../../include/DataExchange/BasicDistributor.hpp"
#include <iostream>

template <typename T>
BasicDistributor<T>::BasicDistributor(std::unique_ptr<HBDIACommunicator<T>> communicator, 
                                      std::unique_ptr<HBDIAExtractor<T>> extractor,
                                      int rootProcess)
    : HBDIADistributor<T>(std::move(communicator), std::move(extractor), rootProcess)
{
}

template <typename T>
BasicDistributor<T>::~BasicDistributor() = default;

template <typename T>
bool BasicDistributor<T>::scatterMatrix(
    HBDIA<T>& globalMatrix,
    HBDIA<T>& localMatrix
) {
    if (!this->communicator_->isInitialized()) {
        std::cerr << "Communicator not initialized" << std::endl;
        return false;
    }
    
    if (!this->isRootProcess()) {
        std::cerr << "scatterMatrix should only be called by root process" << std::endl;
        return false;
    }
    
    // Set the partitioning strategy in the extractor to be the same of this distributor
    this->extractor_->setPartitioningStrategy(this->strategy_);
    
    // Create partitions using the extractor
    int numProcesses = this->getSize();
    if (!this->extractor_->createMatrixPartitions(globalMatrix, numProcesses)) {
        std::cerr << "Failed to create matrix partitions" << std::endl;
        return false;
    }
    
    // Get partitions and matrix data
    const auto& partitions = this->extractor_->getPartitions();
    const auto& matrixDataVec = this->extractor_->getMatrixData();
    
    // Send partitions to all processes using the communicator interface
    if (!this->communicator_->sendPartition(matrixDataVec, partitions, this->rootProcess_)) {
        std::cerr << "Failed to send partitions" << std::endl;
        return false;
    }
    
    // Create local matrix for root process
    const auto& rootPartition = partitions[this->rootProcess_];
    const auto& rootMatrixData = matrixDataVec[this->rootProcess_];
    
    try {
        localMatrix = HBDIA<T>(rootPartition.localRowIndices,
                              rootPartition.localColIndices,
                              rootPartition.localValues,
                              rootMatrixData.numLocalRows,
                              rootMatrixData.numLocalCols,
                              rootPartition.globalRowMapping,
                              true,
                              globalMatrix.getNumGlobalRows(),
                              globalMatrix.getNumGlobalCols(),
                              globalMatrix.getNumGlobalNonZeros(),
                              this->getRank(),
                              this->getSize()
                            );

        localMatrix.convertToHBDIAFormat();
        localMatrix.analyzeDataRanges();
        // Extract partial matrix metadata (stored internally in extractor)
        this->extractor_->extractPartialMatrixMetadata(
            localMatrix, 
            numProcesses
        );
        localMatrix.calculateVectorOffsets(this->getRank(), this->getSize());
        localMatrix.prepareForGPU();
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create local matrix for root process: " << e.what() << std::endl;
        return false;
    }

    // Clear matrix partitions to prepare for next operation
    this->extractor_->clearMatrixPartitions();
    
    return true;
}

template <typename T>
bool BasicDistributor<T>::receiveMatrix(
    HBDIA<T>& localMatrix
) {
    if (!this->communicator_->isInitialized()) {
        std::cerr << "Communicator not initialized" << std::endl;
        return false;
    }
    
    if (this->isRootProcess()) {
        std::cerr << "receiveMatrix should not be called by root process" << std::endl;
        return false;
    }
    
    // Receive partition data using the communicator interface
    MatrixData matrixData;
    Partition<T> partition;
    
    if (!this->communicator_->receivePartition(matrixData, partition, this->rootProcess_)) {
        std::cerr << "Failed to receive partition" << std::endl;
        return false;
    }
    
    // Create local matrix from received partition
    try {
        localMatrix = HBDIA<T>(partition.localRowIndices, 
                         partition.localColIndices, 
                         partition.localValues, 
                         matrixData.numLocalRows, 
                         matrixData.numLocalCols,
                         partition.globalRowMapping,
                         true,
                         matrixData.numGlobalRows,
                         matrixData.numGlobalCols,
                         matrixData.numGlobalNonZeros,
                         this->getRank(),
                         this->getSize()
                        );

        localMatrix.convertToHBDIAFormat();
        localMatrix.analyzeDataRanges();
        // Extract partial matrix metadata (stored internally in extractor)
        this->extractor_->extractPartialMatrixMetadata(
            localMatrix, 
            this->communicator_->getSize()
        );
        localMatrix.calculateVectorOffsets(this->getRank(), this->getSize());
        localMatrix.prepareForGPU();


    } catch (const std::exception& e) {
        std::cerr << "Failed to create local matrix: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

template <typename T>
bool BasicDistributor<T>::scatterVector(
    const std::vector<T>& globalVector,
    std::vector<T>& localVector
) {
    if (!this->communicator_->isInitialized()) {
        std::cerr << "Communicator not initialized" << std::endl;
        return false;
    }
    
    if (!this->isRootProcess()) {
        std::cerr << "scatterVector should only be called by root process" << std::endl;
        return false;
    }
    
    // Set the partitioning strategy in the extractor
    this->extractor_->setPartitioningStrategy(this->strategy_);
    
    // Create vector partitions using the extractor
    if (!this->extractor_->createVectorPartitions(globalVector, this->communicator_->getSize())) {
        std::cerr << "Failed to create vector partitions" << std::endl;
        return false;
    }
    
    // Get the partitioned data (no need for vectorDataVec anymore)
    const auto& vectorPartitions = this->extractor_->getVectorPartitions();
    
    // Send partitions to other processes using the communicator interface
    if (!this->communicator_->sendVectorPartition(vectorPartitions, this->rootProcess_)) {
        std::cerr << "Failed to send vector partitions" << std::endl;
        return false;
    }
    
    // Create local vector for root process
    const auto& rootVectorPartition = vectorPartitions[this->rootProcess_];
    localVector = rootVectorPartition.localValues;
    
    // Clear vector partitions to prepare for next operation
    this->extractor_->clearVectorPartitions();
    
    return true;
}

template <typename T>
bool BasicDistributor<T>::receiveVector(
    std::vector<T>& vector
) {
    if (!this->communicator_->isInitialized()) {
        std::cerr << "Communicator not initialized" << std::endl;
        return false;
    }
    
    if (this->isRootProcess()) {
        std::cerr << "receiveVector should not be called by root process" << std::endl;
        return false;
    }
    
    // Receive partition data using the communicator interface (no VectorData needed)
    VectorPartition<T> vectorPartition;
    
    if (!this->communicator_->receiveVectorPartition(vectorPartition, this->rootProcess_)) {
        std::cerr << "Failed to receive vector partition" << std::endl;
        return false;
    }
    
    // Set the local vector from received partition
    vector = vectorPartition.localValues;
    
    return true;
}

template <typename T>
void BasicDistributor<T>::setPartitioningStrategy(typename HBDIAExtractor<T>::PartitioningStrategy strategy) {
    this->strategy_ = strategy;
    this->extractor_->setPartitioningStrategy(strategy);
}

template <typename T>
bool BasicDistributor<T>::exchangeData(const HBDIA<T>& matrix, HBDIAVector<T>& vector) {
    if (!this->communicator_->isInitialized()) {
        std::cerr << "Communicator not initialized" << std::endl;
        return false;
    }
    
    // Delegate to the communicator's exchangeData method
    return this->communicator_->exchangeData(matrix, vector);
}

// Gather vector from all ranks to root
// Fills globalVector on root, leaves it empty on others
template <typename T>
bool BasicDistributor<T>::gatherVector(
    const HBDIAVector<T>& localVector,
    std::vector<T>& globalVector
) {
    int root = this->rootProcess_;
    return this->communicator_->gatherVectorData(localVector, globalVector, root);
}

// Explicit template instantiations
template class BasicDistributor<int>;
template class BasicDistributor<float>;
template class BasicDistributor<double>;