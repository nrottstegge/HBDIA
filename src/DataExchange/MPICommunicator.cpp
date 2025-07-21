// MPICommunicator.cpp

#include "../../include/DataExchange/MPICommunicator.hpp"
#include "../../include/DataExchange/HBDIAExtractor.hpp"
#include <iostream>
#include <vector>
#include <cassert>

template <typename T>
bool MPICommunicator<T>::initialize(int argc, char* argv[]) {
    int provided;
    CHECK_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &this->size_));

    this->mpi_datatype = this->getMPIDataType();
    
    this->initialized_ = true;
    
    return true;
}

template <typename T>
bool MPICommunicator<T>::finalize() {
    if (!this->initialized_) {
        return false;
    }
    
    CHECK_MPI(MPI_Finalize());
    this->initialized_ = false;
    return true;
}

// Helper function to get MPI data type
template <typename T>
MPI_Datatype MPICommunicator<T>::getMPIDataType() {
    if constexpr (std::is_same_v<T, int>) {
        return MPI_INT;
    } else if constexpr (std::is_same_v<T, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long>) {
        return MPI_LONG;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return MPI_UNSIGNED;
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported data type for MPI communication");
        return MPI_BYTE; // This line will never be reached
    }
}

template <typename T>
void MPICommunicator<T>::barrier() {
    if (!this->initialized_) {
        std::cerr << "MPICommunicator not initialized" << std::endl;
        return;
    }
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    return;
}


template <typename T>
bool MPICommunicator<T>::sendPartition(const std::vector<MatrixData>& matrixDataVec, const std::vector<Partition<T>>& partitions, int senderRank) {
    // Check that we have data for all processes
    if (matrixDataVec.size() != this->size_ || partitions.size() != this->size_) {
        std::cerr << "Error: Number of partitions/matrixData must match number of processes" << std::endl;
        return false;
    }
    
    // Check that the current process is the sender
    if (this->rank_ != senderRank) {
        std::cerr << "Error: Only the sender process can call sendPartition" << std::endl;
        return false;
    }
    
    std::vector<MPI_Request> requests;
    // Store sizes persistently to avoid scope issues with async MPI operations
    std::vector<int> globalRowMappingSizes;
    globalRowMappingSizes.reserve(this->size_);
    
    // Send to all processes (excluding the sender)
    for (int dest = 0; dest < this->size_; ++dest) {
        if (dest == senderRank) continue; // Skip the sender itself
        
        const MatrixData& matrixData = matrixDataVec[dest];
        const Partition<T>& partition = partitions[dest];
        
        // Send matrix metadata first so each process knows how much data to expect
        MPI_Request metadata_request;
        CHECK_MPI(MPI_Isend(&matrixData, sizeof(MatrixData), MPI_BYTE, dest, 0, MPI_COMM_WORLD, &metadata_request));
        requests.push_back(metadata_request);
        
        // Use numLocalNonZeros from MatrixData instead of sending partition size separately
        int partitionSize = matrixData.numLocalNonZeros;
        
        if (partitionSize > 0) {
            // Send partition.rowIndices
            MPI_Request row_request;
            CHECK_MPI(MPI_Isend(partition.localRowIndices.data(), partitionSize, MPI_INT, dest, 2, MPI_COMM_WORLD, &row_request));
            requests.push_back(row_request);
            
            // Send partition.colIndices
            MPI_Request col_request;
            CHECK_MPI(MPI_Isend(partition.localColIndices.data(), partitionSize, MPI_INT, dest, 3, MPI_COMM_WORLD, &col_request));
            requests.push_back(col_request);
            
            // Send partition.values
            MPI_Request val_request;
            CHECK_MPI(MPI_Isend(partition.localValues.data(), partitionSize, this->mpi_datatype, dest, 4, MPI_COMM_WORLD, &val_request));
            requests.push_back(val_request);
            
            // Store global row mapping size persistently for async MPI operations
            globalRowMappingSizes.push_back(partition.globalRowMapping.size());
            int& globalRowMappingSize = globalRowMappingSizes.back();
            
            // Send global row mapping size
            MPI_Request global_row_size_request;
            CHECK_MPI(MPI_Isend(&globalRowMappingSize, 1, MPI_INT, dest, 8, MPI_COMM_WORLD, &global_row_size_request));
            requests.push_back(global_row_size_request);
            
            if (globalRowMappingSize > 0) {
                MPI_Request global_row_request;
                CHECK_MPI(MPI_Isend(partition.globalRowMapping.data(), globalRowMappingSize, MPI_INT, dest, 9, MPI_COMM_WORLD, &global_row_request));
                requests.push_back(global_row_request);
            }
        } else {
            // Even if partitionSize is 0, we still need a placeholder for globalRowMappingSize
            globalRowMappingSizes.push_back(0);
        }
    }
    
    // Wait for all sends to complete
    if (!requests.empty()) {
        std::vector<MPI_Status> statuses(requests.size());
        CHECK_MPI(MPI_Waitall(requests.size(), requests.data(), statuses.data()));
    }
    
    return true;
}

template <typename T>
bool MPICommunicator<T>::receivePartition(MatrixData& matrixData, Partition<T>& partition, int senderRank) {
    MPI_Status status;
    
    // Receive matrix metadata first
    CHECK_MPI(MPI_Recv(&matrixData, sizeof(MatrixData), MPI_BYTE, senderRank, 0, MPI_COMM_WORLD, &status));
    
    // Use numLocalNonZeros from MatrixData instead of receiving partition size separately
    int partitionSize = matrixData.numLocalNonZeros;
    
    if (partitionSize > 0) {
        // Resize vectors to accommodate the data
        partition.localRowIndices.resize(partitionSize);
        partition.localColIndices.resize(partitionSize);
        partition.localValues.resize(partitionSize);
        
        // Receive partition.rowIndices
        CHECK_MPI(MPI_Recv(partition.localRowIndices.data(), partitionSize, MPI_INT, senderRank, 2, MPI_COMM_WORLD, &status));
        
        // Receive partition.colIndices  
        CHECK_MPI(MPI_Recv(partition.localColIndices.data(), partitionSize, MPI_INT, senderRank, 3, MPI_COMM_WORLD, &status));
        
        // Receive partition.values
        CHECK_MPI(MPI_Recv(partition.localValues.data(), partitionSize, this->mpi_datatype, senderRank, 4, MPI_COMM_WORLD, &status));
        
        // Receive global row mapping size
        int globalRowMappingSize;
        CHECK_MPI(MPI_Recv(&globalRowMappingSize, 1, MPI_INT, senderRank, 8, MPI_COMM_WORLD, &status));
        
        if (globalRowMappingSize > 0) {
            // Resize and receive global row mapping
            partition.globalRowMapping.resize(globalRowMappingSize);
            CHECK_MPI(MPI_Recv(partition.globalRowMapping.data(), globalRowMappingSize, MPI_INT, senderRank, 9, MPI_COMM_WORLD, &status));
        }
    }
    
    return true;
}

template <typename T>
bool MPICommunicator<T>::sendVectorPartition(const std::vector<VectorPartition<T>>& vectorPartitions, int senderRank) {
    // Check that we have data for all processes
    if (vectorPartitions.size() != this->size_) {
        std::cerr << "Error: Number of vector partitions must match number of processes" << std::endl;
        return false;
    }
    
    // Check that the current process is the sender
    if (this->rank_ != senderRank) {
        std::cerr << "Error: Only the sender process can call sendVectorPartition" << std::endl;
        return false;
    }
    
    std::vector<MPI_Request> requests;
    // Store partition sizes persistently to avoid scope issues with async MPI operations
    std::vector<int> vectorPartitionSizes;
    vectorPartitionSizes.reserve(this->size_);
    
    // Send to all processes (excluding the sender)
    for (int dest = 0; dest < this->size_; ++dest) {
        if (dest == senderRank) continue; // Skip the sender itself
        
        const VectorPartition<T>& vectorPartition = vectorPartitions[dest];
        
        // Store partition size persistently for async MPI operations
        vectorPartitionSizes.push_back(vectorPartition.localValues.size());
        int& vectorPartitionSize = vectorPartitionSizes.back();
        
        // Send vector partition size directly (no metadata needed)
        MPI_Request size_request;
        CHECK_MPI(MPI_Isend(&vectorPartitionSize, 1, MPI_INT, dest, 6, MPI_COMM_WORLD, &size_request));
        requests.push_back(size_request);
        
        if (vectorPartitionSize > 0) {
            // Send vector partition values
            MPI_Request val_request;
            CHECK_MPI(MPI_Isend(vectorPartition.localValues.data(), vectorPartitionSize, this->mpi_datatype, dest, 7, MPI_COMM_WORLD, &val_request));
            requests.push_back(val_request);
        }
    }
    
    // Wait for all sends to complete
    if (!requests.empty()) {
        std::vector<MPI_Status> statuses(requests.size());
        CHECK_MPI(MPI_Waitall(requests.size(), requests.data(), statuses.data()));
    }
    
    return true;
}

template <typename T>
bool MPICommunicator<T>::receiveVectorPartition(VectorPartition<T>& vectorPartition, int senderRank) {
    MPI_Status status;
    
    // Receive vector partition size directly (no metadata needed)
    int vectorPartitionSize;
    CHECK_MPI(MPI_Recv(&vectorPartitionSize, 1, MPI_INT, senderRank, 6, MPI_COMM_WORLD, &status));
    
    if (vectorPartitionSize > 0) {
        // Resize vector to accommodate the data
        vectorPartition.localValues.resize(vectorPartitionSize);
        
        // Receive vector partition values
        CHECK_MPI(MPI_Recv(vectorPartition.localValues.data(), vectorPartitionSize, this->mpi_datatype, senderRank, 7, MPI_COMM_WORLD, &status));
    }
    
    return true;
}

template <typename T>
bool MPICommunicator<T>::exchangeData(const HBDIA<T>& matrix, HBDIAVector<T>& vector) {
    if (!matrix.isPartialMatrix() || !matrix.hasPartialMatrixMetadata()) {
        std::cerr << "Matrix must be a partial matrix with metadata for data exchange" << std::endl;
        return false;
    }
    
    if (!vector.getUnifiedDataPtr()) {
        std::cerr << "Vector must have unified memory allocated" << std::endl;
        return false;
    }
    
    const auto& metadata = matrix.getPartialMatrixMetadata();
    const auto& processDataRanges = metadata.processDataRanges;  // Per process, which tuples to request from that process
    const auto& globalRowMapping = matrix.getGlobalRowMapping();
    
    // Step 1: Call MPI_recv on every rank - receive how many tuples each rank will request from us
    std::vector<int> numTuplesFromEachRank(this->size_);
    std::vector<MPI_Request> recvCountRequests;
    
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank != this->rank_) {
            MPI_Request req;
            CHECK_MPI(MPI_Irecv(&numTuplesFromEachRank[rank], 1, MPI_INT, rank, 100, MPI_COMM_WORLD, &req));
            recvCountRequests.push_back(req);
        } else {
            numTuplesFromEachRank[rank] = 0; // We don't request from ourselves
        }
    }
    
    // Step 2: Send to each rank how many tuples we will request from them
    std::vector<MPI_Request> sendCountRequests;
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank != this->rank_) {
            int numTuplesWeRequestFromThisRank = 0;
            if (rank < static_cast<int>(processDataRanges.size())) {
                numTuplesWeRequestFromThisRank = static_cast<int>(processDataRanges[rank].size());
            }
            MPI_Request req;
            CHECK_MPI(MPI_Isend(&numTuplesWeRequestFromThisRank, 1, MPI_INT, rank, 100, MPI_COMM_WORLD, &req));
            sendCountRequests.push_back(req);
        }
    }
    
    // Wait for receiving count requests to complete
    if (!recvCountRequests.empty()) {
        CHECK_MPI(MPI_Waitall(static_cast<int>(recvCountRequests.size()), recvCountRequests.data(), MPI_STATUSES_IGNORE));
    }
    
    // Step 4: Post recv calls for the tuples to get from each rank
    std::vector<std::vector<std::tuple<int, int>>> tuplesFromEachRank(this->size_);
    std::vector<MPI_Request> recvTupleRequests;
    
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank != this->rank_ && numTuplesFromEachRank[rank] > 0) {
            tuplesFromEachRank[rank].resize(numTuplesFromEachRank[rank]);
            MPI_Request req;
            CHECK_MPI(MPI_Irecv(tuplesFromEachRank[rank].data(), numTuplesFromEachRank[rank] * 2, MPI_INT, rank, 101, MPI_COMM_WORLD, &req));
            recvTupleRequests.push_back(req);
        }
    }
    
    // Step 5: Post recv calls for data from processDataRanges (going rank by rank)
    int receiveOffset = 0;  // Start at left buffer (offset 0)
    std::vector<MPI_Request> recvDataRequests;
    
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank == this->rank_) {
            // When rank == myRank, add localVector.size() to move to right_rcv_buff
            receiveOffset += static_cast<int>(globalRowMapping.size());
        } else if (rank < static_cast<int>(processDataRanges.size())) {
            const auto& tuplesToRequestFromThisRank = processDataRanges[rank];
            
            // Receive data from this rank for each tuple we're requesting
            for (const auto& tuple : tuplesToRequestFromThisRank) {
                int rangeStart = std::get<0>(tuple);
                int rangeEnd = std::get<1>(tuple);
                int rangeSize = rangeEnd - rangeStart;
                
                if (rangeSize > 0) {
                    T* recvPtr = vector.getUnifiedDataPtr() + receiveOffset;
                    MPI_Request req;
                    CHECK_MPI(MPI_Irecv(recvPtr, rangeSize, this->mpi_datatype, rank, 200, MPI_COMM_WORLD, &req));
                    recvDataRequests.push_back(req);
                    receiveOffset += rangeSize;
                }
            }
        }
    }
    
    // Send our tuple requests to all ranks
    std::vector<MPI_Request> sendTupleRequests;
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank != this->rank_ && rank < static_cast<int>(processDataRanges.size())) {
            const auto& tuplesToRequestFromThisRank = processDataRanges[rank];
            if (!tuplesToRequestFromThisRank.empty()) {
                MPI_Request req;
                CHECK_MPI(MPI_Isend(tuplesToRequestFromThisRank.data(), static_cast<int>(tuplesToRequestFromThisRank.size()) * 2, MPI_INT, rank, 101, MPI_COMM_WORLD, &req));
                sendTupleRequests.push_back(req);
            }
        }
    }
    
    // Step 6: Wait for the tuples
    if (!recvTupleRequests.empty()) {
        CHECK_MPI(MPI_Waitall(static_cast<int>(recvTupleRequests.size()), recvTupleRequests.data(), MPI_STATUSES_IGNORE));
    }

    
    // Step 7: Send our own data based on what each rank requested
    std::vector<MPI_Request> sendDataRequests;
    
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank != this->rank_ && numTuplesFromEachRank[rank] > 0) {
            // Process each tuple request from this rank
            for (const auto& requestedRange : tuplesFromEachRank[rank]) {
                int rangeStart = std::get<0>(requestedRange);
                int rangeEnd = std::get<1>(requestedRange);
                int rangeSize = rangeEnd - rangeStart;
                
                // Find the local index where globalIdx == rangeStart
                size_t startLocalIdx = SIZE_MAX;
                for (size_t localIdx = 0; localIdx < globalRowMapping.size(); ++localIdx) {
                    int globalIdx = globalRowMapping[localIdx];
                    if (globalIdx == rangeStart) {
                        startLocalIdx = localIdx;
                        break;
                    }
                }
                
                // Send directly from the unified vector starting at startLocalIdx
                if (startLocalIdx != SIZE_MAX && rangeSize > 0) {
                    T* sendPtr = vector.unified_local_ptr_ + startLocalIdx;
                    MPI_Request req;
                    CHECK_MPI(MPI_Isend(sendPtr, rangeSize, this->mpi_datatype, rank, 200, MPI_COMM_WORLD, &req));
                    sendDataRequests.push_back(req);
                }
            }
        }
    }

    if (!sendTupleRequests.empty()) {
        CHECK_MPI(MPI_Waitall(static_cast<int>(sendTupleRequests.size()), sendTupleRequests.data(), MPI_STATUSES_IGNORE));
    }
    
    // Step 8: Wait for everything to finish
    if (!recvDataRequests.empty()) {
        CHECK_MPI(MPI_Waitall(static_cast<int>(recvDataRequests.size()), recvDataRequests.data(), MPI_STATUSES_IGNORE));
    }
    
    if (!sendDataRequests.empty()) {
        CHECK_MPI(MPI_Waitall(static_cast<int>(sendDataRequests.size()), sendDataRequests.data(), MPI_STATUSES_IGNORE));
    }
    
    // Wait for count requests to finish
    if (!sendCountRequests.empty()) {
        CHECK_MPI(MPI_Waitall(static_cast<int>(sendCountRequests.size()), sendCountRequests.data(), MPI_STATUSES_IGNORE));
    }
    
    return true;
}

template <typename T>
bool MPICommunicator<T>::verifyDataExchange(const HBDIA<T>& matrix, const HBDIAVector<T>& vector) {
    if (!matrix.isPartialMatrix() || !matrix.hasPartialMatrixMetadata()) {
        std::cerr << "Matrix must be a partial matrix with metadata for verification" << std::endl;
        return false;
    }
    
    if (!vector.getUnifiedDataPtr()) {
        std::cerr << "Vector must have unified memory allocated for verification" << std::endl;
        return false;
    }
    
    const auto& metadata = matrix.getPartialMatrixMetadata();
    const auto& dataRanges = metadata.dataRanges;  // What this process needs (unsorted)
    const auto& processDataRanges = metadata.processDataRanges;  // Per process, which tuples to request from that process
    const auto& globalRowMapping = matrix.getGlobalRowMapping();
    
    bool allCorrect = true;
    int verificationErrors = 0;
    
    // Verify left and right receive buffers
    int receiveOffset = 0;  // Start at left buffer (offset 0)
    
    for (int rank = 0; rank < this->size_; ++rank) {
        if (rank == this->rank_) {
            // When rank == myRank, skip localVector.size() to move to right_rcv_buff
            receiveOffset += static_cast<int>(globalRowMapping.size());
        } else if (rank < static_cast<int>(processDataRanges.size())) {
            const auto& tuplesToRequestFromThisRank = processDataRanges[rank];
            
            // Verify data from this rank for each tuple we requested
            for (const auto& tuple : tuplesToRequestFromThisRank) {
                int rangeStart = std::get<0>(tuple);
                int rangeEnd = std::get<1>(tuple);
                int rangeSize = rangeEnd - rangeStart;
                
                if (rangeSize > 0) {
                    T* recvPtr = vector.getUnifiedDataPtr() + receiveOffset;
                    
                    // Verify that each received value matches its expected global index
                    for (int i = 0; i < rangeSize; ++i) {
                        T expectedValue = static_cast<T>(rangeStart + i);
                        T actualValue = recvPtr[i];
                        
                        if (actualValue != expectedValue) {
                            std::cerr << "Verification ERROR: Rank " << this->rank_ 
                                     << " expected value " << expectedValue 
                                     << " at global index " << (rangeStart + i)
                                     << " but received " << actualValue << std::endl;
                            allCorrect = false;
                            verificationErrors++;
                        }
                    }
                    receiveOffset += rangeSize;
                }
            }
        }
    }
    
    // Also verify our local data is correct (should be globalRowMapping values)
    for (size_t localIdx = 0; localIdx < globalRowMapping.size(); ++localIdx) {
        int globalIdx = globalRowMapping[localIdx];
        T expectedValue = static_cast<T>(globalIdx);
        T actualValue = vector.unified_local_ptr_[localIdx];
        
        if (actualValue != expectedValue) {
            std::cerr << "Verification ERROR: Rank " << this->rank_ 
                     << " local data at index " << localIdx 
                     << " (global index " << globalIdx << ") expected " << expectedValue
                     << " but found " << actualValue << std::endl;
            allCorrect = false;
            verificationErrors++;
        }
    }
    
    if (allCorrect) {
        std::cout << "Data exchange verification PASSED for rank " << this->rank_ << std::endl;
    } else {
        std::cerr << "Data exchange verification FAILED for rank " << this->rank_ 
                  << " with " << verificationErrors << " errors" << std::endl;
    }
    
    return allCorrect;
}

// Gather vector data from all ranks to root
// Each rank sends its local vector size, then the data
// Only root fills globalVector

template <typename T>
bool MPICommunicator<T>::gatherVectorData(const HBDIAVector<T>& localVector, std::vector<T>& globalVector, int rootRank) {
    assert(rootRank == 0);
    int localSize = static_cast<int>(localVector.getLocalSize());
    std::vector<int> recvCounts;
    std::vector<int> displs;

    if (this->rank_ == rootRank) {
        recvCounts.resize(this->size_);
        displs.resize(this->size_);
    }
    
    // Gather sizes first
    CHECK_MPI(MPI_Gather(&localSize, 1, MPI_INT, this->rank_ == rootRank ? recvCounts.data() : nullptr, 1, MPI_INT, rootRank, MPI_COMM_WORLD));

    if (this->rank_ == rootRank) {
        // Calculate displacements and total size
        int totalSize = 0;
        for (int i = 0; i < this->size_; ++i) {
            displs[i] = totalSize;
            totalSize += recvCounts[i];
        }
        globalVector.resize(totalSize);
    }

    // Gather the actual data
    const T* sendbuf = localVector.getLocalDataPtr();
    CHECK_MPI(MPI_Gatherv(sendbuf, localSize, this->mpi_datatype,
                         this->rank_ == rootRank ? globalVector.data() : nullptr,
                         this->rank_ == rootRank ? recvCounts.data() : nullptr,
                         this->rank_ == rootRank ? displs.data() : nullptr,
                         this->mpi_datatype,
                         rootRank, MPI_COMM_WORLD));
    
    return true;
}

// Explicit template instantiations
template class MPICommunicator<int>;
template class MPICommunicator<float>;
template class MPICommunicator<double>;