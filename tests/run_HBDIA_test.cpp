// run_HBDIA.cpp
#include "../include/Format/HBDIA.hpp"
#include "../include/Format/HBDIAVector.hpp"
#include "../include/Format/HBDIAPrinter.hpp"
#include "../include/DataExchange/BasicDistributor.hpp"
#include "../include/DataExchange/BasicExtractor.hpp"
#include "../include/DataExchange/MPICommunicator.hpp"
#include "../include/Operations/HBDIASpMV.cuh"

#include <iostream>
#include <chrono>
#include <cmath>

using DataType = double;

int main(int argc, char *argv[]){

    int myRank = 0;
    std::string filename = "/users/nrottstegge/SuiteSparseMatrixCollection/channel-500x100x100-b050/channel-500x100x100-b050.mtx";

    auto communicator = std::make_unique<MPICommunicator<DataType>>();
    communicator->initialize(argc, argv);

    
    auto extractor = std::make_unique<BasicExtractor<DataType>>();
    
    int rootProcess = 0;
    BasicDistributor<DataType> distributor(std::move(communicator), std::move(extractor), rootProcess);
    distributor.setPartitioningStrategy(HBDIAExtractor<DataType>::PartitioningStrategy::ROW_WISE);
    
    myRank = distributor.getRank();

    HBDIA<DataType> localMatrix;
    std::vector<DataType> localVector;
    HBDIA<DataType> globalMatrix;

    if (distributor.getRank() == rootProcess) {
        globalMatrix.loadMTX(filename);
        globalMatrix.convertToHBDIAFormat();
        //globalMatrix.printHBDIA();

        // Create global vector initialized with values 0...numRows-1
        int numRows = globalMatrix.getNumRows();
        std::vector<DataType> globalVector(numRows);
        for (int i = 0; i < numRows; i++) {
            globalVector[i] = static_cast<DataType>(i);
        }
        
        std::cout << "\nGlobal vector: [";
        for (int i = 0; i < std::min(10, numRows); i++) {
            std::cout << globalVector[i];
            if (i < std::min(10, numRows) - 1) std::cout << ", ";
        }
        if (numRows > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        if (!distributor.scatterMatrix(globalMatrix, localMatrix)) {
            std::cerr << "Failed to scatter matrix" << std::endl;
            distributor.getCommunicator().finalize();
            return 1;
        }
        
        if (!distributor.scatterVector(globalVector, localVector)) {
            std::cerr << "Failed to scatter vector" << std::endl;
            distributor.getCommunicator().finalize();
            return 1;
        }
    } else {
        if (!distributor.receiveMatrix(localMatrix)) {
            std::cerr << "Failed to receive matrix" << std::endl;
            distributor.getCommunicator().finalize();
            return 1;
        }
        
        if (!distributor.receiveVector(localVector)) {
            std::cerr << "Failed to receive vector" << std::endl;
            distributor.getCommunicator().finalize();
            return 1;
        }
    }

    // Create HBDIAVector from the local vector data with matrix integration
    HBDIAVector<DataType> hbdiaVector(localVector, localMatrix, distributor.getRank(), distributor.getSize());
    
    // Print HBDIAVector information
    if(myRank == 0)  std::cout << "HBDIAVector created with local size: " << hbdiaVector.getLocalSize() 
              << ", total size: " << hbdiaVector.getTotalSize() << std::endl;

    // Exchange data to fill left and right buffers
    if(myRank == 0) std::cout << "Exchanging vector data between processes..." << std::endl;
    if (!distributor.exchangeData(localMatrix, hbdiaVector)) {
        std::cerr << myRank << " :Failed to exchange vector data" << std::endl;
        distributor.getCommunicator().finalize();
        return 1;
    }
    if(myRank == 0)  std::cout << "Data exchange completed successfully" << std::endl;

    for(int i = 0; i< distributor.getSize(); ++i) {
        distributor.getCommunicator().barrier();
        if (distributor.getRank() == i) {
            std::cout << "\n========== Process " << i << " Matrix Information ==========" << std::endl;
            //if(myRank == 1) localMatrix.print();
            std::cout << "\n========== Process " << i << " Vector Information ==========" << std::endl;
            //if(myRank == 1) hbdiaVector.print();
        }
        distributor.getCommunicator().barrier();
    }

    for(int i = 0; i< distributor.getSize(); ++i) {
        distributor.getCommunicator().barrier();
        if (distributor.getRank() == i) {
            // Verify that the data exchange worked correctly
            std::cout << myRank << "Verifying data exchange..." << std::endl;
            if (!distributor.getCommunicator().verifyDataExchange(localMatrix, hbdiaVector)) {
                std::cerr << myRank << "Data exchange verification failed!" << std::endl;
                distributor.getCommunicator().finalize();
                return 1;
            }
            std::cout << "Data exchange verification completed" << std::endl;
        }
        distributor.getCommunicator().barrier();
    }



    // Test hybrid SpMV
    if(myRank == 0) std::cout << "\n========== Testing Hybrid GPU+CPU SpMV ==========" << std::endl;
    
    // Create test vectors
    HBDIAVector<DataType> outputVector(std::vector<DataType>(localMatrix.getNumRows(), 0.0));
    
    // Run hybrid SpMV with timing
    auto start = std::chrono::high_resolution_clock::now();
    bool success = hbdiaSpMV(localMatrix, hbdiaVector, outputVector, false, false);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (!success) {
        std::cerr << myRank << "❌ SpMV failed!" << std::endl;
        return 1;
    }

    // gather global result
    std::vector<DataType> globalResult;
    if (!distributor.gatherVector(outputVector, globalResult)) {
        std::cerr << "gatherVector failed!" << std::endl;
    } else if (myRank == rootProcess) {
        //print global result
        std::cout << "gatherVector succeeded. Gathered vector size: " << globalResult.size() << std::endl;
    }
    
    if(myRank == 0){
        // Load reference matrix and data
        int numRows = globalMatrix.getNumRows();
        const auto& rowIndices = globalMatrix.getRowIndices();
        const auto& colIndices = globalMatrix.getColIndices();
        const auto& values = globalMatrix.getValues();
        // Compute reference using original COO data
        std::vector<DataType> reference(numRows, 0.0);
        for (size_t i = 0; i < rowIndices.size(); i++) {
            int row = rowIndices[i];
            int col = colIndices[i];
            DataType val = values[i];
            reference[row] += val * col;
        }
        // Check if reference has non-zero values
        DataType refSum = 0.0;
        for (int i = 0; i < numRows; i++) refSum += std::abs(reference[i]);
        std::cout << "Reference result sum: " << refSum << std::endl;
        if (refSum == 0.0) {
            std::cout << "⚠️ WARNING: Reference result is all zeros! Test may be meaningless." << std::endl;
        }
        // Compare results
        const DataType* result = globalResult.data();
        DataType resultSum = 0.0;
        for (int i = 0; i < numRows; i++) resultSum += std::abs(result[i]);
        std::cout << "Hybrid SpMV result sum: " << resultSum << std::endl;
        DataType maxError = 0.0;
        int errors = 0;
        for (int i = 0; i < numRows; i++) {
            DataType error = std::abs(result[i] - reference[i]);
            maxError = std::max(maxError, error);
            if (error > 1e-10) {
                errors++;
                std::cout << "Row " << i << ": got=" << result[i] << ", ref=" << reference[i] << std::endl;
            }
        }
        std::cout << "Max error: " << maxError << ", Error count: " << errors << std::endl;
        if (maxError < 1e-10) {
            std::cout << "✅ TEST PASSED!" << std::endl;
        } else {
            std::cout << "❌ TEST FAILED! Max error: " << maxError << ", Errors: " << errors << std::endl;
        }
    }

    distributor.getCommunicator().finalize();
    
    return 0;
}