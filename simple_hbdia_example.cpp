#include "include/Format/HBDIA.hpp"
#include "include/Format/HBDIAVector.hpp"
#include "include/Operations/HBDIASpMV.hpp"
#include "include/DataExchange/BasicDistributor.hpp"
#include "include/DataExchange/BasicExtractor.hpp"
#include "include/DataExchange/MPICommunicator.hpp"
#include "include/types.hpp"
#include <iostream>
#include <vector>
#include <memory>

using DataType = double;

int main(int argc, char* argv[]) {
    // Single GPU example
    {
        HBDIA<DataType> matrix;
        matrix.create3DStencil27Point(8, 8, 8, 0.0);
        matrix.convertToHBDIAFormat(32, 2);
        
        std::vector<DataType> inputData(matrix.getNumRows());
        for (int i = 0; i < matrix.getNumRows(); i++) {
            inputData[i] = static_cast<DataType>(i + 1);
        }
        
        HBDIAVector<DataType> vecX(inputData);
        HBDIAVector<DataType> vecY(std::vector<DataType>(matrix.getNumRows(), 0.0));
        
        bool execCOOCPU = true;
        bool execCOOGPU = false;
        
        hbdiaSpMV(matrix, vecX, vecY, execCOOCPU, execCOOGPU);
    }
    
    // Multi GPU example
    {
        auto communicator = std::make_unique<MPICommunicator<DataType>>();
        communicator->initialize(argc, argv);
        
        auto extractor = std::make_unique<BasicExtractor<DataType>>();
        BasicDistributor<DataType> distributor(std::move(communicator), std::move(extractor), 0);
        distributor.setPartitioningStrategy(HBDIAExtractor<DataType>::PartitioningStrategy::ROW_WISE);
        
        int myRank = distributor.getRank();
        
        HBDIA<DataType> globalMatrix, localMatrix;
        std::vector<DataType> globalVector, localVector;
        
        if (myRank == 0) {
            globalMatrix.create3DStencil27Point(16, 16, 16, 0.0);
            globalVector.resize(globalMatrix.getNumRows());
            for (int i = 0; i < globalMatrix.getNumRows(); i++) {
                globalVector[i] = static_cast<DataType>(i + 1);
            }
        }
        
        if (myRank == 0) {
            distributor.scatterMatrix(globalMatrix, localMatrix);
            distributor.scatterVector(globalVector, localVector);
        } else {
            distributor.receiveMatrix(localMatrix);
            distributor.receiveVector(localVector);
        }
        
        HBDIAVector<DataType> hbdiaVecX(localVector, localMatrix, myRank, distributor.getSize());
        distributor.exchangeData(localMatrix, hbdiaVecX);
        
        HBDIAVector<DataType> hbdiaVecY(std::vector<DataType>(localMatrix.getNumRows(), 0.0));
        
        hbdiaSpMV(localMatrix, hbdiaVecX, hbdiaVecY, true, false);
        
        std::vector<DataType> globalResult;
        distributor.gatherVector(hbdiaVecY, globalResult);
        
        distributor.getCommunicator().finalize();
    }
    
    return 0;
}
