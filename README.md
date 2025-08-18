# HBDIA: Heterogenous Blocked Diagonal Sparse Matrix Library

A high-performance sparse matrix library that combines GPU acceleration with CPU fallback for  sparse matrix-vector multiplication (SpMV) on heterogenous architectures.

## What is HBDIA Format?

HBDIA (Hybrid Block-Diagonal) format is a sparse matrix storage scheme that:
- **Organizes matrix into fixed-width blocks** for GPU-friendly memory access patterns
- **Stores frequent diagonal patterns efficiently** using block-diagonal storage
- **Falls back to COO format** for irregular/sparse patterns that don't fit the block structure

### Visual Example

**Original Dense Matrix (20×20):**
```
Dense Matrix Visualization:
Matrix size: 20 x 20
        0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19
  0: 5.00 2.00 0.50    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
  1: 1.05 5.10 2.05    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
  2:    . 1.10 5.20 2.10    .    .    .    .    .    .    .    .    .    .    . 0.30    .    .    .    .
  3:    .    . 1.15 5.30 2.15 0.56    .    .    .    .    .    .    .    .    .    .    .    .    .    .
  4:    .    .    . 1.20 5.40 2.20    .    .    .    .    .    .    .    .    .    .    .    .    .    .
  5:    .    .    .    . 1.25 5.50 2.25    .    .    .    .    .    .    .    .    .    .    . 0.30    .
  6:    .    .    .    .    . 1.30 5.60 2.30 0.62    .    .    .    .    .    .    .    .    .    .    .
  7:    .    .    .    .    .    . 1.35 5.70 2.35    .    .    .    .    .    .    .    .    .    .    .
  8:    .    .    .    .    .    .    . 1.40 5.80 2.40    .    .    .    .    .    .    .    .    .    .
  9:    .    .    .    .    .    .    .    . 1.45 5.90 2.45 0.68    .    .    .    .    .    .    .    .
 10:    .    .    . 0.30    .    .    .    .    . 1.50 6.00 2.50    .    .    .    .    .    .    .    .
 11:    .    .    .    .    .    .    .    .    .    . 1.55 6.10 2.55    .    .    .    .    .    .    .
 12:    .    .    .    .    .    .    .    .    .    .    . 1.60 6.20 2.60 0.74    .    .    .    .    .
 13:    .    .    .    .    .    .    .    .    .    .    .    . 1.65 6.30 2.65    .    .    .    .    .
 14:    .    .    .    .    .    .    .    .    .    .    .    .    . 1.70 6.40 2.70    .    .    .    .
 15:    .    .    .    .    .    .    .    . 0.30    .    .    .    .    . 1.75 6.50 2.75 0.80    .    .
 16:    .    .    .    .    .    .    .    .    .    .    .    .    .    .    . 1.80 6.60 2.80    .    .
 17:    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    . 1.85 6.70 2.85    .
 18:    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    . 1.90 6.80 0.30
 19:    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    . 1.95 6.90
 ```
**Normal DIA format**
```
Diagonal 0 (offset -7, length 20):                 .    .    . 0.30    .    .    .    . | 0.30    .    .    .    .    .    .    . |    .    .    .    .
Diagonal 1 (offset -1, length 20):              1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 | 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 | 1.85 1.90 1.95    .
Diagonal 2 (offset 0, length 20):               5.00 5.10 5.20 5.30 5.40 5.50 5.60 5.70 | 5.80 5.90 6.00 6.10 6.20 6.30 6.40 6.50 | 6.60 6.70 6.80 6.90
Diagonal 3 (offset 1, length 20):                  . 2.00 2.05 2.10 2.15 2.20 2.25 2.30 | 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 | 2.75 2.80 2.85 3.20
Diagonal 4 (offset 2, length 20):                  .    . 0.50    .    . 0.56    .    . | 0.62    .    . 0.68    .    . 0.74    . |    . 0.80    .    .
Diagonal 5 (offset 13, length 20):                 .    .    .    .    .    .    .    . |    .    .    .    .    .    .    . 0.30 |    .    . 0.30    .
```
**HBDIA Block Storage (Block width: 8, Threshold: 3):**
```
Block 0 (columns 0-7):
  Offsets:      -1 0 1 
  Offset -1:            1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40
  Offset 0:             5.00 5.10 5.20 5.30 5.40 5.50 5.60 5.70
  Offset 1:                . 2.00 2.05 2.10 2.15 2.20 2.25 2.30

Block 1 (columns 8-15):
  Offsets:      -1 0 1 2 
  Offset -1:            1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80
  Offset 0:             5.80 5.90 6.00 6.10 6.20 6.30 6.40 6.50
  Offset 1:             2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70
  Offset 2:             0.62    .    . 0.68    .    . 0.74    .

Block 2 (columns 16-23):
  Offsets:      -1 0 1 
  Offset -1:            1.85 1.90 1.95    .    .    .    .    .
  Offset 0:             6.60 6.70 6.80 6.90    .    .    .    .
  Offset 1:             2.75 2.80 2.85 3.20    .    .    .    .

Summary: 3 active blocks out of 3

CPU fallback entries:
Row     Col     Value
0       2       0.500000
2       15      0.300000
3       5       0.560000
5       18      0.300000
10      3       0.300000
15      8       0.300000
15      17      0.800000
```

## Quick Start

### Basic Usage

```cpp
#include "Format/HBDIA.hpp"
#include "Operations/HBDIASpMV.cuh"

// 1. Load matrix from file
HBDIA<double> matrix;
matrix.loadMTX("matrix.mtx");
matrix.convertToHBDIA();

// 2. Create vectors
std::vector<double> input(matrix.getNumRows(), 1.0);
HBDIAVector<double> inputVec(input);
HBDIAVector<double> outputVec(std::vector<double>(matrix.getNumRows(), 0.0));

// 3. Perform SpMV
bool success = hbdiaSpMV(matrix, inputVec, outputVec);
```

### Distributed Multi-GPU Usage

```cpp
#include "DataExchange/BasicDistributor.hpp"
#include "DataExchange/MPICommunicator.hpp"

// 1. Initialize MPI and create distributor
auto communicator = std::make_unique<MPICommunicator<double>>();
communicator->initialize(argc, argv);

auto extractor = std::make_unique<BasicExtractor<double>>();
BasicDistributor<double> distributor(std::move(communicator), std::move(extractor), 0);

// 2. Load and distribute matrix (on root process)
HBDIA<double> globalMatrix, localMatrix;
if (distributor.getRank() == 0) {
    globalMatrix.loadMTX("large_matrix.mtx");
    globalMatrix.convertToHBDIAFormat();
    distributor.scatterMatrix(globalMatrix, localMatrix);
} else {
    distributor.receiveMatrix(localMatrix);
}

// 3. Create and distribute vector
std::vector<double> localVector;
if (distributor.getRank() == 0) {
    std::vector<double> globalVector(globalMatrix.getNumRows());
    // Initialize global vector...
    distributor.scatterVector(globalVector, localVector);
} else {
    distributor.receiveVector(localVector);
}

// 4. Create distributed vector with communication buffers
HBDIAVector<double> hbdiaVector(localVector, localMatrix, 
                               distributor.getRank(), distributor.getSize());

// 5. Exchange boundary data
distributor.exchangeData(localMatrix, hbdiaVector);

// 6. Perform distributed SpMV
HBDIAVector<double> outputVector(std::vector<double>(localMatrix.getNumRows(), 0.0));
bool success = hbdiaSpMV(localMatrix, hbdiaVector, outputVector);

// 7. Gather results
std::vector<double> globalResult;
distributor.gatherVector(outputVector, globalResult);
```

## Build Instructions

```bash
# Clone and build
git clone <repository-url>
cd HBDIA
mkdir build && cd build
cmake ..
make -j
```