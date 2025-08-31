# HBDIA: Hybrid Blocked-DIA Sparse Matrix Library

A high-performance sparse matrix library for GPU-accelerated sparse matrix-vector multiplication (SpMV) with multi-GPU support.

## What is HBDIA Format?

HBDIA (Hybrid Blocked-DIA) format is a sparse matrix storage scheme that:
- **Organizes matrix into fixed-width blocks** for GPU-friendly memory access patterns  
- **Falls back to COO format** for irregular/sparse patterns (can execute on CPU with unified memory)
- **Supports multi-GPU** distributed execution via MPI

### Visual Example (dwt_59 from SuiteSparse Collection*)

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

*Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663

## Example Usage

See `simple_hbdia_example.cpp` for complete single-GPU and multi-GPU examples.

```cpp
// Single GPU example
HBDIA<double> matrix;
matrix.create3DStencil27Point(8, 8, 8, 0.0);
matrix.convertToHBDIAFormat(32, 2);

std::vector<double> inputData(matrix.getNumRows());
for (int i = 0; i < matrix.getNumRows(); i++) {
    inputData[i] = static_cast<double>(i + 1);
}

HBDIAVector<double> vecX(inputData);
HBDIAVector<double> vecY(std::vector<double>(matrix.getNumRows(), 0.0));

bool execCOOCPU = true;
bool execCOOGPU = false;

hbdiaSpMV(matrix, vecX, vecY, execCOOCPU, execCOOGPU);
```

## Build Requirements

Load required modules:
```bash
module load cmake/3.30.5 gcc/13.3.0 cuda/12.6.2 cray-mpich/8.1.30 nccl/2.22.3-1
```

Build:
```bash
mkdir build && cd build
cmake ..
make -j
```