#ifndef BENCH_CUSPARSE_HPP
#define BENCH_CUSPARSE_HPP

#include <vector>
#include "../include/types.hpp"

// Benchmark cuSPARSE SpMV implementation
// Returns execution times in milliseconds (excluding first warmup iteration)
void benchCusparse(const std::vector<int>& rowIndices, 
                   const std::vector<int>& colIndices,
                   const std::vector<DataType>& values,
                   const std::vector<DataType>& inputVector,
                   std::vector<DataType>& outputVector,
                   int numRows, int numCols,
                   std::vector<double>& measurements);

#endif // BENCH_CUSPARSE_HPP
