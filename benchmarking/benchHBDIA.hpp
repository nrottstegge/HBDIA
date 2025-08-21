#ifndef BENCH_HBDIA_HPP
#define BENCH_HBDIA_HPP

#include "../include/Format/HBDIA.hpp"
#include "../include/Format/HBDIAVector.hpp"
#include <vector>

using DataType = double;

// Benchmark HBDIA SpMV implementation  
// Returns execution times in milliseconds (excluding first warmup iteration)
void benchHBDIA(HBDIA<DataType>& matrix,
                const std::vector<DataType>& inputVector,
                std::vector<DataType>& outputVector,
                bool execCOOCPU, bool execCOOGPU,
                std::vector<double>& measurements);

#endif // BENCH_HBDIA_HPP
