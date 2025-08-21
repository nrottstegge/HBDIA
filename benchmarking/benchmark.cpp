#include "benchCusparse.hpp"
#include "benchHBDIA.hpp"
#include "../include/Format/HBDIA.hpp"
#include "../include/types.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>

using DataType = double;

std::string generateTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void saveCSV(const std::string& filename, const std::vector<std::string>& headers, 
             const std::vector<std::vector<std::string>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write headers
    for (size_t i = 0; i < headers.size(); i++) {
        file << headers[i];
        if (i < headers.size() - 1) file << ",";
    }
    file << "\n";
    
    // Write data rows
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); i++) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Data saved to: " << filename << std::endl;
}

bool createDirectories(const std::string& path) {
    std::string command = "mkdir -p " + path;
    return system(command.c_str()) == 0;
}

void saveMeasurementData(const std::string& folderPath, const std::string& algorithmName,
                        const std::vector<double>& measurements) {
    if (measurements.empty()) return;
    
    std::vector<std::string> headers = {"iteration", "time_ms"};
    std::vector<std::vector<std::string>> data;
    
    for (size_t i = 0; i < measurements.size(); i++) {
        data.push_back({
            std::to_string(i),
            std::to_string(measurements[i])
        });
    }
    
    saveCSV(folderPath + "/" + algorithmName + "_measurements.csv", headers, data);
}

void printStats(const std::string& name, const std::vector<double>& measurements) {
    if (measurements.empty()) return;
    
    double mean = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    double variance = 0.0;
    for (double m : measurements) {
        variance += (m - mean) * (m - mean);
    }
    variance /= measurements.size();
    double stddev = std::sqrt(variance);
    
    auto minmax = std::minmax_element(measurements.begin(), measurements.end());
    
    std::cout << name << " Results:" << std::endl;
    std::cout << "  Mean: " << mean << " ms" << std::endl;
    std::cout << "  Std:  " << stddev << " ms" << std::endl;
    std::cout << "  Min:  " << *minmax.first << " ms" << std::endl;
    std::cout << "  Max:  " << *minmax.second << " ms" << std::endl;
}

void printMatrixInfo(const HBDIA<DataType>& matrix, const std::string& name, double noise,
                     int nx, int ny, int nz, const std::string& timestamp) {
    std::cout << "=== Matrix Information ===" << std::endl;
    std::cout << "Name: " << name << std::endl;
    std::cout << "Rows: " << matrix.getNumRows() << std::endl;
    std::cout << "Cols: " << matrix.getNumCols() << std::endl;
    std::cout << "NNZ: " << matrix.getNumNonZeros() << std::endl;
    std::cout << "Noise: " << noise << std::endl;
    std::cout << "Number of Diagonals: " << matrix.getNumberDiagonals() << std::endl;
    
    if (matrix.isHBDIAFormat()) {
        std::cout << std::endl << "=== HBDIA Metrics ===" << std::endl;
        std::cout << "Number of Blocks: " << matrix.getNumBlocks() << std::endl;
        std::cout << "Block Width: " << matrix.getBlockWidth() << std::endl;
        std::cout << "Threshold: " << matrix.getThreshold() << std::endl;
        std::cout << "Max COO Entries: " << matrix.getMaxCooEntries() << std::endl;
        std::cout << "COO Fallback Entries: " << matrix.getCpuValues().size() << std::endl;
        
        // Print histograms
        const auto& histBlocks = matrix.getHistogramBlocks();
        const auto& histNnz = matrix.getHistogramNnz();
        
        std::cout << std::endl << "Histogram - Blocks with X diagonals:" << std::endl;
        for (size_t i = 0; i < std::min(histBlocks.size(), size_t(10)); i++) {
            if (histBlocks[i] > 0) {
                std::cout << "  " << i << " diagonals: " << histBlocks[i] << " blocks" << std::endl;
            }
        }
        
        std::cout << std::endl << "Histogram - NNZ in blocks with X diagonals:" << std::endl;
        for (size_t i = 0; i < std::min(histNnz.size(), size_t(10)); i++) {
            if (histNnz[i] > 0) {
                std::cout << "  " << i << " diagonals: " << histNnz[i] << " nnz" << std::endl;
            }
        }
        
        // Calculate storage costs
        std::cout << std::endl << "=== Storage Costs (bytes) ===" << std::endl;
        
        // HBDIA cost calculation
        int sumHistBlocks = 0;
        for(size_t i = 0; i < histBlocks.size(); i++) {
            sumHistBlocks += i * histBlocks[i];
        }
        size_t hbdiaCost = matrix.getHBDIAData().size() * sizeof(DataType) +           // actual HBDIA data storage
                          (matrix.getNumBlocks() + 1) * sizeof(int) +                    // block pointers
                          sumHistBlocks * sizeof(int) +                                  // offsets (one per diagonal per block)
                          matrix.getCpuValues().size() * sizeof(int) * 2 +                  // COO fallback row + col indices
                          matrix.getCpuValues().size() * sizeof(DataType);              // COO fallback values
        
        // COO cost
        size_t cooCost = matrix.getNumNonZeros() * sizeof(int) * 2 +  // row + col indices
                        matrix.getNumNonZeros() * sizeof(DataType);   // values
        
        // DIA cost
        size_t diaCost = matrix.getNumberDiagonals() * std::min(matrix.getNumRows(), matrix.getNumCols()) * sizeof(DataType) +  // diagonal data
                        matrix.getNumberDiagonals() * sizeof(int);  // offsets
        
        // SELL cost (if threshold == 1)
        size_t sellCost = 0;
        std::vector<int> nnzPerRow = matrix.getNnzPerRow();
        int max = 0;
        int numSlices = (matrix.getNumRows() + matrix.getBlockWidth() - 1) / matrix.getBlockWidth();
        
        for (int i = 0; i < nnzPerRow.size(); i++) {
            if(i % matrix.getBlockWidth() == 0 && i > 0) {
                // Add cost for the previous slice
                sellCost += max * matrix.getBlockWidth() * sizeof(DataType) +  // data
                           max * matrix.getBlockWidth() * sizeof(int);        // column indices
                max = 0; // Reset for new slice
            }
            max = std::max(max, nnzPerRow[i]);
        }
        // Add cost for the last slice
        sellCost += max * matrix.getBlockWidth() * sizeof(DataType) +  // data
                   max * matrix.getBlockWidth() * sizeof(int);        // column indices
        sellCost += numSlices * sizeof(int);                          // slice pointers
        
        // CSR cost
        size_t csrCost = matrix.getNumNonZeros() * sizeof(DataType) +      // values
                        matrix.getNumNonZeros() * sizeof(int) +           // column indices
                        (matrix.getNumRows() + 1) * sizeof(int);          // row pointers
        
        std::cout << "HBDIA: " << hbdiaCost << " bytes" << std::endl;
        std::cout << "COO:   " << cooCost << " bytes" << std::endl;
        std::cout << "DIA:   " << diaCost << " bytes" << std::endl;
        std::cout << "SELL:  " << sellCost << " bytes" << (matrix.getThreshold() == 1 ? "" : " (N/A, threshold != 1)") << std::endl;
        std::cout << "CSR:   " << csrCost << " bytes" << std::endl;
        
        // Print relative costs
        std::cout << std::endl << "Storage efficiency vs COO:" << std::endl;
        std::cout << "HBDIA: " << std::fixed << std::setprecision(2) << (double)hbdiaCost / cooCost << "x" << std::endl;
        std::cout << "DIA:   " << std::fixed << std::setprecision(2) << (double)diaCost / cooCost << "x" << std::endl;
        if (matrix.getThreshold() == 1) {
            std::cout << "SELL:  " << std::fixed << std::setprecision(2) << (double)sellCost / cooCost << "x" << std::endl;
        }
        std::cout << "CSR:   " << std::fixed << std::setprecision(2) << (double)csrCost / cooCost << "x" << std::endl;
        
        // Create data directory and save matrix info as CSV
        std::string folderName = "/users/nrottste/HBDIA/benchmarking/data/" + name + "_" + std::to_string(nx) + "_" + 
                                std::to_string(ny) + "_" + std::to_string(nz) + "_" + 
                                std::to_string(noise) + "_" + timestamp;
        createDirectories(folderName);
        
        // Prepare matrix info data for CSV
        std::vector<std::string> matrixHeaders = {
            "name", "nx", "ny", "nz", "noise", "rows", "cols", "nnz", "num_diagonals",
            "num_blocks", "block_width", "threshold", "max_coo_entries", "coo_fallback_entries",
            "hbdia_data_size", "hbdia_cost_bytes", "coo_cost_bytes", "dia_cost_bytes", 
            "sell_cost_bytes", "csr_cost_bytes", "hbdia_vs_coo", "dia_vs_coo", "csr_vs_coo", "sell_vs_coo"
        };
        
        std::vector<std::vector<std::string>> matrixData = {{
            name,
            std::to_string(nx),
            std::to_string(ny), 
            std::to_string(nz),
            std::to_string(noise),
            std::to_string(matrix.getNumRows()),
            std::to_string(matrix.getNumCols()),
            std::to_string(matrix.getNumNonZeros()),
            std::to_string(matrix.getNumberDiagonals()),
            std::to_string(matrix.getNumBlocks()),
            std::to_string(matrix.getBlockWidth()),
            std::to_string(matrix.getThreshold()),
            std::to_string(matrix.getMaxCooEntries()),
            std::to_string(matrix.getCpuValues().size()),
            std::to_string(matrix.getHBDIAData().size()),
            std::to_string(hbdiaCost),
            std::to_string(cooCost),
            std::to_string(diaCost),
            std::to_string(sellCost),
            std::to_string(csrCost),
            std::to_string((double)hbdiaCost / cooCost),
            std::to_string((double)diaCost / cooCost),
            std::to_string((double)csrCost / cooCost),
            std::to_string((double)sellCost / cooCost)
        }};
        
        saveCSV(folderName + "/matrix_info.csv", matrixHeaders, matrixData);
        
        // Save histogram data
        std::vector<std::string> histHeaders = {"diagonal_count", "blocks_with_diagonals", "nnz_in_blocks"};
        std::vector<std::vector<std::string>> histData;
                
        for (size_t i = 0; i < histBlocks.size(); i++) {
            if (histBlocks[i] > 0 || (i < histNnz.size() && histNnz[i] > 0)) {
                histData.push_back({
                    std::to_string(i),
                    std::to_string(histBlocks[i]),
                    i < histNnz.size() ? std::to_string(histNnz[i]) : "0"
                });
            }
        }
        
        saveCSV(folderName + "/histogram.csv", histHeaders, histData);
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Default configuration
    int nx = 32, ny = 32, nz = 32;
    double noise = 0.1;
    int threshold = 2;
    
    // Parse command line arguments
    if (argc >= 2) nx = std::atoi(argv[1]);
    if (argc >= 3) ny = std::atoi(argv[2]);
    if (argc >= 4) nz = std::atoi(argv[3]);
    if (argc >= 5) threshold = std::atoi(argv[4]);
    if (argc >= 6) noise = std::atof(argv[5]);
    
    // Execution flags (default values)
    bool execCOOCPU = false;
    bool execCOOGPU = true;
    bool unifiedMemory = false;
    bool unifiedMemory_malloc = false;
    bool unifiedMemory_managedMalloc = false;
    bool unifiedMemory_mallocOnNode = false;
    
    // Parse execution flags (optional arguments 7-13)
    if (argc >= 7) execCOOCPU = (std::atoi(argv[6]) != 0);
    if (argc >= 8) execCOOGPU = (std::atoi(argv[7]) != 0);
    if (argc >= 9) unifiedMemory = (std::atoi(argv[8]) != 0);
    if (argc >= 10) unifiedMemory_malloc = (std::atoi(argv[9]) != 0);
    if (argc >= 11) unifiedMemory_managedMalloc = (std::atoi(argv[10]) != 0);
    if (argc >= 12) unifiedMemory_mallocOnNode = (std::atoi(argv[11]) != 0);
    
    // Other configuration
    int block_width = 32;
    int max_coo_entries = INT_MAX;
    if (argc >= 13) max_coo_entries = std::atoi(argv[12]);
    int iteration = 0;
    if (argc >= 14) iteration = std::atoi(argv[13]);
    
    // Print usage if incorrect number of arguments
    if (argc > 14) {
        std::cout << "Usage: " << argv[0] << " [nx] [ny] [nz] [threshold] [noise] [execCOOCPU] [execCOOGPU] [unifiedMemory] [unifiedMemory_malloc] [unifiedMemory_managedMalloc] [unifiedMemory_mallocOnNode] [max_coo_entries] [seed]" << std::endl;
        std::cout << "Defaults: nx=32, ny=32, nz=32, threshold=16, noise=0.1, max_coo_entries=INT_MAX, iteration=0" << std::endl;
        std::cout << "Execution flags: execCOOCPU=0, execCOOGPU=1, others=0 (0=false, 1=true)" << std::endl;
        return 1;
    }

    // Generate timestamp for this run
    std::string timestamp = generateTimestamp();
    
    std::cout << "=== HBDIA vs cuSPARSE Benchmark ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Matrix: " << nx << "x" << ny << "x" << nz << " 27-point stencil" << std::endl;
    std::cout << "  Noise: " << noise << std::endl;
    std::cout << "  Threshold: " << threshold << std::endl;
    std::cout << "  Block Width: " << block_width << std::endl;
    std::cout << "  Timestamp: " << timestamp << std::endl << std::endl;
    
    // Create matrix
    HBDIA<DataType> matrix;
    matrix.create3DStencil27Point(nx, ny, nz, noise, iteration);
    
    // Determine execution mode based on flags
    ExecutionMode execMode = execCOOGPU ? ExecutionMode::GPU_COO : ExecutionMode::CPU_COO;
    matrix.convertToHBDIAFormat(block_width, threshold, max_coo_entries, true, execMode);
    
    // Create test vector
    std::vector<DataType> inputVector(matrix.getNumRows());
    for (int i = 0; i < matrix.getNumRows(); i++) {
        inputVector[i] = static_cast<DataType>(i);
    }
    
    std::cout << "Matrix size: " << matrix.getNumRows() << "x" << matrix.getNumCols() << std::endl;
    std::cout << "Non-zeros: " << matrix.getNumNonZeros() << std::endl << std::endl;
    
    // Print detailed matrix information and save to CSV
    printMatrixInfo(matrix, "3D27Stencil", noise, nx, ny, nz, timestamp);
    
    // Benchmark cuSPARSE
    std::vector<DataType> cusparseResult;
    std::vector<double> cusparseMeasurements;
    
    std::cout << "Running cuSPARSE benchmark..." << std::endl;
    benchCusparse(matrix.getRowIndices(), matrix.getColIndices(), matrix.getValues(),
                  inputVector, cusparseResult, matrix.getNumRows(), matrix.getNumCols(),
                  cusparseMeasurements);
    
    // Benchmark HBDIA
    std::vector<DataType> hbdiaResult;
    std::vector<double> hbdiaMeasurements;
    
    std::cout << "Running HBDIA benchmark..." << std::endl;
    benchHBDIA(matrix, inputVector, hbdiaResult, execCOOCPU, execCOOGPU, hbdiaMeasurements);
    
    // Print results
    std::cout << std::endl;
    printStats("cuSPARSE", cusparseMeasurements);
    std::cout << std::endl;
    printStats("HBDIA", hbdiaMeasurements);
    
    // Save measurement data to CSV
    std::string folderName = "/users/nrottste/HBDIA/benchmarking/data/3D27Stencil_" + std::to_string(nx) + "_" + 
                            std::to_string(ny) + "_" + std::to_string(nz) + "_" + 
                            std::to_string(noise) + "_" + timestamp;
    saveMeasurementData(folderName, "cusparse", cusparseMeasurements);
    saveMeasurementData(folderName, "hbdia", hbdiaMeasurements);
    
    // Performance comparison
    if (!cusparseMeasurements.empty() && !hbdiaMeasurements.empty()) {
        double cusparseMean = std::accumulate(cusparseMeasurements.begin() + 1, cusparseMeasurements.end(), 0.0) / cusparseMeasurements.size();
        double hbdiaMean = std::accumulate(hbdiaMeasurements.begin() + 1, hbdiaMeasurements.end(), 0.0) / hbdiaMeasurements.size();
        
        std::cout << std::endl << "=== Performance Comparison ===" << std::endl;
        std::cout << "Speedup: " << cusparseMean / hbdiaMean << "x" << std::endl;
    }
    
    // Accuracy check
    if (cusparseResult.size() == hbdiaResult.size()) {
        DataType maxError = 0.0;
        for (size_t i = 0; i < cusparseResult.size(); i++) {
            maxError = std::max(maxError, std::abs(cusparseResult[i] - hbdiaResult[i]));
        }
        std::cout << "Max error: " << maxError << std::endl;
        std::cout << (maxError < 1e-6 ? "✅ ACCURACY PASSED" : "❌ ACCURACY FAILED") << std::endl;
    }
    
    return 0;
}
