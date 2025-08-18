// HBDIAPrinter.cpp
#include "HBDIA.hpp"
#include "HBDIAPrinter.hpp"
#include "HBDIAVector.hpp"
#include <map>
#include <iomanip>
#include <cuda_runtime.h>

template <typename T>
void HBDIAPrinter<T>::print(const HBDIA<T>& matrix) {
    std::cout << "=========================================" << std::endl;
    std::cout << "     UNIFIED HBDIA MATRIX DEBUG INFO     " << std::endl;  
    std::cout << "=========================================" << std::endl;
    
    printMatrixHeader(matrix);
    printAvailableFormats(matrix);
    
    // If no formats available, return early
    if (!matrix.hasCOO && !matrix.hasDIA && !matrix.hasHBDIA) {
        std::cout << "No storage format available" << std::endl;
        return;
    }

    // Print all available matrix formats in detail
    std::cout << "\n========== MATRIX FORMATS ==========" << std::endl;
    
    if (matrix.hasCOO) {
        std::cout << "\n--- COO Format ---" << std::endl;
        printCOO(matrix);
    }

    if (matrix.hasHBDIA) {
        std::cout << "\n--- HBDIA Format ---" << std::endl;
        printHBDIA(matrix);
    }

    if (matrix.hasDIA) {
        std::cout << "\n--- DIA Format ---" << std::endl;
        printDIA(matrix);
    }

    // Print data ranges and process ownership information
    std::cout << "\n========== DATA RANGES & OWNERSHIP ==========" << std::endl;
    printDataRanges(matrix);
    
    // Print flattened GPU arrays for CUDA kernel parameters
    if (matrix.hasHBDIA) {
        printFlattenedGPUArrays(matrix);
    }
    
    // Print dense visualization if small enough
    std::cout << "\n========== MATRIX VISUALIZATION ==========" << std::endl;
    printDense(matrix);
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << "       END HBDIA MATRIX DEBUG INFO       " << std::endl;  
    std::cout << "=========================================" << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printCOO(const HBDIA<T>& matrix) {
    if (!matrix.hasCOO) {
        std::cout << "Matrix is not in COO format or is empty." << std::endl;
        return;
    }
    
    if (matrix.values.empty()) {
        std::cout << "Empty matrix" << std::endl;
        return;
    }
    
    std::cout << "\nCOO Format Storage:" << std::endl;
    std::cout << "Number of entries: " << matrix.values.size() << std::endl;
    std::cout << "Storage vectors:" << std::endl;
    std::cout << "  rowIndices: " << matrix.rowIndices.size() << " elements" << std::endl;
    std::cout << "  colIndices: " << matrix.colIndices.size() << " elements" << std::endl;
    std::cout << "  values: " << matrix.values.size() << " elements" << std::endl;
    
    // Show first 10 entries
    std::cout << "\nFirst " << std::min(static_cast<size_t>(10), matrix.values.size()) << " entries:" << std::endl;
    std::cout << "Row\tCol\tValue" << std::endl;
    std::cout << "---\t---\t-----" << std::endl;
    
    size_t printCount = std::min(static_cast<size_t>(10), matrix.values.size());
    for (size_t i = 0; i < printCount; ++i) {
        std::cout << matrix.rowIndices[i] << "\t" << matrix.colIndices[i] << "\t";
        if (matrix.values[i] == T(0)) {
            std::cout << "." << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(6) << matrix.values[i] << std::endl;
        }
    }
    
    if (matrix.values.size() > 10) {
        std::cout << "... and " << (matrix.values.size() - 10) << " more entries" << std::endl;
    }
    
    std::cout << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printDIA(const HBDIA<T>& matrix, int block_width) {
    if (!matrix.hasDIA) {
        std::cout << "Matrix is not in DIA format. Call convertToDIAFormat() first." << std::endl;
        return;
    }
    
    if (matrix.diagonals.empty()) {
        std::cout << "Empty DIA matrix" << std::endl;
        return;
    }
    
    std::cout << "\nDIA Format Storage:" << std::endl;
    std::cout << "Number of diagonals: " << matrix.diagonals.size() << std::endl;
    std::cout << "Matrix size: " << matrix.numRows << " x " << matrix.numCols << std::endl;
    
    if (block_width > 0) {
        std::cout << "Block width: " << block_width << std::endl;
    }
    
    std::cout << "\nDiagonal storage details:" << std::endl;
    
    // Calculate the maximum width needed for any value to ensure proper alignment
    int maxWidth = 4; // Minimum width for "0.00"
    for (size_t i = 0; i < matrix.diagonals.size(); ++i) {
        for (size_t j = 0; j < matrix.diagonals[i].size(); ++j) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << matrix.diagonals[i][j];
            maxWidth = std::max(maxWidth, static_cast<int>(oss.str().length()));
        }
    }
    
    // Print each diagonal vector (limit to first few for readability)
    const size_t maxDiagonalsToShow = 10; // Limit to first 10 diagonals for readability
    size_t diagonalsShown = 0;
    
    for (size_t i = 0; i < matrix.diagonals.size(); ++i) {
        // Only show details for first few diagonals
        if (diagonalsShown < maxDiagonalsToShow) {
            std::cout << "Diagonal " << i << " (offset " << matrix.offsets[i] << ", length " 
                      << matrix.diagonals[i].size() << "): \t\t";
            
            // Limit elements per diagonal to first 50 for readability
            const size_t maxElementsPerDiagonal = 50;
            size_t elementsToShow = std::min(maxElementsPerDiagonal, matrix.diagonals[i].size());
            
            if (block_width > 0) {
                // Print in blocked format with separators
                for (size_t j = 0; j < elementsToShow; ++j) {
                    if (j > 0 && j % block_width == 0) {
                        std::cout << " | ";
                    }
                    
                    if (matrix.diagonals.size() > 20) {
                        std::cout << std::setw(maxWidth) << (matrix.diagonals[i][j] != 0 ? "*" : ".");
                    } else {
                        if (matrix.diagonals[i][j] == T(0)) {
                            std::cout << std::setw(maxWidth) << ".";
                        } else {
                            std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << matrix.diagonals[i][j];
                        }
                    }
                    
                    if (j < elementsToShow - 1 && (j + 1) % block_width != 0) {
                        std::cout << " ";
                    }
                }
            } else {
                // Print normally without blocking
                for (size_t j = 0; j < elementsToShow; ++j) {
                    if (matrix.diagonals.size() > 20) {
                        std::cout << (matrix.diagonals[i][j] != 0 ? "*" : ".");
                    } else {
                        if (matrix.diagonals[i][j] == T(0)) {
                            std::cout << ".";
                        } else {
                            std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << matrix.diagonals[i][j];
                        }
                    }
                    if (j < elementsToShow - 1) {
                        std::cout << " ";
                    }
                }
            }
            
            // Show truncation message if diagonal was cut off
            if (matrix.diagonals[i].size() > maxElementsPerDiagonal) {
                std::cout << " ... (" << (matrix.diagonals[i].size() - maxElementsPerDiagonal) << " more elements)";
            }
            std::cout << std::endl;
            diagonalsShown++;
        }
    }
    
    if (matrix.diagonals.size() > maxDiagonalsToShow) {
        std::cout << "... and " << (matrix.diagonals.size() - maxDiagonalsToShow) << " more diagonals (details omitted for readability)" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Offset explanation:" << std::endl;
    std::cout << "  offset = 0:  Main diagonal" << std::endl;
    std::cout << "  offset > 0:  Super-diagonals (above main)" << std::endl;
    std::cout << "  offset < 0:  Sub-diagonals (below main)" << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printHBDIA(const HBDIA<T>& matrix) {
    if (!matrix.hasHBDIA) {
        std::cout << "Matrix is not in HBDIA format. Call convertToHBDIAFormat() first." << std::endl;
        return;
    }
    
    std::cout << "\nHBDIA Format Storage:" << std::endl;
    std::cout << "Block width: " << matrix.blockWidth << std::endl;
    std::cout << "Threshold: " << matrix.threshold << std::endl;
    std::cout << "Total data size: " << matrix.hbdiaData.size() << " elements" << std::endl;
    std::cout << "CPU fallback entries: " << matrix.cpuValues.size() << std::endl;

    // Compute max width for aligned printing
    int maxWidth = 4;
    for (size_t b = 0; b < matrix.ptrToBlock.size(); ++b) {
        if (matrix.ptrToBlock[b] != nullptr && !matrix.offsetsPerBlock[b].empty()) {
            for (size_t oi = 0; oi < matrix.offsetsPerBlock[b].size() && oi < 3; ++oi) {
                for (int lane = 0; lane < std::min(matrix.blockWidth, 8); ++lane) {
                    int dataIndex = oi * matrix.blockWidth + lane;
                    T value = matrix.ptrToBlock[b][dataIndex];
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << value;
                    maxWidth = std::max(maxWidth, static_cast<int>(oss.str().length()));
                }
            }
        }
    }
    
    std::cout << "\nBlock storage details:" << std::endl;
    int activeBlocks = 0;
    int blocksShown = 0;
    const int maxBlocksToShow = 5; // Limit to first 5 blocks for readability
    
    for (size_t b = 0; b < matrix.ptrToBlock.size(); ++b) {
        if (matrix.ptrToBlock[b] != nullptr && !matrix.offsetsPerBlock[b].empty()) {
            activeBlocks++;
            
            // Only show details for first few blocks
            if (blocksShown < maxBlocksToShow) {
                std::cout << "\nBlock " << b << " (columns " << b * matrix.blockWidth 
                          << "-" << (b + 1) * matrix.blockWidth - 1 << "):" << std::endl;
                std::cout << "  Offsets: \t";
                for (int offset : matrix.offsetsPerBlock[b]) {
                    std::cout << offset << " ";
                }
                std::cout << std::endl;
                
                // Show first few values of each offset in this block
                for (size_t oi = 0; oi < matrix.offsetsPerBlock[b].size() && oi < 10; ++oi) {
                    std::cout << "  Offset " << std::setw(6) << matrix.offsetsPerBlock[b][oi] << ": \t";
                    for (int lane = 0; lane < std::min(matrix.blockWidth, 8); ++lane) {
                        int dataIndex = oi * matrix.blockWidth + lane;
                        T value = matrix.ptrToBlock[b][dataIndex];
                        if (value == T(0)) {
                            std::cout << std::setw(maxWidth) << ".";
                        } else {
                            std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << value;
                        }
                        if (lane < std::min(matrix.blockWidth, 8) - 1) {
                            std::cout << " ";
                        }
                    }
                    if (matrix.blockWidth > 8) std::cout << "...";
                    std::cout << std::endl;
                }
                if (matrix.offsetsPerBlock[b].size() > 10) {
                    std::cout << "  ... and " << (matrix.offsetsPerBlock[b].size() - 10) << " more offsets" << std::endl;
                }
                blocksShown++;
            }
        }
    }
    
    if (activeBlocks > maxBlocksToShow) {
        std::cout << "\n... and " << (activeBlocks - maxBlocksToShow) << " more active blocks (details omitted for readability)" << std::endl;
    }
    
    std::cout << "\nSummary: " << activeBlocks << " active blocks out of " << matrix.ptrToBlock.size() << std::endl;
    
    if (!matrix.cpuValues.empty()) {
        std::cout << "\nCPU fallback entries (first 10):" << std::endl;
        std::cout << "Row\tCol\tValue" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), matrix.cpuValues.size()); ++i) {
            std::cout << matrix.cpuRowIndices[i] << "\t" << matrix.cpuColIndices[i] << "\t";
            if (matrix.cpuValues[i] == T(0)) {
                std::cout << "." << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(6) << matrix.cpuValues[i] << std::endl;
            }
        }
        if (matrix.cpuValues.size() > 10) {
            std::cout << "... and " << (matrix.cpuValues.size() - static_cast<size_t>(10)) << " more CPU entries" << std::endl;
        }
    }
}

template <typename T>
void HBDIAPrinter<T>::printDense(const HBDIA<T>& matrix) {
    if (!matrix.hasCOO && !matrix.hasDIA && !matrix.hasHBDIA) {
        std::cout << "No matrix data available to display." << std::endl;
        return;
    }
    
    std::cout << "\nDense Matrix Visualization (ASCII art):" << std::endl;
    std::cout << "Matrix size: " << matrix.numRows << " x " << matrix.numCols << std::endl;
    
    if (matrix.numRows > 100 || matrix.numCols > 100) {
        std::cout << "Matrix too large for ASCII art display (max 100x100). Showing structure only." << std::endl;
        std::cout << "Non-zero density: " << static_cast<double>(matrix.numNonZeros) / (matrix.numRows * matrix.numCols) * 100.0 << "%" << std::endl;
        return;
    }
    
    // Create a dense matrix representation initialized to zero
    std::vector<std::vector<T>> denseMatrix(matrix.numRows, std::vector<T>(matrix.numCols, T(0)));
    
    // Reconstruct from available format (prioritize order: COO, DIA, HBDIA)
    if (matrix.hasCOO) {
        // Reconstruct from COO format
        for (size_t i = 0; i < matrix.values.size(); ++i) {
            if (matrix.rowIndices[i] >= 0 && matrix.rowIndices[i] < matrix.numRows && 
                matrix.colIndices[i] >= 0 && matrix.colIndices[i] < matrix.numCols) {
                denseMatrix[matrix.rowIndices[i]][matrix.colIndices[i]] = matrix.values[i];
            }
        }
    } else if (matrix.hasDIA) {
        // Reconstruct from DIA format
        for (size_t d = 0; d < matrix.diagonals.size(); ++d) {
            int offset = matrix.offsets[d];
            for (size_t pos = 0; pos < matrix.diagonals[d].size(); ++pos) {
                int row, col;
                if (offset >= 0) {
                    row = static_cast<int>(pos);
                    col = row + offset;
                } else {
                    col = static_cast<int>(pos);
                    row = col - offset;
                }
                
                if (row >= 0 && row < matrix.numRows && col >= 0 && col < matrix.numCols && matrix.diagonals[d][pos] != T(0)) {
                    denseMatrix[row][col] = matrix.diagonals[d][pos];
                }
            }
        }
    } else if (matrix.hasHBDIA) {
        // Reconstruct from HBDIA format
        for (size_t b = 0; b < matrix.ptrToBlock.size(); ++b) {
            if (matrix.ptrToBlock[b] != nullptr && !matrix.offsetsPerBlock[b].empty()) {
                for (size_t oi = 0; oi < matrix.offsetsPerBlock[b].size(); ++oi) {
                    int offset = matrix.offsetsPerBlock[b][oi];
                    for (int lane = 0; lane < matrix.blockWidth; ++lane) {
                        int col = static_cast<int>(b) * matrix.blockWidth + lane;
                        int row = col + offset;
                        
                        if (row >= 0 && row < matrix.numRows && col >= 0 && col < matrix.numCols) {
                            int dataIndex = oi * matrix.blockWidth + lane;
                            T value = matrix.ptrToBlock[b][dataIndex];
                            if (value != T(0)) {
                                denseMatrix[row][col] = value;
                            }
                        }
                    }
                }
            }
        }
        
        // Add CPU fallback entries
        for (size_t i = 0; i < matrix.cpuValues.size(); ++i) {
            int row = matrix.cpuRowIndices[i];
            int col = matrix.cpuColIndices[i];
            if (row >= 0 && row < matrix.numRows && col >= 0 && col < matrix.numCols) {
                denseMatrix[row][col] = matrix.cpuValues[i];
            }
        }
    }
    
    // Calculate the maximum width needed for any value to ensure proper alignment
    int maxWidth = 4; // Minimum width for "0.00" and "."
    if (matrix.numCols <= 20) {
        for (int i = 0; i < matrix.numRows; ++i) {
            for (int j = 0; j < matrix.numCols; ++j) {
                if (denseMatrix[i][j] != T(0)) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << denseMatrix[i][j];
                    maxWidth = std::max(maxWidth, static_cast<int>(oss.str().length()));
                }
            }
        }
    }
    
    // Print column headers for small matrices
    if (matrix.numCols <= 20) {
        std::cout << "     ";
        for (int j = 0; j < matrix.numCols; ++j) {
            std::cout << std::setw(maxWidth) << j;
            if (j < matrix.numCols - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }
    
    // Print the matrix
    for (int i = 0; i < matrix.numRows; ++i) {
        if (matrix.numCols <= 20) {
            std::cout << std::setw(3) << i << ": ";
        }
        
        for (int j = 0; j < matrix.numCols; ++j) {
            if (denseMatrix[i][j] == T(0)) {
                if (matrix.numCols <= 20) {
                    std::cout << std::setw(maxWidth) << ".";
                } else {
                    std::cout << ".";
                }
            } else {
                if (matrix.numCols <= 20) {
                    std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << denseMatrix[i][j];
                } else {
                    std::cout << "*";
                }
            }
            if (j < matrix.numCols - 1 && matrix.numCols <= 20) std::cout << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printMatrixHeader(const HBDIA<T>& matrix) {
    std::cout << "Matrix size: " << matrix.numRows << " x " << matrix.numCols
              << " with " << matrix.numNonZeros << " non-zeros" << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printAvailableFormats(const HBDIA<T>& matrix) {
    std::vector<std::string> formats;

    if (matrix.hasCOO)   formats.push_back("COO");
    if (matrix.hasHBDIA) formats.push_back("HBDIA");
    if (matrix.hasDIA)   formats.push_back("DIA");

    if (!formats.empty()) {
        std::cout << "Storage formats: ";
        for (size_t i = 0; i < formats.size(); ++i) {
            std::cout << formats[i];
            if (i + 1 < formats.size()) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void HBDIAPrinter<T>::printDataRanges(const HBDIA<T>& matrix) {
    std::cout << "=== PARTIAL MATRIX DATA RANGES ===" << std::endl;
    
    // Check if this is a partial matrix
    if (!matrix.isPartialMatrix()) {
        std::cout << "Matrix is not configured as a partial matrix" << std::endl;
        return;
    }
    
    // Print basic partial matrix information
    std::cout << "Matrix Type: Partial Matrix" << std::endl;
    std::cout << "Matrix dimensions: " << matrix.getNumRows() << "x" << matrix.getNumCols() << std::endl;
    std::cout << "Local non-zeros: " << matrix.getNumNonZeros() << std::endl;
    
    // Print data ranges needed by this process
    const auto& dataRanges = matrix.getDataRanges();
    std::cout << "\nData ranges needed by this process:" << std::endl;
    
    if (dataRanges.empty()) {
        std::cout << "  No data ranges analyzed yet" << std::endl;
        std::cout << "  Note: Data ranges are automatically analyzed when matrix is created with partialMatrix=true" << std::endl;
        return;
    }
    
    std::cout << "  Total ranges: " << dataRanges.size() << std::endl;
    
    int totalElements = 0;
    for (size_t i = 0; i < dataRanges.size(); ++i) {
        auto [start, end] = dataRanges[i];
        int rangeSize = end - start;
        totalElements += rangeSize;
    }
    
    std::cout << "  Total elements needed: " << totalElements << std::endl;
    
    // Show only first few ranges and summary
    const size_t maxRangesToShow = 3;
    if (!dataRanges.empty()) {
        std::cout << "  Sample ranges (first " << std::min(maxRangesToShow, dataRanges.size()) << "):" << std::endl;
        for (size_t i = 0; i < std::min(maxRangesToShow, dataRanges.size()); ++i) {
            auto [start, end] = dataRanges[i];
            int rangeSize = end - start;
            std::cout << "    [" << start << ", " << end << ") -> " << rangeSize << " elements" << std::endl;
        }
        if (dataRanges.size() > maxRangesToShow) {
            std::cout << "    ... and " << (dataRanges.size() - maxRangesToShow) << " more ranges" << std::endl;
        }
    }
    
    std::cout << "=================================" << std::endl;

    // Print processDataRanges for all processes if available
    if (!matrix.partialMatrixMetadata_.processDataRanges.empty()) {
        std::cout << "\n=== DATA RANGES PROVIDED BY EACH PROCESS ===" << std::endl;
        for (size_t i = 0; i < matrix.partialMatrixMetadata_.processDataRanges.size(); ++i) {
            std::cout << "Rank " << i << " will provide:" << std::endl;
            for (const auto& tuple : matrix.partialMatrixMetadata_.processDataRanges[i]) {
                std::cout << "  [" << std::get<0>(tuple) << ", " << std::get<1>(tuple) << ") -> " << (std::get<1>(tuple) - std::get<0>(tuple)) << " elements" << std::endl;
            }
        }
        std::cout << "=================================" << std::endl;
    }else{
        std::cout << "No processDataRanges available. This may be due to the matrix not being configured as a partial matrix." << std::endl;
    }
}

template <typename T>
void HBDIAPrinter<T>::printVector(const HBDIAVector<T>& vector, const std::string& vectorName) {
    std::cout << "=================================" << std::endl;
    std::cout << vectorName << " Debug Information" << std::endl;
    std::cout << "=================================" << std::endl;
    
    printVectorBufferInfo(vector);
    printVectorMemoryLayout(vector);
    printVectorData(vector);
    
    std::cout << "=================================" << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printVectorMemoryLayout(const HBDIAVector<T>& vector) {
    std::cout << "\nMemory Layout:" << std::endl;
    std::cout << "  Total unified memory size: " << vector.getTotalSize() << " elements" << std::endl;
    std::cout << "  Layout: [left_buffer|local_data|right_buffer]" << std::endl;
    std::cout << "           " << vector.size_recv_left_ << " + " << vector.getLocalSize() 
              << " + " << vector.size_recv_right_ << " = " << vector.getTotalSize() << std::endl;
    
    std::cout << "\nPointer Status:" << std::endl;
    std::cout << "  unified_data_ptr_: " << (vector.unified_data_ptr_ ? "allocated" : "null") << std::endl;
    std::cout << "  unified_left_ptr_: " << (vector.unified_left_ptr_ ? "set" : "null") << std::endl;
    std::cout << "  unified_local_ptr_: " << (vector.unified_local_ptr_ ? "set" : "null") << std::endl;
    std::cout << "  unified_right_ptr_: " << (vector.unified_right_ptr_ ? "set" : "null") << std::endl;
    
    std::cout << "\nDevice Memory Status:" << std::endl;
    std::cout << "  data_ptr_d_: " << (vector.data_ptr_d_ ? "allocated" : "null") << std::endl;
    std::cout << "  left_ptr_d_: " << (vector.left_ptr_d_ ? "allocated" : "null") << std::endl;
    std::cout << "  local_ptr_d_: " << (vector.local_ptr_d_ ? "allocated" : "null") << std::endl;
    std::cout << "  right_ptr_d_: " << (vector.right_ptr_d_ ? "allocated" : "null") << std::endl;
    std::cout << "  cpu_result_ptr_d_: " << (vector.cpu_result_ptr_d_ ? "allocated" : "null") << std::endl;
    
    std::cout << "\nHost Memory Status:" << std::endl;
    std::cout << "  data_ptr_h_: " << (vector.data_ptr_h_ ? "allocated" : "null") << std::endl;
    std::cout << "  left_ptr_h_: " << (vector.left_ptr_h_ ? "allocated" : "null") << std::endl;
    std::cout << "  local_ptr_h_: " << (vector.local_ptr_h_ ? "allocated" : "null") << std::endl;
    std::cout << "  right_ptr_h_: " << (vector.right_ptr_h_ ? "allocated" : "null") << std::endl;
    std::cout << "  cpu_result_ptr_h_: " << (vector.cpu_result_ptr_h_ ? "allocated" : "null") << std::endl;
    
    if (vector.unified_data_ptr_) {
        std::cout << "\nUnified Memory Addresses:" << std::endl;
        std::cout << "  Base address: " << static_cast<void*>(vector.unified_data_ptr_) << std::endl;
        if (vector.unified_left_ptr_) {
            std::cout << "  Left buffer: " << static_cast<void*>(vector.unified_left_ptr_) << std::endl;
        }
        if (vector.unified_local_ptr_) {
            std::cout << "  Local data: " << static_cast<void*>(vector.unified_local_ptr_) << std::endl;
        }
        if (vector.unified_right_ptr_) {
            std::cout << "  Right buffer: " << static_cast<void*>(vector.unified_right_ptr_) << std::endl;
        }
    }
    
    if (vector.data_ptr_d_) {
        std::cout << "\nDevice Memory Addresses:" << std::endl;
        std::cout << "  Base device address: " << static_cast<void*>(vector.data_ptr_d_) << std::endl;
        if (vector.left_ptr_d_) {
            std::cout << "  Left buffer device: " << static_cast<void*>(vector.left_ptr_d_) << std::endl;
        }
        if (vector.local_ptr_d_) {
            std::cout << "  Local data device: " << static_cast<void*>(vector.local_ptr_d_) << std::endl;
        }
        if (vector.right_ptr_d_) {
            std::cout << "  Right buffer device: " << static_cast<void*>(vector.right_ptr_d_) << std::endl;
        }
        if (vector.cpu_result_ptr_d_) {
            std::cout << "  CPU result device: " << static_cast<void*>(vector.cpu_result_ptr_d_) << std::endl;
        }
    }
    
    if (vector.data_ptr_h_) {
        std::cout << "\nHost Memory Addresses:" << std::endl;
        std::cout << "  Base host address: " << static_cast<void*>(vector.data_ptr_h_) << std::endl;
        if (vector.left_ptr_h_) {
            std::cout << "  Left buffer host: " << static_cast<void*>(vector.left_ptr_h_) << std::endl;
        }
        if (vector.local_ptr_h_) {
            std::cout << "  Local data host: " << static_cast<void*>(vector.local_ptr_h_) << std::endl;
        }
        if (vector.right_ptr_h_) {
            std::cout << "  Right buffer host: " << static_cast<void*>(vector.right_ptr_h_) << std::endl;
        }
        if (vector.cpu_result_ptr_h_) {
            std::cout << "  CPU result host: " << static_cast<void*>(vector.cpu_result_ptr_h_) << std::endl;
        }
    }
}

template <typename T>
void HBDIAPrinter<T>::printVectorBufferInfo(const HBDIAVector<T>& vector) {
    std::cout << "\nBuffer Information:" << std::endl;
    std::cout << "  Left buffer size: " << vector.size_recv_left_ << " elements" << std::endl;
    std::cout << "  Local vector size: " << vector.getLocalSize() << " elements" << std::endl;
    std::cout << "  Right buffer size: " << vector.size_recv_right_ << " elements" << std::endl;
    std::cout << "  Total size: " << vector.getTotalSize() << " elements" << std::endl;
    
    // Calculate memory usage
    size_t totalMemory = vector.getTotalSize() * sizeof(T);
    std::cout << "  Total memory usage: " << totalMemory << " bytes";
    if (totalMemory >= 1024 * 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2) << totalMemory / (1024.0 * 1024.0) << " MB)";
    } else if (totalMemory >= 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2) << totalMemory / 1024.0 << " KB)";
    }
    std::cout << std::endl;
}

template <typename T>
void HBDIAPrinter<T>::printVectorData(const HBDIAVector<T>& vector, size_t maxElements) {
    const auto& localVec = vector.getLocalVector();
    
    std::cout << "\nLocal Vector Data (original std::vector):" << std::endl;
    if (localVec.empty()) {
        std::cout << "  (empty - data moved to managed memory)" << std::endl;
    } else {
        size_t printCount = std::min(maxElements, localVec.size());
        std::cout << "  First " << printCount << " elements:" << std::endl;
        std::cout << "  [";
        
        for (size_t i = 0; i < printCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << localVec[i];
        }
        
        if (localVec.size() > maxElements) {
            std::cout << ", ... (" << (localVec.size() - maxElements) << " more)";
        }
        std::cout << "]" << std::endl;
    }
    
    // Show unified memory content if available
    if (vector.unified_data_ptr_ && vector.getTotalSize() > 0) {
        std::cout << "\nUnified Memory Buffer Content:" << std::endl;
        std::cout << "  Total size: " << vector.getTotalSize() << " elements" << std::endl;
        std::cout << "  Layout: [left_buffer(" << vector.size_recv_left_ << ") | local_data(" << vector.getLocalSize() 
                  << ") | right_buffer(" << vector.size_recv_right_ << ")]" << std::endl;
        
        // Print Left Buffer
        if (vector.size_recv_left_ > 0 && vector.unified_left_ptr_) {
            size_t leftPrintCount = std::min(maxElements, vector.size_recv_left_);
            std::cout << "\n  Left Buffer (" << vector.size_recv_left_ << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < leftPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.unified_left_ptr_[i];
            }
            if (vector.size_recv_left_ > maxElements) {
                std::cout << ", ... (" << (vector.size_recv_left_ - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "\n  Left Buffer: (empty)" << std::endl;
        }
        
        // Print Local Data Buffer
        if (vector.getLocalSize() > 0 && vector.unified_local_ptr_) {
            size_t localPrintCount = std::min(maxElements, vector.getLocalSize());
            std::cout << "\n  Local Data (" << vector.getLocalSize() << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < localPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.unified_local_ptr_[i];
            }
            if (vector.getLocalSize() > maxElements) {
                std::cout << ", ... (" << (vector.getLocalSize() - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "\n  Local Data: (empty)" << std::endl;
        }
        
        // Print Right Buffer
        if (vector.size_recv_right_ > 0 && vector.unified_right_ptr_) {
            size_t rightPrintCount = std::min(maxElements, vector.size_recv_right_);
            std::cout << "\n  Right Buffer (" << vector.size_recv_right_ << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < rightPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.unified_right_ptr_[i];
            }
            if (vector.size_recv_right_ > maxElements) {
                std::cout << ", ... (" << (vector.size_recv_right_ - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "\n  Right Buffer: (empty)" << std::endl;
        }
        
        // Show complete unified buffer with section markers for reference
        std::cout << "\n  Complete Unified Buffer (first " << std::min(static_cast<size_t>(maxElements), vector.getTotalSize()) << " elements):" << std::endl;
        size_t totalPrintCount = std::min(static_cast<size_t>(maxElements), vector.getTotalSize());
        std::cout << "    [";
        
        for (size_t i = 0; i < totalPrintCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << vector.unified_data_ptr_[i];
            
            // Add section markers to show buffer boundaries
            if (i == vector.size_recv_left_ - 1 && vector.size_recv_left_ > 0 && i < totalPrintCount - 1) {
                std::cout << " |"; // Mark end of left buffer
            } else if (i == vector.size_recv_left_ + vector.getLocalSize() - 1 && vector.size_recv_right_ > 0 && i < totalPrintCount - 1) {
                std::cout << " |"; // Mark end of local data
            }
        }
        
        if (vector.getTotalSize() > maxElements) {
            std::cout << ", ... (" << (vector.getTotalSize() - maxElements) << " more)";
        }
        std::cout << "]" << std::endl;
        
        // Show the local portion consistency check
        if (vector.unified_local_ptr_ && vector.getLocalSize() > 0 && !localVec.empty()) {
            bool matches = true;
            size_t localCheckCount = std::min(static_cast<size_t>(maxElements), vector.getLocalSize());
            localCheckCount = std::min(localCheckCount, localVec.size()); // Ensure we don't exceed localVec size
            
            for (size_t i = 0; i < localCheckCount; ++i) {
                if (localVec[i] != vector.unified_local_ptr_[i]) {
                    matches = false;
                    break;
                }
            }
            std::cout << "  Local data consistency: " << (matches ? "✓ matches local vector" : "✗ differs from local vector") << std::endl;
        } else if (vector.unified_local_ptr_ && vector.getLocalSize() > 0 && localVec.empty()) {
            std::cout << "  Local data consistency: (local vector empty - data moved to unified memory)" << std::endl;
        }
    }
    
    // Show separate host memory content if available
    if (vector.data_ptr_h_ && vector.getTotalSize() > 0) {
        std::cout << "\nHost Memory Buffer Content:" << std::endl;
        std::cout << "  Total size: " << vector.getTotalSize() << " elements" << std::endl;
        std::cout << "  Layout: [left_buffer(" << vector.size_recv_left_ << ") | local_data(" << vector.getLocalSize() 
                  << ") | right_buffer(" << vector.size_recv_right_ << ")]" << std::endl;
        
        // Print Host Left Buffer
        if (vector.size_recv_left_ > 0 && vector.left_ptr_h_) {
            size_t leftPrintCount = std::min(maxElements, vector.size_recv_left_);
            std::cout << "\n  Host Left Buffer (" << vector.size_recv_left_ << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < leftPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.left_ptr_h_[i];
            }
            if (vector.size_recv_left_ > maxElements) {
                std::cout << ", ... (" << (vector.size_recv_left_ - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        }
        
        // Print Host Local Data Buffer
        if (vector.getLocalSize() > 0 && vector.local_ptr_h_) {
            size_t localPrintCount = std::min(maxElements, vector.getLocalSize());
            std::cout << "\n  Host Local Data (" << vector.getLocalSize() << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < localPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.local_ptr_h_[i];
            }
            if (vector.getLocalSize() > maxElements) {
                std::cout << ", ... (" << (vector.getLocalSize() - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        }
        
        // Print Host Right Buffer
        if (vector.size_recv_right_ > 0 && vector.right_ptr_h_) {
            size_t rightPrintCount = std::min(maxElements, vector.size_recv_right_);
            std::cout << "\n  Host Right Buffer (" << vector.size_recv_right_ << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < rightPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.right_ptr_h_[i];
            }
            if (vector.size_recv_right_ > maxElements) {
                std::cout << ", ... (" << (vector.size_recv_right_ - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        }
        
        // Show host CPU results if available
        if (vector.cpu_result_ptr_h_) {
            size_t resultPrintCount = std::min(maxElements, vector.getLocalSize());
            std::cout << "\n  Host CPU Results (" << vector.getLocalSize() << " elements):" << std::endl;
            std::cout << "    [";
            for (size_t i = 0; i < resultPrintCount; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << vector.cpu_result_ptr_h_[i];
            }
            if (vector.getLocalSize() > maxElements) {
                std::cout << ", ... (" << (vector.getLocalSize() - maxElements) << " more)";
            }
            std::cout << "]" << std::endl;
        }
    }
}

template <typename T>
void HBDIAPrinter<T>::printFlattenedGPUArrays(const HBDIA<T>& matrix) {
    std::cout << "\n=== FLATTENED GPU ARRAYS (CUDA Kernel Parameters) ===" << std::endl;
    
    if (!matrix.hasHBDIA) {
        std::cout << "Matrix not in HBDIA format - no flattened GPU arrays available" << std::endl;
        return;
    }
    
    // Check if GPU data is prepared
    if (matrix.getHBDIADataDevice() == nullptr) {
        std::cout << "GPU data not prepared - call matrix.prepareForGPU() first" << std::endl;
        return;
    }
    
    std::cout << "These are the actual flattened data structures stored on GPU device:" << std::endl;
    std::cout << std::endl;
    
    // 1. Copy and display hbdiaData from device
    const auto& hbdiaData = matrix.getHBDIAData();
    if (!hbdiaData.empty() && matrix.getHBDIADataDevice() != nullptr) {
        std::cout << "1. hbdiaData_d_ (" << hbdiaData.size() << " elements):" << std::endl;
        std::vector<T> hbdiaData_h(hbdiaData.size());
        cudaMemcpy(hbdiaData_h.data(), matrix.getHBDIADataDevice(), hbdiaData.size() * sizeof(T), cudaMemcpyDeviceToHost);
        
        size_t showCount = std::min(static_cast<size_t>(20), hbdiaData_h.size());
        std::cout << "   [";
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(3) << hbdiaData_h[i];
        }
        if (hbdiaData_h.size() > 20) {
            std::cout << ", ... (" << (hbdiaData_h.size() - 20) << " more)";
        }
        std::cout << "]" << std::endl;
    }
    
    // 2. Copy and display flattenedOffsets from device
    const auto& offsetsPerBlock = matrix.getOffsetsPerBlock();
    size_t totalOffsets = 0;
    for (const auto& blockOffsets : offsetsPerBlock) {
        totalOffsets += blockOffsets.size();
    }
    
    if (totalOffsets > 0 && matrix.getFlattenedOffsetsDevice() != nullptr) {
        std::cout << "\n2. flattenedOffsets_d_ (" << totalOffsets << " elements):" << std::endl;
        std::vector<int> flattenedOffsets_h(totalOffsets);
        cudaMemcpy(flattenedOffsets_h.data(), matrix.getFlattenedOffsetsDevice(), totalOffsets * sizeof(int), cudaMemcpyDeviceToHost);
        
        size_t showCount = std::min(static_cast<size_t>(30), flattenedOffsets_h.size());
        std::cout << "   [";
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << flattenedOffsets_h[i];
        }
        if (flattenedOffsets_h.size() > 30) {
            std::cout << ", ... (" << (flattenedOffsets_h.size() - 30) << " more)";
        }
        std::cout << "]" << std::endl;
    }
    
    // 3. Copy and display blockStartIndices from device
    size_t numBlocks = 0;
    for (const auto& blockOffsets : offsetsPerBlock) {
        if (!blockOffsets.empty()) {
            numBlocks++;
        }
    }
    
    if (numBlocks > 0 && matrix.getBlockStartIndicesDevice() != nullptr) {
        std::cout << "\n3. blockStartIndices_d_ (" << numBlocks << " elements):" << std::endl;
        std::vector<int> blockStartIndices_h(numBlocks);
        cudaMemcpy(blockStartIndices_h.data(), matrix.getBlockStartIndicesDevice(), numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        
        size_t showCount = std::min(static_cast<size_t>(10), blockStartIndices_h.size());
        std::cout << "   [";
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << blockStartIndices_h[i];
        }
        if (blockStartIndices_h.size() > 10) {
            std::cout << ", ... (" << (blockStartIndices_h.size() - 10) << " more)";
        }
        std::cout << "]" << std::endl;
    }
    
    // 4. Copy and display blockSizes from device
    if (numBlocks > 0 && matrix.getBlockSizesDevice() != nullptr) {
        std::cout << "\n4. blockSizes_d_ (" << numBlocks << " elements):" << std::endl;
        std::vector<int> blockSizes_h(numBlocks);
        cudaMemcpy(blockSizes_h.data(), matrix.getBlockSizesDevice(), numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        
        size_t showCount = std::min(static_cast<size_t>(10), blockSizes_h.size());
        std::cout << "   [";
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << blockSizes_h[i];
        }
        if (blockSizes_h.size() > 10) {
            std::cout << ", ... (" << (blockSizes_h.size() - 10) << " more)";
        }
        std::cout << "]" << std::endl;
    }
    
    // 5. Copy and display flattenedVectorOffsets from device (if available)
    if (totalOffsets > 0 && matrix.getFlattenedVectorOffsetsDevice() != nullptr) {
        std::cout << "\n5. flattenedVectorOffsets_d_ (" << totalOffsets << " elements):" << std::endl;
        std::vector<int> flattenedVectorOffsets_h(totalOffsets);
        cudaMemcpy(flattenedVectorOffsets_h.data(), matrix.getFlattenedVectorOffsetsDevice(), totalOffsets * sizeof(int), cudaMemcpyDeviceToHost);
        
        size_t showCount = std::min(static_cast<size_t>(30), flattenedVectorOffsets_h.size());
        std::cout << "   [";
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << flattenedVectorOffsets_h[i];
        }
        if (flattenedVectorOffsets_h.size() > 30) {
            std::cout << ", ... (" << (flattenedVectorOffsets_h.size() - 30) << " more)";
        }
        std::cout << "]" << std::endl;
        
    } else {
        std::cout << "\n5. flattenedVectorOffsets_d_: not allocated or not available" << std::endl;
    }
    
    std::cout << "\nNote: These are the actual contiguous GPU device memory arrays used by CUDA kernels" << std::endl;
    std::cout << "===============================================" << std::endl;
}

// Explicit template instantiations for common types
template class HBDIAPrinter<double>;
template class HBDIAPrinter<float>;
template class HBDIAPrinter<int>;

