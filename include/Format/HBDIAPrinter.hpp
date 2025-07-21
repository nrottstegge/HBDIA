// HBDIAPrinter.hpp
#ifndef HBDIAPRINTER_HPP
#define HBDIAPRINTER_HPP

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// Forward declarations
template <typename T>
class HBDIA;

template <typename T>
class HBDIAVector;

/**
 * @brief Printer utility class for HBDIA matrices
 * 
 * This class provides all printing functionality for HBDIA matrices,
 * including format-specific printing and dense visualization.
 * It's designed as a friend class to access private members of HBDIA.
 */
template <typename T>
class HBDIAPrinter {
public:
    /**
     * @brief Print comprehensive matrix information including all available formats
     * @param matrix Reference to the HBDIA matrix to print
     */
    static void print(const HBDIA<T>& matrix);
    
    /**
     * @brief Print COO format details
     * @param matrix Reference to the HBDIA matrix to print
     */
    static void printCOO(const HBDIA<T>& matrix);
    
    /**
     * @brief Print DIA format details
     * @param matrix Reference to the HBDIA matrix to print
     * @param block_width Optional block width for sliced format display (default: 0)
     */
    static void printDIA(const HBDIA<T>& matrix, int block_width = 0);
    
    /**
     * @brief Print HBDIA format details
     * @param matrix Reference to the HBDIA matrix to print
     */
    static void printHBDIA(const HBDIA<T>& matrix);
    
    /**
     * @brief Print dense matrix visualization (ASCII art)
     * @param matrix Reference to the HBDIA matrix to print
     */
    static void printDense(const HBDIA<T>& matrix);
    
    /**
     * @brief Print partial matrix data ranges information
     * @param matrix Reference to the HBDIA matrix to print
     */
    static void printDataRanges(const HBDIA<T>& matrix);

    /**
     * @brief Print HBDIAVector debug information
     * @param vector Reference to the HBDIAVector to print
     * @param vectorName Optional name for the vector (default: "HBDIAVector")
     */
    static void printVector(const HBDIAVector<T>& vector, const std::string& vectorName = "HBDIAVector");
    
    /**
     * @brief Print HBDIAVector memory layout information
     * @param vector Reference to the HBDIAVector to print
     */
    static void printVectorMemoryLayout(const HBDIAVector<T>& vector);
    
    /**
     * @brief Print HBDIAVector buffer sizes and pointer status
     * @param vector Reference to the HBDIAVector to print
     */
    static void printVectorBufferInfo(const HBDIAVector<T>& vector);
    
    /**
     * @brief Print local vector data (first N elements)
     * @param vector Reference to the HBDIAVector to print
     * @param maxElements Maximum number of elements to print (default: 10)
     */
    static void printVectorData(const HBDIAVector<T>& vector, size_t maxElements = 10);

    /**
     * @brief Print GPU buffer status and flattened data structures
     * @param matrix Reference to the HBDIA matrix to print
     */
    static void printGPUBufferStatus(const HBDIA<T>& matrix);

    /**
     * @brief Print all flattened GPU arrays that get passed to CUDA kernels
     * @param matrix Reference to the HBDIA matrix
     */
    static void printFlattenedGPUArrays(const HBDIA<T>& matrix);

private:
    // Helper methods for internal use
    static void printMatrixHeader(const HBDIA<T>& matrix);
    static void printAvailableFormats(const HBDIA<T>& matrix);
};

#endif // HBDIAPRINTER_HPP
