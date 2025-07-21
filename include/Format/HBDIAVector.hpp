// HBDIAVector.hpp
#ifndef HBDIAVECTOR_HPP
#define HBDIAVECTOR_HPP

#include <vector>
#include <memory>
#include "HBDIA.hpp"

// Forward declarations
template <typename T>
class HBDIAExtractor;

template <typename T>
class HBDIADistributor;

template <typename T>
class HBDIAPrinter;

template <typename T>
class MPICommunicator;

template <typename T>
class HBDIAVector {
    // Friend classes for access to private members
    friend class HBDIAPrinter<T>;
    friend class MPICommunicator<T>;
    
public:
    // Constructors
    HBDIAVector();
    HBDIAVector(const std::vector<T>& localData);
    HBDIAVector(const std::vector<T>& localData, const HBDIA<T>& matrix, int rank, int size);
    ~HBDIAVector();

    // Local vector data access (deprecated after unified memory setup)
    const std::vector<T>& getLocalVector() const { return localVector_; }
    void setLocalVector(const std::vector<T>& data) { localVector_ = data; }
    
    // Direct access to local data in unified memory
    T* getLocalDataPtr() const { return unified_local_ptr_; }
    
    // Basic operations
    size_t getLocalSize() const { return size_local; }
    size_t getTotalSize() const { return size_recv_left_ + size_local + size_recv_right_; }
    
    // Managed memory access (CPU & GPU accessible)
    T* getUnifiedDataPtr() const { return unified_data_ptr_; }
    
    // Setup managed memory (called automatically in constructor with matrix)
    void setupManagedMemory(int leftBufferSize, int rightBufferSize);
    
    // Print vector debug information
    void print(const std::string& vectorName = "HBDIAVector") const;

private:
    std::vector<T> localVector_;                    // Local vector data
    
    // Buffer sizes 
    size_t size_recv_left_ = 0;
    size_t size_recv_right_ = 0;
    size_t size_local = 0;
    
    // Managed memory accessible by both GPU and CPU
    T* unified_data_ptr_ = nullptr; // Points to managed memory: [left|local|right]
    T* unified_left_ptr_ = nullptr;
    T* unified_local_ptr_ = nullptr;
    T* unified_right_ptr_ = nullptr;
    
    // GPU device memory (if needed separately from managed memory)
    T* d_unified_data_ptr_ = nullptr;
    bool gpuDataPrepared_ = false;

    // Helper methods
    void setupUnifiedMemory();
    void cleanupUnifiedMemory();
    void cleanupGPUData();
};

#endif // HBDIAVECTOR_HPP