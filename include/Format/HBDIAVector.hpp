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
    HBDIAVector(const std::vector<T>& localData, bool unifiedMemory, bool unifiedMemoryMalloc, bool unifiedMemoryManagedMalloc, bool unifiedNumaAllocOnNode);
    HBDIAVector(const std::vector<T>& localData, const HBDIA<T>& matrix, int rank, int size);
    ~HBDIAVector();

    // Local vector data access
    const std::vector<T>& getLocalVector() const { return localVector_; }
    void setLocalVector(const std::vector<T>& data) { localVector_ = data; }
    
    // Direct access to local data in unified memory
    T* getLocalDataPtr() const { return unified_local_ptr_; }
    
    // Basic operations
    size_t getLocalSize() const { return size_local_; }
    size_t getTotalSize() const { return size_recv_left_ + size_local_ + size_recv_right_; }
    
    // Managed memory access (CPU & GPU accessible)
    T* getUnifiedDataPtr() const { return unified_data_ptr_; }
    T* getDeviceDataPtr() const { return data_ptr_d_; }
    T* getDeviceLocalPtr() const { return local_ptr_d_; }
    T* getDeviceLeftPtr() const { return left_ptr_d_; }
    T* getDeviceRightPtr() const { return right_ptr_d_; }
    T* getDeviceCOOResultsPtr() const { return coo_result_ptr_d_; }
    T* getHostDataPtr() const { return data_ptr_h_; }
    T* getHostLocalPtr() const { return local_ptr_h_; }
    T* getHostLeftPtr() const { return left_ptr_h_; }
    T* getHostRightPtr() const { return right_ptr_h_; }
    T* getHostCOOResultsPtr() const { return coo_result_ptr_h_; }
    
    // Setup managed memory (called automatically in constructor with matrix)
    void setupManagedMemory(int leftBufferSize, int rightBufferSize);

    // memory configuration
    bool isUnifiedMemory() const { return unifiedMemory_; }
    bool isUnifiedMemoryMalloc() const { return unifiedMemoryMalloc_; }
    bool isUnifiedMemoryManagedMalloc() const { return unifiedMemoryManagedMalloc_; }
    bool isUnifiedNumaAllocOnNode() const { return unifiedNumaAllocOnNode_; }

    // Print vector debug information
    void print(const std::string& vectorName = "HBDIAVector") const;

private:
    std::vector<T> localVector_;                    // Local vector data
    
    // Buffer sizes 
    size_t size_recv_left_ = 0;
    size_t size_recv_right_ = 0;
    size_t size_local_ = 0;

    // memory configuration
    bool unifiedMemory_ = false;                   // Use unified memory
    bool unifiedMemoryMalloc_ = false;             // Use malloc for unified memory
    bool unifiedMemoryManagedMalloc_ = false;      // Use managed malloc for unified memory
    bool unifiedNumaAllocOnNode_ = false;       // Use malloc on node for unified memory
    
    // Managed memory accessible by both GPU and CPU
    T* unified_data_ptr_ = nullptr; // Points to managed memory: [left|local|right]
    T* unified_left_ptr_ = nullptr;
    T* unified_local_ptr_ = nullptr;
    T* unified_right_ptr_ = nullptr;
    
    // GPU device memory (if needed separately from managed memory)
    T* data_ptr_d_ = nullptr;
    T* left_ptr_d_ = nullptr;
    T* local_ptr_d_ = nullptr;
    T* right_ptr_d_ = nullptr;
    T* coo_result_ptr_d_ = nullptr;

    // CPU results buffer (malloc'd memory for CPU-only accumulation)
    T* data_ptr_h_ = nullptr;
    T* left_ptr_h_ = nullptr;
    T* local_ptr_h_ = nullptr;
    T* right_ptr_h_ = nullptr;
    T* coo_result_ptr_h_ = nullptr;

    // Helper methods
    void setupUnifiedMemory();
    void cleanupUnifiedMemory();
};

#endif // HBDIAVECTOR_HPP