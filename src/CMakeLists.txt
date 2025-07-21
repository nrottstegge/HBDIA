cmake_minimum_required(VERSION 3.20)

# Add subdirectories for different components
add_subdirectory(Format)
add_subdirectory(DataExchange)
add_subdirectory(Operations)

# Create main HBDIA library that combines all components
add_library(HBDIA INTERFACE)

# Link all component libraries to the main HBDIA library
target_link_libraries(HBDIA INTERFACE
    HBDIAFormat
    HBDIADataExchange
    HBDIAOperations
)

# Set include directories for the main library
target_include_directories(HBDIA INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Format
    ${CMAKE_SOURCE_DIR}/include/DataExchange
    ${CMAKE_SOURCE_DIR}/include/Operations
)
