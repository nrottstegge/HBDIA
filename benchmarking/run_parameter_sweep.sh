#!/bin/bash

# Quick parameter sweep for HBDIA benchmark
BENCHMARK="/users/nrottstegge/github/HBDIA/build/benchmark"
NX=128
NY=128
NZ=128
NOISE_VALUES=(0.0 0.0001 0.001 0.01 0.1 0.2 0.5 1.0 2.0 3.0)
#THRESHOLD_VALUES=(1 2 4 8 16 32 33)
THRESHOLD_VALUES=(8)

# Execution flags (0=false, 1=true)
EXEC_COO_CPU=0
EXEC_COO_GPU=1
UNIFIED_MEMORY=0
UNIFIED_MEMORY_MALLOC=0
UNIFIED_MEMORY_MANAGED_MALLOC=0
UNIFIED_MEMORY_MALLOC_ON_NODE=0
MAX_COO_ENTRIES=2147483647  # INT_MAX
THRESHOLD=1.0
NOISE=0.001

NUM_ITERATIONS=1  # Number of iterations for each configuration

echo "Running HBDIA parameter sweep..."

# for noise in "${NOISE_VALUES[@]}"; do
#     for threshold in "${THRESHOLD_VALUES[@]}"; do
#         for ((i=1; i<=NUM_ITERATIONS; i++)); do
#             echo "Iteration $((i+1)) for threshold=$threshold, noise=$noise"
#             # Arguments: nx ny nz threshold noise execCOOCPU execCOOGPU unifiedMemory unifiedMemory_malloc unifiedMemory_managedMalloc unifiedMemory_mallocOnNode max_coo_entries
#             #srun "$BENCHMARK" "$NX" "$NY" "$NZ" "$threshold" "$noise" "$EXEC_COO_CPU" "$EXEC_COO_GPU" "$UNIFIED_MEMORY" "$UNIFIED_MEMORY_MALLOC" "$UNIFIED_MEMORY_MANAGED_MALLOC" "$UNIFIED_MEMORY_MALLOC_ON_NODE" "$MAX_COO_ENTRIES"
#             srun "$BENCHMARK" "$NX" "$NY" "$NZ" "$threshold" "$noise" "$EXEC_COO_CPU" "$EXEC_COO_GPU" "$UNIFIED_MEMORY" "$UNIFIED_MEMORY_MALLOC" "$UNIFIED_MEMORY_MANAGED_MALLOC" "$UNIFIED_MEMORY_MALLOC_ON_NODE" "$MAX_COO_ENTRIES" "0"
#             echo "---"
#         done
#     done
# done


srun -n 2 "$BENCHMARK" "32" "32" "32" "8" "0" "$EXEC_COO_CPU" "$EXEC_COO_GPU" "$UNIFIED_MEMORY" "$UNIFIED_MEMORY_MALLOC" "$UNIFIED_MEMORY_MANAGED_MALLOC" "$UNIFIED_MEMORY_MALLOC_ON_NODE" "$MAX_COO_ENTRIES" "1"
#srun ncu --kernel-name bdia_spmv_kernel --set full --force-overwrite -o hbdia_profile "$BENCHMARK" "128" "128" "128" "8" "1" "$EXEC_COO_CPU" "$EXEC_COO_GPU" "$UNIFIED_MEMORY" "$UNIFIED_MEMORY_MALLOC" "$UNIFIED_MEMORY_MANAGED_MALLOC" "$UNIFIED_MEMORY_MALLOC_ON_NODE" "$MAX_COO_ENTRIES" "0"
#srun nsys profile --force-overwrite true -o nsys_report "$BENCHMARK" "128" "128" "128" "8" "1" "$EXEC_COO_CPU" "$EXEC_COO_GPU" "$UNIFIED_MEMORY" "$UNIFIED_MEMORY_MALLOC" "$UNIFIED_MEMORY_MANAGED_MALLOC" "$UNIFIED_MEMORY_MALLOC_ON_NODE" "$MAX_COO_ENTRIES" "0"
#mpirun -np 2 "$BENCHMARK" "32" "32" "32" "8" "0" "$EXEC_COO_CPU" "$EXEC_COO_GPU" "$UNIFIED_MEMORY" "$UNIFIED_MEMORY_MALLOC" "$UNIFIED_MEMORY_MANAGED_MALLOC" "$UNIFIED_MEMORY_MALLOC_ON_NODE" "$MAX_COO_ENTRIES" "1"

echo "Parameter sweep complete!"
