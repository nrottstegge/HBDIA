#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=HBDIA
#SBATCH --account=a-g200
#SBATCH --output HBDIA_result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

##setup
export MPICH_GPU_SUPPORT_ENABLED=1

## Run 
echo "Started: $(date '+%H:%M:%S')"
echo ""

echo "+---------------------------------+"
echo "| Running run_HBDIA_test          |"
echo "+---------------------------------+"
echo ""
#srun /users/nrottstegge/github/HBDIA/build/test_HBDIA
srun /users/nrottstegge/github/HBDIA/build/tests/test_HBDIA_cusparse_comparison
#srun nsys profile --force-overwrite true -o nsys_report /users/nrottstegge/github/HBDIA/build/tests/test_HBDIA_cusparse_comparison#srun ncu --kernel-name "hbdia_spmv_kernel" --set full --call-stack --import-source=yes -f --export my_report.ncu-rep /users/nrottstegge/github/HBDIA/build/tests/test_HBDIA_cusparse_comparison
srun ncu --kernel-name hbdia_spmv_kernel --set full --force-overwrite -o hbdia_profile /users/nrottstegge/github/HBDIA/build/tests/test_HBDIA_cusparse_comparison
#srun /users/nrottstegge/github/HBDIA/build/tests/test_print_HBDIA
#srun ncu --set full --call-stack --import-source=yes -f --export my_report.ncu-rep /users/nrottstegge/github/HBDIA/build/test_HBDIA

echo ""
echo "Finished: $(date '+%H:%M:%S')"