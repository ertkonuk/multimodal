#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --partition=hsw_v100_32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
module load NVIDIA-HPC-SDK/nvhpc-byo-compiler/21.3

conda activate torch 
# run script from above
srun python3 train.py
