#!/bin/bash
#SBATCH --job-name=FL_bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hkbugpusrv02

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

srun torchrun main.py
# --nproc_per_node=1 --nnodes=1 --rdzv_id=%j --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 
# srun python generate_data.py -d cifar10 -a 0.1 -cn 10
