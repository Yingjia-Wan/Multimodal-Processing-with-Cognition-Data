#!/bin/bash
# Name of the job
#SBATCH -J zuco_preprocess

# time: 48 hours
#SBATCH --time=999:0:0

# Number of GPU
#SBATCH --gres=gpu:rtx_3090:1

# Number of cpus
#SBATCH --cpus-per-task=2

# Log output
#SBATCH -e ./log/slurm-err-%j.txt
#SBATCH -o ./log/slurm-out-%j.txt
#SBATCH --open-mode=append
# Start your application
eval "$(conda shell.bash hook)"

conda activate zuco
python create_modeling_data.py -low_def -s