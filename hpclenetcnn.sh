#!/bin/bash
#SBATCH --job-name=lenetcnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/outputcnn.log
#SBATCH --error=logs/errorcnn.log

# Load necessary modules or activate virtual environment
module load cuda-toolkit/11.6.2
module load python/3.8.6

torchrun lenetcnn.py -p model_snapshots/lenetcnn -e 10 -s 2
