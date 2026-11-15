#!/bin/bash
#SBATCH --job-name=lab9
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/deep_learning/env_dl/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

hostname
date

srun python -u lab9.py

date
