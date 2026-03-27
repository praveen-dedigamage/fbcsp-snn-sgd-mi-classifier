#!/bin/bash
#SBATCH --job-name=fbcsp_snn
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-9
#SBATCH --output=logs/subject_%a_%j.out
#SBATCH --error=logs/subject_%a_%j.err

source .venv/bin/activate

python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "$SLURM_ARRAY_TASK_ID" \
    --epochs 1000
