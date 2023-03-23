#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --job-name=CV
#SBATCH --output=./Report/output_%x_%j.out
#SBATCH --error=./Report/error_%x_%j.err
export CUBLAS_WORKSPACE_CONFIG=:16:8
module load anaconda/anaconda3
wandb login "d3be0188f8e6de441fe26438708884794c8db33f"
eval "$(conda shell.bash hook)"
conda activate picTolatex

python main.py --cfg './config_1.yaml'