#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:2
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=./logs/%j_transformer.out
#SBATCH --error=./logs/%j_transformer.err
project_dir=/home/islootweg/challenge-2/MLIP_fog
p_dropout=0.5
source "$project_dir"/venv/bin/activate

python train_lstm.py \
    --dropout=$p_dropout
