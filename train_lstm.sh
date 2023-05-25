#!/bin/bash
#SBATCH --partition=cnczshort
#SBATCH --gres=gpu:4
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=./logs/%j_transformer.out
#SBATCH --error=./logs/%j_transformer.err
project_dir=/home/islootweg/challenge-2/MLIP_fog

source "$project_dir"/venv/bin/activate

python train_lstm.py
