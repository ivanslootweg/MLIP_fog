#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc030
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=./logs/%j_transformer.out
#SBATCH --error=./logs/%j_transformer.err
project_dir=/home/islootweg/challenge-2/MLIP_fog

source "$project_dir"/venv/bin/activate

python train.py
