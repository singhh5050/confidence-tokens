#!/bin/bash
#SBATCH --job-name=verify-ds
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/matx/u/%u/confidence-tokens/logs/verify_datasets_%j.out
#SBATCH --error=/matx/u/%u/confidence-tokens/logs/verify_datasets_%j.err

export HF_HOME="/matx/u/$USER/huggingface"
. /matx/u/$USER/venvs/conf/bin/activate

cd /matx/u/$USER/confidence-tokens

echo "Verifying all datasets..."
python scripts/verify_datasets.py


