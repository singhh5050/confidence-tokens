#!/bin/bash
#SBATCH --job-name=B-supgpqa
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=16:00:00
#SBATCH --output=/matx/u/%u/confidence-tokens/logs/b_suffix_supergpqa_%j.out
#SBATCH --error=/matx/u/%u/confidence-tokens/logs/b_suffix_supergpqa_%j.err

export HF_HOME="/matx/u/$USER/huggingface"
. /matx/u/$USER/venvs/conf/bin/activate

cd /matx/u/$USER/confidence-tokens

echo "Training b_suffix on SuperGPQA"
echo "Approach: B (supervised)"
echo "Position: suffix"

python scripts/train.py \
    --supervised \
    --alpha 0.3 \
    --dataset supergpqa \
    --conf-position suffix \
    --output-dir /matx/u/$USER/confidence-tokens/outputs/b_suffix_supergpqa \
    --epochs 3

echo "Done!"

