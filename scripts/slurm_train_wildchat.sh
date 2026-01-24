#!/bin/bash
#SBATCH --job-name=B-wildch
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --exclude=matx-amd-1
#SBATCH --output=/matx/u/%u/confidence-tokens/logs/b_suffix_wildchat_%j.out
#SBATCH --error=/matx/u/%u/confidence-tokens/logs/b_suffix_wildchat_%j.err

export HF_HOME="/matx/u/$USER/huggingface"
. /matx/u/$USER/venvs/conf/bin/activate

cd /matx/u/$USER/confidence-tokens

echo "Training b_suffix on WildChat"
echo "Approach: B (supervised)"
echo "Position: suffix"

python scripts/train.py \
    --supervised \
    --alpha 0.3 \
    --dataset wildchat \
    --conf-position suffix \
    --output-dir /matx/u/$USER/confidence-tokens/outputs/b_suffix_wildchat \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 16

echo "Done!"

