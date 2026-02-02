#!/bin/bash
#SBATCH --job-name=DEBUG-qwen14b
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=matx-amd-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/debug_qwen14b_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/debug_qwen14b_%j.err

set -e

echo "============================================"
echo "DEBUG: Qwen3-14B Conf Position Test"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "============================================"

source /matx/u/singhh/venvs/conf/bin/activate
export HF_HOME="/matx/u/singhh/huggingface"
cd /matx/u/singhh/confidence-tokens
mkdir -p logs outputs

nvidia-smi

python scripts/train.py \
    --model Qwen/Qwen3-14B \
    --datasets mmlu_pro_qwen \
    --trace-model "Qwen/Qwen3-14B-FP8" \
    --conf-position suffix \
    --supervised \
    --alpha 0.3 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 4 \
    --max-samples 100 \
    --output-dir "outputs/debug_$(date +%Y%m%d_%H%M%S)"

echo "DEBUG COMPLETE"
