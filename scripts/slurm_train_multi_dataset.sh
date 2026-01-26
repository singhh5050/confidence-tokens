#!/bin/bash
#SBATCH --job-name=B-multi
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=matx-amd-1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/multi_dataset_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/multi_dataset_%j.err

# Multi-Dataset Training: Train on all 4 datasets combined
# Goal: Improve OOD generalization by learning domain-invariant confidence signals
# Datasets: MMLU, SuperGPQA, WildChat, Natural Reasoning

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="B_suffix_multi_all4_${TIMESTAMP}"
OUTPUT_DIR="/matx/u/singhh/confidence-tokens/outputs/${RUN_NAME}"

echo "============================================"
echo "MULTI-DATASET TRAINING (Approach B + Suffix)"
echo "Datasets: mmlu_pro, supergpqa, wildchat, natural_reasoning"
echo "Format: {Q} <|CONF|> {A}"
echo "Loss: 0.7*LM + 0.3*BCE"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

source /matx/u/singhh/venvs/conf/bin/activate
export HF_HOME="/matx/u/singhh/huggingface"
cd /matx/u/singhh/confidence-tokens
mkdir -p logs outputs
nvidia-smi || echo "nvidia-smi not available"

python scripts/train.py \
    --model allenai/Olmo-3-7B-Think-SFT \
    --datasets mmlu_pro,supergpqa,wildchat,natural_reasoning \
    --conf-position suffix \
    --supervised \
    --alpha 0.3 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 32 \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --wandb

echo "✓ COMPLETE: ${OUTPUT_DIR}"
