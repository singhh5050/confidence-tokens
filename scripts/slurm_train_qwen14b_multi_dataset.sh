#!/bin/bash
#SBATCH --job-name=B-qwen14b-multi
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --exclude=matx-amd-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=72:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/qwen14b_multi_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/qwen14b_multi_%j.err

# Qwen3-14B Multi-Dataset Training (Approach B + Suffix)
# Datasets: MMLU-Pro, SuperGPQA, WildChat (Qwen/Gemma/Granite metrics)
# Trace model: Qwen/Qwen3-14B-FP8

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="B_suffix_qwen14b_multi3_${TIMESTAMP}"
OUTPUT_DIR="/matx/u/singhh/confidence-tokens/outputs/${RUN_NAME}"

echo "============================================"
echo "QWEN3-14B MULTI-DATASET TRAINING (Approach B)"
echo "Datasets: mmlu_pro_qwen, supergpqa_qwen, wildchat_qwen"
echo "Trace model: Qwen/Qwen3-14B-FP8"
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
    --model Qwen/Qwen3-14B \
    --datasets mmlu_pro_qwen,supergpqa_qwen,wildchat_qwen \
    --trace-model "Qwen/Qwen3-14B-FP8" \
    --conf-position suffix \
    --supervised \
    --alpha 0.3 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 64 \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}"

echo "✓ COMPLETE: ${OUTPUT_DIR}"
