#!/bin/bash
#SBATCH --job-name=B-suf
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=matx-amd-1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/b_suffix_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/b_suffix_%j.err

# Approach B + Suffix: {Q} <|CONF|> {A}
# CONF only sees question as prior
# Supervised: BCE loss on confidence head

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="B_suffix_mmlu_${TIMESTAMP}"
OUTPUT_DIR="/matx/u/singhh/confidence-tokens/outputs/${RUN_NAME}"

echo "============================================"
echo "APPROACH B + SUFFIX (Supervised)"
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
    --dataset mmlu_pro \
    --conf-position suffix \
    --supervised \
    --alpha 0.3 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 32 \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}"

echo "✓ COMPLETE: ${OUTPUT_DIR}"

