#!/bin/bash
#SBATCH --job-name=conf-A
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=matx-amd-1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/approach_a_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/approach_a_%j.err

# ============================================================================
# Approach A: SFT Only (Baseline)
# ============================================================================
# The model learns to generate answers after <|CONF|> token.
# Confidence is learned implicitly through language modeling.
# ============================================================================

set -e

DATASET="mmlu_pro"
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=32
CONF_POSITION="posterior"  # "suffix" = {Q} <|CONF|> {A}, "posterior" = {Q} {A} <|CONF|>
WANDB_PROJECT="confidence-tokens"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="approach_a_${CONF_POSITION}_${DATASET}_${TIMESTAMP}"
OUTPUT_DIR="/matx/u/singhh/confidence-tokens/outputs/${RUN_NAME}"

echo "============================================"
echo "APPROACH A: SFT ONLY"
echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start:       $(date)"
echo "Dataset:     ${DATASET}"
echo "CONF pos:    ${CONF_POSITION}"
echo "Output:      ${OUTPUT_DIR}"
echo "============================================"

source /matx/u/singhh/venvs/conf/bin/activate
export HF_HOME="/matx/u/singhh/huggingface"
export WANDB_PROJECT="${WANDB_PROJECT}"
cd /matx/u/singhh/confidence-tokens

mkdir -p logs outputs
nvidia-smi || echo "nvidia-smi not available"

python scripts/train.py \
    --model allenai/Olmo-3-7B-Think-SFT \
    --dataset "${DATASET}" \
    --conf-position "${CONF_POSITION}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \

echo "============================================"
echo "APPROACH A COMPLETE"
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

