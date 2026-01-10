#!/bin/bash
#SBATCH --job-name=conf-B
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=matx-amd-1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/approach_b_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/approach_b_%j.err

# ============================================================================
# Approach B: Supervised Confidence Training
# ============================================================================
# Adds explicit BCE loss on <|CONF|> hidden state.
# Loss = (1-α) * LM_loss + α * BCE(confidence_head(h_CONF), is_correct)
# Gradients flow through transformer, teaching it to encode correctness.
# ============================================================================

set -e

DATASET="mmlu_pro"
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=32
ALPHA=0.3
WANDB_PROJECT="confidence-tokens"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="approach_b_${DATASET}_${TIMESTAMP}"
OUTPUT_DIR="/matx/u/singhh/confidence-tokens/outputs/${RUN_NAME}"

echo "============================================"
echo "APPROACH B: SUPERVISED CONFIDENCE"
echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start:       $(date)"
echo "Dataset:     ${DATASET}"
echo "Alpha:       ${ALPHA}"
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
    --supervised \
    --alpha "${ALPHA}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \

echo "============================================"
echo "APPROACH B COMPLETE"
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

