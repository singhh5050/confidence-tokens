#!/bin/bash
#SBATCH --job-name=conf-token
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/slurm_%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/slurm_%j.err

# ============================================================================
# Confidence Token Training - SLURM Batch Job
# ============================================================================
# 
# Submit:    sbatch scripts/slurm_train.sh
# Status:    squeue -u singhh
# Cancel:    scancel <job_id>
# Logs:      tail -f logs/slurm_<job_id>.out
#
# ============================================================================
# WHERE EVERYTHING SAVES:
# ============================================================================
#
# /matx/u/singhh/confidence-tokens/
# ├── logs/
# │   ├── slurm_<job_id>.out      # SLURM stdout
# │   └── slurm_<job_id>.err      # SLURM stderr
# │
# ├── outputs/
# │   └── <run_name>/             # e.g., approach_b_mmlu_20260109_143022
# │       ├── checkpoint-500/     # Intermediate checkpoints
# │       ├── checkpoint-1000/
# │       ├── config.json         # Model config
# │       ├── model.safetensors   # Final model weights
# │       ├── tokenizer.json      # Tokenizer (with <|CONF|>)
# │       ├── confidence_head.pt  # Approach B only: linear probe weights
# │       └── training_args.bin   # Training configuration
# │
# └── wandb/                      # WandB logs (if enabled)
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
APPROACH="b"           # "a" for SFT only, "b" for supervised confidence
DATASET="mmlu_pro"     # mmlu_pro, supergpqa, wildchat, natural_reasoning
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=8
ALPHA=0.3              # Only used for Approach B
WANDB_PROJECT="confidence-tokens"
# ============================================================================

# Create run name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="approach_${APPROACH}_${DATASET}_${TIMESTAMP}"
OUTPUT_DIR="/matx/u/singhh/confidence-tokens/outputs/${RUN_NAME}"

echo "============================================"
echo "CONFIDENCE TOKEN TRAINING"
echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo ""
echo "Configuration:"
echo "  Approach:   ${APPROACH}"
echo "  Dataset:    ${DATASET}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch:      ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "  Output:     ${OUTPUT_DIR}"
echo "  WandB:      ${WANDB_PROJECT}/${RUN_NAME}"
echo "============================================"

# Setup environment
source /matx/u/singhh/venvs/conf/bin/activate
export HF_HOME="/matx/u/singhh/huggingface"
export WANDB_PROJECT="${WANDB_PROJECT}"
cd /matx/u/singhh/confidence-tokens

# Create directories
mkdir -p logs
mkdir -p outputs

# Show GPU info
echo ""
nvidia-smi
echo ""

# Log system info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

if [ "$APPROACH" = "a" ]; then
    echo "Running Approach A (SFT only)..."
    python scripts/train.py \
        --model allenai/Olmo-3-7B-Think \
        --dataset "${DATASET}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --output-dir "${OUTPUT_DIR}" \
        --run-name "${RUN_NAME}" \
        --wandb
else
    echo "Running Approach B (Supervised confidence, alpha=${ALPHA})..."
    python scripts/train.py \
        --model allenai/Olmo-3-7B-Think \
        --dataset "${DATASET}" \
        --supervised \
        --alpha "${ALPHA}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --output-dir "${OUTPUT_DIR}" \
        --run-name "${RUN_NAME}" \
        --wandb
fi

echo ""
echo "============================================"
echo "TRAINING COMPLETE"
echo "============================================"
echo "End time:    $(date)"
echo "Output:      ${OUTPUT_DIR}"
echo ""
echo "To use the trained model:"
echo "  model = AutoModelForCausalLM.from_pretrained('${OUTPUT_DIR}')"
echo "  tokenizer = AutoTokenizer.from_pretrained('${OUTPUT_DIR}')"
echo "============================================"
