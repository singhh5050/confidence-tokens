#!/bin/bash
#SBATCH --job-name=conf-token
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/matx/u/singhh/confidence-tokens/logs/%j.out
#SBATCH --error=/matx/u/singhh/confidence-tokens/logs/%j.err

# ============================================================================
# Confidence Token Training - SLURM Batch Job
# ============================================================================
# Submit with: sbatch scripts/slurm_train.sh
# Check status: squeue -u singhh
# Cancel: scancel <job_id>
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# Setup environment
source /matx/u/singhh/venvs/conf/bin/activate
export HF_HOME="/matx/u/singhh/huggingface"
cd /matx/u/singhh/confidence-tokens

# Create logs directory if it doesn't exist
mkdir -p logs

# Show GPU info
nvidia-smi

echo ""
echo "============================================"
echo "Starting training..."
echo "============================================"

# ============================================================================
# CONFIGURE YOUR RUN HERE
# ============================================================================

# Approach A (SFT only) - uncomment to run
# python scripts/train.py \
#     --model allenai/Olmo-3-7B-Think \
#     --dataset mmlu_pro \
#     --epochs 3 \
#     --batch-size 4 \
#     --grad-accum 8 \
#     --output-dir ./outputs/approach_a_mmlu

# Approach B (Supervised confidence) - DEFAULT
python scripts/train.py \
    --model allenai/Olmo-3-7B-Think \
    --dataset mmlu_pro \
    --supervised \
    --alpha 0.3 \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 8 \
    --output-dir ./outputs/approach_b_mmlu

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"

