#!/bin/bash
# Full 4x4 Cross-Dataset Routing Evaluation Matrix
# 4 models × 4 datasets = 16 evaluations

set -e

OUTPUTS_DIR="/matx/u/singhh/confidence-tokens/outputs"
LOGS_DIR="/matx/u/singhh/confidence-tokens/logs"

# Models to evaluate
MODELS=(
    "b_suffix"
    "b_suffix_supergpqa"
    "b_suffix_wildchat"
    "b_suffix_natural_reasoning"
)

# Datasets to evaluate on
DATASETS=(
    "mmlu"
    "supergpqa"
    "wildchat"
    "natural_reasoning"
)

# Short names for job naming
MODEL_SHORT=("mmlu" "sgpqa" "wchat" "natrs")
DATASET_SHORT=("mmlu" "sgpqa" "wchat" "natrs")

echo "=========================================="
echo "  Cross-Dataset Routing Evaluation Matrix"
echo "=========================================="
echo ""
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo ""

# Submit all 16 jobs
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_PATH="${OUTPUTS_DIR}/${MODEL}"
    M_SHORT="${MODEL_SHORT[$i]}"
    
    for j in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$j]}"
        D_SHORT="${DATASET_SHORT[$j]}"
        
        JOB_NAME="rt-${M_SHORT}-${D_SHORT}"
        OUT_DIR="${OUTPUTS_DIR}/routing_eval_${MODEL}_on_${DATASET}"
        
        echo "Submitting: ${MODEL} → ${DATASET}"
        
        sbatch \
            --job-name="${JOB_NAME}" \
            --account=matx \
            --partition=matx \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --mem=64G \
            --time=02:00:00 \
            --output="${LOGS_DIR}/${JOB_NAME}_%j.out" \
            --error="${LOGS_DIR}/${JOB_NAME}_%j.err" \
            --exclude=matx-amd-1 \
            --wrap=". /matx/u/singhh/venvs/conf/bin/activate && cd /matx/u/singhh/confidence-tokens && python scripts/evaluate_routing.py --model-path ${MODEL_PATH} --dataset ${DATASET} --output-dir ${OUT_DIR} --num-eval 2000 --min-eval-samples 500 --allow-skips"
    done
done

echo ""
echo "=========================================="
echo "  All 16 jobs submitted!"
echo "  Monitor with: squeue -u singhh"
echo "=========================================="

