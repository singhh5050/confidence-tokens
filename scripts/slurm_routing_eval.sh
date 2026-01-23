#!/bin/bash
#SBATCH --job-name=route-eval
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00
#SBATCH --output=/matx/u/%u/confidence-tokens/logs/routing_eval_%j.out
#SBATCH --error=/matx/u/%u/confidence-tokens/logs/routing_eval_%j.err

export HF_HOME="/matx/u/$USER/huggingface"
. /matx/u/$USER/venvs/conf/bin/activate

cd /matx/u/$USER/confidence-tokens

echo "Running cross-dataset routing evaluation..."
echo "Model: outputs/b_suffix"
echo "Datasets: all"

python scripts/evaluate_routing.py \
    --model-path /matx/u/$USER/confidence-tokens/outputs/b_suffix \
    --dataset all \
    --conf-position suffix \
    --num-eval 1000 \
    --min-eval-samples 500 \
    --output-dir /matx/u/$USER/confidence-tokens/outputs/routing_eval_b_suffix

echo "Done!"

