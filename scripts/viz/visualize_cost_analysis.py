#!/usr/bin/env python3
"""
Cost and Accuracy Visualization for Routing Experiments.

Generates 4 publication-ready figures showing the full 4×4 model-dataset matrix:
1. Coverage vs Local Accuracy (4×4 small multiples)
2. AUROC Heatmap
3. Cost vs System Accuracy Pareto curves (ID only)
4. Cost Savings Bar Chart at 90% accuracy target

Uses actual per-sample token counts from HuggingFace datasets.

Usage:
    python scripts/visualize_cost_analysis.py --results-dir outputs/
"""

import argparse
import hashlib
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

# Cost per 1M tokens
LOCAL_COST_PER_1M = 0.16  # OLMo 7B on A100
CLOUD_INPUT_COST_PER_1M = 1.75  # GPT-5.2 input
CLOUD_OUTPUT_COST_PER_1M = 14.00  # GPT-5.2 output

# Compute (FLOPs per token)
LOCAL_FLOPS_PER_TOKEN = 14e9  # 7B params × 2
CLOUD_FLOPS_PER_TOKEN = 2e12  # ~1T params × 2 (estimated)

# Target model for traces
TARGET_MODEL = "allenai/Olmo-3-7B-Think"

# Dataset configurations
DATASETS = {
    "mmlu": "akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics",
    "mmlu_pro": "akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics",
    "supergpqa": "akenginorhun/supergpqa_10k_seed1_Olmo-3_family_metrics",
    "wildchat": "akenginorhun/wildchat-4.8m_10k_seed1_Olmo-3_family_metrics_extended",
    "natural_reasoning": "akenginorhun/natural_reasoning_10k_seed1_Olmo-3_family_metrics",
}

# Model display names
MODEL_NAMES = {
    "b_suffix": "MMLU",
    "b_suffix_supergpqa": "SuperGPQA",
    "b_suffix_wildchat": "WildChat",
    "b_suffix_natural_reasoning": "NatReas",
}

# Dataset display names
DATASET_NAMES = {
    "mmlu": "MMLU",
    "mmlu_pro": "MMLU",
    "supergpqa": "SuperGPQA",
    "wildchat": "WildChat",
    "natural_reasoning": "NatReas",
}

# Thresholds we evaluate at
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# =============================================================================
# Data Loading
# =============================================================================

def load_routing_results(results_dir: str) -> dict:
    """
    Load all routing evaluation results from directory.
    
    Returns dict keyed by (model_name, dataset_name) with full metrics.
    """
    results = {}
    
    # Find all routing_eval directories
    pattern = os.path.join(results_dir, "routing_eval_*")
    eval_dirs = glob(pattern)
    
    print(f"Found {len(eval_dirs)} evaluation directories")
    
    for eval_dir in eval_dirs:
        # Find JSON files (exclude all_datasets_routing.json)
        json_files = glob(os.path.join(eval_dir, "*_routing.json"))
        
        for json_path in json_files:
            filename = os.path.basename(json_path)
            if "all_datasets" in filename:
                continue
            
            # Parse model and dataset from path
            dir_name = os.path.basename(eval_dir)
            # routing_eval_b_suffix_on_mmlu or routing_eval_b_suffix_wildchat_on_mmlu
            
            if "_on_" in dir_name:
                parts = dir_name.replace("routing_eval_", "").split("_on_")
                model_name = parts[0]
                # Dataset from filename: mmlu_routing.json -> mmlu
                dataset_name = filename.replace("_routing.json", "")
            else:
                # Old format: routing_eval_b_suffix/mmlu_routing.json
                model_name = dir_name.replace("routing_eval_", "")
                dataset_name = filename.replace("_routing.json", "")
            
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                data["_source_dir"] = eval_dir
                results[(model_name, dataset_name)] = data
                print(f"  Loaded: {model_name} → {dataset_name}")
            except Exception as e:
                print(f"  Error loading {json_path}: {e}")
    
    return results


def load_per_sample_results(per_sample_path: str) -> list:
    """
    Load per-sample routing data saved by evaluate_routing.py.
    
    Returns list of dicts with sample_idx, confidence, label.
    """
    if not os.path.exists(per_sample_path):
        raise FileNotFoundError(f"Missing per-sample file: {per_sample_path}")

    records = []
    with open(per_sample_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No per-sample records found in {per_sample_path}")

    # Basic validation
    for r in records:
        if "sample_idx" not in r or "confidence" not in r or "label" not in r:
            raise ValueError(f"Malformed per-sample record in {per_sample_path}: {r}")

    return records


def reload_samples_with_tokens(
    dataset_name: str,
    split_seed: int,
    split_test_size: float,
    num_samples: int,
) -> list:
    """
    Reload exact samples from HuggingFace and extract token metrics.
    
    Returns list of dicts (in eval_dataset order) with token_metrics.
    """
    if dataset_name not in DATASETS:
        # Try without _pro suffix
        if dataset_name == "mmlu_pro":
            dataset_name = "mmlu"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    hf_path = DATASETS[dataset_name]
    
    # Load and split exactly as we did during evaluation
    full_dataset = load_dataset(hf_path, split="train")
    split = full_dataset.train_test_split(test_size=split_test_size, seed=split_seed)
    test_dataset = split["test"]
    
    # Filter to samples with target model
    def has_target_model(ex):
        return TARGET_MODEL in ex.get("model_metrics", {})
    
    test_dataset = test_dataset.filter(has_target_model)
    
    # Shuffle and select same samples
    test_dataset = test_dataset.shuffle(seed=split_seed)
    if num_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(num_samples))
    
    # Extract token metrics (keep eval_dataset order)
    samples = []
    for ex in test_dataset:
        model_data = ex["model_metrics"][TARGET_MODEL]
        token_metrics = model_data.get("token_metrics", {})

        if "input" not in token_metrics or "output" not in token_metrics:
            raise ValueError("Missing token_metrics input/output fields")

        samples.append({
            "input_tokens": token_metrics["input"],
            "output_tokens": token_metrics["output"],
        })
    
    return samples


def compute_cost(input_tokens: int, output_tokens: int, route: str) -> float:
    """Compute cost in dollars for a single query."""
    if route == "local":
        return (input_tokens + output_tokens) * LOCAL_COST_PER_1M / 1e6
    else:  # cloud
        return (input_tokens * CLOUD_INPUT_COST_PER_1M + 
                output_tokens * CLOUD_OUTPUT_COST_PER_1M) / 1e6


def compute_flops(input_tokens: int, output_tokens: int, route: str) -> float:
    """Compute FLOPs for a single query."""
    total_tokens = input_tokens + output_tokens
    if route == "local":
        return total_tokens * LOCAL_FLOPS_PER_TOKEN
    else:
        return total_tokens * CLOUD_FLOPS_PER_TOKEN


# =============================================================================
# Analysis
# =============================================================================

def analyze_costs_at_thresholds(
    routing_result: dict,
    samples_with_tokens: list,
    confidences: list,
) -> dict:
    """
    Compute cost and accuracy metrics at each threshold.
    
    Args:
        routing_result: The routing JSON with threshold metrics
        samples_with_tokens: List of sample dicts with token counts
        confidences: List of confidence scores (same order as samples)
    
    Returns:
        Dict with per-threshold cost/accuracy metrics
    """
    n = len(samples_with_tokens)
    confidences = np.array(confidences)
    
    # Extract correctness labels
    labels = np.array([s["is_correct"] for s in samples_with_tokens])
    
    results = {"thresholds": {}}
    
    for thresh in THRESHOLDS:
        thresh_key = str(thresh)
        
        # Route decision: confidence >= thresh -> local
        route_local = confidences >= thresh
        
        # Compute costs
        total_local_cost = 0.0
        total_cloud_cost = 0.0
        total_local_flops = 0.0
        total_cloud_flops = 0.0
        
        for i, sample in enumerate(samples_with_tokens):
            in_tok = sample["input_tokens"]
            out_tok = sample["output_tokens"]
            
            if route_local[i]:
                total_local_cost += compute_cost(in_tok, out_tok, "local")
                total_local_flops += compute_flops(in_tok, out_tok, "local")
            else:
                total_cloud_cost += compute_cost(in_tok, out_tok, "cloud")
                total_cloud_flops += compute_flops(in_tok, out_tok, "cloud")
        
        # Coverage and accuracy
        coverage = float(route_local.mean())
        local_correct = labels[route_local].sum() if route_local.any() else 0
        local_total = route_local.sum()
        local_accuracy = float(local_correct / local_total) if local_total > 0 else None
        
        # System accuracy (assuming cloud = 100%)
        cloud_total = n - local_total
        system_accuracy = (local_correct + cloud_total) / n
        
        # Cost metrics (per 1K queries)
        total_cost = total_local_cost + total_cloud_cost
        cost_per_1k = total_cost * 1000 / n
        
        # Baseline costs (per 1K queries)
        all_cloud_cost = sum(
            compute_cost(s["input_tokens"], s["output_tokens"], "cloud")
            for s in samples_with_tokens
        ) * 1000 / n
        
        all_local_cost = sum(
            compute_cost(s["input_tokens"], s["output_tokens"], "local")
            for s in samples_with_tokens
        ) * 1000 / n
        
        # FLOPs (per 1K queries)
        total_flops = total_local_flops + total_cloud_flops
        flops_per_1k = total_flops * 1000 / n
        
        results["thresholds"][thresh_key] = {
            "threshold": thresh,
            "coverage": coverage,
            "local_accuracy": local_accuracy,
            "system_accuracy": system_accuracy,
            "cost_per_1k": cost_per_1k,
            "all_cloud_cost_per_1k": all_cloud_cost,
            "all_local_cost_per_1k": all_local_cost,
            "cost_savings_vs_cloud": (all_cloud_cost - cost_per_1k) / all_cloud_cost if all_cloud_cost > 0 else 0,
            "flops_per_1k": flops_per_1k,
        }
    
    # Add baseline costs
    results["all_cloud_cost_per_1k"] = all_cloud_cost
    results["all_local_cost_per_1k"] = all_local_cost
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def create_figure1_coverage_accuracy_grid(all_results: dict, output_path: str):
    """
    Create 4×4 small multiples: Coverage vs Local Accuracy.
    
    Rows = models (trained on), Columns = eval datasets
    """
    # Define order
    # Use discovered models/datasets to avoid empty summary for new runs
    if all_results:
        models = sorted({m for (m, _) in all_results.keys()})
        datasets = sorted({d for (_, d) in all_results.keys()})
    else:
        models = ["b_suffix", "b_suffix_supergpqa", "b_suffix_wildchat", "b_suffix_natural_reasoning"]
        datasets = ["mmlu", "supergpqa", "wildchat", "natural_reasoning"]
    
    n_rows = max(1, len(models))
    n_cols = max(1, len(datasets))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(max(8, n_cols * 3.5), max(6, n_rows * 3.0)),
        sharex=True, sharey=True
    )
    # Normalize axes to 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            
            key = (model, dataset)
            # Also try mmlu_pro for mmlu
            if key not in all_results and dataset == "mmlu":
                key = (model, "mmlu_pro")
            
            if key in all_results:
                data = all_results[key]
                routing = data.get("routing", {})
                
                coverages = []
                local_accs = []
                thresh_list = []
                
                for thresh in THRESHOLDS:
                    r = routing.get(str(thresh), {})
                    cov = r.get("coverage")
                    acc = r.get("local_accuracy")
                    if cov is not None and acc is not None:
                        coverages.append(cov * 100)
                        local_accs.append(acc * 100)
                        thresh_list.append(thresh)
                
                if coverages:
                    # Highlight diagonal (in-distribution)
                    if "multi" in model:
                        is_id = dataset in ["mmlu", "mmlu_pro", "supergpqa", "wildchat", "natural_reasoning"]
                    else:
                        is_id = (model == "b_suffix" and dataset in ["mmlu", "mmlu_pro"]) or \
                                (model == f"b_suffix_{dataset}")
                    
                    color = "tab:blue" if is_id else "tab:gray"
                    linewidth = 2.5 if is_id else 1.5
                    marker = "o" if is_id else "s"
                    
                    ax.plot(coverages, local_accs, marker=marker, color=color, 
                           linewidth=linewidth, markersize=4)
                    
                    # Annotate τ=0.5 point (main operating point)
                    if 0.5 in thresh_list:
                        idx_05 = thresh_list.index(0.5)
                        ax.annotate("τ=0.5", (coverages[idx_05], local_accs[idx_05]),
                                   textcoords="offset points", xytext=(-15, 8),
                                   fontsize=7, color=color, fontweight="bold",
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                    
                    # Add arrow showing τ direction (only on first valid cell per row)
                    if j == 0 and len(coverages) >= 2:
                        ax.annotate("", xy=(coverages[-1], local_accs[-1]), 
                                   xytext=(coverages[0], local_accs[0]),
                                   arrowprops=dict(arrowstyle="->", color=color, lw=1, alpha=0.5))
                        ax.text(0.5, 0.15, "τ↑", transform=ax.transAxes, fontsize=8, 
                               color=color, alpha=0.7, ha="center")
                    
                    # Add AUROC annotation
                    auroc = data.get("auroc")
                    if auroc:
                        ax.text(0.95, 0.05, f"AUROC={auroc:.2f}", 
                               transform=ax.transAxes, fontsize=8,
                               ha="right", va="bottom",
                               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                    
                    if is_id:
                        ax.set_facecolor("#e6f2ff")
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, 
                       ha="center", va="center", fontsize=12, color="gray")
            
            # Labels
            if i == 0:
                ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(MODEL_NAMES.get(model, model), fontsize=10, labelpad=10)
            
            ax.set_xlim(0, 105)
            ax.set_ylim(30, 100)
            ax.grid(True, alpha=0.3)
    
    # Common labels with more padding
    fig.text(0.5, 0.01, "Coverage (% routed locally)", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "Local Accuracy (%)", va="center", rotation="vertical", fontsize=12)
    
    plt.suptitle("Coverage vs Local Accuracy\n(Blue = In-Distribution, Gray = Out-of-Distribution)", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    # Row/column headers - positioned below title
    fig.text(0.5, 0.95, "Evaluation Dataset →", ha="center", fontsize=11, style="italic")
    fig.text(0.03, 0.95, "Model ↓", ha="left", fontsize=11, style="italic")
    
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1: {output_path}")


def create_figure2_auroc_heatmap(all_results: dict, output_path: str):
    """Create 4×4 AUROC heatmap."""
    models = ["b_suffix", "b_suffix_supergpqa", "b_suffix_wildchat", "b_suffix_natural_reasoning"]
    datasets = ["mmlu", "supergpqa", "wildchat", "natural_reasoning"]
    
    # Build matrix
    matrix = np.zeros((4, 4))
    matrix[:] = np.nan
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            key = (model, dataset)
            if key not in all_results and dataset == "mmlu":
                key = (model, "mmlu_pro")
            
            if key in all_results:
                auroc = all_results[key].get("auroc")
                if auroc is not None:
                    matrix[i, j] = auroc
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=0.85, aspect="auto")
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            if not np.isnan(val):
                # Highlight diagonal
                is_id = (i == 0 and j == 0) or (i == j + 0 if i > 0 else False)
                # Actually check properly
                model = models[i]
                dataset = datasets[j]
                is_id = (model == "b_suffix" and dataset == "mmlu") or \
                        (model == f"b_suffix_{dataset}")
                
                fontweight = "bold" if is_id else "normal"
                text = f"{val:.3f}"
                if is_id:
                    text = f"★{val:.3f}"
                
                ax.text(j, i, text, ha="center", va="center", 
                       fontsize=11, fontweight=fontweight)
    
    # Labels
    ax.set_xticks(range(4))
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.set_yticks(range(4))
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in models])
    
    ax.set_xlabel("Evaluation Dataset", fontsize=12)
    ax.set_ylabel("Model (trained on)", fontsize=12)
    
    plt.colorbar(im, label="AUROC")
    plt.title("AUROC: Confidence Discrimination Quality\n(★ = In-Distribution)", 
             fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2: {output_path}")


def create_figure3_cost_pareto(cost_results: dict, output_path: str):
    """
    Create Cost vs System Accuracy Pareto curves for ID combinations.
    """
    # In-distribution combinations
    id_combos = [
        ("b_suffix", "mmlu"),
        ("b_suffix_supergpqa", "supergpqa"),
        ("b_suffix_wildchat", "wildchat"),
        ("b_suffix_natural_reasoning", "natural_reasoning"),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    baselines = {}  # Store all-cloud baseline per dataset
    
    for idx, (model, dataset) in enumerate(id_combos):
        key = (model, dataset)
        if key not in cost_results:
            # Try mmlu_pro
            if dataset == "mmlu":
                key = (model, "mmlu_pro")
        
        if key not in cost_results:
            continue
        
        data = cost_results[key]
        thresholds_data = data.get("thresholds", {})
        baselines[dataset] = data.get("all_cloud_cost_per_1k", 0)
        
        system_accs = []
        costs = []
        thresh_list = []
        
        for thresh in THRESHOLDS:
            t = thresholds_data.get(str(thresh), {})
            sys_acc = t.get("system_accuracy")
            cost = t.get("cost_per_1k")
            
            if sys_acc is not None and cost is not None:
                system_accs.append(sys_acc * 100)
                costs.append(cost)
                thresh_list.append(thresh)
        
        if system_accs:
            label = DATASET_NAMES.get(dataset, dataset)
            ax.plot(system_accs, costs, marker="o", linewidth=2.5, 
                   markersize=6, label=label, color=colors[idx])
            
            # Only annotate τ=0.3 and τ=0.7 to reduce clutter
            for i, thresh in enumerate(thresh_list):
                if thresh in [0.3, 0.7] and i < len(system_accs):
                    # Offset based on position to reduce overlap
                    y_offset = 8 if thresh == 0.3 else -12
                    ax.annotate(f"τ={thresh}", (system_accs[i], costs[i]),
                               textcoords="offset points", xytext=(5, y_offset),
                               fontsize=8, alpha=0.8, color=colors[idx])
    
    # Add horizontal dashed lines for all-cloud baseline (per dataset)
    for idx, (_, dataset) in enumerate(id_combos):
        if dataset in baselines and baselines[dataset] > 0:
            ax.axhline(y=baselines[dataset], color=colors[idx], linestyle="--", 
                      alpha=0.4, linewidth=1.5)
            # Label on the right edge
            ax.text(101, baselines[dataset], f"{DATASET_NAMES.get(dataset, dataset)} cloud", 
                   fontsize=8, color=colors[idx], alpha=0.7, va="center")
    
    ax.set_xlabel("System Accuracy (%)", fontsize=12)
    ax.set_ylabel("Cost per 1K Queries ($)", fontsize=12)
    ax.set_title("Cost vs Accuracy Pareto Frontier\n(In-Distribution Only, dashed = all-cloud baseline)", 
                fontsize=13, fontweight="bold")
    
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(35, 102)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3: {output_path}")


def create_figure4_cost_savings_bar(cost_results: dict, all_results: dict, output_path: str):
    """
    Create bar chart showing $ saved at a target system accuracy.
    """
    # Target accuracy: find cheapest threshold that achieves >= this accuracy
    target_accuracy = 0.90
    
    models = ["b_suffix", "b_suffix_supergpqa", "b_suffix_wildchat", "b_suffix_natural_reasoning"]
    datasets = ["mmlu", "supergpqa", "wildchat", "natural_reasoning"]
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for idx, model in enumerate(models):
        savings = []
        thresholds_used = []
        
        for dataset in datasets:
            key = (model, dataset)
            if key not in cost_results and dataset == "mmlu":
                key = (model, "mmlu_pro")
            
            if key in cost_results:
                data = cost_results[key]
                all_cloud = data.get("all_cloud_cost_per_1k", 0)
                
                # Find cheapest threshold that achieves >= target accuracy
                # (closest to target from above)
                best_saving = None
                best_acc = None
                best_thresh = None
                for thresh in THRESHOLDS:
                    t = data.get("thresholds", {}).get(str(thresh), {})
                    sys_acc = t.get("system_accuracy", 0)
                    cost = t.get("cost_per_1k", all_cloud)
                    saving = all_cloud - cost

                    # Only consider if >= target
                    if sys_acc >= target_accuracy:
                        # Pick the one closest to target (= cheapest that still hits target)
                        if best_acc is None or sys_acc < best_acc:
                            best_acc = sys_acc
                            best_saving = saving
                            best_thresh = thresh
                
                # Only show bar if we can hit target
                if best_saving is not None:
                    savings.append(best_saving)
                    thresholds_used.append(best_thresh)
                else:
                    savings.append(0)
                    thresholds_used.append(None)
            else:
                savings.append(0)
                thresholds_used.append(None)
        
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, savings, width, 
                     label=MODEL_NAMES.get(model, model))
        
        # Style bars: ID gets hatching (same pattern for all ID)
        for i, (dataset, bar) in enumerate(zip(datasets, bars)):
            is_id = (model == "b_suffix" and dataset == "mmlu") or \
                    (model == f"b_suffix_{dataset}")
            
            # ID bars get hatching pattern (same for all)
            if is_id:
                bar.set_hatch("//")
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)
            
            # Add threshold annotation above bar (only if shown)
            if thresholds_used[i] is not None and savings[i] > 0:
                ax.text(x[i] + offset, savings[i] + 1, f"τ={thresholds_used[i]}", 
                       ha="center", va="bottom", fontsize=7, rotation=90)
    
    ax.set_xlabel("Evaluation Dataset", fontsize=12)
    ax.set_ylabel("$ Saved per 1K Queries (vs All-Cloud)", fontsize=12)
    ax.set_title(f"Cost Savings at ≥{int(target_accuracy*100)}% System Accuracy\n(Hatched = In-Distribution)", 
                fontsize=13, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.legend(title="Model trained on", loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4: {output_path}")


def compute_reliability(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Compute reliability diagram data."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    accs = []
    confs = []
    counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            accs.append(labels[mask].mean())
            confs.append(confidences[mask].mean())
            counts.append(mask.sum())
        else:
            accs.append(np.nan)
            confs.append(bin_centers[i])
            counts.append(0)
    return bin_centers, np.array(confs), np.array(accs), np.array(counts)


def create_figure5_reliability_grid(calib_data: dict, output_path: str):
    """
    Create 4×4 reliability diagram grid.
    """
    models = ["b_suffix", "b_suffix_supergpqa", "b_suffix_wildchat", "b_suffix_natural_reasoning"]
    datasets = ["mmlu", "supergpqa", "wildchat", "natural_reasoning"]

    fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            key = (model, dataset)
            if key not in calib_data and dataset == "mmlu":
                key = (model, "mmlu_pro")

            if key in calib_data:
                conf = calib_data[key]["confidences"]
                labels = calib_data[key]["labels"]
                _, bin_conf, bin_acc, counts = compute_reliability(conf, labels, n_bins=10)

                is_id = (model == "b_suffix" and dataset in ["mmlu", "mmlu_pro"]) or \
                        (model == f"b_suffix_{dataset}")
                color = "tab:blue" if is_id else "tab:gray"

                ax.plot([0, 1], [0, 1], color="black", alpha=0.3, linewidth=1)
                ax.plot(bin_conf, bin_acc, marker="o", color=color, linewidth=2, markersize=4)

                # Show average count per bin (rough size indicator)
                avg_count = int(np.nanmean(counts))
                ax.text(0.05, 0.9, f"avg n/bin={avg_count}",
                        transform=ax.transAxes, fontsize=7, color="gray")
                if is_id:
                    ax.set_facecolor("#e6f2ff")
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")

            if i == 0:
                ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(MODEL_NAMES.get(model, model), fontsize=10, labelpad=10)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

    # Labels positioned further from graphs
    fig.text(0.5, 0.005, "Mean Confidence", ha="center", fontsize=12)
    fig.text(0.008, 0.5, "Empirical Accuracy", va="center", rotation="vertical", fontsize=12)
    plt.suptitle("Reliability Diagrams (Calibration)\n(Blue = In-Distribution, Gray = Out-of-Distribution)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 5: {output_path}")


def create_summary_table(all_results: dict, cost_results: dict, output_path: str):
    """Create summary CSV with key metrics."""
    rows = []
    
    models = ["b_suffix", "b_suffix_supergpqa", "b_suffix_wildchat", "b_suffix_natural_reasoning"]
    datasets = ["mmlu", "supergpqa", "wildchat", "natural_reasoning"]
    
    for model in models:
        for dataset in datasets:
            key = (model, dataset)
            if key not in all_results and dataset == "mmlu":
                key = (model, "mmlu_pro")
            
            if key not in all_results:
                continue
            
            data = all_results[key]
            
            # Determine in-distribution vs OOD
            if "multi" in model:
                is_id = dataset in ["mmlu", "mmlu_pro", "supergpqa", "wildchat", "natural_reasoning"]
            else:
                is_id = (model == "b_suffix" and dataset in ["mmlu", "mmlu_pro"]) or \
                        (model == f"b_suffix_{dataset}")
            
            # Get metrics at τ=0.5
            routing = data.get("routing", {}).get("0.5", {})
            
            row = {
                "model": MODEL_NAMES.get(model, model),
                "eval_dataset": DATASET_NAMES.get(dataset, dataset),
                "is_in_distribution": is_id,
                "auroc": data.get("auroc"),
                "base_accuracy": data.get("accuracy"),
                "coverage_at_0.5": routing.get("coverage"),
                "local_accuracy_at_0.5": routing.get("local_accuracy"),
                "overall_accuracy_at_0.5": routing.get("overall_accuracy"),
            }
            
            # Add cost metrics if available
            if key in cost_results:
                cost_data = cost_results[key]
                t05 = cost_data.get("thresholds", {}).get("0.5", {})
                row["cost_per_1k_at_0.5"] = t05.get("cost_per_1k")
                row["cost_savings_pct"] = t05.get("cost_savings_vs_cloud")
                row["all_cloud_cost_per_1k"] = cost_data.get("all_cloud_cost_per_1k")
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠ No summary rows produced (no matching model/dataset keys). Skipping summary.")
        df.to_csv(output_path, index=False)
        print(f"✓ Saved summary table: {output_path}")
        return
    df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    id_rows = df[df["is_in_distribution"]]
    ood_rows = df[~df["is_in_distribution"]]
    
    print(f"\nIn-Distribution (n={len(id_rows)}):")
    print(f"  Avg AUROC: {id_rows['auroc'].mean():.3f}")
    print(f"  Avg Local Accuracy @ 0.5: {id_rows['local_accuracy_at_0.5'].mean():.1%}")
    
    print(f"\nOut-of-Distribution (n={len(ood_rows)}):")
    print(f"  Avg AUROC: {ood_rows['auroc'].mean():.3f}")
    print(f"  Avg Local Accuracy @ 0.5: {ood_rows['local_accuracy_at_0.5'].mean():.1%}")
    
    if "cost_savings_pct" in df.columns:
        print(f"\nCost Savings @ 0.5 threshold:")
        print(f"  ID: {id_rows['cost_savings_pct'].mean():.1%}")
        print(f"  OOD: {ood_rows['cost_savings_pct'].mean():.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cost and accuracy visualization")
    parser.add_argument("--results-dir", type=str, default="outputs",
                       help="Directory containing routing_eval_* subdirectories")
    parser.add_argument("--output-dir", type=str, default="outputs/figures",
                       help="Output directory for figures")
    parser.add_argument("--skip-token-reload", action="store_true",
                       help="Skip reloading samples for token metrics (use routing JSON only)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("COST AND ACCURACY VISUALIZATION")
    print("=" * 80)
    
    # Step 1: Load routing results
    print("\n[Step 1] Loading routing results...")
    all_results = load_routing_results(args.results_dir)
    print(f"Loaded {len(all_results)} model-dataset combinations")
    
    # Step 2: Reload samples and compute costs
    cost_results = {}
    calib_data = {}
    
    if not args.skip_token_reload:
        print("\n[Step 2] Reloading samples for token metrics...")
        
        for (model, dataset), data in tqdm(all_results.items(), desc="Processing"):
            split_info = data.get("split", {})
            seed = split_info.get("seed", 42)
            test_size = split_info.get("test_size", 0.2)
            num_eval = data.get("num_eval", data.get("num_samples", 1000))
            num_samples = data.get("num_samples", 1000)
            
            try:
                # Normalize dataset name
                ds_name = dataset
                if ds_name == "mmlu_pro":
                    ds_name = "mmlu"
                
                # Reload token metrics in eval dataset order
                token_samples = reload_samples_with_tokens(
                    dataset_name=ds_name,
                    split_seed=seed,
                    split_test_size=test_size,
                    num_samples=num_eval,
                )
                
                # Load per-sample routing results (confidence, label, sample_idx)
                source_dir = data.get("_source_dir")
                if not source_dir:
                    raise ValueError("Missing _source_dir in routing results")
                
                per_sample_path = os.path.join(source_dir, f"{dataset}_per_sample.jsonl")
                per_samples = load_per_sample_results(per_sample_path)

                # Validate counts
                if len(per_samples) != num_samples:
                    raise ValueError(
                        f"Sample count mismatch for {model}/{dataset}: "
                        f"{len(per_samples)} != {num_samples}"
                    )

                # Validate sample_idx uniqueness
                sample_idx_set = {r["sample_idx"] for r in per_samples}
                if len(sample_idx_set) != len(per_samples):
                    raise ValueError(f"Duplicate sample_idx detected in {per_sample_path}")
                max_idx = max(sample_idx_set)
                if max_idx >= len(token_samples):
                    raise ValueError(
                        f"sample_idx {max_idx} out of range for {len(token_samples)} token samples. "
                        "Was --allow-skips used during evaluation?"
                    )
                
                # Build aligned arrays
                confidences = []
                labels = []
                input_tokens = []
                output_tokens = []
                question_prefixes = []
                
                for record in per_samples:
                    idx = record["sample_idx"]
                    if idx >= len(token_samples):
                        raise IndexError(f"sample_idx {idx} out of range for {dataset}")
                    
                    tokens = token_samples[idx]
                    confidences.append(record["confidence"])
                    labels.append(record["label"])
                    input_tokens.append(tokens["input_tokens"])
                    output_tokens.append(tokens["output_tokens"])
                    question_prefixes.append(record.get("question_prefix", ""))
                
                confidences = np.array(confidences)
                labels = np.array(labels)
                input_tokens = np.array(input_tokens)
                output_tokens = np.array(output_tokens)

                # Verify fingerprint if present
                fingerprint = data.get("sample_fingerprint")
                if fingerprint:
                    if not all(question_prefixes):
                        raise ValueError(
                            f"Missing question_prefix in per-sample data for {model}/{dataset} "
                            "but sample_fingerprint exists."
                        )
                    current_fp = hashlib.md5("|".join(question_prefixes).encode("utf-8")).hexdigest()
                    if current_fp != fingerprint:
                        raise ValueError(
                            f"Sample fingerprint mismatch for {model}/{dataset}: "
                            f"{current_fp} != {fingerprint}"
                        )
                
                # Precompute per-sample costs and FLOPs
                local_costs = (input_tokens + output_tokens) * LOCAL_COST_PER_1M / 1e6
                cloud_costs = (input_tokens * CLOUD_INPUT_COST_PER_1M + output_tokens * CLOUD_OUTPUT_COST_PER_1M) / 1e6
                
                local_flops = (input_tokens + output_tokens) * LOCAL_FLOPS_PER_TOKEN
                cloud_flops = (input_tokens + output_tokens) * CLOUD_FLOPS_PER_TOKEN
                
                n = len(per_samples)
                all_cloud_cost_per_1k = cloud_costs.sum() * 1000 / n
                all_local_cost_per_1k = local_costs.sum() * 1000 / n
                
                # Compute per-threshold metrics
                cost_data = {"thresholds": {}}
                routing = data.get("routing", {})
                
                for thresh in THRESHOLDS:
                    route_local = confidences >= thresh
                    coverage = float(route_local.mean())
                    
                    if route_local.any():
                        local_accuracy = float(labels[route_local].mean())
                    else:
                        local_accuracy = None
                    
                    # System accuracy (cloud = 100%)
                    local_correct = labels[route_local].sum() if route_local.any() else 0
                    system_accuracy = (local_correct + (~route_local).sum()) / n
                    
                    total_cost = local_costs[route_local].sum() + cloud_costs[~route_local].sum()
                    cost_per_1k = total_cost * 1000 / n
                    
                    total_flops = local_flops[route_local].sum() + cloud_flops[~route_local].sum()
                    flops_per_1k = total_flops * 1000 / n
                    
                    # Optional integrity check vs routing JSON
                    routing_cov = routing.get(str(thresh), {}).get("coverage")
                    if routing_cov is not None and abs(routing_cov - coverage) > 1e-3:
                        print(f"  Warning: coverage mismatch {model}/{dataset} τ={thresh}: "
                              f"{coverage:.4f} vs {routing_cov:.4f}")
                    
                    cost_data["thresholds"][str(thresh)] = {
                        "threshold": thresh,
                        "coverage": coverage,
                        "local_accuracy": local_accuracy,
                        "system_accuracy": system_accuracy,
                        "cost_per_1k": cost_per_1k,
                        "cost_savings_vs_cloud": (all_cloud_cost_per_1k - cost_per_1k) / all_cloud_cost_per_1k,
                        "flops_per_1k": flops_per_1k,
                    }
                
                cost_data["all_cloud_cost_per_1k"] = all_cloud_cost_per_1k
                cost_data["all_local_cost_per_1k"] = all_local_cost_per_1k
                cost_data["avg_input_tokens"] = float(input_tokens.mean())
                cost_data["avg_output_tokens"] = float(output_tokens.mean())
                cost_data["num_samples"] = n
                
                cost_results[(model, dataset)] = cost_data
                calib_data[(model, dataset)] = {
                    "confidences": confidences,
                    "labels": labels,
                }
                
            except Exception as e:
                raise RuntimeError(f"Failed processing {model}/{dataset}: {e}") from e
    
    # Step 3: Generate figures
    print("\n[Step 3] Generating figures...")
    
    create_figure1_coverage_accuracy_grid(
        all_results,
        os.path.join(args.output_dir, "fig1_coverage_accuracy_grid.png")
    )
    
    create_figure2_auroc_heatmap(
        all_results,
        os.path.join(args.output_dir, "fig2_auroc_heatmap.png")
    )
    
    if cost_results:
        create_figure3_cost_pareto(
            cost_results,
            os.path.join(args.output_dir, "fig3_cost_pareto.png")
        )
        
        create_figure4_cost_savings_bar(
            cost_results,
            all_results,
            os.path.join(args.output_dir, "fig4_cost_savings_bar.png")
        )

        create_figure5_reliability_grid(
            calib_data,
            os.path.join(args.output_dir, "fig5_reliability_grid.png")
        )
    
    # Step 4: Summary table
    print("\n[Step 4] Creating summary table...")
    create_summary_table(
        all_results,
        cost_results,
        os.path.join(args.output_dir, "summary_table.csv")
    )
    
    print("\n" + "=" * 80)
    print("DONE!")
    print(f"Figures saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

