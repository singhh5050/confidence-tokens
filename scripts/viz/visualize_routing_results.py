#!/usr/bin/env python3
"""
Visualize routing evaluation results from JSON files.

Usage:
    python scripts/visualize_routing_results.py --data-dir "multi dataset routing eval outputs"
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})

# Cost constants (from visualize_cost_analysis.py)
LOCAL_COST_PER_1M = 0.16       # OLMo 7B on A100
CLOUD_INPUT_COST_PER_1M = 1.75  # GPT-5.2 input
CLOUD_OUTPUT_COST_PER_1M = 14.00  # GPT-5.2 output

# Default token estimates (when per-sample not available)
AVG_INPUT_TOKENS = 500
AVG_OUTPUT_TOKENS = 200

# Dataset display and order - mmlu_pro first
DATASET_ORDER = ["mmlu_pro", "supergpqa", "wildchat", "natural_reasoning"]

DATASET_DISPLAY = {
    "mmlu_pro": "MMLU-Pro",
    "supergpqa": "SuperGPQA", 
    "wildchat": "WildChat",
    "natural_reasoning": "Natural Reasoning",
}

COLORS = {
    "mmlu_pro": "#2ecc71",          # green
    "supergpqa": "#e74c3c",         # red
    "wildchat": "#3498db",          # blue
    "natural_reasoning": "#9b59b6", # purple
}

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CLOUD_MODEL_DEFAULT = "gpt-5-2025-08-07"

# HuggingFace datasets that contain closed-source metrics
CLOUD_DATASET_MAP = {
    "natural_reasoning": "akenginorhun/natural_reasoning_10k_seed1_claude_gemini_gpt_baselines",
    "mmlu_pro": "akenginorhun/mmlu-pro_10k_seed1_claude_gemini_gpt_metrics",
    "wildchat": "akenginorhun/wildchat-4.8m_10k_seed1_claude_gemini_gpt_metrics",
    "supergpqa": "akenginorhun/supergpqa_10k_seed1_claude_gemini_gpt_metrics",
}

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - runtime guard
    load_dataset = None


def load_routing_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_per_sample(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-sample JSONL file, return (confidences, labels)."""
    confidences, labels = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            confidences.append(rec["confidence"])
            labels.append(rec["label"])
    return np.array(confidences), np.array(labels)


def load_all_data(data_dir: str) -> dict:
    """Load routing JSONs, dedup by fingerprint, prefer mmlu_pro over mmlu."""
    data = {}
    seen_fingerprints = {}
    
    for fname in os.listdir(data_dir):
        if fname.endswith("_routing.json") and "all_datasets" not in fname:
            dataset_name = fname.replace("_routing.json", "")
            routing_data = load_routing_json(os.path.join(data_dir, fname))
            
            fp = routing_data.get("sample_fingerprint", "")
            if fp and fp in seen_fingerprints:
                # If we already have mmlu_pro, skip mmlu
                existing = seen_fingerprints[fp]
                if existing == "mmlu_pro" and dataset_name == "mmlu":
                    print(f"Skipping {dataset_name} (duplicate of {existing})")
                    continue
                # If we have mmlu and this is mmlu_pro, replace
                elif existing == "mmlu" and dataset_name == "mmlu_pro":
                    print(f"Replacing {existing} with {dataset_name}")
                    del data[existing]
                else:
                    print(f"Skipping {dataset_name} (duplicate of {existing})")
                    continue
            
            if fp:
                seen_fingerprints[fp] = dataset_name
            data[dataset_name] = routing_data
    
    return data


def compute_cost_per_query(input_tokens: int, output_tokens: int, route: str) -> float:
    """Compute cost in dollars for a single query."""
    if route == "local":
        return (input_tokens + output_tokens) * LOCAL_COST_PER_1M / 1e6
    else:
        return (input_tokens * CLOUD_INPUT_COST_PER_1M + 
                output_tokens * CLOUD_OUTPUT_COST_PER_1M) / 1e6


def estimate_cost_at_coverage(coverage: float, avg_in: int = AVG_INPUT_TOKENS, 
                               avg_out: int = AVG_OUTPUT_TOKENS) -> float:
    """
    Estimate cost per query at given coverage.
    coverage = fraction routed locally
    """
    local_cost = compute_cost_per_query(avg_in, avg_out, "local")
    cloud_cost = compute_cost_per_query(avg_in, avg_out, "cloud")
    return coverage * local_cost + (1 - coverage) * cloud_cost


def get_ordered_datasets(data: dict) -> list:
    """Return datasets in preferred order."""
    return [d for d in DATASET_ORDER if d in data]


def load_sample_prefixes(data_dir: str, dataset_name: str) -> dict:
    per_sample_path = os.path.join(data_dir, f"{dataset_name}_per_sample.jsonl")
    if not os.path.exists(per_sample_path):
        return {"prefixes": [], "prefix_map": {}}

    prefixes = []
    with open(per_sample_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            prefix = rec.get("question_prefix")
            if prefix:
                prefixes.append(prefix)

    prefix_map = {}
    for prefix in prefixes:
        key = prefix[:16]
        prefix_map.setdefault(key, []).append(prefix)

    return {"prefixes": prefixes, "prefix_map": prefix_map}


def load_cloud_metrics(dataset_name: str, cloud_model: str, cache: dict, data_dir: str) -> dict:
    if dataset_name not in CLOUD_DATASET_MAP:
        raise ValueError(f"No cloud dataset mapping for {dataset_name}")
    cache_entry = cache.get(dataset_name, {}).get(cloud_model)
    if cache_entry:
        return cache_entry
    if load_dataset is None:
        raise RuntimeError("datasets library not available; install it to load cloud metrics.")
    hf_name = CLOUD_DATASET_MAP[dataset_name]
    ds = load_dataset(hf_name, split="train", streaming=True)

    sample_filter = load_sample_prefixes(data_dir, dataset_name)
    prefixes = sample_filter["prefixes"]
    prefix_map = sample_filter["prefix_map"]
    remaining_prefixes = set(prefixes)

    correct = 0
    total = 0
    input_tokens = 0.0
    output_tokens = 0.0
    token_rows = 0
    matched = 0

    for row in ds:
        if prefixes:
            problem = row.get("problem", "")
            key = problem[:16]
            candidates = prefix_map.get(key, [])
            matched_prefix = None
            for p in candidates:
                if p in remaining_prefixes and problem.startswith(p):
                    matched_prefix = p
                    break
            if matched_prefix is None:
                continue
            remaining_prefixes.remove(matched_prefix)
        model_metrics = row.get("model_metrics", {})
        metrics = model_metrics.get(cloud_model)
        if not metrics:
            continue
        evaluation = metrics.get("evaluation", {})
        is_correct = evaluation.get("is_correct")
        if is_correct is not None:
            total += 1
            correct += 1 if is_correct else 0
        token_metrics = metrics.get("token_metrics") or {}
        in_tok = token_metrics.get("input")
        out_tok = token_metrics.get("output")
        if in_tok is not None and out_tok is not None:
            input_tokens += float(in_tok)
            output_tokens += float(out_tok)
            token_rows += 1
        matched += 1

    if prefixes and matched != len(prefixes):
        raise ValueError(
            f"Cloud match mismatch for {dataset_name}: matched {matched} of {len(prefixes)} "
            f"prefixes. Consider using longer/unique prefixes."
        )

    cloud_acc = (correct / total) if total > 0 else None
    avg_in = (input_tokens / token_rows) if token_rows > 0 else None
    avg_out = (output_tokens / token_rows) if token_rows > 0 else None

    cache.setdefault(dataset_name, {})[cloud_model] = {
        "cloud_accuracy": cloud_acc,
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "num_eval": total,
        "num_token_rows": token_rows,
        "num_matched": matched,
        "num_prefixes": len(prefixes),
        "dataset": hf_name,
    }
    return cache[dataset_name][cloud_model]


def compute_threshold_table(data: dict, cloud_info: dict) -> dict:
    tables = {}
    for dataset_name, routing_data in data.items():
        routing = routing_data.get("routing", {})
        cloud = cloud_info.get(dataset_name, {})
        cloud_acc = cloud.get("cloud_accuracy")
        avg_in = cloud.get("avg_input_tokens") or AVG_INPUT_TOKENS
        avg_out = cloud.get("avg_output_tokens") or AVG_OUTPUT_TOKENS

        if cloud_acc is None:
            raise ValueError(f"Missing cloud accuracy for {dataset_name}")

        cloud_cost = compute_cost_per_query(avg_in, avg_out, "cloud")
        local_cost = compute_cost_per_query(avg_in, avg_out, "local")

        rows = []
        for thresh_str in sorted(routing.keys(), key=float):
            r = routing[thresh_str]
            cov = r.get("coverage", 0)
            local_acc = r.get("local_accuracy")
            if local_acc is None:
                system_acc = cloud_acc
            else:
                system_acc = cov * local_acc + (1 - cov) * cloud_acc
            acc_delta = system_acc - cloud_acc
            router_cost = cov * local_cost + (1 - cov) * cloud_cost
            savings = cloud_cost - router_cost
            savings_pct = (savings / cloud_cost) if cloud_cost > 0 else 0.0

            rows.append({
                "threshold": float(thresh_str),
                "coverage": cov,
                "local_accuracy": local_acc,
                "system_accuracy": system_acc,
                "accuracy_delta": acc_delta,
                "cloud_accuracy": cloud_acc,
                "cloud_cost": cloud_cost,
                "router_cost": router_cost,
                "savings": savings,
                "savings_pct": savings_pct,
                "avg_input_tokens": avg_in,
                "avg_output_tokens": avg_out,
            })

        tables[dataset_name] = rows
    return tables


def fig1_coverage_accuracy_curves(data: dict, output_path: str):
    """
    Coverage vs Local Accuracy curves for all datasets on one plot.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    datasets = get_ordered_datasets(data)
    
    for dataset_name in datasets:
        routing_data = data[dataset_name]
        routing = routing_data.get("routing", {})
        
        coverages = []
        local_accs = []
        thresholds = []
        
        for thresh_str in sorted(routing.keys(), key=float):
            r = routing[thresh_str]
            cov = r.get("coverage")
            acc = r.get("local_accuracy")
            if cov is not None and acc is not None:
                coverages.append(cov * 100)
                local_accs.append(acc * 100)
                thresholds.append(float(thresh_str))
        
        if coverages:
            color = COLORS.get(dataset_name, "#333333")
            label = DATASET_DISPLAY.get(dataset_name, dataset_name)
            ax.plot(coverages, local_accs, 'o-', color=color, label=label, 
                    linewidth=2.5, markersize=7, alpha=0.9)
            
            # Annotate τ=0.5 point
            if 0.5 in thresholds:
                idx = thresholds.index(0.5)
                ax.annotate(f"τ=0.5", (coverages[idx], local_accs[idx]),
                           textcoords="offset points", xytext=(8, -12),
                           fontsize=9, color=color, fontweight='bold')
    
    ax.set_xlabel("Coverage (%)", fontweight='bold')
    ax.set_ylabel("Local Accuracy (%)", fontweight='bold')
    ax.set_title("Routing Tradeoff: Coverage vs Local Accuracy", fontweight='bold', pad=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(50, 100)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig2_cost_pareto(tables: dict, output_path: str):
    """
    Cost vs System Accuracy Pareto curves.
    System accuracy uses closed-source cloud accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = get_ordered_datasets(tables)

    all_system_accs = []
    all_costs = []

    for dataset_name in datasets:
        rows = tables[dataset_name]
        system_accs = [r["system_accuracy"] * 100 for r in rows if r["system_accuracy"] is not None]
        costs = [r["router_cost"] * 1000 for r in rows if r["system_accuracy"] is not None]
        thresholds = [r["threshold"] for r in rows if r["system_accuracy"] is not None]

        if system_accs:
            color = COLORS.get(dataset_name, "#333")
            label = DATASET_DISPLAY.get(dataset_name, dataset_name)
            ax.plot(system_accs, costs, 'o-', color=color, label=label,
                    linewidth=2.5, markersize=7)
            all_system_accs.extend(system_accs)
            all_costs.extend(costs)

            for i, thresh in enumerate(thresholds):
                if thresh in [0.3, 0.5, 0.7]:
                    y_off = 0.3 if thresh == 0.5 else (-0.2 if thresh == 0.7 else 0.15)
                    ax.annotate(f"τ={thresh}", (system_accs[i], costs[i]),
                                textcoords="offset points", xytext=(5, 8),
                                fontsize=8, color=color, alpha=0.8)

            cloud_acc = rows[0]["cloud_accuracy"] * 100
            cloud_cost = rows[0]["cloud_cost"] * 1000
            ax.scatter([cloud_acc], [cloud_cost], marker="*", s=130, color=color, zorder=4)
            ax.annotate("All cloud", (cloud_acc, cloud_cost),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=8, color=color, alpha=0.8)
    
    ax.set_xlabel("System Accuracy (%)", fontweight='bold')
    ax.set_ylabel("Cost per 1K Queries ($)", fontweight='bold')
    ax.set_title("Cost vs System Accuracy Pareto Frontier\n(cloud model = GPT-5)", 
                fontweight='bold', pad=10)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    if all_system_accs:
        min_acc = min(all_system_accs)
        max_acc = max(all_system_accs)
        padding = max(2, (max_acc - min_acc) * 0.1)
        ax.set_xlim(max(0, min_acc - padding), min(105, max_acc + padding))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig3_cost_savings_at_target(tables: dict, output_path: str, target_accuracy: float = 0.90):
    """
    Bar chart showing $ saved per 1K queries at >= target system accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = get_ordered_datasets(tables)
    n = len(datasets)

    savings = []
    thresholds_used = []
    coverages = []
    all_cloud_costs = []
    
    for dataset_name in datasets:
        rows = tables[dataset_name]

        # Find the threshold that achieves >= target with lowest cost
        best_saving = None
        best_thresh = None
        best_cov = None
        best_cloud_cost = None
        
        for r in rows:
            sys_acc = r["system_accuracy"]
            cov = r["coverage"]
            if sys_acc is not None and sys_acc >= target_accuracy:
                saving = r["savings"] * 1000
                cloud_cost = r["cloud_cost"] * 1000
                if best_saving is None or cov > best_cov:
                    best_saving = saving
                    best_thresh = r["threshold"]
                    best_cov = cov
                    best_cloud_cost = cloud_cost
        
        if best_saving is not None:
            savings.append(best_saving)
            thresholds_used.append(best_thresh)
            coverages.append(best_cov)
            all_cloud_costs.append(best_cloud_cost)
        else:
            savings.append(0)
            thresholds_used.append(None)
            coverages.append(0)
            all_cloud_costs.append(0)
    
    x = np.arange(n)
    colors = [COLORS.get(d, "#333") for d in datasets]
    bars = ax.bar(x, savings, color=colors, edgecolor='black', linewidth=1)
    
    # Add annotations
    for i, (bar, thresh, cov) in enumerate(zip(bars, thresholds_used, coverages)):
        height = bar.get_height()
        if thresh is not None and height > 0:
            ax.annotate(f"τ={thresh}\n{cov*100:.0f}% local", 
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        elif height == 0:
            ax.annotate("Cannot\nhit target", 
                       xy=(bar.get_x() + bar.get_width()/2, 0.1),
                       ha='center', va='bottom', fontsize=8, color='red')
    
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets])
    ax.set_ylabel("$ Saved per 1K Queries (vs All-Cloud)", fontweight='bold')
    ax.set_title(f"Cost Savings at ≥{int(target_accuracy*100)}% System Accuracy", 
                fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    if any(all_cloud_costs):
        baseline = float(np.mean([v for v in all_cloud_costs if v > 0]))
        ax.axhline(y=baseline, color='red', linestyle=':', alpha=0.3)
        ax.text(n-0.5, baseline, f"Mean all-cloud: ${baseline:.2f}/1K", 
               fontsize=8, color='red', alpha=0.7, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig4_metrics_comparison(data: dict, output_path: str):
    """
    Bar chart comparing key metrics across datasets.
    """
    datasets = get_ordered_datasets(data)
    n = len(datasets)
    
    metrics = {
        "Accuracy": [data[d].get("accuracy", 0) for d in datasets],
        "AUROC": [data[d].get("auroc", 0) for d in datasets],
        "ECE": [data[d].get("ece", 0) for d in datasets],
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    x = np.arange(n)
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(x, values, color=[COLORS.get(d, "#333") for d in datasets], 
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets], 
                          rotation=25, ha='right')
        ax.set_title(metric_name, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle("Performance Metrics by Dataset", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig5_reliability_diagrams(data: dict, data_dir: str, output_path: str):
    """
    Reliability diagrams (calibration plots) for each dataset.
    """
    datasets = get_ordered_datasets(data)
    datasets = [d for d in datasets if os.path.exists(
        os.path.join(data_dir, f"{d}_per_sample.jsonl"))]
    
    if not datasets:
        print("No per_sample.jsonl files found, skipping reliability diagrams")
        return
    
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    axes = axes.flatten()
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for i, dataset_name in enumerate(datasets):
        ax = axes[i]
        per_sample_path = os.path.join(data_dir, f"{dataset_name}_per_sample.jsonl")
        confidences, labels = load_per_sample(per_sample_path)
        
        # Compute reliability
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for j in range(n_bins):
            if j == n_bins - 1:
                mask = (confidences >= bin_edges[j]) & (confidences <= bin_edges[j+1])
            else:
                mask = (confidences >= bin_edges[j]) & (confidences < bin_edges[j+1])
            
            count = mask.sum()
            bin_counts.append(count)
            if count > 0:
                bin_accs.append(labels[mask].mean())
                bin_confs.append(confidences[mask].mean())
            else:
                bin_accs.append(np.nan)
                bin_confs.append((bin_edges[j] + bin_edges[j+1]) / 2)
        
        color = COLORS.get(dataset_name, "#333")
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect')
        
        # Reliability curve
        ax.plot(bin_confs, bin_accs, 'o-', color=color, linewidth=2.5, markersize=7)
        
        # Histogram of confidences
        bar_heights = np.array(bin_counts) / max(bin_counts) * 0.25
        bar_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bar_centers, bar_heights, width=0.08, alpha=0.3, color=color, bottom=0)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(DATASET_DISPLAY.get(dataset_name, dataset_name), fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # ECE annotation
        ece = data[dataset_name].get("ece", 0)
        ax.text(0.95, 0.05, f"ECE={ece:.3f}", transform=ax.transAxes,
               ha='right', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle("Reliability Diagrams (Calibration)", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig6_confidence_distributions(data: dict, data_dir: str, output_path: str):
    """
    Confidence score distributions for correct vs incorrect predictions.
    """
    datasets = get_ordered_datasets(data)
    datasets = [d for d in datasets if os.path.exists(
        os.path.join(data_dir, f"{d}_per_sample.jsonl"))]
    
    if not datasets:
        print("No per_sample.jsonl files found, skipping confidence distributions")
        return
    
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    axes = axes.flatten()
    
    for i, dataset_name in enumerate(datasets):
        ax = axes[i]
        per_sample_path = os.path.join(data_dir, f"{dataset_name}_per_sample.jsonl")
        confidences, labels = load_per_sample(per_sample_path)
        
        correct_conf = confidences[labels == 1]
        incorrect_conf = confidences[labels == 0]
        
        bins = np.linspace(0, 1, 21)
        ax.hist(correct_conf, bins=bins, alpha=0.6, label='Correct', color='#27ae60', density=True)
        ax.hist(incorrect_conf, bins=bins, alpha=0.6, label='Incorrect', color='#c0392b', density=True)
        
        ax.axvline(correct_conf.mean(), color='#27ae60', linestyle='--', linewidth=2.5)
        ax.axvline(incorrect_conf.mean(), color='#c0392b', linestyle='--', linewidth=2.5)
        
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.set_title(DATASET_DISPLAY.get(dataset_name, dataset_name), fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)
        
        # AUROC annotation
        auroc = data[dataset_name].get("auroc", 0)
        ax.text(0.05, 0.95, f"AUROC={auroc:.3f}", transform=ax.transAxes,
               va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle("Confidence Distributions: Correct vs Incorrect", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def write_tables(tables: dict, output_dir: str):
    json_path = os.path.join(output_dir, "router_cost_accuracy_tables.json")
    with open(json_path, "w") as f:
        json.dump(tables, f, indent=2)

    csv_path = os.path.join(output_dir, "router_cost_accuracy_tables.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,threshold,coverage,local_accuracy,system_accuracy,accuracy_delta,"
                "cloud_accuracy,cloud_cost,router_cost,savings,savings_pct,avg_input_tokens,avg_output_tokens\n")
        for dataset_name, rows in tables.items():
            for r in rows:
                f.write(
                    f"{dataset_name},{r['threshold']},{r['coverage']},{r['local_accuracy']},"
                    f"{r['system_accuracy']},{r['accuracy_delta']},{r['cloud_accuracy']},"
                    f"{r['cloud_cost']},{r['router_cost']},{r['savings']},{r['savings_pct']},"
                    f"{r['avg_input_tokens']},{r['avg_output_tokens']}\n"
                )

    print(f"Saved tables: {json_path}, {csv_path}")


def print_summary(data: dict, tables: dict):
    """Print a text summary of the results."""
    datasets = get_ordered_datasets(data)

    print("\n" + "="*96)
    print("ROUTING EVALUATION SUMMARY")
    print("="*96)

    print(f"\nCost assumptions: Local=${LOCAL_COST_PER_1M}/1M tok, "
          f"Cloud=${CLOUD_INPUT_COST_PER_1M}/${CLOUD_OUTPUT_COST_PER_1M} (in/out)")

    print(f"\n{'Dataset':<18} {'Acc':>7} {'CloudAcc':>9} {'Δ@0.5':>7} {'Cov@0.5':>9} {'SysAcc@0.5':>11}")
    print("-"*96)

    for dataset_name in datasets:
        d = data[dataset_name]
        rows = tables[dataset_name]
        row_05 = next((r for r in rows if abs(r["threshold"] - 0.5) < 1e-9), None)
        cov = (row_05["coverage"] * 100) if row_05 else 0
        sys_acc = (row_05["system_accuracy"] * 100) if row_05 else 0
        delta = (row_05["accuracy_delta"] * 100) if row_05 else 0
        cloud_acc = (row_05["cloud_accuracy"] * 100) if row_05 else 0

        print(f"{DATASET_DISPLAY.get(dataset_name, dataset_name):<18} "
              f"{d.get('accuracy', 0)*100:>6.1f}% "
              f"{cloud_acc:>8.1f}% "
              f"{delta:>6.1f}% "
              f"{cov:>8.1f}% "
              f"{sys_acc:>10.1f}%")

    print("="*96 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize routing evaluation results")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing *_routing.json files")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for figures (default: same as data-dir)")
    parser.add_argument("--target-accuracy", type=float, default=0.90,
                       help="Target system accuracy for cost savings chart (default: 0.90)")
    parser.add_argument("--cloud-model", type=str, default=CLOUD_MODEL_DEFAULT,
                       help=f"Cloud model to compare against (default: {CLOUD_MODEL_DEFAULT})")
    parser.add_argument("--cloud-cache", type=str, default=None,
                       help="Optional path to cache cloud metrics JSON")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_all_data(data_dir)
    
    if not data:
        raise ValueError(f"No *_routing.json files found in {data_dir}")
    
    print(f"Loaded {len(data)} datasets: {list(data.keys())}")

    # Load cloud metrics
    cache_path = args.cloud_cache or os.path.join(output_dir, "cloud_metrics_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cloud_cache = json.load(f)
    else:
        cloud_cache = {}

    cloud_info = {}
    for dataset_name in get_ordered_datasets(data):
        cloud_info[dataset_name] = load_cloud_metrics(dataset_name, args.cloud_model, cloud_cache, data_dir)

    with open(cache_path, "w") as f:
        json.dump(cloud_cache, f, indent=2)

    # Build tables
    tables = compute_threshold_table(data, cloud_info)
    write_tables(tables, output_dir)

    # Print summary
    print_summary(data, tables)

    # Generate figures
    fig1_coverage_accuracy_curves(data, os.path.join(output_dir, "fig1_routing_curves.png"))
    fig2_cost_pareto(tables, os.path.join(output_dir, "fig2_cost_pareto.png"))
    fig3_cost_savings_at_target(tables, output_dir + "/fig3_cost_savings.png", args.target_accuracy)
    fig4_metrics_comparison(data, os.path.join(output_dir, "fig4_metrics.png"))
    fig5_reliability_diagrams(data, data_dir, os.path.join(output_dir, "fig5_reliability.png"))
    fig6_confidence_distributions(data, data_dir, os.path.join(output_dir, "fig6_distributions.png"))
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
