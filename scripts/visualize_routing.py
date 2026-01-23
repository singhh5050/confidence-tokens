#!/usr/bin/env python3
"""
Visualize routing evaluation results.

Creates Pareto curves and domain breakdown charts.

Usage:
    python scripts/visualize_routing.py \
        --results-dir outputs/routing_eval_b_suffix \
        --output-dir outputs/routing_figures
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']


def load_results(results_dir: str) -> dict:
    """Load all routing evaluation results from directory."""
    results = {}
    
    # Look for individual dataset files
    for filename in os.listdir(results_dir):
        if filename.endswith("_routing.json") and filename != "all_datasets_routing.json":
            dataset_name = filename.replace("_routing.json", "")
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                results[dataset_name] = json.load(f)
    
    # Also try combined file
    combined_path = os.path.join(results_dir, "all_datasets_routing.json")
    if os.path.exists(combined_path) and not results:
        with open(combined_path, "r") as f:
            results = json.load(f)
    
    return results


def plot_pareto_curves(results: dict, output_path: str):
    """
    Plot Pareto curves: Coverage vs Overall Accuracy for each dataset.
    
    X-axis: Coverage (% handled locally)
    Y-axis: Overall Accuracy (assuming cloud is always right)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (dataset_name, metrics) in enumerate(results.items()):
        if "error" in metrics or "routing" not in metrics:
            continue
        
        routing = metrics["routing"]
        thresholds = sorted(routing.keys(), key=float)
        
        coverages = [routing[t]["coverage"] for t in thresholds]
        overall_accs = [routing[t]["overall_accuracy"] for t in thresholds]
        
        color = COLORS[i % len(COLORS)]
        ax.plot(coverages, overall_accs, 'o-', 
                label=dataset_name, color=color, linewidth=2, markersize=8)
        
        # Annotate threshold 0.5 point
        if "0.5" in routing:
            cov_05 = routing["0.5"]["coverage"]
            acc_05 = routing["0.5"]["overall_accuracy"]
            ax.annotate(f'τ=0.5', (cov_05, acc_05), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color=color)
    
    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Cloud only (100%)')
    
    ax.set_xlabel('Coverage (% queries handled locally)', fontsize=12)
    ax.set_ylabel('Overall Accuracy', fontsize=12)
    ax.set_title('Routing Performance: Coverage vs Accuracy Trade-off', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 1.02)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Pareto curves to {output_path}")


def plot_local_accuracy_curves(results: dict, output_path: str):
    """
    Plot Local Accuracy vs Threshold for each dataset.
    
    Shows how accurate the local model is at different confidence thresholds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (dataset_name, metrics) in enumerate(results.items()):
        if "error" in metrics or "routing" not in metrics:
            continue
        
        routing = metrics["routing"]
        thresholds = sorted(routing.keys(), key=float)
        
        thresh_vals = [float(t) for t in thresholds]
        local_accs = [routing[t]["local_accuracy"] or 0 for t in thresholds]
        coverages = [routing[t]["coverage"] for t in thresholds]
        
        color = COLORS[i % len(COLORS)]
        ax.plot(thresh_vals, local_accs, 'o-', 
                label=f'{dataset_name} (acc={metrics["accuracy"]:.1%})', 
                color=color, linewidth=2, markersize=8)
    
    # Reference: baseline accuracy line
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Local Accuracy (on routed queries)', fontsize=12)
    ax.set_title('Local Model Accuracy at Different Confidence Thresholds', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved local accuracy curves to {output_path}")


def plot_coverage_comparison(results: dict, output_path: str, threshold: float = 0.5):
    """
    Bar chart comparing coverage and accuracy across datasets at fixed threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets = []
    coverages = []
    local_accs = []
    overall_accs = []
    aurocs = []
    
    for dataset_name, metrics in results.items():
        if "error" in metrics or "routing" not in metrics:
            continue
        
        routing = metrics["routing"].get(str(threshold), {})
        datasets.append(dataset_name)
        coverages.append(routing.get("coverage", 0) * 100)
        local_accs.append((routing.get("local_accuracy", 0) or 0) * 100)
        overall_accs.append(routing.get("overall_accuracy", 0) * 100)
        aurocs.append(metrics.get("auroc", 0) or 0)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot 1: Coverage and Local Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, coverages, width, label='Coverage', color=COLORS[0])
    bars2 = ax1.bar(x + width/2, local_accs, width, label='Local Accuracy', color=COLORS[1])
    
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_title(f'Coverage vs Local Accuracy (threshold={threshold})', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 105)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Plot 2: AUROC comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x, aurocs, width*2, color=COLORS[2])
    
    ax2.set_ylabel('AUROC', fontsize=11)
    ax2.set_title('Confidence Discrimination (AUROC)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved coverage comparison to {output_path}")


def plot_category_breakdown(results: dict, output_path: str, dataset_name: str = None):
    """
    Plot accuracy and AUROC by category for a specific dataset.
    """
    # Find dataset with category breakdown
    target = None
    for name, metrics in results.items():
        if "by_category" in metrics and len(metrics["by_category"]) > 1:
            if dataset_name is None or name == dataset_name:
                target = (name, metrics)
                break
    
    if target is None:
        print("⚠ No category breakdown available")
        return
    
    name, metrics = target
    categories = metrics["by_category"]
    
    # Sort by count
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True)
    
    # Limit to top 15 categories
    if len(sorted_cats) > 15:
        sorted_cats = sorted_cats[:15]
    
    cat_names = [c[0][:25] for c in sorted_cats]  # Truncate long names
    counts = [c[1]["count"] for c in sorted_cats]
    accs = [c[1]["accuracy"] * 100 for c in sorted_cats]
    aurocs = [c[1].get("auroc", 0) or 0 for c in sorted_cats]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(cat_names))
    
    # Plot 1: Accuracy by category
    ax1 = axes[0]
    bars1 = ax1.barh(x, accs, color=COLORS[0])
    ax1.set_xlabel('Accuracy (%)', fontsize=11)
    ax1.set_title(f'{name}: Accuracy by Category', fontsize=12)
    ax1.set_yticks(x)
    ax1.set_yticklabels(cat_names)
    ax1.set_xlim(0, 105)
    ax1.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        width = bar.get_width()
        ax1.annotate(f'{width:.0f}% (n={count})', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=8)
    
    # Plot 2: AUROC by category
    ax2 = axes[1]
    bars2 = ax2.barh(x, aurocs, color=COLORS[2])
    ax2.set_xlabel('AUROC', fontsize=11)
    ax2.set_title(f'{name}: AUROC by Category', fontsize=12)
    ax2.set_yticks(x)
    ax2.set_yticklabels(cat_names)
    ax2.set_xlim(0, 1.05)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()
    
    for bar in bars2:
        width = bar.get_width()
        if width > 0:
            ax2.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved category breakdown to {output_path}")


def create_summary_table(results: dict, output_path: str):
    """Create a markdown summary table of results."""
    lines = [
        "# Routing Evaluation Results",
        "",
        "## Summary Table",
        "",
        "| Dataset | AUROC | Accuracy | Coverage@0.5 | Local Acc@0.5 | Overall Acc@0.5 |",
        "|---------|-------|----------|--------------|---------------|-----------------|",
    ]
    
    for dataset_name, metrics in results.items():
        if "error" in metrics:
            lines.append(f"| {dataset_name} | ERROR | - | - | - | - |")
            continue
        
        auroc = f"{metrics['auroc']:.3f}" if metrics.get('auroc') else "N/A"
        acc = f"{metrics['accuracy']:.1%}"
        
        routing = metrics.get("routing", {}).get("0.5", {})
        cov = f"{routing.get('coverage', 0):.1%}"
        local_acc = f"{routing.get('local_accuracy', 0):.1%}" if routing.get('local_accuracy') else "N/A"
        overall = f"{routing.get('overall_accuracy', 0):.1%}"
        
        lines.append(f"| {dataset_name} | {auroc} | {acc} | {cov} | {local_acc} | {overall} |")
    
    lines.extend([
        "",
        "## Key Metrics",
        "",
        "- **AUROC**: How well confidence discriminates correct vs incorrect (1.0 = perfect)",
        "- **Accuracy**: Base accuracy of the local model",
        "- **Coverage@0.5**: % of queries handled locally at threshold 0.5",
        "- **Local Acc@0.5**: Accuracy on locally-handled queries",
        "- **Overall Acc@0.5**: Combined accuracy (local + cloud fallback)",
        "",
        "## Interpretation",
        "",
        "Higher coverage with maintained accuracy = better confidence calibration.",
        "Overall accuracy assumes cloud model is always correct.",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize routing evaluation results")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing routing evaluation JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for figures (default: results_dir/figures)")
    args = parser.parse_args()
    
    # Set output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("ROUTING VISUALIZATION")
    print("="*60)
    print(f"Results: {args.results_dir}")
    print(f"Output: {args.output_dir}")
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"\nFound results for {len(results)} datasets: {list(results.keys())}")
    
    # Generate visualizations
    plot_pareto_curves(results, os.path.join(args.output_dir, "pareto_curves.png"))
    plot_local_accuracy_curves(results, os.path.join(args.output_dir, "local_accuracy.png"))
    plot_coverage_comparison(results, os.path.join(args.output_dir, "coverage_comparison.png"))
    
    # Category breakdown for each dataset with categories
    for name in results:
        if "by_category" in results[name]:
            plot_category_breakdown(results, os.path.join(args.output_dir, f"category_{name}.png"), name)
    
    # Summary table
    create_summary_table(results, os.path.join(args.output_dir, "summary.md"))
    
    print(f"\n✓ All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

