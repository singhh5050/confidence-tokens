#!/usr/bin/env python3
"""
Analyze cross-dataset routing evaluation results.

Reads all *_routing.json files and produces a summary matrix.

Usage:
    python scripts/analyze_routing.py outputs/routing_eval_*/*_routing.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def extract_model_dataset(filepath: str):
    """Extract model and dataset names from filepath."""
    # Path like: outputs/routing_eval_b_suffix_wildchat_on_mmlu/mmlu_routing.json
    dirname = Path(filepath).parent.name
    filename = Path(filepath).stem  # e.g., "mmlu_routing"
    
    # Skip combined results files
    if "all_datasets" in filename:
        return None, None
    
    dataset = filename.replace("_routing", "")
    
    # Extract model from dirname
    # routing_eval_b_suffix_wildchat_on_mmlu -> b_suffix_wildchat
    if "_on_" in dirname:
        model = dirname.replace("routing_eval_", "").split("_on_")[0]
    else:
        model = dirname.replace("routing_eval_", "")
    
    return model, dataset


def main():
    parser = argparse.ArgumentParser(description="Analyze routing results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Routing threshold")
    args = parser.parse_args()
    
    # Collect results
    results = {}  # {(model, dataset): metrics}
    
    for filepath in args.files:
        model, dataset = extract_model_dataset(filepath)
        if model is None:
            continue
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            results[(model, dataset)] = data
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
    
    if not results:
        print("No valid results found!")
        return
    
    # Get unique models and datasets
    models = sorted(set(m for m, d in results.keys()))
    datasets = sorted(set(d for m, d in results.keys()))
    
    # Print AUROC matrix
    print("\n" + "=" * 80)
    print("AUROC MATRIX (rows=model trained on, cols=evaluated on)")
    print("=" * 80)
    print(f"{'Model':<30}", end="")
    for d in datasets:
        print(f"{d[:12]:>14}", end="")
    print()
    print("-" * 80)
    
    for m in models:
        print(f"{m:<30}", end="")
        for d in datasets:
            key = (m, d)
            if key in results:
                auroc = results[key].get("auroc")
                if auroc is not None:
                    # Highlight diagonal (in-distribution)
                    is_diagonal = (m == "b_suffix" and d == "mmlu") or \
                                  (m == "b_suffix" and d == "mmlu_pro") or \
                                  (d in m)
                    val = f"{auroc:.3f}"
                    if is_diagonal:
                        val = f"*{auroc:.3f}*"
                    print(f"{val:>14}", end="")
                else:
                    print(f"{'N/A':>14}", end="")
            else:
                print(f"{'-':>14}", end="")
        print()
    
    # Print Coverage @ threshold matrix
    thresh_key = str(args.threshold)
    print("\n" + "=" * 80)
    print(f"COVERAGE @ {args.threshold} (fraction routed to local model)")
    print("=" * 80)
    print(f"{'Model':<30}", end="")
    for d in datasets:
        print(f"{d[:12]:>14}", end="")
    print()
    print("-" * 80)
    
    for m in models:
        print(f"{m:<30}", end="")
        for d in datasets:
            key = (m, d)
            if key in results and "routing" in results[key]:
                routing = results[key]["routing"].get(thresh_key, {})
                cov = routing.get("coverage")
                if cov is not None:
                    print(f"{cov:.1%}".rjust(14), end="")
                else:
                    print(f"{'N/A':>14}", end="")
            else:
                print(f"{'-':>14}", end="")
        print()
    
    # Print Local Accuracy @ threshold matrix
    print("\n" + "=" * 80)
    print(f"LOCAL ACCURACY @ {args.threshold} (accuracy on samples routed locally)")
    print("=" * 80)
    print(f"{'Model':<30}", end="")
    for d in datasets:
        print(f"{d[:12]:>14}", end="")
    print()
    print("-" * 80)
    
    for m in models:
        print(f"{m:<30}", end="")
        for d in datasets:
            key = (m, d)
            if key in results and "routing" in results[key]:
                routing = results[key]["routing"].get(thresh_key, {})
                acc = routing.get("local_accuracy")
                if acc is not None:
                    print(f"{acc:.1%}".rjust(14), end="")
                else:
                    print(f"{'N/A':>14}", end="")
            else:
                print(f"{'-':>14}", end="")
        print()
    
    # Print base accuracy for context
    print("\n" + "=" * 80)
    print("BASE ACCURACY (Olmo-7B-Think on each dataset)")
    print("=" * 80)
    print(f"{'Model':<30}", end="")
    for d in datasets:
        print(f"{d[:12]:>14}", end="")
    print()
    print("-" * 80)
    
    for m in models:
        print(f"{m:<30}", end="")
        for d in datasets:
            key = (m, d)
            if key in results:
                acc = results[key].get("accuracy")
                if acc is not None:
                    print(f"{acc:.1%}".rjust(14), end="")
                else:
                    print(f"{'N/A':>14}", end="")
            else:
                print(f"{'-':>14}", end="")
        print()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Average AUROC by model
    print("\nAverage AUROC by model (across all eval datasets):")
    for m in models:
        aurocs = [results[(m, d)].get("auroc") for d in datasets 
                  if (m, d) in results and results[(m, d)].get("auroc") is not None]
        if aurocs:
            print(f"  {m}: {sum(aurocs)/len(aurocs):.3f} (n={len(aurocs)})")
    
    # In-distribution vs out-of-distribution
    print("\nIn-distribution vs Out-of-distribution AUROC:")
    id_aurocs = []
    ood_aurocs = []
    for (m, d), data in results.items():
        auroc = data.get("auroc")
        if auroc is None:
            continue
        # Check if in-distribution (model trained on same dataset)
        is_id = (m == "b_suffix" and d in ["mmlu", "mmlu_pro"]) or (d in m)
        if is_id:
            id_aurocs.append(auroc)
        else:
            ood_aurocs.append(auroc)
    
    if id_aurocs:
        print(f"  In-distribution:  {sum(id_aurocs)/len(id_aurocs):.3f} (n={len(id_aurocs)})")
    if ood_aurocs:
        print(f"  Out-of-distribution: {sum(ood_aurocs)/len(ood_aurocs):.3f} (n={len(ood_aurocs)})")


if __name__ == "__main__":
    main()

