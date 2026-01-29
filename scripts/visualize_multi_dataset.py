#!/usr/bin/env python3
"""
Visualize multi-dataset routing evaluation results.

Creates figures that focus on a multi-dataset trained model and compares it
against single-dataset baselines when available.

Usage:
    python scripts/visualize_multi_dataset.py --results-dir outputs
    python scripts/visualize_multi_dataset.py --results-dir outputs --model-name B_suffix_multi_all4_20250101_120000
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")

DATASET_ORDER = ["mmlu", "supergpqa", "wildchat", "natural_reasoning"]
DATASET_DISPLAY = {
    "mmlu": "MMLU",
    "mmlu_pro": "MMLU",
    "supergpqa": "SuperGPQA",
    "wildchat": "WildChat",
    "natural_reasoning": "NatReas",
}

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def discover_models(results_dir: str) -> dict:
    """
    Discover model directories (training outputs) and classify as single/multi.

    Returns:
        {
            "single": {model_name: {"path": ..., "metadata": {...}}},
            "multi": {model_name: {"path": ..., "metadata": {...}}},
        }
    """
    models = {"single": {}, "multi": {}}
    for entry in os.listdir(results_dir):
        model_path = os.path.join(results_dir, entry)
        if not os.path.isdir(model_path):
            continue
        meta_path = os.path.join(model_path, "split_metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        is_multi = meta.get("metadata_schema") == "multi" or meta.get("is_multi_dataset", False)
        key = "multi" if is_multi else "single"
        models[key][entry] = {"path": model_path, "metadata": meta}
    return models


def load_routing_results(results_dir: str) -> dict:
    """
    Load routing evaluation JSONs from routing_eval_* directories.

    Returns dict keyed by (model_name, dataset_name).
    """
    results = {}
    eval_dirs = glob(os.path.join(results_dir, "routing_eval_*"))
    for eval_dir in eval_dirs:
        json_files = glob(os.path.join(eval_dir, "*_routing.json"))
        for json_path in json_files:
            filename = os.path.basename(json_path)
            if "all_datasets" in filename:
                continue
            dir_name = os.path.basename(eval_dir)
            if "_on_" in dir_name:
                model_name = dir_name.replace("routing_eval_", "").split("_on_")[0]
                dataset_name = filename.replace("_routing.json", "")
            else:
                model_name = dir_name.replace("routing_eval_", "")
                dataset_name = filename.replace("_routing.json", "")
            try:
                data = read_json(json_path)
            except Exception:
                continue
            data["_source_dir"] = eval_dir
            results[(model_name, dataset_name)] = data
    return results


def normalize_dataset_name(dataset_name: str, available_names: set) -> str:
    if dataset_name in available_names:
        return dataset_name
    if dataset_name == "mmlu_pro" and "mmlu" in available_names:
        return "mmlu"
    if dataset_name == "mmlu" and "mmlu_pro" in available_names:
        return "mmlu_pro"
    return dataset_name


def get_result(results: dict, model_name: str, dataset_name: str) -> dict | None:
    available = {d for (_, d) in results.keys() if _ == model_name}
    ds_key = normalize_dataset_name(dataset_name, available)
    return results.get((model_name, ds_key))


def load_per_sample(eval_dir: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(eval_dir, f"{dataset_name}_per_sample.jsonl")
    if not os.path.exists(path):
        alt = os.path.join(eval_dir, f"mmlu_per_sample.jsonl")
        if dataset_name == "mmlu_pro" and os.path.exists(alt):
            path = alt
        else:
            return None, None
    confidences = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            confidences.append(record["confidence"])
            labels.append(record["label"])
    if not confidences:
        return None, None
    return np.array(confidences), np.array(labels)


def compute_reliability(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10):
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


def plot_multi_performance_grid(
    model_name: str,
    dataset_list: list,
    results: dict,
    output_path: str,
):
    n_cols = max(1, len(dataset_list))
    fig, axes = plt.subplots(1, n_cols, figsize=(max(8, n_cols * 3.4), 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for j, dataset in enumerate(dataset_list):
        ax = axes[j]
        data = get_result(results, model_name, dataset)
        if not data:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", color="gray")
            continue

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
            ax.plot(coverages, local_accs, marker="o", color="tab:blue", linewidth=2, markersize=4)
            if 0.5 in thresh_list:
                idx_05 = thresh_list.index(0.5)
                ax.annotate(
                    "τ=0.5",
                    (coverages[idx_05], local_accs[idx_05]),
                    textcoords="offset points",
                    xytext=(-15, 8),
                    fontsize=8,
                    color="tab:blue",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
            auroc = data.get("auroc")
            if auroc:
                ax.text(
                    0.95,
                    0.05,
                    f"AUROC={auroc:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.set_ylim(30, 100)
        ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01, "Coverage (% routed locally)", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "Local Accuracy (%)", va="center", rotation="vertical", fontsize=12)
    plt.suptitle(
        f"Multi-Dataset Model Performance ({model_name})\n(All datasets in-distribution)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0.04, 0.05, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1: {output_path}")


def build_single_baselines(single_models: dict) -> dict:
    """
    Map dataset -> single-dataset model trained on it.
    """
    mapping = {}
    for model_name, info in single_models.items():
        meta = info["metadata"]
        dataset_name = meta.get("dataset_name")
        if dataset_name:
            mapping[dataset_name] = model_name
    return mapping


def plot_single_vs_multi_comparison(
    model_name: str,
    dataset_list: list,
    results: dict,
    single_models: dict,
    output_path: str,
):
    baselines = build_single_baselines(single_models)
    metrics = [
        ("AUROC", "auroc", (0.4, 1.0)),
        ("Coverage@0.5", "coverage", (0, 1.0)),
        ("Local Acc@0.5", "local_accuracy", (0, 1.0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(dataset_list))
    width = 0.35

    for ax, (title, key, ylim) in zip(axes, metrics):
        multi_vals = []
        single_vals = []
        for dataset in dataset_list:
            multi_data = get_result(results, model_name, dataset) or {}
            routing = multi_data.get("routing", {}).get("0.5", {})
            multi_val = routing.get(key) if key != "auroc" else multi_data.get("auroc")
            if multi_val is None:
                multi_vals.append(np.nan)
            else:
                multi_vals.append(multi_val)

            baseline_model = baselines.get(dataset) or baselines.get("mmlu_pro") if dataset == "mmlu" else baselines.get(dataset)
            base_data = get_result(results, baseline_model, dataset) if baseline_model else None
            if base_data:
                base_routing = base_data.get("routing", {}).get("0.5", {})
                base_val = base_routing.get(key) if key != "auroc" else base_data.get("auroc")
                single_vals.append(base_val if base_val is not None else np.nan)
            else:
                single_vals.append(np.nan)

        ax.bar(x - width / 2, single_vals, width, label="Single-dataset", color="tab:gray")
        ax.bar(x + width / 2, multi_vals, width, label="Multi-dataset", color="tab:blue")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in dataset_list], rotation=45, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, axis="y")

        for i, val in enumerate(single_vals):
            if np.isnan(val):
                ax.text(i - width / 2, ylim[0] + 0.02, "N/A", ha="center", fontsize=8, color="gray", rotation=90)
        for i, val in enumerate(multi_vals):
            if np.isnan(val):
                ax.text(i + width / 2, ylim[0] + 0.02, "N/A", ha="center", fontsize=8, color="gray", rotation=90)

    axes[0].legend(loc="lower left", fontsize=9)
    plt.suptitle("Single vs Multi-Dataset Comparison (τ=0.5)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2: {output_path}")


def plot_ood_generalization(
    results: dict,
    single_models: dict,
    multi_model_name: str,
    dataset_list: list,
    output_path: str,
):
    model_names = list(single_models.keys()) + [multi_model_name]
    id_vals = []
    ood_vals = []
    labels = []

    for model_name in model_names:
        if model_name == multi_model_name:
            vals = []
            for dataset in dataset_list:
                data = get_result(results, model_name, dataset)
                if data and data.get("auroc") is not None:
                    vals.append(data.get("auroc"))
            id_vals.append(np.mean(vals) if vals else np.nan)
            ood_vals.append(np.nan)
            labels.append("multi")
            continue

        meta = single_models[model_name]["metadata"]
        id_dataset = meta.get("dataset_name")
        if not id_dataset:
            continue
        id_data = get_result(results, model_name, id_dataset) or {}
        id_vals.append(id_data.get("auroc", np.nan))

        ood_list = []
        for dataset in dataset_list:
            if dataset in [id_dataset, "mmlu"] and id_dataset == "mmlu_pro":
                continue
            if dataset == "mmlu" and id_dataset == "mmlu":
                continue
            data = get_result(results, model_name, dataset)
            if data and data.get("auroc") is not None:
                ood_list.append(data.get("auroc"))
        ood_vals.append(np.mean(ood_list) if ood_list else np.nan)
        labels.append(DATASET_DISPLAY.get(id_dataset, id_dataset))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, id_vals, width, label="In-distribution", color="tab:blue")
    ax.bar(x + width / 2, ood_vals, width, label="Out-of-distribution", color="tab:orange")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("OOD Generalization (AUROC)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3: {output_path}")


def plot_multi_reliability(
    model_name: str,
    dataset_list: list,
    results: dict,
    output_path: str,
):
    n_cols = max(1, len(dataset_list))
    fig, axes = plt.subplots(1, n_cols, figsize=(max(8, n_cols * 3.4), 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for j, dataset in enumerate(dataset_list):
        ax = axes[j]
        data = get_result(results, model_name, dataset)
        if not data:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", color="gray")
            continue
        eval_dir = data.get("_source_dir")
        confidences, labels = load_per_sample(eval_dir, dataset)
        if confidences is None:
            ax.text(0.5, 0.5, "No per-sample data", transform=ax.transAxes, ha="center", va="center", color="gray")
            continue

        _, bin_conf, bin_acc, counts = compute_reliability(confidences, labels, n_bins=10)
        ax.plot([0, 1], [0, 1], color="black", alpha=0.3, linewidth=1)
        ax.plot(bin_conf, bin_acc, marker="o", color="tab:blue", linewidth=2, markersize=4)
        avg_count = int(np.nanmean(counts)) if np.nanmean(counts) else 0
        ax.text(0.05, 0.9, f"avg n/bin={avg_count}", transform=ax.transAxes, fontsize=7, color="gray")
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01, "Mean Confidence", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "Empirical Accuracy", va="center", rotation="vertical", fontsize=12)
    plt.suptitle(
        f"Reliability Diagrams (Multi-Dataset: {model_name})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0.04, 0.05, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4: {output_path}")


def write_summary_csv(
    model_name: str,
    dataset_list: list,
    results: dict,
    single_models: dict,
    output_path: str,
):
    baselines = build_single_baselines(single_models)
    lines = [
        "dataset,multi_auroc,multi_accuracy,multi_coverage_0.5,multi_local_acc_0.5,"
        "single_model,single_auroc,single_accuracy,single_coverage_0.5,single_local_acc_0.5"
    ]
    for dataset in dataset_list:
        multi_data = get_result(results, model_name, dataset) or {}
        multi_routing = multi_data.get("routing", {}).get("0.5", {})
        baseline_model = baselines.get(dataset) or baselines.get("mmlu_pro") if dataset == "mmlu" else baselines.get(dataset)
        base_data = get_result(results, baseline_model, dataset) if baseline_model else {}
        base_routing = base_data.get("routing", {}).get("0.5", {}) if base_data else {}
        line = [
            DATASET_DISPLAY.get(dataset, dataset),
            multi_data.get("auroc"),
            multi_data.get("accuracy"),
            multi_routing.get("coverage"),
            multi_routing.get("local_accuracy"),
            baseline_model or "",
            base_data.get("auroc") if base_data else "",
            base_data.get("accuracy") if base_data else "",
            base_routing.get("coverage") if base_data else "",
            base_routing.get("local_accuracy") if base_data else "",
        ]
        lines.append(",".join("" if v is None else str(v) for v in line))
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved summary CSV: {output_path}")


def select_multi_model(multi_models: dict, requested: str | None) -> str:
    if requested:
        if requested in multi_models:
            return requested
        raise ValueError(f"Requested model '{requested}' not found in multi-dataset models: {list(multi_models.keys())}")
    if len(multi_models) == 1:
        return next(iter(multi_models.keys()))
    if not multi_models:
        raise ValueError("No multi-dataset models found. Train or specify --model-name.")
    # Pick most recently modified
    latest = max(multi_models.items(), key=lambda x: os.path.getmtime(x[1]["path"]))
    print(f"ℹ Multiple multi-dataset models found; using most recent: {latest[0]}")
    return latest[0]


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-dataset routing results")
    parser.add_argument("--results-dir", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--model-name", type=str, default=None, help="Multi-dataset model directory name")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for figures and summary")
    args = parser.parse_args()

    results_dir = args.results_dir
    models = discover_models(results_dir)
    all_results = load_routing_results(results_dir)

    if not all_results:
        raise ValueError(f"No routing_eval_* results found in {results_dir}")

    multi_model = select_multi_model(models["multi"], args.model_name)
    meta = models["multi"][multi_model]["metadata"]
    dataset_list = meta.get("dataset_names", DATASET_ORDER)
    dataset_list = [normalize_dataset_name(d, set(DATASET_ORDER + ["mmlu_pro"])) for d in dataset_list]

    output_dir = args.output_dir or os.path.join(results_dir, f"figures_multi_{multi_model}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MULTI-DATASET VISUALIZATION")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    print(f"Multi model: {multi_model}")
    print(f"Datasets: {dataset_list}")
    print(f"Output dir: {output_dir}")

    plot_multi_performance_grid(
        model_name=multi_model,
        dataset_list=dataset_list,
        results=all_results,
        output_path=os.path.join(output_dir, "fig1_multi_performance.png"),
    )

    plot_single_vs_multi_comparison(
        model_name=multi_model,
        dataset_list=dataset_list,
        results=all_results,
        single_models=models["single"],
        output_path=os.path.join(output_dir, "fig2_single_vs_multi.png"),
    )

    plot_ood_generalization(
        results=all_results,
        single_models=models["single"],
        multi_model_name=multi_model,
        dataset_list=dataset_list,
        output_path=os.path.join(output_dir, "fig3_ood_generalization.png"),
    )

    plot_multi_reliability(
        model_name=multi_model,
        dataset_list=dataset_list,
        results=all_results,
        output_path=os.path.join(output_dir, "fig4_multi_reliability.png"),
    )

    write_summary_csv(
        model_name=multi_model,
        dataset_list=dataset_list,
        results=all_results,
        single_models=models["single"],
        output_path=os.path.join(output_dir, "multi_summary.csv"),
    )

    print("\n✓ All multi-dataset visualizations complete.")


if __name__ == "__main__":
    main()
