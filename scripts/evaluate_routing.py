#!/usr/bin/env python3
"""
Cross-dataset routing evaluation.

Evaluates a trained confidence model (b_suffix) on any dataset for routing analysis.
Does NOT use split_metadata - evaluates on fresh data from specified dataset.

For Approach B models only (uses pre-trained confidence_head).

Usage:
    # Evaluate b_suffix model on SuperGPQA
    python scripts/evaluate_routing.py \
        --model-path outputs/b_suffix \
        --dataset supergpqa \
        --output-dir outputs/routing_eval/b_suffix_on_supergpqa

    # Evaluate on all datasets
    python scripts/evaluate_routing.py \
        --model-path outputs/b_suffix \
        --dataset all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Target model whose traces we use
TARGET_MODEL = "allenai/Olmo-3-7B-Think"

# Dataset configurations
DATASETS = {
    "mmlu_pro": {
        "path": "akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics",
        "category_field": "category",  # or check dataset_metadata
    },
    "supergpqa": {
        "path": "akenginorhun/supergpqa_10k_seed1_Olmo-3_family_metrics",
        "category_field": "discipline",
    },
    "wildchat": {
        "path": "akenginorhun/wildchat-4.8m_10k_seed1_Olmo-3_family_metrics_extended",
        "category_field": None,  # Will check
    },
    "natural_reasoning": {
        "path": "akenginorhun/natural_reasoning_10k_seed1_Olmo-3_family_metrics",
        "category_field": None,  # Will check
    },
}


def compute_ece(confidences, correctness, n_bins=10):
    """Expected Calibration Error with fixed-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)
    
    return ece / len(confidences)


def load_confidence_head(model_path: str, hidden_size: int, device, dtype):
    """Load pre-trained confidence head for Approach B."""
    head_path = os.path.join(model_path, "confidence_head.pt")
    
    if not os.path.exists(head_path):
        raise FileNotFoundError(
            f"No confidence_head.pt found at {head_path}. "
            "This script only works with Approach B models."
        )
    
    confidence_head = torch.nn.Linear(hidden_size, 1).to(device).to(dtype)
    state_dict = torch.load(head_path, map_location=device)
    confidence_head.load_state_dict(state_dict)
    confidence_head.eval()
    
    print(f"✓ Loaded confidence head from {head_path}")
    return confidence_head


def get_confidence_from_hidden(model, inputs, conf_token_position: int, confidence_head):
    """Extract confidence using linear probe on <|CONF|> hidden state."""
    with torch.no_grad():
        base_model = getattr(model, "model", None) or getattr(model, "base_model", None)
        outputs = base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        conf_hidden = last_hidden[0, conf_token_position, :]
        conf_logit = confidence_head(conf_hidden.to(confidence_head.weight.dtype))
        confidence = torch.sigmoid(conf_logit).item()
    
    return confidence


def extract_category(example, dataset_name: str):
    """Extract category/domain from example based on dataset structure."""
    # Try direct fields first
    for field in ["category", "subject", "discipline", "domain"]:
        if field in example and example[field]:
            return str(example[field])
    
    # Try dataset_metadata
    if "dataset_metadata" in example and isinstance(example["dataset_metadata"], dict):
        dm = example["dataset_metadata"]
        for field in ["category", "subject", "discipline", "domain"]:
            if field in dm and dm[field]:
                return str(dm[field])
    
    return "unknown"


def evaluate_on_dataset(
    model,
    tokenizer,
    confidence_head,
    dataset_name: str,
    conf_position: str = "suffix",
    num_eval: int = 1000,
    seed: int = 42,
):
    """
    Evaluate confidence model on a dataset for routing analysis.
    
    Returns detailed results including per-sample predictions.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    config = DATASETS[dataset_name]
    
    print(f"\n{'='*60}")
    print(f"Evaluating on: {dataset_name}")
    print(f"Dataset path: {config['path']}")
    print(f"{'='*60}")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = load_dataset(config["path"], split="train")
    print(f"Total samples in dataset: {len(full_dataset)}")
    
    # Filter to samples with target model traces
    def has_target_model(ex):
        return TARGET_MODEL in ex.get("model_metrics", {})
    
    dataset = full_dataset.filter(has_target_model, desc=f"Filtering by {TARGET_MODEL}")
    print(f"Samples with {TARGET_MODEL} traces: {len(dataset)}")
    
    if len(dataset) < 100:
        raise ValueError(f"Only {len(dataset)} valid samples - not enough for evaluation")
    
    # Create test split (use 20% as test, consistent with training)
    # This ensures we don't accidentally evaluate on training data if same dataset
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    test_dataset = split["test"]
    print(f"Test split: {len(test_dataset)} samples (20%, seed={seed})")
    
    # Sample for evaluation
    if num_eval < len(test_dataset):
        eval_dataset = test_dataset.shuffle(seed=seed).select(range(num_eval))
    else:
        eval_dataset = test_dataset.shuffle(seed=seed)
    print(f"Evaluating on: {len(eval_dataset)} samples")
    
    # Get device
    device = next(model.parameters()).device
    conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
    
    # Evaluate
    results = {
        "confidences": [],
        "labels": [],
        "categories": [],
        "questions": [],
    }
    
    skipped = 0
    for example in tqdm(eval_dataset, desc="Evaluating"):
        question = example["problem"]
        model_data = example["model_metrics"][TARGET_MODEL]
        answer = model_data.get("lm_response", "")
        is_correct = model_data.get("evaluation", {}).get("is_correct", False)
        category = extract_category(example, dataset_name)
        
        # Format prompt based on position
        if conf_position == "suffix":
            prompt = f"{question} <|CONF|>"
        else:
            prompt = f"{question} {answer} <|CONF|>"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        
        # Find <|CONF|> position
        input_ids = inputs["input_ids"][0].tolist()
        if conf_token_id not in input_ids:
            skipped += 1
            continue
        conf_pos = input_ids.index(conf_token_id)
        
        # Get confidence
        try:
            conf = get_confidence_from_hidden(model, inputs, conf_pos, confidence_head)
        except Exception as e:
            print(f"Error computing confidence: {e}")
            skipped += 1
            continue
        
        results["confidences"].append(conf)
        results["labels"].append(1 if is_correct else 0)
        results["categories"].append(category)
        results["questions"].append(question[:200])  # Truncate for storage
    
    print(f"\nEvaluated: {len(results['confidences'])} samples")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} samples (truncation or errors)")
    
    # Convert to numpy
    confidences = np.array(results["confidences"])
    labels = np.array(results["labels"])
    categories = results["categories"]
    
    # Compute metrics
    metrics = {
        "dataset": dataset_name,
        "num_samples": len(confidences),
        "num_skipped": skipped,
        "accuracy": float(labels.mean()),
        "confidence_mean": float(confidences.mean()),
        "confidence_std": float(confidences.std()),
    }
    
    # Calibration metrics
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, confidences))
        metrics["brier_score"] = float(brier_score_loss(labels, confidences))
        metrics["ece"] = float(compute_ece(confidences, labels))
    else:
        print("⚠ All labels same value - cannot compute AUROC")
        metrics["auroc"] = None
        metrics["brier_score"] = None
        metrics["ece"] = None
    
    # Routing metrics at each threshold
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics["routing"] = {}
    
    for thresh in thresholds:
        route_local = confidences >= thresh
        coverage = float(route_local.mean())
        
        if route_local.any():
            local_accuracy = float(labels[route_local].mean())
        else:
            local_accuracy = None
        
        # Overall accuracy: local handles high-conf, cloud handles low-conf (assume cloud always right)
        # overall = coverage * local_acc + (1 - coverage) * 1.0
        if local_accuracy is not None:
            overall_accuracy = coverage * local_accuracy + (1 - coverage) * 1.0
        else:
            overall_accuracy = 1.0  # All routed to cloud
        
        metrics["routing"][str(thresh)] = {
            "threshold": thresh,
            "coverage": coverage,
            "local_accuracy": local_accuracy,
            "overall_accuracy": overall_accuracy,
        }
    
    # Category breakdown
    unique_categories = list(set(categories))
    if len(unique_categories) > 1 and unique_categories != ["unknown"]:
        metrics["by_category"] = {}
        for cat in unique_categories:
            cat_mask = np.array([c == cat for c in categories])
            if cat_mask.sum() >= 10:  # Minimum samples for meaningful stats
                cat_conf = confidences[cat_mask]
                cat_labels = labels[cat_mask]
                metrics["by_category"][cat] = {
                    "count": int(cat_mask.sum()),
                    "accuracy": float(cat_labels.mean()),
                    "confidence_mean": float(cat_conf.mean()),
                }
                if len(np.unique(cat_labels)) > 1:
                    metrics["by_category"][cat]["auroc"] = float(roc_auc_score(cat_labels, cat_conf))
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"AUROC: {metrics['auroc']:.4f}" if metrics['auroc'] else "AUROC: N/A")
    print(f"Brier: {metrics['brier_score']:.4f}" if metrics['brier_score'] else "Brier: N/A")
    print(f"\nRouting at threshold 0.5:")
    r = metrics["routing"]["0.5"]
    print(f"  Coverage: {r['coverage']:.1%}")
    print(f"  Local accuracy: {r['local_accuracy']:.1%}" if r['local_accuracy'] else "  Local accuracy: N/A")
    print(f"  Overall accuracy: {r['overall_accuracy']:.1%}")
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset routing evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model directory (e.g., outputs/b_suffix)")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASETS.keys()) + ["all"],
                       help="Dataset to evaluate on, or 'all' for all datasets")
    parser.add_argument("--conf-position", type=str, default="suffix",
                       choices=["suffix", "posterior"],
                       help="Position of <|CONF|> token")
    parser.add_argument("--num-eval", type=int, default=1000,
                       help="Number of samples to evaluate")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: model_path/routing_eval/)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()
    
    # Set output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "routing_eval")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("CROSS-DATASET ROUTING EVALUATION")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset(s): {args.dataset}")
    print(f"Position: {args.conf_position}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Add <|CONF|> if not present
    if "<|CONF|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|CONF|>"]})
    
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    
    # Load confidence head
    confidence_head = load_confidence_head(args.model_path, hidden_size, device, torch_dtype)
    
    # Determine datasets to evaluate
    if args.dataset == "all":
        datasets_to_eval = list(DATASETS.keys())
    else:
        datasets_to_eval = [args.dataset]
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name in datasets_to_eval:
        try:
            metrics, raw_results = evaluate_on_dataset(
                model=model,
                tokenizer=tokenizer,
                confidence_head=confidence_head,
                dataset_name=dataset_name,
                conf_position=args.conf_position,
                num_eval=args.num_eval,
                seed=args.seed,
            )
            all_results[dataset_name] = metrics
            
            # Save per-dataset results
            output_path = os.path.join(args.output_dir, f"{dataset_name}_routing.json")
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"✓ Saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed on {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_datasets_routing.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined results saved to {combined_path}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("CROSS-DATASET COMPARISON")
    print("="*60)
    print(f"{'Dataset':<20} {'AUROC':>8} {'Acc':>8} {'Cov@0.5':>10} {'OA@0.5':>10}")
    print("-"*60)
    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"{name:<20} ERROR: {metrics['error'][:30]}")
        else:
            auroc = f"{metrics['auroc']:.3f}" if metrics.get('auroc') else "N/A"
            acc = f"{metrics['accuracy']:.1%}"
            cov = f"{metrics['routing']['0.5']['coverage']:.1%}"
            oa = f"{metrics['routing']['0.5']['overall_accuracy']:.1%}"
            print(f"{name:<20} {auroc:>8} {acc:>8} {cov:>10} {oa:>10}")


if __name__ == "__main__":
    main()

