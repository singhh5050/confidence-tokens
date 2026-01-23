#!/usr/bin/env python3
"""
Verify dataset structure for routing experiments.

Checks that all datasets have the required fields and sufficient samples.
Fails hard if any dataset doesn't match expected structure.

Usage:
    python scripts/verify_datasets.py
"""

import sys
from datasets import load_dataset

# Target model whose traces we need
TARGET_MODEL = "allenai/Olmo-3-7B-Think"

# Minimum samples for meaningful evaluation
MIN_SAMPLES = 500

# Datasets to verify
DATASETS = {
    "mmlu_pro": "akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics",
    "supergpqa": "akenginorhun/supergpqa_10k_seed1_Olmo-3_family_metrics",
    "wildchat": "akenginorhun/wildchat-4.8m_10k_seed1_Olmo-3_family_metrics_extended",
    "natural_reasoning": "akenginorhun/natural_reasoning_10k_seed1_Olmo-3_family_metrics",
}


def verify_dataset(name: str, path: str) -> dict:
    """
    Verify a single dataset has the required structure.
    
    Returns dict with verification results.
    Raises ValueError if critical fields are missing.
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")
    
    # Load dataset
    try:
        dataset = load_dataset(path, split="train")
    except Exception as e:
        raise ValueError(f"Failed to load {name}: {e}")
    
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    if total_samples < MIN_SAMPLES:
        raise ValueError(f"{name} has only {total_samples} samples, need at least {MIN_SAMPLES}")
    
    # Check first example structure
    example = dataset[0]
    
    # 1. Check 'problem' field exists
    if "problem" not in example:
        raise ValueError(f"{name} missing 'problem' field")
    print(f"✓ 'problem' field exists")
    
    # 2. Check 'model_metrics' field exists
    if "model_metrics" not in example:
        raise ValueError(f"{name} missing 'model_metrics' field")
    print(f"✓ 'model_metrics' field exists")
    
    model_metrics = example["model_metrics"]
    
    # 3. Check target model exists in model_metrics
    if TARGET_MODEL not in model_metrics:
        available = list(model_metrics.keys())
        raise ValueError(
            f"{name} missing traces for {TARGET_MODEL}. "
            f"Available models: {available}"
        )
    print(f"✓ '{TARGET_MODEL}' traces exist")
    
    target_data = model_metrics[TARGET_MODEL]
    
    # 4. Check 'lm_response' exists
    if "lm_response" not in target_data:
        raise ValueError(f"{name} missing 'lm_response' in model_metrics[{TARGET_MODEL}]")
    print(f"✓ 'lm_response' field exists")
    
    # 5. Check 'evaluation' and 'is_correct' exist
    if "evaluation" not in target_data:
        raise ValueError(f"{name} missing 'evaluation' in model_metrics[{TARGET_MODEL}]")
    
    evaluation = target_data["evaluation"]
    if "is_correct" not in evaluation:
        raise ValueError(f"{name} missing 'is_correct' in evaluation")
    print(f"✓ 'evaluation.is_correct' field exists")
    
    # 6. Check for category/domain metadata
    metadata_fields = []
    
    # Check common metadata field names
    possible_metadata = ["category", "subject", "discipline", "domain", "topic", "dataset_metadata"]
    for field in possible_metadata:
        if field in example:
            metadata_fields.append(field)
    
    # Also check inside dataset_metadata if it exists
    if "dataset_metadata" in example and isinstance(example["dataset_metadata"], dict):
        dm = example["dataset_metadata"]
        for field in ["category", "subject", "discipline", "domain", "topic"]:
            if field in dm:
                metadata_fields.append(f"dataset_metadata.{field}")
    
    if metadata_fields:
        print(f"✓ Category/domain metadata found: {metadata_fields}")
    else:
        print(f"⚠ No category/domain metadata found (will skip domain breakdown)")
    
    # 7. Sample 100 examples to verify structure consistency and compute accuracy
    print(f"\nSampling 100 examples to verify consistency...")
    correct_count = 0
    missing_target = 0
    
    sample_indices = list(range(min(100, total_samples)))
    for idx in sample_indices:
        ex = dataset[idx]
        
        if "model_metrics" not in ex:
            raise ValueError(f"{name} example {idx} missing 'model_metrics'")
        
        if TARGET_MODEL not in ex["model_metrics"]:
            missing_target += 1
            continue
        
        is_correct = ex["model_metrics"][TARGET_MODEL].get("evaluation", {}).get("is_correct", None)
        if is_correct is None:
            raise ValueError(f"{name} example {idx} missing 'is_correct'")
        
        if is_correct:
            correct_count += 1
    
    valid_samples = 100 - missing_target
    if valid_samples == 0:
        raise ValueError(f"{name} has no valid samples with {TARGET_MODEL} traces in first 100")
    
    accuracy = correct_count / valid_samples
    print(f"✓ Structure consistent across sample")
    print(f"  - Valid samples: {valid_samples}/100")
    print(f"  - Accuracy in sample: {accuracy:.1%}")
    
    if missing_target > 0:
        print(f"  ⚠ {missing_target}/100 samples missing {TARGET_MODEL} traces")
    
    # 8. Estimate total valid samples
    # Filter to count exact number with target model
    print(f"\nCounting samples with {TARGET_MODEL} traces...")
    
    def has_target_model(ex):
        return TARGET_MODEL in ex.get("model_metrics", {})
    
    # This is slow but necessary for accurate counts
    valid_dataset = dataset.filter(has_target_model, desc=f"Filtering {name}")
    valid_count = len(valid_dataset)
    
    print(f"✓ Valid samples with {TARGET_MODEL}: {valid_count}/{total_samples}")
    
    if valid_count < MIN_SAMPLES:
        raise ValueError(
            f"{name} has only {valid_count} valid samples with {TARGET_MODEL}, "
            f"need at least {MIN_SAMPLES}"
        )
    
    # Compute overall accuracy (handle None evaluation gracefully)
    correct_total = 0
    null_eval_count = 0
    for ex in valid_dataset:
        evaluation = ex["model_metrics"][TARGET_MODEL].get("evaluation")
        if evaluation is None:
            null_eval_count += 1
            continue
        if evaluation.get("is_correct", False):
            correct_total += 1
    
    if null_eval_count > 0:
        print(f"⚠ {null_eval_count}/{valid_count} samples have null 'evaluation' field")
        valid_for_accuracy = valid_count - null_eval_count
    else:
        valid_for_accuracy = valid_count
    
    if valid_for_accuracy > 0:
        overall_accuracy = correct_total / valid_for_accuracy
        print(f"✓ Overall accuracy ({TARGET_MODEL}): {overall_accuracy:.1%} ({correct_total}/{valid_for_accuracy})")
    else:
        overall_accuracy = 0.0
        print(f"⚠ No valid samples to compute accuracy")
    
    return {
        "name": name,
        "path": path,
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "null_eval_samples": null_eval_count,
        "accuracy": overall_accuracy,
        "metadata_fields": metadata_fields,
    }


def main():
    print("="*60)
    print("DATASET VERIFICATION FOR ROUTING EXPERIMENTS")
    print(f"Target model: {TARGET_MODEL}")
    print(f"Minimum samples: {MIN_SAMPLES}")
    print("="*60)
    
    results = {}
    failed = []
    
    for name, path in DATASETS.items():
        try:
            result = verify_dataset(name, path)
            results[name] = result
        except ValueError as e:
            print(f"\n❌ FAILED: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            failed.append((name, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Valid samples: {result['valid_samples']}")
        if result.get('null_eval_samples', 0) > 0:
            print(f"  ⚠ Null evaluations: {result['null_eval_samples']}")
        print(f"  Accuracy: {result['accuracy']:.1%}")
        print(f"  Metadata: {result['metadata_fields'] or 'None'}")
    
    if failed:
        print(f"\n❌ FAILED DATASETS ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(results)} datasets verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

