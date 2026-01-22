#!/usr/bin/env python3
"""
Evaluate confidence extraction from fine-tuned models.

Usage:
    python scripts/evaluate_confidence.py --experiment a_suffix
    python scripts/evaluate_confidence.py --experiment b_posterior --method hidden
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np


def compute_ece(confidences, correctness, n_bins=10):
    """Expected Calibration Error (ECE) with fixed-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)
    
    return ece / len(confidences)


def compute_ace(confidences, correctness, n_bins=10):
    """Adaptive Calibration Error (ACE) with equal-mass bins."""
    n = len(confidences)
    indices = np.argsort(confidences)
    sorted_conf = confidences[indices]
    sorted_corr = correctness[indices]
    
    ace = 0.0
    bin_size = n // n_bins
    if bin_size == 0:
        return float("nan")
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        if end <= start:
            continue
        
        bin_conf = sorted_conf[start:end].mean()
        bin_acc = sorted_corr[start:end].mean()
        ace += (end - start) * abs(bin_conf - bin_acc)
    
    return ace / n


def format_prompt_for_inference(question: str, conf_position: str) -> str:
    """Format prompt for confidence extraction (no answer, just up to <|CONF|>)."""
    if conf_position == "suffix":
        # {question} <|CONF|>
        return f"{question} <|CONF|>"
    elif conf_position == "posterior":
        # For posterior, we need the answer to get confidence
        # At inference without answer, we can't do posterior properly
        # So we'll generate the answer first, then extract confidence
        raise ValueError("Posterior position requires answer for confidence extraction")
    else:
        raise ValueError(f"Unknown conf_position: {conf_position}")


def format_prompt_with_answer(question: str, answer: str, conf_position: str) -> str:
    """Format prompt with answer for posterior position."""
    if conf_position == "suffix":
        return f"{question} <|CONF|> {answer}"
    elif conf_position == "posterior":
        return f"{question} {answer} <|CONF|>"
    else:
        raise ValueError(f"Unknown conf_position: {conf_position}")


def get_confidence_from_hidden(model, inputs, conf_token_position: int, confidence_head):
    """Extract confidence using linear probe on <|CONF|> hidden state."""
    with torch.no_grad():
        # Get hidden states from base model
        base_model = getattr(model, "model", None) or getattr(model, "base_model", None)
        outputs = base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
        
        # Get hidden state at <|CONF|> position
        conf_hidden = last_hidden[0, conf_token_position, :]  # (hidden_size,)
        
        # Apply confidence head
        conf_logit = confidence_head(conf_hidden.to(confidence_head.weight.dtype))
        confidence = torch.sigmoid(conf_logit).item()
    
    return confidence


def get_confidence_from_entropy(model, inputs):
    """Extract confidence using inverse entropy of next-token distribution."""
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Logits at last position (predicting next token after <|CONF|>)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Normalize to [0, 1] range
        vocab_size = logits.shape[0]
        max_entropy = np.log(vocab_size)
        normalized_entropy = entropy.item() / max_entropy
        
        # Confidence = 1 - normalized_entropy
        confidence = 1 - normalized_entropy
    
    return confidence


def get_confidence_from_topk(model, inputs, k=10):
    """Confidence based on probability mass in top-k tokens."""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k probability mass
        topk_probs, _ = torch.topk(probs, k)
        confidence = topk_probs.sum().item()
    
    return confidence


def load_confidence_head(model_path: str, hidden_size: int, device, dtype):
    """Load trained confidence head for Approach B."""
    head_path = os.path.join(model_path, "confidence_head.pt")
    
    if not os.path.exists(head_path):
        print(f"WARNING: No confidence_head.pt found at {head_path}")
        return None
    
    # Create head with same architecture
    confidence_head = torch.nn.Linear(hidden_size, 1).to(device).to(dtype)
    
    # Load weights
    state_dict = torch.load(head_path, map_location=device)
    confidence_head.load_state_dict(state_dict)
    confidence_head.eval()
    
    print(f"Loaded confidence head from {head_path}")
    return confidence_head


def train_probe_from_data(model, tokenizer, dataset, conf_position: str, device, dtype, num_samples=500, target_model_name=None, debug_log=None):
    """Train a simple linear probe for Approach A (post-hoc)."""
    print(f"Training post-hoc probe on {num_samples} samples...")
    
    hidden_states = []
    labels = []
    probe_training_log = []  # DEBUG: log every training sample
    
    # Get model name for the dataset
    sample = dataset[0]['model_metrics']
    available_models = list(sample.keys())
    
    if target_model_name and target_model_name in available_models:
        model_name = target_model_name
    else:
        # Find the 7B Think model
        model_name = None
        for m in available_models:
            if "7B-Think" in m and "SFT" not in m:
                model_name = m
                break
        if model_name is None:
            model_name = available_models[0]
    
    print(f"Using model metrics from: {model_name}")
    
    base_model = getattr(model, "model", None) or getattr(model, "base_model", None)
    
    skipped_count = 0
    for i, example in enumerate(tqdm(dataset.select(range(min(num_samples, len(dataset)))), desc="Collecting hidden states")):
        question = example['problem']
        answer = example['model_metrics'][model_name].get('lm_response', '')
        is_correct = example['model_metrics'][model_name].get('evaluation', {}).get('is_correct', False)
        
        # Format prompt with answer to get hidden state at <|CONF|> position
        prompt = format_prompt_with_answer(question, answer, conf_position)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        
        # Find <|CONF|> position
        conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
        input_ids = inputs["input_ids"][0].tolist()
        if conf_token_id not in input_ids:
            skipped_count += 1
            continue
        conf_pos = input_ids.index(conf_token_id)
        
        with torch.no_grad():
            outputs = base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            hidden = outputs.last_hidden_state[0, conf_pos, :].cpu().float()
        
        hidden_states.append(hidden)
        labels.append(1.0 if is_correct else 0.0)
        
        # DEBUG: Log this training sample
        probe_training_log.append({
            "dataset_index": i,
            "question_snippet": question[:100],
            "answer_snippet": answer[:100] if answer else "",
            "label": 1.0 if is_correct else 0.0,
            "conf_token_pos": conf_pos,
            "seq_len": len(input_ids),
            "hidden_norm": float(hidden.norm().item()),
        })
    
    print(f"  Skipped {skipped_count} samples (no <|CONF|> token after truncation)")
    
    if len(hidden_states) < 10:
        print("ERROR: Not enough valid samples to train probe")
        return None
    
    # Stack into tensors
    X = torch.stack(hidden_states)  # (N, hidden_size)
    y = torch.tensor(labels).unsqueeze(1)  # (N, 1)
    
    # DEBUG: Log training data statistics
    if debug_log is not None:
        debug_log["probe_training"] = {
            "num_samples_requested": num_samples,
            "num_samples_valid": len(hidden_states),
            "num_samples_skipped": skipped_count,
            "label_distribution": {
                "num_correct": int(sum(labels)),
                "num_incorrect": int(len(labels) - sum(labels)),
                "accuracy": float(sum(labels) / len(labels)),
            },
            "samples": probe_training_log,
        }
    
    # Train simple logistic regression
    hidden_size = X.shape[1]
    probe = torch.nn.Linear(hidden_size, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"Training probe on {len(X)} samples...")
    training_losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        logits = probe(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        training_losses.append(float(loss.item()))
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
    
    # DEBUG: Log training curve and final predictions on training data
    if debug_log is not None:
        with torch.no_grad():
            final_logits = probe(X)
            final_probs = torch.sigmoid(final_logits).squeeze().tolist()
        debug_log["probe_training"]["training_losses"] = training_losses
        debug_log["probe_training"]["final_train_predictions"] = final_probs
        debug_log["probe_training"]["final_train_labels"] = labels
        
        # Check if probe achieves perfect separation on training data
        train_preds = np.array(final_probs)
        train_labels = np.array(labels)
        if len(np.unique(train_labels)) > 1:
            train_auroc = roc_auc_score(train_labels, train_preds)
            debug_log["probe_training"]["train_auroc"] = float(train_auroc)
            print(f"  Probe AUROC on training data: {train_auroc:.4f}")
    
    probe = probe.to(device).to(dtype)
    probe.eval()
    return probe


def evaluate_confidence(
    model_path: str,
    conf_position: str,
    method: str = "entropy",
    num_eval: int = 500,
    approach: str = "a",
    seed: int = 42,
):
    """Run confidence evaluation on test data."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Position: {conf_position}, Method: {method}, Approach: {approach}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden_size = model.config.hidden_size
    
    # Add <|CONF|> token if not present
    if "<|CONF|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|CONF|>"]})
        # Model should already have this token from training
    
    conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
    print(f"<|CONF|> token ID: {conf_token_id}")
    
    # Load confidence head (for Approach B or trained probe for A)
    confidence_head = None
    if method == "hidden":
        if approach == "b":
            confidence_head = load_confidence_head(model_path, hidden_size, device, dtype)
        # For approach A, we'll train a probe below
    
    # Load test dataset using split metadata (ensures same test set as training)
    print("Loading dataset...")
    
    split_metadata_path = os.path.join(model_path, "split_metadata.json")
    run_config_path = os.path.join(model_path, "run_config.json")
    train_dataset = None
    trace_model_name = None
    if not os.path.exists(split_metadata_path):
        raise FileNotFoundError(
            f"Missing split_metadata.json at {split_metadata_path}. "
            "Evaluation requires the original train/test split to avoid contamination."
        )
    # Use the saved split to get exact same test set
    print(f"✓ Found split metadata at {split_metadata_path}")
    with open(split_metadata_path, "r") as f:
        split_metadata = json.load(f)
    
    print(f"  Dataset: {split_metadata['dataset_name']}")
    print(f"  Test size: {split_metadata['test_size']} (seed={split_metadata['seed']})")
    print(f"  Expected test samples: {split_metadata['test_samples']}")
    
    # Recreate the exact split
    full_dataset = load_dataset(split_metadata['dataset_path'], split='train')
    split = full_dataset.train_test_split(
        test_size=split_metadata['test_size'],
        seed=split_metadata['seed']
    )
    train_dataset = split['train']
    dataset = split['test']
    print(f"✓ Loaded test set: {len(dataset)} samples (held out during training)")

    # Get model name - use the 7B model we trained on, not the 32B
    sample = dataset[0]['model_metrics']
    available_models = list(sample.keys())
    print(f"Available models in dataset: {available_models}")
    
    # Try to get trace_model from run_config.json, fall back to 7B-Think if not available
    trace_model_name = None
    if os.path.exists(run_config_path):
        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        trace_model_name = run_config.get("args", {}).get("trace_model")
        print(f"✓ Found run_config.json, trace_model: {trace_model_name}")
    else:
        print(f"⚠ No run_config.json found at {run_config_path}, using fallback")
    
    # Use trace_model from run_config if available; otherwise find 7B-Think model
    model_name = trace_model_name
    if model_name is None or model_name not in available_models:
        # Fallback: find the 7B-Think model
        for m in available_models:
            if "7B-Think" in m and "SFT" not in m:
                model_name = m
                print(f"  Using fallback model: {model_name}")
                break
        if model_name is None or model_name not in available_models:
            raise ValueError(
                f"Could not find suitable model in dataset. "
                f"Available: {available_models}"
            )
    print(f"Using answers from: {model_name}")
    
    # DEBUG: Create comprehensive debug log
    debug_log = {
        "config": {
            "model_path": model_path,
            "conf_position": conf_position,
            "method": method,
            "approach": approach,
            "num_eval": num_eval,
            "seed": seed,
        },
        "dataset_info": {
            "split_metadata": split_metadata,
            "trace_model": model_name,
            "train_size": len(train_dataset),
            "test_size": len(dataset),
        },
    }
    
    # For Approach A with hidden method, train probe first
    if method == "hidden" and approach == "a" and confidence_head is None:
        if train_dataset is None:
            raise ValueError(
                "Cannot train probe: train_dataset is None. "
                "Training probe on test set would cause data leakage."
            )
        confidence_head = train_probe_from_data(
            model, tokenizer, train_dataset, conf_position, device, dtype, num_samples=500, target_model_name=model_name,
            debug_log=debug_log
        )
        if confidence_head is None:
            print("Failed to train probe, falling back to entropy method")
            method = "entropy"
    
    # Evaluate
    print(f"\nEvaluating on {num_eval} samples...")
    confidences = []
    labels = []
    eval_samples_log = []  # DEBUG: log every eval sample
    
    eval_dataset = dataset.shuffle(seed=seed).select(range(min(num_eval, len(dataset))))
    
    skipped_eval = 0
    for idx, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        question = example['problem']
        answer = example['model_metrics'][model_name].get('lm_response', '')
        is_correct = example['model_metrics'][model_name].get('evaluation', {}).get('is_correct', False)
        
        try:
            # Format prompt based on position
            if conf_position == "suffix":
                # For suffix: we can extract confidence before seeing answer
                prompt = f"{question} <|CONF|>"
            else:
                # For posterior: need answer first
                prompt = f"{question} {answer} <|CONF|>"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
            
            # Find <|CONF|> position
            input_ids = inputs["input_ids"][0].tolist()
            if conf_token_id not in input_ids:
                skipped_eval += 1
                continue
            conf_pos = input_ids.index(conf_token_id)
            
            # Get confidence based on method
            if method == "hidden" and confidence_head is not None:
                conf = get_confidence_from_hidden(model, inputs, conf_pos, confidence_head)
            elif method == "entropy":
                conf = get_confidence_from_entropy(model, inputs)
            elif method == "topk":
                conf = get_confidence_from_topk(model, inputs)
            else:
                conf = get_confidence_from_entropy(model, inputs)
            
            confidences.append(conf)
            labels.append(1 if is_correct else 0)
            
            # DEBUG: Log this eval sample
            eval_samples_log.append({
                "eval_index": idx,
                "question_snippet": question[:100],
                "answer_snippet": answer[:100] if answer else "",
                "label": 1 if is_correct else 0,
                "confidence": float(conf),
                "conf_token_pos": conf_pos,
                "seq_len": len(input_ids),
            })
            
        except Exception as e:
            print(f"Error on example: {e}")
            skipped_eval += 1
            continue
    
    print(f"  Skipped {skipped_eval} eval samples")
    
    # DEBUG: Add evaluation data to debug log
    debug_log["evaluation"] = {
        "num_requested": num_eval,
        "num_evaluated": len(confidences),
        "num_skipped": skipped_eval,
        "samples": eval_samples_log,
    }
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(confidences)}")
    
    confidences = np.array(confidences)
    labels = np.array(labels)
    
    # Basic stats
    print(f"\nConfidence stats:")
    print(f"  Mean: {confidences.mean():.4f}")
    print(f"  Std:  {confidences.std():.4f}")
    print(f"  Min:  {confidences.min():.4f}")
    print(f"  Max:  {confidences.max():.4f}")
    
    print(f"\nAccuracy in dataset: {labels.mean():.4f}")
    
    # Calibration metrics
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, confidences)
        brier = brier_score_loss(labels, confidences)
        print(f"\nCalibration metrics:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
    else:
        print("\nWARNING: All labels are same value, can't compute AUROC")
        auroc = None
        brier = None
    
    ece = compute_ece(confidences, labels) if len(confidences) > 0 else None
    ace = compute_ace(confidences, labels) if len(confidences) > 0 else None
    if ece is not None:
        print(f"  ECE:   {ece:.4f}")
    if ace is not None:
        print(f"  ACE:   {ace:.4f}")
    
    # Save results
    results = {
        "model_path": model_path,
        "conf_position": conf_position,
        "method": method,
        "approach": approach,
        "num_samples": len(confidences),
        "confidence_mean": float(confidences.mean()),
        "confidence_std": float(confidences.std()),
        "accuracy": float(labels.mean()),
        "auroc": float(auroc) if auroc else None,
        "brier_score": float(brier) if brier else None,
        "ece": float(ece) if ece is not None else None,
        "ace": float(ace) if ace is not None else None,
    }
    
    # Coverage analysis at thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        route_self = confidences >= threshold
        results[f"coverage@{threshold}"] = float(route_self.mean())
        results[f"accuracy_self@{threshold}"] = (
            float(labels[route_self].mean()) if route_self.any() else None
        )
    
    results_path = os.path.join(model_path, f"eval_results_{method}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # DEBUG: Save comprehensive debug log
    debug_log["results"] = results
    debug_log["raw_data"] = {
        "confidences": confidences.tolist(),
        "labels": labels.tolist(),
    }
    
    # DEBUG: Analyze confidence/label correlation
    if len(confidences) > 0:
        correct_confs = confidences[labels == 1]
        incorrect_confs = confidences[labels == 0]
        
        debug_log["analysis"] = {
            "num_correct": int((labels == 1).sum()),
            "num_incorrect": int((labels == 0).sum()),
            "correct_conf_mean": float(correct_confs.mean()) if len(correct_confs) > 0 else None,
            "correct_conf_std": float(correct_confs.std()) if len(correct_confs) > 0 else None,
            "correct_conf_min": float(correct_confs.min()) if len(correct_confs) > 0 else None,
            "correct_conf_max": float(correct_confs.max()) if len(correct_confs) > 0 else None,
            "incorrect_conf_mean": float(incorrect_confs.mean()) if len(incorrect_confs) > 0 else None,
            "incorrect_conf_std": float(incorrect_confs.std()) if len(incorrect_confs) > 0 else None,
            "incorrect_conf_min": float(incorrect_confs.min()) if len(incorrect_confs) > 0 else None,
            "incorrect_conf_max": float(incorrect_confs.max()) if len(incorrect_confs) > 0 else None,
        }
        
        # Check for perfect separation
        if len(correct_confs) > 0 and len(incorrect_confs) > 0:
            min_correct = correct_confs.min()
            max_incorrect = incorrect_confs.max()
            debug_log["analysis"]["separation_gap"] = float(min_correct - max_incorrect)
            debug_log["analysis"]["perfect_separation"] = bool(min_correct > max_incorrect)
            
            print(f"\n{'='*60}")
            print("DEBUG ANALYSIS")
            print(f"{'='*60}")
            print(f"Correct answers ({len(correct_confs)}): conf range [{correct_confs.min():.4f}, {correct_confs.max():.4f}], mean={correct_confs.mean():.4f}")
            print(f"Incorrect answers ({len(incorrect_confs)}): conf range [{incorrect_confs.min():.4f}, {incorrect_confs.max():.4f}], mean={incorrect_confs.mean():.4f}")
            print(f"Separation gap (min_correct - max_incorrect): {min_correct - max_incorrect:.4f}")
            if min_correct > max_incorrect:
                print("⚠️  PERFECT SEPARATION DETECTED - this is suspicious!")
    
    debug_log_path = os.path.join(model_path, f"eval_debug_{method}.json")
    with open(debug_log_path, "w") as f:
        json.dump(debug_log, f, indent=2)
    print(f"Debug log saved to: {debug_log_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate confidence extraction")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["a_suffix", "a_posterior", "b_suffix", "b_posterior"],
                       help="Which experiment to evaluate")
    parser.add_argument("--method", type=str, default="entropy",
                       choices=["hidden", "entropy", "topk"],
                       help="Confidence extraction method")
    parser.add_argument("--num-eval", type=int, default=500,
                       help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for evaluation sampling")
    parser.add_argument("--output-base", type=str, 
                       default="/workspace/confidence-tokens/outputs",
                       help="Base path for outputs")
    args = parser.parse_args()
    
    # Parse experiment name
    approach = args.experiment[0]  # 'a' or 'b'
    conf_position = args.experiment[2:]  # 'suffix' or 'posterior'
    model_path = os.path.join(args.output_base, args.experiment)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    evaluate_confidence(
        model_path=model_path,
        conf_position=conf_position,
        method=args.method,
        num_eval=args.num_eval,
        approach=approach,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

