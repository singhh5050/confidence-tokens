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


def train_probe_from_data(model, tokenizer, dataset, conf_position: str, device, dtype, num_samples=500, target_model_name=None):
    """Train a simple linear probe for Approach A (post-hoc)."""
    print(f"Training post-hoc probe on {num_samples} samples...")
    
    hidden_states = []
    labels = []
    
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
    
    if len(hidden_states) < 10:
        print("ERROR: Not enough valid samples to train probe")
        return None
    
    # Stack into tensors
    X = torch.stack(hidden_states)  # (N, hidden_size)
    y = torch.tensor(labels).unsqueeze(1)  # (N, 1)
    
    # Train simple logistic regression
    hidden_size = X.shape[1]
    probe = torch.nn.Linear(hidden_size, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"Training probe on {len(X)} samples...")
    for epoch in range(100):
        optimizer.zero_grad()
        logits = probe(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
    
    probe = probe.to(device).to(dtype)
    probe.eval()
    return probe


def evaluate_confidence(
    model_path: str,
    conf_position: str,
    method: str = "entropy",
    num_eval: int = 500,
    approach: str = "a",
):
    """Run confidence evaluation on test data."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Position: {conf_position}, Method: {method}, Approach: {approach}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
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
    
    # Load test dataset
    print("Loading dataset...")
    dataset = load_dataset(
        'akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics',
        split='train'
    )
    
    # Get model name - use the 7B model we trained on, not the 32B
    sample = dataset[0]['model_metrics']
    available_models = list(sample.keys())
    print(f"Available models in dataset: {available_models}")
    
    # Find the 7B Think model (what we trained on)
    model_name = None
    for m in available_models:
        if "7B-Think" in m and "SFT" not in m:
            model_name = m
            break
    
    if model_name is None:
        model_name = available_models[0]
        print(f"WARNING: Could not find 7B-Think model, using {model_name}")
    
    print(f"Using answers from: {model_name}")
    
    # For Approach A with hidden method, train probe first
    if method == "hidden" and approach == "a" and confidence_head is None:
        confidence_head = train_probe_from_data(
            model, tokenizer, dataset, conf_position, device, dtype, num_samples=500, target_model_name=model_name
        )
        if confidence_head is None:
            print("Failed to train probe, falling back to entropy method")
            method = "entropy"
    
    # Evaluate
    print(f"\nEvaluating on {num_eval} samples...")
    confidences = []
    labels = []
    
    eval_dataset = dataset.select(range(min(num_eval, len(dataset))))
    
    for example in tqdm(eval_dataset, desc="Evaluating"):
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
            
        except Exception as e:
            print(f"Error on example: {e}")
            continue
    
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
    }
    
    results_path = os.path.join(model_path, f"eval_results_{method}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
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
    )


if __name__ == "__main__":
    main()

