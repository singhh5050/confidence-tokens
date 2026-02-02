#!/usr/bin/env python3
"""
Training script for confidence token models.

Two approaches available:
- Approach A (SFT): Standard language modeling, confidence learned implicitly
- Approach B (Supervised): Explicit supervision on <|CONF|> hidden state

Usage:
    # Approach A: SFT only (default)
    python scripts/train.py --max-samples 100 --epochs 1
    
    # Approach B: Supervised confidence training
    python scripts/train.py --supervised --alpha 0.3
    
    # Full training with Olmo-3-7B-Think (default)
    python scripts/train.py --supervised --dataset mmlu_pro
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokenizer_utils import add_conf_token
from data import (
    prepare_confidence_dataset, 
    get_tokenized_dataset, 
    CONF_POSITIONS,
    create_train_test_split,
    create_multi_dataset_split,
    FINETUNE_MODEL,
    DEFAULT_TRACE_MODEL,
)
from training import ConfidenceTrainingConfig, train_confidence_model


def main():
    parser = argparse.ArgumentParser(
        description="Train a confidence token model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (Approach A, SFT only)
    python scripts/train.py --max-samples 100 --epochs 1
    
    # Quick test (Approach B, Supervised)
    python scripts/train.py --supervised --max-samples 100 --epochs 1
    
    # Full training with Olmo-3-7B-Think (default model)
    python scripts/train.py --supervised --dataset mmlu_pro
    
    # Lightweight testing with Qwen
    python scripts/train.py --model Qwen/Qwen3-0.6B --supervised --epochs 3
    
    # Resume from checkpoint
    python scripts/train.py --resume ./output/checkpoint-500
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=FINETUNE_MODEL,
        help=f"Model to fine-tune (default: {FINETUNE_MODEL})"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="mmlu_pro",
        choices=[
            "mmlu_pro",
            "supergpqa",
            "wildchat",
            "natural_reasoning",
            "mmlu_pro_qwen",
            "supergpqa_qwen",
            "wildchat_qwen",
        ],
        help="Dataset to train on (default: mmlu_pro). Ignored if --datasets is provided."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets for multi-dataset training (e.g., 'mmlu_pro,supergpqa,wildchat,natural_reasoning')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (for quick testing)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test/eval set (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)"
    )
    parser.add_argument(
        "--trace-model",
        type=str,
        default=DEFAULT_TRACE_MODEL,
        help=f"Model whose traces to use from dataset (default: {DEFAULT_TRACE_MODEL})"
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output/conf-sft",
        help="Output directory (default: ./output/conf-sft)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Max sequence length (default: 4096)"
    )
    
    # Training approach
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Use Approach B: Supervised confidence training (default: Approach A, SFT only)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Confidence loss weight for Approach B: loss = (1-α)*LM + α*Conf (default: 0.3)"
    )
    parser.add_argument(
        "--conf-position",
        type=str,
        default="suffix",
        choices=CONF_POSITIONS,
        help="Where to place <|CONF|> token: 'suffix' = {Q} <|CONF|> {A}, 'posterior' = {Q} {A} <|CONF|> (default: suffix)"
    )
    
    # Precision
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 precision (default: True; disable with --no-bf16)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision instead of bf16"
    )
    
    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to Weights & Biases"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name"
    )
    
    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )
    
    args = parser.parse_args()

    # Seed everything for reproducibility
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(args.seed)
    
    # Determine dtype FIRST (before printing config)
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.fp16:
        dtype = torch.float16
    elif args.bf16 and bf16_supported:
        dtype = torch.bfloat16
    elif args.bf16 and not bf16_supported:
        dtype = torch.float16  # Will warn later
    else:
        dtype = torch.float32
    
    # Print configuration
    approach = "B (Supervised)" if args.supervised else "A (SFT only)"
    conf_pos = args.conf_position
    
    # Format descriptions for conf position
    pos_fmt = "{Q} <|CONF|> {A}" if conf_pos == "suffix" else "{Q} {A} <|CONF|>"
    
    # Parse multi-dataset argument if provided
    dataset_list = None
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(",")]
        valid_datasets = [
            "mmlu_pro",
            "supergpqa",
            "wildchat",
            "natural_reasoning",
            "mmlu_pro_qwen",
            "supergpqa_qwen",
            "wildchat_qwen",
        ]
        for d in dataset_list:
            if d not in valid_datasets:
                print(f"❌ Unknown dataset: {d}. Valid options: {valid_datasets}")
                sys.exit(1)
    
    print("=" * 70)
    print("CONFIDENCE TOKEN TRAINING")
    print("=" * 70)
    print(f"\nApproach: {approach}")
    print(f"\nConfiguration:")
    print(f"  Model to fine-tune: {args.model}")
    print(f"  Trace model (data): {args.trace_model}")
    if dataset_list:
        print(f"  Datasets (multi): {', '.join(dataset_list)}")
    else:
        print(f"  Dataset: {args.dataset}")
    print(f"  Train/Test split: {100*(1-args.test_size):.0f}%/{100*args.test_size:.0f}% (seed={args.seed})")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} (effective: {args.batch_size * args.grad_accum})")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  CONF position: {conf_pos} → {pos_fmt}")
    print(f"  bf16 supported: {bf16_supported}")
    print(f"  Precision: {'bf16' if dtype == torch.bfloat16 else ('fp16' if dtype == torch.float16 else 'fp32')}")
    print(f"  Logging: {'wandb' if args.wandb else 'none'}")
    if args.supervised:
        print(f"  Confidence loss weight (α): {args.alpha}")
        print(f"  Loss = {1 - args.alpha:.1f} * LM + {args.alpha:.1f} * Conf")
    
    # Load model and tokenizer
    print("\n" + "-" * 70)
    print("Loading model and tokenizer...")
    print("-" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warn if bf16 was requested but not supported (dtype already set above)
    if args.bf16 and not bf16_supported:
        print("⚠ bf16 requested but not supported on this hardware; falling back to fp16")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print(f"✓ Model loaded ({dtype})")
    except RuntimeError as e:
        if "buffer size" in str(e).lower() or "memory" in str(e).lower():
            print(f"\n❌ Model too large for available memory")
            print(f"   Try: python scripts/train.py --model Qwen/Qwen3-0.6B")
            sys.exit(1)
        raise
    
    # Fix generation config conflicts (Olmo has do_sample=False but temperature/top_p set)
    model.config.use_cache = False
    if hasattr(model, 'generation_config'):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    
    # Add confidence token
    print("\nAdding <|CONF|> token...")
    conf_token_id = add_conf_token(tokenizer, model)
    print(f"✓ <|CONF|> token ID: {conf_token_id}")

    # Ensure output embeddings match tokenizer size
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_size = output_embeddings.weight.shape[0]
        vocab_size = len(tokenizer)
        if output_size != vocab_size:
            print(
                f"⚠ Output embeddings size mismatch: {output_size} != {vocab_size}. "
                "Resizing and tying weights."
            )
            model.resize_token_embeddings(vocab_size)
            try:
                model.tie_weights()
            except Exception:
                print("⚠ Unable to tie weights; continuing with resized embeddings.")
    
    # Prepare datasets with proper train/test split
    print("\n" + "-" * 70)
    if dataset_list:
        print(f"Preparing multi-dataset ({', '.join(dataset_list)}) with train/test split...")
    else:
        print(f"Preparing {args.dataset} dataset with train/test split...")
    print("-" * 70)
    
    # Create the split and save metadata
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use multi-dataset or single-dataset loading based on args
    if dataset_list:
        raw_train, raw_test, split_metadata = create_multi_dataset_split(
            dataset_names=dataset_list,
            test_size=args.test_size,
            seed=args.seed,
            output_dir=args.output_dir,  # Saves split_metadata.json here
            trace_model=args.trace_model,
        )
    else:
        raw_train, raw_test, split_metadata = create_train_test_split(
            dataset_name=args.dataset,
            test_size=args.test_size,
            seed=args.seed,
            output_dir=args.output_dir,  # Saves split_metadata.json here
        )
    
    # Limit samples if requested (for quick testing)
    if args.max_samples:
        raw_train = raw_train.select(range(min(args.max_samples, len(raw_train))))
        raw_test = raw_test.select(range(min(args.max_samples // 5, len(raw_test))))
        print(f"⚠ Limited to {len(raw_train)} train / {len(raw_test)} test samples (--max-samples)")
    
    # Filter out samples missing the requested trace model (no fallback)
    # Multi-dataset splits may already be flattened; handle both cases.
    def has_trace_model(example):
        if "model_metrics" in example:
            return args.trace_model in example.get("model_metrics", {})
        if "is_correct" in example and "lm_response" in example and "problem" in example:
            return True
        return False

    pre_filter_train = len(raw_train)
    raw_train = raw_train.filter(has_trace_model, desc=f"Filtering train by {args.trace_model}")
    pre_filter_test = len(raw_test)
    raw_test = raw_test.filter(has_trace_model, desc=f"Filtering test by {args.trace_model}")

    dropped_train = pre_filter_train - len(raw_train)
    dropped_test = pre_filter_test - len(raw_test)
    if dropped_train > 0 or dropped_test > 0:
        print(
            f"⚠ Dropped samples missing model_metrics for {args.trace_model}: "
            f"train={dropped_train}, test={dropped_test}"
        )
    
    # Format datasets with CONF token
    from data import format_prompt, extract_from_nested
    
    def format_example(example):
        if "model_metrics" in example:
            extracted = extract_from_nested(example, args.trace_model)
            question = str(extracted["question"])
            answer = str(extracted["answer"])
            is_correct = float(extracted["is_correct"])
        else:
            question = str(example.get("problem", ""))
            answer = str(example.get("lm_response", ""))
            is_correct = float(example.get("is_correct", 0.0))

        text = format_prompt(question, answer, args.conf_position)
        return {
            "text": text,
            "confidence_label": is_correct,
            "conf_token_position": -1,
        }
    
    # Get columns to remove (handle multi-dataset case where _source_dataset is added)
    train_columns = raw_train.column_names
    test_columns = raw_test.column_names
    
    train_dataset = raw_train.map(
        format_example, 
        remove_columns=train_columns,
        desc=f"Formatting train set ({args.conf_position})"
    )
    eval_dataset = raw_test.map(
        format_example,
        remove_columns=test_columns,
        desc=f"Formatting test set ({args.conf_position})"
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} examples")
    print(f"✓ Eval dataset:  {len(eval_dataset)} examples")
    
    # FAIL LOUDLY if no eval dataset
    if eval_dataset is None or len(eval_dataset) == 0:
        raise ValueError(
            "❌ FATAL: No evaluation dataset! "
            "Training without held-out data is invalid. "
            "Check your --test-size setting."
        )
    
    # Tokenize
    print("\nTokenizing datasets...")
    pre_tokenize_train = len(train_dataset)
    train_dataset = get_tokenized_dataset(
        train_dataset,
        tokenizer,
        args.max_length,
        include_conf_fields=args.supervised,
        drop_invalid_conf=True,
        drop_truncated=True,
    )
    dropped_tokenize_train = pre_tokenize_train - len(train_dataset)
    if eval_dataset:
        pre_tokenize_eval = len(eval_dataset)
        eval_dataset = get_tokenized_dataset(
            eval_dataset,
            tokenizer,
            args.max_length,
            include_conf_fields=args.supervised,
            drop_invalid_conf=True,
            drop_truncated=True,
        )
        dropped_tokenize_eval = pre_tokenize_eval - len(eval_dataset)
    else:
        dropped_tokenize_eval = 0
    print("✓ Tokenization complete")

    if args.supervised:
        print("\n" + "=" * 60)
        print("CONF POSITION DIAGNOSTICS")
        print("=" * 60)

        conf_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
        unk_id = tokenizer.unk_token_id
        print(f"<|CONF|> token ID: {conf_id}")
        print(f"UNK token ID: {unk_id}")
        if conf_id == unk_id:
            raise ValueError("FATAL: <|CONF|> is UNK! Token not in vocabulary.")

        n_samples = min(10, len(train_dataset))
        positions = train_dataset["conf_token_position"][:n_samples]
        lengths = [len(ids) for ids in train_dataset["input_ids"][:n_samples]]

        print(f"\nFirst {n_samples} samples:")
        for i, (pos, length) in enumerate(zip(positions, lengths)):
            valid = "OK" if 0 <= pos < length else "INVALID"
            print(f"  [{i}] pos={pos}, seq_len={length} -> {valid}")

        all_pos = train_dataset["conf_token_position"]
        all_lens = [len(ids) for ids in train_dataset["input_ids"]]
        invalid = sum(1 for p, l in zip(all_pos, all_lens) if p < 0 or p >= l)
        invalid_pct = (100 * invalid / len(all_pos)) if all_pos else 0.0
        print(f"\nOverall: {invalid}/{len(all_pos)} invalid positions ({invalid_pct:.1f}%)")
        print("=" * 60 + "\n")
    
    # Configure training
    # Set eval_strategy based on whether we have an eval dataset
    eval_strat = "steps" if eval_dataset is not None else "no"
    
    # Determine actual precision for TrainingArguments
    # Must match what we actually loaded the model with
    use_bf16 = args.bf16 and not args.fp16 and bf16_supported
    use_fp16 = args.fp16 or (args.bf16 and not bf16_supported)
    
    config = ConfidenceTrainingConfig(
        output_dir=args.output_dir,
        supervised=args.supervised,
        confidence_loss_weight=args.alpha,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy=eval_strat,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
    )

    # Save run config for accountability/reproducibility
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent.parent), text=True
        ).strip()
    except Exception:
        git_commit = None

    run_config = {
        "args": vars(args),
        "precision": "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32"),
        "git_commit": git_commit,
        "split_metadata": split_metadata,
        "is_multi_dataset": dataset_list is not None,
        "dataset_list": dataset_list,
        "dropped_counts": {
            "missing_trace_model_train": dropped_train,
            "missing_trace_model_test": dropped_test,
            "tokenized_drop_train": dropped_tokenize_train,
            "tokenized_drop_eval": dropped_tokenize_eval,
        },
    }
    config_path = Path(args.output_dir) / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"✓ Saved run config to: {config_path}")
    
    # Train
    print("\n" + "-" * 70)
    print(f"Starting training (Approach {approach})...")
    print("-" * 70)
    
    metrics = train_confidence_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        conf_token_id=conf_token_id,
        resume_from_checkpoint=args.resume,
    )
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

