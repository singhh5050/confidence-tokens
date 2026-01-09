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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokenizer_utils import add_conf_token
from data import prepare_suffix_confidence_dataset, get_tokenized_dataset
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
        default="allenai/Olmo-3-7B-Think-SFT",
        help="Model name (default: allenai/Olmo-3-7B-Think-SFT)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="mmlu_pro",
        choices=["mmlu_pro", "supergpqa", "wildchat", "natural_reasoning"],
        help="Dataset to train on (default: mmlu_pro)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (for quick testing)"
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
    
    # Print configuration
    approach = "B (Supervised)" if args.supervised else "A (SFT only)"
    
    print("=" * 70)
    print("CONFIDENCE TOKEN TRAINING")
    print("=" * 70)
    print(f"\nApproach: {approach}")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} (effective: {args.batch_size * args.grad_accum})")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max samples: {args.max_samples or 'all'}")
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
    
    # Determine dtype
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.fp16:
        dtype = torch.float16
    elif args.bf16 and bf16_supported:
        dtype = torch.bfloat16
    elif args.bf16 and not bf16_supported:
        print("⚠ bf16 requested but not supported on this hardware; falling back to fp16")
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
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
    
    # Prepare datasets
    print("\n" + "-" * 70)
    print(f"Preparing {args.dataset} dataset...")
    print("-" * 70)
    
    train_dataset = prepare_suffix_confidence_dataset(
        args.dataset, 
        tokenizer, 
        "train",
        args.max_samples
    )
    print(f"✓ Train dataset: {len(train_dataset)} examples")
    
    # Try to load eval dataset
    eval_dataset = None
    try:
        eval_max = args.max_samples // 10 if args.max_samples else 1000
        eval_dataset = prepare_suffix_confidence_dataset(
            args.dataset,
            tokenizer,
            "test",
            eval_max
        )
        print(f"✓ Eval dataset: {len(eval_dataset)} examples")
    except Exception as e:
        print(f"⚠ Could not load eval dataset: {e}")
    
    # Tokenize
    print("\nTokenizing datasets...")
    train_dataset = get_tokenized_dataset(
        train_dataset,
        tokenizer,
        args.max_length,
        include_conf_fields=args.supervised,
    )
    if eval_dataset:
        eval_dataset = get_tokenized_dataset(
            eval_dataset,
            tokenizer,
            args.max_length,
            include_conf_fields=args.supervised,
        )
    print("✓ Tokenization complete")
    
    # Configure training
    config = ConfidenceTrainingConfig(
        output_dir=args.output_dir,
        supervised=args.supervised,
        confidence_loss_weight=args.alpha,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
    )
    
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

