#!/usr/bin/env python3
"""
Quick end-to-end test for both Approach A and B.
Run this on RunPod to verify everything works before submitting batch job.

Usage:
    python scripts/test_pipeline.py
    
Expected: Both approaches complete without errors, ~2-5 min total.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokenizer_utils import add_conf_token
from data import prepare_suffix_confidence_dataset, get_tokenized_dataset
from training import ConfidenceTrainingConfig, train_confidence_model


def test_data_loading():
    """Test that dataset loading and extraction works."""
    print("\n" + "=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)
    
    # Use a small model just for tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|CONF|>"]})
    
    # Load 3 samples
    ds = prepare_suffix_confidence_dataset("mmlu_pro", tokenizer, "train", max_samples=3)
    
    print(f"✓ Loaded {len(ds)} samples")
    for i, ex in enumerate(ds):
        print(f"\n--- Sample {i} ---")
        print(f"  confidence_label: {ex['confidence_label']}")
        print(f"  conf_token_position: {ex['conf_token_position']}")
        print(f"  text (first 150 chars): {ex['text'][:150]}...")
    
    print("\n✓ Data loading test PASSED")
    return True


def test_approach_a(model, tokenizer, conf_token_id):
    """Test Approach A: SFT only."""
    print("\n" + "=" * 60)
    print("TEST 2: Approach A (SFT Only)")
    print("=" * 60)
    
    # Prepare tiny dataset
    train_ds = prepare_suffix_confidence_dataset("mmlu_pro", tokenizer, "train", max_samples=10)
    train_ds = get_tokenized_dataset(train_ds, tokenizer, max_length=512)
    
    config = ConfidenceTrainingConfig(
        output_dir="./test_output_a",
        supervised=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=5,  # Just 5 steps
        logging_steps=1,
        save_steps=100,  # Don't save
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    metrics = train_confidence_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        config=config,
        conf_token_id=conf_token_id,
    )
    
    print(f"\n✓ Approach A test PASSED (loss: {metrics.get('train_loss', 'N/A'):.4f})")
    return True


def test_approach_b(model, tokenizer, conf_token_id):
    """Test Approach B: Supervised confidence."""
    print("\n" + "=" * 60)
    print("TEST 3: Approach B (Supervised Confidence)")
    print("=" * 60)
    
    # Prepare tiny dataset
    train_ds = prepare_suffix_confidence_dataset("mmlu_pro", tokenizer, "train", max_samples=10)
    train_ds = get_tokenized_dataset(train_ds, tokenizer, max_length=512)
    
    config = ConfidenceTrainingConfig(
        output_dir="./test_output_b",
        supervised=True,
        confidence_loss_weight=0.3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=5,  # Just 5 steps
        logging_steps=1,
        save_steps=100,  # Don't save
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    metrics = train_confidence_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        config=config,
        conf_token_id=conf_token_id,
    )
    
    print(f"\n✓ Approach B test PASSED (loss: {metrics.get('train_loss', 'N/A'):.4f})")
    return True


def main():
    print("=" * 60)
    print("CONFIDENCE TOKEN PIPELINE TEST")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test 1: Data loading
    test_data_loading()
    
    # Load model for remaining tests
    print("\n" + "=" * 60)
    print("Loading model (Qwen/Qwen3-0.6B for quick testing)...")
    print("=" * 60)
    
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    
    # Add confidence token
    conf_token_id = add_conf_token(tokenizer, model)
    print(f"✓ Model loaded, <|CONF|> token ID: {conf_token_id}")
    
    # Test 2: Approach A
    test_approach_a(model, tokenizer, conf_token_id)
    
    # Reload model (training modifies it)
    print("\nReloading fresh model for Approach B test...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    conf_token_id = add_conf_token(tokenizer, model)
    
    # Test 3: Approach B
    test_approach_b(model, tokenizer, conf_token_id)
    
    # Cleanup
    import shutil
    for d in ["./test_output_a", "./test_output_b"]:
        if Path(d).exists():
            shutil.rmtree(d)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nYou're ready to submit the batch job!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

