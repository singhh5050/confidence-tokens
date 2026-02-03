#!/usr/bin/env python3
"""
Smoke test for <|CONF|> token addition.

This script verifies that:
1. The <|CONF|> token has a valid token ID
2. The token exists in the vocabulary
3. The embedding matrix has been properly resized
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizer_utils import add_conf_token, verify_conf_token


def run_smoke_test(model_name: str = "allenai/Olmo-3-7B-Think-SFT"):
    """
    Run smoke test for confidence token addition.
    
    Args:
        model_name: Name of the model to use for testing (default: allenai/Olmo-3-7B-Think-SFT)
    """
    print("=" * 70)
    print("CONFIDENCE TOKEN SMOKE TEST")
    print("=" * 70)
    print(f"\nLoading model: {model_name}")
    print("-" * 70)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"✓ Model loaded successfully")
    print(f"  Original vocab size: {len(tokenizer)}")
    print(f"  Original embedding size: {model.get_input_embeddings().weight.shape[0]}")
    
    # Add confidence token
    print("\n" + "-" * 70)
    print("Adding <|CONF|> token...")
    print("-" * 70)
    
    conf_token_id = add_conf_token(tokenizer, model)
    
    # Verify token addition
    print("\n" + "-" * 70)
    print("VERIFICATION RESULTS")
    print("-" * 70)
    
    # Test 1: Token ID
    print(f"\n1. Token ID of <|CONF|>: {conf_token_id}")
    if conf_token_id is not None and conf_token_id >= 0:
        print("   ✓ Valid token ID")
    else:
        print("   ✗ Invalid token ID")
        return False
    
    # Test 2: Token in vocabulary
    print(f"\n2. Token in vocabulary:")
    decoded = tokenizer.decode([conf_token_id])
    print(f"   Decoded token: '{decoded}'")
    roundtrip = tokenizer.convert_tokens_to_ids("<|CONF|>")
    print(f"   Roundtrip conversion: {roundtrip}")
    
    if roundtrip == conf_token_id:
        print("   ✓ Token exists in vocabulary")
    else:
        print("   ✗ Token not properly added to vocabulary")
        return False
    
    # Test 3: Embedding matrix resized
    print(f"\n3. Embedding matrix resize:")
    new_vocab_size = len(tokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    embedding_dim = model.get_input_embeddings().weight.shape[1]
    
    print(f"   New vocab size: {new_vocab_size}")
    print(f"   Embedding matrix shape: ({embedding_size}, {embedding_dim})")
    
    if new_vocab_size == embedding_size:
        print("   ✓ Embedding matrix properly resized")
    else:
        print(f"   ✗ Size mismatch: vocab={new_vocab_size}, embeddings={embedding_size}")
        return False
    
    # Test 4: Embedding initialization
    print(f"\n4. Embedding initialization:")
    conf_embedding = model.get_input_embeddings().weight[conf_token_id]
    print(f"   Embedding shape: {conf_embedding.shape}")
    print(f"   Embedding norm: {conf_embedding.norm().item():.4f}")
    print(f"   Embedding mean: {conf_embedding.mean().item():.4f}")
    print(f"   Embedding std: {conf_embedding.std().item():.4f}")
    
    if conf_embedding.norm().item() > 0:
        print("   ✓ Embedding properly initialized (non-zero)")
    else:
        print("   ✗ Embedding not initialized (zero vector)")
        return False
    
    # Additional verification using utility function
    print("\n" + "-" * 70)
    print("AUTOMATED VERIFICATION")
    print("-" * 70)
    
    results = verify_conf_token(tokenizer, model)
    print(f"\nVerification results:")
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # Final summary
    print("\n" + "=" * 70)
    all_passed = (
        results["token_exists"] and
        results["sizes_match"] and
        results["token_id"] == conf_token_id
    )
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smoke test for <|CONF|> token addition"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Think-SFT",
        help="Model name to use for testing (default: allenai/Olmo-3-7B-Think-SFT). "
             "Try 'Qwen/Qwen3-0.6B' for faster download (~600MB)."
    )
    
    args = parser.parse_args()
    
    success = run_smoke_test(args.model)
    sys.exit(0 if success else 1)

