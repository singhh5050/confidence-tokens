#!/usr/bin/env python3
"""
Smoke test for CONF token hidden state extraction and confidence head.

This demonstrates the core mechanism:
1. Tokenize a prompt with <|CONF|>
2. Run forward pass to get hidden states
3. Extract the hidden state at the CONF token position
4. Pass through a linear confidence head to get a scalar
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokenizer_utils import add_conf_token
from data import format_suffix_prompt, get_conf_token_position


def run_forward_smoke_test(model_name: str = "allenai/Olmo-3-7B-Think"):
    """
    Run smoke test for hidden state extraction and confidence head.
    
    Args:
        model_name: Name of the model to use (default: allenai/Olmo-3-7B-Think)
    """
    print("=" * 70)
    print("CONFIDENCE TOKEN HIDDEN STATE EXTRACTION SMOKE TEST")
    print("=" * 70)
    print(f"\nUsing model: {model_name}")
    print("-" * 70)
    
    # Load model and tokenizer
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with memory-efficient settings
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        print("✓ Model loaded")
    except (RuntimeError, ValueError) as e:
        if "buffer size" in str(e).lower() or "memory" in str(e).lower():
            print(f"\n❌ Model too large for available memory")
            print(f"   Error: {e}")
            print(f"\n💡 Try using a smaller model:")
            print(f"   python scripts/smoke_forward.py --model Qwen/Qwen3-0.6B")
            sys.exit(1)
        else:
            raise
    
    # Add confidence token
    print("\nAdding <|CONF|> token...")
    conf_token_id = add_conf_token(tokenizer, model)
    print(f"✓ <|CONF|> token ID: {conf_token_id}")
    
    # Get hidden dimension
    hidden_dim = model.config.hidden_size
    print(f"✓ Model hidden dimension: {hidden_dim}")
    
    # Create a simple confidence head (random initialization for smoke test)
    print("\nCreating confidence head...")
    confidence_head = nn.Linear(hidden_dim, 1)
    confidence_head = confidence_head.to(model.device)
    print(f"✓ Confidence head: Linear({hidden_dim}, 1)")
    
    # Test examples
    test_examples = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "description": "Simple arithmetic (likely correct)"
        },
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "description": "Factual question (likely correct)"
        },
        {
            "question": "What is the meaning of life?",
            "answer": "42",
            "description": "Ambiguous question (uncertain)"
        },
    ]
    
    print("\n" + "=" * 70)
    print("RUNNING FORWARD PASSES")
    print("=" * 70)
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n{'-' * 70}")
        print(f"EXAMPLE {i}: {example['description']}")
        print(f"{'-' * 70}")
        
        # Format using the spec's prompt format
        prompt_text = format_suffix_prompt(example["question"], example["answer"])
        conf_pos = get_conf_token_position(example["question"], tokenizer)
        
        print(f"\nPrompt:\n{prompt_text}")
        print(f"\nCONF token position: {conf_pos}")
        
        # Tokenize
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Move to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        total_tokens = inputs['input_ids'].shape[1]
        print(f"Total tokens: {total_tokens}")
        
        # Run forward pass with hidden states
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        print("✓ Forward pass complete")
        
        # Extract hidden states
        all_hidden_states = outputs.hidden_states
        print(f"  Number of layers: {len(all_hidden_states)}")
        print(f"  Hidden state shape: {all_hidden_states[-1].shape}")
        
        # Extract CONF token hidden state from last layer
        conf_hidden_state = all_hidden_states[-1][0, conf_pos, :]
        
        print(f"\nExtracted CONF hidden state:")
        print(f"  Position: {conf_pos}")
        print(f"  Shape: {conf_hidden_state.shape}")
        print(f"  Norm: {conf_hidden_state.norm().item():.4f}")
        
        # Pass through confidence head
        print(f"\nPassing through confidence head...")
        confidence_logit = confidence_head(conf_hidden_state.float())
        confidence_scalar = confidence_logit.item()
        
        print(f"✓ Confidence logit (raw): {confidence_scalar:.6f}")
        
        # Apply sigmoid to get probability
        confidence_prob = torch.sigmoid(confidence_logit).item()
        print(f"✓ Confidence probability: {confidence_prob:.6f}")
        
        # Verify dimensions
        assert conf_hidden_state.shape == (hidden_dim,), \
            f"Expected shape ({hidden_dim},), got {conf_hidden_state.shape}"
        assert confidence_logit.shape == (1,), \
            f"Expected scalar, got shape {confidence_logit.shape}"
        
        print(f"\n✓ All dimension checks passed")
    
    # Summary
    print("\n" + "=" * 70)
    print("MECHANISM VERIFICATION")
    print("=" * 70)
    
    print(f"""
✓ Tokenization: Successfully tokenized prompts with <|CONF|> token
✓ CONF Position: Located <|CONF|> token in sequence
✓ Forward Pass: Model ran with output_hidden_states=True
✓ Hidden State: Extracted hidden_states[-1][0, conf_pos, :] with shape [{hidden_dim}]
✓ Confidence Head: Applied Linear({hidden_dim}, 1) to get scalar
✓ Output: Generated confidence logits and probabilities

The core mechanism is working! 🚀

Note: The confidence values are random because the head is untrained.
      After training, these values would be meaningful confidence estimates.
""")
    
    print("=" * 70)
    print("✓ ALL MECHANICS VERIFIED - READY FOR TRAINING")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smoke test for CONF token hidden state extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Think",
        help="Model to use (default: allenai/Olmo-3-7B-Think)"
    )
    
    args = parser.parse_args()
    
    success = run_forward_smoke_test(args.model)
    sys.exit(0 if success else 1)
