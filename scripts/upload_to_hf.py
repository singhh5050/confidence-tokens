#!/usr/bin/env python3
"""Push trained model to HuggingFace Hub."""

import os
import sys
from huggingface_hub import HfApi, login

# === CONFIGURATION ===
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable")
    print("Usage: HF_TOKEN='your_token' python scripts/upload_to_hf.py")
    sys.exit(1)

HF_USERNAME = "singhh5050"
OUTPUT_BASE = "/workspace/confidence-tokens/outputs"

# Auto-detect experiment from available folders
experiments = [d for d in os.listdir(OUTPUT_BASE) if os.path.isdir(f"{OUTPUT_BASE}/{d}")]
if len(experiments) == 1:
    EXPERIMENT = experiments[0]
else:
    print(f"Found experiments: {experiments}")
    EXPERIMENT = input("Enter experiment name: ").strip()

MODEL_PATH = f"{OUTPUT_BASE}/{EXPERIMENT}"
REPO_NAME = f"{HF_USERNAME}/olmo-conf-{EXPERIMENT}"

print(f"\n{'='*60}")
print(f"Pushing: {EXPERIMENT}")
print(f"From:    {MODEL_PATH}")
print(f"To:      https://huggingface.co/{REPO_NAME}")
print(f"{'='*60}\n")

# === LOGIN ===
login(token=HF_TOKEN)
api = HfApi()

# === CREATE REPO ===
print("Creating repository...")
api.create_repo(repo_id=REPO_NAME, exist_ok=True, private=False)

# === UPLOAD MODEL ===
print("Uploading model (this may take a while)...")
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_NAME,
    ignore_patterns=["checkpoint-*", "*.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
)

# === ADD MODEL CARD ===
is_approach_b = EXPERIMENT.startswith("b_")
conf_position = "posterior" if "posterior" in EXPERIMENT else "suffix"

model_card = f"""---
license: apache-2.0
base_model: allenai/Olmo-3-7B-Think-SFT
tags:
- confidence-estimation
- olmo
- fine-tuned
---

# OLMo Confidence Token Model: {EXPERIMENT}

Fine-tuned [allenai/Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) with confidence token training.

## Training Details

- **Approach**: {"B (Supervised Confidence Loss)" if is_approach_b else "A (SFT Only)"}
- **CONF Position**: {conf_position} (`{{question}} {"{{answer}} <|CONF|>" if conf_position == "posterior" else "<|CONF|> {{answer}}"}`)
- **Dataset**: MMLU-Pro 10k with OLMo-3-7B-Think traces
- **Epochs**: 3
- **Batch Size**: 1 (gradient accumulation: 32)
- **Learning Rate**: 2e-5

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{REPO_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

# The model has a special <|CONF|> token for confidence estimation
conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
```

{"## Confidence Head" + chr(10) + chr(10) + "This model includes a trained confidence head (`confidence_head.pt`) for binary correctness prediction." if is_approach_b else ""}
"""

print("Adding model card...")
api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=REPO_NAME,
)

print(f"\n✅ Done! Model available at: https://huggingface.co/{REPO_NAME}")

