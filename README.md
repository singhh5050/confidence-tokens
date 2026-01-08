# Confidence Tokens

Training models with `<|CONF|>` suffix token for confidence estimation.

## Overview

Insert a `<|CONF|>` token after the question but before the answer. The token's hidden state encodes confidence information about the subsequent answer.

**Prompt Format:**
```
Problem to solve: {question}
Confidence: <|CONF|>
Answer: {answer}
```

**Two Training Approaches:**
- **Approach A (SFT)**: Standard language modeling, confidence learned implicitly
- **Approach B (Supervised)**: Explicit supervision on `<|CONF|>` hidden state

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
confidence-tokens/
├── src/
│   ├── tokenizer_utils.py   # add_conf_token()
│   ├── data.py              # Dataset prep & formatting
│   └── training.py          # SFT + Supervised trainers
├── scripts/
│   ├── smoke_token.py       # Verify token addition
│   ├── smoke_forward.py     # Verify hidden state extraction
│   └── train.py             # Training entry point
└── requirements.txt
```

## Quick Start

### 1. Verify Token Mechanics

```bash
# Verify token addition (use --model Qwen/Qwen3-0.6B for quick testing)
python scripts/smoke_token.py --model Qwen/Qwen3-0.6B

# Verify hidden state extraction
python scripts/smoke_forward.py --model Qwen/Qwen3-0.6B
```

### 2. Train

```bash
# Approach A: SFT only (default)
python scripts/train.py --max-samples 100 --epochs 1

# Approach B: Supervised confidence training (recommended)
python scripts/train.py --supervised --max-samples 100 --epochs 1

# Full training with default model (Olmo-3-7B-Think)
python scripts/train.py --supervised --dataset mmlu_pro
```

## Usage

### Add Confidence Token

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tokenizer_utils import add_conf_token

tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Think")

conf_token_id = add_conf_token(tokenizer, model)
```

### Extract Confidence

```python
from src.data import format_suffix_prompt, get_conf_token_position

# Format prompt
prompt = format_suffix_prompt("What is 2+2?", "4")
conf_pos = get_conf_token_position("What is 2+2?", tokenizer)

# Forward pass
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)

# Extract hidden state at CONF position
conf_hidden = outputs.hidden_states[-1][0, conf_pos, :]  # Shape: [hidden_dim]
```

### Train Model

```python
from src.data import prepare_suffix_confidence_dataset, get_tokenized_dataset
from src.training import ConfidenceTrainingConfig, train_confidence_model

# Prepare data
train_data = prepare_suffix_confidence_dataset("mmlu_pro", tokenizer, "train")
train_data = get_tokenized_dataset(train_data, tokenizer)

# Approach A: SFT only
config = ConfidenceTrainingConfig(output_dir="./output", supervised=False)

# Approach B: Supervised (recommended)
config = ConfidenceTrainingConfig(
    output_dir="./output", 
    supervised=True, 
    confidence_loss_weight=0.3  # α: loss = (1-α)*LM + α*Conf
)

train_confidence_model(model, tokenizer, train_data, config=config, conf_token_id=conf_token_id)
```

## Datasets

All datasets contain Olmo-3-7B-Think generated responses with correctness labels.

| Dataset | Path |
|---------|------|
| `mmlu_pro` | `akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics` |
| `supergpqa` | `akenginorhun/supergpqa_10k_seed1_Olmo-3_family_metrics` |
| `wildchat` | `akenginorhun/wildchat-4.8m_10k_seed1_Olmo-3_family_metrics_extended` |
| `natural_reasoning` | `akenginorhun/natural_reasoning_10k_seed1_Olmo-3_family_metrics` |

## Training Approaches

### Approach A: SFT Only
- Standard next-token prediction loss
- Hope that `<|CONF|>` hidden state implicitly encodes confidence
- Probe later to extract confidence signal

### Approach B: Supervised (Recommended)
- Combines LM loss with explicit confidence prediction
- `Loss = (1-α) * LM_loss + α * BCE(confidence_head(h_CONF), is_correct)`
- The confidence head gradient backprops through the transformer
- Model learns to encode correctness signal at `<|CONF|>` position
