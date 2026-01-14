"""
Training data preparation for confidence token training.

Handles:
- Dataset loading and formatting with <|CONF|> token
- Position tracking for confidence loss computation

Dataset Structure (akenginorhun/* datasets):
    - problem: The question/prompt text
    - answer: Ground truth answer (letter for MCQ)
    - model_metrics: Dict keyed by model name, each containing:
        - evaluation.is_correct: Boolean correctness label
        - lm_response: Model's generated answer
    - dataset_metadata: Original question metadata
"""

from typing import Optional, Dict, List
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


# =============================================================================
# Dataset Configuration
# =============================================================================

# Default model to use for extracting responses and labels
DEFAULT_MODEL = "allenai/Olmo-3-7B-Think-SFT"

DATASET_CONFIGS = {
    "mmlu_pro": {
        "path": "akenginorhun/mmlu-pro_10k_seed1_Olmo-3_family_metrics",
        "split_map": {"train": "train", "test": "test"},
    },
    "supergpqa": {
        "path": "akenginorhun/supergpqa_10k_seed1_Olmo-3_family_metrics",
        "split_map": {"train": "train", "test": "test"},
    },
    "wildchat": {
        "path": "akenginorhun/wildchat-4.8m_10k_seed1_Olmo-3_family_metrics_extended",
        "split_map": {"train": "train", "test": "test"},
    },
    "natural_reasoning": {
        "path": "akenginorhun/natural_reasoning_10k_seed1_Olmo-3_family_metrics",
        "split_map": {"train": "train", "test": "test"},
    },
}


def extract_from_nested(example: Dict, model_name: str = DEFAULT_MODEL) -> Dict:
    """
    Extract question, answer, and correctness from nested dataset structure.
    
    Args:
        example: Raw dataset row
        model_name: Which model's responses/labels to use
        
    Returns:
        Dict with:
            - question
            - answer
            - is_correct
            - source_model: which model was used
            - used_fallback: bool, true if default model was missing
    """
    # Question is always in 'problem' column
    question = example.get("problem", "")
    
    # Get model-specific metrics
    model_metrics = example.get("model_metrics", {})
    
    used_fallback = False
    if model_name not in model_metrics:
        # Fallback: try to find any available model
        available_models = list(model_metrics.keys())
        if available_models:
            model_name = available_models[0]
            used_fallback = True
        else:
            raise ValueError(f"No model metrics found in example. Available: {model_metrics.keys()}")
    
    model_data = model_metrics[model_name]
    
    # Extract model's response (answer)
    answer = model_data.get("lm_response", "")
    
    # Extract correctness label from evaluation
    evaluation = model_data.get("evaluation", {})
    is_correct = evaluation.get("is_correct", False)
    
    return {
        "question": question,
        "answer": answer,
        "is_correct": float(is_correct),  # Convert bool to float for training
        "source_model": model_name,
        "used_fallback": used_fallback,
    }


# =============================================================================
# Prompt Formatting
# =============================================================================

# Available CONF token positions
CONF_POSITIONS = ["suffix", "posterior"]


def format_suffix_prompt(question: str, answer: str) -> str:
    """
    Format prompt with <|CONF|> in suffix position (after question, before answer).
    
    Format:
        {question} <|CONF|> {answer}
    """
    return f"{question} <|CONF|> {answer}"


def format_posterior_prompt(question: str, answer: str) -> str:
    """
    Format prompt with <|CONF|> in posterior position (after question AND answer).
    
    This gives the CONF token access to both question and answer as prior context,
    which may help encode correctness information better.
    
    Format:
        {question} {answer} <|CONF|>
    """
    return f"{question} {answer} <|CONF|>"


def format_prompt(question: str, answer: str, conf_position: str = "suffix") -> str:
    """
    Format prompt with <|CONF|> token at specified position.
    
    Args:
        question: The question text
        answer: The answer text
        conf_position: Where to place <|CONF|>
            - "suffix": {question} <|CONF|> {answer}
            - "posterior": {question} {answer} <|CONF|>
    
    Returns:
        Formatted prompt string
    """
    if conf_position == "suffix":
        return format_suffix_prompt(question, answer)
    elif conf_position == "posterior":
        return format_posterior_prompt(question, answer)
    else:
        raise ValueError(f"Unknown conf_position: {conf_position}. Choose from: {CONF_POSITIONS}")


def format_suffix_prompt_inference(question: str) -> str:
    """
    Format prompt for inference (no answer yet) - suffix position.
    
    Format:
        {question} <|CONF|>
    """
    return f"{question} <|CONF|>"


def format_posterior_prompt_inference(question: str, answer: str) -> str:
    """
    Format prompt for inference - posterior position (needs answer to compute CONF position).
    
    Format:
        {question} {answer} <|CONF|>
    """
    return f"{question} {answer} <|CONF|>"


def format_prompt_inference(question: str, answer: str = "", conf_position: str = "suffix") -> str:
    """
    Format prompt for inference with <|CONF|> token at specified position.
    
    Args:
        question: The question text
        answer: The answer text (required for posterior, optional for suffix)
        conf_position: Where to place <|CONF|>
    
    Returns:
        Formatted prompt string for inference
    """
    if conf_position == "suffix":
        return format_suffix_prompt_inference(question)
    elif conf_position == "posterior":
        return format_posterior_prompt_inference(question, answer)
    else:
        raise ValueError(f"Unknown conf_position: {conf_position}. Choose from: {CONF_POSITIONS}")


def get_conf_token_position(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Find the position of <|CONF|> token in any formatted text.
    
    Works for both suffix and posterior positions by scanning for the token.
    
    Args:
        text: The formatted prompt containing <|CONF|>
        tokenizer: Tokenizer with <|CONF|> added
        
    Returns:
        Position (0-indexed) of the <|CONF|> token, or -1 if not found
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
    
    if conf_token_id in tokens:
        return tokens.index(conf_token_id)
    return -1


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_confidence_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
    conf_position: str = "suffix",
    custom_config: Optional[Dict] = None,
) -> Dataset:
    """
    Prepare dataset with <|CONF|> token at specified position.
    
    Args:
        dataset_name: Name of dataset (key in DATASET_CONFIGS) or custom
        tokenizer: Tokenizer with <|CONF|> token added
        split: Dataset split ("train" or "test")
        max_samples: Optional limit on number of samples
        model_name: Which model's responses/labels to use from model_metrics
        conf_position: Where to place <|CONF|> token
            - "suffix": {question} <|CONF|> {answer}
            - "posterior": {question} {answer} <|CONF|>
        custom_config: Optional custom dataset configuration
        
    Returns:
        Formatted Dataset with columns:
            - text: Formatted prompt string
            - confidence_label: Target confidence value [0, 1]
            - conf_token_position: Position of <|CONF|> token
    """
    if conf_position not in CONF_POSITIONS:
        raise ValueError(f"Unknown conf_position: {conf_position}. Choose from: {CONF_POSITIONS}")
    
    # Get dataset configuration
    if custom_config:
        config = custom_config
    elif dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")
    
    # Load raw dataset
    actual_split = config.get("split_map", {}).get(split, split)
    dataset = load_dataset(config["path"], split=actual_split)
    
    # Limit samples if requested
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    fallback_counter = {"used": 0, "total": 0}
    
    def format_example(example):
        fallback_counter["total"] += 1
        # Extract from nested structure
        extracted = extract_from_nested(example, model_name)
        if extracted.get("used_fallback"):
            fallback_counter["used"] += 1
        
        question = str(extracted["question"])
        answer = str(extracted["answer"])
        confidence_label = float(extracted["is_correct"])
        
        # Clamp to [0, 1]
        confidence_label = max(0.0, min(1.0, confidence_label))
        
        # Format with <|CONF|> at specified position
        text = format_prompt(question, answer, conf_position)
        
        # conf_token_position is computed after tokenization, placeholder here
        return {
            "text": text,
            "confidence_label": confidence_label,
            "conf_token_position": -1,  # Will be recomputed during tokenization
        }
    
    # Apply formatting
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc=f"Formatting {dataset_name} ({split}) [{conf_position}] using {model_name}"
    )
    
    if fallback_counter["used"] > 0:
        print(
            f"⚠ Model metrics fallback used for {fallback_counter['used']}/{fallback_counter['total']} "
            f"samples (requested model: {model_name})"
        )
    else:
        print(f"✓ Model metrics all from requested model: {model_name}")
    print(f"✓ CONF position: {conf_position}")
    
    return formatted_dataset


# Alias for backwards compatibility
def prepare_suffix_confidence_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
    custom_config: Optional[Dict] = None,
) -> Dataset:
    """Backwards-compatible alias for prepare_confidence_dataset with suffix position."""
    return prepare_confidence_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split=split,
        max_samples=max_samples,
        model_name=model_name,
        conf_position="suffix",
        custom_config=custom_config,
    )


def prepare_multiple_datasets(
    dataset_names: List[str],
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_samples_per_dataset: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
    conf_position: str = "suffix",
) -> Dataset:
    """
    Prepare and concatenate multiple datasets.
    
    Args:
        dataset_names: List of dataset names to load
        tokenizer: Tokenizer with <|CONF|> token added
        split: Dataset split ("train" or "test")
        max_samples_per_dataset: Max samples per dataset
        model_name: Which model's responses/labels to use
        conf_position: Where to place <|CONF|> ("suffix" or "posterior")
    """
    from datasets import concatenate_datasets
    
    datasets = []
    for name in dataset_names:
        ds = prepare_confidence_dataset(
            name, tokenizer, split, max_samples_per_dataset, model_name, conf_position
        )
        datasets.append(ds)
        print(f"✓ Loaded {name}: {len(ds)} examples")
    
    combined = concatenate_datasets(datasets)
    print(f"✓ Combined dataset: {len(combined)} total examples")
    
    return combined


# =============================================================================
# Tokenization for Training
# =============================================================================

def tokenize_for_training(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 4096,
) -> Dict[str, List]:
    """
    Tokenize examples for training and recompute CONF positions after tokenization.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Recompute CONF token position post-tokenization to avoid special-token offsets
    conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
    conf_positions = []
    valid_conf = []
    truncated = []
    for ids in tokenized["input_ids"]:
        truncated.append(len(ids) == max_length)
        if conf_token_id in ids:
            conf_positions.append(ids.index(conf_token_id))
            valid_conf.append(True)
        else:
            conf_positions.append(-1)
            valid_conf.append(False)
    
    tokenized["confidence_label"] = examples["confidence_label"]
    tokenized["conf_token_position"] = conf_positions
    tokenized["valid_conf"] = valid_conf
    tokenized["truncated"] = truncated
    
    return tokenized


def get_tokenized_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 4096,
    include_conf_fields: bool = True,
    drop_invalid_conf: bool = True,
) -> Dataset:
    """
    Tokenize a formatted dataset for training.
    """
    tokenized = dataset.map(
        lambda x: tokenize_for_training(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    total_rows = len(tokenized)
    valid_conf_mask = tokenized["valid_conf"]
    truncated_mask = tokenized["truncated"]
    invalid_conf = sum(1 for v in valid_conf_mask if not v)
    truncated = sum(1 for t in truncated_mask if t)
    
    if include_conf_fields and drop_invalid_conf:
        tokenized = tokenized.filter(lambda ex: ex["valid_conf"])
    
    print(
        f"Tokenization summary: total={total_rows}, "
        f"invalid_conf={invalid_conf}, truncated={truncated}, "
        f"kept={len(tokenized)}"
    )
    if invalid_conf > 0:
        print(f"⚠ Dropped {invalid_conf} samples without a valid <|CONF|> token")
    if truncated > 0:
        print(f"⚠ {truncated} samples were truncated to max_length={max_length}")
    
    # Remove helper fields that the trainer/collator should not see
    drop_cols = ["valid_conf", "truncated"]
    if include_conf_fields:
        # keep confidence_label and conf_token_position
        cols_to_remove = [c for c in drop_cols if c in tokenized.column_names]
        tokenized = tokenized.remove_columns(cols_to_remove)
    else:
        # remove all confidence-related columns
        cols_to_remove = [
            c for c in ["confidence_label", "conf_token_position"] + drop_cols
            if c in tokenized.column_names
        ]
        tokenized = tokenized.remove_columns(cols_to_remove)
    
    return tokenized
