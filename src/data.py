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
DEFAULT_MODEL = "allenai/Olmo-3-7B-Think"

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
        Dict with 'question', 'answer', 'is_correct'
    """
    # Question is always in 'problem' column
    question = example.get("problem", "")
    
    # Get model-specific metrics
    model_metrics = example.get("model_metrics", {})
    
    if model_name not in model_metrics:
        # Fallback: try to find any available model
        available_models = list(model_metrics.keys())
        if available_models:
            model_name = available_models[0]
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
    }


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_suffix_prompt(question: str, answer: str) -> str:
    """
    Format prompt with <|CONF|> in suffix position (after question, before answer).
    
    Format:
        Problem to solve: {question}
        Confidence: <|CONF|>
        Answer: {answer}
    """
    return f"Problem to solve: {question}\nConfidence: <|CONF|>\nAnswer: {answer}"


def format_suffix_prompt_inference(question: str) -> str:
    """
    Format prompt for inference (no answer yet).
    
    Format:
        Problem to solve: {question}
        Confidence: <|CONF|>
        Answer:
    """
    return f"Problem to solve: {question}\nConfidence: <|CONF|>\nAnswer:"


def get_conf_token_position(question: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Find the position of <|CONF|> token in the formatted prompt.
    
    Args:
        question: The question text
        tokenizer: Tokenizer with <|CONF|> added
        
    Returns:
        Position (0-indexed) of the <|CONF|> token
    """
    prefix = f"Problem to solve: {question}\nConfidence: "
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    return len(prefix_tokens)


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_suffix_confidence_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
    custom_config: Optional[Dict] = None,
) -> Dataset:
    """
    Prepare dataset with <|CONF|> in suffix position (after question, before answer).
    
    Args:
        dataset_name: Name of dataset (key in DATASET_CONFIGS) or custom
        tokenizer: Tokenizer with <|CONF|> token added
        split: Dataset split ("train" or "test")
        max_samples: Optional limit on number of samples
        model_name: Which model's responses/labels to use from model_metrics
        custom_config: Optional custom dataset configuration
        
    Returns:
        Formatted Dataset with columns:
            - text: Formatted prompt string
            - confidence_label: Target confidence value [0, 1]
            - conf_token_position: Position of <|CONF|> token
    """
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
    
    def format_example(example):
        # Extract from nested structure
        extracted = extract_from_nested(example, model_name)
        
        question = str(extracted["question"])
        answer = str(extracted["answer"])
        confidence_label = float(extracted["is_correct"])
        
        # Clamp to [0, 1]
        confidence_label = max(0.0, min(1.0, confidence_label))
        
        # Format with suffix token
        text = format_suffix_prompt(question, answer)
        
        # Find position of <|CONF|> token
        conf_token_position = get_conf_token_position(question, tokenizer)
        
        return {
            "text": text,
            "confidence_label": confidence_label,
            "conf_token_position": conf_token_position,
        }
    
    # Apply formatting
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc=f"Formatting {dataset_name} ({split}) using {model_name}"
    )
    
    return formatted_dataset


def prepare_multiple_datasets(
    dataset_names: List[str],
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_samples_per_dataset: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dataset:
    """
    Prepare and concatenate multiple datasets.
    
    Args:
        dataset_names: List of dataset names to load
        tokenizer: Tokenizer with <|CONF|> token added
        split: Dataset split ("train" or "test")
        max_samples_per_dataset: Max samples per dataset
        model_name: Which model's responses/labels to use
    """
    from datasets import concatenate_datasets
    
    datasets = []
    for name in dataset_names:
        ds = prepare_suffix_confidence_dataset(
            name, tokenizer, split, max_samples_per_dataset, model_name
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
    max_length: int = 2048,
) -> Dict[str, List]:
    """
    Tokenize examples for training.
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
    
    # Pass through confidence labels and positions
    tokenized["confidence_label"] = examples["confidence_label"]
    tokenized["conf_token_position"] = examples["conf_token_position"]
    
    return tokenized


def get_tokenized_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Tokenize a formatted dataset for training.
    """
    return dataset.map(
        lambda x: tokenize_for_training(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
