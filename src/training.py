"""
Training utilities for confidence token models.

Two approaches:
- Approach A (SFT): Standard language modeling, confidence learned implicitly
- Approach B (Supervised): Explicit supervision on <|CONF|> hidden state
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from datasets import Dataset


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class ConfidenceTrainingConfig:
    """Configuration for confidence token training."""
    
    # Output
    output_dir: str = "./olmo3-7b-conf-suffix-sft"
    
    # Training approach
    supervised: bool = False  # False = Approach A (SFT), True = Approach B (Supervised)
    confidence_loss_weight: float = 0.3  # Alpha for Approach B: (1-α)*LM + α*Conf
    
    # Batch size
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Learning rate
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Reporting
    report_to: str = "wandb"
    run_name: Optional[str] = None
    
    # Other
    dataloader_num_workers: int = 4
    seed: int = 42
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.eval_strategy,
            bf16=self.bf16,
            fp16=self.fp16,
            report_to=self.report_to,
            run_name=self.run_name,
            dataloader_num_workers=self.dataloader_num_workers,
            seed=self.seed,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )


# =============================================================================
# Approach B: Supervised Confidence Trainer
# =============================================================================

class ConfidenceDataCollator:
    """
    Data collator that handles both LM inputs and confidence labels.
    Pads sequences and extracts confidence-specific fields.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm: bool = False):
        self.tokenizer = tokenizer
        self.lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract confidence-specific fields before LM collation
        confidence_labels = torch.tensor(
            [f.pop("confidence_label") for f in features], 
            dtype=torch.float
        )
        conf_token_positions = torch.tensor(
            [f.pop("conf_token_position") for f in features],
            dtype=torch.long
        )
        
        # Standard LM collation for input_ids, attention_mask, labels
        batch = self.lm_collator(features)
        
        # Add back confidence fields
        batch["confidence_labels"] = confidence_labels
        batch["conf_token_positions"] = conf_token_positions
        
        return batch


class SuffixConfidenceTrainer(Trainer):
    """
    Custom trainer that adds supervised loss on the <|CONF|> token's hidden state.
    
    Approach B: Combines LM loss with explicit confidence prediction loss.
    
    Loss = (1 - alpha) * LM_loss + alpha * BCE(confidence_head(h_CONF), is_correct)
    
    The confidence head learns to predict correctness from the hidden state at 
    the <|CONF|> position. Gradients flow back through the transformer, training
    it to encode confidence-relevant information at that position.
    """
    
    def __init__(
        self, 
        *args, 
        conf_token_id: Optional[int] = None,
        alpha: float = 0.3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conf_token_id = conf_token_id
        self.alpha = alpha  # Weight for confidence supervision loss
        
        # Linear probe to predict confidence from hidden state
        hidden_size = self.model.config.hidden_size
        self.confidence_head = nn.Linear(hidden_size, 1)
        
        # Move to same device as model
        if hasattr(self.model, 'device'):
            self.confidence_head = self.confidence_head.to(self.model.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract custom labels (pop so they don't go to model forward)
        confidence_labels = inputs.pop("confidence_labels").float()  # (batch,)
        conf_token_positions = inputs.pop("conf_token_positions")    # (batch,)
        
        # Ensure confidence head is on correct device
        if self.confidence_head.weight.device != model.device:
            self.confidence_head = self.confidence_head.to(model.device)
        
        # Standard forward pass with hidden states
        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss
        
        # Extract hidden states at <|CONF|> positions
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        batch_size = last_hidden.size(0)
        
        # Gather hidden states at the <|CONF|> position for each example
        conf_hiddens = []
        for i in range(batch_size):
            pos = conf_token_positions[i].item()
            conf_hiddens.append(last_hidden[i, pos, :])
        conf_hidden = torch.stack(conf_hiddens)  # (batch, hidden_size)
        
        # Predict confidence from hidden state
        conf_logits = self.confidence_head(conf_hidden).squeeze(-1)  # (batch,)
        
        # Confidence supervision loss (Binary Cross-Entropy)
        conf_loss = F.binary_cross_entropy_with_logits(
            conf_logits, 
            confidence_labels.to(conf_logits.device)
        )
        
        # Combined loss: (1-α) * LM + α * Confidence
        total_loss = (1 - self.alpha) * lm_loss + self.alpha * conf_loss
        
        # Log both losses for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "lm_loss": lm_loss.item(),
                "conf_loss": conf_loss.item(),
                "conf_accuracy": ((conf_logits > 0).float() == confidence_labels.to(conf_logits.device)).float().mean().item(),
            })
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save both the model and the confidence head."""
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # Save confidence head separately
        confidence_head_path = f"{output_dir}/confidence_head.pt"
        torch.save(self.confidence_head.state_dict(), confidence_head_path)
        print(f"✓ Saved confidence head to {confidence_head_path}")


# =============================================================================
# Approach A: Standard SFT Trainer
# =============================================================================

def create_sft_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[ConfidenceTrainingConfig] = None,
) -> Trainer:
    """
    Create a standard SFT trainer (Approach A).
    
    The confidence token learns representations implicitly through standard 
    language modeling. The token's hidden state naturally encodes question 
    difficulty as the model learns to predict answers.
    """
    if config is None:
        config = ConfidenceTrainingConfig()
    
    training_args = config.to_training_arguments()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer


def create_supervised_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[ConfidenceTrainingConfig] = None,
    conf_token_id: Optional[int] = None,
) -> SuffixConfidenceTrainer:
    """
    Create a supervised confidence trainer (Approach B).
    
    Combines LM loss with explicit confidence prediction loss:
    Loss = (1 - alpha) * LM_loss + alpha * BCE(confidence_head(h_CONF), is_correct)
    
    The confidence head learns to predict correctness from the hidden state at 
    the <|CONF|> position. Gradients backprop through the transformer.
    """
    if config is None:
        config = ConfidenceTrainingConfig(supervised=True)
    
    training_args = config.to_training_arguments()
    
    # Use custom data collator that handles confidence labels
    data_collator = ConfidenceDataCollator(tokenizer=tokenizer)
    
    trainer = SuffixConfidenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        conf_token_id=conf_token_id,
        alpha=config.confidence_loss_weight,
    )
    
    return trainer


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[ConfidenceTrainingConfig] = None,
    conf_token_id: Optional[int] = None,
) -> Trainer:
    """
    Create the appropriate trainer based on config.
    
    Args:
        config.supervised = False → Approach A (SFT only)
        config.supervised = True  → Approach B (Supervised confidence)
    """
    if config is None:
        config = ConfidenceTrainingConfig()
    
    if config.supervised:
        return create_supervised_trainer(
            model, tokenizer, train_dataset, eval_dataset, config, conf_token_id
        )
    else:
        return create_sft_trainer(
            model, tokenizer, train_dataset, eval_dataset, config
        )


# =============================================================================
# Training Entry Point
# =============================================================================

def train_confidence_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[ConfidenceTrainingConfig] = None,
    conf_token_id: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a confidence token model.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer with <|CONF|> token added
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        config: Training configuration (config.supervised controls approach)
        conf_token_id: ID of the <|CONF|> token (required for supervised training)
        resume_from_checkpoint: Optional checkpoint path to resume from
        
    Returns:
        Training metrics dictionary
    """
    print("=" * 70)
    print("CONFIDENCE TOKEN TRAINING")
    print("=" * 70)
    
    if config is None:
        config = ConfidenceTrainingConfig()
    
    approach = "B (Supervised)" if config.supervised else "A (SFT only)"
    
    print(f"\nApproach: {approach}")
    print(f"\nConfiguration:")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  BF16: {config.bf16}")
    
    if config.supervised:
        print(f"  Confidence loss weight (α): {config.confidence_loss_weight}")
        print(f"  Loss = {1 - config.confidence_loss_weight:.1f} * LM + {config.confidence_loss_weight:.1f} * Conf")
    
    print(f"\nDatasets:")
    print(f"  Train: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"  Eval: {len(eval_dataset)} examples")
    
    # Create appropriate trainer based on config
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        conf_token_id=conf_token_id,
    )
    
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("\n" + "-" * 70)
    print("Saving model...")
    print("-" * 70)
    
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nApproach: {approach}")
    print(f"Model saved to: {config.output_dir}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    
    if config.supervised:
        print(f"(Confidence head saved to: {config.output_dir}/confidence_head.pt)")
    
    return metrics
