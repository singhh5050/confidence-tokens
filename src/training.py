"""
Training utilities for confidence token models.

Two approaches:
- Approach A (SFT): Standard language modeling, confidence learned implicitly
- Approach B (Supervised): Explicit supervision on <|CONF|> hidden state
"""

import os
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
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
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
    save_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Memory optimization (CRITICAL for 7B models on single GPU)
    gradient_checkpointing: bool = True  # Recompute activations to save memory
    optim: str = "paged_adamw_8bit"  # Paged 8-bit optimizer (bnb), stable across HF versions
    
    # Reporting
    report_to: str = "wandb"
    run_name: Optional[str] = None
    
    # Other
    dataloader_num_workers: int = 4
    seed: int = 42
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        # Only load best model if we're doing evaluation
        load_best = self.eval_strategy != "no"
        
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
            save_strategy=self.save_strategy,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.eval_strategy,
            bf16=self.bf16,
            fp16=self.fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            optim=self.optim,
            report_to=self.report_to,
            run_name=self.run_name,
            dataloader_num_workers=self.dataloader_num_workers,
            seed=self.seed,
            remove_unused_columns=False,
            load_best_model_at_end=load_best,
            metric_for_best_model="eval_loss" if load_best else None,
            greater_is_better=False,
        )


# =============================================================================
# Optimizer helper
# =============================================================================

class SFTTrainer(Trainer):
    """
    Trainer subclass that force-enables bitsandbytes 8-bit optimizers when requested.
    
    HuggingFace will silently fall back to torch.optim.AdamW if bitsandbytes is
    unavailable. We override create_optimizer to:
      1) Detect the 8-bit request (optim contains "8bit")
      2) Import and construct the bnb optimizer explicitly
      3) Raise a clear error if bitsandbytes is missing
    """

    def create_optimizer(self):
        if self.optimizer is not None:
            return

        use_8bit = "8bit" in str(self.args.optim).lower()

        # Build parameter groups (match HF Trainer logic)
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        if use_8bit:
            try:
                import bitsandbytes as bnb  # type: ignore
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Requested 8-bit optimizer but bitsandbytes is not available. "
                    "Install with `pip install bitsandbytes` and ensure CUDA is present."
                ) from exc

            if "paged" in str(self.args.optim).lower():
                optimizer_cls = bnb.optim.PagedAdamW8bit
            else:
                optimizer_cls = bnb.optim.AdamW8bit
            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
                "weight_decay": self.args.weight_decay,
            }
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        else:
            # Defer to HF for non-8bit paths
            super().create_optimizer()


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
        self.mlm = mlm
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract confidence-specific fields before padding
        confidence_labels = torch.tensor(
            [f.pop("confidence_label") for f in features], 
            dtype=torch.float
        )
        conf_token_positions = torch.tensor(
            [f.pop("conf_token_position") for f in features],
            dtype=torch.long
        )
        
        # Manually pad input_ids and labels to same length
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        input_ids_padded = []
        labels_padded = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        for f in features:
            seq_len = len(f["input_ids"])
            padding_len = max_len - seq_len
            
            # Pad input_ids
            input_ids_padded.append(f["input_ids"] + [pad_token_id] * padding_len)
            
            # Pad labels with -100 (ignored in loss)
            labels_padded.append(f["labels"] + [-100] * padding_len)
            
            # Create attention mask
            attention_masks.append([1] * seq_len + [0] * padding_len)
        
        batch = {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "confidence_labels": confidence_labels,
            "conf_token_positions": conf_token_positions,
        }
        
        return batch


class SuffixConfidenceTrainer(SFTTrainer):
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
        self._invalid_label_warning_count = 0
        
        # Linear probe to predict confidence from hidden state
        hidden_size = self.model.config.hidden_size
        
        # Create with same dtype as model to avoid dtype mismatch
        model_dtype = next(self.model.parameters()).dtype
        self.confidence_head = nn.Linear(hidden_size, 1, dtype=model_dtype)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        self.confidence_head = self.confidence_head.to(device)
    
    def create_optimizer(self):
        """Override to include confidence_head parameters in optimizer."""
        if self.optimizer is not None:
            return
        
        # First, call parent to set up model parameters
        super().create_optimizer()
        
        # Add confidence_head parameters to the optimizer
        # Use same LR as model, no weight decay (it's a small probe)
        conf_head_params = list(self.confidence_head.parameters())
        
        # Add to existing optimizer's param_groups
        self.optimizer.add_param_group({
            "params": conf_head_params,
            "weight_decay": 0.0,  # No weight decay for probe
            "lr": self.args.learning_rate,
        })
        
        # Verify confidence head is actually in optimizer (fail loudly if not)
        num_param_groups = len(self.optimizer.param_groups)
        conf_head_param_count = sum(p.numel() for p in conf_head_params)
        print(f"✓ Added confidence_head to optimizer ({len(conf_head_params)} tensors, {conf_head_param_count} params)")
        print(f"✓ Optimizer has {num_param_groups} param groups")
        
        # Should have at least 3 groups: decay, no_decay, conf_head
        if num_param_groups < 3:
            raise RuntimeError(
                f"Expected at least 3 param groups in optimizer, got {num_param_groups}. "
                "Confidence head may not be in optimizer!"
            )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract custom labels (pop so they don't go to model forward)
        confidence_labels = inputs.pop("confidence_labels").float()  # (batch,)
        conf_token_positions = inputs.pop("conf_token_positions")    # (batch,)
        labels = inputs.get("labels")
        
        # IMPORTANT: Disable cache for gradient checkpointing compatibility
        model.config.use_cache = False
        
        # Memory-efficient forward: get last_hidden_state directly from base model
        # This avoids storing ALL layer hidden states (massive memory savings)
        base_model = getattr(model, "model", None) or getattr(model, "base_model", None)
        if base_model is None:
            raise AttributeError(
                "Could not locate base transformer on model; expected `model.model` "
                "or `model.base_model`. Please verify the architecture."
            )
        
        base_outputs = base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        )
        last_hidden = base_outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Ensure confidence head and labels align with last_hidden device/dtype
        hidden_device = last_hidden.device
        hidden_dtype = last_hidden.dtype
        if (
            self.confidence_head.weight.device != hidden_device
            or self.confidence_head.weight.dtype != hidden_dtype
        ):
            self.confidence_head = self.confidence_head.to(
                device=hidden_device,
                dtype=hidden_dtype,
            )
        
        # Compute LM logits and loss manually
        lm_head_device = model.lm_head.weight.device
        logits = model.lm_head(last_hidden.to(lm_head_device))  # (batch, seq_len, vocab)
        
        # Shift for autoregressive LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if shift_labels.device != shift_logits.device:
            shift_labels = shift_labels.to(shift_logits.device)
        # Guard against out-of-range labels (prevents CUDA device asserts)
        vocab_size = shift_logits.size(-1)
        invalid_mask_labels = (shift_labels != -100) & (
            (shift_labels < 0) | (shift_labels >= vocab_size)
        )
        if invalid_mask_labels.any():
            invalid_count = int(invalid_mask_labels.sum().item())
            valid_labels = shift_labels[shift_labels != -100]
            max_label = int(valid_labels.max().item()) if valid_labels.numel() > 0 else -1
            min_label = int(valid_labels.min().item()) if valid_labels.numel() > 0 else -1
            if self._invalid_label_warning_count == 0:
                print(
                    f"⚠ Found {invalid_count} labels out of vocab "
                    f"(range [{min_label}, {max_label}], vocab_size={vocab_size}); "
                    "masking to -100. (warning shown once)"
                )
            self._invalid_label_warning_count += 1
            shift_labels = shift_labels.masked_fill(invalid_mask_labels, -100)

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # Vectorized gather of CONF hidden states (no loop needed)
        batch_size = last_hidden.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_device)
        seq_len = last_hidden.size(1)
        conf_positions = conf_token_positions.to(hidden_device)
        
        # Guard against any out-of-bounds or invalid positions (should be rare after filtering)
        # Note: position -1 would index the last token in Python, so we must check < 0 too
        invalid_mask = (conf_positions >= seq_len) | (conf_positions < 0)
        if invalid_mask.any():
            # Clamp to valid range to avoid crashes; loss masking handles invalid entries
            conf_positions = torch.clamp(conf_positions, min=0, max=seq_len - 1)
        
        conf_hidden = last_hidden[batch_idx, conf_positions]  # (batch, hidden)
        
        # Predict confidence from hidden state
        conf_logits = self.confidence_head(conf_hidden).squeeze(-1)  # (batch,)
        
        # Confidence supervision loss (Binary Cross-Entropy)
        conf_targets = confidence_labels.to(hidden_device)
        bce = F.binary_cross_entropy_with_logits(
            conf_logits, 
            conf_targets,
            reduction="none",
        )
        valid_mask = (~invalid_mask).float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        conf_loss = (bce * valid_mask).sum() / valid_count
        
        # Combined loss: (1-α) * LM + α * Confidence
        total_loss = (1 - self.alpha) * lm_loss + self.alpha * conf_loss
        
        # Log both losses for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                conf_probs = torch.sigmoid(conf_logits)
            self.log({
                "lm_loss": lm_loss.item(),
                "conf_loss": conf_loss.item(),
                "conf_accuracy": ((conf_logits > 0).float() == confidence_labels.to(hidden_device)).float().mean().item(),
                "conf_logit_mean": conf_logits.mean().item(),
                "conf_logit_std": conf_logits.std().item(),
                "conf_prob_mean": conf_probs.mean().item(),
                "conf_prob_std": conf_probs.std().item(),
                "conf_invalid_frac": invalid_mask.float().mean().item(),
            })
        
        if return_outputs:
            # Return CausalLMOutput for compatibility with Trainer eval loop
            from transformers.modeling_outputs import CausalLMOutput
            return total_loss, CausalLMOutput(loss=total_loss, logits=logits)
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
    
    # Simple padding collator for LM (handles variable length sequences)
    def lm_collator(features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        input_ids_padded = []
        labels_padded = []
        attention_masks = []
        
        for f in features:
            seq_len = len(f["input_ids"])
            padding_len = max_len - seq_len
            input_ids_padded.append(f["input_ids"] + [pad_token_id] * padding_len)
            labels_padded.append(f["labels"] + [-100] * padding_len)
            attention_masks.append([1] * seq_len + [0] * padding_len)
        
        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lm_collator,
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
    print(f"  Optimizer: {config.optim}")
    
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
    
    # Force optimizer creation to confirm 8-bit optimizer is used
    trainer.create_optimizer()
    optim_cls = type(trainer.optimizer).__name__ if trainer.optimizer else "None"
    print(f"Optimizer check: args.optim={trainer.args.optim}, class={optim_cls}")
    if optim_cls == "AdamW":
        print("⚠ Warning: 8-bit optimizer not in use (got torch.optim.AdamW). Expect higher memory.")
    
    # If resuming supervised training, restore the confidence head
    if config.supervised and resume_from_checkpoint:
        head_path = f"{resume_from_checkpoint}/confidence_head.pt"
        if os.path.exists(head_path) and isinstance(trainer, SuffixConfidenceTrainer):
            state_dict = torch.load(head_path, map_location=next(model.parameters()).device)
            trainer.confidence_head.load_state_dict(state_dict)
            print(f"✓ Loaded confidence head from {head_path}")
        else:
            print(f"⚠ confidence_head.pt not found at {head_path}, starting head from scratch")
    
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
