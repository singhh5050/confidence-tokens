"""
Utilities for adding and managing the confidence token <|CONF|>.
"""

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


def add_conf_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> int:
    """
    Add the <|CONF|> special token to the tokenizer and resize model embeddings.
    
    This token is designed to be placed after the question but before the answer
    in autoregressive models. Its hidden state serves as a "confidence embedding"
    that can predict whether the subsequent answer will be correct.
    
    Args:
        tokenizer: The tokenizer to add the token to
        model: The model whose embeddings will be resized
        
    Returns:
        The token ID of the newly added <|CONF|> token
        
    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
        >>> model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Think")
        >>> conf_token_id = add_conf_token(tokenizer, model)
        >>> print(f"<|CONF|> token ID: {conf_token_id}")
    """
    # Store original vocab size for verification
    original_vocab_size = len(tokenizer)
    
    # Add special token
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|CONF|>"]
    })
    
    if num_added == 0:
        print("Warning: <|CONF|> token was already in the vocabulary")
    else:
        print(f"Added {num_added} new token(s) to tokenizer")
    
    # Resize model embeddings to match new vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings: {original_vocab_size} -> {len(tokenizer)}")
    
    # Get the token ID
    conf_token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
    
    # Initialize the new token embedding using mean initialization
    # This provides a reasonable starting point for training
    # Only do this if we actually added new tokens (avoid NaN from empty slice)
    if num_added > 0:
    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        
        # Compute mean of all existing embeddings (excluding the newly added ones)
        mean_embedding = embedding_layer.weight[:-num_added].mean(dim=0)
        
        # Assign mean embedding to the new token
        embedding_layer.weight[-1] = mean_embedding
        
        print(f"Initialized <|CONF|> embedding with mean of existing embeddings")
    else:
        print("Skipped embedding init (token already existed)")
    
    return conf_token_id


def get_conf_token_id(tokenizer: PreTrainedTokenizer) -> int:
    """
    Get the token ID for <|CONF|>.
    
    Args:
        tokenizer: The tokenizer containing the <|CONF|> token
        
    Returns:
        The token ID of <|CONF|>
        
    Raises:
        ValueError: If <|CONF|> is not in the tokenizer vocabulary
    """
    token_id = tokenizer.convert_tokens_to_ids("<|CONF|>")
    
    # Check if token exists (unknown tokens typically return a special unknown token ID)
    if tokenizer.convert_ids_to_tokens(token_id) != "<|CONF|>":
        raise ValueError("<|CONF|> token not found in tokenizer vocabulary")
    
    return token_id


def verify_conf_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> dict:
    """
    Verify that the <|CONF|> token has been properly added.
    
    Args:
        tokenizer: The tokenizer to verify
        model: The model to verify
        
    Returns:
        Dictionary containing verification results
    """
    results = {
        "token_exists": False,
        "token_id": None,
        "vocab_size": len(tokenizer),
        "embedding_size": model.get_input_embeddings().weight.shape[0],
        "sizes_match": False,
    }
    
    try:
        token_id = get_conf_token_id(tokenizer)
        results["token_exists"] = True
        results["token_id"] = token_id
    except ValueError:
        results["token_exists"] = False
    
    results["sizes_match"] = results["vocab_size"] == results["embedding_size"]
    
    return results

