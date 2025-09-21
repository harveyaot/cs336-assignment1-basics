#!/usr/bin/env python3
"""
Training script for a small Transformer language model using all implemented components.

This script demonstrates the complete training pipeline using:
- BPE tokenizer training
- Data loading with get_batch
- TransformerLM model
- AdamW optimizer
- Learning rate scheduling
- Checkpointing
- Training loop with gradient clipping
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from cs336_basics.simple_bpe import BPETokenizerParams, BPETokenizer
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.simple_data import get_batch
from cs336_basics.simple_model import TransformerLM


def create_gpt2_tokenizer(vocab_path: str, merges_path: str) -> BPETokenizer:
    """Create a BPE tokenizer using GPT-2 vocabulary and merges."""
    import json

    print(f"Loading GPT-2 vocabulary from {vocab_path}")

    # Get the GPT-2 byte decoder (inverse of gpt2_bytes_to_unicode)
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)

    # Convert the vocab format: gpt2_token_string -> token_id to token_id -> bytes
    vocab = {}
    for gpt2_token_str, token_id in vocab_dict.items():
        # Decode the GPT-2 printable representation back to actual bytes
        token_bytes = bytes([gpt2_byte_decoder[char] for char in gpt2_token_str])
        vocab[token_id] = token_bytes

    print(f"Loaded vocabulary with {len(vocab)} tokens")

    print(f"Loading GPT-2 merges from {merges_path}")
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip the version line
    merge_lines = lines[1:]

    for line in merge_lines:
        line = line.strip()
        if not line:
            continue

        # Parse "token1 token2" format
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        token1_str, token2_str = parts

        # Decode the GPT-2 printable representations back to actual bytes
        token1_bytes = bytes([gpt2_byte_decoder[char] for char in token1_str])
        token2_bytes = bytes([gpt2_byte_decoder[char] for char in token2_str])
        merges.append((token1_bytes, token2_bytes))

    print(f"Loaded {len(merges)} merge rules")

    # Create the BPE tokenizer
    special_tokens = ["<|endoftext|>"]
    params = BPETokenizerParams(vocab, merges, special_tokens)
    tokenizer = BPETokenizer(params)
    return tokenizer


from cs336_basics.simple_train import (
    AdamW,
    cross_entropy,
    gradient_clipping,
    get_lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)


def setup_device():
    """Setup the device for training (CPU, CUDA, or MPS)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")

    return device


def load_and_tokenize_data(
    data_path: str, tokenizer: BPETokenizer = None, max_tokens: Optional[int] = None
):
    """
    Load pre-tokenized data from a numpy file using memory mapping.

    This function loads data that has been pre-tokenized and saved as uint16
    numpy arrays. It uses memory mapping to avoid loading the entire dataset
    into memory at once, which is essential for large datasets.

    Args:
        data_path: Path to the pre-tokenized .npy file
        tokenizer: BPE tokenizer instance (not used, kept for compatibility)
        max_tokens: Maximum number of tokens to load (for memory constraints)

    Returns:
        memory-mapped numpy array of token IDs
    """
    print(f"Loading pre-tokenized data from {data_path}...")

    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Pre-tokenized data file not found: {data_path}")

    # Load the data using memory mapping
    # mmap_mode='r' ensures the array is memory-mapped and read-only
    tokens = np.load(data_path, mmap_mode="r")

    print(f"Loaded {len(tokens):,} tokens from memory-mapped file")
    print(f"Data type: {tokens.dtype}")
    print(f"File size: {os.path.getsize(data_path):,} bytes")

    # Convert to int64 for consistency with training code, but keep memory-mapped
    # Note: This creates a view, not a copy, so it's still memory-efficient
    return tokens.astype(np.int64)


def load_dataset_splits(
    train_data_path: str, val_data_path: str, max_tokens: Optional[int] = None
):
    """
    Load both training and validation datasets using memory mapping.

    Args:
        train_data_path: Path to the pre-tokenized training data .npy file
        val_data_path: Path to the pre-tokenized validation data .npy file
        max_tokens: Maximum number of tokens to load per split (for memory constraints)

    Returns:
        tuple of (train_tokens, val_tokens) as memory-mapped arrays
    """
    print("Loading dataset splits...")

    train_tokens = load_and_tokenize_data(train_data_path, max_tokens=max_tokens)
    val_tokens = load_and_tokenize_data(val_data_path, max_tokens=max_tokens)

    print(f"Training set: {len(train_tokens):,} tokens")
    print(f"Validation set: {len(val_tokens):,} tokens")

    return train_tokens, val_tokens


def create_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    device: str,
) -> TransformerLM:
    """
    Create a Transformer language model.

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        device: Device to place the model on

    Returns:
        TransformerLM model
    """
    print(
        f"Creating model with {num_layers} layers, {num_heads} heads, d_model={d_model}"
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device,
    )

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def train_model(
    model: TransformerLM,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    tokenizer: BPETokenizer,
    device: str,
    args,
):
    """
    Train the language model.

    Args:
        model: The TransformerLM model to train
        train_tokens: Training token IDs
        val_tokens: Validation token IDs
        tokenizer: BPE tokenizer
        device: Training device
        args: Training arguments
    """
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # Training parameters
    batch_size = args.batch_size
    context_length = args.context_length
    max_iters = args.max_iters
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    gradient_clip_val = args.gradient_clip

    # Learning rate schedule parameters
    warmup_iters = args.warmup_iters
    cosine_cycle_iters = max_iters

    # Training state
    iteration = 0
    best_val_loss = float("inf")

    # Load checkpoint if resuming
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        iteration = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {iteration}")

    # Training loop
    model.train()
    start_time = time.time()

    for iter_num in range(iteration, max_iters):
        # Get batch
        x, y = get_batch(train_tokens, batch_size, context_length, device)

        # Forward pass
        logits = model(x)

        # Compute loss
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip_val > 0:
            gradient_clipping(model.parameters(), gradient_clip_val)

        # Optimizer step
        optimizer.step()

        # Learning rate scheduling
        lr = get_lr_cosine_schedule(
            iter_num, args.learning_rate, args.min_lr, warmup_iters, cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Logging
        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"iter {iter_num:4d} | loss {loss.item():.4f} | lr {lr:.6f} | time {elapsed:.2f}s"
            )

        # Evaluation
        if iter_num % eval_interval == 0:
            val_loss = evaluate_model(
                model, val_tokens, batch_size, context_length, device
            )
            print(f"iter {iter_num:4d} | val loss {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save_best:
                    save_checkpoint(
                        model, optimizer, iter_num, f"{args.output_dir}/best_model.pt"
                    )

        # Regular checkpointing
        if iter_num % save_interval == 0 and iter_num > 0:
            save_checkpoint(
                model,
                optimizer,
                iter_num,
                f"{args.output_dir}/checkpoint_iter_{iter_num}.pt",
            )

    # Save final checkpoint
    save_checkpoint(model, optimizer, max_iters, f"{args.output_dir}/final_model.pt")

    print(
        f"\nTraining completed! Final model saved to {args.output_dir}/final_model.pt"
    )


def evaluate_model(
    model: TransformerLM,
    val_tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    """
    Evaluate the model on validation data.

    Args:
        model: The model to evaluate
        val_tokens: Validation token IDs
        batch_size: Batch size for evaluation
        context_length: Context length for evaluation
        device: Device to use

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        # Evaluate on multiple batches
        for _ in range(min(10, len(val_tokens) // (batch_size * context_length))):
            try:
                x, y = get_batch(val_tokens, batch_size, context_length, device)
                logits = model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                num_batches += 1
            except ValueError:
                # Skip if not enough tokens for a full batch
                break

    model.train()
    return total_loss / num_batches if num_batches > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Train a small Transformer language model"
    )

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="tokenized_data/TinyStoriesV2-GPT4-train_tokenized.npy",
        help="Path to training data file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="tokenized_data/TinyStoriesV2-GPT4-valid_tokenized.npy",
        help="Path to validation data file",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000000,
        help="Maximum tokens to load (for memory constraints)",
    )

    # Tokenizer arguments
    parser.add_argument(
        "--num_merges",
        type=int,
        default=1000,
        help="Number of BPE merges for tokenizer",
    )
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<pad>", "<unk>"],
        help="Special tokens to add to vocabulary",
    )

    # Model arguments
    parser.add_argument(
        "--context_length", type=int, default=128, help="Context length for training"
    )
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Maximum training iterations"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=100, help="Warmup iterations"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0, help="Gradient clipping value"
    )

    # Logging and saving
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument(
        "--eval_interval", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument(
        "--save_interval", type=int, default=500, help="Checkpoint saving interval"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save_best",
        action="store_true",
        help="Save best model based on validation loss",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = setup_device()

    # Create GPT-2 tokenizer
    print("Creating GPT-2 tokenizer...")
    tokenizer = create_gpt2_tokenizer(
        "tests/fixtures/gpt2_vocab.json", "tests/fixtures/gpt2_merges.txt"
    )

    # Load pre-tokenized data
    print("Loading pre-tokenized data...")
    train_tokens = load_and_tokenize_data(
        "tokenized_data/TinyStoriesV2-GPT4-train_tokenized.npy",
        tokenizer,
        args.max_tokens,
    )
    val_tokens = load_and_tokenize_data(
        "tokenized_data/TinyStoriesV2-GPT4-valid_tokenized.npy",
        tokenizer,
        args.max_tokens,
    )

    # Create model
    model = create_model(
        vocab_size=len(tokenizer.params.vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
    )

    # Train model
    train_model(model, train_tokens, val_tokens, tokenizer, device, args)

    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()
