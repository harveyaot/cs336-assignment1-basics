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

from cs336_basics.simple_train_bpe import train_bpe
from cs336_basics.simple_bpe import BPETokenizerParams
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.simple_data import get_batch
from cs336_basics.simple_model import TransformerLM
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
    data_path: str, tokenizer: BPETokenizer, max_tokens: Optional[int] = None
):
    """
    Load and tokenize data from a text file.

    Args:
        data_path: Path to the text file
        tokenizer: BPE tokenizer instance
        max_tokens: Maximum number of tokens to load (for memory constraints)

    Returns:
        numpy array of token IDs
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text length: {len(text):,} characters")

    # Tokenize the text
    tokens = tokenizer.encode(text)
    print(f"Tokenized into {len(tokens):,} tokens")

    # Limit tokens if specified
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"Limited to {len(tokens):,} tokens")

    return np.array(tokens, dtype=np.int64)


def train_tokenizer(
    text_path: str, num_merges: int, special_tokens: list[str] = None
) -> BPETokenizer:
    """
    Train a BPE tokenizer on the given text.

    Args:
        text_path: Path to the training text file
        num_merges: Number of BPE merges to perform
        special_tokens: List of special tokens to add

    Returns:
        Trained BPE tokenizer
    """
    print(f"Training BPE tokenizer with {num_merges} merges...")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Train BPE
    bpe_params = train_bpe(text, num_merges, special_tokens)

    # Convert BPETokenizerParams to the format expected by BPETokenizer constructor
    vocab = bpe_params.vocab
    merges = []
    for (id1, id2), _ in bpe_params.merges.items():
        bytes1 = vocab.get(id1, b"")
        bytes2 = vocab.get(id2, b"")
        merges.append((bytes1, bytes2))

    special_tokens = (
        list(bpe_params.special_tokens.keys()) if bpe_params.special_tokens else None
    )

    # Create tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    return tokenizer


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
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to training data file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
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

    # Train or load tokenizer
    tokenizer_path = f"{args.output_dir}/tokenizer.pt"
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = torch.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer = train_tokenizer(
            args.train_data, args.num_merges, args.special_tokens
        )
        torch.save(tokenizer, tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    # Load data
    train_tokens = load_and_tokenize_data(args.train_data, tokenizer, args.max_tokens)
    val_tokens = load_and_tokenize_data(args.val_data, tokenizer, args.max_tokens)

    # Create model
    model = create_model(
        vocab_size=len(tokenizer.vocab),
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
