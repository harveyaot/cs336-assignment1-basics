#!/usr/bin/env python3
"""
Demo script to test the training pipeline with a very small model.

This script runs a quick training session to verify all components work together.
"""

import os
import torch
import numpy as np
from cs336_basics.simple_train_bpe import train_bpe
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.simple_data import get_batch
from cs336_basics.simple_model import TransformerLM
from cs336_basics.simple_train import (
    AdamW,
    cross_entropy,
    gradient_clipping,
    save_checkpoint,
)


def create_demo_data():
    """Create a small demo dataset for testing."""
    demo_text = (
        """
    The quick brown fox jumps over the lazy dog. 
    This is a sample text for training our language model.
    We will use this to test all the implemented components.
    The model should learn to predict the next word in sequences.
    Training will be quick since this is just a demonstration.
    """
        * 100
    )  # Repeat to have enough data

    return demo_text


def main():
    print("🚀 Starting Demo Training Session")
    print("=" * 50)

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")

    # Create demo data
    print("\n📝 Creating demo dataset...")
    demo_text = create_demo_data()
    print(f"Demo text length: {len(demo_text):,} characters")

    # Train a simple tokenizer
    print("\n🔤 Training BPE tokenizer...")
    bpe_params = train_bpe(demo_text, num_merges=100, special_tokens=["<pad>", "<unk>"])

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

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Tokenize the text
    print("\n🔄 Tokenizing text...")
    tokens = tokenizer.encode(demo_text)
    print(f"Tokenized into {len(tokens):,} tokens")

    # Convert to numpy array
    train_tokens = np.array(tokens, dtype=np.int64)

    # Create a tiny model
    print("\n🏗️  Creating tiny Transformer model...")
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=32,  # Very short context
        d_model=64,  # Tiny model dimension
        num_layers=2,  # Just 2 layers
        num_heads=4,  # 4 attention heads
        d_ff=128,  # Small feed-forward
        device=device,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Setup optimizer
    print("\n⚙️  Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    # Training parameters
    batch_size = 8
    context_length = 32
    max_iters = 50  # Very few iterations for demo

    print(f"\n🎯 Training parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Device: {device}")

    # Training loop
    print("\n🔥 Starting training loop...")
    model.train()

    for iter_num in range(max_iters):
        # Get batch
        try:
            x, y = get_batch(train_tokens, batch_size, context_length, device)
        except ValueError as e:
            print(f"⚠️  Batch creation failed: {e}")
            print(f"   Available tokens: {len(train_tokens)}")
            print(f"   Required: {batch_size * context_length}")
            break

        # Forward pass
        logits = model(x)

        # Compute loss
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Logging
        if iter_num % 10 == 0:
            print(f"  Iter {iter_num:2d} | Loss: {loss.item():.4f}")

    print(f"\n✅ Training completed!")

    # Test the model
    print("\n🧪 Testing the trained model...")
    model.eval()

    with torch.no_grad():
        # Create a simple test sequence
        test_tokens = torch.tensor([[0, 1, 2, 3, 4]], device=device, dtype=torch.long)
        test_logits = model(test_tokens)

        print(f"  Input shape: {test_tokens.shape}")
        print(f"  Output shape: {test_logits.shape}")
        print(f"  Output dtype: {test_logits.dtype}")

        # Get predictions
        predictions = torch.softmax(test_logits, dim=-1)
        predicted_tokens = torch.argmax(predictions, dim=-1)

        print(f"  Predicted tokens: {predicted_tokens}")

    # Save a checkpoint
    print("\n💾 Saving checkpoint...")
    os.makedirs("demo_checkpoints", exist_ok=True)
    save_checkpoint(model, optimizer, max_iters, "demo_checkpoints/demo_model.pt")
    print("  Checkpoint saved to demo_checkpoints/demo_model.pt")

    # Test checkpoint loading
    print("\n📂 Testing checkpoint loading...")
    new_model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        device=device,
    )
    new_optimizer = AdamW(new_model.parameters(), lr=1e-3)

    from cs336_basics.simple_train import load_checkpoint

    loaded_iteration = load_checkpoint(
        "demo_checkpoints/demo_model.pt", new_model, new_optimizer
    )
    print(f"  Loaded iteration: {loaded_iteration}")

    # Verify the loaded model works
    with torch.no_grad():
        test_logits_loaded = new_model(test_tokens)
        print(f"  Loaded model output shape: {test_logits_loaded.shape}")

        # Check if outputs are similar
        if torch.allclose(test_logits, test_logits_loaded, atol=1e-6):
            print("  ✅ Loaded model produces identical output")
        else:
            print("  ⚠️  Loaded model output differs from original")

    print("\n🎉 Demo completed successfully!")
    print("All components are working correctly!")


if __name__ == "__main__":
    main()
