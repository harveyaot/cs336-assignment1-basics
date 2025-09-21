#!/usr/bin/env python3
"""
Data preprocessing script to tokenize train and validation data using GPT-2 tokenizer.

This script:
1. Loads the GPT-2 vocabulary and merges from the test fixtures
2. Creates a BPE tokenizer using these pretrained components
3. Tokenizes the train and validation data files
4. Converts the tokenized data to uint16 numpy arrays
5. Saves the arrays to disk for memory-mapped loading during training
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from tqdm import tqdm

from cs336_basics.bpe_tokenizer import BPETokenizer
from tests.common import gpt2_bytes_to_unicode


def create_gpt2_tokenizer(vocab_path: str, merges_path: str) -> BPETokenizer:
    """Create a BPE tokenizer using GPT-2 vocabulary and merges."""

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
    tokenizer = BPETokenizer(vocab, merges)
    return tokenizer


def _tokenize_chunk(chunk_and_tokenizer: Tuple[str, BPETokenizer]) -> List[int]:
    """
    Helper function to tokenize a single chunk of text.
    This function is designed to be used with multiprocessing.

    Args:
        chunk_and_tokenizer: Tuple containing (text_chunk, tokenizer)

    Returns:
        List of token IDs for the chunk
    """
    chunk, tokenizer = chunk_and_tokenizer
    return tokenizer.encode(chunk)


def tokenize_file(
    file_path: str,
    tokenizer: BPETokenizer,
    chunk_size: int = 1024 * 1024,
    num_processes: int = None,
) -> List[int]:
    """
    Tokenize a large text file in chunks using multiprocessing to avoid memory issues.

    Args:
        file_path: Path to the text file
        tokenizer: BPE tokenizer
        chunk_size: Size of text chunks to process at once (in characters)
        num_processes: Number of processes to use (defaults to CPU count)

    Returns:
        List of token IDs
    """
    if num_processes is None:
        num_processes = cpu_count()

    print(f"Tokenizing {file_path} using {num_processes} processes")

    # Read file in chunks and prepare for multiprocessing
    chunks = []
    file_size = os.path.getsize(file_path)

    print("Reading file into chunks...")
    with open(file_path, "r", encoding="utf-8") as f:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Reading chunks"
        ) as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                pbar.update(len(chunk.encode("utf-8")))

    print(f"Split into {len(chunks)} chunks, processing with multiprocessing...")

    # Prepare arguments for multiprocessing
    chunk_tokenizer_pairs = [(chunk, tokenizer) for chunk in chunks]

    # Process chunks in parallel
    all_tokens = []
    with Pool(processes=num_processes) as pool:
        # Use imap for better memory efficiency and progress tracking
        with tqdm(total=len(chunks), desc="Tokenizing chunks") as pbar:
            for tokens in pool.imap(_tokenize_chunk, chunk_tokenizer_pairs):
                all_tokens.extend(tokens)
                pbar.update(1)

    print(f"Tokenized into {len(all_tokens):,} tokens")
    return all_tokens


def save_tokens_as_uint16(tokens: List[int], output_path: str):
    """
    Convert tokens to uint16 and save as numpy array.

    Args:
        tokens: List of token IDs
        output_path: Path to save the numpy array
    """
    print(f"Converting {len(tokens):,} tokens to uint16")

    # Check if any tokens exceed uint16 range
    max_token = max(tokens)
    if max_token >= 65536:
        print(f"Warning: Maximum token ID {max_token} exceeds uint16 range (65535)")
        print("Clipping to uint16 range...")
        tokens = [min(token, 65535) for token in tokens]

    # Convert to numpy array
    tokens_array = np.array(tokens, dtype=np.uint16)

    # Save the array
    print(f"Saving tokenized data to {output_path}")
    np.save(output_path, tokens_array)

    # Verify the saved file
    file_size = os.path.getsize(output_path)
    print(f"Saved {file_size:,} bytes ({file_size / (1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data using GPT-2 tokenizer"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing train and validation data files",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="tests/fixtures/gpt2_vocab.json",
        help="Path to GPT-2 vocabulary JSON file",
    )
    parser.add_argument(
        "--merges-path",
        type=str,
        default="tests/fixtures/gpt2_merges.txt",
        help="Path to GPT-2 merges text file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenized_data",
        help="Directory to save tokenized data",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Chunk size for processing large files (in characters)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to use for tokenization (defaults to CPU count)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create GPT-2 tokenizer
    print("Creating GPT-2 tokenizer...")
    tokenizer = create_gpt2_tokenizer(args.vocab_path, args.merges_path)

    # Find data files
    data_dir = Path(args.data_dir)
    data_files = {
        "train": [
            data_dir / "TinyStoriesV2-GPT4-train.txt",
        ],
        "valid": [
            data_dir / "TinyStoriesV2-GPT4-valid.txt",
        ],
    }

    # Process each dataset
    for split_name, file_paths in data_files.items():
        for file_path in file_paths:
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist, skipping...")
                continue

            print(f"\n=== Processing {split_name} dataset: {file_path.name} ===")

            # Tokenize the file
            tokens = tokenize_file(
                str(file_path), tokenizer, args.chunk_size, args.num_processes
            )

            # Create output filename
            output_name = f"{file_path.stem}_tokenized.npy"
            output_path = output_dir / output_name

            # Save as uint16 numpy array
            save_tokens_as_uint16(tokens, str(output_path))

    print(f"\n=== Preprocessing complete! ===")
    print(f"Tokenized data saved to: {output_dir}")
    print("\nFiles created:")
    for file_path in output_dir.glob("*.npy"):
        file_size = file_path.stat().st_size
        print(
            f"  {file_path.name}: {file_size:,} bytes ({file_size / (1024**2):.1f} MB)"
        )


if __name__ == "__main__":
    main()
