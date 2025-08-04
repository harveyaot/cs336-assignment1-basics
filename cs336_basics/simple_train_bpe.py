from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set
import regex
from cs336_basics.simple_bpe import merge, BPETokenizerParams, GPT2_TOKENIZER_REGEX


def train_bpe(string: str, num_merges: int, special_tokens: list[str] = None) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    # Initialize vocabulary and merges
    vocab: dict[int, bytes] = {}  # index -> bytes
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    
    # Initialize vocabulary with byte tokens first (like the optimized version)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # Add special tokens after byte tokens (like the optimized version)
    special_tokens_dict = {}
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            vocab[next_id] = byte_encoded_special_token
            special_tokens_dict[special_token] = next_id
            next_id += 1
    
    # Pre-tokenize the text using the GPT-2 regex pattern
    pre_tokens = regex.findall(GPT2_TOKENIZER_REGEX, string)
    
    # Convert each pre-token to a sequence of byte indices
    pre_token_byte_sequences = []
    for pre_token in pre_tokens:
        # Convert pre-token to UTF-8 bytes, then to byte indices
        pre_token_bytes = pre_token.encode("utf-8")
        byte_indices = []
        for byte in pre_token_bytes:
            # Find the vocabulary index that corresponds to this byte
            for idx, vocab_bytes in vocab.items():
                if vocab_bytes == bytes([byte]):
                    byte_indices.append(idx)
                    break
            else:
                # If byte not found in vocabulary, use the byte value directly
                byte_indices.append(byte)
        pre_token_byte_sequences.append(byte_indices)
    
    # Train BPE merges
    for i in range(num_merges):
        # Count all pairs within each pre-token (not across boundaries)
        pair_counts = defaultdict(int)
        for byte_sequence in pre_token_byte_sequences:
            for j in range(len(byte_sequence) - 1):
                pair = (byte_sequence[j], byte_sequence[j + 1])
                pair_counts[pair] += 1
        
        if not pair_counts:
            break
        
        # Find the most frequent pair, breaking ties by preferring lexicographically greater pair
        max_count = max(pair_counts.values())
        max_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
        
        # For tie-breaking, convert bytes to GPT-2 unicode for comparison
        from tests.common import gpt2_bytes_to_unicode
        gpt2_encoder = gpt2_bytes_to_unicode()
        
        def pair_to_bytes(pair):
            # Convert each token in the pair to its byte representation
            token1_bytes = vocab[pair[0]]
            token2_bytes = vocab[pair[1]]
            
            # Return the byte sequences for comparison
            return (token1_bytes, token2_bytes)
        
        # Print detailed information when there are multiple pairs with the same max count
        if len(max_pairs) > 1:
            print(f"\n--- Iteration {i} ---")
            print(f"Max count: {max_count}")
            print(f"Pairs with max count ({len(max_pairs)} pairs):")
            
            # Sort by lexicographic order for clarity
            max_pairs_sorted = sorted(max_pairs, key=lambda x: x[0])
            
            for pair_item in max_pairs_sorted:
                # Convert to printable format using GPT-2 unicode
                token1_bytes = vocab[pair_item[0]]
                token2_bytes = vocab[pair_item[1]]
                p1 = ''.join([gpt2_encoder[b] for b in token1_bytes])
                p2 = ''.join([gpt2_encoder[b] for b in token2_bytes])
                print(f"  {pair_item}: '{p1}' '{p2}' (count: {max_count})")
        
        # Sort by byte representation and choose the lexicographically greater
        max_pairs.sort(key=pair_to_bytes)
        pair = max_pairs[-1]  # Lexicographically greater pair for tie-breaking
        
        # Print the selected pair when there were ties
        if len(max_pairs) > 1:
            token1_bytes = vocab[pair[0]]
            token2_bytes = vocab[pair[1]]
            p1 = ''.join([gpt2_encoder[b] for b in token1_bytes])
            p2 = ''.join([gpt2_encoder[b] for b in token2_bytes])
            print(f"Selected pair: {pair} ('{p1}' '{p2}')")
        index1, index2 = pair
        
        # Create new merged token
        new_index = len(vocab)
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        
        # Apply the merge to all pre-token byte sequences
        for j, byte_sequence in enumerate(pre_token_byte_sequences):
            pre_token_byte_sequences[j] = merge(byte_sequence, pair, new_index)
    
    return BPETokenizerParams(vocab=vocab, merges=merges, special_tokens=special_tokens_dict if special_tokens_dict else None)