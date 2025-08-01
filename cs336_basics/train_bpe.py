import json
import pickle
import regex
import heapq
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Iterator
from pathlib import Path


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: Dict[int, bytes]  # index -> bytes
    merges: Dict[Tuple[int, int], int]  # index1,index2 -> new_index
    special_tokens: Dict[str, int] = None  # special_token_string -> index
    vocab_size: int = None
    pretokenization_pattern: str = None

    def __post_init__(self):
        if self.special_tokens is None:
            object.__setattr__(self, "special_tokens", {})


# Canonical BPE training loop, matches tiktoken logic.
def train_bpe(
    texts: Union[str, List[str]],
    vocab_size: int = 50000,
    special_tokens: Optional[List[str]] = None,
    pretokenization_pattern: Optional[str] = None,
    min_frequency: int = 2,
) -> BPETokenizerParams:
    """Simple BPE training function for backward compatibility."""
    if isinstance(texts, str):
        texts = [texts]
    
    return train_bpe_optimized(
        texts=texts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        pretokenization_pattern=pretokenization_pattern,
        min_frequency=min_frequency,
    )


def train_bpe_optimized(
    texts: Union[str, List[str]],
    vocab_size: int = 50000,
    special_tokens: Optional[List[str]] = None,
    pretokenization_pattern: Optional[str] = None,
    min_frequency: int = 2,
    progress_callback: Optional[callable] = None,
) -> BPETokenizerParams:
    """
    Highly optimized BPE training with advanced data structures.

    Optimizations:
    1. Max-heap for O(log n) pair selection instead of O(n) scanning
    2. Pair-to-words index to avoid full word_freqs rebuilding
    3. Incremental updates only for affected words
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        raise ValueError("Training texts cannot be empty")

    # Initialize vocabulary with byte tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # Add special tokens first
    special_token_map = {}
    if special_tokens:
        for token in special_tokens:
            special_token_map[token] = next_id
            next_id += 1

    # Pre-tokenize all texts
    pretokenization_regex = None
    if pretokenization_pattern:
        pretokenization_regex = regex.compile(pretokenization_pattern)

    all_tokens = []
    for text in texts:
        if pretokenization_regex:
            tokens = pretokenization_regex.findall(text)
        else:
            tokens = [text]
        all_tokens.extend(tokens)

    # Convert all tokens to byte sequences
    word_freqs = Counter()
    for token in all_tokens:
        if token.strip():  # Skip empty tokens
            byte_sequence = list(token.encode("utf-8"))
            word_freqs[tuple(byte_sequence)] += 1

    # Calculate number of merges needed
    current_vocab_size = len(vocab) + len(special_token_map)
    num_merges = vocab_size - current_vocab_size

    if num_merges <= 0:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be larger than base vocab size ({current_vocab_size})"
        )

    merges: Dict[Tuple[int, int], int] = {}

    # OPTIMIZATION 1: Use max-heap for efficient pair selection
    # Note: heapq is min-heap, so we use negative counts for max-heap behavior
    pair_heap = []
    pair_counts = defaultdict(int)

    # OPTIMIZATION 2: Maintain index of which words contain which pairs
    pair_to_words = defaultdict(set)  # pair -> set of word_ids containing this pair
    word_id_to_word = {}  # word_id -> word tuple
    word_to_id = {}  # word tuple -> word_id

    def _get_pairs_from_word(word: List[int]) -> List[Tuple[int, int]]:
        """Get all adjacent pairs in a word."""
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    def _initialize_data_structures():
        """Initialize the heap and indices."""
        nonlocal pair_heap, pair_counts, pair_to_words, word_id_to_word, word_to_id

        # Assign IDs to words
        word_id = 0
        for word_tuple, freq in word_freqs.items():
            word_id_to_word[word_id] = word_tuple
            word_to_id[word_tuple] = word_id
            word_id += 1

        # Count pairs and build indices
        for word_tuple, freq in word_freqs.items():
            word_id = word_to_id[word_tuple]
            word = list(word_tuple)
            pairs = _get_pairs_from_word(word)

            for pair in pairs:
                pair_counts[pair] += freq
                pair_to_words[pair].add(word_id)

        # Build initial heap
        for pair, count in pair_counts.items():
            if count >= min_frequency:
                heapq.heappush(pair_heap, (-count, pair))  # Negative for max-heap

    def _get_best_pair():
        """Get the pair with highest frequency using heap."""
        while pair_heap:
            neg_count, pair = heapq.heappop(pair_heap)
            current_count = pair_counts.get(pair, 0)

            # Check if this entry is still valid
            if current_count == -neg_count and current_count >= min_frequency:
                return pair, current_count
            # If not valid, the correct count might still be in heap, continue

        return None, 0

    def _update_affected_words(merged_pair: Tuple[int, int], new_token_id: int):
        """Update only the words that contain the merged pair."""
        affected_word_ids = pair_to_words[merged_pair].copy()

        # Track pair count changes
        pair_count_deltas = defaultdict(int)

        for word_id in affected_word_ids:
            old_word_tuple = word_id_to_word[word_id]
            old_word = list(old_word_tuple)
            freq = word_freqs[old_word_tuple]

            # Remove old word
            del word_freqs[old_word_tuple]
            del word_to_id[old_word_tuple]

            # Get old pairs and update counts
            old_pairs = _get_pairs_from_word(old_word)
            for pair in old_pairs:
                pair_count_deltas[pair] -= freq
                pair_to_words[pair].discard(word_id)

            # Apply merge to create new word
            new_word = []
            i = 0
            while i < len(old_word):
                if (
                    i < len(old_word) - 1
                    and old_word[i] == merged_pair[0]
                    and old_word[i + 1] == merged_pair[1]
                ):
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1

            new_word_tuple = tuple(new_word)

            # Add new word
            word_freqs[new_word_tuple] += freq
            word_to_id[new_word_tuple] = word_id
            word_id_to_word[word_id] = new_word_tuple

            # Get new pairs and update counts
            new_pairs = _get_pairs_from_word(new_word)
            for pair in new_pairs:
                pair_count_deltas[pair] += freq
                pair_to_words[pair].add(word_id)

        # Apply count deltas and update heap
        for pair, delta in pair_count_deltas.items():
            old_count = pair_counts[pair]
            new_count = old_count + delta
            pair_counts[pair] = new_count

            # Add new count to heap if it meets threshold
            if new_count >= min_frequency:
                heapq.heappush(pair_heap, (-new_count, pair))

            # Clean up zero/negative counts
            if new_count <= 0:
                del pair_counts[pair]
                if pair in pair_to_words:
                    del pair_to_words[pair]

    # Initialize data structures
    _initialize_data_structures()

    # Training loop - HIGHLY OPTIMIZED VERSION
    for i in range(num_merges):
        if progress_callback:
            progress_callback(i, num_merges)

        # Get best pair using heap
        best_pair, best_count = _get_best_pair()

        if best_pair is None:
            break

        # Add merge
        new_token_id = next_id
        merges[best_pair] = new_token_id
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        next_id += 1

        # Update only affected words
        _update_affected_words(best_pair, new_token_id)

    if progress_callback:
        progress_callback(num_merges, num_merges)

    return BPETokenizerParams(
        vocab=vocab,
        merges=merges,
        special_tokens=special_token_map,
        vocab_size=len(vocab) + len(special_token_map),
        pretokenization_pattern=pretokenization_pattern,
    )


# GPT-2 style pre-tokenization pattern with Unicode support (requires regex library)
GPT_PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def create_gpt_style_tokenizer(
    texts: Union[str, List[str]],
    vocab_size: int = 50257,
    special_tokens: Optional[List[str]] = None,
) -> "BPETokenizer":
    """Create a GPT-style tokenizer with standard settings."""
    from .bpe_tokenizer import BPETokenizer
    
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    params = train_bpe_optimized(
        texts=texts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        pretokenization_pattern=GPT_PRETOKENIZATION_PATTERN,
        min_frequency=2,
    )

    return BPETokenizer.from_params(params) 