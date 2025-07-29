import json
import pickle
import regex
from abc import ABC
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def encode(self, string: str) -> List[int]:
        raise NotImplementedError

    def decode(self, indices: List[int]) -> str:
        raise NotImplementedError


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


class BPETokenizer(Tokenizer):
    """Production-ready BPE tokenizer with performance optimizations and special tokens."""

    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self._validate_params()

        # Create reverse mappings for efficiency
        self._token_to_id = {
            token: idx for token, idx in self.params.special_tokens.items()
        }
        self._id_to_token = {
            idx: token for token, idx in self.params.special_tokens.items()
        }

        # Pre-compile regex if provided
        self._pretokenization_regex = None
        if self.params.pretokenization_pattern:
            self._pretokenization_regex = regex.compile(
                self.params.pretokenization_pattern
            )

        # Create efficient merge lookup
        self._merge_ranks = {
            pair: rank for rank, (pair, _) in enumerate(self.params.merges.items())
        }

    def _validate_params(self):
        """Validate tokenizer parameters."""
        if not self.params.vocab:
            raise ValueError("Vocabulary cannot be empty")

        if not all(
            isinstance(k, int) and isinstance(v, bytes)
            for k, v in self.params.vocab.items()
        ):
            raise ValueError("Vocabulary must map int -> bytes")

        if self.params.merges and not all(
            isinstance(k, tuple) and len(k) == 2 and isinstance(v, int)
            for k, v in self.params.merges.items()
        ):
            raise ValueError("Merges must map (int, int) -> int")

    def encode(
        self, text: str, allowed_special: Union[str, Set[str]] = "none_raise"
    ) -> List[int]:
        """
        Encode text to token IDs with special token handling.

        Args:
            text: Input text to encode
            allowed_special: How to handle special tokens:
                - "none_raise": Raise error if special tokens found
                - "all": Allow all special tokens
                - set of strings: Allow only these special tokens
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Handle special tokens
        special_tokens = set()
        if allowed_special == "all":
            special_tokens = set(self.params.special_tokens.keys())
        elif isinstance(allowed_special, set):
            special_tokens = allowed_special
        elif allowed_special != "none_raise":
            raise ValueError(
                "allowed_special must be 'none_raise', 'all', or a set of strings"
            )

        # Split on special tokens first
        if special_tokens:
            text_parts = self._split_on_special_tokens(text, special_tokens)
        else:
            # Check for disallowed special tokens
            for special in self.params.special_tokens:
                if special in text:
                    raise ValueError(
                        f"Special token '{special}' found in text but not allowed"
                    )
            text_parts = [text]

        # Encode each part
        all_tokens = []
        for part in text_parts:
            if part in self.params.special_tokens:
                all_tokens.append(self.params.special_tokens[part])
            else:
                all_tokens.extend(self._encode_ordinary(part))

        return all_tokens

    def _split_on_special_tokens(
        self, text: str, special_tokens: Set[str]
    ) -> List[str]:
        """Split text on special tokens, keeping the special tokens."""
        if not special_tokens:
            return [text]

        # Sort special tokens by length (longest first) to handle overlaps
        sorted_specials = sorted(special_tokens, key=len, reverse=True)

        parts = [text]
        for special in sorted_specials:
            new_parts = []
            for part in parts:
                if part not in special_tokens:  # Don't split special tokens themselves
                    subparts = part.split(special)
                    for i, subpart in enumerate(subparts):
                        if i > 0:
                            new_parts.append(special)
                        if subpart:  # Don't add empty strings
                            new_parts.append(subpart)
                else:
                    new_parts.append(part)
            parts = new_parts

        return [p for p in parts if p]  # Remove empty strings

    def _encode_ordinary(self, text: str) -> List[int]:
        """Encode ordinary text (no special tokens)."""
        if not text:
            return []

        # Pre-tokenization
        if self._pretokenization_regex:
            tokens = self._pretokenization_regex.findall(text)
        else:
            tokens = [text]

        # Encode each token with BPE
        all_ids = []
        for token in tokens:
            if not token:
                continue
            token_ids = self._encode_token(token)
            all_ids.extend(token_ids)

        return all_ids

    def _encode_token(self, token: str) -> List[int]:
        """Apply BPE encoding to a single token."""
        if not token:
            return []

        # Convert to bytes and then to individual byte tokens
        token_bytes = token.encode("utf-8")
        ids = list(token_bytes)

        # Apply merges efficiently
        return self._apply_merges(ids)

    def _apply_merges(self, ids: List[int]) -> List[int]:
        """Efficiently apply BPE merges using priority-based approach."""
        if len(ids) <= 1:
            return ids

        # Get all pairs and their merge ranks
        pairs = self._get_pairs(ids)
        if not pairs:
            return ids

        while True:
            # Find the pair with the lowest merge rank (earliest in training)
            best_pair = min(
                pairs, key=lambda pair: self._merge_ranks.get(pair, float("inf"))
            )

            # If no valid merge found, stop
            if best_pair not in self._merge_ranks:
                break

            # Apply the merge
            new_token = self.params.merges[best_pair]
            ids = self._merge_pair(ids, best_pair, new_token)

            if len(ids) <= 1:
                break

            pairs = self._get_pairs(ids)
            if not pairs:
                break

        return ids

    def _get_pairs(self, word: List[int]) -> Set[Tuple[int, int]]:
        """Get all adjacent pairs in the word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def _merge_pair(
        self, word: List[int], pair: Tuple[int, int], new_token: int
    ) -> List[int]:
        """Merge all instances of pair in word with new_token."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not isinstance(ids, list):
            raise ValueError("Input must be a list of integers")

        # Convert IDs to bytes, handling special tokens
        bytes_parts = []
        current_bytes = []

        for token_id in ids:
            if token_id in self._id_to_token:
                # Special token - flush current bytes and add special token
                if current_bytes:
                    try:
                        bytes_parts.append(b"".join(current_bytes).decode("utf-8"))
                        current_bytes = []
                    except UnicodeDecodeError:
                        # If we can't decode, keep as bytes for now
                        pass
                bytes_parts.append(self._id_to_token[token_id])
            elif token_id in self.params.vocab:
                current_bytes.append(self.params.vocab[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        # Flush remaining bytes
        if current_bytes:
            try:
                bytes_parts.append(b"".join(current_bytes).decode("utf-8"))
            except UnicodeDecodeError as e:
                raise ValueError(f"Cannot decode bytes to UTF-8: {e}")

        return "".join(bytes_parts)

    def get_vocab_size(self) -> int:
        """Get the total vocabulary size including special tokens."""
        return len(self.params.vocab) + len(self.params.special_tokens)

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)

        # Convert params to serializable format
        data = {
            "vocab": {str(k): list(v) for k, v in self.params.vocab.items()},
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.params.merges.items()},
            "special_tokens": self.params.special_tokens,
            "vocab_size": self.params.vocab_size,
            "pretokenization_pattern": self.params.pretokenization_pattern,
        }

        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load tokenizer from file."""
        path = Path(path)

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        # Convert back to proper format
        vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        merges = {}
        for k, v in data["merges"].items():
            parts = k.split(",")
            merges[(int(parts[0]), int(parts[1]))] = v

        params = BPETokenizerParams(
            vocab=vocab,
            merges=merges,
            special_tokens=data.get("special_tokens", {}),
            vocab_size=data.get("vocab_size"),
            pretokenization_pattern=data.get("pretokenization_pattern"),
        )

        return cls(params)


def train_bpe(
    texts: Union[str, List[str]],
    vocab_size: int = 50000,
    special_tokens: Optional[List[str]] = None,
    pretokenization_pattern: Optional[str] = None,
    min_frequency: int = 2,
    progress_callback: Optional[callable] = None,
) -> BPETokenizerParams:
    """
    Train a BPE tokenizer with production-ready features and optimized performance.

    Args:
        texts: Training text(s)
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to add
        pretokenization_pattern: Regex pattern for pre-tokenization
        min_frequency: Minimum frequency for a pair to be considered for merging
        progress_callback: Optional callback for training progress
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

    # OPTIMIZATION: Calculate initial pair counts once
    def _get_pairs_from_word(word: List[int]) -> List[Tuple[int, int]]:
        """Get all adjacent pairs in a word."""
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    def _calculate_all_pair_counts(word_freqs: Counter) -> defaultdict:
        """Calculate all pair counts from word frequencies."""
        pair_counts = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            word = list(word_tuple)
            for pair in _get_pairs_from_word(word):
                pair_counts[pair] += freq
        return pair_counts

    def _apply_merge_to_word(
        word: List[int], pair: Tuple[int, int], new_token: int
    ) -> List[int]:
        """Apply a single merge to a word, returning the new word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    # Calculate initial pair counts
    pair_counts = _calculate_all_pair_counts(word_freqs)

    # Training loop - OPTIMIZED VERSION
    for i in range(num_merges):
        if progress_callback:
            progress_callback(i, num_merges)

        # Find most frequent pair
        if not pair_counts:
            break

        # Filter by minimum frequency and find the best pair
        valid_pairs = {
            pair: count for pair, count in pair_counts.items() if count >= min_frequency
        }
        if not valid_pairs:
            break

        best_pair = max(valid_pairs.items(), key=lambda x: x[1])[0]

        # Add merge
        new_token_id = next_id
        merges[best_pair] = new_token_id
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        next_id += 1

        # OPTIMIZATION: Incrementally update word frequencies and pair counts
        new_word_freqs = Counter()

        # Track which pairs need count updates
        pairs_to_update = defaultdict(int)  # pair -> delta_count

        for word_tuple, freq in word_freqs.items():
            word = list(word_tuple)

            # Check if this word contains the pair to merge
            contains_merge = False
            for j in range(len(word) - 1):
                if word[j] == best_pair[0] and word[j + 1] == best_pair[1]:
                    contains_merge = True
                    break

            if contains_merge:
                # This word will change - update pair counts
                old_pairs = _get_pairs_from_word(word)
                new_word = _apply_merge_to_word(word, best_pair, new_token_id)
                new_pairs = _get_pairs_from_word(new_word)

                # Calculate pair count deltas
                for pair in old_pairs:
                    pairs_to_update[pair] -= freq
                for pair in new_pairs:
                    pairs_to_update[pair] += freq

                new_word_freqs[tuple(new_word)] += freq
            else:
                # Word unchanged
                new_word_freqs[word_tuple] += freq

        # Apply incremental updates to pair counts
        for pair, delta in pairs_to_update.items():
            pair_counts[pair] += delta
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

        word_freqs = new_word_freqs

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
) -> BPETokenizer:
    """Create a GPT-style tokenizer with standard settings."""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    params = train_bpe(
        texts=texts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        pretokenization_pattern=GPT_PRETOKENIZATION_PATTERN,
        min_frequency=2,
    )

    return BPETokenizer(params)


def bpe_tokenizer():
    """Demo function showing the improved tokenizer capabilities."""
    print("Training improved BPE tokenizer...")

    # Training data
    training_texts = [
        "the cat in the hat",
        "the quick brown fox jumps over the lazy dog",
        "hello world! How are you doing today?",
        "This is a test of the tokenizer.",
    ]

    # Train tokenizer with special tokens
    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]

    def progress_callback(current, total):
        if current % 100 == 0 or current == total:
            print(f"Training progress: {current}/{total}")

    params = train_bpe(
        texts=training_texts,
        vocab_size=300,  # Small for demo
        special_tokens=special_tokens,
        pretokenization_pattern=GPT_PRETOKENIZATION_PATTERN,
        progress_callback=progress_callback,
    )

    tokenizer = BPETokenizer(params)

    print(f"\nTokenizer trained! Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of merges: {len(params.merges)}")

    # Test encoding/decoding
    test_text = "Hello world! <|endoftext|> This is a test."
    print(f"\nOriginal text: '{test_text}'")

    # Test with special tokens allowed
    tokens = tokenizer.encode(test_text, allowed_special="all")
    print(f"Encoded tokens: {tokens}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: '{decoded}'")

    # Test serialization
    print("\nTesting save/load...")
    tokenizer.save("test_tokenizer.json")
    loaded_tokenizer = BPETokenizer.load("test_tokenizer.json")

    # Verify loaded tokenizer works
    loaded_tokens = loaded_tokenizer.encode(test_text, allowed_special="all")
    print(f"Loaded tokenizer tokens: {loaded_tokens}")
    print(f"Tokens match: {tokens == loaded_tokens}")

    # Clean up
    Path("test_tokenizer.json").unlink(missing_ok=True)


def benchmark_training_performance():
    """Benchmark to demonstrate the performance improvement of optimized training."""
    import time

    print("=== BPE Training Performance Benchmark ===\n")

    # Create some sample training data
    training_texts = [
        "the cat in the hat sat on the mat",
        "the quick brown fox jumps over the lazy dog",
        "hello world how are you doing today",
        "this is a test of the tokenizer system",
        "machine learning is a subset of artificial intelligence",
        "natural language processing involves computational linguistics",
        "deep learning uses neural networks with multiple layers",
        "transformers have revolutionized the field of AI",
    ] * 100  # Multiply to make it more substantial

    # Test with a reasonable vocab size
    vocab_size = 1000

    print(f"Training data: {len(training_texts)} texts")
    print(f"Target vocab size: {vocab_size}")
    print(f"Expected merges: ~{vocab_size - 256 - 3}")  # 256 bytes + 3 special tokens

    # Time the optimized version
    start_time = time.time()

    params = train_bpe(
        texts=training_texts,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        pretokenization_pattern=GPT_PRETOKENIZATION_PATTERN,
        min_frequency=2,
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nOptimized training completed in: {training_time:.2f} seconds")
    print(f"Final vocab size: {len(params.vocab) + len(params.special_tokens)}")
    print(f"Number of merges learned: {len(params.merges)}")

    # Test the tokenizer
    tokenizer = BPETokenizer(params)
    test_text = "Hello world! This is a test."
    tokens = tokenizer.encode(test_text, allowed_special="all")
    decoded = tokenizer.decode(tokens)

    print(f"\nTest encoding/decoding:")
    print(f"Original: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")
    print(f"Round-trip successful: {test_text == decoded}")


def test_multilingual_pretokenization():
    """Test the multilingual capabilities of the GPT-2 Unicode-aware pattern."""
    print("=== Testing Multilingual Pretokenization ===\n")

    # Test cases with different languages
    test_cases = [
        ("English", "Hello world! I'm testing 123 tokens."),
        ("Chinese", "你好世界！我正在测试123个令牌。"),
        ("Arabic", "مرحبا بالعالم! أنا أختبر 123 رمزا."),
        ("Japanese", "こんにちは世界！私は123トークンをテストしています。"),
        ("Hindi", "नमस्ते दुनिया! मैं 123 टोकन का परीक्षण कर रहा हूं।"),
        ("Mixed", "Hello 你好 مرحبا こんにちは नमस्ते 123!"),
    ]

    pattern = GPT_PRETOKENIZATION_PATTERN
    compiled_regex = regex.compile(pattern)

    for language, text in test_cases:
        print(f"{language}: '{text}'")
        tokens = compiled_regex.findall(text)
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print()


if __name__ == "__main__":
    bpe_tokenizer()
    print("\n" + "=" * 50 + "\n")
    benchmark_training_performance()
    print("\n" + "=" * 50 + "\n")
    test_multilingual_pretokenization()
