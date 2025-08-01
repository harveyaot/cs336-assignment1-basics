import json
import pickle
import regex
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Union, Iterator
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

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and special tokens.

        Args:
            vocab: dict[int, bytes] - mapping from token ID to byte sequence
            merges: list[tuple[bytes, bytes]] - list of merge rules as (bytes1, bytes2) pairs
            special_tokens: list[str] | None - optional list of special token strings
        """
        # Convert to internal format for compatibility with existing implementation
        self.vocab = vocab

        # Convert merges from list[tuple[bytes, bytes]] to Dict[Tuple[int, int], int]
        self.merges = {}
        self._merge_ranks = {}

        # Create byte-to-id mapping for merge conversion and encoding
        self.byte_to_id = {}
        for token_id, byte_seq in vocab.items():
            if len(byte_seq) == 1:
                self.byte_to_id[byte_seq] = token_id

        # Convert merges and assign new token IDs
        next_id = max(vocab.keys()) + 1 if vocab else 256

        for i, (bytes1, bytes2) in enumerate(merges):
            # Find token IDs for the byte sequences
            id1 = self.byte_to_id.get(bytes1)
            id2 = self.byte_to_id.get(bytes2)

            if id1 is not None and id2 is not None:
                merged_bytes = bytes1 + bytes2
                
                # Check if the merged token already exists in vocab
                existing_id = None
                for vid, vbytes in self.vocab.items():
                    if vbytes == merged_bytes:
                        existing_id = vid
                        break
                
                if existing_id is not None:
                    # Use existing ID
                    self.merges[(id1, id2)] = existing_id
                else:
                    # Assign new ID
                    self.merges[(id1, id2)] = next_id
                    self.vocab[next_id] = merged_bytes
                    next_id += 1
                
                self._merge_ranks[(id1, id2)] = i  # Track merge order

        # Handle special tokens
        self.special_tokens = {}
        self._token_to_id = {}
        self._id_to_token = {}

        if special_tokens:
            for token in special_tokens:
                # Check if the special token already exists in vocab
                token_bytes = token.encode("utf-8")
                existing_id = None
                for vid, vbytes in self.vocab.items():
                    if vbytes == token_bytes:
                        existing_id = vid
                        break
                
                if existing_id is not None:
                    # Use existing ID
                    self.special_tokens[token] = existing_id
                    self._token_to_id[token] = existing_id
                    self._id_to_token[existing_id] = token
                else:
                    # Assign new ID
                    self.special_tokens[token] = next_id
                    self._token_to_id[token] = next_id
                    self._id_to_token[next_id] = token
                    # Add to vocab
                    self.vocab[next_id] = token_bytes
                    # Update byte_to_id mapping if it's a single byte
                    if len(token_bytes) == 1:
                        self.byte_to_id[token_bytes] = next_id
                    next_id += 1

        # Set pretokenization to GPT-2 pattern
        self._pretokenization_regex = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @classmethod
    def from_params(cls, params: BPETokenizerParams) -> "BPETokenizer":
        """Create tokenizer from BPETokenizerParams (for backward compatibility)."""
        # Convert from internal format to required format
        vocab = params.vocab

        # Convert merges from Dict[Tuple[int, int], int] to list[tuple[bytes, bytes]]
        merges = []
        for (id1, id2), _ in params.merges.items():
            bytes1 = vocab.get(id1, b"")
            bytes2 = vocab.get(id2, b"")
            merges.append((bytes1, bytes2))

        # Convert special_tokens from Dict[str, int] to list[str]
        special_tokens = (
            list(params.special_tokens.keys()) if params.special_tokens else None
        )

        instance = cls(vocab, merges, special_tokens)

        # Set pretokenization pattern if available
        if params.pretokenization_pattern:
            instance._pretokenization_regex = regex.compile(
                params.pretokenization_pattern
            )

        return instance

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Handle special tokens (allow all by default for required interface)
        special_tokens = (
            set(self.special_tokens.keys()) if self.special_tokens else set()
        )

        # Split on special tokens first
        if special_tokens:
            text_parts = self._split_on_special_tokens(text, special_tokens)
        else:
            text_parts = [text]

        # Encode each part
        all_tokens = []
        for part in text_parts:
            if part in self.special_tokens:
                all_tokens.append(self.special_tokens[part])
            else:
                all_tokens.extend(self._encode_ordinary(part))

        return all_tokens

    def encode_iterable(self, iterable) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.

        This is required for memory-efficient tokenization of large files that cannot
        be directly loaded into memory.

        Args:
            iterable: An iterable of strings (e.g., a file handle)

        Yields:
            Token IDs one by one
        """
        for text in iterable:
            if isinstance(text, str):
                tokens = self.encode(text)
                for token_id in tokens:
                    yield token_id
            else:
                raise ValueError(f"Expected string, got {type(text)}")

    def decode(self, ids: List[int]) -> str:
        """Decode a sequence of token IDs into text."""
        if not isinstance(ids, list):
            raise ValueError("Input must be a list of integers")

        # Convert IDs to bytes, handling special tokens
        result_parts = []
        all_bytes = []

        for token_id in ids:
            if token_id in self._id_to_token:
                # Special token - flush accumulated bytes and add special token
                if all_bytes:
                    try:
                        result_parts.append(b"".join(all_bytes).decode("utf-8"))
                        all_bytes = []
                    except UnicodeDecodeError as e:
                        raise ValueError(f"Cannot decode bytes to UTF-8: {e}")
                result_parts.append(self._id_to_token[token_id])
            elif token_id in self.vocab:
                all_bytes.append(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        # Flush remaining bytes
        if all_bytes:
            try:
                result_parts.append(b"".join(all_bytes).decode("utf-8"))
            except UnicodeDecodeError as e:
                # Try to decode with error handling
                try:
                    result_parts.append(b"".join(all_bytes).decode("utf-8", errors="replace"))
                except:
                    raise ValueError(f"Cannot decode bytes to UTF-8: {e}")

        return "".join(result_parts)

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
        ids = []
        
        # Convert each byte to its corresponding token ID
        for byte in token_bytes:
            byte_seq = bytes([byte])
            if byte_seq in self.byte_to_id:
                ids.append(self.byte_to_id[byte_seq])
            else:
                # Fallback: use the byte value directly
                ids.append(byte)

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
            new_token = self.merges[best_pair]
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

    def get_vocab_size(self) -> int:
        """Get the total vocabulary size including special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)

        # Convert to serializable format
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
            "vocab_size": len(self.vocab) + len(self.special_tokens),
            "pretokenization_pattern": None,  # Not stored in new format
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

        # IMPORTANT: We need to preserve the exact original merges and token IDs
        # Instead of using the constructor which assigns new IDs, we'll reconstruct manually

        # Convert special tokens from dict to list
        special_tokens = None
        if "special_tokens" in data and data["special_tokens"]:
            special_tokens = list(data["special_tokens"].keys())

        # Find base vocab (without merged tokens) for constructor
        # Base vocab should be bytes 0-255 plus any single-byte tokens
        base_vocab = {}
        merged_tokens = {}

        for token_id, token_bytes in vocab.items():
            if len(token_bytes) == 1 and token_id < 256:
                base_vocab[token_id] = token_bytes
            else:
                # This is either a merged token or a special token
                base_vocab[token_id] = token_bytes

        # Convert merges preserving original structure
        original_merges = {}
        merges_list = []
        for k, v in data["merges"].items():
            parts = k.split(",")
            id1, id2 = int(parts[0]), int(parts[1])
            merged_token_id = int(v)
            if id1 in vocab and id2 in vocab:
                bytes1 = vocab[id1]
                bytes2 = vocab[id2]
                merges_list.append((bytes1, bytes2))
                original_merges[(id1, id2)] = merged_token_id

        # Create instance with minimal constructor to avoid token ID conflicts
        instance = cls.__new__(cls)

        # Set all attributes directly to preserve original state
        instance.vocab = vocab
        instance.merges = original_merges
        instance.special_tokens = data.get("special_tokens", {})
        instance._token_to_id = {
            token: idx for token, idx in instance.special_tokens.items()
        }
        instance._id_to_token = {
            idx: token for token, idx in instance.special_tokens.items()
        }

        # Create merge ranks preserving original order
        instance._merge_ranks = {}
        for i, (k, v) in enumerate(data["merges"].items()):
            parts = k.split(",")
            id1, id2 = int(parts[0]), int(parts[1])
            if (id1, id2) in original_merges:
                instance._merge_ranks[(id1, id2)] = i

        # Set pretokenization pattern if available
        instance._pretokenization_regex = None
        if data.get("pretokenization_pattern"):
            instance._pretokenization_regex = regex.compile(
                data["pretokenization_pattern"]
            )

        return instance

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """
        Construct and return a Tokenizer from separate serialized vocabulary and merges files.

        Args:
            vocab_filepath: Path to vocabulary file (JSON or pickle format)
            merges_filepath: Path to merges file (JSON or pickle format)
            special_tokens: Optional list of special tokens to add

        Returns:
            BPETokenizer instance

        File Formats Expected:
            vocab_filepath: {"0": [0], "1": [1], ...} mapping token_id -> byte_list
            merges_filepath: {"116,104": 256, ...} mapping "id1,id2" -> new_token_id
        """
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        # Load vocabulary file
        if vocab_path.suffix == ".json":
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
        else:
            with open(vocab_path, "rb") as f:
                vocab_data = pickle.load(f)

        # Load merges file
        if merges_path.suffix == ".json":
            with open(merges_path, "r", encoding="utf-8") as f:
                merges_data = json.load(f)
        else:
            with open(merges_path, "rb") as f:
                merges_data = pickle.load(f)

        # Convert vocabulary to proper format: int -> bytes
        vocab = {}
        for k, v in vocab_data.items():
            token_id = int(k)
            if isinstance(v, list):
                # Format: [byte1, byte2, ...]
                vocab[token_id] = bytes(v)
            elif isinstance(v, bytes):
                # Already in bytes format
                vocab[token_id] = v
            elif isinstance(v, str):
                # String format, encode to bytes
                vocab[token_id] = v.encode("utf-8")
            else:
                raise ValueError(
                    f"Unsupported vocab value format: {type(v)} for key {k}"
                )

        # Convert merges to required format: list[tuple[bytes, bytes]]
        merges = []
        for k, v in merges_data.items():
            if isinstance(k, str) and "," in k:
                # Format: "id1,id2" -> new_id
                parts = k.split(",")
                if len(parts) == 2:
                    id1, id2 = int(parts[0]), int(parts[1])
                    if id1 in vocab and id2 in vocab:
                        bytes1 = vocab[id1]
                        bytes2 = vocab[id2]
                        merges.append((bytes1, bytes2))
                else:
                    raise ValueError(f"Invalid merge key format: {k}")
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                # Format: [id1, id2] -> new_id
                id1, id2 = int(k[0]), int(k[1])
                if id1 in vocab and id2 in vocab:
                    bytes1 = vocab[id1]
                    bytes2 = vocab[id2]
                    merges.append((bytes1, bytes2))
            else:
                raise ValueError(f"Unsupported merge key format: {type(k)} for key {k}")

        return cls(vocab, merges, special_tokens)

    def save_to_files(
        self,
        vocab_filepath: str,
        merges_filepath: str,
        include_special_tokens: bool = False,
    ) -> None:
        """
        Save tokenizer vocabulary and merges to separate files.

        Args:
            vocab_filepath: Path to save vocabulary file (JSON or pickle based on extension)
            merges_filepath: Path to save merges file (JSON or pickle based on extension)
            include_special_tokens: Whether to include special tokens in vocab file
        """
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        # Prepare vocabulary data
        vocab_data = {str(k): list(v) for k, v in self.vocab.items()}

        # Optionally include special tokens in vocab
        if include_special_tokens and self.special_tokens:
            max_vocab_id = max(self.vocab.keys()) if self.vocab else 255
            for token, token_id in self.special_tokens.items():
                if token_id > max_vocab_id:
                    # Special tokens are typically stored as strings, not bytes
                    vocab_data[str(token_id)] = list(token.encode("utf-8"))

        # Prepare merges data
        merges_data = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

        # Save vocabulary file
        if vocab_path.suffix == ".json":
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        else:
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab_data, f)

        # Save merges file
        if merges_path.suffix == ".json":
            with open(merges_path, "w", encoding="utf-8") as f:
                json.dump(merges_data, f, indent=2, ensure_ascii=False)
        else:
            with open(merges_path, "wb") as f:
                pickle.dump(merges_data, f)
