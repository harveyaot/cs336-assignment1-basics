from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Set
import regex


GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError
        
@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
    special_tokens: dict[str, int] | None = None  # special_token_string -> index

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def _split_on_special_tokens(text: str, special_tokens: Set[str]) -> List[str]:
    """Split text on special tokens, preserving the special tokens as separate parts."""
    if not special_tokens:
        return [text]
    
    # Sort special tokens by length (longest first) to handle overlapping tokens
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    parts = [text]
    for special_token in sorted_specials:
        new_parts = []
        for part in parts:
            if part not in special_tokens:  # Don't split special tokens themselves
                # Split on the special token
                split_parts = part.split(special_token)
                for i, split_part in enumerate(split_parts):
                    if split_part:  # Add non-empty parts
                        new_parts.append(split_part)
                    if i < len(split_parts) - 1:  # Add the special token between parts
                        new_parts.append(special_token)
            else:
                new_parts.append(part)
        parts = new_parts
    
    return parts

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        # Initialize special_tokens if not provided
        if self.params.special_tokens is None:
            object.__setattr__(self.params, 'special_tokens', {})

    def encode(self, string: str) -> list[int]:
        # Handle special tokens first
        special_tokens = set(self.params.special_tokens.keys()) if self.params.special_tokens else set()
        
        if special_tokens:
            # Split text on special tokens
            text_parts = _split_on_special_tokens(string, special_tokens)
            all_tokens = []
            
            for part in text_parts:
                if part in self.params.special_tokens:
                    # This is a special token, add its ID directly
                    all_tokens.append(self.params.special_tokens[part])
                else:
                    # This is regular text, encode it normally
                    all_tokens.extend(self._encode_text(part))
            
            return all_tokens
        else:
            # No special tokens, encode normally
            return self._encode_text(string)

    def _encode_text(self, string: str) -> list[int]:
        """Encode regular text (without special token handling)."""
        # First, apply GPT-2 pre-tokenization using regex
        segments = regex.findall(GPT2_TOKENIZER_REGEX, string)
        
        all_indices = []
        for segment in segments:
            # Convert segment to bytes and find corresponding vocabulary indices
            input_bytes = segment.encode("utf-8")
            indices = []
            for byte in input_bytes:
                # Find the vocabulary index that corresponds to this byte
                for idx, vocab_bytes in self.params.vocab.items():
                    if vocab_bytes == bytes([byte]):
                        indices.append(idx)
                        break
                else:
                    # If byte not found in vocabulary, use the byte value directly
                    indices.append(byte)
            
            # Apply BPE merges to this segment
            for pair, new_index in self.params.merges.items():
                indices = merge(indices, pair, new_index)
            
            all_indices.extend(indices)
        
        return all_indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string

    def encode_iterable(self, iterable) -> list[int]:
        """Encode an iterable (like a file) yielding token IDs one by one."""
        for line in iterable:
            # Encode each line and yield the token IDs
            ids = self.encode(line)
            for token_id in ids:
                yield token_id