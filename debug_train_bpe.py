from cs336_basics.simple_train_bpe import train_bpe
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import json

# Test with a simple string first
print("=== Testing with simple string ===")
simple_string = "the cat in the hat"
params = train_bpe(simple_string, num_merges=3)
print(f"Vocab size: {len(params.vocab)}")
print(f"Merges: {params.merges}")

# Test with the actual corpus
print("\n=== Testing with actual corpus ===")
input_path = FIXTURES_PATH / "corpus.en"
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)

print(f"Vocab size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")
print(f"First 10 merges: {merges[:10]}")

# Compare with reference
print("\n=== Comparing with reference ===")
reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

with open(reference_merges_path) as f:
    gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
    reference_merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_reference_merges
    ]

print(f"Reference merges: {len(reference_merges)}")
print(f"First 10 reference merges: {reference_merges[:10]}")

# Find first difference
for i, (our_merge, ref_merge) in enumerate(zip(merges, reference_merges)):
    if our_merge != ref_merge:
        print(f"First difference at index {i}:")
        print(f"  Our merge: {our_merge}")
        print(f"  Ref merge: {ref_merge}")
        break 