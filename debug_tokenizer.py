from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
from cs336_basics.simple_bpe import _split_on_special_tokens
import tiktoken

# Test the specific problematic text
test_string = 'happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.\n<|endoftext|>\n\nOnce upon a time there was a friendly little boy called Bob. Bob loved to pick flowers and lookfo'

print("=== TESTING WITH SPECIAL TOKENS ===")
tokenizer_with_special = get_tokenizer_from_vocab_merges_path(
    vocab_path='tests/fixtures/gpt2_vocab.json',
    merges_path='tests/fixtures/gpt2_merges.txt',
    special_tokens=["<|endoftext|>"]
)

our_ids_with_special = tokenizer_with_special.encode(test_string)
reference_tokenizer = tiktoken.get_encoding("gpt2")
reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})

print(f"With special tokens - Our IDs length: {len(our_ids_with_special)}")
print(f"With special tokens - Reference IDs length: {len(reference_ids)}")
print(f"With special tokens - Match: {our_ids_with_special == reference_ids}")

# Test without special tokens using a simple string
simple_string = "Once upon a time there was a friendly little boy called Bob."
print(f"\n=== TESTING WITHOUT SPECIAL TOKENS ===")
print(f"Simple string: {repr(simple_string)}")

tokenizer_without_special = get_tokenizer_from_vocab_merges_path(
    vocab_path='tests/fixtures/gpt2_vocab.json',
    merges_path='tests/fixtures/gpt2_merges.txt',
    special_tokens=None
)

our_ids_without_special = tokenizer_without_special.encode(simple_string)
reference_ids_without_special = reference_tokenizer.encode(simple_string, allowed_special=set())

print(f"Without special tokens - Our IDs: {our_ids_without_special}")
print(f"Without special tokens - Reference IDs: {reference_ids_without_special}")
print(f"Without special tokens - Match: {our_ids_without_special == reference_ids_without_special}")

# Find differences in the special token case
if our_ids_with_special != reference_ids:
    for i, (our_id, ref_id) in enumerate(zip(our_ids_with_special, reference_ids)):
        if our_id != ref_id:
            print(f"First difference at index {i}: our_id={our_id}, ref_id={ref_id}")
            # Show what text these tokens represent
            our_context = tokenizer_with_special.decode([our_id])
            ref_context = reference_tokenizer.decode([ref_id])
            print(f"Our token {our_id} represents: {repr(our_context)}")
            print(f"Ref token {ref_id} represents: {repr(ref_context)}")
            break 