import json
from tests.common import gpt2_bytes_to_unicode

# Load vocabulary
with open('tests/fixtures/gpt2_vocab.json', 'r') as f:
    vocab = json.load(f)

# Create byte decoder
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

# Check for endoftext token
endoftext_bytes = '<|endoftext|>'.encode('utf-8')
print('endoftext bytes:', repr(endoftext_bytes))

found = False
for token_str, token_id in vocab.items():
    try:
        token_bytes = bytes([gpt2_byte_decoder[token] for token in token_str])
        if token_bytes == endoftext_bytes:
            print(f'Found <|endoftext|> at ID {token_id}')
            found = True
            break
    except:
        continue

if not found:
    print('<|endoftext|> not found in vocab') 