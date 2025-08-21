import numpy as np
import torch


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding labels from a dataset.

    Args:
        x: 1D numpy array of integer token IDs in the dataset
        batch_size: Desired batch size to sample
        context_length: Desired context length of each sampled example
        device: PyTorch device string (e.g., 'cpu', 'cuda:0', or 'mps')

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length):
        - First tensor: sampled input sequences
        - Second tensor: corresponding next-token targets (offset by 1)
    """
    # Calculate the maximum valid starting index
    # We can start at most at index len(x) - context_length - 1
    # because we need context_length tokens starting from that index
    max_start_idx = len(x) - context_length - 1

    if max_start_idx < 0:
        raise ValueError(
            f"Dataset length {len(x)} is too short for context length {context_length}"
        )

    # Randomly sample starting indices for each batch element
    # Valid indices are from 0 to max_start_idx (inclusive)
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Create input sequences by extracting context_length tokens starting from each start index
    input_sequences = np.array(
        [x[start_idx : start_idx + context_length] for start_idx in start_indices]
    )

    # Create target sequences by shifting input sequences by 1 position
    # Each target position should contain the next token in the sequence
    target_sequences = np.array(
        [
            x[start_idx + 1 : start_idx + context_length + 1]
            for start_idx in start_indices
        ]
    )

    # Convert to PyTorch tensors and move to specified device
    input_tensor = torch.tensor(input_sequences, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_sequences, dtype=torch.long, device=device)

    return input_tensor, target_tensor
