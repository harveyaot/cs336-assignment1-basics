import torch
from torch import nn
from einops import rearrange, einsum
import math


class Linear(nn.Module):
    """
    A linear transformation module without bias.

    This module performs a linear transformation: y = xW^T
    where x is the input tensor and W is the weight matrix.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Initialize the Linear module.

        Args:
            in_features: Final dimension of the input
            out_features: Final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()

        # Create weight parameter W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Perform linear transformation: y = xW^T
        # We use W^T in the computation but store W for memory efficiency
        return einsum(
            self.weight,
            x,
            "out_features in_features, ... in_features -> ... out_features",
        )


class Embedding(nn.Module):
    """
    An embedding module that performs embedding lookup.

    This module maps token IDs to dense embedding vectors.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        """
        Initialize the Embedding module.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()

        # Create embedding weight matrix
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: Input tensor of token IDs of shape (*, )

        Returns:
            Output tensor of shape (*, embedding_dim)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization module.

    RMSNorm normalizes the input by dividing by the root mean square
    of the input elements, then applies a learnable scaling factor.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Initialize the RMSNorm module.

        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()

        self.eps = eps

        # Create learnable weight parameter (scaling factor)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape (*, d_model)

        Returns:
            Output tensor of the same shape as input
        """
        # Store original dtype for later restoration
        original_dtype = x.dtype

        # Upcast to float32 for numerical stability
        x_float32 = x.float()

        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x_float32**2, dim=-1, keepdim=True) + self.eps)

        # Normalize and apply learnable scaling
        normalized = (x_float32 / rms) * self.weight

        # Downcast back to original dtype
        return normalized.to(original_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Swish) activation function.

    SiLU(x) = x * σ(x) = x / (1 + e^(-x))

    Args:
        x: Input tensor

    Returns:
        Output tensor with SiLU activation applied
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network module.

    Implements: FFN(x) = W2(SiLU(W1x) ⊙ W3x)
    where ⊙ represents element-wise multiplication.
    """

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        Initialize the SwiGLU module.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension of the feed-forward layer. If None,
                  defaults to 8/3 * d_model rounded to nearest multiple of 64
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()

        # Calculate d_ff if not provided: 8/3 * d_model, rounded to multiple of 64
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear transformations (no bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate projection
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # Output projection
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Value projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation to the input.

        Args:
            x: Input tensor of shape (*, d_model)

        Returns:
            Output tensor of shape (*, d_model)
        """
        # Compute SiLU(W1x) ⊙ W3x
        gate = silu(self.w1(x))  # Apply W1 then SiLU
        value = self.w3(x)  # Apply W3
        gated = gate * value  # Element-wise multiplication

        # Apply final linear transformation W2
        return self.w2(gated)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) module.

    Applies rotation matrices to inject positional information into query and key vectors.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize the RoPE module.

        Args:
            theta: Θ value for the RoPE
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Pre-compute frequency values
        # Standard RoPE frequency calculation: 1 / θ^(2j/d) for j = 0, 1, 2, ...
        # This gives frequencies: 1/θ^0/d, 1/θ^2/d, 1/θ^4/d, ...
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device).float()

        # Compute angles: outer product of positions and frequencies
        # Shape: (max_seq_len, d_k//2)
        angles = torch.outer(positions, freqs)

        # Pre-compute cos and sin values
        cos_vals = torch.cos(angles)  # (max_seq_len, d_k//2)
        sin_vals = torch.sin(angles)  # (max_seq_len, d_k//2)

        # Register as buffers (not parameters, don't learn these values)
        self.register_buffer("cos_vals", cos_vals, persistent=False)
        self.register_buffer("sin_vals", sin_vals, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)

        Returns:
            Output tensor of the same shape as input
        """
        # Get cos and sin values for the given positions
        cos = self.cos_vals[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_vals[token_positions]  # (..., seq_len, d_k//2)

        # Reshape x to separate even/odd dimensions
        # Split into pairs: (x_0, x_1), (x_2, x_3), ..., (x_{d_k-2}, x_{d_k-1})
        x_even = x[..., 0::2]  # (..., seq_len, d_k//2) - even indices
        x_odd = x[..., 1::2]  # (..., seq_len, d_k//2) - odd indices

        # Apply rotation:
        # For each pair (x_even, x_odd), apply 2D rotation:
        # [x_even_new]   [cos  -sin] [x_even]
        # [x_odd_new ] = [sin   cos] [x_odd ]
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_even * sin + x_odd * cos

        # Interleave the results back to original shape
        result = torch.zeros_like(x)
        result[..., 0::2] = x_even_new
        result[..., 1::2] = x_odd_new

        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation to a tensor along the specified dimension.

    Uses numerical stability trick by subtracting the maximum value
    to prevent overflow in the exponential function.

    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax

    Returns:
        Output tensor with same shape as input, where the specified
        dimension contains normalized probability distributions
    """
    # Subtract max for numerical stability
    # keepdim=True maintains the dimension for broadcasting
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max

    # Compute exponentials
    exp_x = torch.exp(x_shifted)

    # Compute sum of exponentials along the specified dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    # Normalize to get probabilities
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Scaled dot-product attention mechanism.

    Computes attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: Query tensor of shape (..., seq_len, d_k)
        K: Key tensor of shape (..., seq_len, d_k)
        V: Value tensor of shape (..., seq_len, d_v)
        mask: Optional boolean mask of shape (seq_len, seq_len).
              True positions are kept, False positions are masked out.

    Returns:
        Output tensor of shape (..., seq_len, d_v)
    """
    # Get the key dimension for scaling
    d_k = Q.shape[-1]

    # Compute attention scores: QK^T / sqrt(d_k)
    # Q: (..., seq_len, d_k), K: (..., seq_len, d_k)
    # K.transpose(-2, -1): (..., d_k, seq_len)
    # scores: (..., seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        # mask: (seq_len, seq_len), scores: (..., seq_len, seq_len)
        # Set False positions to -inf so they become 0 after softmax
        # ~mask inverts: True->False, False->True, so we mask out False positions
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax to get attention probabilities
    # Softmax along the last dimension (keys dimension)
    attn_probs = softmax(scores, dim=-1)

    # Apply attention probabilities to values
    # attn_probs: (..., seq_len, seq_len), V: (..., seq_len, d_v)
    # output: (..., seq_len, d_v)
    output = torch.matmul(attn_probs, V)

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention module with RoPE.

    Implements the multi-head attention mechanism from "Attention Is All You Need"
    with causal masking and rotary position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        """
        Initialize the multi-head self-attention module.

        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE precomputation
            rope_theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / num_heads

        # Query, Key, Value projection layers
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE for positional embeddings
        self.rope = RotaryPositionalEmbedding(
            rope_theta, self.d_head, max_seq_len, device=device
        )

        # Register causal mask as a buffer
        self.register_buffer("causal_mask", None, persistent=False)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask for the given sequence length."""
        # Create lower triangular mask: True where j <= i, False where j > i
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply causal multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Token positions for RoPE of shape (batch_size, seq_len).
                           If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Default token positions if not provided
        if token_positions is None:
            token_positions = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        # From (batch_size, seq_len, d_model) to (batch_size, num_heads, seq_len, d_head)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        # Apply RoPE to each head separately to avoid dimension issues
        Q_list = []
        K_list = []

        for head_idx in range(self.num_heads):
            # Extract one head at a time: (batch_size, seq_len, d_head)
            Q_head = Q[:, head_idx, :, :]  # (batch_size, seq_len, d_head)
            K_head = K[:, head_idx, :, :]  # (batch_size, seq_len, d_head)

            # Apply RoPE to this head
            Q_head_rope = self.rope(Q_head, token_positions)
            K_head_rope = self.rope(K_head, token_positions)

            Q_list.append(Q_head_rope)
            K_list.append(K_head_rope)

        # Stack heads back: (batch_size, num_heads, seq_len, d_head)
        Q = torch.stack(Q_list, dim=1)
        K = torch.stack(K_list, dim=1)

        # Apply scaled dot-product attention with causal mask
        # Q, K, V: (batch_size, num_heads, seq_len, d_head)
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Concatenate heads: (batch_size, num_heads, seq_len, d_head) -> (batch_size, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Apply output projection
        output = self.output_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Implements the architecture:
    x1 = x + MultiHeadSelfAttention(RMSNorm(x))
    x2 = x1 + SwiGLU(RMSNorm(x1))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        """
        Initialize the Transformer block.

        Args:
            d_model: Dimensionality of the Transformer block inputs
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            max_seq_len: Maximum sequence length for RoPE precomputation
            rope_theta: RoPE theta parameter
            eps: Epsilon for RMSNorm
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # First RMSNorm (before attention)
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype
        )

        # Second RMSNorm (before feed-forward)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply the Transformer block to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Token positions for RoPE of shape (batch_size, seq_len).
                           If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-norm multi-head self-attention with residual connection
        # x1 = x + MultiHeadSelfAttention(RMSNorm(x))
        attn_input = self.ln1(x)
        attn_output = self.attn(attn_input, token_positions)
        x = x + attn_output

        # Pre-norm feed-forward network with residual connection
        # x2 = x1 + SwiGLU(RMSNorm(x1))
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output

        return x


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.

    Implements the full architecture:
    1. Token embeddings
    2. Multiple Transformer blocks
    3. Final RMSNorm
    4. Language model head (linear projection to vocabulary)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        """
        Initialize the Transformer Language Model.

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            context_length: Maximum context length (sequence length)
            d_model: Dimensionality of the model embeddings and sublayer outputs
            num_layers: Number of Transformer blocks to stack
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the feed-forward inner layer
            rope_theta: RoPE theta parameter
            eps: Epsilon for RMSNorm
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Token embeddings
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        # Language model head (output projection to vocabulary)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, input_ids: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer language model.

        Args:
            input_ids: Token indices of shape (batch_size, sequence_length)
            token_positions: Token positions for RoPE of shape (batch_size, sequence_length).
                           If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Generate default token positions if not provided
        if token_positions is None:
            token_positions = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embeddings(input_ids)

        # Pass through each Transformer block
        for layer in self.layers:
            x = layer(x, token_positions)

        # Final layer normalization
        x = self.ln_final(x)

        # Language model head: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
