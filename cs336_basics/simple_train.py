import torch
import torch.nn as nn
import math
import os
from typing import Union, BinaryIO, IO


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss with numerical stability.

    Computes ℓi = -log(softmax(oi)[xi+1]) for each sample in the batch.

    Args:
        logits: Predicted logits of shape (..., vocab_size) where ... represents
                any number of batch dimensions
        targets: Target indices of shape (...) where each element is an index
                in [0, vocab_size)

    Returns:
        Average cross entropy loss across the batch (scalar tensor)
    """
    # Get the original shapes
    original_shape = logits.shape[:-1]  # All dimensions except vocab_size
    vocab_size = logits.shape[-1]

    # Flatten batch dimensions for easier processing
    # logits: (..., vocab_size) -> (batch_total, vocab_size)
    logits_flat = logits.view(-1, vocab_size)
    # targets: (...) -> (batch_total,)
    targets_flat = targets.view(-1)

    batch_size = logits_flat.shape[0]

    # Numerical stability: subtract max value along vocabulary dimension
    # Shape: (batch_total, 1)
    max_logits = torch.max(logits_flat, dim=-1, keepdim=True)[0]
    # Shape: (batch_total, vocab_size)
    logits_shifted = logits_flat - max_logits

    # Compute log-sum-exp for denominator of softmax
    # Shape: (batch_total, 1)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True))

    # Gather target logits for each sample
    # Shape: (batch_size, 1)
    target_logits = logits_flat.gather(1, targets_flat.unsqueeze(1))
    target_logits_shifted = target_logits - max_logits

    # Compute cross entropy loss: -log(softmax(logits)[target])
    # = -(target_logit - max_logit - log_sum_exp)
    # = -target_logit + max_logit + log_sum_exp
    losses = -target_logits_shifted + log_sum_exp

    # Remove the extra dimension and return average loss
    losses = losses.squeeze(1)  # Shape: (batch_size,)

    return torch.mean(losses)


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer implementation.

    AdamW decouples weight decay from gradient-based updates, applying
    weight decay directly to parameters rather than incorporating it into gradients.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Initialize first moment vector (momentum)
                    state["exp_avg"] = torch.zeros_like(p.data).float()
                    # Initialize second moment vector (RMSprop)
                    state["exp_avg_sq"] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute bias-corrected first moment estimate
                corrected_exp_avg = exp_avg / bias_correction1

                # Compute bias-corrected second moment estimate
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Compute the denominator (with numerical stability)
                denom = corrected_exp_avg_sq.sqrt().add_(group["eps"])

                # Compute step size
                step_size = group["lr"]

                # AdamW: Apply weight decay directly to parameters (decoupled weight decay)
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Apply update
                p.data.addcdiv_(corrected_exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the total number of iterations for the schedule.
            The actual cosine phase duration is cosine_cycle_iters - warmup_iters.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # Calculate the actual cosine phase duration
    actual_cosine_iters = cosine_cycle_iters - warmup_iters

    # Warmup phase: linear increase from 0 to max_learning_rate
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    # Cosine decay phase: cosine decay from max_learning_rate to min_learning_rate
    elif it < warmup_iters + actual_cosine_iters:
        # Progress through cosine cycle (0 to 1) over actual_cosine_iters steps
        progress = (it - warmup_iters) / actual_cosine_iters
        # Cosine decay: cos(π * progress) goes from 1 to -1, so we transform it
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return (
            min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor
        )

    # After cosine cycle: stay at minimum learning rate
    else:
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients of parameters to have L2 norm at most max_l2_norm.

    This function modifies the gradients in-place. If the total L2 norm
    of all gradients exceeds max_l2_norm, all gradients are scaled down
    proportionally to satisfy the constraint.

    Args:
        parameters: Iterable of parameters with gradients to clip
        max_l2_norm: Maximum L2 norm allowed for the combined gradients
    """
    # Convert to list and filter parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return

    # Compute the total L2 norm of all gradients
    total_norm = 0.0
    for p in params_with_grad:
        param_norm = torch.norm(p.grad.data)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5

    # Clip gradients if the total norm exceeds the maximum
    eps = 1e-6  # PyTorch default epsilon for numerical stability
    clip_coef = max_l2_norm / (total_norm + eps)

    if clip_coef < 1.0:
        # Scale down all gradients by the clipping coefficient
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    """
    Save a checkpoint containing model state, optimizer state, and iteration number.

    Args:
        model: PyTorch module to save
        optimizer: PyTorch optimizer to save
        iteration: Current iteration number
        out: Path or file-like object to save the checkpoint to
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a checkpoint and restore model and optimizer states.

    Args:
        src: Path or file-like object to load the checkpoint from
        model: PyTorch module to restore state to
        optimizer: PyTorch optimizer to restore state to

    Returns:
        The iteration number that was saved in the checkpoint
    """
    checkpoint = torch.load(src)

    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Return the saved iteration number
    return checkpoint["iteration"]
