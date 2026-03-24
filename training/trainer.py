"""Network trainer: batched SGD on the replay buffer.

Loss = cross-entropy(policy) + MSE(value) + L2 regularisation (via Adam).
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam

from nn.encoding import ACTION_SPACE_SIZE, encode_policy
from nn.model import BlokusNet
from .config import TrainingConfig
from .self_play import TrainingExample


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of training examples."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[TrainingExample] = deque(maxlen=capacity)

    def extend(self, examples: List[TrainingExample]) -> None:
        self.buffer.extend(examples)

    def sample(self, n: int) -> List[TrainingExample]:
        k = min(n, len(self.buffer))
        return random.sample(list(self.buffer), k)

    def __len__(self) -> int:
        return len(self.buffer)


def _collate(
    batch: List[TrainingExample],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a batch into tensors.

    Returns (states, policy_targets, legal_masks, value_targets).
    """
    states = np.stack([ex.state for ex in batch])
    values = np.stack([ex.value for ex in batch])

    policy_dense = np.zeros(
        (len(batch), ACTION_SPACE_SIZE), dtype=np.float32
    )
    mask = np.zeros(
        (len(batch), ACTION_SPACE_SIZE), dtype=np.float32
    )

    for i, ex in enumerate(batch):
        for idx, prob in ex.policy.items():
            policy_dense[i, idx] = prob
            mask[i, idx] = 1.0

    return (
        torch.from_numpy(states).to(device),
        torch.from_numpy(policy_dense).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(values).to(device),
    )


def train_epoch(
    network: BlokusNet,
    optimizer: Adam,
    buffer: ReplayBuffer,
    config: TrainingConfig,
    iteration: int,
    logger=None,
    scaler: GradScaler | None = None,
) -> Tuple[float, float]:
    """Train one epoch over the replay buffer.

    When *scaler* is provided, uses mixed-precision (FP16) for ~2× faster
    training on CUDA.  Returns ``(avg_policy_loss, avg_value_loss)``.
    """
    device = next(network.parameters()).device
    use_amp = scaler is not None and device.type == "cuda"
    network.train()

    indices = list(range(len(buffer.buffer)))
    random.shuffle(indices)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for start in range(0, len(indices), config.batch_size):
        batch_idx = indices[start : start + config.batch_size]
        if len(batch_idx) < 2:
            continue  # skip tiny trailing batch (BN needs ≥2)

        batch = [buffer.buffer[i] for i in batch_idx]
        states, policy_targets, masks, value_targets = _collate(batch, device)

        with autocast("cuda", enabled=use_amp):
            # Forward
            policy_logits, value_pred = network(states)

            # --- policy loss: cross-entropy with MCTS target distribution ---
            masked_logits = policy_logits + (masks - 1.0) * 1e9
            log_probs = F.log_softmax(masked_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

            # --- value loss: MSE ---
            value_loss = F.mse_loss(value_pred, value_targets)

            loss = policy_loss + value_loss

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pl = policy_loss.item()
        vl = value_loss.item()
        total_policy_loss += pl
        total_value_loss += vl
        num_batches += 1

        if logger is not None:
            lr = optimizer.param_groups[0]["lr"]
            logger.log_training_batch(
                iteration=iteration,
                batch=num_batches,
                policy_loss=pl,
                value_loss=vl,
                total_loss=pl + vl,
                lr=lr,
            )

    avg_pl = total_policy_loss / max(num_batches, 1)
    avg_vl = total_value_loss / max(num_batches, 1)
    return avg_pl, avg_vl
