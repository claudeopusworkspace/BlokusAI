"""BlokusNet — ResNet with policy + value heads.

Input  : ``(batch, 10, 20, 20)`` board-state feature planes.
Output : ``(policy_logits, value)``
    * policy_logits : ``(batch, 36 400)`` raw logits over the action space.
    * value         : ``(batch, 4)``  expected normalised score per player ∈ [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blokus.board import BOARD_SIZE, NUM_PLAYERS
from .encoding import NUM_PAIRS


class _ResBlock(nn.Module):
    """Pre-activation residual block (conv → BN → ReLU) × 2 + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class BlokusNet(nn.Module):
    """ResNet-style network for Blokus policy + value prediction."""

    def __init__(
        self,
        in_channels: int = 10,
        num_filters: int = 64,
        num_blocks: int = 5,
        policy_channels: int = 32,
        value_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.board_size = BOARD_SIZE

        # --- shared trunk ---
        self.conv_in = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList(
            [_ResBlock(num_filters) for _ in range(num_blocks)]
        )

        # --- policy head (fully-convolutional) ---
        self.policy_conv = nn.Conv2d(num_filters, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_out = nn.Conv2d(policy_channels, NUM_PAIRS, 1)
        # shape: (batch, 91, 20, 20) → flatten → (batch, 36 400)

        # --- value head ---
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, NUM_PLAYERS)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # trunk
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_out(p)                       # (B, 91, 20, 20)
        p = p.view(p.size(0), -1)                    # (B, 36400)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)                    # (B, 400)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))            # (B, 4) ∈ [-1, 1]

        return p, v
