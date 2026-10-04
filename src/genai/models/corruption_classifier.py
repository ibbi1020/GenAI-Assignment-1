"""Four-class corruption classifier for Task 2 hard routing."""

from __future__ import annotations

import torch
from torch import nn

from genai.data.corruptions import CONDITIONS

IMAGE_SIZE = 128
N_CLASSES = len(CONDITIONS)


class CorruptionClassifier(nn.Module):
    """Small CNN: stride-2 conv blocks, global average pool, linear to 4 logits."""

    def __init__(
        self,
        base_channels: int = 32,
        n_blocks: int = 4,
        dropout: float = 0.1,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        if base_channels < 1:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, got {n_blocks}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {n_classes}")

        self.base_channels = base_channels
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.n_classes = n_classes

        blocks = []
        in_channels = 3
        channels = base_channels
        for _ in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=False),
                    nn.Dropout2d(dropout),
                )
            )
            in_channels = channels
            channels = min(channels * 2, base_channels * 8)
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, n_classes),
        )
        self.apply(_init_weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.features(image)
        hidden = self.pool(hidden)
        return self.head(hidden)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
