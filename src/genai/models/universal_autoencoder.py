"""Universal denoising autoencoder for Task 1.

The encoder compresses a 128×128 RGB image to one vector. The decoder rebuilds
the image from that vector alone. There are no skip connections, so the network
cannot copy the corrupted input around the bottleneck.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGE_SIZE = 128


class UniversalAutoencoder(nn.Module):
    def __init__(self, base_channels: int = 32, bottleneck_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if base_channels < 1:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if bottleneck_dim < 1:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base_channels = base_channels
        self.bottleneck_dim = bottleneck_dim
        self.spatial = 8
        stage_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoded_channels = stage_channels[-1]

        encoder = []
        in_channels = 3
        for out_channels in stage_channels:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.Dropout2d(dropout),
                )
            )
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder)

        flat_dim = self.spatial * self.spatial * self.encoded_channels
        self.to_latent = nn.Linear(flat_dim, bottleneck_dim)
        self.latent_dropout = nn.Dropout(dropout)
        self.from_latent = nn.Linear(bottleneck_dim, flat_dim)

        decoder = []
        reversed_channels = list(reversed(stage_channels))
        for in_channels, out_channels in zip(reversed_channels, reversed_channels[1:]):
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False),
                )
            )
        self.decoder = nn.ModuleList(decoder)
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(stage_channels[0], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        hidden = image
        for layer in self.encoder:
            hidden = layer(hidden)
        return self.latent_dropout(self.to_latent(hidden.flatten(1)))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.from_latent(latent)
        hidden = hidden.view(-1, self.encoded_channels, self.spatial, self.spatial)
        for layer in self.decoder:
            hidden = layer(hidden)
        return self.to_rgb(hidden)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(image))


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
