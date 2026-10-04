"""Soft mixture of experts for Task 3.

The gate is the Task 2 classifier. The three specialists are the Task 2
autoencoders. Clean is an identity branch: it returns the input unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from genai.data.corruptions import CONDITIONS
from genai.models.corruption_classifier import CorruptionClassifier
from genai.models.denoising_autoencoder import DenoisingAutoencoder

# An expert "contributed" when it receives at least an equal share of the mix.
CONTRIBUTION_FLOOR = 0.25


class SoftMixture(nn.Module):
    """Weighted sum of the input and the three specialist restorations.

    Gate weights are ``softmax(logits / temperature)``, in the order
    clean, salt, blur, occlusion.
    """

    def __init__(
        self,
        gate: CorruptionClassifier,
        specialists: dict[str, DenoisingAutoencoder],
        temperature: float,
    ):
        super().__init__()
        if CONDITIONS[0] != "clean" or len(CONDITIONS) != 4:
            raise ValueError(f"expected clean plus three corruptions, got {CONDITIONS}")
        expected = set(CONDITIONS[1:])
        missing = expected - set(specialists)
        extra = set(specialists) - expected
        if missing or extra:
            raise ValueError(f"specialists must be {sorted(expected)}, got {sorted(specialists)}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.gate = gate
        self.experts = nn.ModuleDict({name: specialists[name] for name in CONDITIONS[1:]})
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def experts_frozen(self) -> bool:
        parameters = list(self.experts.parameters())
        return bool(parameters) and all(not parameter.requires_grad for parameter in parameters)

    def components(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the restoration, the four weights, and the raw gate logits.

        Frozen experts run without a gradient so warm-up only trains the gate.
        """
        logits = self.gate(image)
        weights = torch.softmax(logits / self.temperature, dim=1)
        if self.experts_frozen():
            with torch.no_grad():
                salt, blur, occlusion = self._expert_images(image)
        else:
            salt, blur, occlusion = self._expert_images(image)
        restored = (
            weights[:, 0:1, None, None] * image
            + weights[:, 1:2, None, None] * salt
            + weights[:, 2:3, None, None] * blur
            + weights[:, 3:4, None, None] * occlusion
        )
        return restored, weights, logits

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        restored, weights, _logits = self.components(image)
        return restored, weights

    def _expert_images(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.experts["salt_and_pepper"](image),
            self.experts["gaussian_blur"](image),
            self.experts["rectangular_occlusion"](image),
        )


def freeze_experts(model: SoftMixture) -> None:
    """Stop expert updates. The identity branch has no weights to freeze."""
    for expert in model.experts.values():
        expert.requires_grad_(False)
        expert.eval()


def unfreeze_experts(model: SoftMixture) -> None:
    for expert in model.experts.values():
        expert.requires_grad_(True)


def contributed_most(
    weights: dict[str, float], floor: float = CONTRIBUTION_FLOOR
) -> list[str]:
    """Class names, in gate order, whose weight is at least ``floor``."""
    return [name for name in CONDITIONS if float(weights.get(name, 0.0)) >= floor]
