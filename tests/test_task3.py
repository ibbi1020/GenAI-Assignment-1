"""Unit tests for the Task 3 mixture, its loss, and the routing summaries."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from genai.data.corruptions import CONDITIONS
from genai.models.corruption_classifier import CorruptionClassifier
from genai.models.denoising_autoencoder import DenoisingAutoencoder
from genai.models.soft_mixture import SoftMixture, contributed_most, freeze_experts, unfreeze_experts
from genai.training.task3 import (
    average_gate_weights,
    balance_penalty,
    expert_activity,
    export_mixture_onnx,
    initialize_from_task2,
    mixture_loss,
    pick_routing_examples,
)


class _FixedGate(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits.float())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(image.shape[0], -1)


class _Fill(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.fill = value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.full_like(image, self.fill)


def _mixture(logits: list[float], fills: tuple[float, float, float], temperature: float) -> SoftMixture:
    specialists = {
        "salt_and_pepper": _Fill(fills[0]),
        "gaussian_blur": _Fill(fills[1]),
        "rectangular_occlusion": _Fill(fills[2]),
    }
    return SoftMixture(_FixedGate(torch.tensor(logits)), specialists, temperature)


def test_weights_follow_clean_salt_blur_occlusion_order():
    image = torch.full((1, 3, 8, 8), 0.2)
    # Logits peak on blur, index 2. Blur's fill is the only large value.
    model = _mixture([0.0, 0.0, 6.0, 0.0], (0.0, 0.9, 0.0), temperature=0.5)
    restored, weights = model(image)
    assert weights.shape == (1, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-5)
    assert int(weights.argmax(dim=1)) == CONDITIONS.index("gaussian_blur")
    assert restored.mean().item() > 0.8


def test_lower_temperature_makes_the_gate_sharper():
    image = torch.zeros(1, 3, 4, 4)
    sharp, sharp_weights = _mixture([2.0, 0.0, 0.0, 0.0], (0.0, 0.0, 0.0), 0.5)(image)
    _soft, soft_weights = _mixture([2.0, 0.0, 0.0, 0.0], (0.0, 0.0, 0.0), 2.0)(image)
    assert sharp_weights[0, 0] > soft_weights[0, 0]
    assert sharp.shape == image.shape


def test_output_is_the_weighted_sum_of_the_four_branches():
    image = torch.full((2, 3, 4, 4), 0.2)
    target = torch.tensor([0.1, 0.2, 0.3, 0.4])
    model = _mixture(target.log().tolist(), (1.0, 0.0, 0.5), temperature=1.0)
    restored, weights = model(image)
    assert torch.allclose(weights[0], target, atol=1e-5)
    expected = 0.1 * image + 0.2 * 1.0 + 0.3 * 0.0 + 0.4 * 0.5
    assert torch.allclose(restored, expected, atol=1e-5)


def test_clean_identity_is_the_input_when_that_weight_is_one():
    image = torch.rand(1, 3, 4, 4)
    model = _mixture([8.0, -8.0, -8.0, -8.0], (0.0, 0.0, 0.0), temperature=0.5)
    restored, weights = model(image)
    assert weights[0, 0] > 0.99
    assert torch.allclose(restored, image, atol=1e-4)


def test_balance_penalty_uses_the_batch_mean():
    peaked_but_balanced = torch.eye(4)
    assert balance_penalty(peaked_but_balanced).item() == pytest.approx(0.0)
    one_expert = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # (1 - 0.25)^2 + 3 * (0 - 0.25)^2 = 0.75
    assert balance_penalty(one_expert).item() == pytest.approx(0.75)


def test_joint_loss_matches_the_four_terms():
    restored = torch.zeros(2, 3, 16, 16)
    clean = torch.zeros(2, 3, 16, 16)
    logits = torch.zeros(2, 4)
    labels = torch.tensor([0, 1])
    weights = torch.full((2, 4), 0.25)
    params = {"lambda_l1": 0.8, "lambda_ssim": 0.1, "lambda_cls": 0.01, "lambda_balance": 0.01}
    total, parts = mixture_loss(restored, clean, logits, labels, weights, params)
    expected = 0.8 * parts["l1"] + 0.1 * (1 - parts["ssim"]) + 0.01 * parts["cls"] + 0.01 * parts["balance"]
    assert torch.allclose(total, expected)
    assert parts["balance"].item() == pytest.approx(0.0)
    assert parts["l1"].item() == pytest.approx(0.0)


def test_frozen_experts_get_no_gradient_and_the_gate_does():
    gate = CorruptionClassifier(base_channels=4, n_blocks=2, dropout=0.0)
    specialists = {
        name: DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group")
        for name in ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion")
    }
    model = SoftMixture(gate, specialists, temperature=1.0)
    freeze_experts(model)
    assert model.experts_frozen()
    image = torch.rand(4, 3, 128, 128)
    clean = torch.rand(4, 3, 128, 128)
    labels = torch.arange(4)
    restored, weights, logits = model.components(image)
    loss, _parts = mixture_loss(
        restored,
        clean,
        logits,
        labels,
        weights,
        {"lambda_l1": 0.8, "lambda_ssim": 0.1, "lambda_cls": 0.01, "lambda_balance": 0.01},
    )
    loss.backward()
    for parameter in model.experts.parameters():
        assert parameter.grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.gate.parameters())

    unfreeze_experts(model)
    model.zero_grad(set_to_none=True)
    restored, weights, logits = model.components(image)
    loss, _parts = mixture_loss(
        restored,
        clean,
        logits,
        labels,
        weights,
        {"lambda_l1": 0.8, "lambda_ssim": 0.1, "lambda_cls": 0.01, "lambda_balance": 0.01},
    )
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.experts.parameters())


def test_routing_summaries_group_weights_and_name_a_foreign_expert():
    def row(corruption: str, severity: str | None, weights: list[float]) -> dict:
        named = {name: value for name, value in zip(CONDITIONS, weights, strict=True)}
        return {
            "image_id": corruption,
            "corruption": corruption,
            "severity": severity,
            "weights": named,
            "dominant": max(named, key=named.get),
            "l1": 0.1,
            "ssim": 0.8,
            "psnr": 20.0,
        }

    records = [
        row("clean", None, [0.9, 0.02, 0.04, 0.04]),
        row("salt_and_pepper", "low", [0.05, 0.15, 0.7, 0.1]),
        row("gaussian_blur", "low", [0.25, 0.25, 0.25, 0.25]),
    ]
    table = average_gate_weights(records)
    assert [item["corruption"] for item in table] == ["clean", "salt_and_pepper", "gaussian_blur"]
    assert table[1]["weights"]["gaussian_blur"] == pytest.approx(0.7)

    activity = expert_activity(records)
    foreign = activity["takes_foreign_inputs"]
    assert any(
        item["true_corruption"] == "salt_and_pepper" and item["expert"] == "gaussian_blur"
        for item in foreign
    )

    picks = pick_routing_examples(records, limit=4)
    assert picks["dominant"][0]["corruption"] == "clean"
    assert picks["shared"][0]["corruption"] == "gaussian_blur"
    assert contributed_most(records[2]["weights"]) == list(CONDITIONS)
    assert contributed_most(records[0]["weights"]) == ["clean"]


def test_initialize_from_task2_copies_checkpoint_weights(tmp_path):
    gate = CorruptionClassifier(base_channels=4, n_blocks=2, dropout=0.0)
    specialists = {
        name: DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group")
        for name in ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion")
    }
    classifier_path = tmp_path / "corruption_classifier.pt"
    torch.save(
        {"params": {"base_channels": 4, "n_blocks": 2, "dropout": 0.0}, "state_dict": gate.state_dict(), "score": 0.5},
        classifier_path,
    )
    specialist_paths = {}
    for name, expert in specialists.items():
        path = tmp_path / f"specialist_{name}.pt"
        torch.save(
            {
                "params": {"encoder_channels": 8, "bottleneck_dimension": 32, "norm": "group"},
                "state_dict": expert.state_dict(),
                "score": 0.2,
            },
            path,
        )
        specialist_paths[name] = path

    loaded, meta = initialize_from_task2(
        temperature=1.5,
        classifier_path=classifier_path,
        specialist_paths=specialist_paths,
    )
    assert meta["gate_params"]["base_channels"] == 4
    assert torch.allclose(loaded.temperature, torch.tensor(1.5))
    for key, value in gate.state_dict().items():
        assert torch.equal(loaded.gate.state_dict()[key], value)
    for name, expert in specialists.items():
        for key, value in expert.state_dict().items():
            assert torch.equal(loaded.experts[name].state_dict()[key], value)


def test_onnx_export_matches_the_mixture_on_both_outputs(tmp_path):
    model = _mixture([0.0, 1.0, 0.0, 0.0], (0.2, 0.4, 0.6), temperature=1.0)
    sample = torch.rand(2, 3, 8, 8)
    diff = export_mixture_onnx(model, tmp_path / "soft_mixture.onnx", sample)
    assert diff["restored"] < 1e-4
    assert diff["weights"] < 1e-4


def test_initialize_from_task2_names_missing_files(tmp_path):
    try:
        initialize_from_task2(
            temperature=1.0,
            classifier_path=tmp_path / "missing.pt",
            specialist_paths={"salt_and_pepper": tmp_path / "also_missing.pt"},
        )
    except FileNotFoundError as error:
        assert "missing.pt" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")
