"""Unit tests for Task 2 data, classifier metrics, and hard routing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from genai.data.corruptions import CONDITIONS, sample_training_corruption
from genai.data.pet_dataset import (
    BalancedBatchSampler,
    ClassifierTrainDataset,
    TrainPetDataset,
    filter_manifest,
)
from genai.models.corruption_classifier import CorruptionClassifier
from genai.models.denoising_autoencoder import DenoisingAutoencoder
from genai.training.routing import HardRouter
from genai.training.task2_classifier import classification_metrics


def _write_solid(path: Path, color: tuple[int, int, int] = (120, 80, 40)) -> None:
    Image.new("RGB", (128, 128), color).save(path)


def test_forced_corruption_sampler_only_draws_that_class():
    for corruption in CONDITIONS:
        for seed in range(20):
            spec = sample_training_corruption(random.Random(seed), corruption=corruption)
            assert spec.corruption == corruption


def test_specialist_dataset_only_emits_its_corruption(tmp_path: Path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    _write_solid(image_dir / "pet.png")
    dataset = TrainPetDataset(
        ["pet"],
        image_dir,
        rng_factory=lambda: random.Random(7),
        corruption="gaussian_blur",
    )
    for _ in range(5):
        sample = dataset[0]
        assert sample["corruption"] == "gaussian_blur"
        assert sample["label"] == CONDITIONS.index("gaussian_blur")


def test_balanced_batch_sampler_has_equal_classes():
    labels = [0, 1, 2, 3] * 20
    sampler = BalancedBatchSampler(labels, batch_size=8, seed=42, drop_last=True)
    batches = list(sampler)
    # 20 samples per class, 2 of each class per batch -> 10 batches.
    assert len(batches) == 10
    for batch in batches:
        assert len(batch) == 8
        batch_labels = [labels[index] for index in batch]
        counts = [batch_labels.count(class_index) for class_index in range(4)]
        assert counts == [2, 2, 2, 2]


def test_classifier_train_dataset_covers_all_classes(tmp_path: Path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    _write_solid(image_dir / "a.png")
    _write_solid(image_dir / "b.png", (10, 20, 30))
    dataset = ClassifierTrainDataset(["a", "b"], image_dir, seed=3)
    assert len(dataset) == 8
    assert dataset.labels == [0, 1, 2, 3, 0, 1, 2, 3]
    sample = dataset[1]
    assert sample["corruption"] == "salt_and_pepper"
    assert sample["label"] == 1


def test_filter_manifest_keeps_requested_rows():
    rows = [
        {"corruption": "clean", "image_id": "a"},
        {"corruption": "salt_and_pepper", "image_id": "b"},
        {"corruption": "gaussian_blur", "image_id": "c"},
    ]
    filtered = filter_manifest(rows, ["salt_and_pepper", "gaussian_blur"])
    assert [row["image_id"] for row in filtered] == ["b", "c"]


def test_classifier_output_shape_and_metrics():
    model = CorruptionClassifier(base_channels=8, n_blocks=3, dropout=0.0)
    model.eval()
    images = torch.rand(4, 3, 128, 128)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (4, 4)

    confusion = np.array(
        [
            [5, 1, 0, 0],
            [0, 4, 1, 0],
            [0, 0, 6, 0],
            [1, 0, 0, 3],
        ],
        dtype=np.int64,
    )
    metrics = classification_metrics(confusion)
    assert metrics["accuracy"] == 18 / 21
    assert 0 < metrics["macro_f1"] <= 1
    assert len(metrics["per_class"]) == 4
    assert abs(sum(metrics["normalized_confusion"][0]) - 1.0) < 1e-9


def test_hard_router_clean_uses_identity_bypass():
    specialists = {
        "salt_and_pepper": DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group"),
        "gaussian_blur": DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group"),
        "rectangular_occlusion": DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group"),
    }
    for model in specialists.values():
        model.eval()
    router = HardRouter(classifier=None, specialists=specialists)
    corrupted = torch.rand(2, 3, 128, 128)
    labels = torch.tensor([0, 0])
    with torch.no_grad():
        restored, routes, probs = router.restore(corrupted, labels=labels)
    assert torch.equal(restored, corrupted)
    assert torch.equal(routes, labels)
    assert probs.shape == (2, 4)


def test_hard_router_oracle_sends_salt_to_salt_expert():
    salt = DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group").eval()
    blur = DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group").eval()
    occlusion = DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group").eval()
    router = HardRouter(
        classifier=None,
        specialists={
            "salt_and_pepper": salt,
            "gaussian_blur": blur,
            "rectangular_occlusion": occlusion,
        },
    )
    corrupted = torch.rand(1, 3, 128, 128)
    labels = torch.tensor([1])
    with torch.no_grad():
        expected = salt(corrupted)
        restored, routes, _ = router.restore(corrupted, labels=labels)
    assert torch.equal(routes, labels)
    assert torch.allclose(restored, expected)
