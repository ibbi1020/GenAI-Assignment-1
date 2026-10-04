import random
from pathlib import Path

import torch
from PIL import Image

from genai.data.pet_dataset import ManifestPetDataset, TrainPetDataset, pil_to_tensor
from genai.models.denoising_autoencoder import DenoisingAutoencoder
from genai.training.losses import reconstruction_loss, structural_similarity, validation_objective


def test_autoencoder_shrinks_to_the_bottleneck_size_and_rebuilds_the_image():
    model = DenoisingAutoencoder(encoder_channels=8, bottleneck_dimension=32, norm="group")
    model.eval()
    hidden = torch.rand(2, 3, 128, 128)
    smallest = 128
    with torch.no_grad():
        for layer in model.dae_e2e:
            hidden = layer(hidden)
            smallest = min(smallest, hidden.shape[-1])
    assert smallest == 32
    assert hidden.shape == (2, 3, 128, 128)
    assert hidden.min() >= 0
    assert hidden.max() <= 1


def test_reconstruction_loss_is_zero_for_a_perfect_copy_and_follows_alpha():
    image = torch.rand(2, 3, 128, 128)
    total, l1, ssim = reconstruction_loss(image, image, alpha=0.8)
    assert torch.allclose(l1, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(ssim, torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(total, torch.tensor(0.0), atol=1e-4)

    blank = torch.zeros_like(image)
    solid = torch.ones_like(image)
    total_l1, l1_only, _ = reconstruction_loss(blank, solid, alpha=1.0)
    assert torch.allclose(l1_only, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(total_l1, l1_only)


def test_ssim_of_a_constant_image_matches_itself():
    image = torch.full((1, 3, 128, 128), 0.4)
    score = structural_similarity(image, image)
    assert torch.allclose(score, torch.ones(1), atol=1e-5)


def test_validation_objective_improves_when_error_falls():
    worse = validation_objective(mean_l1=0.2, mean_ssim=0.7)
    better = validation_objective(mean_l1=0.05, mean_ssim=0.95)
    assert better < worse


def test_training_corruption_is_repeatable_for_the_same_seed(tmp_path: Path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    _write_gradient(image_dir / "pet.png")
    first = TrainPetDataset(["pet"], image_dir, rng_factory=lambda: random.Random(4))
    second = TrainPetDataset(["pet"], image_dir, rng_factory=lambda: random.Random(4))
    assert torch.equal(first[0]["corrupted"], second[0]["corrupted"])
    assert torch.equal(first[0]["clean"], second[0]["clean"])
    assert first[0]["corrupted"].shape == (3, 128, 128)


def test_manifest_dataset_applies_the_stored_corruption(tmp_path: Path):
    image_dir = tmp_path / "val"
    image_dir.mkdir()
    _write_gradient(image_dir / "pet.png")
    row = {
        "image_id": "pet",
        "split": "val",
        "corruption": "salt_and_pepper",
        "severity": "high",
        "sample_seed": 5,
        "seed": 5,
        "salt_probability": 0.15,
        "blur_kernel": None,
        "blur_sigma": None,
        "n_rectangles": None,
        "rectangles": None,
        "covered_pixels": None,
        "coverage": None,
    }
    dataset = ManifestPetDataset([row], image_dir)
    first = dataset[0]
    second = dataset[0]
    assert first["corruption"] == "salt_and_pepper"
    assert torch.equal(first["corrupted"], second["corrupted"])
    assert not torch.equal(first["corrupted"], first["clean"])
    assert first["clean"].shape == pil_to_tensor(
        Image.open(image_dir / "pet.png").convert("RGB")
    ).shape


def _write_gradient(path: Path) -> None:
    rows = []
    for y in range(128):
        row = [(y * 2, x * 2, 80) for x in range(128)]
        rows.append(row)
    Image.fromarray(__import__("numpy").array(rows, dtype="uint8"), mode="RGB").save(path)
