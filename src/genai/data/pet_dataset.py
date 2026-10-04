"""Pet restoration datasets for Tasks 1–3.

Training samples a new corruption every time an image is loaded. Validation and
test apply the frozen manifest rows.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from genai.data.corruptions import apply_corruption, sample_training_corruption
from genai.data.pets import read_official_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
PETS_ROOT = REPO_ROOT / "data" / "pets"


def fresh_rng() -> random.Random:
    """New unseeded generator. A named function can be sent to DataLoader workers."""
    return random.Random()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


class TrainPetDataset(Dataset):
    def __init__(
        self,
        image_ids: list[str],
        image_dir: Path,
        seed: int = 42,
        rng_factory=None,
    ):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.seed = seed
        self.epoch = 0
        self.rng_factory = rng_factory

    def __len__(self) -> int:
        return len(self.image_ids)

    def _rng_for(self, index: int) -> random.Random:
        if self.rng_factory is not None:
            return self.rng_factory()
        mixed = (self.seed + self.epoch * 1_000_003 + index * 17) % (2**31 - 1)
        return random.Random(mixed or 1)

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        clean = _open_rgb(self.image_dir / f"{image_id}.png")
        spec = sample_training_corruption(self._rng_for(index))
        corrupted = apply_corruption(clean, spec.as_row(image_id, "train"))
        return {
            "corrupted": pil_to_tensor(corrupted),
            "clean": pil_to_tensor(clean),
            "corruption": spec.corruption,
            "image_id": image_id,
            "severity": "",
        }


class ManifestPetDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path):
        self.rows = list(rows)
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_id = row["image_id"]
        clean = _open_rgb(self.image_dir / f"{image_id}.png")
        corrupted = apply_corruption(clean, row)
        severity = row.get("severity") or ""
        return {
            "corrupted": pil_to_tensor(corrupted),
            "clean": pil_to_tensor(clean),
            "corruption": row["corruption"],
            "image_id": image_id,
            "severity": severity,
        }


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        loaded = image.convert("RGB")
        loaded.load()
        return loaded


def load_task1_splits(pets_root: Path = PETS_ROOT) -> dict[str, list[str]]:
    return {
        "train": read_official_ids(pets_root / "splits" / "train.txt"),
        "val": read_official_ids(pets_root / "splits" / "val.txt"),
        "test": read_official_ids(pets_root / "splits" / "test.txt"),
    }
