"""Pet restoration datasets for Tasks 1–3.

Training samples a new corruption every time an image is loaded. Validation and
test apply the frozen manifest rows.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from genai.data.corruptions import CONDITIONS, apply_corruption, sample_training_corruption
from genai.data.pets import read_official_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
PETS_ROOT = REPO_ROOT / "data" / "pets"

CLASS_TO_INDEX = {name: index for index, name in enumerate(CONDITIONS)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}


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


class ClassifierTrainDataset(Dataset):
    """Four training views of every image, one for each corruption class.

    Index ``4 * image_index + class_index`` always uses that class, so a
    ``BalancedBatchSampler`` can build equal-class batches without guessing.
    """

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
        self.labels = [class_index for _ in image_ids for class_index in range(len(CONDITIONS))]

    def __len__(self) -> int:
        return len(self.labels)

    def _rng_for(self, index: int) -> random.Random:
        if self.rng_factory is not None:
            return self.rng_factory()
        mixed = (self.seed + self.epoch * 1_000_003 + index * 17) % (2**31 - 1)
        return random.Random(mixed or 1)

    def __getitem__(self, index: int) -> dict:
        image_index = index // len(CONDITIONS)
        class_index = index % len(CONDITIONS)
        image_id = self.image_ids[image_index]
        corruption = CONDITIONS[class_index]
        clean = _open_rgb(self.image_dir / f"{image_id}.png")
        spec = sample_training_corruption(self._rng_for(index), corruption=corruption)
        corrupted = apply_corruption(clean, spec.as_row(image_id, "train"))
        return {
            "corrupted": pil_to_tensor(corrupted),
            "clean": pil_to_tensor(clean),
            "corruption": corruption,
            "label": class_index,
            "image_id": image_id,
            "severity": "",
        }


class TrainPetDataset(Dataset):
    def __init__(
        self,
        image_ids: list[str],
        image_dir: Path,
        seed: int = 42,
        rng_factory=None,
        corruption: str | None = None,
    ):
        if corruption is not None and corruption not in CONDITIONS:
            raise ValueError(f"unknown corruption {corruption!r}")
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.seed = seed
        self.epoch = 0
        self.rng_factory = rng_factory
        self.corruption = corruption

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
        spec = sample_training_corruption(
            self._rng_for(index),
            corruption=self.corruption,
        )
        corrupted = apply_corruption(clean, spec.as_row(image_id, "train"))
        return {
            "corrupted": pil_to_tensor(corrupted),
            "clean": pil_to_tensor(clean),
            "corruption": spec.corruption,
            "label": CLASS_TO_INDEX[spec.corruption],
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
            "label": CLASS_TO_INDEX[row["corruption"]],
            "image_id": image_id,
            "severity": severity,
        }


def filter_manifest(rows: Sequence[dict], corruptions: Sequence[str]) -> list[dict]:
    """Keep only rows whose corruption is in ``corruptions``."""
    allowed = set(corruptions)
    unknown = allowed - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown corruptions: {sorted(unknown)}")
    return [row for row in rows if row["corruption"] in allowed]


class BalancedBatchSampler(Sampler[list[int]]):
    """Each batch has the same number of samples from every class.

    ``batch_size`` must be divisible by the number of classes. Indices come from
    a flat list of class labels, one label per dataset index.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n_classes = len(CONDITIONS)
        if batch_size % n_classes != 0:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by {n_classes} classes"
            )
        self.labels = [int(label) for label in labels]
        self.batch_size = batch_size
        self.per_class = batch_size // n_classes
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.indices_by_class: dict[int, list[int]] = {index: [] for index in range(n_classes)}
        for index, label in enumerate(self.labels):
            if label not in self.indices_by_class:
                raise ValueError(f"label {label} is outside 0..{n_classes - 1}")
            self.indices_by_class[label].append(index)
        for label, indices in self.indices_by_class.items():
            if len(indices) < self.per_class:
                raise ValueError(
                    f"class {label} has only {len(indices)} samples, need {self.per_class}"
                )

    def __len__(self) -> int:
        counts = [len(indices) // self.per_class for indices in self.indices_by_class.values()]
        if self.drop_last:
            return min(counts)
        return min(
            (len(indices) + self.per_class - 1) // self.per_class
            for indices in self.indices_by_class.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random((self.seed + self.epoch * 1_000_003) % (2**31 - 1) or 1)
        pools: dict[int, list[int]] = {}
        for label, indices in self.indices_by_class.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            pools[label] = shuffled
        n_batches = len(self)
        for batch_index in range(n_batches):
            batch: list[int] = []
            for label in range(len(CONDITIONS)):
                start = batch_index * self.per_class
                end = start + self.per_class
                pool = pools[label]
                if end <= len(pool):
                    batch.extend(pool[start:end])
                else:
                    # Wrap only when drop_last is False and the last partial round
                    # needs more samples than remain.
                    need = self.per_class
                    chosen = pool[start:]
                    while len(chosen) < need:
                        extra = list(pool)
                        rng.shuffle(extra)
                        chosen.extend(extra)
                    batch.extend(chosen[:need])
            rng.shuffle(batch)
            yield batch


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
