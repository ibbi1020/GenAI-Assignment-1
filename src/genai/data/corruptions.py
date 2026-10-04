"""Corruption rules shared by Tasks 1, 2, and 3.

Training calls ``sample_training_corruption`` when an image is loaded.
Validation and test call the manifest builders once. Applying a saved row
uses the stored numbers and rectangles, not a second random draw.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

IMAGE_SIZE = 128
MASTER_SEED = 42
CONDITIONS: tuple[str, ...] = (
    "clean",
    "salt_and_pepper",
    "gaussian_blur",
    "rectangular_occlusion",
)

# Fixed final-test settings from the assignment brief.
TEST_SALT = {"low": 0.03, "medium": 0.08, "high": 0.15}
TEST_BLUR = {
    "low": (3, 0.7),
    "medium": (5, 1.5),
    "high": (7, 2.5),
}
TEST_OCCLUSION = {
    "low": (1, 0.10),
    "medium": (2, 0.20),
    "high": (3, 0.35),
}
TEST_SEVERITY_ORDER: tuple[str, ...] = ("low", "medium", "high")
TEST_COVERAGE_TOLERANCE = 0.01
TRAIN_COVERAGE_MIN = 0.10
TRAIN_COVERAGE_MAX = 0.35

MANIFEST_FIELDS: tuple[str, ...] = (
    "image_id",
    "split",
    "corruption",
    "severity",
    "sample_seed",
    "seed",
    "salt_probability",
    "blur_kernel",
    "blur_sigma",
    "n_rectangles",
    "rectangles",
    "covered_pixels",
    "coverage",
)


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CorruptionSpec:
    """One fully specified corruption.

    ``sample_seed`` is the draw from the master stream that created the row.
    ``seed`` is the seed that reproduces a salt pattern. Rectangle geometry is
    stored on the spec, so occlusion does not need to be sampled again.
    """

    corruption: str
    severity: str | None
    sample_seed: int
    seed: int
    salt_probability: float | None = None
    blur_kernel: int | None = None
    blur_sigma: float | None = None
    rectangles: tuple[Rectangle, ...] = ()
    covered_pixels: int | None = None
    coverage: float | None = None

    def as_row(self, image_id: str, split: str) -> dict:
        return {
            "image_id": image_id,
            "split": split,
            "corruption": self.corruption,
            "severity": self.severity,
            "sample_seed": self.sample_seed,
            "seed": self.seed,
            "salt_probability": self.salt_probability,
            "blur_kernel": self.blur_kernel,
            "blur_sigma": self.blur_sigma,
            "n_rectangles": len(self.rectangles) if self.rectangles else None,
            "rectangles": [rect.as_dict() for rect in self.rectangles] or None,
            "covered_pixels": self.covered_pixels,
            "coverage": self.coverage,
        }


def preprocess_image(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Match the shared-data rule: RGB, bicubic stretch to a square."""
    converted = image.convert("RGB")
    if converted.size == (size, size):
        return converted
    return converted.resize((size, size), Image.Resampling.BICUBIC)


def sample_training_corruption(
    rng: random.Random,
    size: int = IMAGE_SIZE,
    corruption: str | None = None,
) -> CorruptionSpec:
    """Draw one training condition. Call this again on every load.

    Pass ``corruption`` to force one class (used by Task 2 specialists).
    """
    sample_seed = rng.randrange(1, 2**31)
    return _sample_from_seed(sample_seed, size=size, corruption=corruption)


def build_val_manifest(val_ids: Sequence[str], size: int = IMAGE_SIZE) -> list[dict]:
    """One frozen condition per validation id, ids walked in sorted order."""
    rng = random.Random(MASTER_SEED)
    rows: list[dict] = []
    for image_id in sorted(val_ids):
        sample_seed = rng.randrange(1, 2**31)
        spec = _sample_from_seed(sample_seed, size=size)
        rows.append(spec.as_row(image_id, "val"))
    return rows


def build_test_manifest(test_ids: Sequence[str], size: int = IMAGE_SIZE) -> list[dict]:
    """Ten frozen rows per test id. ``test_ids`` must stay in official file order."""
    rng = random.Random(MASTER_SEED)
    rows: list[dict] = []
    for image_id in test_ids:
        for corruption, severity in _test_condition_order():
            sample_seed = rng.randrange(1, 2**31)
            spec = _spec_for_test(sample_seed, corruption, severity, size=size)
            rows.append(spec.as_row(image_id, "test"))
    return rows


def apply_corruption(image: Image.Image, row: dict) -> Image.Image:
    """Apply a manifest row or a ``CorruptionSpec.as_row`` dict."""
    if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"expected RGB {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.mode} {image.size}"
        )
    corruption = row["corruption"]
    if corruption == "clean":
        return image.copy()
    pixels = np.asarray(image, dtype=np.uint8)
    if corruption == "salt_and_pepper":
        corrupted = _apply_salt_and_pepper(
            pixels, float(row["salt_probability"]), int(row["seed"])
        )
    elif corruption == "gaussian_blur":
        corrupted = _apply_gaussian_blur(
            pixels, int(row["blur_kernel"]), float(row["blur_sigma"])
        )
    elif corruption == "rectangular_occlusion":
        corrupted = _apply_rectangles(pixels, row["rectangles"])
    else:
        raise ValueError(f"unknown corruption {corruption!r}")
    return Image.fromarray(corrupted, mode="RGB")


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size not in (3, 5, 7):
        raise ValueError(f"kernel size must be 3, 5, or 7, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    center = kernel_size // 2
    positions = np.arange(kernel_size, dtype=np.float64) - center
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _test_condition_order() -> list[tuple[str, str | None]]:
    order: list[tuple[str, str | None]] = [("clean", None)]
    for corruption in ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion"):
        for severity in TEST_SEVERITY_ORDER:
            order.append((corruption, severity))
    return order


def _sample_from_seed(
    sample_seed: int,
    size: int,
    corruption: str | None = None,
) -> CorruptionSpec:
    rng = random.Random(sample_seed)
    if corruption is None:
        corruption = rng.choice(CONDITIONS)
    elif corruption not in CONDITIONS:
        raise ValueError(f"unknown corruption {corruption!r}")
    if corruption == "clean":
        return _clean_spec(sample_seed)
    if corruption == "salt_and_pepper":
        probability = rng.uniform(0.02, 0.15)
        pattern_seed = rng.randrange(1, 2**31)
        return CorruptionSpec(
            corruption=corruption,
            severity=None,
            sample_seed=sample_seed,
            seed=pattern_seed,
            salt_probability=probability,
        )
    if corruption == "gaussian_blur":
        kernel = rng.choice((3, 5, 7))
        sigma = rng.uniform(0.5, 2.5)
        return CorruptionSpec(
            corruption=corruption,
            severity=None,
            sample_seed=sample_seed,
            seed=sample_seed,
            blur_kernel=kernel,
            blur_sigma=sigma,
        )
    return _sample_training_occlusion(rng, sample_seed, size)


def _clean_spec(sample_seed: int) -> CorruptionSpec:
    return CorruptionSpec(
        corruption="clean",
        severity=None,
        sample_seed=sample_seed,
        seed=sample_seed,
    )


def _sample_training_occlusion(
    rng: random.Random, sample_seed: int, size: int
) -> CorruptionSpec:
    last_coverage: float | None = None
    for _ in range(100):
        n_rectangles = rng.randint(1, 3)
        target = rng.uniform(TRAIN_COVERAGE_MIN, TRAIN_COVERAGE_MAX)
        place_seed = rng.randrange(1, 2**31)
        placed = place_rectangles(place_seed, n_rectangles, target, size=size)
        if placed is None:
            continue
        rectangles, covered_pixels, coverage = placed
        last_coverage = coverage
        if TRAIN_COVERAGE_MIN <= coverage <= TRAIN_COVERAGE_MAX:
            return CorruptionSpec(
                corruption="rectangular_occlusion",
                severity=None,
                sample_seed=sample_seed,
                seed=place_seed,
                rectangles=tuple(rectangles),
                covered_pixels=covered_pixels,
                coverage=coverage,
            )
    raise RuntimeError(
        f"could not place a training occlusion inside "
        f"[{TRAIN_COVERAGE_MIN}, {TRAIN_COVERAGE_MAX}] (last coverage {last_coverage})"
    )


def _spec_for_test(
    sample_seed: int, corruption: str, severity: str | None, size: int
) -> CorruptionSpec:
    if corruption == "clean":
        return _clean_spec(sample_seed)
    if corruption == "salt_and_pepper":
        assert severity is not None
        return CorruptionSpec(
            corruption=corruption,
            severity=severity,
            sample_seed=sample_seed,
            seed=sample_seed,
            salt_probability=TEST_SALT[severity],
        )
    if corruption == "gaussian_blur":
        assert severity is not None
        kernel, sigma = TEST_BLUR[severity]
        return CorruptionSpec(
            corruption=corruption,
            severity=severity,
            sample_seed=sample_seed,
            seed=sample_seed,
            blur_kernel=kernel,
            blur_sigma=sigma,
        )
    assert severity is not None
    n_rectangles, target = TEST_OCCLUSION[severity]
    rectangles, covered_pixels, coverage, place_seed = _place_test_occlusion(
        sample_seed, n_rectangles, target, size
    )
    return CorruptionSpec(
        corruption=corruption,
        severity=severity,
        sample_seed=sample_seed,
        seed=place_seed,
        rectangles=tuple(rectangles),
        covered_pixels=covered_pixels,
        coverage=coverage,
    )


def _place_test_occlusion(
    sample_seed: int, n_rectangles: int, target: float, size: int
) -> tuple[list[Rectangle], int, float, int]:
    rng = random.Random(sample_seed)
    last_coverage: float | None = None
    for _ in range(200):
        place_seed = rng.randrange(1, 2**31)
        placed = place_rectangles(place_seed, n_rectangles, target, size=size)
        if placed is None:
            continue
        rectangles, covered_pixels, coverage = placed
        last_coverage = coverage
        if abs(coverage - target) <= TEST_COVERAGE_TOLERANCE:
            return rectangles, covered_pixels, coverage, place_seed
    raise RuntimeError(
        f"could not place {n_rectangles} rectangles near coverage {target} "
        f"(last coverage {last_coverage})"
    )


def place_rectangles(
    seed: int, n_rectangles: int, target_fraction: float, size: int = IMAGE_SIZE
) -> tuple[list[Rectangle], int, float] | None:
    """Place ``n_rectangles`` non-overlapping axis-aligned rectangles.

    Returns None when a rectangle cannot be seated. Coverage is the exact
    share of pixels painted, which can differ slightly from ``target_fraction``
    because widths and heights are integers.
    """
    if n_rectangles < 1:
        raise ValueError("n_rectangles must be at least 1")
    if not 0 < target_fraction < 1:
        raise ValueError(f"target_fraction out of range: {target_fraction}")

    rng = random.Random(seed)
    target_pixels = int(round(target_fraction * size * size))
    target_pixels = min(size * size, max(n_rectangles, target_pixels))
    budgets = _split_pixel_budget(rng, target_pixels, n_rectangles)
    occupied = np.zeros((size, size), dtype=bool)
    rectangles: list[Rectangle] = []

    for budget in budgets:
        seated = _seat_rectangle(rng, occupied, budget, size)
        if seated is None:
            return None
        rectangles.append(seated)

    covered_pixels = int(occupied.sum())
    coverage = covered_pixels / (size * size)
    return rectangles, covered_pixels, coverage


def _split_pixel_budget(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    weights = [rng.random() + 0.05 for _ in range(parts)]
    weight_sum = sum(weights)
    budgets = [max(1, int(round(total * weight / weight_sum))) for weight in weights]
    _repair_budget_sum(budgets, total)
    return budgets


def _repair_budget_sum(budgets: list[int], total: int) -> None:
    while sum(budgets) > total:
        index = max(range(len(budgets)), key=budgets.__getitem__)
        if budgets[index] == 1:
            break
        budgets[index] -= 1
    while sum(budgets) < total:
        index = max(range(len(budgets)), key=budgets.__getitem__)
        budgets[index] += 1


def _seat_rectangle(
    rng: random.Random, occupied: np.ndarray, budget: int, size: int
) -> Rectangle | None:
    for _ in range(400):
        width, height = _choose_shape(rng, budget, size)
        x = rng.randrange(0, size - width + 1)
        y = rng.randrange(0, size - height + 1)
        region = occupied[y : y + height, x : x + width]
        if region.any():
            continue
        occupied[y : y + height, x : x + width] = True
        return Rectangle(x, y, width, height)
    return None


def _choose_shape(rng: random.Random, budget: int, size: int) -> tuple[int, int]:
    """Pick a width and height whose area is close to ``budget``."""
    aspect = math.exp(rng.uniform(math.log(0.4), math.log(2.5)))
    base_height = min(size, max(1, int(round(math.sqrt(budget / aspect)))))
    best: tuple[int, int, int] | None = None
    for delta in range(-6, 7):
        height = base_height + delta
        if not 1 <= height <= size:
            continue
        width = min(size, max(1, int(round(budget / height))))
        error = abs(width * height - budget)
        if best is None or error < best[0]:
            best = (error, width, height)
    assert best is not None
    return best[1], best[2]


def _apply_salt_and_pepper(pixels: np.ndarray, probability: float, seed: int) -> np.ndarray:
    if not 0 <= probability <= 1:
        raise ValueError(f"salt probability out of range: {probability}")
    output = pixels.copy()
    if probability == 0:
        return output
    rng = random.Random(seed)
    height, width = output.shape[:2]
    for y in range(height):
        for x in range(width):
            if rng.random() >= probability:
                continue
            value = 0 if rng.random() < 0.5 else 255
            output[y, x] = value
    return output


def _apply_gaussian_blur(pixels: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    blurred = _separable_convolve(pixels.astype(np.float64), kernel)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def _separable_convolve(pixels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    horizontal = _convolve_axis(pixels, kernel, radius, axis=1)
    return _convolve_axis(horizontal, kernel, radius, axis=0)


def _convolve_axis(
    pixels: np.ndarray, kernel: np.ndarray, radius: int, axis: int
) -> np.ndarray:
    length = pixels.shape[axis]
    output = np.zeros_like(pixels)
    positions = np.arange(length)
    for offset, weight in enumerate(kernel):
        source = _reflect101(positions + (offset - radius), length)
        if axis == 1:
            output += weight * pixels[:, source, :]
        else:
            output += weight * pixels[source, :, :]
    return output


def _reflect101(indices: np.ndarray, length: int) -> np.ndarray:
    """Reflect padding that does not repeat the edge pixel."""
    if length == 1:
        return np.zeros_like(indices)
    reflected = indices.astype(np.int64, copy=True)
    for _ in range(4):
        reflected = np.where(reflected < 0, -reflected, reflected)
        reflected = np.where(reflected >= length, (2 * length - 2) - reflected, reflected)
    return reflected


def _apply_rectangles(pixels: np.ndarray, rectangles: Iterable[dict]) -> np.ndarray:
    output = pixels.copy()
    height, width = output.shape[:2]
    for rect in rectangles:
        x = int(rect["x"])
        y = int(rect["y"])
        rect_width = int(rect["width"])
        rect_height = int(rect["height"])
        if rect_width < 1 or rect_height < 1:
            raise ValueError(f"rectangle has non-positive size: {rect}")
        if x < 0 or y < 0 or x + rect_width > width or y + rect_height > height:
            raise ValueError(f"rectangle falls outside the image: {rect}")
        output[y : y + rect_height, x : x + rect_width] = 0
    return output
