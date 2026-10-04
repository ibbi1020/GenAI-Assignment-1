import random

import numpy as np
from PIL import Image

from genai.data.corruptions import (
    TEST_BLUR,
    TEST_OCCLUSION,
    TEST_SALT,
    apply_corruption,
    build_test_manifest,
    build_val_manifest,
    gaussian_kernel_1d,
    place_rectangles,
    sample_training_corruption,
)


def _solid(color: tuple[int, int, int]) -> Image.Image:
    array = np.zeros((128, 128, 3), dtype=np.uint8)
    array[:, :] = color
    return Image.fromarray(array, mode="RGB")


def test_gaussian_kernel_sums_to_one_and_blur_keeps_a_flat_image():
    for kernel_size, sigma in ((3, 0.7), (5, 1.5), (7, 2.5)):
        kernel = gaussian_kernel_1d(kernel_size, sigma)
        assert abs(float(kernel.sum()) - 1.0) < 1e-12

    image = _solid((40, 90, 140))
    row = {
        "corruption": "gaussian_blur",
        "blur_kernel": 5,
        "blur_sigma": 1.5,
        "seed": 1,
    }
    output = np.asarray(apply_corruption(image, row))
    assert output.shape == (128, 128, 3)
    assert np.max(np.abs(output.astype(int) - np.asarray(image).astype(int))) <= 1


def test_salt_and_pepper_is_deterministic_and_uses_only_black_or_white():
    image = _solid((10, 20, 30))
    row = {"corruption": "salt_and_pepper", "salt_probability": 0.15, "seed": 123}
    first = np.asarray(apply_corruption(image, row))
    second = np.asarray(apply_corruption(image, row))
    assert np.array_equal(first, second)

    changed = np.any(first != np.asarray(image), axis=2)
    assert changed.any()
    painted = first[changed]
    assert set(map(tuple, painted.reshape(-1, 3))).issubset({(0, 0, 0), (255, 255, 255)})

    other = np.asarray(apply_corruption(image, {**row, "seed": 456}))
    assert not np.array_equal(first, other)


def test_occlusion_rectangles_do_not_overlap_and_cover_the_requested_band():
    for seed in range(30):
        placed = place_rectangles(seed + 1, n_rectangles=(seed % 3) + 1, target_fraction=0.1 + (seed % 5) * 0.05)
        assert placed is not None
        rectangles, covered_pixels, coverage = placed
        mask = np.zeros((128, 128), dtype=np.int32)
        for rect in rectangles:
            mask[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] += 1
        assert int(mask.max()) == 1
        assert int(mask.sum()) == covered_pixels
        assert abs(coverage - covered_pixels / (128 * 128)) < 1e-12

    image = _solid((200, 180, 160))
    placed = place_rectangles(7, 2, 0.20)
    assert placed is not None
    rectangles, covered_pixels, coverage = placed
    row = {
        "corruption": "rectangular_occlusion",
        "rectangles": [rect.as_dict() for rect in rectangles],
    }
    output = np.asarray(apply_corruption(image, row))
    painted = np.all(output == 0, axis=2)
    assert int(painted.sum()) == covered_pixels
    assert 0 <= coverage <= 1


def test_validation_manifest_is_one_row_per_id_and_repeatable():
    ids = ["b", "a", "c", "d"]
    first = build_val_manifest(ids)
    second = build_val_manifest(ids)
    assert first == second
    assert [row["image_id"] for row in first] == ["a", "b", "c", "d"]
    for row in first:
        if row["corruption"] == "salt_and_pepper":
            assert 0.02 <= row["salt_probability"] < 0.15
        if row["corruption"] == "gaussian_blur":
            assert row["blur_kernel"] in (3, 5, 7)
            assert 0.5 <= row["blur_sigma"] < 2.5
        if row["corruption"] == "rectangular_occlusion":
            assert 0.10 <= row["coverage"] <= 0.35
            assert 1 <= row["n_rectangles"] <= 3


def test_test_manifest_uses_the_brief_severity_grid():
    ids = ["pet_a", "pet_b"]
    rows = build_test_manifest(ids)
    assert len(rows) == 20
    per_id: dict[str, list[dict]] = {}
    for row in rows:
        per_id.setdefault(row["image_id"], []).append(row)
    assert list(per_id) == ids

    for image_rows in per_id.values():
        assert image_rows[0]["corruption"] == "clean"
        salts = [row for row in image_rows if row["corruption"] == "salt_and_pepper"]
        blurs = [row for row in image_rows if row["corruption"] == "gaussian_blur"]
        occlusions = [row for row in image_rows if row["corruption"] == "rectangular_occlusion"]
        assert [row["salt_probability"] for row in salts] == [
            TEST_SALT["low"],
            TEST_SALT["medium"],
            TEST_SALT["high"],
        ]
        assert [(row["blur_kernel"], row["blur_sigma"]) for row in blurs] == [
            TEST_BLUR["low"],
            TEST_BLUR["medium"],
            TEST_BLUR["high"],
        ]
        for row, severity in zip(occlusions, ("low", "medium", "high")):
            n_rectangles, target = TEST_OCCLUSION[severity]
            assert row["n_rectangles"] == n_rectangles
            assert abs(row["coverage"] - target) <= 0.01


def test_training_sampler_can_draw_every_condition():
    rng = random.Random(0)
    seen = set()
    for _ in range(80):
        spec = sample_training_corruption(rng)
        seen.add(spec.corruption)
        if spec.corruption == "rectangular_occlusion":
            assert spec.coverage is not None
            assert 0.10 <= spec.coverage <= 0.35
    assert seen == {
        "clean",
        "salt_and_pepper",
        "gaussian_blur",
        "rectangular_occlusion",
    }
