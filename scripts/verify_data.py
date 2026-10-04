"""Check the prepared Oxford Pets and FS2K files against the assignment rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.corruptions import (
    TEST_BLUR,
    TEST_SALT,
    TEST_OCCLUSION,
    TEST_COVERAGE_TOLERANCE,
    apply_corruption,
)
from genai.data.fs2k import STYLES, count_styles
from genai.data.pets import assert_disjoint, read_official_ids

PETS = ROOT / "data" / "pets"
FS2K = ROOT / "data" / "fs2k"


def main() -> None:
    problems: list[str] = []
    problems.extend(_verify_pets())
    if (FS2K / "splits" / "train.jsonl").exists():
        problems.extend(_verify_fs2k())
    else:
        problems.append("FS2K splits are missing. Run scripts/prepare_fs2k.py.")
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("shared data checks passed")


def _verify_pets() -> list[str]:
    problems: list[str] = []
    try:
        train_ids = read_official_ids(PETS / "splits" / "train.txt")
        val_ids = read_official_ids(PETS / "splits" / "val.txt")
        test_ids = read_official_ids(PETS / "splits" / "test.txt")
        assert_disjoint(train_ids, val_ids, test_ids)
    except (OSError, ValueError) as error:
        return [f"pet splits: {error}"]

    official_test = read_official_ids(PETS / "raw" / "annotations" / "test.txt")
    official_trainval = read_official_ids(PETS / "raw" / "annotations" / "trainval.txt")
    if set(test_ids) != set(official_test):
        problems.append("pet test split does not match official test.txt")
    if set(train_ids) | set(val_ids) != set(official_trainval):
        problems.append("train + val is not the official trainval list")
    expected_val = int(round(len(official_trainval) * 0.2))
    if len(val_ids) != expected_val:
        problems.append(f"val size {len(val_ids)} != {expected_val}")

    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        problems.extend(_check_png_dir(PETS / "processed" / split, ids, "{image_id}.png"))

    val_rows = _read_jsonl(PETS / "manifests" / "val_corruptions.jsonl")
    test_rows = _read_jsonl(PETS / "manifests" / "test_corruptions.jsonl")
    if [row["image_id"] for row in val_rows] != sorted(val_ids):
        problems.append("val manifest is not one sorted row per val id")
    if len(test_rows) != len(test_ids) * 10:
        problems.append(
            f"test manifest has {len(test_rows)} rows, expected {len(test_ids) * 10}"
        )
    problems.extend(_check_test_manifest(test_rows, test_ids))
    problems.extend(_check_val_manifest(val_rows))
    if val_rows and test_rows:
        problems.extend(_check_apply(PETS / "processed" / "val" / f"{val_rows[0]['image_id']}.png", val_rows[0]))
    return problems


def _verify_fs2k() -> list[str]:
    problems: list[str] = []
    splits = {name: _read_jsonl(FS2K / "splits" / f"{name}.jsonl") for name in ("train", "val", "test")}
    train_names = {row["image_name"] for row in splits["train"]}
    val_names = {row["image_name"] for row in splits["val"]}
    test_names = {row["image_name"] for row in splits["test"]}
    if train_names & val_names or train_names & test_names or val_names & test_names:
        problems.append("FS2K splits overlap")
    if len(test_names) != 1046:
        problems.append(f"FS2K test size {len(test_names)} != 1046")
    if len(train_names) + len(val_names) != 1058:
        problems.append(
            f"FS2K train+val is {len(train_names) + len(val_names)}, expected 1058"
        )

    for split_name, rows in splits.items():
        for style in STYLES:
            style_rows = [row for row in rows if row["style"] == style]
            if split_name == "val":
                official_n = count_styles(splits["train"] + splits["val"])[style]
                expected = int(round(official_n * 0.15))
                if len(style_rows) != expected:
                    problems.append(
                        f"FS2K val style {style} has {len(style_rows)} rows, expected {expected}"
                    )
        problems.extend(
            _check_png_dir(
                FS2K / "processed" / split_name,
                [row["image_id"] for row in rows],
                "{image_id}_photo.png",
            )
        )
        problems.extend(
            _check_png_dir(
                FS2K / "processed" / split_name,
                [row["image_id"] for row in rows],
                "{image_id}_sketch.png",
            )
        )
    return problems


def _check_png_dir(directory: Path, image_ids: list[str], pattern: str) -> list[str]:
    problems: list[str] = []
    if not directory.is_dir():
        return [f"missing directory {directory}"]
    for image_id in image_ids:
        path = directory / pattern.format(image_id=image_id)
        if not path.is_file():
            problems.append(f"missing {path.relative_to(ROOT)}")
            if len(problems) >= 8:
                return problems
            continue
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (128, 128):
                problems.append(f"{path.name} is {image.mode} {image.size}")
                if len(problems) >= 8:
                    return problems
    return problems


def _check_val_manifest(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    for row in rows:
        corruption = row["corruption"]
        if corruption == "salt_and_pepper" and not 0.02 <= row["salt_probability"] < 0.15:
            problems.append(f"{row['image_id']} salt probability out of range")
        if corruption == "gaussian_blur" and row["blur_kernel"] not in (3, 5, 7):
            problems.append(f"{row['image_id']} blur kernel out of range")
        if corruption == "gaussian_blur" and not 0.5 <= row["blur_sigma"] < 2.5:
            problems.append(f"{row['image_id']} blur sigma out of range")
        if corruption == "rectangular_occlusion" and not 0.10 <= row["coverage"] <= 0.35:
            problems.append(f"{row['image_id']} occlusion coverage {row['coverage']}")
        if len(problems) >= 8:
            break
    return problems


def _check_test_manifest(rows: list[dict], test_ids: list[str]) -> list[str]:
    problems: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["image_id"], []).append(row)
    expected_order = ["clean"] + [
        f"{corruption}:{severity}"
        for corruption in ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion")
        for severity in ("low", "medium", "high")
    ]
    for image_id in test_ids:
        image_rows = grouped.get(image_id, [])
        if len(image_rows) != 10:
            problems.append(f"{image_id} has {len(image_rows)} test rows")
            continue
        actual = [
            row["corruption"] if row["severity"] is None else f"{row['corruption']}:{row['severity']}"
            for row in image_rows
        ]
        if actual != expected_order:
            problems.append(f"{image_id} test row order is {actual}")
            continue
        for row in image_rows:
            if row["corruption"] == "salt_and_pepper" and row["salt_probability"] != TEST_SALT[row["severity"]]:
                problems.append(f"{image_id} salt settings differ from the brief")
            if row["corruption"] == "gaussian_blur":
                kernel, sigma = TEST_BLUR[row["severity"]]
                if row["blur_kernel"] != kernel or row["blur_sigma"] != sigma:
                    problems.append(f"{image_id} blur settings differ from the brief")
            if row["corruption"] == "rectangular_occlusion":
                _n_rectangles, target = TEST_OCCLUSION[row["severity"]]
                if abs(row["coverage"] - target) > TEST_COVERAGE_TOLERANCE:
                    problems.append(f"{image_id} occlusion {row['severity']} coverage {row['coverage']}")
                if row["n_rectangles"] != _n_rectangles:
                    problems.append(f"{image_id} occlusion rectangle count {row['n_rectangles']}")
        if problems:
            break
    return problems


def _check_apply(image_path: Path, row: dict) -> list[str]:
    if not image_path.is_file():
        return [f"cannot apply corruption, missing {image_path}"]
    with Image.open(image_path) as image:
        output = apply_corruption(image, row)
    if output.mode != "RGB" or output.size != (128, 128):
        return ["applying a manifest row did not return a 128x128 RGB image"]
    return []


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()
