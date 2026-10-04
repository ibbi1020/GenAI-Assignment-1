"""Download Oxford-IIIT Pet, resize it, and write the shared Task 1–3 splits."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.corruptions import build_test_manifest, build_val_manifest, preprocess_image
from genai.data.pets import assert_disjoint, read_official_ids, split_train_val

ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW_DIR = ROOT / "data" / "pets" / "raw"
PROCESSED_DIR = ROOT / "data" / "pets" / "processed"
SPLIT_DIR = ROOT / "data" / "pets" / "splits"
MANIFEST_DIR = ROOT / "data" / "pets" / "manifests"

ARCHIVES = {
    "images.tar.gz": {
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "md5": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    },
    "annotations.tar.gz": {
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        "md5": "95a8c909bbe2e81eed6a22bccdf3f68f",
    },
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, info in ARCHIVES.items():
        archive_path = RAW_DIR / name
        _ensure_archive(archive_path, info["url"], info["md5"])
        _extract_archive(archive_path, RAW_DIR)

    trainval_ids = read_official_ids(RAW_DIR / "annotations" / "trainval.txt")
    test_ids = read_official_ids(RAW_DIR / "annotations" / "test.txt")
    train_ids, val_ids = split_train_val(trainval_ids)
    assert_disjoint(train_ids, val_ids, test_ids)

    _write_ids(SPLIT_DIR / "train.txt", train_ids)
    _write_ids(SPLIT_DIR / "val.txt", val_ids)
    _write_ids(SPLIT_DIR / "test.txt", test_ids)

    failures = _resize_split("train", train_ids) + _resize_split("val", val_ids)
    failures += _resize_split("test", test_ids)
    if failures:
        preview = "\n".join(failures[:20])
        raise SystemExit(f"{len(failures)} images failed to convert:\n{preview}")

    val_rows = build_val_manifest(val_ids)
    test_rows = build_test_manifest(test_ids)
    _write_jsonl(MANIFEST_DIR / "val_corruptions.jsonl", val_rows)
    _write_jsonl(MANIFEST_DIR / "test_corruptions.jsonl", test_rows)

    meta = {
        "dataset": "oxford-iiit-pet",
        "source": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "archives": ARCHIVES,
        "split_rule": (
            "Shuffle official trainval.txt order with random.Random(42). "
            "Validation size is round(0.2 * N). Returned id files are sorted. "
            "Test membership is the official test.txt list."
        ),
        "image_rule": "RGB, 128x128, PIL bicubic stretch, saved as PNG.",
        "counts": {
            "official_trainval": len(trainval_ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "val_manifest_rows": len(val_rows),
            "test_manifest_rows": len(test_rows),
        },
    }
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["counts"], indent=2))


def _ensure_archive(path: Path, url: str, expected_md5: str) -> None:
    if path.exists() and _md5(path) == expected_md5:
        print(f"archive ok: {path.name}")
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    actual = _md5(path)
    if actual != expected_md5:
        path.unlink(missing_ok=True)
        raise SystemExit(f"{path.name} md5 {actual} does not match {expected_md5}")


def _extract_archive(archive_path: Path, destination: Path) -> None:
    marker_name = "images" if archive_path.name.startswith("images") else "annotations"
    marker = destination / marker_name
    if marker.exists():
        print(f"already extracted: {marker_name}")
        return
    print(f"extracting {archive_path.name}")
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination, filter="data")


def _resize_split(split: str, image_ids: list[str]) -> list[str]:
    destination_dir = PROCESSED_DIR / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for index, image_id in enumerate(image_ids, start=1):
        destination = destination_dir / f"{image_id}.png"
        if _png_is_ready(destination):
            continue
        source = _find_source(image_id)
        if source is None:
            failures.append(f"{image_id}: source file missing")
            continue
        try:
            _write_png(source, destination)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            failures.append(f"{image_id}: {error}")
        if index % 500 == 0 or index == len(image_ids):
            print(f"resized {split}: {index}/{len(image_ids)}")
    return failures


def _find_source(image_id: str) -> Path | None:
    images_dir = RAW_DIR / "images"
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _write_png(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        prepared = preprocess_image(image)
        prepared.save(destination, format="PNG")


def _png_is_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.mode == "RGB" and image.size == (128, 128)
    except (OSError, UnidentifiedImageError):
        return False


def _write_ids(path: Path, image_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image_id}\n" for image_id in image_ids))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
