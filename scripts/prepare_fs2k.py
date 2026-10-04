"""Download FS2K, resize photo-sketch pairs, and write the Task 4 splits."""

from __future__ import annotations

import json
import sys
import tarfile
import zipfile
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.corruptions import preprocess_image
from genai.data.fs2k import count_styles, image_id_for, sketch_location, split_train_val, style_name

ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW_DIR = ROOT / "data" / "fs2k" / "raw"
PROCESSED_DIR = ROOT / "data" / "fs2k" / "processed"
SPLIT_DIR = ROOT / "data" / "fs2k" / "splits"
DRIVE_FILE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_root = _find_annotation_root(RAW_DIR)
    if dataset_root is None:
        archive_path = _download_archive()
        _extract_archive(archive_path, RAW_DIR)
        dataset_root = _find_annotation_root(RAW_DIR)
    if dataset_root is None:
        raise SystemExit(f"anno_train.json was not found under {RAW_DIR}")

    official_train = _read_annotations(dataset_root / "anno_train.json")
    official_test = _read_annotations(dataset_root / "anno_test.json")
    _assert_official_lists(official_train, official_test)
    train_records, val_records = split_train_val(official_train)

    photo_dir = _require_dir(dataset_root, "photo")
    sketch_dir = _require_dir(dataset_root, "sketch")

    split_rows = {
        "train": _materialize("train", train_records, photo_dir, sketch_dir),
        "val": _materialize("val", val_records, photo_dir, sketch_dir),
        "test": _materialize("test", official_test, photo_dir, sketch_dir),
    }
    for split_name, rows in split_rows.items():
        _write_jsonl(SPLIT_DIR / f"{split_name}.jsonl", rows)

    meta = {
        "dataset": "FS2K",
        "source": "https://github.com/DengPingFan/FS2K",
        "download": f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view",
        "split_rule": (
            "Official anno_test.json is the test set. From anno_train.json only, "
            "hold out round(0.15 * n) per style. Styles are visited in order 0, 1, 2. "
            "Within a style, records are sorted by image_name and shuffled with "
            "random.Random(42)."
        ),
        "style_names": {"0": "Style 1", "1": "Style 2", "2": "Style 3"},
        "image_rule": "Photo and sketch converted to RGB and stretched to 128x128 with PIL bicubic.",
        "counts": {
            "official_train": len(official_train),
            "official_test": len(official_test),
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
            "official_train_styles": count_styles(official_train),
            "train_styles": count_styles(train_records),
            "val_styles": count_styles(val_records),
            "test_styles": count_styles(official_test),
        },
    }
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["counts"], indent=2))


def _download_archive() -> Path:
    destination = RAW_DIR / "fs2k_download"
    if destination.exists() and destination.stat().st_size > 0:
        print(f"using existing download: {destination}")
        return destination
    print("downloading FS2K from Google Drive")
    import gdown

    downloaded = gdown.download(id=DRIVE_FILE_ID, output=str(destination), quiet=False)
    if not downloaded or not destination.exists():
        raise SystemExit("FS2K download failed")
    return destination


def _extract_archive(archive_path: Path, destination: Path) -> None:
    if _find_annotation_root(destination) is not None:
        return
    print(f"extracting {archive_path.name}")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination, filter="data")
        return
    raise SystemExit(f"{archive_path} is not a zip or tar archive")


def _find_annotation_root(search_root: Path) -> Path | None:
    matches = sorted(search_root.rglob("anno_train.json"))
    if not matches:
        return None
    return matches[0].parent


def _read_annotations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise SystemExit(f"{path} is not a list of annotations")
    return payload


def _assert_official_lists(official_train: list[dict], official_test: list[dict]) -> None:
    train_names = {record["image_name"] for record in official_train}
    test_names = {record["image_name"] for record in official_test}
    if len(train_names) != len(official_train) or len(test_names) != len(official_test):
        raise SystemExit("FS2K annotations contain duplicate image_name values")
    if train_names & test_names:
        raise SystemExit("FS2K official train and test lists overlap")
    if len(official_train) != 1058 or len(official_test) != 1046:
        raise SystemExit(
            f"unexpected FS2K sizes: train {len(official_train)}, test {len(official_test)}"
        )


def _require_dir(dataset_root: Path, name: str) -> Path:
    matches = [path for path in dataset_root.rglob(name) if path.is_dir() and path.name == name]
    if not matches:
        raise SystemExit(f"could not find a {name} directory under {dataset_root}")
    return sorted(matches, key=lambda path: len(path.parts))[0]


def _materialize(
    split: str,
    records: list[dict],
    photo_dir: Path,
    sketch_dir: Path,
) -> list[dict]:
    destination_dir = PROCESSED_DIR / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, record in enumerate(records, start=1):
        image_name = record["image_name"]
        style = int(record["style"])
        image_id = image_id_for(image_name)
        photo_source = _find_photo(photo_dir, image_name)
        sketch_source = _find_sketch(sketch_dir, image_name)
        photo_file = f"{image_id}_photo.png"
        sketch_file = f"{image_id}_sketch.png"
        _write_png(photo_source, destination_dir / photo_file)
        _write_png(sketch_source, destination_dir / sketch_file)
        rows.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "style": style,
                "style_name": style_name(style),
                "official_split": "test" if split == "test" else "train",
                "photo_file": photo_file,
                "sketch_file": sketch_file,
            }
        )
        if index % 200 == 0 or index == len(records):
            print(f"resized {split}: {index}/{len(records)}")
    return rows


def _find_photo(photo_dir: Path, image_name: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = photo_dir / f"{image_name}{suffix}"
        if candidate.is_file():
            return candidate
    direct = photo_dir / image_name
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"photo not found for {image_name} under {photo_dir}")


def _find_sketch(sketch_dir: Path, image_name: str) -> Path:
    sketch_folder, sketch_stem = sketch_location(image_name)
    directory = sketch_dir / sketch_folder
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{sketch_stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"sketch not found for {image_name} under {directory} (stem {sketch_stem})"
    )


def _write_png(source: Path, destination: Path) -> None:
    if destination.is_file():
        with Image.open(destination) as existing:
            if existing.mode == "RGB" and existing.size == (128, 128):
                return
    with Image.open(source) as image:
        prepared = preprocess_image(image)
        prepared.save(destination, format="PNG")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
        raise SystemExit(str(error)) from error
