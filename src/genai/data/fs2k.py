"""FS2K split used by Task 4.

The official test list is never passed into ``split_train_val``. Validation is
15% of the official training list, stratified by sketch style.
"""

from __future__ import annotations

import random

TRAIN_VAL_SEED = 42
VAL_FRACTION = 0.15
STYLES: tuple[int, ...] = (0, 1, 2)


def style_name(style: int) -> str:
    if style not in STYLES:
        raise ValueError(f"FS2K style must be 0, 1, or 2, got {style}")
    return f"Style {style + 1}"


def image_id_for(image_name: str) -> str:
    return image_name.replace("/", "_").replace("\\", "_")


def sketch_location(image_name: str) -> tuple[str, str]:
    """Map ``photo1/image0110`` to the sketch folder and file stem.

    FS2K stores sketches beside the photo source folders, not beside the style
    label. ``photo1/image0110`` pairs with ``sketch1/sketch0110``.
    """
    photo_folder, stem = image_name.split("/")
    if not photo_folder.startswith("photo") or not stem.startswith("image"):
        raise ValueError(f"unexpected FS2K image_name {image_name}")
    number = stem[len("image") :]
    sketch_folder = "sketch" + photo_folder[len("photo") :]
    return sketch_folder, f"sketch{number}"


def split_train_val(
    records: list[dict],
    seed: int = TRAIN_VAL_SEED,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[dict], list[dict]]:
    """Hold out a stratified validation set from the official training records.

    Styles are visited in order 0, 1, 2. Inside a style, records are sorted by
    ``image_name`` and then shuffled with one shared ``random.Random(seed)``.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction out of range: {val_fraction}")

    grouped: dict[int, list[dict]] = {style: [] for style in STYLES}
    seen_names: set[str] = set()
    for record in records:
        image_name = record["image_name"]
        if image_name in seen_names:
            raise ValueError(f"duplicate FS2K image_name {image_name}")
        seen_names.add(image_name)
        style = int(record["style"])
        if style not in grouped:
            raise ValueError(f"unexpected FS2K style {style} on {image_name}")
        grouped[style].append(record)

    rng = random.Random(seed)
    train_records: list[dict] = []
    val_records: list[dict] = []
    for style in STYLES:
        group = sorted(grouped[style], key=lambda record: record["image_name"])
        if not group:
            raise ValueError(f"official training list has no records for style {style}")
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        if n_val <= 0 or n_val >= len(group):
            raise ValueError(
                f"validation size {n_val} is invalid for style {style} ({len(group)} records)"
            )
        val_records.extend(group[:n_val])
        train_records.extend(group[n_val:])
    return train_records, val_records


def count_styles(records: list[dict]) -> dict[int, int]:
    counts = {style: 0 for style in STYLES}
    for record in records:
        counts[int(record["style"])] += 1
    return counts
