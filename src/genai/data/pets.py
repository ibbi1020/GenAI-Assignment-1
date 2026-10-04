"""Oxford-IIIT Pet split used by Tasks 1, 2, and 3."""

from __future__ import annotations

import random
from pathlib import Path

TRAIN_VAL_SEED = 42
VAL_FRACTION = 0.2


def read_official_ids(list_path: Path) -> list[str]:
    """Read the first column of an official trainval.txt or test.txt file."""
    ids: list[str] = []
    for line_number, raw_line in enumerate(list_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        image_id = line.split()[0]
        if not image_id:
            raise ValueError(f"{list_path}:{line_number} has an empty id")
        ids.append(image_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{list_path} contains duplicate ids")
    return ids


def split_train_val(
    trainval_ids: list[str],
    seed: int = TRAIN_VAL_SEED,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Split official trainval into train and val.

    Membership comes from shuffling the official file order with
    ``random.Random(seed)``. Returned lists are sorted so the files stay stable.
    """
    if len(trainval_ids) != len(set(trainval_ids)):
        raise ValueError("trainval ids contain duplicates")
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction out of range: {val_fraction}")

    shuffled = list(trainval_ids)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    if n_val <= 0 or n_val >= len(shuffled):
        raise ValueError(f"validation size {n_val} is invalid for {len(shuffled)} ids")

    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def assert_disjoint(train_ids: list[str], val_ids: list[str], test_ids: list[str]) -> None:
    train, val, test = set(train_ids), set(val_ids), set(test_ids)
    if train & val or train & test or val & test:
        raise ValueError("pet splits overlap")
    if len(train) != len(train_ids) or len(val) != len(val_ids) or len(test) != len(test_ids):
        raise ValueError("a pet split contains duplicate ids")
