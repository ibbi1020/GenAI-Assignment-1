"""Score the corrupted input against the clean target (the "do nothing" baseline).

A restorer is only useful if its output is closer to the clean image than the
corrupted input was. This prints L1, SSIM, and PSNR of the corrupted input per
condition, so a model can be compared against it.

Default split is validation. Use ``--split test`` only for the final report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.pet_dataset import PETS_ROOT, ManifestPetDataset, load_manifest
from genai.training.losses import per_image_scores

def summarize_by_class(records: list[dict]) -> list[dict]:
    """Mean scores per (corruption, severity). Works for val (no severity) and test."""
    groups: dict[tuple[str, str | None], list[dict]] = {}
    for record in records:
        groups.setdefault((record["corruption"], record["severity"]), []).append(record)
    rows = []
    for (corruption, severity), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "count": len(group),
                **{key: sum(r[key] for r in group) / len(group) for key in ("l1", "ssim", "psnr")},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(PETS_ROOT / "manifests" / f"{args.split}_corruptions.jsonl")
    dataset = ManifestPetDataset(manifest, PETS_ROOT / "processed" / args.split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    records = []
    for batch in loader:
        scores = per_image_scores(batch["corrupted"], batch["clean"])
        for index, image_id in enumerate(batch["image_id"]):
            records.append(
                {
                    "image_id": image_id,
                    "corruption": batch["corruption"][index],
                    "severity": batch["severity"][index] or None,
                    "l1": float(scores["l1"][index]),
                    "ssim": float(scores["ssim"][index]),
                    "psnr": float(scores["psnr"][index]),
                }
            )
    summary = summarize_by_class(records)
    print(f"corrupted input vs clean, split={args.split}")
    for row in summary:
        print(
            f"{row['corruption']:<24}{str(row['severity']):<8}n={row['count']:<6}"
            f"L1 {row['l1']:.4f}  SSIM {row['ssim']:.4f}  PSNR {row['psnr']:.2f}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
