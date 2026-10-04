"""Score a saved Task 1 checkpoint against the corrupted input (validation only).

Use this when a training run was stopped early. ``train_task1_fixed.py`` only
prints its comparison table and writes its preview at the very end of the run,
but the best-validation checkpoint is saved after every improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from genai.data.pet_dataset import PETS_ROOT, load_manifest
from genai.training.task1 import build_model, collect_records, get_device, make_loaders, save_example_grid
from train_task1_fixed import CLASSES, input_records, mean_by_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="dae_b32_a05_gn_20ep", help="checkpoint is outputs/task1/task1_<tag>.pt")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    checkpoint = ROOT / "outputs" / "task1" / f"task1_{args.tag}.pt"
    results_dir = ROOT / "results" / "task1"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    device = get_device()
    print(f"checkpoint {checkpoint.name}  val score {payload['score']:.4f}  params {params}", flush=True)

    model = build_model(params)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    loaders = make_loaders(params, num_workers=args.workers, include_test=False)

    records = collect_records(model, loaders["val"], device)
    restored = mean_by_class(records)
    baseline = mean_by_class(input_records(loaders["val"], device))

    print("\nvalidation: restored vs corrupted input (both compared with the clean image)")
    verdict = {}
    for name in ("clean",) + CLASSES:
        r, b = restored[name], baseline[name]
        wins = {"l1": r["l1"] < b["l1"], "ssim": r["ssim"] > b["ssim"], "psnr": r["psnr"] > b["psnr"]}
        verdict[name] = {"restored": r, "input": b, "beats_input": wins}
        print(
            f"{name:<24} n={r['count']:<4}"
            f" L1 {r['l1']:.4f} vs {b['l1']:.4f} {'OK ' if wins['l1'] else 'NO '}"
            f" SSIM {r['ssim']:.3f} vs {b['ssim']:.3f} {'OK ' if wins['ssim'] else 'NO '}"
            f" PSNR {r['psnr']:.2f} vs {b['psnr']:.2f} {'OK' if wins['psnr'] else 'NO'}"
        )

    preview = []
    for name in ("clean",) + CLASSES:
        group = sorted((x for x in records if x["corruption"] == name), key=lambda x: x["l1"])
        preview.append(group[len(group) // 2])
    save_example_grid(
        results_dir / f"val_preview_{args.tag}.png",
        model,
        preview,
        load_manifest(PETS_ROOT / "manifests" / "val_corruptions.jsonl"),
        PETS_ROOT / "processed" / "val",
        device,
        f"Validation preview: {args.tag}",
    )
    (results_dir / f"eval_{args.tag}.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "params": params, "validation_vs_input": verdict}, indent=2) + "\n"
    )
    print(f"\nwrote results/task1/val_preview_{args.tag}.png and eval_{args.tag}.json")


if __name__ == "__main__":
    main()
