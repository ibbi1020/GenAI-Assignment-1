"""Train Task 1 once with a fixed config (no Optuna) and compare to the corrupted input.

Validation only. The official test set is not loaded.
For each corruption, "beats input" means the restored image is closer to the
clean image than the corrupted input was (lower L1, higher SSIM, higher PSNR).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.pet_dataset import PETS_ROOT, load_manifest
from genai.training.losses import per_image_scores
from genai.training.task1 import (
    build_model,
    collect_records,
    get_device,
    make_loaders,
    save_example_grid,
    seed_everything,
    train_epochs,
)

CONFIG = {
    "alpha": 0.5,
    "bottleneck_dimension": 32,  # spatial size of the bottleneck map (32 x 32)
    "batch_size": 32,
    "encoder_channels": 256,  # channels in the first encoder layer
    "lr": 5e-4,
    "norm": "group",
    "train_seed": 42,
}
EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 2e-3
CLASSES = ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion")


def mean_by_class(records: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for name in ("clean",) + CLASSES:
        group = [r for r in records if r["corruption"] == name]
        out[name] = {k: sum(r[k] for r in group) / len(group) for k in ("l1", "ssim", "psnr")}
        out[name]["count"] = len(group)
    return out


def input_records(loader, device) -> list[dict]:
    records = []
    for batch in loader:
        scores = per_image_scores(batch["corrupted"], batch["clean"])
        for i, corruption in enumerate(batch["corruption"]):
            records.append({"corruption": corruption, **{k: float(scores[k][i]) for k in ("l1", "ssim", "psnr")}})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", default="b32_a05_gn")
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "task1"
    results_dir = ROOT / "results" / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("task1-universal-restoration")

    device = get_device()
    print(f"device {device}  config {CONFIG}  epochs {args.epochs}", flush=True)
    seed_everything(42)
    model = build_model(CONFIG)
    loaders = make_loaders(CONFIG, num_workers=args.workers, include_test=False)
    checkpoint = output_dir / f"task1_{args.tag}.pt"

    with mlflow.start_run(run_name=f"fixed-{args.tag}"):
        mlflow.log_params(CONFIG)
        mlflow.log_params({"epochs": args.epochs, "patience": PATIENCE, "min_delta": MIN_DELTA})
        final = train_epochs(
            model,
            loaders,
            CONFIG,
            device,
            epochs=args.epochs,
            checkpoint_path=checkpoint,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )
        mlflow.log_metric("best_objective", final["best_objective"])

    restored = mean_by_class(collect_records(model, loaders["val"], device))
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

    manifest = load_manifest(PETS_ROOT / "manifests" / "val_corruptions.jsonl")
    preview = []
    val_records = collect_records(model, loaders["val"], device)
    for name in ("clean",) + CLASSES:
        group = sorted((x for x in val_records if x["corruption"] == name), key=lambda x: x["l1"])
        preview.append(group[len(group) // 2])
    save_example_grid(
        results_dir / f"val_preview_{args.tag}.png",
        model,
        preview,
        manifest,
        PETS_ROOT / "processed" / "val",
        device,
        f"Validation preview: {args.tag}",
    )
    report = {
        "config": CONFIG,
        "epochs_budget": args.epochs,
        "epochs_ran": len(final["history"]),
        "best_objective": final["best_objective"],
        "checkpoint": str(checkpoint),
        "validation_vs_input": verdict,
        "history": final["history"],
    }
    (results_dir / f"fixed_run_{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
