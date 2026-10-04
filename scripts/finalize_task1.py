"""Final Task 1 evaluation of the denoising autoencoder checkpoint.

The checkpoint was chosen on validation (best epoch), so the official test set
is used here once, for reporting. Writes to ``results/task1/dae/`` and exports
``outputs/task1/denoising_autoencoder.onnx`` with a parity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.pet_dataset import PETS_ROOT, load_manifest
from genai.training.losses import per_image_scores
from genai.training.task1 import (
    build_model,
    collect_records,
    export_onnx,
    get_device,
    make_loaders,
    save_example_grid,
    select_failures,
    select_representative,
    summarize_records,
)

LOG_LINE = re.compile(
    r"epoch (\d+)\s+train L1 ([\d.]+) SSIM ([\d.]+)\s+val L1 ([\d.]+) SSIM ([\d.]+)\s+score ([\d.]+)"
)


def input_records(loader) -> list[dict]:
    """Score the corrupted input against the clean image (the do-nothing baseline)."""
    records = []
    for batch in loader:
        scores = per_image_scores(batch["corrupted"], batch["clean"])
        for i in range(len(batch["image_id"])):
            records.append(
                {
                    "corruption": batch["corruption"][i],
                    "severity": batch["severity"][i] or None,
                    **{k: float(scores[k][i]) for k in ("l1", "ssim", "psnr")},
                }
            )
    return records


def compare(restored: list[dict], baseline: list[dict]) -> list[dict]:
    base = {(r["corruption"], r["severity"]): r for r in baseline}
    rows = []
    for row in restored:
        key = (row["corruption"], row["severity"])
        b = base[key]
        rows.append(
            {
                "corruption": row["corruption"],
                "severity": row["severity"],
                "count": row["count"],
                "restored": {k: row[k] for k in ("l1", "ssim", "psnr")},
                "input": {k: b[k] for k in ("l1", "ssim", "psnr")},
                "beats_input": {
                    "l1": row["l1"] < b["l1"],
                    "ssim": row["ssim"] > b["ssim"],
                    "psnr": row["psnr"] > b["psnr"],
                },
            }
        )
    return rows


def parse_log(path: Path) -> list[dict]:
    history = []
    for line in path.read_text().splitlines():
        match = LOG_LINE.search(line)
        if match:
            e, tl1, tss, vl1, vss, score = match.groups()
            history.append(
                {
                    "epoch": int(e),
                    "train_l1": float(tl1),
                    "train_ssim": float(tss),
                    "val_l1": float(vl1),
                    "val_ssim": float(vss),
                    "score": float(score),
                }
            )
    return history


def plot_curves(history: list[dict], path: Path, best_epoch: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (title, train_key, val_key) in zip(
        axes,
        [
            ("L1 (lower is better)", "train_l1", "val_l1"),
            ("SSIM (higher is better)", "train_ssim", "val_ssim"),
            ("Validation score L1 + (1 - SSIM)", None, "score"),
        ],
    ):
        if train_key:
            axis.plot(epochs, [h[train_key] for h in history], marker="o", ms=3, label="train")
        axis.plot(epochs, [h[val_key] for h in history], marker="o", ms=3, label="val")
        axis.axvline(best_epoch, color="gray", ls="--", lw=1)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f"Task 1 denoising autoencoder (dashed = checkpoint, epoch {best_epoch})")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="dae_b32_a05_gn_20ep")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = ROOT / "outputs" / "task1" / f"task1_{args.tag}.pt"
    out = ROOT / "results" / "task1" / "dae"
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    device = get_device()
    model = build_model(params)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    history = parse_log(ROOT / "logs" / f"task1_{args.tag}.log")
    best = min(history, key=lambda h: h["score"])
    plot_curves(history, out / "loss_curves.png", best["epoch"])

    loaders = make_loaders(params, num_workers=args.workers, include_test=True)
    test_loader = loaders["test"]
    print("scoring the test set (restored)", flush=True)
    records = collect_records(model, test_loader, device)
    print("scoring the test set (corrupted input baseline)", flush=True)
    baseline = summarize_records(input_records(test_loader))
    restored = summarize_records(records)
    table = compare(restored, baseline)

    print(f"\n{'condition':<28}{'L1 out/in':<20}{'SSIM out/in':<20}{'PSNR out/in':<20}")
    for row in table:
        name = row["corruption"] + (f" {row['severity']}" if row["severity"] else "")
        r, b, w = row["restored"], row["input"], row["beats_input"]
        print(
            f"{name:<28}{r['l1']:.4f}/{b['l1']:.4f} {'OK' if w['l1'] else 'NO':<4}"
            f"{r['ssim']:.3f}/{b['ssim']:.3f} {'OK' if w['ssim'] else 'NO':<4}"
            f"{r['psnr']:.2f}/{b['psnr']:.2f} {'OK' if w['psnr'] else 'NO'}"
        )

    manifest = load_manifest(PETS_ROOT / "manifests" / "test_corruptions.jsonl")
    image_dir = PETS_ROOT / "processed" / "test"
    representative = select_representative(records)
    failures = select_failures(records)
    save_example_grid(out / "representative_examples.png", model, representative, manifest, image_dir, device,
                      "Task 1 denoising autoencoder: 12 representative test cases")
    save_example_grid(out / "failure_cases.png", model, failures, manifest, image_dir, device,
                      "Task 1 denoising autoencoder: highest-error test case per condition")
    (out / "representative_examples.json").write_text(json.dumps(representative, indent=2) + "\n")
    (out / "failure_cases.json").write_text(json.dumps(failures, indent=2) + "\n")

    with (out / "test_vs_input.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corruption", "severity", "count", "l1", "l1_input", "ssim", "ssim_input", "psnr", "psnr_input"])
        for row in table:
            r, b = row["restored"], row["input"]
            writer.writerow([row["corruption"], row["severity"] or "", row["count"],
                             f"{r['l1']:.6f}", f"{b['l1']:.6f}", f"{r['ssim']:.6f}", f"{b['ssim']:.6f}",
                             f"{r['psnr']:.4f}", f"{b['psnr']:.4f}"])

    sample = next(iter(test_loader))["corrupted"][:4]
    onnx_path = ROOT / "outputs" / "task1" / "denoising_autoencoder.onnx"
    max_abs = export_onnx(model, params, onnx_path, sample)
    print(f"\nONNX max abs diff vs PyTorch: {max_abs:.2e}")

    summary = {
        "model": "DenoisingAutoencoder",
        "config": params,
        "epochs_planned": 20,
        "epochs_trained": history[-1]["epoch"],
        "stopped_early_by_user": history[-1]["epoch"] < 20,
        "checkpoint_epoch": best["epoch"],
        "checkpoint_validation": best,
        "history": history,
        "test_vs_input": table,
        "onnx": str(onnx_path),
        "onnx_max_abs_diff": max_abs,
        "files": {
            "loss_curves": "loss_curves.png",
            "representative_examples": "representative_examples.png",
            "failure_cases": "failure_cases.png",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote results/task1/dae/")
    if max_abs > 1e-4:
        raise SystemExit(f"ONNX output differs from PyTorch by {max_abs}")


if __name__ == "__main__":
    main()
