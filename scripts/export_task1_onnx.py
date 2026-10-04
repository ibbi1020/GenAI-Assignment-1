"""Export a saved Task 1 checkpoint to ONNX and check it against PyTorch.

Also redraws the training curves from the run log, because a run that was
stopped early never wrote its own history file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.pet_dataset import PETS_ROOT, ManifestPetDataset, load_manifest
from genai.training.task1 import build_model, export_onnx

EPOCH_LINE = re.compile(
    r"epoch (\d+)\s+train L1 ([\d.]+) SSIM ([\d.]+)\s+val L1 ([\d.]+) SSIM ([\d.]+)\s+score ([\d.]+)"
)


def plot_curves(log_path: Path, out_path: Path, title: str) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [tuple(float(x) for x in m.groups()) for m in map(EPOCH_LINE.search, log_path.read_text().splitlines()) if m]
    epochs = [r[0] for r in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for axis, (name, train, val) in zip(
        axes,
        [("L1 (lower is better)", 1, 3), ("SSIM (higher is better)", 2, 4), ("Selection score L1 + (1 - SSIM)", None, 5)],
    ):
        if train is not None:
            axis.plot(epochs, [r[train] for r in rows], marker="o", ms=3, label="train")
        axis.plot(epochs, [r[val] for r in rows], marker="o", ms=3, label="val")
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="dae_b32_a05_gn_20ep")
    args = parser.parse_args()

    checkpoint = ROOT / "outputs" / "task1" / f"task1_{args.tag}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    model = build_model(params)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    rows = load_manifest(PETS_ROOT / "manifests" / "val_corruptions.jsonl")[:4]
    batch = ManifestPetDataset(rows, PETS_ROOT / "processed" / "val")
    sample = torch.stack([batch[i]["corrupted"] for i in range(len(rows))])

    onnx_path = ROOT / "outputs" / "task1" / "denoising_autoencoder.onnx"
    diff = export_onnx(model, params, onnx_path, sample)
    shutil.copyfile(checkpoint, ROOT / "outputs" / "task1" / "denoising_autoencoder.pt")
    print(f"wrote {onnx_path.name}  max abs diff vs PyTorch {diff:.2e}  (limit 1e-4)")
    if diff > 1e-4:
        raise SystemExit("ONNX output differs from PyTorch")

    log = ROOT / "logs" / f"task1_{args.tag}.log"
    if log.exists():
        count = plot_curves(
            log,
            ROOT / "results" / "task1" / f"loss_curves_{args.tag}.png",
            f"Task 1 denoising autoencoder, {count_label(params)}",
        )
        print(f"plotted {count} epochs from {log.name}")


def count_label(params: dict) -> str:
    return (
        f"bottleneck {params['bottleneck_dimension']}x{params['bottleneck_dimension']}, "
        f"{params['encoder_channels']} first-layer channels, alpha {params['alpha']}, {params['norm']} norm"
    )


if __name__ == "__main__":
    main()
