"""Evaluate Task 2 oracle and predicted hard routing on the official test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.pet_dataset import (
    PETS_ROOT,
    ManifestPetDataset,
    load_manifest,
    tensor_to_pil,
)
from genai.training.losses import per_image_scores
from genai.training.routing import (
    collect_routing_records,
    load_hard_router,
    select_routing_failures,
)
from genai.training.task1 import (
    build_model,
    get_device,
    save_example_grid,
    summarize_records,
    write_table,
)
from genai.training.task2_specialists import SPECIALIST_CORRUPTIONS


def _load_universal(path: Path, device: torch.device):
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload["params"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model


def _collect_passthrough(loader: DataLoader, device: torch.device) -> list[dict]:
    records = []
    with torch.no_grad():
        for batch in loader:
            scores = per_image_scores(batch["corrupted"].to(device), batch["clean"].to(device))
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
    return records


def _collect_universal(model, loader: DataLoader, device: torch.device) -> list[dict]:
    records = []
    with torch.no_grad():
        for batch in loader:
            restored = model(batch["corrupted"].to(device))
            scores = per_image_scores(restored, batch["clean"].to(device))
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
    return records


def _routing_to_metric_records(records: list[dict]) -> list[dict]:
    return [
        {
            "image_id": row["image_id"],
            "corruption": row["true_corruption"],
            "severity": row["severity"],
            "l1": row["l1"],
            "ssim": row["ssim"],
            "psnr": row["psnr"],
        }
        for row in records
    ]


def _save_failure_grid(
    path: Path,
    router,
    failures: list[dict],
    manifest_rows: list[dict],
    image_dir: Path,
    device: torch.device,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (row["image_id"], row["corruption"], row.get("severity") or None): row
        for row in manifest_rows
    }
    figure, axes = plt.subplots(len(failures), 4, figsize=(10, 2.6 * max(len(failures), 1)))
    if len(failures) == 1:
        axes = axes.reshape(1, 4)
    with torch.no_grad():
        for row_index, record in enumerate(failures):
            manifest_row = lookup[
                (record["image_id"], record["true_corruption"], record["severity"])
            ]
            sample = ManifestPetDataset([manifest_row], image_dir)[0]
            corrupted = sample["corrupted"].unsqueeze(0).to(device)
            restored, route_indices, _ = router.restore(corrupted)
            restored = restored.squeeze(0).cpu()
            error = (restored - sample["clean"]).abs().mean(dim=0).numpy()
            panels = [
                tensor_to_pil(sample["clean"]),
                tensor_to_pil(sample["corrupted"]),
                tensor_to_pil(restored),
                error,
            ]
            label = (
                f"true {record['true_corruption']}\n"
                f"route {record['route_corruption']}\n"
                f"L1 {record['l1']:.3f} gap {record['l1_gap']:.3f}"
            )
            axes[row_index, 0].set_ylabel(label, fontsize=8)
            for column, panel in enumerate(panels):
                axis = axes[row_index, column]
                if column == 3:
                    axis.imshow(panel, cmap="magma", vmin=0, vmax=max(float(error.max()), 1e-6))
                else:
                    axis.imshow(panel)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(["clean", "corrupted", "restored", "absolute error"][column])
    figure.suptitle("Task 2 misrouting failures (predicted worse than oracle)", fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "task2"
    results_dir = ROOT / "results" / "task2"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device {device}", flush=True)

    classifier_ckpt = output_dir / "corruption_classifier.pt"
    specialist_ckpts = {
        name: output_dir / f"specialist_{name}.pt" for name in SPECIALIST_CORRUPTIONS
    }
    for path in [classifier_ckpt, *specialist_ckpts.values()]:
        if not path.exists():
            raise SystemExit(f"missing checkpoint: {path}")

    router = load_hard_router(classifier_ckpt, specialist_ckpts, device)
    manifest = load_manifest(PETS_ROOT / "manifests" / "test_corruptions.jsonl")
    image_dir = PETS_ROOT / "processed" / "test"
    loader = DataLoader(
        ManifestPetDataset(manifest, image_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False,
    )

    print("scoring do-nothing baseline", flush=True)
    identity = summarize_records(_collect_passthrough(loader, device))
    write_table(results_dir / "identity_test_metrics.csv", identity)

    universal_path = ROOT / "outputs" / "task1" / "denoising_autoencoder.pt"
    universal_summary = None
    universal = _load_universal(universal_path, device)
    if universal is not None:
        print("scoring Task 1 universal model", flush=True)
        universal_summary = summarize_records(_collect_universal(universal, loader, device))
        write_table(results_dir / "universal_test_metrics.csv", universal_summary)
        del universal
        if device.type == "mps":
            torch.mps.empty_cache()

    print("scoring oracle routing", flush=True)
    oracle_records = collect_routing_records(router, loader, device, mode="oracle")
    oracle_summary = summarize_records(_routing_to_metric_records(oracle_records))
    write_table(results_dir / "oracle_test_metrics.csv", oracle_summary)
    (results_dir / "oracle_test_metrics.json").write_text(json.dumps(oracle_summary, indent=2) + "\n")

    print("scoring predicted routing", flush=True)
    predicted_records = collect_routing_records(router, loader, device, mode="predicted")
    predicted_summary = summarize_records(_routing_to_metric_records(predicted_records))
    write_table(results_dir / "predicted_test_metrics.csv", predicted_summary)
    (results_dir / "predicted_test_metrics.json").write_text(
        json.dumps(predicted_summary, indent=2) + "\n"
    )

    failures = select_routing_failures(oracle_records, predicted_records, limit=8)
    (results_dir / "routing_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        _save_failure_grid(
            results_dir / "routing_failures.png",
            router,
            failures[:4],
            manifest,
            image_dir,
            device,
        )

    # Representative predicted restorations: one median-L1 case per corruption.
    representatives = []
    for corruption in ("clean", *SPECIALIST_CORRUPTIONS):
        group = sorted(
            (row for row in predicted_records if row["true_corruption"] == corruption),
            key=lambda row: row["l1"],
        )
        if group:
            chosen = group[len(group) // 2]
            representatives.append(
                {
                    "image_id": chosen["image_id"],
                    "corruption": chosen["true_corruption"],
                    "severity": chosen["severity"],
                    "l1": chosen["l1"],
                    "ssim": chosen["ssim"],
                    "psnr": chosen["psnr"],
                }
            )
    if representatives:
        save_example_grid(
            results_dir / "predicted_examples.png",
            # save_example_grid expects an autoencoder; for clean bypass we wrap the router.
            _RouterAsModel(router, device),
            representatives,
            manifest,
            image_dir,
            device,
            "Task 2 predicted-routing restorations",
        )

    report = {
        "device": str(device),
        "identity": identity,
        "universal": universal_summary,
        "oracle": oracle_summary,
        "predicted": predicted_summary,
        "n_routing_failures_saved": len(failures),
        "misroute_rate": sum(row["misrouted"] for row in predicted_records) / len(predicted_records),
    }
    (results_dir / "routing_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "misroute_rate": report["misroute_rate"],
                "oracle_all": next(row for row in oracle_summary if row["corruption"] == "all"),
                "predicted_all": next(row for row in predicted_summary if row["corruption"] == "all"),
                "identity_all": next(row for row in identity if row["corruption"] == "all"),
            },
            indent=2,
        )
    )


class _RouterAsModel(torch.nn.Module):
    """Adapter so save_example_grid can call predicted routing."""

    def __init__(self, router, device: torch.device):
        super().__init__()
        self.router = router
        self._device = device

    def eval(self):
        return self

    def __call__(self, corrupted: torch.Tensor) -> torch.Tensor:
        restored, _, _ = self.router.restore(corrupted)
        return restored


if __name__ == "__main__":
    main()
