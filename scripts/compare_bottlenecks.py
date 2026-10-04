"""Train the same Task 1 autoencoder at two other bottleneck sizes.

Every other setting matches the 32 run. Validation chooses the checkpoint.
The official test set is not loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.pet_dataset import PETS_ROOT, load_manifest
from genai.training.task1 import (
    build_model,
    collect_records,
    make_loaders,
    save_example_grid,
    seed_everything,
    summarize_records,
    train_epochs,
    get_device,
)

# Same settings as the main 32 run. Only the bottleneck changes.
FIXED = {
    "lr": 5e-4,
    "batch_size": 32,
    "encoder_channels": 256,
    "norm": "group",
    "alpha": 0.5,
    "train_seed": 42,
}
DIMENSIONS = (16, 64)  # spatial bottleneck sizes; 32 is the main run
EPOCHS = 25
PATIENCE = 6


def median_per_condition(records: list[dict]) -> list[dict]:
    chosen = []
    for corruption in ("clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"):
        group = sorted(
            (record for record in records if record["corruption"] == corruption),
            key=lambda record: record["l1"],
        )
        if group:
            chosen.append(group[len(group) // 2])
    return chosen


def main() -> None:
    output_dir = ROOT / "outputs" / "task1"
    results_dir = ROOT / "results" / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("task1-universal-restoration")

    reference_path = results_dir / "run_summary.json"
    reference_score = None
    if reference_path.exists():
        reference_score = json.loads(reference_path.read_text())["final_validation_objective"]

    device = get_device()
    print(f"device {device}", flush=True)
    print("test set is not used", flush=True)
    runs = []
    manifest = load_manifest(PETS_ROOT / "manifests" / "val_corruptions.jsonl")
    image_dir = PETS_ROOT / "processed" / "val"

    for dimension in DIMENSIONS:
        params = dict(FIXED)
        params["bottleneck_dimension"] = dimension
        print(f"start bottleneck {dimension} params {params}", flush=True)
        seed_everything(42)
        model = build_model(params)
        loaders = make_loaders(params, num_workers=2, include_test=False)
        checkpoint = output_dir / f"bottleneck_{dimension}.pt"
        with mlflow.start_run(run_name=f"bottleneck-{dimension}"):
            mlflow.log_params(params)
            mlflow.log_param("final_epochs", EPOCHS)
            final = train_epochs(
                model,
                loaders,
                params,
                device,
                epochs=EPOCHS,
                checkpoint_path=checkpoint,
                patience=PATIENCE,
            )
            mlflow.log_metric("best_objective", final["best_objective"])

        records = collect_records(model, loaders["val"], device)
        summary = summarize_records(records)
        preview = median_per_condition(records)
        save_example_grid(
            results_dir / f"bottleneck_{dimension}_val.png",
            model,
            preview,
            manifest,
            image_dir,
            device,
            f"Validation preview, bottleneck {dimension}",
        )
        best_epoch = min(final["history"], key=lambda row: row["objective"])["epoch"]
        run = {
            "bottleneck_dimension": dimension,
            "params": params,
            "best_objective": final["best_objective"],
            "best_epoch": best_epoch,
            "epochs_ran": len(final["history"]),
            "val_metrics": summary,
            "checkpoint": str(checkpoint),
            "preview": str(results_dir / f"bottleneck_{dimension}_val.png"),
        }
        runs.append(run)
        print(
            f"done bottleneck {dimension}  best score {final['best_objective']:.4f} at epoch {best_epoch}",
            flush=True,
        )
        del model
        del loaders

    report = {
        "reference_bottleneck_256_validation_objective": reference_score,
        "lower_is_better": True,
        "fixed_settings": FIXED,
        "runs": runs,
    }
    (results_dir / "bottleneck_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote results/task1/bottleneck_comparison.json", flush=True)
    if reference_score is not None:
        print(f"256 reference validation score {reference_score:.4f}", flush=True)
    for run in runs:
        print(f"{run['bottleneck_dimension']} validation score {run['best_objective']:.4f}", flush=True)


if __name__ == "__main__":
    main()
