"""Run the Task 1 Optuna search, final training, test evaluation, and ONNX export.

The official test set is loaded only after the final checkpoint has been chosen
from validation.
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
from genai.training.task1 import (
    BASELINE_PARAMS,
    SEARCH_SPACE,
    build_model,
    collect_records,
    completed_trial_count,
    export_onnx,
    get_device,
    make_loaders,
    run_optuna,
    save_example_grid,
    seed_everything,
    select_failures,
    select_representative,
    study_rows,
    summarize_records,
    train_epochs,
    write_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "task1"
    results_dir = ROOT / "results" / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("task1-universal-restoration")

    device = get_device()
    print(f"device {device}", flush=True)
    study = run_optuna(output_dir, args.trials, args.trial_epochs, device, args.workers)
    best_params = study.best_params
    print(f"best trial {study.best_trial.number} score {study.best_value:.4f} params {best_params}", flush=True)
    (results_dir / "optuna_trials.json").write_text(json.dumps(study_rows(study), indent=2) + "\n")

    seed_everything(42)
    best_params = dict(best_params)
    best_params["train_seed"] = 42
    model = build_model(best_params)
    loaders = make_loaders(best_params, num_workers=args.workers, include_test=False)
    checkpoint = output_dir / "universal_autoencoder.pt"
    with mlflow.start_run(run_name="final"):
        mlflow.log_params(best_params)
        mlflow.log_param("final_epochs", args.final_epochs)
        final = train_epochs(
            model,
            loaders,
            best_params,
            device,
            epochs=args.final_epochs,
            checkpoint_path=checkpoint,
            patience=args.patience,
        )
        mlflow.log_metric("best_objective", final["best_objective"])

    test_loader = make_loaders(best_params, num_workers=args.workers, include_test=True)["test"]
    records = collect_records(model, test_loader, device)
    summary = summarize_records(records)
    write_table(results_dir / "test_metrics.csv", summary)
    (results_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    manifest = load_manifest(PETS_ROOT / "manifests" / "test_corruptions.jsonl")
    image_dir = PETS_ROOT / "processed" / "test"
    representative = select_representative(records)
    failures = select_failures(records)
    save_example_grid(
        results_dir / "representative_examples.png",
        model,
        representative,
        manifest,
        image_dir,
        device,
        "Task 1 representative restorations",
    )
    save_example_grid(
        results_dir / "failure_cases.png",
        model,
        failures,
        manifest,
        image_dir,
        device,
        "Task 1 highest-error cases",
    )
    (results_dir / "representative_examples.json").write_text(json.dumps(representative, indent=2) + "\n")
    (results_dir / "failure_cases.json").write_text(json.dumps(failures, indent=2) + "\n")

    sample = next(iter(test_loader))["corrupted"][:4]
    max_abs = export_onnx(model, best_params, output_dir / "universal_autoencoder.onnx", sample)
    report = {
        "device": str(device),
        "search_space": SEARCH_SPACE,
        "baseline_enqueued": BASELINE_PARAMS,
        "completed_trials": completed_trial_count(study),
        "best_trial": study.best_trial.number,
        "best_trial_objective": study.best_value,
        "selected_params": best_params,
        "final_validation_objective": final["best_objective"],
        "final_history": final["history"],
        "test_summary": summary,
        "onnx_max_abs_diff": max_abs,
        "checkpoint": str(checkpoint),
        "onnx": str(output_dir / "universal_autoencoder.onnx"),
    }
    (results_dir / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test": summary, "onnx_max_abs_diff": max_abs}, indent=2))
    if max_abs > 1e-4:
        raise SystemExit(f"ONNX output differs from PyTorch by {max_abs}")


if __name__ == "__main__":
    main()
