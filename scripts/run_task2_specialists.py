"""Shared specialist Optuna search, independent final training, and ONNX export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.training.task1 import get_device, seed_everything, study_rows
from genai.training.task2_specialists import (
    BASELINE_PARAMS,
    SEARCH_SPACE,
    SPECIALIST_CORRUPTIONS,
    export_specialist_onnx,
    make_specialist_loaders,
    run_specialist_optuna,
    train_one_specialist,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--trial-epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "task2"
    results_dir = ROOT / "results" / "task2"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("task2-specialists")

    device = get_device()
    print(f"device {device}", flush=True)
    study = run_specialist_optuna(
        output_dir,
        args.trials,
        args.trial_epochs,
        device,
        args.workers,
    )
    best_params = dict(study.best_params)
    print(
        f"best specialist trial {study.best_trial.number} mean score {study.best_value:.4f} params {best_params}",
        flush=True,
    )
    (results_dir / "specialist_optuna_trials.json").write_text(
        json.dumps(study_rows(study), indent=2) + "\n"
    )

    specialist_report = {
        "device": str(device),
        "search_space": SEARCH_SPACE,
        "baseline_enqueued": BASELINE_PARAMS,
        "best_trial": study.best_trial.number,
        "best_trial_mean_objective": study.best_value,
        "selected_params": best_params,
        "specialists": {},
    }
    onnx_diffs = {}
    for offset, corruption in enumerate(SPECIALIST_CORRUPTIONS):
        params = dict(best_params)
        params["train_seed"] = 100 + offset
        seed_everything(params["train_seed"])
        checkpoint = output_dir / f"specialist_{corruption}.pt"
        print(f"final train {corruption} params {params}", flush=True)
        with mlflow.start_run(run_name=f"specialist-final-{corruption}"):
            mlflow.log_params(params)
            mlflow.log_param("final_epochs", args.final_epochs)
            result = train_one_specialist(
                corruption,
                params,
                device,
                epochs=args.final_epochs,
                checkpoint_path=checkpoint,
                num_workers=args.workers,
                patience=args.patience,
            )
            mlflow.log_metric("best_objective", result["best_objective"])
        sample_loader = make_specialist_loaders(params, corruption, num_workers=0, include_test=False)["val"]
        sample = next(iter(sample_loader))["corrupted"][:4]
        onnx_path = output_dir / f"specialist_{corruption}.onnx"
        max_abs = export_specialist_onnx(result["model"], params, onnx_path, sample)
        onnx_diffs[corruption] = max_abs
        specialist_report["specialists"][corruption] = {
            "params": params,
            "best_objective": result["best_objective"],
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_path),
            "onnx_max_abs_diff": max_abs,
        }
        print(f"done {corruption} score {result['best_objective']:.4f} onnx_diff {max_abs:.2e}", flush=True)
        del result["model"]
        if device.type == "mps":
            torch.mps.empty_cache()

    specialist_report["onnx_max_abs_diff"] = onnx_diffs
    (results_dir / "specialist_summary.json").write_text(json.dumps(specialist_report, indent=2) + "\n")
    print(json.dumps({"onnx_max_abs_diff": onnx_diffs}, indent=2))
    if any(value > 1e-4 for value in onnx_diffs.values()):
        raise SystemExit(f"ONNX parity failed: {onnx_diffs}")


if __name__ == "__main__":
    main()
