"""Run the Task 2 classifier Optuna search, final training, test eval, and ONNX export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.training.task1 import get_device, seed_everything, study_rows
from genai.training.task2_classifier import (
    BASELINE_PARAMS,
    SEARCH_SPACE,
    accuracy_by_severity,
    build_classifier,
    collect_classifier_records,
    evaluate_classifier,
    export_classifier_onnx,
    make_classifier_loaders,
    run_classifier_optuna,
    save_confusion_matrix,
    train_classifier_epochs,
    write_confusion_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "task2"
    results_dir = ROOT / "results" / "task2"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("task2-classifier")

    device = get_device()
    print(f"device {device}", flush=True)
    study = run_classifier_optuna(
        output_dir,
        args.trials,
        args.trial_epochs,
        device,
        args.workers,
    )
    best_params = dict(study.best_params)
    print(
        f"best trial {study.best_trial.number} macro-F1 {study.best_value:.4f} params {best_params}",
        flush=True,
    )
    (results_dir / "classifier_optuna_trials.json").write_text(
        json.dumps(study_rows(study), indent=2) + "\n"
    )

    seed_everything(42)
    best_params["train_seed"] = 42
    model = build_classifier(best_params)
    loaders = make_classifier_loaders(best_params, num_workers=args.workers, include_test=False)
    checkpoint = output_dir / "corruption_classifier.pt"
    with mlflow.start_run(run_name="classifier-final"):
        mlflow.log_params(best_params)
        mlflow.log_param("final_epochs", args.final_epochs)
        final = train_classifier_epochs(
            model,
            loaders,
            best_params,
            device,
            epochs=args.final_epochs,
            checkpoint_path=checkpoint,
            patience=args.patience,
        )
        mlflow.log_metric("best_macro_f1", final["best_objective"])

    test_loader = make_classifier_loaders(best_params, num_workers=args.workers, include_test=True)["test"]
    test_metrics = evaluate_classifier(model, test_loader, device)
    records = collect_classifier_records(model, test_loader, device)
    severity_rows = accuracy_by_severity(records)
    confusion = test_metrics["confusion"]
    write_confusion_csv(results_dir / "confusion_matrix.csv", confusion)
    save_confusion_matrix(
        results_dir / "confusion_matrix.png",
        confusion,
        "Task 2 classifier normalized confusion matrix",
    )

    sample = next(iter(test_loader))["corrupted"][:4]
    max_abs = export_classifier_onnx(
        model,
        best_params,
        output_dir / "corruption_classifier.onnx",
        sample,
    )
    report = {
        "device": str(device),
        "search_space": SEARCH_SPACE,
        "baseline_enqueued": BASELINE_PARAMS,
        "best_trial": study.best_trial.number,
        "best_trial_macro_f1": study.best_value,
        "selected_params": best_params,
        "final_validation_macro_f1": final["best_objective"],
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class": test_metrics["per_class"],
            "normalized_confusion": test_metrics["normalized_confusion"],
        },
        "accuracy_by_severity": severity_rows,
        "onnx_max_abs_diff": max_abs,
        "checkpoint": str(checkpoint),
        "onnx": str(output_dir / "corruption_classifier.onnx"),
    }
    (results_dir / "classifier_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test": report["test_metrics"], "onnx_max_abs_diff": max_abs}, indent=2))
    if max_abs > 1e-4:
        raise SystemExit(f"ONNX output differs from PyTorch: max abs {max_abs}")


if __name__ == "__main__":
    main()
