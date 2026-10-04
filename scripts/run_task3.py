"""Train the Task 3 soft mixture with one fixed hyperparameter set.

Copies the Task 2 classifier into the gate and the three specialists in as
experts, warms up the gate, then fine-tunes the whole mixture. Exits before
any training if a Task 2 checkpoint is missing. The test split is scored once,
after the saved epoch is chosen on validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.training.task1 import get_device, seed_everything
from genai.training.task3 import (
    FIXED_PARAMS,
    average_gate_weights,
    collect_mixture_records,
    expert_activity,
    export_mixture_onnx,
    initialize_from_task2,
    make_mixture_loaders,
    missing_task2_checkpoints,
    pick_routing_examples,
    save_routing_examples,
    save_weight_heatmap,
    summarize_reconstruction,
    train_mixture,
    write_gate_weight_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    missing = missing_task2_checkpoints()
    if missing:
        listed = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(
            "Task 3 copies the Task 2 classifier and specialists. These files are not ready yet:\n"
            f"{listed}"
        )

    params = dict(FIXED_PARAMS)
    if args.warmup_epochs is not None:
        params["warmup_epochs"] = args.warmup_epochs
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.patience is not None:
        params["patience"] = args.patience
    output_dir = ROOT / "outputs" / "task3"
    results_dir = ROOT / "results" / "task3"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("task3-soft-mixture")

    device = get_device()
    print(f"device {device}", flush=True)
    print(f"fixed params {params}", flush=True)
    seed_everything(int(params["train_seed"]))
    model, init_meta = initialize_from_task2(float(params["temperature"]))
    loaders = make_mixture_loaders(params, num_workers=args.workers, include_test=False)
    checkpoint = output_dir / "soft_mixture.pt"
    with mlflow.start_run(run_name="mixture-final"):
        mlflow.log_params(params)
        final = train_mixture(
            model,
            loaders,
            params,
            device,
            warmup_epochs=int(params["warmup_epochs"]),
            joint_epochs=int(params["epochs"]),
            checkpoint_path=checkpoint,
            patience=int(params["patience"]),
            min_delta=float(params["min_delta"]),
            init_meta=init_meta,
        )
        mlflow.log_metric("best_objective", final["best_objective"])

    sample = next(iter(loaders["val"]))["corrupted"][:4]
    onnx_path = output_dir / "soft_mixture.onnx"
    onnx_diff = export_mixture_onnx(model, onnx_path, sample)
    print(f"onnx max abs diff {onnx_diff}", flush=True)

    test_loader = make_mixture_loaders(params, num_workers=args.workers, include_test=True)["test"]
    records = collect_mixture_records(model, test_loader, device)
    gate_rows = average_gate_weights(records)
    reconstruction = summarize_reconstruction(records)
    activity = expert_activity(records)
    examples = pick_routing_examples(records)
    write_gate_weight_csv(results_dir / "gate_weights.csv", gate_rows)
    save_weight_heatmap(results_dir / "weight_heatmap.png", gate_rows)
    save_routing_examples(
        results_dir / "dominant_examples.png",
        model,
        test_loader,
        device,
        examples["dominant"],
        "One expert dominates",
    )
    save_routing_examples(
        results_dir / "shared_examples.png",
        model,
        test_loader,
        device,
        examples["shared"],
        "Weight is shared",
    )

    summary = {
        "split_scored": "test",
        "selected_params": params,
        "validation_objective": final["best_objective"],
        "gate_weights": gate_rows,
        "reconstruction": reconstruction,
        "expert_activity": activity,
        "onnx_max_abs_diff": onnx_diff,
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"validation_objective": final["best_objective"], "onnx_max_abs_diff": onnx_diff}, indent=2))
    if any(value > 1e-4 for value in onnx_diff.values()):
        raise SystemExit(f"ONNX parity failed: {onnx_diff}")


if __name__ == "__main__":
    main()
