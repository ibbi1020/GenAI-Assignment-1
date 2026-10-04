"""Optuna search, training, evaluation, and ONNX export for Task 1."""

from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from PIL import Image
from torch.utils.data import DataLoader

from genai.data.corruptions import CONDITIONS
from genai.data.pet_dataset import (
    PETS_ROOT,
    ManifestPetDataset,
    TrainPetDataset,
    load_manifest,
    load_task1_splits,
    tensor_to_pil,
)
from genai.data.pets import assert_disjoint
from genai.models.denoising_autoencoder import DenoisingAutoencoder
from genai.training.losses import per_image_scores, reconstruction_loss, validation_objective

SEARCH_SPACE = {
    "lr": "log uniform from 1e-4 to 2e-3",
    "batch_size": [16, 32, 64],
    "bottleneck_dimension": [16, 32, 64],
    "encoder_channels": [64, 128, 256],
    "norm": ["group", "batch"],
    "alpha": "uniform from 0.5 to 0.95",
}
BASELINE_PARAMS = {
    "lr": 5e-4,
    "batch_size": 32,
    "bottleneck_dimension": 32,
    "encoder_channels": 256,
    "norm": "group",
    "alpha": 0.5,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(params: dict) -> DenoisingAutoencoder:
    return DenoisingAutoencoder(
        encoder_channels=int(params["encoder_channels"]),
        bottleneck_dimension=int(params["bottleneck_dimension"]),
        norm=str(params["norm"]),
    )


def make_loaders(
    params: dict,
    pets_root: Path = PETS_ROOT,
    num_workers: int = 2,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    splits = load_task1_splits(pets_root)
    assert_disjoint(splits["train"], splits["val"], splits["test"])
    processed = pets_root / "processed"
    train_dataset = TrainPetDataset(
        splits["train"],
        processed / "train",
        seed=int(params.get("train_seed", 42)),
    )
    val_dataset = ManifestPetDataset(
        load_manifest(pets_root / "manifests" / "val_corruptions.jsonl"),
        processed / "val",
    )
    generator = torch.Generator()
    generator.manual_seed(42)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            drop_last=True,
            generator=generator,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=int(params["batch_size"]),
            shuffle=False,
            **loader_kwargs,
        ),
    }
    if include_test:
        loaders["test"] = DataLoader(
            ManifestPetDataset(
                load_manifest(pets_root / "manifests" / "test_corruptions.jsonl"),
                processed / "test",
            ),
            batch_size=int(params["batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )
    return loaders


def train_epochs(
    model: DenoisingAutoencoder,
    loaders: dict[str, DataLoader],
    params: dict,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path | None = None,
    patience: int | None = None,
    trial: optuna.Trial | None = None,
    mlflow_active: bool = True,
    min_delta: float = 1e-4,
) -> dict:
    """Train and return the best validation score seen in this run."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    alpha = float(params["alpha"])
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_dataset = loaders["train"].dataset
        if hasattr(train_dataset, "epoch"):
            train_dataset.epoch = epoch
        train_metrics = _run_epoch(model, loaders["train"], optimizer, alpha, device, train=True)
        val_metrics = _run_epoch(model, loaders["val"], None, alpha, device, train=False)
        score = validation_objective(val_metrics["l1"], val_metrics["ssim"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "objective": score}
        history.append(row)
        print(
            f"epoch {epoch:02d}  train L1 {train_metrics['l1']:.4f} SSIM {train_metrics['ssim']:.4f}"
            f"  val L1 {val_metrics['l1']:.4f} SSIM {val_metrics['ssim']:.4f}  score {score:.4f}",
            flush=True,
        )
        if mlflow_active:
            mlflow.log_metrics(
                {
                    "train_l1": train_metrics["l1"],
                    "train_ssim": train_metrics["ssim"],
                    "train_loss": train_metrics["loss"],
                    "val_l1": val_metrics["l1"],
                    "val_ssim": val_metrics["ssim"],
                    "val_objective": score,
                },
                step=epoch,
            )
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if score < best_score - min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"params": params, "state_dict": best_state, "score": best_score}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return {"best_objective": best_score, "history": history}


def collect_records(model: DenoisingAutoencoder, loader: DataLoader, device: torch.device) -> list[dict]:
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            restored = model(batch["corrupted"].to(device))
            scores = per_image_scores(restored, batch["clean"].to(device))
            for index, image_id in enumerate(batch["image_id"]):
                severity = batch["severity"][index] or None
                records.append(
                    {
                        "image_id": image_id,
                        "corruption": batch["corruption"][index],
                        "severity": severity,
                        "l1": float(scores["l1"][index]),
                        "ssim": float(scores["ssim"][index]),
                        "psnr": float(scores["psnr"][index]),
                    }
                )
    return records


def summarize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str | None], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["corruption"], record["severity"]), []).append(record)
    rows = []
    for corruption, severity in _summary_order():
        group = grouped.get((corruption, severity), [])
        if not group:
            continue
        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "count": len(group),
                "l1": _mean(group, "l1"),
                "ssim": _mean(group, "ssim"),
                "psnr": _mean(group, "psnr"),
            }
        )
    rows.append(
        {
            "corruption": "all",
            "severity": None,
            "count": len(records),
            "l1": _mean(records, "l1"),
            "ssim": _mean(records, "ssim"),
            "psnr": _mean(records, "psnr"),
        }
    )
    return rows


def select_representative(records: list[dict]) -> list[dict]:
    """Three typical test cases per condition, near the 25th, 50th, and 75th percentile of L1."""
    chosen = []
    for corruption in CONDITIONS:
        group = sorted(
            (record for record in records if record["corruption"] == corruption),
            key=lambda record: record["l1"],
        )
        if not group:
            continue
        for fraction in (0.25, 0.5, 0.75):
            chosen.append(group[int(round(fraction * (len(group) - 1)))])
    return chosen


def select_failures(records: list[dict]) -> list[dict]:
    """The highest-L1 test case for each of the four conditions."""
    failures = []
    for corruption in CONDITIONS:
        group = [record for record in records if record["corruption"] == corruption]
        if group:
            failures.append(max(group, key=lambda record: record["l1"]))
    return failures


def save_example_grid(
    path: Path,
    model: DenoisingAutoencoder,
    chosen: list[dict],
    manifest_rows: list[dict],
    image_dir: Path,
    device: torch.device,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (row["image_id"], row["corruption"], row.get("severity") or None): row for row in manifest_rows
    }
    columns = ["clean", "corrupted", "restored", "absolute error"]
    figure, axes = plt.subplots(len(chosen), 4, figsize=(10, 2.4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, 4)
    model.eval()
    with torch.no_grad():
        for row_index, record in enumerate(chosen):
            manifest_row = lookup[(record["image_id"], record["corruption"], record["severity"])]
            sample = ManifestPetDataset([manifest_row], image_dir)[0]
            restored = model(sample["corrupted"].unsqueeze(0).to(device)).squeeze(0).cpu()
            error = (restored - sample["clean"]).abs().mean(dim=0).numpy()
            panels = [
                tensor_to_pil(sample["clean"]),
                tensor_to_pil(sample["corrupted"]),
                tensor_to_pil(restored),
                error,
            ]
            severity = record["severity"] or "none"
            axes[row_index, 0].set_ylabel(
                f"{record['corruption']}\n{severity}\nL1 {record['l1']:.3f}",
                fontsize=8,
            )
            for column, panel in enumerate(panels):
                axis = axes[row_index, column]
                if column == 3:
                    axis.imshow(panel, cmap="magma", vmin=0, vmax=max(float(error.max()), 1e-6))
                else:
                    axis.imshow(panel)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(columns[column], fontsize=10)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def export_onnx(model: DenoisingAutoencoder, params: dict, onnx_path: Path, sample: torch.Tensor) -> float:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = build_model(params)
    export_model.load_state_dict(model.state_dict())
    export_model.cpu().eval()
    export_sample = sample.detach().cpu()
    export_kwargs = {
        "input_names": ["corrupted"],
        "output_names": ["restored"],
        "dynamic_axes": {"corrupted": {0: "batch"}, "restored": {0: "batch"}},
        "opset_version": 17,
        "dynamo": False,
    }
    try:
        torch.onnx.export(export_model, export_sample, str(onnx_path), **export_kwargs)
    except TypeError:
        export_kwargs.pop("dynamo")
        torch.onnx.export(export_model, export_sample, str(onnx_path), **export_kwargs)
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = export_model(export_sample).numpy()
    exported = session.run(None, {"corrupted": export_sample.numpy()})[0]
    return float(np.max(np.abs(reference - exported)))


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corruption", "severity", "count", "l1", "ssim", "psnr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "corruption": row["corruption"],
                    "severity": "" if row["severity"] is None else row["severity"],
                    "count": row["count"],
                    "l1": f"{row['l1']:.6f}",
                    "ssim": f"{row['ssim']:.6f}",
                    "psnr": f"{row['psnr']:.4f}",
                }
            )


def run_optuna(
    output_dir: Path,
    n_trials: int,
    trial_epochs: int,
    device: torch.device,
    num_workers: int,
) -> optuna.Study:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir / 'optuna.db'}"
    study = optuna.create_study(
        study_name="task1-universal-restoration",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "bottleneck_dimension": trial.suggest_categorical("bottleneck_dimension", [16, 32, 64]),
            "encoder_channels": trial.suggest_categorical("encoder_channels", [64, 128, 256]),
            "norm": trial.suggest_categorical("norm", ["group", "batch"]),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
        }
        seed_everything(42 + trial.number)
        params["train_seed"] = 42 + trial.number
        print(f"trial {trial.number} params {params}", flush=True)
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_param("trial_epochs", trial_epochs)
            model = build_model(params)
            loaders = make_loaders(params, num_workers=num_workers, include_test=False)
            try:
                result = train_epochs(
                    model,
                    loaders,
                    params,
                    device,
                    epochs=trial_epochs,
                    patience=None,
                    trial=trial,
                )
            finally:
                del model
                del loaders
                if device.type == "mps":
                    torch.mps.empty_cache()
            mlflow.log_metric("best_objective", result["best_objective"])
            return result["best_objective"]

    study.optimize(objective, n_trials=n_trials)
    return study


def study_rows(study: optuna.Study) -> list[dict]:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
            }
        )
    return rows


def _run_epoch(model, loader, optimizer, alpha: float, device: torch.device, train: bool) -> dict:
    model.train(train)
    totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    count = 0
    for batch in loader:
        corrupted = batch["corrupted"].to(device)
        clean = batch["clean"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            restored = model(corrupted)
            loss, l1, ssim = reconstruction_loss(restored, clean, alpha)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = clean.size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["l1"] += float(l1.detach()) * batch_size
        totals["ssim"] += float(ssim.detach()) * batch_size
        count += batch_size
    return {key: value / count for key, value in totals.items()}


def _summary_order() -> list[tuple[str, str | None]]:
    order: list[tuple[str, str | None]] = [("clean", None)]
    for corruption in CONDITIONS:
        if corruption == "clean":
            continue
        for severity in ("low", "medium", "high"):
            order.append((corruption, severity))
    return order


def _mean(records: list[dict], key: str) -> float:
    return sum(record[key] for record in records) / len(records)


def completed_trial_count(study: optuna.Study) -> int:
    return sum(trial.state == TrialState.COMPLETE for trial in study.trials)
