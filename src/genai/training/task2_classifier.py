"""Optuna search, training, evaluation, and ONNX export for the Task 2 classifier."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genai.data.corruptions import CONDITIONS
from genai.data.pet_dataset import (
    PETS_ROOT,
    BalancedBatchSampler,
    ClassifierTrainDataset,
    INDEX_TO_CLASS,
    ManifestPetDataset,
    load_manifest,
    load_task1_splits,
)
from genai.data.pets import assert_disjoint
from genai.models.corruption_classifier import CorruptionClassifier
from genai.training.task1 import get_device, seed_everything, study_rows

SEARCH_SPACE = {
    "lr": "log uniform from 1e-4 to 2e-3",
    "batch_size": [16, 32, 64],
    "base_channels": [16, 32, 64],
    "n_blocks": [3, 4, 5],
    "dropout": "uniform from 0.0 to 0.4",
    "weight_decay": "log uniform from 1e-6 to 1e-3",
}
BASELINE_PARAMS = {
    "lr": 1e-3,
    "batch_size": 32,
    "base_channels": 32,
    "n_blocks": 4,
    "dropout": 0.1,
    "weight_decay": 1e-4,
}


def build_classifier(params: dict) -> CorruptionClassifier:
    return CorruptionClassifier(
        base_channels=int(params["base_channels"]),
        n_blocks=int(params["n_blocks"]),
        dropout=float(params["dropout"]),
    )


def make_classifier_loaders(
    params: dict,
    pets_root: Path = PETS_ROOT,
    num_workers: int = 2,
    include_test: bool = False,
) -> dict:
    splits = load_task1_splits(pets_root)
    assert_disjoint(splits["train"], splits["val"], splits["test"])
    processed = pets_root / "processed"
    train_dataset = ClassifierTrainDataset(
        splits["train"],
        processed / "train",
        seed=int(params.get("train_seed", 42)),
    )
    sampler = BalancedBatchSampler(
        train_dataset.labels,
        batch_size=int(params["batch_size"]),
        seed=int(params.get("train_seed", 42)),
        drop_last=True,
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    loaders = {
        "train": DataLoader(train_dataset, batch_sampler=sampler, **loader_kwargs),
        "train_sampler": sampler,
        "val": DataLoader(
            ManifestPetDataset(
                load_manifest(pets_root / "manifests" / "val_corruptions.jsonl"),
                processed / "val",
            ),
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


def train_classifier_epochs(
    model: CorruptionClassifier,
    loaders: dict,
    params: dict,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path | None = None,
    patience: int | None = None,
    trial: optuna.Trial | None = None,
    mlflow_active: bool = True,
) -> dict:
    """Train and keep the checkpoint with the best validation macro-F1."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []
    sampler = loaders.get("train_sampler")

    for epoch in range(1, epochs + 1):
        train_dataset = loaders["train"].dataset
        if hasattr(train_dataset, "epoch"):
            train_dataset.epoch = epoch
        if sampler is not None and hasattr(sampler, "epoch"):
            sampler.epoch = epoch
        train_metrics = _run_epoch(model, loaders["train"], optimizer, device, train=True)
        val_metrics = evaluate_classifier(model, loaders["val"], device)
        score = val_metrics["macro_f1"]
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "objective": score}
        history.append(row)
        print(
            f"epoch {epoch:02d}  train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f}"
            f"  val macro-F1 {score:.4f} acc {val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if mlflow_active:
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": score,
                },
                step=epoch,
            )
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"params": params, "state_dict": best_state, "score": best_score},
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return {"best_objective": best_score, "history": history}


def evaluate_classifier(
    model: CorruptionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    n_classes = len(CONDITIONS)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["corrupted"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach()) * labels.size(0)
            count += labels.size(0)
            for true, pred in zip(labels.tolist(), preds.tolist()):
                confusion[true, pred] += 1
    metrics = classification_metrics(confusion)
    metrics["loss"] = total_loss / max(count, 1)
    metrics["confusion"] = confusion
    return metrics


def classification_metrics(confusion: np.ndarray) -> dict:
    """Accuracy, macro and per-class precision / recall / F1 from a confusion matrix."""
    n_classes = confusion.shape[0]
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    precision = np.divide(
        correct,
        predicted,
        out=np.zeros(n_classes, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        correct,
        support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(n_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    total = confusion.sum()
    accuracy = float(correct.sum() / total) if total else 0.0
    per_class = []
    for index, name in enumerate(CONDITIONS):
        per_class.append(
            {
                "class": name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "normalized_confusion": normalize_confusion(confusion),
    }


def normalize_confusion(confusion: np.ndarray) -> list[list[float]]:
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion.astype(np.float64),
        row_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums > 0,
    )
    return normalized.tolist()


def accuracy_by_severity(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str | None], list[bool]] = {}
    for record in records:
        key = (record["true_corruption"], record.get("severity"))
        groups.setdefault(key, []).append(record["true_corruption"] == record["pred_corruption"])
    rows = []
    for (corruption, severity), hits in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "count": len(hits),
                "accuracy": sum(hits) / len(hits),
            }
        )
    return rows


def collect_classifier_records(
    model: CorruptionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["corrupted"].to(device))
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            for index, image_id in enumerate(batch["image_id"]):
                pred = int(preds[index])
                true = int(batch["label"][index])
                severity = batch["severity"][index] or None
                records.append(
                    {
                        "image_id": image_id,
                        "true_corruption": INDEX_TO_CLASS[true],
                        "pred_corruption": INDEX_TO_CLASS[pred],
                        "severity": severity,
                        "probabilities": {
                            name: float(probs[index, class_index])
                            for class_index, name in enumerate(CONDITIONS)
                        },
                    }
                )
    return records


def save_confusion_matrix(path: Path, confusion: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = np.array(normalize_confusion(confusion))
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(CONDITIONS)))
    axis.set_yticks(range(len(CONDITIONS)))
    axis.set_xticklabels(CONDITIONS, rotation=30, ha="right")
    axis.set_yticklabels(CONDITIONS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(len(CONDITIONS)):
        for column in range(len(CONDITIONS)):
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.2f}\n({confusion[row, column]})",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if normalized[row, column] < 0.6 else "white",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_confusion_csv(path: Path, confusion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CONDITIONS])
        for row_index, name in enumerate(CONDITIONS):
            writer.writerow([name, *[int(value) for value in confusion[row_index]]])


def export_classifier_onnx(
    model: CorruptionClassifier,
    params: dict,
    onnx_path: Path,
    sample: torch.Tensor,
) -> float:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = build_classifier(params)
    export_model.load_state_dict(model.state_dict())
    export_model.cpu().eval()
    export_sample = sample.detach().cpu()
    export_kwargs = {
        "input_names": ["corrupted"],
        "output_names": ["logits"],
        "dynamic_axes": {"corrupted": {0: "batch"}, "logits": {0: "batch"}},
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


def run_classifier_optuna(
    output_dir: Path,
    n_trials: int,
    trial_epochs: int,
    device: torch.device,
    num_workers: int,
) -> optuna.Study:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir / 'optuna_classifier.db'}"
    study = optuna.create_study(
        study_name="task2-classifier",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # No pruner: in a 4-epoch race the median pruner killed 10 of 14 trials, so the
        # baseline-like settings won by default. Same flaw as the Task 1 search.
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 64]),
            "n_blocks": trial.suggest_categorical("n_blocks", [3, 4, 5]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
        seed_everything(42 + trial.number)
        params["train_seed"] = 42 + trial.number
        print(f"trial {trial.number} params {params}", flush=True)
        with mlflow.start_run(run_name=f"classifier-trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_param("trial_epochs", trial_epochs)
            model = build_classifier(params)
            loaders = make_classifier_loaders(params, num_workers=num_workers, include_test=False)
            try:
                result = train_classifier_epochs(
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
            mlflow.log_metric("best_macro_f1", result["best_objective"])
            return result["best_objective"]

    study.optimize(objective, n_trials=n_trials)
    return study


def _run_epoch(model, loader, optimizer, device: torch.device, train: bool) -> dict:
    model.train(train)
    totals = {"loss": 0.0, "correct": 0.0}
    count = 0
    for batch in loader:
        images = batch["corrupted"].to(device)
        labels = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["correct"] += float((preds == labels).sum().detach())
        count += batch_size
    return {
        "loss": totals["loss"] / count,
        "accuracy": totals["correct"] / count,
    }
