"""Shared Optuna search and independent training for Task 2 specialists."""

from __future__ import annotations

from pathlib import Path

import mlflow
import optuna
import torch
from torch.utils.data import DataLoader

from genai.data.corruptions import CONDITIONS
from genai.data.pet_dataset import (
    PETS_ROOT,
    ManifestPetDataset,
    TrainPetDataset,
    filter_manifest,
    load_manifest,
    load_task1_splits,
)
from genai.data.pets import assert_disjoint
from genai.training.task1 import (
    build_model,
    export_onnx,
    seed_everything,
    study_rows,
    train_epochs,
)

SPECIALIST_CORRUPTIONS: tuple[str, ...] = (
    "salt_and_pepper",
    "gaussian_blur",
    "rectangular_occlusion",
)

SEARCH_SPACE = {
    "lr": "log uniform from 1e-4 to 2e-3",
    "batch_size": [16, 32, 64],
    "bottleneck_dimension": [16, 32, 64],
    "encoder_channels": [32, 64, 128],
    "norm": ["group", "batch"],
    "alpha": "uniform from 0.5 to 0.95",
}
BASELINE_PARAMS = {
    "lr": 5e-4,
    "batch_size": 32,
    "bottleneck_dimension": 32,
    "encoder_channels": 64,
    "norm": "group",
    "alpha": 0.5,
}


def make_specialist_loaders(
    params: dict,
    corruption: str,
    pets_root: Path = PETS_ROOT,
    num_workers: int = 2,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    if corruption not in SPECIALIST_CORRUPTIONS:
        raise ValueError(f"specialist corruption must be one of {SPECIALIST_CORRUPTIONS}, got {corruption}")
    splits = load_task1_splits(pets_root)
    assert_disjoint(splits["train"], splits["val"], splits["test"])
    processed = pets_root / "processed"
    train_dataset = TrainPetDataset(
        splits["train"],
        processed / "train",
        seed=int(params.get("train_seed", 42)),
        corruption=corruption,
    )
    val_rows = filter_manifest(
        load_manifest(pets_root / "manifests" / "val_corruptions.jsonl"),
        [corruption],
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    generator = torch.Generator()
    generator.manual_seed(42)
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
            ManifestPetDataset(val_rows, processed / "val"),
            batch_size=int(params["batch_size"]),
            shuffle=False,
            **loader_kwargs,
        ),
    }
    if include_test:
        test_rows = filter_manifest(
            load_manifest(pets_root / "manifests" / "test_corruptions.jsonl"),
            [corruption],
        )
        loaders["test"] = DataLoader(
            ManifestPetDataset(test_rows, processed / "test"),
            batch_size=int(params["batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )
    return loaders


def train_one_specialist(
    corruption: str,
    params: dict,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    num_workers: int,
    patience: int | None = None,
    trial: optuna.Trial | None = None,
    mlflow_active: bool = True,
) -> dict:
    model = build_model(params)
    loaders = make_specialist_loaders(
        params,
        corruption,
        num_workers=num_workers,
        include_test=False,
    )
    try:
        result = train_epochs(
            model,
            loaders,
            params,
            device,
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            patience=patience,
            trial=trial,
            mlflow_active=mlflow_active,
        )
    finally:
        del loaders
        if device.type == "mps":
            torch.mps.empty_cache()
    return {"model": model, **result}


def run_specialist_optuna(
    output_dir: Path,
    n_trials: int,
    trial_epochs: int,
    device: torch.device,
    num_workers: int,
) -> optuna.Study:
    """Shared architecture search. No median pruner: short races favour tiny nets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir / 'optuna_specialists.db'}"
    study = optuna.create_study(
        study_name="task2-specialists",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
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
            "bottleneck_dimension": trial.suggest_categorical("bottleneck_dimension", [16, 32, 64]),
            "encoder_channels": trial.suggest_categorical("encoder_channels", [32, 64, 128]),
            "norm": trial.suggest_categorical("norm", ["group", "batch"]),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
        }
        print(f"specialist trial {trial.number} params {params}", flush=True)
        scores = []
        with mlflow.start_run(run_name=f"specialist-trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_param("trial_epochs", trial_epochs)
            for offset, corruption in enumerate(SPECIALIST_CORRUPTIONS):
                local = dict(params)
                local["train_seed"] = 42 + trial.number * 10 + offset
                seed_everything(local["train_seed"])
                checkpoint = output_dir / f"trial_{trial.number}_{corruption}.pt"
                print(f"  training {corruption}", flush=True)
                result = train_one_specialist(
                    corruption,
                    local,
                    device,
                    epochs=trial_epochs,
                    checkpoint_path=checkpoint,
                    num_workers=num_workers,
                    patience=None,
                    trial=None,
                    mlflow_active=False,
                )
                scores.append(result["best_objective"])
                mlflow.log_metric(f"{corruption}_objective", result["best_objective"])
                del result["model"]
                if device.type == "mps":
                    torch.mps.empty_cache()
            mean_score = float(sum(scores) / len(scores))
            mlflow.log_metric("mean_objective", mean_score)
            return mean_score

    study.optimize(objective, n_trials=n_trials)
    return study


def export_specialist_onnx(
    model,
    params: dict,
    onnx_path: Path,
    sample: torch.Tensor,
) -> float:
    return export_onnx(model, params, onnx_path, sample)
