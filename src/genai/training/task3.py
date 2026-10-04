"""Warm-up, joint fine-tune, evaluation, and ONNX export for Task 3.

Every run starts from the Task 2 classifier and the three specialist
checkpoints. Nothing here is trained from random weights. The hyperparameters
are fixed: there is no search.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genai.data.corruptions import CONDITIONS
from genai.data.pet_dataset import REPO_ROOT
from genai.models.soft_mixture import SoftMixture, freeze_experts, unfreeze_experts
from genai.training.losses import per_image_scores, structural_similarity, validation_objective
from genai.training.task1 import build_model
from genai.training.task2_classifier import build_classifier, make_classifier_loaders
from genai.training.task2_specialists import SPECIALIST_CORRUPTIONS

COLLAPSE_MEAN_WEIGHT = 0.85
INACTIVE_MEAN_WEIGHT = 0.05
DOMINANT_WEIGHT = 0.70
SHARED_WEIGHT = 0.55
TARGET_BRANCH_WEIGHT = 0.25

# One config. `lr` is recorded with the others; the two stages use warmup_lr
# and finetune_lr. lambda_l1 and lambda_ssim are separate weights, not a mix
# that has to sum to 1.
FIXED_PARAMS = {
    "lr": 3e-3,
    "batch_size": 64,
    "temperature": 5.0,
    "lambda_l1": 0.8,
    "lambda_ssim": 0.1,
    "lambda_cls": 0.01,
    "lambda_balance": 0.01,
    "epochs": 200,
    "patience": 20,
    "min_delta": 1e-3,
    "warmup_epochs": 20,
    "warmup_lr": 1e-3,
    "finetune_lr": 1e-4,
    "train_seed": 42,
}


def task2_checkpoint_paths(task2_dir: Path | None = None) -> tuple[Path, dict[str, Path]]:
    root = task2_dir or (REPO_ROOT / "outputs" / "task2")
    classifier = root / "corruption_classifier.pt"
    specialists = {name: root / f"specialist_{name}.pt" for name in SPECIALIST_CORRUPTIONS}
    return classifier, specialists


def missing_task2_checkpoints(task2_dir: Path | None = None) -> list[Path]:
    classifier, specialists = task2_checkpoint_paths(task2_dir)
    return [path for path in (classifier, *specialists.values()) if not path.exists()]


def initialize_from_task2(
    temperature: float,
    task2_dir: Path | None = None,
    classifier_path: Path | None = None,
    specialist_paths: dict[str, Path] | None = None,
) -> tuple[SoftMixture, dict]:
    """Copy Task 2 weights into a new mixture. Does not train."""
    default_classifier, default_specialists = task2_checkpoint_paths(task2_dir)
    classifier_path = classifier_path or default_classifier
    specialist_paths = specialist_paths or default_specialists
    paths = [classifier_path, *specialist_paths.values()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        listed = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Task 3 needs the Task 2 checkpoints. Missing: {listed}")

    classifier_payload = torch.load(classifier_path, map_location="cpu", weights_only=False)
    specialist_payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in specialist_paths.items()
    }
    gate_params = classifier_payload["params"]
    specialist_params = {name: payload["params"] for name, payload in specialist_payloads.items()}
    model = build_mixture(gate_params, specialist_params, temperature)
    model.gate.load_state_dict(classifier_payload["state_dict"])
    for name, payload in specialist_payloads.items():
        model.experts[name].load_state_dict(payload["state_dict"])
    meta = {"gate_params": gate_params, "specialist_params": specialist_params}
    return model, meta


def build_mixture(
    gate_params: dict,
    specialist_params: dict[str, dict],
    temperature: float,
) -> SoftMixture:
    gate = build_classifier(gate_params)
    specialists = {name: build_model(specialist_params[name]) for name in SPECIALIST_CORRUPTIONS}
    return SoftMixture(gate, specialists, temperature)


def balance_penalty(weights: torch.Tensor) -> torch.Tensor:
    """Sum of squared gaps between each branch's batch-mean weight and 0.25.

    The mean is over the batch, so the penalty is only meaningful when the
    batch contains every class in equal count.
    """
    if weights.ndim != 2 or weights.size(1) != len(CONDITIONS):
        raise ValueError(f"expected weights shaped (batch, 4), got {tuple(weights.shape)}")
    mean_weight = weights.mean(dim=0)
    return (mean_weight - TARGET_BRANCH_WEIGHT).pow(2).sum()


def mixture_loss(
    restored: torch.Tensor,
    clean: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint loss: λ_L1 * L1 + λ_SSIM * (1 - SSIM) + λ_cls * CE + λ_balance * balance.

    Cross-entropy uses the raw gate logits. Temperature only scales the
    softmax that mixes the experts, so it does not also rescale this term.
    """
    lambda_l1 = float(params["lambda_l1"])
    lambda_ssim = float(params["lambda_ssim"])
    if lambda_l1 < 0 or lambda_ssim < 0:
        raise ValueError(f"loss weights must be non-negative, got L1 {lambda_l1} SSIM {lambda_ssim}")
    l1 = (restored - clean).abs().mean()
    ssim = structural_similarity(restored, clean).mean()
    classification = F.cross_entropy(logits, labels)
    balance = balance_penalty(weights)
    total = (
        lambda_l1 * l1
        + lambda_ssim * (1.0 - ssim)
        + float(params["lambda_cls"]) * classification
        + float(params["lambda_balance"]) * balance
    )
    return total, {
        "loss": total,
        "l1": l1,
        "ssim": ssim,
        "cls": classification,
        "balance": balance,
    }


def make_mixture_loaders(
    params: dict,
    num_workers: int = 2,
    include_test: bool = False,
) -> dict:
    """Class-balanced training loader. Test is off unless a final eval asks for it."""
    loader_params = {
        "batch_size": int(params["batch_size"]),
        "train_seed": int(params.get("train_seed", 42)),
    }
    return make_classifier_loaders(
        loader_params,
        num_workers=num_workers,
        include_test=include_test,
    )


def train_mixture(
    model: SoftMixture,
    loaders: dict,
    params: dict,
    device: torch.device,
    warmup_epochs: int,
    joint_epochs: int,
    checkpoint_path: Path | None = None,
    patience: int | None = None,
    min_delta: float = 1e-3,
    mlflow_active: bool = True,
    init_meta: dict | None = None,
) -> dict:
    """Freeze the experts and train the gate, then fine-tune everything.

    The checkpoint is the joint-stage weights with the best validation score.
    The post-warm-up weights are the first candidate, before any joint step.
    """
    if warmup_epochs < 1:
        raise ValueError(f"warmup_epochs must be at least 1, got {warmup_epochs}")
    if joint_epochs < 1:
        raise ValueError(f"joint_epochs must be at least 1, got {joint_epochs}")

    model.to(device)
    history: list[dict] = []
    step = 0
    warmup_lr = float(params["warmup_lr"])
    finetune_lr = float(params["finetune_lr"])

    freeze_experts(model)
    warmup_optimizer = torch.optim.Adam(model.gate.parameters(), lr=warmup_lr)
    for epoch in range(1, warmup_epochs + 1):
        step += 1
        row = _run_epoch_pair(
            model,
            loaders,
            warmup_optimizer,
            params,
            device,
            epoch=epoch,
            step=step,
            stage="warmup",
            mlflow_active=mlflow_active,
        )
        history.append(row)

    unfreeze_experts(model)
    warmed = _run_epoch(model, loaders["val"], None, params, device, train=False)
    best_score = validation_objective(warmed["l1"], warmed["ssim"])
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    print(
        f"after warm-up  val L1 {warmed['l1']:.4f} SSIM {warmed['ssim']:.4f}  score {best_score:.4f}",
        flush=True,
    )

    joint_optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=finetune_lr,
    )
    for epoch in range(1, joint_epochs + 1):
        step += 1
        row = _run_epoch_pair(
            model,
            loaders,
            joint_optimizer,
            params,
            device,
            epoch=epoch,
            step=step,
            stage="joint",
            mlflow_active=mlflow_active,
        )
        history.append(row)
        score = row["objective"]
        if score < best_score - min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            _save_checkpoint(checkpoint_path, params, best_state, best_score, init_meta, warmup_lr)
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(f"early stop at joint epoch {epoch}", flush=True)
                break

    if checkpoint_path is not None and not checkpoint_path.exists():
        _save_checkpoint(checkpoint_path, params, best_state, best_score, init_meta, warmup_lr)
    model.load_state_dict(best_state)
    return {"best_objective": best_score, "history": history, "warmup_lr": warmup_lr}


def collect_mixture_records(
    model: SoftMixture,
    loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            corrupted = batch["corrupted"].to(device)
            clean = batch["clean"].to(device)
            restored, weights = model(corrupted)
            scores = per_image_scores(restored, clean)
            for index, image_id in enumerate(batch["image_id"]):
                per_image = {
                    name: float(weights[index, class_index])
                    for class_index, name in enumerate(CONDITIONS)
                }
                severity = batch["severity"][index] or None
                records.append(
                    {
                        "image_id": image_id,
                        "corruption": batch["corruption"][index],
                        "severity": severity,
                        "weights": per_image,
                        "dominant": max(per_image, key=per_image.get),
                        "l1": float(scores["l1"][index]),
                        "ssim": float(scores["ssim"][index]),
                        "psnr": float(scores["psnr"][index]),
                    }
                )
    return records


def average_gate_weights(records: list[dict]) -> list[dict]:
    """Mean gate weight for every true corruption and severity present in ``records``."""
    return _summarize(
        records,
        lambda group: {"weights": _mean_weights(group)},
    )


def summarize_reconstruction(records: list[dict]) -> list[dict]:
    if not records:
        return []
    rows = _summarize(
        records,
        lambda group: {
            "l1": _mean(group, "l1"),
            "ssim": _mean(group, "ssim"),
            "psnr": _mean(group, "psnr"),
        },
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


def expert_activity(records: list[dict]) -> dict:
    """Which experts are unused, and which ones take another class's images."""
    if not records:
        return {
            "mean_weights": {name: 0.0 for name in CONDITIONS},
            "inactive": [],
            "takes_foreign_inputs": [],
        }
    overall = _mean_weights(records)
    foreign = []
    for corruption in CONDITIONS:
        subset = [record for record in records if record["corruption"] == corruption]
        if not subset:
            continue
        means = _mean_weights(subset)
        matching = means[corruption]
        for expert, value in means.items():
            if expert == corruption or value <= matching:
                continue
            foreign.append(
                {
                    "true_corruption": corruption,
                    "expert": expert,
                    "mean_weight_on_true_expert": matching,
                    "mean_weight_on_this_expert": value,
                }
            )
    return {
        "mean_weights": overall,
        "inactive": [name for name, value in overall.items() if value < INACTIVE_MEAN_WEIGHT],
        "takes_foreign_inputs": foreign,
    }


def pick_routing_examples(records: list[dict], limit: int = 4) -> dict[str, list[dict]]:
    """A few images where one expert owns the mix, and a few where it is shared."""
    dominant = [record for record in records if max(record["weights"].values()) >= DOMINANT_WEIGHT]
    shared = [record for record in records if max(record["weights"].values()) <= SHARED_WEIGHT]
    return {
        "dominant": _spread(dominant, limit, reverse=True),
        "shared": _spread(shared, limit, reverse=False),
    }


def write_gate_weight_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["corruption", "severity", "count", *CONDITIONS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "weights" not in row:
                continue
            writer.writerow(
                {
                    "corruption": row["corruption"],
                    "severity": "" if row["severity"] is None else row["severity"],
                    "count": row["count"],
                    **{name: f"{row['weights'][name]:.6f}" for name in CONDITIONS},
                }
            )


def save_weight_heatmap(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = [row for row in rows if row["corruption"] != "all" and "weights" in row]
    if not plotted:
        return
    matrix = np.array(
        [[row["weights"][name] for name in CONDITIONS] for row in plotted],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(7, max(3, 0.45 * len(plotted))))
    image = axis.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(CONDITIONS)))
    axis.set_xticklabels(CONDITIONS, rotation=30, ha="right")
    axis.set_yticks(range(len(plotted)))
    axis.set_yticklabels(
        [
            f"{row['corruption']} / {row['severity'] or 'none'}"
            for row in plotted
        ]
    )
    axis.set_xlabel("Gate branch")
    axis.set_title("Mean gate weight")
    for row_index, row in enumerate(plotted):
        for column, name in enumerate(CONDITIONS):
            value = row["weights"][name]
            axis.text(
                column,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if value < 0.6 else "white",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_routing_examples(
    path: Path,
    model: SoftMixture,
    loader: DataLoader,
    device: torch.device,
    picks: list[dict],
    title: str,
) -> None:
    """Input, restoration, and the four weights for the chosen images."""
    if not picks:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genai.data.pet_dataset import tensor_to_pil

    wanted = {
        (record["image_id"], record["corruption"], record["severity"]): record for record in picks
    }
    found: dict[tuple, tuple] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for index, image_id in enumerate(batch["image_id"]):
                severity = batch["severity"][index] or None
                key = (image_id, batch["corruption"][index], severity)
                if key not in wanted or key in found:
                    continue
                image = batch["corrupted"][index : index + 1].to(device)
                restored, weights = model(image)
                found[key] = (
                    tensor_to_pil(image[0].cpu()),
                    tensor_to_pil(restored[0].cpu()),
                    {
                        name: float(weights[0, class_index])
                        for class_index, name in enumerate(CONDITIONS)
                    },
                )
            if len(found) == len(wanted):
                break
    ordered = [found[key] for key in wanted if key in found]
    if not ordered:
        return
    figure, axes = plt.subplots(len(ordered), 2, figsize=(6, 3 * len(ordered)))
    if len(ordered) == 1:
        axes = np.array([axes])
    for row_index, (corrupted, restored, weights) in enumerate(ordered):
        axes[row_index, 0].imshow(corrupted)
        axes[row_index, 1].imshow(restored)
        caption = "  ".join(f"{name[:4]} {weights[name]:.2f}" for name in CONDITIONS)
        axes[row_index, 0].set_ylabel(caption, fontsize=8)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Restoration")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def export_mixture_onnx(model: SoftMixture, onnx_path: Path, sample: torch.Tensor) -> dict[str, float]:
    """Export the full mixture. Returns max absolute error for both outputs."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    unfreeze_experts(export_model)
    export_sample = sample.detach().cpu()
    export_kwargs = {
        "input_names": ["corrupted"],
        "output_names": ["restored", "weights"],
        "dynamic_axes": {
            "corrupted": {0: "batch"},
            "restored": {0: "batch"},
            "weights": {0: "batch"},
        },
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
        reference_image, reference_weights = export_model(export_sample)
    exported_image, exported_weights = session.run(None, {"corrupted": export_sample.numpy()})
    return {
        "restored": float(np.max(np.abs(reference_image.numpy() - exported_image))),
        "weights": float(np.max(np.abs(reference_weights.numpy() - exported_weights))),
    }


def _run_epoch_pair(
    model: SoftMixture,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    params: dict,
    device: torch.device,
    epoch: int,
    step: int,
    stage: str,
    mlflow_active: bool,
) -> dict:
    _set_loader_epoch(loaders["train"], epoch)
    train_metrics = _run_epoch(model, loaders["train"], optimizer, params, device, train=True)
    val_metrics = _run_epoch(model, loaders["val"], None, params, device, train=False)
    score = validation_objective(val_metrics["l1"], val_metrics["ssim"])
    print(
        f"{stage} epoch {epoch:02d}  train L1 {train_metrics['l1']:.4f} "
        f"SSIM {train_metrics['ssim']:.4f} cls {train_metrics['cls']:.4f} "
        f"balance {train_metrics['balance']:.4f}  "
        f"val L1 {val_metrics['l1']:.4f} SSIM {val_metrics['ssim']:.4f}  score {score:.4f}",
        flush=True,
    )
    if mlflow_active:
        metrics = {
            "train_loss": train_metrics["loss"],
            "train_l1": train_metrics["l1"],
            "train_ssim": train_metrics["ssim"],
            "train_cls": train_metrics["cls"],
            "train_balance": train_metrics["balance"],
            "val_l1": val_metrics["l1"],
            "val_ssim": val_metrics["ssim"],
            "val_objective": score,
        }
        for name, value in train_metrics["mean_weights"].items():
            metrics[f"train_weight_{name}"] = value
        mlflow.log_metrics(metrics, step=step)
    if _collapsed(train_metrics["mean_weights"]):
        print(
            f"{stage} epoch {epoch:02d} collapsed onto one expert: {train_metrics['mean_weights']}",
            flush=True,
        )
    return {
        "stage": stage,
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
        "objective": score,
    }


def _run_epoch(model, loader, optimizer, params: dict, device: torch.device, train: bool) -> dict:
    _set_mode(model, train)
    totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "cls": 0.0, "balance": 0.0}
    weight_sum = torch.zeros(len(CONDITIONS), device=device)
    count = 0
    for batch in loader:
        corrupted = batch["corrupted"].to(device)
        clean = batch["clean"].to(device)
        labels = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            restored, weights, logits = model.components(corrupted)
            loss, parts = mixture_loss(restored, clean, logits, labels, weights, params)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    1.0,
                )
                optimizer.step()
        batch_size = clean.size(0)
        for key in totals:
            totals[key] += float(parts[key].detach()) * batch_size
        weight_sum += weights.detach().sum(dim=0)
        count += batch_size
    mean_weights = {
        name: float(weight_sum[index] / count) for index, name in enumerate(CONDITIONS)
    }
    return {key: value / count for key, value in totals.items()} | {"mean_weights": mean_weights}


def _set_mode(model: SoftMixture, train: bool) -> None:
    model.gate.train(train)
    for expert in model.experts.values():
        trainable = any(parameter.requires_grad for parameter in expert.parameters())
        expert.train(train and trainable)


def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = loader.dataset
    if hasattr(dataset, "epoch"):
        dataset.epoch = epoch
    sampler = loader.batch_sampler
    if hasattr(sampler, "epoch"):
        sampler.epoch = epoch


def _collapsed(mean_weights: dict[str, float]) -> bool:
    return max(mean_weights.values()) > COLLAPSE_MEAN_WEIGHT


def _save_checkpoint(
    path: Path | None,
    params: dict,
    state: dict,
    score: float,
    init_meta: dict | None,
    warmup_lr: float,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {**params, "warmup_lr": warmup_lr},
        "state_dict": state,
        "score": score,
    }
    if init_meta is not None:
        payload.update(init_meta)
    torch.save(payload, path)


def _summarize(records: list[dict], value_fn) -> list[dict]:
    grouped: dict[tuple[str, str | None], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["corruption"], record["severity"]), []).append(record)
    rows = []
    for corruption, severity in _ordered_keys(set(grouped)):
        group = grouped[(corruption, severity)]
        row = {"corruption": corruption, "severity": severity, "count": len(group)}
        row.update(value_fn(group))
        rows.append(row)
    return rows


def _ordered_keys(keys: set[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    preferred: list[tuple[str, str | None]] = [("clean", None)]
    for name in CONDITIONS[1:]:
        preferred.append((name, None))
        for severity in ("low", "medium", "high"):
            preferred.append((name, severity))
    ordered = [key for key in preferred if key in keys]
    leftover = sorted(keys - set(ordered), key=lambda item: (item[0], item[1] or ""))
    return ordered + leftover


def _mean_weights(records: list[dict]) -> dict[str, float]:
    return {
        name: sum(record["weights"][name] for record in records) / len(records)
        for name in CONDITIONS
    }


def _mean(records: list[dict], key: str) -> float:
    return sum(record[key] for record in records) / len(records)


def _spread(records: list[dict], limit: int, reverse: bool) -> list[dict]:
    ordered = sorted(records, key=lambda record: max(record["weights"].values()), reverse=reverse)
    chosen: list[dict] = []
    seen: set[str] = set()
    for record in ordered:
        if record["corruption"] in seen:
            continue
        chosen.append(record)
        seen.add(record["corruption"])
        if len(chosen) == limit:
            return chosen
    for record in ordered:
        if record in chosen:
            continue
        chosen.append(record)
        if len(chosen) == limit:
            break
    return chosen


__all__ = [
    "FIXED_PARAMS",
    "average_gate_weights",
    "collect_mixture_records",
    "expert_activity",
    "export_mixture_onnx",
    "initialize_from_task2",
    "make_mixture_loaders",
    "missing_task2_checkpoints",
    "mixture_loss",
    "pick_routing_examples",
    "train_mixture",
]
