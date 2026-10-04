"""Hard routing: classifier picks a specialist, clean uses an identity bypass."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genai.data.corruptions import CONDITIONS
from genai.data.pet_dataset import INDEX_TO_CLASS
from genai.models.corruption_classifier import CorruptionClassifier
from genai.models.denoising_autoencoder import DenoisingAutoencoder
from genai.training.losses import per_image_scores
from genai.training.task1 import build_model
from genai.training.task2_classifier import build_classifier
from genai.training.task2_specialists import SPECIALIST_CORRUPTIONS


class HardRouter:
    """Oracle or predicted routing over three specialists plus a clean bypass."""

    def __init__(
        self,
        classifier: CorruptionClassifier | None,
        specialists: dict[str, DenoisingAutoencoder],
    ):
        missing = set(SPECIALIST_CORRUPTIONS) - set(specialists)
        if missing:
            raise ValueError(f"missing specialists for {sorted(missing)}")
        self.classifier = classifier
        self.specialists = specialists

    def restore(
        self,
        corrupted: torch.Tensor,
        route: str | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return restored images, route indices, and class probabilities.

        Pass ``labels`` for oracle routing. Pass nothing for predicted routing.
        ``route`` is unused; kept for call-site clarity in tests.
        """
        del route
        batch = corrupted.size(0)
        device = corrupted.device
        if labels is not None:
            route_indices = labels.to(device)
            probs = F.one_hot(route_indices, num_classes=len(CONDITIONS)).float()
        else:
            if self.classifier is None:
                raise ValueError("predicted routing needs a classifier")
            logits = self.classifier(corrupted)
            probs = F.softmax(logits, dim=1)
            route_indices = logits.argmax(dim=1)

        restored = corrupted.clone()
        for class_index, name in enumerate(CONDITIONS):
            mask = route_indices == class_index
            if not bool(mask.any()):
                continue
            if name == "clean":
                continue
            specialist = self.specialists[name]
            restored[mask] = specialist(corrupted[mask])
        return restored, route_indices, probs


def load_hard_router(
    classifier_checkpoint: Path,
    specialist_checkpoints: dict[str, Path],
    device: torch.device,
) -> HardRouter:
    classifier_payload = torch.load(classifier_checkpoint, map_location="cpu", weights_only=False)
    classifier = build_classifier(classifier_payload["params"])
    classifier.load_state_dict(classifier_payload["state_dict"])
    classifier.to(device).eval()

    specialists: dict[str, DenoisingAutoencoder] = {}
    for corruption, path in specialist_checkpoints.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(payload["params"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        specialists[corruption] = model
    return HardRouter(classifier, specialists)


def collect_routing_records(
    router: HardRouter,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> list[dict]:
    if mode not in {"oracle", "predicted"}:
        raise ValueError(f"mode must be oracle or predicted, got {mode}")
    records = []
    with torch.no_grad():
        for batch in loader:
            corrupted = batch["corrupted"].to(device)
            clean = batch["clean"].to(device)
            labels = batch["label"].to(device)
            if mode == "oracle":
                restored, route_indices, probs = router.restore(corrupted, labels=labels)
            else:
                restored, route_indices, probs = router.restore(corrupted)
            scores = per_image_scores(restored, clean)
            input_scores = per_image_scores(corrupted, clean)
            for index, image_id in enumerate(batch["image_id"]):
                true = int(labels[index])
                pred = int(route_indices[index])
                severity = batch["severity"][index] or None
                records.append(
                    {
                        "image_id": image_id,
                        "mode": mode,
                        "true_corruption": INDEX_TO_CLASS[true],
                        "route_corruption": INDEX_TO_CLASS[pred],
                        "severity": severity,
                        "misrouted": true != pred,
                        "l1": float(scores["l1"][index]),
                        "ssim": float(scores["ssim"][index]),
                        "psnr": float(scores["psnr"][index]),
                        "input_l1": float(input_scores["l1"][index]),
                        "input_ssim": float(input_scores["ssim"][index]),
                        "input_psnr": float(input_scores["psnr"][index]),
                        "probabilities": {
                            name: float(probs[index, class_index])
                            for class_index, name in enumerate(CONDITIONS)
                        },
                    }
                )
    return records


def select_routing_failures(
    oracle_records: list[dict],
    predicted_records: list[dict],
    limit: int = 8,
) -> list[dict]:
    """Cases where a wrong class makes predicted routing much worse than oracle."""
    oracle_lookup = {
        (row["image_id"], row["true_corruption"], row.get("severity")): row
        for row in oracle_records
    }
    failures = []
    for predicted in predicted_records:
        if not predicted["misrouted"]:
            continue
        key = (predicted["image_id"], predicted["true_corruption"], predicted.get("severity"))
        oracle = oracle_lookup[key]
        gap = predicted["l1"] - oracle["l1"]
        if gap <= 0.01:
            continue
        failures.append(
            {
                **predicted,
                "oracle_l1": oracle["l1"],
                "oracle_ssim": oracle["ssim"],
                "l1_gap": gap,
            }
        )
    failures.sort(key=lambda row: row["l1_gap"], reverse=True)
    return failures[:limit]
