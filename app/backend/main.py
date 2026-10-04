"""FastAPI app for the four assignment workspaces.

The React UI is the only client. Uploads are checked, resized to 128×128 RGB
the same way as training, and passed to ONNX when the weights are on disk.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genai.data.corruptions import (  # noqa: E402
    _spec_for_test,
    apply_corruption,
    preprocess_image,
)

OUTPUTS = ROOT / "outputs"
SAMPLE_DIR = ROOT / "data" / "pets" / "processed" / "val"
MAX_BYTES = 8 * 1024 * 1024
CLASSES = ("clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion")

app = FastAPI(title="Four Plates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, object] = {}


def _session(path: Path):
    key = str(path)
    if key in _sessions:
        return _sessions[key]
    if not path.exists():
        return None
    import onnxruntime as ort

    _sessions[key] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions[key]


def _png_data_url(image: Image.Image) -> str:
    import base64

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_nchw(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None]


def _from_nchw(array: np.ndarray) -> Image.Image:
    image = np.clip(array[0], 0, 1)
    image = np.transpose(image, (1, 2, 0))
    pixels = (image * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _read_upload(upload: UploadFile | None, sample_id: str) -> Image.Image:
    if sample_id:
        path = (SAMPLE_DIR / f"{sample_id}.png").resolve()
        if SAMPLE_DIR.resolve() not in path.parents or not path.exists():
            raise HTTPException(status_code=404, detail="That sample is not in the prepared pet set.")
        return preprocess_image(Image.open(path))
    if upload is None:
        raise HTTPException(status_code=400, detail="Upload a photo or pick a clean sample.")
    content_type = upload.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The upload has to be an image.")
    raw = upload.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Images must be 8 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="That file could not be read as an image.") from error
    return preprocess_image(image)


def _apply_requested(image: Image.Image, corruption: str, severity: str) -> tuple[Image.Image, dict]:
    if corruption in ("", "as_uploaded"):
        return image, {"corruption": "as_uploaded", "severity": None}
    if corruption not in CLASSES[1:]:
        raise HTTPException(status_code=400, detail="Unknown corruption.")
    if severity not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="Severity must be low, medium, or high.")
    seed = int(time.time_ns() % (2**31 - 1)) or 1
    spec = _spec_for_test(seed, corruption, severity, 128)
    row = spec.as_row("upload", "app")
    damaged = apply_corruption(image, row)
    return damaged, {
        "corruption": corruption,
        "severity": severity,
        "salt_probability": row.get("salt_probability"),
        "blur_kernel": row.get("blur_kernel"),
        "blur_sigma": row.get("blur_sigma"),
        "coverage": row.get("coverage"),
        "seed": seed,
    }


def _run_onnx(path: Path, image: Image.Image, output_index: int = 0) -> np.ndarray | None:
    outputs = _run_onnx_outputs(path, image)
    if outputs is None:
        return None
    return outputs[output_index]


def _run_onnx_outputs(path: Path, image: Image.Image) -> list[np.ndarray] | None:
    session = _session(path)
    if session is None:
        return None
    return session.run(None, {"corrupted": _to_nchw(image)})


def _mixture_weights(array: np.ndarray) -> dict[str, float] | None:
    vector = np.asarray(array, dtype=np.float64).reshape(-1)
    if vector.size != len(CLASSES):
        return None
    return {name: float(vector[index]) for index, name in enumerate(CLASSES)}


def _dominant_experts(weights: dict[str, float]) -> list[str]:
    # Equal share of four branches is 0.25. Highlight every branch at or above that.
    return [name for name in CLASSES if weights[name] >= 0.25]


@app.get("/health")
def health() -> dict:
    models = {
        "universal": (OUTPUTS / "task1" / "denoising_autoencoder.onnx").exists(),
        "classifier": (OUTPUTS / "task2" / "corruption_classifier.onnx").exists(),
        "specialists": all(
            (OUTPUTS / "task2" / f"specialist_{name}.onnx").exists()
            for name in ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion")
        ),
        "mixture": (OUTPUTS / "task3" / "soft_mixture.onnx").exists(),
        "sketch": (OUTPUTS / "task4" / "generator.onnx").exists(),
    }
    return {"status": "ok", "models": models}


RESULTS = ROOT / "results"
# Only these files are served to the UI. Nothing else under results/ is exposed.
RESULT_IMAGES = {
    "task1": {
        "val_preview": "val_preview_dae_b32_a05_gn_20ep.png",
        "loss_curves": "loss_curves_dae_b32_a05_gn_20ep.png",
    },
    "task2": {
        "confusion_matrix": "confusion_matrix.png",
        "routing_failures": "routing_failures.png",
        "predicted_examples": "predicted_examples.png",
    },
    "task3": {
        "weight_heatmap": "weight_heatmap.png",
        "dominant_examples": "dominant_examples.png",
        "shared_examples": "shared_examples.png",
    },
}


def _read_json(path: Path):
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def _available_images(task: str) -> dict[str, str]:
    return {
        key: f"/result-images/{task}/{key}"
        for key, filename in RESULT_IMAGES[task].items()
        if (RESULTS / task / filename).exists()
    }


def _read_metric_csv(path: Path) -> dict[tuple[str, str], dict]:
    import csv

    if not path.exists():
        return {}
    with path.open() as handle:
        return {
            (row["corruption"], row["severity"] or "all"): {
                "count": int(row["count"]),
                "l1": float(row["l1"]),
                "ssim": float(row["ssim"]),
                "psnr": float(row["psnr"]),
            }
            for row in csv.DictReader(handle)
        }


def _test_vs_input() -> list[dict] | None:
    """Task 1 model vs. the corrupted input on the test split, written by evaluate_task2.py."""
    restored = _read_metric_csv(RESULTS / "task2" / "universal_test_metrics.csv")
    baseline = _read_metric_csv(RESULTS / "task2" / "identity_test_metrics.csv")
    if not restored or not baseline:
        return None
    rows = []
    for corruption in CLASSES:
        severities = ("all",) if corruption == "clean" else ("low", "medium", "high", "all")
        for severity in severities:
            if severity == "all" and corruption != "clean":
                # Mean over the three severities (each has the same number of images).
                parts = [(corruption, level) for level in ("low", "medium", "high")]
                if not all(part in restored and part in baseline for part in parts):
                    continue
                pick = lambda table: {  # noqa: E731
                    "count": sum(table[part]["count"] for part in parts),
                    **{
                        metric: sum(table[part][metric] * table[part]["count"] for part in parts)
                        / sum(table[part]["count"] for part in parts)
                        for metric in ("l1", "ssim", "psnr")
                    },
                }
                rows.append(
                    {"corruption": corruption, "severity": "all", "restored": pick(restored), "input": pick(baseline)}
                )
                continue
            key = (corruption, severity)
            if key in restored and key in baseline:
                rows.append(
                    {"corruption": corruption, "severity": severity, "restored": restored[key], "input": baseline[key]}
                )
    return rows


@app.get("/results/task1")
def results_task1() -> dict:
    """Validation numbers for the denoising autoencoder, plus test numbers when they exist."""
    report = _read_json(RESULTS / "task1" / "eval_dae_b32_a05_gn_20ep.json")
    return {
        "test_vs_input": _test_vs_input(),
        "split": "validation",
        "params": report["params"] if report else None,
        "validation_vs_input": report["validation_vs_input"] if report else None,
        "images": _available_images("task1"),
    }


@app.get("/results/task2")
def results_task2() -> dict:
    classifier = _read_json(RESULTS / "task2" / "classifier_summary.json")
    routing = _read_json(RESULTS / "task2" / "routing_summary.json")
    specialists = _read_json(RESULTS / "task2" / "specialist_summary.json")
    return {
        "classifier": (
            {
                "selected_params": classifier["selected_params"],
                "validation_macro_f1": classifier["final_validation_macro_f1"],
                "test_metrics": classifier["test_metrics"],
                "accuracy_by_severity": classifier["accuracy_by_severity"],
            }
            if classifier
            else None
        ),
        "specialists": specialists,
        "routing": routing,
        "classes": list(CLASSES),
        "images": _available_images("task2"),
    }


@app.get("/results/task3")
def results_task3() -> dict:
    """Soft-mixture tables. Empty until Task 3 training writes summary.json."""
    return {
        "summary": _read_json(RESULTS / "task3" / "summary.json"),
        "classes": list(CLASSES),
        "images": _available_images("task3"),
    }


@app.get("/result-images/{task}/{key}")
def result_image(task: str, key: str) -> Response:
    filename = RESULT_IMAGES.get(task, {}).get(key)
    path = RESULTS / task / filename if filename else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Result image not found.")
    return Response(path.read_bytes(), media_type="image/png")


@app.get("/samples")
def samples() -> dict:
    if not SAMPLE_DIR.exists():
        return {"samples": []}
    chosen = sorted(path.stem for path in SAMPLE_DIR.glob("*.png"))[:8]
    return {"samples": [{"id": image_id, "url": f"/samples/{image_id}"} for image_id in chosen]}


@app.get("/samples/{image_id}")
def sample_image(image_id: str) -> Response:
    path = (SAMPLE_DIR / f"{image_id}.png").resolve()
    if SAMPLE_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found.")
    return Response(path.read_bytes(), media_type="image/png")


@app.post("/universal-restoration")
async def universal_restoration(
    file: UploadFile | None = File(default=None),
    sample_id: str = Form(default=""),
    corruption: str = Form(default="as_uploaded"),
    severity: str = Form(default=""),
) -> dict:
    started = time.perf_counter()
    image = _read_upload(file, sample_id)
    damaged, settings = _apply_requested(image, corruption, severity)
    output = _run_onnx(OUTPUTS / "task1" / "denoising_autoencoder.onnx", damaged)
    restored = _from_nchw(output) if output is not None else None
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {
        "input_png": _png_data_url(damaged),
        "output_png": _png_data_url(restored) if restored is not None else None,
        "inference_ms": elapsed,
        "model_loaded": restored is not None,
        "corruption_settings": settings,
    }


@app.post("/hard-routing")
async def hard_routing(
    file: UploadFile | None = File(default=None),
    sample_id: str = Form(default=""),
    corruption: str = Form(default="as_uploaded"),
    severity: str = Form(default=""),
) -> dict:
    started = time.perf_counter()
    image = _read_upload(file, sample_id)
    damaged, settings = _apply_requested(image, corruption, severity)
    logits = _run_onnx(OUTPUTS / "task2" / "corruption_classifier.onnx", damaged)
    specialists_ready = all(
        (OUTPUTS / "task2" / f"specialist_{name}.onnx").exists()
        for name in ("salt_and_pepper", "gaussian_blur", "rectangular_occlusion")
    )
    probabilities = None
    predicted = None
    expert = None
    restored = None
    if logits is not None and specialists_ready:
        scores = logits[0]
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / weights.sum()
        probabilities = {name: float(weights[index]) for index, name in enumerate(CLASSES)}
        predicted_index = int(np.argmax(weights))
        predicted = CLASSES[predicted_index]
        expert = "identity" if predicted == "clean" else predicted
        if predicted == "clean":
            restored = damaged
        else:
            specialist = _run_onnx(OUTPUTS / "task2" / f"specialist_{predicted}.onnx", damaged)
            restored = _from_nchw(specialist) if specialist is not None else None
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {
        "input_png": _png_data_url(damaged),
        "output_png": _png_data_url(restored) if restored is not None else None,
        "inference_ms": elapsed,
        "model_loaded": restored is not None,
        "corruption_settings": settings,
        "probabilities": probabilities,
        "predicted_class": predicted,
        "chosen_expert": expert,
    }


@app.post("/soft-mixture")
async def soft_mixture(
    file: UploadFile | None = File(default=None),
    sample_id: str = Form(default=""),
    corruption: str = Form(default="as_uploaded"),
    severity: str = Form(default=""),
) -> dict:
    started = time.perf_counter()
    image = _read_upload(file, sample_id)
    damaged, settings = _apply_requested(image, corruption, severity)
    outputs = _run_onnx_outputs(OUTPUTS / "task3" / "soft_mixture.onnx", damaged)
    restored = _from_nchw(outputs[0]) if outputs is not None else None
    weights = _mixture_weights(outputs[1]) if outputs is not None and len(outputs) > 1 else None
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {
        "input_png": _png_data_url(damaged),
        "output_png": _png_data_url(restored) if restored is not None else None,
        "inference_ms": elapsed,
        "model_loaded": restored is not None,
        "corruption_settings": settings,
        "weights": weights,
        "dominant_experts": _dominant_experts(weights) if weights else [],
    }


@app.post("/face-to-sketch")
async def face_to_sketch(
    file: UploadFile | None = File(default=None),
    style: str = Form(default="Style 1"),
) -> dict:
    if style not in ("Style 1", "Style 2", "Style 3"):
        raise HTTPException(status_code=400, detail="Pick Style 1, Style 2, or Style 3.")
    started = time.perf_counter()
    image = _read_upload(file, "")
    generator = OUTPUTS / "task4" / "generator.onnx"
    sketch = None
    if generator.exists():
        # The style embedding is inside the exported generator. This endpoint
        # passes the style name; the ONNX graph is wired when Task 4 exports it.
        sketch = None
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {
        "input_png": _png_data_url(image),
        "output_png": _png_data_url(sketch) if sketch is not None else None,
        "inference_ms": elapsed,
        "model_loaded": sketch is not None,
        "style": style,
    }
