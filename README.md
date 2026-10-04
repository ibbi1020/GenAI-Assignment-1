# Generative AI Assignment 1

Image restoration on Oxford-IIIT Pet (Tasks 1–3) and face-to-sketch generation on FS2K (Task 4).

## Setup

```bash
uv sync
```

## Prepare the shared data

Oxford-IIIT Pet is downloaded from the official Oxford site. FS2K is downloaded from the official Google Drive link.

```bash
uv run python scripts/prepare_pets.py
uv run python scripts/prepare_fs2k.py
uv run python scripts/verify_data.py
uv run pytest
```

Raw downloads and resized images stay on disk and are gitignored. Split files and corruption manifests are small and are meant to be kept with the code.

Do not train on `data/pets/processed/test` or `data/fs2k/processed/test` until the final evaluation. Do not use those test lists to choose hyperparameters.

Training corruption for Tasks 1–3 is not saved as image files. Apply it at load time with `genai.data.corruptions`.

## Task 1: Universal Restoration

```bash
MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/run_task1.py \
  --trials 10 --trial-epochs 5 --final-epochs 25 --patience 6 --workers 2
```

Optional bottleneck comparison (validation only, no test set):

```bash
MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/compare_bottlenecks.py
```

Checkpoints and ONNX weights land in `outputs/task1/` (gitignored). Metrics and figure grids land in `results/task1/`.

## Task 3: Soft mixture

Needs the Task 2 classifier and specialist checkpoints in `outputs/task2/`.

```bash
MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/run_task3.py --workers 0
```

The run used for the report was shorter: `--warmup-epochs 2 --epochs 8 --patience 4`. The ONNX file is `outputs/task3/soft_mixture.onnx`. Metrics land in `results/task3/`.

## Application

One browser app, four workspaces. The React UI talks only to FastAPI.

```bash
docker compose up --build
```

Open http://127.0.0.1:5173. Put the ONNX files in `outputs/task1/`, `outputs/task2/`, and `outputs/task3/` before starting. Compose mounts that folder and `results/` into the API container.

Without Docker, from this folder:

```bash
uv run --with fastapi --with 'uvicorn[standard]' --with python-multipart --with onnxruntime \
  uvicorn app.backend.main:app --port 8000
```

In another terminal:

```bash
cd app/frontend && npm install && npm run dev
```
