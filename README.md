# RL-ROI-Net v2

RL-ROI-Net is a local video evidence-review tool for deepfake analysis. The deployed path uses a frozen `honi05` EfficientNet-B4 backbone with a head-only classifier and region-localization head. A separate `train.py` path without `--verdict` trains the configured ImageNet EfficientNet-B4 path with its configured frozen blocks; that path is not the deployed checkpoint. The result is deliberately three-way: **REAL**, **FAKE**, or **REVIEW**. REVIEW means a person must assess the clip; no result is proof of authenticity or manipulation.

## Run the local service on Windows

1. Start Docker Desktop with NVIDIA GPU support enabled.
2. Double-click **`start.bat`** in the repository root.
3. It rebuilds/starts the `api` service with `outputs/checkpoints/best.pt`, waits for `/health`, then opens `http://localhost:8000/demo.html`.

The launcher fails clearly if Docker Compose, CUDA support, or `outputs/checkpoints/best.pt` is unavailable. To stop the service:

```text
docker compose stop api
```

## Local review workflow

- Upload one authorized video, up to **30 seconds** and **250 MiB**.
- The service uniformly samples 32 frames, computes a robust video score, and returns a REAL / FAKE / REVIEW outcome.
- The UI displays ROI boxes for sampled-frame localization evidence, supports cancellation, downloads a local JSON report, and can delete the generated video/evidence bundle immediately.
- Review bundles otherwise expire after 24 hours. The local API accepts one analysis at a time to protect the GPU.

The API runs at `http://localhost:8000`; `GET /health` reports checkpoint/CUDA readiness, `POST /analyze` powers the UI, and `DELETE /uploads/{request_id}` deletes an analysis bundle.

## Model and evidence boundary

The production runtime is explicitly configured to use:

```text
outputs/checkpoints/best.pt
```

FF++ held-out-video measurements are retained in `outputs/eval_generalization.json` and `outputs/eval_report.json`. The latest same-FF++ continuation is retained separately with its reports under `outputs/checkpoints/diverse_all4_alpha60_continuation/`; it was **not promoted** because its comparable evidence did not justify replacing the production checkpoint.

The model has no current external-dataset evaluation because raw Celeb-DF is unavailable. Treat this as a local review assistant only: retain REVIEW outcomes for human assessment, and do not use an output as the sole basis for moderation, legal, safety, or identity decisions.

## Development

```text
docker compose build
docker compose run --rm tests
docker compose run --rm evaluate
```

The project uses frozen-backbone, head-only training with Focal classification and Dice/BCE mask supervision. Dataset inputs remain under `data/`; model weights remain under `outputs/models/`. Historical non-production experiments, failed run folders, caches, and stale reports were removed during the current cleanup.
