# Prompt: Prepare RL-ROI-Net for a final autonomous Kaggle training run (research-paper prototype)

You are an ML engineer. Make everything **ready to run on Kaggle** so a human can
start one notebook cell, walk away, and get a complete, paper-grade result set for a
3-seed temporal-region deepfake detector ("RL-ROI-Net v2"). Do NOT redesign the
experiment; the design and code are finalized. Your job is to verify, package, and
hand back a **working, self-healing notebook + dataset bundle** plus a precise
"how to run / what to download" sheet.

---

## 1. What the system is

- A frozen verdict backbone (`honi05`, trained on Celeb-DF v2) provides per-face
  features/probabilities. A **temporal classifier + ROI localization head** is
  trained on top using **same-video frame sequences** (length 8) from FaceForensics++
  (FF++, 10,400-item manifest, C23).
- Training is per **seed** (0, 1, 2). Each seed gets its own data split, its own
  feature cache (keyed by index fingerprint), and its own `outputs/seedN/` tree.
  A 3-seed mean±std is the paper's headline result.
- Feature caching: the backbone runs once per split; every epoch after that is
  head-only tensor shuffling.
- The run happens on **Kaggle: GPU T4 x2, 12h session cap, 30h/week quota**
  (quota = wall-clock, so an idle GPU is wasted quota).

## 2. What has already been done (do not redo, do not regress)

1. **Metrics table PASS/FAIL** — `rlroinet/train_temporal_region.py`:
   `print_metrics_table` now prints an **acceptance-bar legend**, a **pass?**
   column right of `best`, and an **OVERALL acceptance** row. Bar:
   `auc >= 0.85`, `region_iou >= 0.75`, `region_hit >= 0.90`,
   `confident_false_positive_rate <= 0.01`, `confident_false_negative_rate <= 0.01`,
   `reliable_coverage >= 0.50`, `review_rate <= 0.50`. Metrics without a bar show "—".
2. **Early stopping** — new args `--early-stop-patience` (default 12) and
   `--early-stop-min-epochs` (default 6). Stops once val AUC has not improved for
   `patience` epochs. `final.pt` records the actual stop epoch; `best.pt` is always
   saved on improvement and is what calibration/test use. Saves ~40 of 60 epochs.
3. **GPU-correct precision** — `tools/kaggle/kaggle_driver.py::detect_amp()` now
   picks **bf16 only on compute capability >= 8.0** (Ampere+); **T4 (Turing) uses
   fp16 + GradScaler** (native tensor cores). Old `is_bf16_supported()` logic made
   T4 run slow emulated bf16. `--workers` cap raised to 8 (halved during parallel).
4. **Parallel seeds 1+2** — after seed 0 passes the gate, seeds 1 and 2 run
   **concurrently, one per GPU**, pinned via `CUDA_VISIBLE_DEVICES` (seed 1 -> GPU 0,
   seed 2 -> GPU 1). Both T4s are used; the run drops from ~4h to well under an hour.
   Falls back to sequential if only 1 GPU.
5. **Cross-set evaluation after seed 0** — new script
   `rlroinet/eval_cross_set.py` (module: `python -m rlroinet.eval_cross_set`):
   loads the seed-0 `best.pt`, evaluates the held-out FF++ test **per manipulation
   method** (unseen-method transfer) and, if `data/celebdf` exists, raw **Celeb-DF**
   videos verdict-only. Writes `outputs/seed0/eval_cross_set.json`, prints a summary.
   The driver runs it after the gate, before escalating; an optional
   `RLROINET_CROSS_GATE_AUC` floor (default 0 = disabled) can stop the run if the
   cross-set AUC is below it. `RLROINET_CROSS_QUICK` (default 100) caps Celeb-DF
   videos/class.
6. **Results persistence** — trainer writes per-seed `metrics.json`,
   `test_metrics.json`, `decision_policy.json`, `run_config.json` (full repro info);
   driver aggregates to `run_summary.json` (mean±std + per-seed detail) and a
   human-readable `outputs/RESULTS.md`; a notebook cell bundles `outputs/` into
   `results_download.zip` for download.
7. **Known dependency saga is resolved** — the notebook's Cell 3 (repair + verify
   deps) is final: on a FRESH Kaggle session nothing is installed; only if numpy etc.
   are broken does it uninstall the damaged stack and reinstall together unpinned
   with `--no-warn-conflicts`. A corrupted numpy can only be fixed by **restarting
   the session** — never keep repairing in-session.
8. **`temporal_region.py` `_compose` bug fixed** — `forward` calls `_compose` which
   is now defined. The notebook **self-heals stale code**: Cell 4b patches
   `temporal_region.py`, Cell 4c patches `train_temporal_region.py` and writes
   `eval_cross_set.py` from base64 embeds (idempotent; only overwrites when markers
   like `def _compose`, `PASS_BAR`, `early_stop_patience`, `_CHECKPOINT_FORMAT` are
   absent). This means **the user does NOT need to re-upload the dataset** — the
   notebook brings the code up to date.

## 3. File inventory (all current and verified)

- `tools/kaggle/kaggle_driver.py` — autonomous driver (state machine).
- `tools/kaggle/build_notebook.py` — builds the notebook, embeds driver + code patches.
- `tools/kaggle/train_temporal_region_kaggle.ipynb` — **the deliverable the user runs**
  (Cells 1-6; Cell 5 is the autonomous driver).
- `tools/package_kaggle.py` — builds `rlroinet_kaggle.zip` (code + weights + FF++
  manifest + referenced face/mask/landmark files).
- `rlroinet/train_temporal_region.py`, `rlroinet/temporal_region.py`,
  `rlroinet/eval_cross_set.py`, `rlroinet/feature_cache.py`, `rlroinet/precision.py`
  — runtime code.
- `rlroinet_kaggle.zip` — **already rebuilt**: 965.5 MB, 26,040 entries, CRC-clean,
  contains the new `eval_cross_set.py` and updated trainer/temporal_region.
- `tests/` — 38 tests, all passing (run `python -m pytest tests/ -q`).

## 4. The run pipeline (what the notebook does)

1. Cell 1: env checks. 2. Cell 2: unpack zip or extracted tree into `/kaggle/working`.
3. Cell 3: repair+verify deps (no-op on healthy session). 4. Cell 4: write driver
(fresh embed). 5. Cell 4b/4c: patch runtime code. 6. Cell 5 (`kaggle_driver.main()`):
   - pulls prior state (resume),
   - runs **seed 0** alone (GPU 0), early-stopping, PASS/FAIL tables each epoch,
   - gate: `validation_best_auc >= 0.80` else FAILED + stop,
   - **cross-set eval** (per-method FF++ + Celeb-DF if present) -> `eval_cross_set.json`,
   - seeds 1+2 **in parallel** (GPU 0 + GPU 1),
   - aggregates -> `run_summary.json` + `RESULTS.md`, marks DONE.
7. Cell 6: bundle `outputs/` into `results_download.zip` for download.

## 5. Environment knobs (env overrides, defaults shown)

`RLROINET_SEEDS="0 1 2"`, `RLROINET_EPOCHS=60`, `RLROINET_GATE_AUC=0.80`,
`RLROINET_EARLY_STOP_PATIENCE=12`, `RLROINET_BUDGET_MINUTES=600`,
`RLROINET_SNAPSHOT_MINUTES=30`, `RLROINET_CROSS_QUICK=100`,
`RLROINET_CROSS_GATE_AUC=0` (0 = no floor), `RLROINET_WORKERS_MAX=8`,
`RLROINET_BATCH=8`, `RLROINET_GRAD_ACCUM=1`, `RLROINET_SEQUENCE_LENGTH=8`,
`RLROINET_TRAIN_SAMPLE=2400`, `RLROINET_VAL_SAMPLE=600`, `RLROINET_TEST_SAMPLE=600`.

## 6. Constraints & gotchas the other AI must respect

- **Kaggle quota is wall-clock** — 1h in a 2xT4 session = 1h of the 30h/week. The
  parallel-seed + fp16 + early-stop changes cut wall time from ~4h to ~1h total.
- **`KAGGLE_USERNAME` is unset** on the user's account -> dataset push is skipped
  (driver logs this). Results live in `/kaggle/working/outputs/` and MUST be
  downloaded before the 12h cap or a Stop. Do not silently depend on cross-session
  resume; the single run fits in one session.
- The mounted dataset tree may be the **stale** one (missing `_compose`); the
  notebook's patch cells fix that automatically — do not force a re-upload.
- Keep the 3 seeds **identical in config** (only `--seed` differs) so mean±std is
  valid. Early stopping applies identically to all seeds.
- `eval_cross_set.py` needs CUDA and imports from `generalize.py` for the raw
  Celeb-DF path; that path is wrapped so a missing `data/celebdf` degrades to
  `"not run: no Celeb-DF tree at ..."` in the report.
- Do not edit `data/` or the manifests; only code under `rlroinet/`, `tools/`, `tests/`.

## 7. Verification checklist (run all before handing back)

1. `python -m pytest tests/ -q` -> **38 passed**.
2. `python tools/kaggle/build_notebook.py` -> notebook compiles; every code cell
   `ast.parse`s; driver + code-patch base64 embeds are content-identical to the
   local files (line-ending agnostic).
3. `python tools/package_kaggle.py` -> zip rebuilds; verify `eval_cross_set.py`,
   updated `train_temporal_region.py`, fixed `temporal_region.py` are inside; all
   entry CRCs valid.
4. Confirm `detect_amp()` returns `fp16` on T4 (capability 7.x) and `bf16` on
   Ampere+ (capability >= 8.0).
5. Confirm the driver flow in code: seed 0 gate -> cross-set eval -> seeds 1+2
   parallel with per-GPU `CUDA_VISIBLE_DEVICES` pinning -> aggregation -> RESULTS.md.

## 8. Definition of done

A user can: (1) create a Kaggle notebook, attach the `rlroinet-kaggle` dataset
(already has the zip/tree) and this notebook, set GPU T4 x2, run Cells 1-5, wait;
(2) at DONE, run Cell 6 and download `results_download.zip`; (3) open `RESULTS.md`
for the paper: per-metric PASS/FAIL vs the bar, 3-seed mean±std, per-seed
`test_metrics.json`, and `seed0/eval_cross_set.json` (unseen-method + Celeb-DF).

If you need to change anything, change the source files and REBUILD the notebook
and zip, re-run the verification checklist, and report exactly what you changed.