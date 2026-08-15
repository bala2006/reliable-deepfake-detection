# Next Steps — Make RL-ROI-Net a Reliable, Benchmarked Research Prototype

Status baseline: 2026-08-15, from retained artifacts and verified on-disk data.

## Research idea (the story this project must deliver)

> A video is classified as **REAL / FAKE / REVIEW** (REVIEW = mandatory human escalation).
> If FAKE, the system shows sampled-frame evidence and fixed landmark-defined regions
> (PERIOCULAR / JAWLINE / MOUTH / HAIRLINE) associated with the model output. These
> boxes are not causal attribution of the manipulation.

This idea is implemented end-to-end in the code. The project's problem is **reliability**
and **missing evidence**, not missing features.

| Research claim | Existing implementation | Status |
|---|---|---|
| Video -> REAL / FAKE / REVIEW | `video_decision()` + robust 32-frame pooling, 0.30/0.70 band | built |
| Show sampled evidence frames | `predict_frames()` per-frame flag + trace/detections | built, frame-level metric never reported |
| Fixed-ROI localization evidence | Region head -> 4 ROI maps; `_annotate()` draws fixed landmark-defined ROI boxes on saved JPEGs | built; boxes are not causal attribution |
| Reviewable evidence | JSON report + annotated frames + DELETE, 24 h retention | built |

## Root cause of weak accuracy (verified)

The frozen backbone `honi05` (`outputs/models/best_model.pt`) is an EfficientNet-B4 **trained only
on Celeb-DF v2** (per its model card: 590 real / 5,639 fake videos). The earlier claim of
AUC 0.99 on raw Celeb-DF is **unattributable and not citable**: `outputs/verdict_shootout.json`
records 200 videos and ACC 0.9450 / AUC 0.9903, but specifies no dataset, split, disjointness,
or preprocessing.

On the leakage-safe official-pair FF++ diagnostic (`outputs/diagnostics/faceswap_inversion_diagnostic_retry.json`,
1132 records / 142 videos), there is **no FaceSwap inversion or collapse**. The production head
has per-family video AUC Deepfakes 0.5488, Face2Face 0.5833, FaceSwap 0.5310, and NeuralTextures
0.5816, with overall AUC 0.5615. Frozen honi05 alone reaches 0.5377 overall (real mean 0.145 vs
fake mean 0.188), while the spatial adapter reaches 0.6274. The widely cited 0.951 / 0.927 / 0.283
spread came from `outputs/eval_generalization.json`, which is **not** the official-pair split and
must not be treated as official-pair model behavior.

**Implication:** the current official-pair evidence supports a near-chance production baseline and
a measured adapter improvement, not a family-specific FaceSwap collapse. Head-only improvement,
backbone choice, and external validity remain open research questions.

## Architecture paths (both exist)

- `train.py --verdict honi05`: frozen Celeb-DF-trained honi05 verdict backbone with trainable
  classification/region heads; this is the deployed production path.
- `train.py` without `--verdict`: configured ImageNet EfficientNet-B4 path with its configured
  frozen blocks; this is an alternate training path, not the deployed checkpoint.

These paths must be named explicitly in any experiment report.

## Backbone-leakage rule (non-negotiable for the paper)

A backbone's training set must be disjoint from the dataset used to evaluate it.

- `honi05` (Celeb-DF-trained)  -> evaluate on **FF++** (unseen to it).      <- current setup, valid
- `deepfakebench_xception` (FF++-trained) -> evaluate on **Celeb-DF** (unseen to it).
- Never claim "cross-dataset Celeb-DF" with a honi05-based system; that is leakage.

## Dataset decision (checked on disk, no downloads needed)

| Source | On disk | Masks | Use |
|---|---|---|---|
| FF++ c23 | youtube 999 + actors 363 real; Deepfakes/F2F/FaceSwap/NT ~1000 each + DeepFakeDetection 3066 | masks for 100% of its 5,200 fake records; only mask-supervised source | Training + locked eval |
| Celeb-DF-v2 | frames + landmarks only: 588 real / 5,639 fake / 300 yt (256x256) | zero mask files | Classification-only external evaluation; honi05 is contaminated for this source |
| DFDCP | 5,250 records: 4,119 fake / 1,131 real; 4,473 train / 777 test | zero mask files | Classification-only external robustness source |

- **Celeb-DF v1 raw videos: NOT needed.** v2 is the standard benchmark and its extracted frames
  are sufficient for an honest classification-only cross-dataset eval.
- FF++ raw videos were removed from disk; all FF++ paths go through the manifest
  (`data/FaceForensics++/.rlroinet_index.json`, 10,400 records).
- Data frames are aligned face crops (256x256) -> loader resizes to `face_size` (288) ->
  honi05 internally resizes to 224. Consistent with runtime (video -> sample -> face crop -> model).
- Capacity limitation: the FF++ official test contains 142 videos. Valid external capacity is
  605 source-video identities (Celeb-DF-v2 518 + DFDCP 87); DFDCP's 777 test records collapse
  to 87 source-group identities. The >=1,000-test-video and >=2-external-dataset gates are
  therefore unreachable with the data on disk.

## Release-gate status

The numeric release-gate set is **pending approval**. The earlier proposals in this file
(AUC >=0.85 / IoU >=0.75 / hit >=0.90) and `training.md` (AUC >=0.95 / IoU >=0.65)
are withdrawn and must not be presented as approved targets. Do not invent replacement
thresholds. Until approval, report the following without pass/fail claims:

- The official-pair video protocol and its 142-video capacity.
- The 603-face-sample BEST report separately from the honest 0.5592 official-pair video AUC.
- Classification-only external results, with no localization claim for Celeb-DF-v2 or DFDCP.
- Calibration, REVIEW coverage, frame-level metrics, and 3-seed variation when measured.
- API-aligned protocol details and complete provenance for every result.

## Execution plan

### Phase 0 — Foundation (do first, highest value, no model training)

1. **Backbone shootout on FF++ (per-method).** Score frozen `honi05` vs `deepfakebench_xception`
   on the FF++ held-out split. Pick the stronger FF++ backbone, or **ensemble both verdicts**
   (honi05 = Celeb-DF expert, xception = FF++ expert) as the frozen signal for the region head.
   - Add a small script or extend `benchmark_verdict.py` to run on FF++ frames via the manifest.
   - Record per-method AUC/ACC/ECE; this drives every later decision.
2. **Frame-based Celeb-DF eval path.** `generalize.py`/`benchmark_verdict.py` expect raw `.mp4`
   at `data/celebdf`; only extracted frames exist. Add a Celeb-DF frame indexer
   (`data/Celeb-DF-v2/{Celeb-real,Celeb-synthesis,YouTube-real}/frames/*/*.png`,
   case-insensitive) so classification eval runs on stored crops. No masks -> classification only.
3. **One consistent benchmark harness.** A single entry point that evaluates a checkpoint on the
   same locked video-disjoint FF++ split producing: overall + per-method + Celeb-DF + frame-level
   AUC + ECE + REVIEW-policy metrics. Ends the 152-vs-142-vs-603 incomparable protocols.
4. **Quarantine unattributable reports.** `outputs/verdict_shootout.json` records ACC 0.9450 / AUC 0.9903 on 200 videos without dataset, split, disjointness, or preprocessing; `outputs/eval_report.json` is likewise unattributed. Preserve both as historical artifacts, but do not cite either.
5. **Init a git repo** for run/checkpoint traceability.

### Phase 1 — Data

6. Rebuild FF++ manifest: `manifest_frames_per_video` 8 -> 16; use more of the ~1000
   videos/method available (currently ~112 used).
7. Add **DeepFakeDetection (masked) + actors** to training methods for diversity; keep
   FaceShifter classification-only (no masks).
8. Add **per-method sampling weights** upweighting FaceSwap / Deepfakes so hard methods are not
   drowned by easy ones (Face2Face/NeuralTextures).
9. Optionally add DFDCP as an external robustness training source (classification-only,
   keep the synthetic-ROI flag; never report its localization as evidence).

### Phase 2 — Model (training optimizations verified against the code)

10. Train the champion on the chosen backbone: the **64-channel adapter** (already beats
    `best.pt` under the official split) or the **gated `temporal-region-v2` head**
    (measured video AUC 0.834, regionIoU 0.80 — best so far).
11. **Learning-rate schedule (implemented):** every trainer currently uses a constant `head_lr`;
    literature and the honi05 model card itself use cosine decay. Added `WarmupCosine`
    (`rlroinet/snapshot.py`): 1-epoch linear warmup then cosine decay to `head_lr * lr_min_factor`
    (default 0.1). Flags `--lr-warmup-epochs`, `--lr-min-factor` on `train_temporal_region.py` and
    `train_adapter.py`. This is the single highest-leverage training change.
12. **30-minute snapshots (implemented):** `SnapshotScheduler` saves a labeled checkpoint every
    `--snapshot-minutes` (default 30) of wall-clock time — `snapshot_t<minutes>min_epoch<NN>.pt` —
    so you can watch validation AUC improve mid-run without waiting for completion.
    - Every epoch already evaluates **validation video AUC** and appends to `metrics.json`;
      snapshot metrics also carry `elapsed_minutes`. To check progress while training, tail
      `outputs/metrics.json` or the log line `epoch N/N ... val AUC=... elapsed=...min`.
    - Epoch pacing on the RTX 4050: temporal-region ~430 s/epoch (5 epochs ≈ 36 min, so
      snapshots land roughly every ~4 epochs); head-only/adapter faster. Snapshot cadence is
      wall-clock so a 30-min run still yields a mid-run checkpoint.
13. 5-10 epochs, early-stop on **validation video AUC** (head-only best was epoch 16; temporal
    best epoch 3-5). ~430 s/epoch -> full run ~36 min to a few hours on the RTX 4050.
14. Run **3 seeds** for the final candidate; report mean +/- std (set `cfg.train.seed`, keep it
    in the checkpoint).
15. Persist `decision_policy.json` + temperature + seed + config beside every candidate.
16. **Further optimizations worth testing (in priority order):**
    - **EMA of trainable head weights** (~0.999 decay) — cheap, reliably smooths val AUC. Not yet implemented.
    - **Class-balanced / per-method sampling** instead of uniform shuffle — upweights FaceSwap/Deepfakes,
      which are the current AUC failures (0.58/0.28).
    - **Label smoothing** (e.g. 0.05) on the frame/video BCE — helps calibration (ECE is 0.217 today).
    - **Stronger sequence augmentation** — currently only photometric contrast/noise/blur on 35-50% of
      batches; add per-sequence random frame drop and horizontal flip (no-op on symmetric crops).
17. **Precision + feature-cache speedups (implemented):**
    - **bf16 AMP** (`--amp-dtype bf16`, now the default): on Ada/RTX 4050 bf16 runs the same tensor-core
      math as fp16 but keeps fp32-comparable exponent range, needs **no GradScaler**, and never underflows.
      Master weights stay fp32. All trainers + evaluation use the configured dtype via `rlroinet/precision.py`.
      `--amp-dtype fp16` restores the old GradScaler path (only needed on pre-Ampere GPUs).
    - **Frozen-backbone feature caching** (`--feature-cache DIR`): the backbone is frozen, so `build_feature_cache`
      computes `(features, verdict_prob)` once per face (one backbone pass, ~the cost of a single epoch) and saves
      them to `DIR/features_<verdict>_<hash>.pt` keyed by an index fingerprint — stale caches auto-rebuild.
      Training then runs **only the heads** (`RegionAgent/AdapterRegionAgent/TemporalRegionAgent.forward_from_features`),
      skipping the ~430 s/epoch backbone forward entirely.
      - `train_adapter.py --feature-cache DIR` and `train.py --verdict honi05 --feature-cache DIR`: cache the train split only
        (validation/test keep real faces; identical quality).
      - `train_temporal_region.py --feature-cache DIR` requires `--no-train-augmentation` (cached features were computed
        on unaugmented faces) — the cache is incompatible with photometric augmentation.
      - `train.py`/`train_adapter.py` also enable TF32 + `cudnn.benchmark` automatically.
    - Verified end-to-end on CPU smoke tests: cache build/load/fingerprint invalidation, cached region/adapter/temporal
      forwards, and cached training steps all produce correct losses.
18. **GPU-parallel data loading (implemented):**
    - `train.py`, `train_adapter.py` no longer decode faces/masks synchronously on the main thread. Every trainer now
      iterates a DataLoader with `--workers N` (default 4) + `pin_memory` + `prefetch` + `persistent_workers`, so CPU
      PNG decode overlaps the GPU and host-to-device copies are non-blocking (`non_blocking=True`).
    - New `--workers` / `--prefetch-factor` flags on `train.py` and `train_adapter.py`; `train_temporal_region.py` already
      had them. `rlroinet/data/maskdata.py` exports `mask_sample_collate` + `make_data_loader`; cached batches use
      `cached_mask_collate` (`rlroinet/feature_cache.py`).
    - The feature cache now also stores **masks and image-quality features**, so cached epochs are fully RAM-backed:
      zero face decoding, zero quality recomputation, no face tensors transferred to the GPU. `TemporalRegionAgent`
      gained `forward_from_cached(features, verdict_probs, quality, ...)` used by the temporal cached path.
    - `torch.backends.cudnn.benchmark = True` is now enabled in `train.py` too (was only in adapter/temporal trainers).
    - Verified on CPU smoke tests: both collates, worker-backed loaders (2 workers), cached + non-cached training steps,
      loader-driven cached epoch, and the cached temporal forward all produce correct losses/shapes.

### Phase 3 — Evaluation + promotion

17. Calibrate (temperature + REVIEW bounds) on the video-disjoint validation split.
18. Measure against the target bar on the **one consistent protocol** (FF++ locked split +
    Celeb-DF via xception + frame-level AUC + per-method).
19. Promote to `outputs/checkpoints/best.pt` ONLY if the candidate passes the bar on the
    consistent protocol. Otherwise keep `best.pt` and document honestly.

### Phase 4 — Paper artifacts

20. Metrics table: overall + per-method + cross-dataset + frame-level + calibration rows.
21. Figures: ROI boxes on flagged frames; REVIEW-rate behavior on held-out/weak methods.
22. Verify honi05 MIT provenance (HF model card is MIT); never bundle NC weights or datasets.
23. Honest framing: FaceSwap weakness and Celeb-DF drop are *features* of the REVIEW/escalation
    story, not bugs.

19. **Autonomous Kaggle training (implemented):** `tools/kaggle/` is a self-driving pipeline —
    one notebook, no manual steps after launch.
    - `kaggle_driver.py` = a state machine: pull persisted state from a private Kaggle dataset,
      resume the exact seed/epoch where the previous session stopped (covers the 9h session cap
      and 30h/wk quota), run seed 0, evaluate a health gate, and only escalate to the full 3-seed
      run if the gate passes; otherwise mark FAILED and end the session (GPU freed).
    - `train_temporal_region_kaggle.ipynb` = generated notebook that embeds the driver (base64,
      byte-identical) plus a local unpack of the package from `/kaggle/input` and dependency
      install. Rebuild with `python tools/kaggle/build_notebook.py`.
    - The package is delivered as a **private Kaggle dataset** (`rlroinet-kaggle` containing
      `rlroinet_kaggle.zip`), not via Google Drive: `/kaggle/input` is mounted locally, so there
      is no network download and no "Anyone with the link" sharing. The notebook copies the zip
      to `/kaggle/working` each session (input is read-only) — that is the only "download".
    - **Runtime decisions:** GPU type (T4 => `--amp-dtype bf16`, P100 => `fp16`+GradScaler), CPU
      count (`--workers`), resume point, gate threshold (`RLROINET_GATE_AUC`, default 0.80),
      wall-clock budget (`RLROINET_BUDGET_MINUTES`, default 450 < 540).
    - **Gate (seed 0):** max val video AUC >= 0.80 AND finite losses AND valid test metrics.
      Pass => seeds 1, 2 + mean/std aggregation; fail => `state.json` = failed, session stops.
    - **Persistence:** after every 30-min snapshot and at each transition the sync dir (per-seed
      checkpoints + metrics + state) is pushed to a private Kaggle dataset via the `kaggle` CLI
      (pre-authenticated in notebooks). Restarting the notebook pulls it and resumes. Each seed
      owns its own `outputs/seed<N>/` so metrics.json never collides.
    - `rlroinet/feature_cache.py` cache build now **streams in blocks** (~5GB peak instead of
      ~14GB) so the ~8000-face train split fits a 16GB Kaggle session. `train_temporal_region.py`
      gained `--seed` + full RNG seeding for reproducible multi-seed runs (each seed's cache
      auto-rebuilds via its index fingerprint).
    - Verified: 32/32 tests pass; driver state-machine/gate/aggregation logic + streaming cache
      build (workers 1 and 2) + the notebook unpack cell (26,039 entries from a real zip) all
      smoke-tested on CPU.

## Immediate next action (Phase 0, step 1)

Run the frozen-backbone comparison on FF++ held-out (honi05 vs deepfakebench_xception), record
per-method AUC, and let the result choose the backbone for Phase 2 training.

## Quick start for Phase 2 (new flags in place)

```bash
# champion temporal-region head with 30-min snapshots + cosine LR (defaults are already set)
python -m rlroinet.train_temporal_region --checkpoint-dir outputs/checkpoints/tr_v3 \
    --epochs 10 --snapshot-minutes 30 --lr-min-factor 0.1 --lr-warmup-epochs 1

# same, but cache the frozen backbone features once (~1 backbone pass, then heads only).
# Requires --no-train-augmentation (cache is computed on unaugmented faces).
python -m rlroinet.train_temporal_region --checkpoint-dir outputs/checkpoints/tr_v3_fc \
    --epochs 10 --snapshot-minutes 30 --no-train-augmentation --feature-cache outputs/feature_cache

# resume an interrupted run from the NEAREST saved checkpoint (best.pt / final.pt /
# latest snapshot_t*min_epochNN.pt) -- restores model + optimizer + LR schedule + history
python -m rlroinet.train_temporal_region --checkpoint-dir outputs/checkpoints/tr_v3 \
    --epochs 10 --snapshot-minutes 30 --resume

# adapter candidate, same protocol (also supports --resume). Cache the train split freely
# (no augmentation there) for a ~4x epoch speedup:
python -m rlroinet.train_adapter --checkpoint-dir outputs/checkpoints/adapter_v3 \
    --epochs 10 --snapshot-minutes 30 --lr-min-factor 0.1 --lr-warmup-epochs 1 \
    --feature-cache outputs/feature_cache

# production region-head path with the frozen honi05 backbone, cached the same way
python -m rlroinet.train --verdict honi05 --checkpoint-dir outputs/checkpoints/rh_v3 \
    --epochs 10 --feature-cache outputs/feature_cache
```

Precision is **bf16 by default** on the RTX 4050 (`--amp-dtype bf16`, no GradScaler, faster and more
stable than fp16). Use `--amp-dtype fp16` only on pre-Ampere GPUs. Feature caches are keyed by an index
fingerprint + dtype, so a changed split or dtype automatically rebuilds them.

Monitor live: `snapshot_t*min_epochNN.pt` files land in the checkpoint dir every 30 min of
training, and `outputs/metrics.json` appends val AUC per epoch with `elapsed_minutes`.

## Kaggle quick start (autonomous, BIG plan)

1. `python tools/package_kaggle.py` (already done) -> `rlroinet_kaggle.zip` (943 MB).
2. On Kaggle: **Datasets -> New Dataset** -> Upload from file -> drag `rlroinet_kaggle.zip` in.
   Name it `rlroinet-kaggle`, keep it **Private**, wait for the upload to finish (keep the tab open).
3. `python tools/kaggle/build_notebook.py` (already done) -> `tools/kaggle/train_temporal_region_kaggle.ipynb`.
4. On Kaggle: New Notebook -> Import from file. Set **Accelerator = GPU T4 x2**, **Internet ON**,
   and attach the `rlroinet-kaggle` dataset via **Add Input**.
5. Run cells 1-3 once, then **Cell 5** (RUN) and walk away. It unpacks the zip from
   `/kaggle/input`, trains **seed 0 for 60 epochs (~10 h budget)**, gates on val AUC >= 0.80,
   escalates to seeds 1-2 on pass (or stops on fail), resumes across the 12h session cap via a
   private Kaggle dataset, and reports `run_summary.json` (mean +/- std).

### Current plan vs BIG (full) plan

| | Current (balanced) | **BIG (full)** |
|---|---|---|
| Epochs / seed | 40 | **60** |
| Effective batch | 8 (batch 8, grad-accum 1) | 8 (batch 8, grad-accum 1) |
| Seq length / frames-per-video | 8 / 8 | 8 / 8 |
| Train / val / test budget | 2400 / 600 / 600 | 2400 / 600 / 600 |
| Actual split (pool 10,400) | ~7,964 / 1,220 / 1,216 | ~7,964 / 1,220 / 1,216 |
| Session budget | 450 min (was) / 660 min (then) | **600 min (~10 h of 12 h cap)** |
| Snapshot push | every 30 min | every 30 min |
| Gate | val AUC >= 0.80 | val AUC >= 0.80 |
| Seeds | 0 -> gate -> 1,2 | 0 -> gate -> 1,2 |
| Weight decay / head LR | AdamW 3e-4, warmup 2 | AdamW 3e-4, warmup 2 |
| Estimated seed-0 time (T4 x2, bf16, cached features) | ~2.7 h | **~4 h** |

**How to switch profiles:** everything is env-configurable. For the current/balanced plan set
`RLROINET_EPOCHS=40` and `RLROINET_BUDGET_MINUTES=450`; the BIG plan is the notebook default
(60 epochs, 600 min). GPU detection stays automatic (T4 -> bf16, P100 -> fp16).

**Interrupted runs:** every checkpoint (`best.pt`, `final.pt`, and each 30-min snapshot) now
saves the optimizer state, and every trainer stores `optimizer_state` + `epoch`. Rerun with
`--resume` and training continues from the highest-epoch saved checkpoint — LR schedule, AdamW
momentum, best-val-AUC, and epoch history all carry over, so you can stop mid-run, inspect
`snapshot_t*min_epochNN.pt`, and restart without losing progress.