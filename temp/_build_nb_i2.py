"""Emit notebook/i2-EvidenceNet-v1.ipynb (EvidenceNet v1 — i2 redesign).

Follows the same generator pattern as temp/_build_nb.py (which emitted the i1
Stable-RouteNet v4 notebook).

i2 changes from i1 (all justified in architecture/i2-EvidenceNet-v1.md):
  - existence aggregation (top-k / LSE / max / mean / std) replaces the convex average
    that destroyed i1's localized evidence
  - boundary target B = 4*m*(1-m) (LAA-Net / Face X-ray) in addition to the region mask
  - token-level prototype router removed (i1: entropy 1.606 ~= ln 5 -> a no-op)
  - low-level high-pass/NPR residual branch (sub-patch seams)
  - self-consistency patch head (LAA-Net)
  - alignment + uniformity losses on L2-normalised features (GenD)
  - held-out calibration split, temperature scaling, validated threshold
  - frame cache + numpy-only forging + random frames per epoch (i1 was 30x loader-bound)
  - final cross-dataset evaluation is a required, output-producing cell

Implementation notes that matter for correctness under nn.DataParallel:
  * the trunk's batch dimension is FRAMES (B*T): every patch-level output (e_logit, b_logit,
    l_logit, Z, g, sc) and the region/boundary targets are per frame, while the two DECISION
    outputs (bag, recipe) are folded to one value per clip with
    `bag = bag_f.reshape(B, T).mean(1)` -- i1's `Zv = Zf.reshape(B,T,-1).mean(dim=1)` -- so
    they line up with the clip-level labels
  * eval/scoring paths call the UNWRAPPED module, so a batch smaller than the GPU count
    cannot produce an empty replica chunk
  * checkpoints store keys with any "module." prefix stripped, so they reload into the
    unwrapped module
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n") + "\n"))


def code(src):
    CELLS.append(("code", src.strip("\n") + "\n"))


# --------------------------------------------------------------------------- 0
md(r"""
# EvidenceNet v1 — Localization-First Detection with Existence Aggregation

**Iteration `i2`** · supersedes the Stable-RouteNet line (`i1-Stable-RouteNet-v4`).

> i1 learned *where* the manipulation was (pixel AUC 0.938) but could not *decide*
> (worst AUROC 0.578). Its image score was a **convex average** of patch features, so the
> sparse, localized artifact was diluted by roughly `1 / mask-area-fraction` before the
> classifier ever saw it. i1's own diagnostics show it: mean manipulation score on fake
> frames `0.3336` versus real frames `0.3296`, a difference of `0.004`. The information
> existed and the aggregator threw it away.

This notebook rebuilds the decision path around an **existence operator** ("does *any*
patch contain artifact evidence?") and keeps everything else that worked.

## What changed, and the evidence for each change

| # | i1 | EvidenceNet v1 | Evidence |
|---|---|---|---|
| 1 | convex-average pooling | top-k / LSE / max / mean / std aggregation | i1 diagnostics above; MIL pooling literature |
| 2 | "LoRA" = 11k-param rank-8 bottleneck on frozen tokens | frozen trunk; LN-tuning available as flag-gated stage 2 | GenD ablation: linear probe 76.0 → LN-tuning 90.3 mean AUROC; LoRA rank-1 hits 99.99 % train AUROC in 2 epochs |
| 3 | region-mask supervision only | + boundary target `B = 4·m⊙(1−m)` | LAA-Net CVPR'24; Face X-ray CVPR'20 |
| 4 | token router, top-2 of 5 | **removed** | i1 measured router entropy `1.606 ≈ ln 5 = 1.609` → uniform → no-op |
| 5 | trust weights `M×S×R^α` on the decision path | demoted to a two-view consistency regularizer | normalized weights (mean 1) cannot amplify |
| 6 | 0.5 threshold, calibration on the selection split | held-out calibration split + temperature scaling + validated threshold | `record/i1-...-Fail.md` findings #3 and next-step #3 |
| 7 | cross-dataset evaluation never executed | required, output-producing evaluation cell reporting **within-dataset and cross-dataset** tests | i1 cell 14 has no outputs |
| 8 | ~30 s/step, loader-bound | frame cache + numpy forging + random frames | 900 batches / 443 min = 1.1 images/s vs 25–40 images/s compute bound |
| 9 | 8 fixed frames per video | random frames every epoch | 7 200 → ~128 000 effective frame-passes |
| 10 | forgery head supervised on a constant label | recipe head with 15 always-instantiated classes | i1 `N_FORGERY_TYPES = 7` with no real forgery ever seen |

## Protocol (unchanged from i1, deliberately)

Training uses **real videos only** plus self-blended synthesis. **No real manipulated video
is ever trained on.** This is also the setting in which LAA-Net reaches 95.4 AUROC on
Celeb-DF-v2, so it is not the reason i1 failed. DFDCP, Celeb-DF-v2 and all six FF++
manipulation families stay completely unseen until the final evaluation cell.

Design document: `architecture/i2-EvidenceNet-v1.md`.

## Run order

config → frame cache → imports → splits + leakage → synthesis → datasets → model →
losses → metrics → **self-test / preflight (must print `PREFLIGHT PASSED`)** → training →
report + gates + figures → final evaluation → calibration.

Set `DRY_RUN = True` in the config cell first: it runs the entire pipeline on a handful of
videos in a few minutes, which catches shape and data errors before spending the night.
""")

# --------------------------------------------------------------------------- 1
md(r"""
## Protocol, splits and leakage control

| Split | Source | Used for |
|---|---|---|
| **train** | FF++ YouTube-c23 real 110 + Celeb-DF Celeb-real 100 + YouTube-real 100 | gradient updates |
| **val-hard** | held-out videos, video-level disjoint, hard synthesis parameters | monitoring + checkpoint selection |
| **calibration** | held-out videos + fresh synthesis draws, disjoint from train and val | temperature + threshold fitting (**new in i2**) |
| **test — within-dataset** | held-out FF++ real 50 vs self-blended fakes generated from those same 50 videos with the training recipe; plus the pack's own FF++ Deepfakes / Face2Face / FaceShifter / FaceSwap / NeuralTextures / DeepFakeDetection (50 each) | reported once, after freeze |
| **test — cross-dataset** | DFDCP 50 real + 50 fake | reported once, after freeze |
| **test — cross-dataset** | Celeb-DF-v2 50 real + 50 fake | reported once, after freeze |

`assert_no_leakage()` enforces that train, val, calibration and every test set are
disjoint at video level, and that no test video is ever passed to the synthesis generator.
A `manifest.json` hash is written into the checkpoint, so results cannot be reported
against a different split.

## The data pack (`sekhar826/srn-v4b-data`)

A 5.6 GB curated pack (Kaggle ref `sekhar826/srn-v4b-data`, version 1), not raw FF++.
It carries real face videos from three independent sources plus the manipulated benchmark
sets, each already cut into frames with per-frame landmark `.npy` files:

| Pool on disk | Content | Videos |
|---|---|---|
| `FaceForensics++/original_sequences/youtube/...` | FF++ YouTube-c23 **real** | 200 |
| `Celeb-DF-v2/Celeb-real` | Celeb-DF-v2 **real** | 200 |
| `Celeb-DF-v2/YouTube-real` | Celeb-DF-v2 **real** | 200 |
| `Celeb-DF-v2/Celeb-synthesis` | Celeb-DF-v2 **fake** (test only) | — |
| `DFDCP/original_videos` + `method_A` / `method_B` | DFDC-preview real + fake (test only) | — |
| `FaceForensics++/manipulated_sequences/<family>/c23` | 6 FF++ families (test only) | — |

The three real pools give `200 + 200 + 200 = 600` videos, which is exactly what the four
splits consume (`110+25+15+50`, `100+25+20+50`, `100+25+20+50`). Nothing is left over, so
`split_pool` prints and scales the split if the pack turns out smaller than assumed instead
of asserting at 2 a.m. Layout discovery is by sniffing (a directory holding `Celeb-DF-v2`
plus `DFDCP` or `FaceForensics++`), so the pack resolves wherever Kaggle mounts it.
**Two evaluation axes, and both are required.** (A) *Within-dataset*: held-out videos from the
same three real pools, pushed through the training pixel pipeline (synthesis recipe for the fake
side, the training photometric draw for the real side), plus the pack's own real manipulations of
those same held-out videos. No domain shift, so this measures whether the task was learnt at all.
(B) *Cross-dataset*: DFDCP and Celeb-DF-v2 — other corpora and other generators, raw frames, the
published protocol. `assert_no_leakage()` enforces video-level disjointness for train / val /
calibration and every test set, and the split manifest hash is stored inside the checkpoint.

**Note on the batch dimension.** The frozen trunk runs on all `batch × frames` images at once,
so *patch-level* outputs (`e_logit`, `b_logit`, `l_logit`, `Z`, `g`, `sc`) stay frame-level and
line up with the per-frame `region` / `boundary` targets of shape `(batch, frames, 28, 28)`.
The two *decision* outputs are folded to one value per clip — `bag = bag_f.reshape(B, T).mean(1)`,
which is exactly i1's `Zv = Zf.reshape(B, T, -1).mean(dim=1)` — so `bag` and `recipe` line up
with the clip-level `y` and `recipe` labels.
""")

# --------------------------------------------------------------------------- 2
code(r'''
# ============================ CONFIG — EVIDENCENET v1 ============================
import os

# This run is the FULL architecture on a short schedule: one arm, no ablation.
# The full-schedule run uses RUN_NAME = "i2_evidence_v1", so the two never share outputs.
RUN_NAME  = "i2_evidence_v1_short"
ABLATION  = "A0"    # A0 = the FULL architecture (every component on). A0 only, for now.
                    # Other arms: A1 i1 convex-mean pooling | A2 no boundary loss
                    # A3 no low-level branch | A4 no consistency | A5 no recipe head
                    # A7 LN-tuning stage 2 | A8 no alignment/uniformity
SEED      = 0

# Kaggle mounts an attached dataset at /kaggle/input/<slug>. resolve_dataset_root() also
# sniffs /kaggle/input and every subdirectory, so any mount point still resolves.
DATA_ROOT = "/kaggle/input/srn-v4b-data"
OUT_DIR   = f"/kaggle/working/outputs/{RUN_NAME}_{ABLATION}"
# keyed on the split, not on RUN_NAME, so the short and full-schedule runs share one cache
CACHE_DIR = "/kaggle/working/frame_cache_i2"

# --- geometry (392 = 28 x 14 -> a clean 28x28 patch grid) --------------------
IMG, PATCH   = 392, 14
GRID         = IMG // PATCH        # 28
NPATCH       = GRID * GRID         # 784
T_FRAMES     = 8                   # frames per training clip
T_TEST       = 32                  # frames per test video
FRAMES_CACHE = 20                  # frames decoded per video into the cache

# --- splits (video-level, must fit the pack on disk) ------------------------
N_TRAIN = dict(ffpp=110, celeb=100, youtube=100)
N_VAL   = dict(ffpp=25,  celeb=25,  youtube=25)
N_CAL   = dict(ffpp=15,  celeb=20,  youtube=20)
TEST_N  = 50

# --- optimisation -----------------------------------------------------------
EPOCHS          = 8               # short first run: full architecture, one arm
BATCH_VIDEOS    = 2
ACCUM_STEPS     = 2               # effective batch = 4 videos = 32 images
LR_PEAK         = 1.5e-4
LR_FLOOR        = 0.05            # multiplier of LR_PEAK at the end
WD              = 0.01
WARMUP_FRAC     = 0.10
GRAD_CLIP       = 1.0
EMA_DECAY       = 0.99            # ~100-step window, matched to a short schedule
LABEL_SMOOTH    = 0.05
TIME_BUDGET_MIN = 150             # short first run; the full schedule uses 420
VAL_MONITOR_N   = 40              # videos used for the per-epoch validation

# --- model ------------------------------------------------------------------
D_MODEL         = 384
TRUNK_NAME      = "facebook/dinov2-with-registers-large"
ADAPTER_HEADS   = 6
ADAPTER_LAYERS  = 2
ADAPTER_DROPOUT = 0.1
TOPK_FRAC       = 0.03            # 24 of 784 patches
LSE_TAU         = 2.0
AGG_DIM         = 3 * 5 + 1 + 2   # 3 score maps x 5 stats + global logit + 2 summaries
N_RECIPE        = 15              # 5 mask regions x 3 blend-ratio buckets

# --- synthesis --------------------------------------------------------------
RATIO_LO, RATIO_HI = 0.15, 0.60   # i1 used 0.15..1.00 -> a global shortcut
POISSON_BLEND      = False        # i1 strategy: expensive and removes the seam
SHARED_PHOTOMETRIC = True         # same jitter distribution for source and for reals
NUM_WORKERS        = 3

# --- evaluation -------------------------------------------------------------
ROBUST_CONDS  = ["clean", "jpeg30", "blur15", "resize50"]
SELECT_METRIC = "worst_auc"
BOOTSTRAP_N   = 2000

# --- ablation switches ------------------------------------------------------
_ABL = {
    "A0": dict(pool="existence", boundary=True,  lowlevel=True,  consistency=True,  recipe=True,  ln_tune=False, align=True),
    "A1": dict(pool="convex",    boundary=True,  lowlevel=True,  consistency=True,  recipe=True,  ln_tune=False, align=True),
    "A2": dict(pool="existence", boundary=False, lowlevel=True,  consistency=True,  recipe=True,  ln_tune=False, align=True),
    "A3": dict(pool="existence", boundary=True,  lowlevel=False, consistency=True,  recipe=True,  ln_tune=False, align=True),
    "A4": dict(pool="existence", boundary=True,  lowlevel=True,  consistency=False, recipe=True,  ln_tune=False, align=True),
    "A5": dict(pool="existence", boundary=True,  lowlevel=True,  consistency=True,  recipe=False, ln_tune=False, align=True),
    "A7": dict(pool="existence", boundary=True,  lowlevel=True,  consistency=True,  recipe=True,  ln_tune=True,  align=True),
    "A8": dict(pool="existence", boundary=True,  lowlevel=True,  consistency=True,  recipe=True,  ln_tune=False, align=False),
}
assert ABLATION in _ABL, f"ABLATION must be one of {sorted(_ABL)}"
ABL = _ABL[ABLATION]

# --- driver flags -----------------------------------------------------------
DRY_RUN       = False   # True -> tiny subset, few steps, validates the whole pipeline
REBUILD_CACHE = True    # rebuild the decoded-frame cache from scratch
USE_DP        = True    # DataParallel when more than one GPU is visible
LN_TUNE       = ABL["ln_tune"]

for _d in (OUT_DIR, f"{OUT_DIR}/figures", f"{OUT_DIR}/evaluation"):
    os.makedirs(_d, exist_ok=True)

CFG = dict(RUN_NAME=RUN_NAME, ABLATION=ABLATION, ABL=ABL, SEED=SEED, DATA_ROOT=DATA_ROOT,
           OUT_DIR=OUT_DIR, CACHE_DIR=CACHE_DIR, IMG=IMG, PATCH=PATCH, GRID=GRID,
           NPATCH=NPATCH, T_FRAMES=T_FRAMES, T_TEST=T_TEST, FRAMES_CACHE=FRAMES_CACHE,
           N_TRAIN=N_TRAIN, N_VAL=N_VAL, N_CAL=N_CAL, TEST_N=TEST_N, EPOCHS=EPOCHS,
           BATCH_VIDEOS=BATCH_VIDEOS, ACCUM_STEPS=ACCUM_STEPS, LR_PEAK=LR_PEAK,
           LR_FLOOR=LR_FLOOR, WD=WD, WARMUP_FRAC=WARMUP_FRAC, GRAD_CLIP=GRAD_CLIP,
           EMA_DECAY=EMA_DECAY, LABEL_SMOOTH=LABEL_SMOOTH, D_MODEL=D_MODEL,
           TRUNK_NAME=TRUNK_NAME, ADAPTER_HEADS=ADAPTER_HEADS,
           ADAPTER_LAYERS=ADAPTER_LAYERS, ADAPTER_DROPOUT=ADAPTER_DROPOUT,
           TOPK_FRAC=TOPK_FRAC, LSE_TAU=LSE_TAU, LN_TUNE=LN_TUNE, N_RECIPE=N_RECIPE,
           RATIO_LO=RATIO_LO, RATIO_HI=RATIO_HI, POISSON_BLEND=POISSON_BLEND,
           SHARED_PHOTOMETRIC=SHARED_PHOTOMETRIC, ROBUST_CONDS=ROBUST_CONDS,
           SELECT_METRIC=SELECT_METRIC, DRY_RUN=DRY_RUN, VAL_MONITOR_N=VAL_MONITOR_N,
           TIME_BUDGET_MIN=TIME_BUDGET_MIN)

print(f"run={RUN_NAME} ablation={ABLATION} -> {ABL}")
print(f"epochs={EPOCHS} | dry_run={DRY_RUN} | ln_tune={LN_TUNE} | output={OUT_DIR}")
''')

# --------------------------------------------------------------------------- 3
code(r'''
# ============================ IMPORTS + DETERMINISM ============================
import io, json, math, glob, time, random, shutil, hashlib, collections
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TVT
from transformers import AutoModel
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, roc_curve,
                             precision_recall_curve)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
N_GPU = torch.cuda.device_count()
AMP   = torch.float16
assert N_GPU in (0, 1, 2), f"expected up to 2 GPUs, found {N_GPU}"


def worker_init(wid):
    s = SEED * 1000 + wid
    random.seed(s); np.random.seed(s)


IMNET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

print("device:", device, "| GPUs:", N_GPU, "| torch:", torch.__version__,
      "| cv2:", cv2.__version__)
for i in range(N_GPU):
    print(f"  gpu{i}: {torch.cuda.get_device_name(i)}")
''')

# --------------------------------------------------------------------------- 4
code(r'''
# ============================ DATA DISCOVERY, SPLITS, LEAKAGE CONTROL ============================
def find_dir(root, name):
    hits = [p for p in Path(root).rglob(name) if p.is_dir()]
    return hits[0] if hits else None


def _has_layout(path):
    if not path.is_dir():
        return False
    names = {p.name.casefold() for p in path.iterdir() if p.is_dir()}
    return "celeb-df-v2" in names and ("dfdcp" in names or "faceforensics++" in names)


def resolve_dataset_root():
    input_root = Path("/kaggle/input")
    candidates = [Path(DATA_ROOT)]
    if input_root.exists():
        candidates.append(input_root)
        candidates += sorted(p for p in input_root.iterdir() if p.is_dir())
        candidates += sorted(p.parent for p in input_root.rglob("Celeb-DF-v2") if p.is_dir())
    seen = set()
    for c in candidates:
        c = c.resolve()
        if str(c) in seen:
            continue
        seen.add(str(c))
        if _has_layout(c):
            return str(c)
    avail = sorted(str(p) for p in input_root.iterdir()) if input_root.exists() else []
    raise AssertionError(f"could not resolve the srn layout; available inputs: {avail}")


DATA_ROOT  = resolve_dataset_root()
FFPP_REAL  = find_dir(DATA_ROOT, "youtube")
CDF_ROOT   = find_dir(DATA_ROOT, "Celeb-DF-v2")
DFDCP_ROOT = find_dir(DATA_ROOT, "DFDCP")
FFPP_ROOT  = find_dir(DATA_ROOT, "FaceForensics++")
assert FFPP_REAL is not None and CDF_ROOT is not None, "FF++ real / Celeb-DF-v2 not found"
print(f"dataset root: {DATA_ROOT}")


def video_frame_dirs(root):
    # -> [(video_id, frames_dir, landmarks_dir_or_None)] sorted deterministically
    out = []
    if root is None or not Path(root).exists():
        return out
    for fr in sorted(Path(root).rglob("frames")):
        lm = fr.parent / "landmarks"
        for vd in sorted(fr.iterdir()):
            if vd.is_dir():
                out.append((vd.name, str(vd), str(lm / vd.name) if lm.exists() else None))
    return out


ffpp_all  = video_frame_dirs(FFPP_REAL)
celeb_all = video_frame_dirs(CDF_ROOT / "Celeb-real")
yt_all    = video_frame_dirs(CDF_ROOT / "YouTube-real")
print(f"on disk: FF++ real {len(ffpp_all)} | Celeb-real {len(celeb_all)} | "
      f"YouTube-real {len(yt_all)}")


def split_pool(pool, n_tr, n_va, n_ca, n_te, tag):
    # Proportions are fixed by N_TRAIN / N_VAL / N_CAL / TEST_N, but if the pack holds fewer
    # videos than i1 saw the split scales down together instead of crashing mid-run. The
    # slices stay contiguous and non-overlapping, so leakage control is unaffected.
    n = len(pool)
    need = n_tr + n_va + n_ca + n_te
    if n < need:
        scale = n / float(need)
        n_tr, n_va, n_ca = (max(1, int(n_tr * scale)), max(1, int(n_va * scale)),
                            max(1, int(n_ca * scale)))
        assert n_tr + n_va + n_ca < n, f"{tag}: {n} videos cannot fill train/val/cal"
        n_te = n - n_tr - n_va - n_ca
        print(f"  {tag}: {n} videos on disk < {need} assumed -> scaled split "
              f"train {n_tr} / val {n_va} / cal {n_ca} / test {n_te}")
    else:
        n_te = min(n_te, n - n_tr - n_va - n_ca)
    a, b, c = n_tr, n_tr + n_va, n_tr + n_va + n_ca
    return pool[:a], pool[a:b], pool[b:c], pool[c:c + n_te]


ffpp_tr, ffpp_va, ffpp_ca, ffpp_te = split_pool(
    ffpp_all, N_TRAIN["ffpp"], N_VAL["ffpp"], N_CAL["ffpp"], TEST_N, "ffpp")
celeb_tr, celeb_va, celeb_ca, celeb_te = split_pool(
    celeb_all, N_TRAIN["celeb"], N_VAL["celeb"], N_CAL["celeb"], TEST_N, "celeb-real")
yt_tr, yt_va, yt_ca, yt_te = split_pool(
    yt_all, N_TRAIN["youtube"], N_VAL["youtube"], N_CAL["youtube"], TEST_N, "youtube-real")

train_vids = {"ffpp": ffpp_tr, "cdf": celeb_tr + yt_tr}
val_vids   = {"ffpp": ffpp_va, "cdf": celeb_va + yt_va}
cal_vids   = {"ffpp": ffpp_ca, "cdf": celeb_ca + yt_ca}

TEST_SETS = {}
if DFDCP_ROOT is not None:
    TEST_SETS["dfdcp_real"] = video_frame_dirs(DFDCP_ROOT / "original_videos")[:TEST_N]
    TEST_SETS["dfdcp_fake"] = (video_frame_dirs(DFDCP_ROOT / "method_A")
                               + video_frame_dirs(DFDCP_ROOT / "method_B"))[:TEST_N]
TEST_SETS["cdf_test_real"] = celeb_te
TEST_SETS["cdf_test_fake"] = video_frame_dirs(CDF_ROOT / "Celeb-synthesis")[:TEST_N]
TEST_SETS["ffpp_test_real"] = ffpp_te
if FFPP_ROOT is not None:
    for manip in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap",
                  "NeuralTextures", "DeepFakeDetection"]:
        got = video_frame_dirs(FFPP_ROOT / "manipulated_sequences" / manip / "c23")
        if got:
            TEST_SETS[f"ffpp_{manip}"] = got[:TEST_N]


def _keys(groups):
    return {f"{d}/{v[0]}" for d, vs in groups.items() for v in vs}


def assert_no_leakage():
    tr, va, ca = _keys(train_vids), _keys(val_vids), _keys(cal_vids)
    assert not (tr & va), f"train/val overlap: {sorted(tr & va)[:5]}"
    assert not (tr & ca), f"train/cal overlap: {sorted(tr & ca)[:5]}"
    assert not (va & ca), f"val/cal overlap: {sorted(va & ca)[:5]}"
    held = {k.split("/", 1)[1] for k in (tr | va | ca)}
    for name, recs in TEST_SETS.items():
        te = {r[0] for r in recs}
        assert not (te & held), f"{name} leaks into train/val/cal: {sorted(te & held)[:5]}"
    return True


assert_no_leakage()
MANIFEST = dict(data_root=DATA_ROOT,
                train={d: [v[0] for v in vs] for d, vs in train_vids.items()},
                val={d: [v[0] for v in vs] for d, vs in val_vids.items()},
                cal={d: [v[0] for v in vs] for d, vs in cal_vids.items()},
                test={k: [r[0] for r in v] for k, v in TEST_SETS.items()})
MANIFEST["hash"] = hashlib.sha1(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()[:12]
CFG["DATA_ROOT_RESOLVED"] = DATA_ROOT
CFG["MANIFEST_HASH"] = MANIFEST["hash"]
Path(OUT_DIR, "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
Path(OUT_DIR, "config.json").write_text(json.dumps(CFG, indent=2, default=str))

n_tr = sum(len(v) for v in train_vids.values())
n_va = sum(len(v) for v in val_vids.values())
n_ca = sum(len(v) for v in cal_vids.values())
print(f"train {n_tr} | val {n_va} | calibration {n_ca} videos")
print("test subsets: " + ", ".join(f"{k}={len(v)}" for k, v in TEST_SETS.items()))
print(f"leakage check passed | manifest {MANIFEST['hash']}")
''')

# --------------------------------------------------------------------------- 5
code(r'''
# ============================ FRAME CACHE (i1 was 30x loader-bound) ============================
# i1 ran 900 batches in 443 min = 29.6 s/step for 32 images ~= 1.1 images/s, while the frozen
# trunk can do 25-40 images/s on 2x T4. The loader was the bottleneck: globbing frame lists,
# decoding and resizing PNGs, and a PIL photometric chain inside __getitem__, per frame.
# Here every frame is decoded and resized exactly once, then reused by every epoch.
# Frames are cached evenly spaced; the *random* frame sampling that gives i2 its extra
# supervision happens per epoch inside ClipDataset, over these cached frames.

CACHE_INDEX_PATH = Path(CACHE_DIR, "index.json")


def pick_frame_files(fdir, n):
    files = sorted(glob.glob(str(Path(fdir) / "*.png")))
    if not files:
        files = sorted(glob.glob(str(Path(fdir) / "*.jpg")))
    assert files, f"no frames in {fdir}"
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx], files


def _read_resize(path, size=IMG):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, np.uint8)


def build_cache(groups, split_name):
    index = {}
    written = 0
    for dom, recs in groups.items():
        index[dom] = []
        for vid, fdir, ldir in recs:
            out = Path(CACHE_DIR, split_name, dom, vid)
            out.mkdir(parents=True, exist_ok=True)
            frames, _ = pick_frame_files(fdir, FRAMES_CACHE)
            fpaths, lpaths = [], []
            for i, fp in enumerate(frames):
                arr = _read_resize(fp)
                ip = out / f"{i:03d}.png"
                Image.fromarray(arr).save(ip)
                fpaths.append(str(ip))
                lp = ""
                if ldir:
                    cand = Path(ldir) / (Path(fp).stem + ".npy")
                    if cand.exists():
                        dp = out / f"{i:03d}.npy"
                        shutil.copyfile(cand, dp)
                        lp = str(dp)
                lpaths.append(lp)
                written += 1
            index[dom].append(dict(vid=vid, frames=fpaths, lms=lpaths))
    print(f"  {split_name}: {len(index)} domains, {written} frames cached", flush=True)
    return index


def load_or_build_cache():
    if CACHE_INDEX_PATH.exists() and not REBUILD_CACHE:
        print(f"frame cache reused from {CACHE_DIR}")
        return json.loads(CACHE_INDEX_PATH.read_text())
    if Path(CACHE_DIR).exists() and REBUILD_CACHE:
        shutil.rmtree(CACHE_DIR)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"building frame cache (one pass) -> {CACHE_DIR}")
    t0 = time.time()
    idx = {"train": build_cache(train_vids, "train"),
           "val": build_cache(val_vids, "val"),
           "cal": build_cache(cal_vids, "cal")}
    CACHE_INDEX_PATH.write_text(json.dumps(idx))
    print(f"frame cache ready in {time.time() - t0:.1f}s")
    return idx


if DRY_RUN:
    # tiny subset so DRY_RUN finishes in minutes rather than tens of minutes
    for _g in (train_vids, val_vids, cal_vids):
        for _d in list(_g):
            _g[_d] = _g[_d][:6]
    TEST_SETS = {k: v[:6] for k, v in TEST_SETS.items()}
    print("DRY_RUN: splits truncated")

CACHE = load_or_build_cache()

if DRY_RUN:
    for _s in ("train", "val", "cal"):
        for _d in list(CACHE[_s]):
            CACHE[_s][_d] = CACHE[_s][_d][:6]

n_cached = sum(len(v) for s in CACHE.values() for v in s.values())
print(f"cached video entries: {n_cached}")
''')

# --------------------------------------------------------------------------- 6
code(r'''
# ============================ SELF-BLENDED SYNTHESIS (numpy/cv2, hard) ============================
# Every difference from i1 targets a specific shortcut found in the i1 audit:
#   ratio capped at 0.60      (i1 allowed 1.0 -> a global colour cue)
#   identical photometric draw for the source and for real frames (i1 jittered the source only)
#   narrow seam blur 1.5-5.0  (i1 used 2-8 -> the "seam" was a wide gradient)
#   numpy alpha blend         (i1's cv2.seamlessClone removed the very seam we supervise and
#                              cost tens of ms per frame)
# The generator also returns the raw soft mask m, so the LAA-Net / Face X-ray boundary
# target B = 4*m*(1-m) is exactly computable.

MASK_REGIONS  = ["full", "upper", "lower", "middle", "ellipse"]
RATIO_BUCKETS = [(0.15, 0.30), (0.30, 0.45), (0.45, 0.60)]
RECIPE_REGIONS = {r: i for i, r in enumerate(MASK_REGIONS)}


def _ellipse_mask(size, rng):
    m = Image.new("L", size, 0)
    w, h = size
    cx, cy = w * 0.5, h * 0.5
    rx, ry = w * rng.uniform(0.34, 0.42), h * rng.uniform(0.40, 0.47)
    ImageDraw.Draw(m).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return np.asarray(m, np.float32) / 255.0


def _soft_mask(lm_path, size, rng, region):
    # raw geometric mask in [0,1]; 1 = fully source, 0 = fully target
    if region == "ellipse" or not lm_path or not Path(lm_path).exists():
        a = _ellipse_mask(size, rng)
    else:
        m = Image.new("L", size, 0)
        try:
            lm = np.load(lm_path).astype(np.float32).reshape(-1, 2)
            pts = lm * np.array([size[0] / 256.0, size[1] / 256.0], np.float32)
            jit = rng.uniform(0.0, 0.012)
            pts = pts + rng.normal(0.0, jit * size[0], pts.shape).astype(np.float32)
            ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=255)
            a = np.asarray(m, np.float32) / 255.0
            if a.mean() < 0.01:
                a = _ellipse_mask(size, rng)
        except Exception:
            a = _ellipse_mask(size, rng)
    if region not in ("full", "ellipse"):
        h = a.shape[0]
        band = np.zeros_like(a)
        if region == "upper":
            y0, y1 = 0.0, rng.uniform(0.45, 0.62)
        elif region == "lower":
            y0, y1 = rng.uniform(0.38, 0.55), 1.0
        else:
            y0, y1 = rng.uniform(0.25, 0.35), rng.uniform(0.65, 0.78)
        band[int(h * y0):int(h * y1)] = 1.0
        a = a * band
    return np.clip(cv2.GaussianBlur(a, (0, 0), rng.uniform(1.5, 5.0)), 0.0, 1.0)


def sample_photometric(rng):
    return dict(bright=rng.uniform(0.85, 1.18) if rng.random() < 0.8 else 1.0,
                contrast=rng.uniform(0.85, 1.18) if rng.random() < 0.8 else 1.0,
                noise=rng.uniform(0.5, 3.0) if rng.random() < 0.5 else 0.0,
                blur=rng.uniform(0.2, 1.0) if rng.random() < 0.4 else 0.0)


def apply_photometric(arr_u8, p):
    out = arr_u8.astype(np.float32) * p["bright"]
    out = 128.0 + (out - 128.0) * p["contrast"]
    if p["noise"] > 0:
        out = out + np.random.normal(0.0, p["noise"], out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if p["blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    return out


def make_sbi(target_u8, lm_path, src_u8, rng, hard=False):
    # -> blended uint8, raw soft mask m, boundary target B, binary region target, recipe id
    H = W = IMG
    region = rng.choice(MASK_REGIONS[1:] if hard else MASK_REGIONS)
    m = _soft_mask(lm_path, (W, H), rng, region)
    bkt = rng.randrange(2 if hard else len(RATIO_BUCKETS))   # "hard" = a subtler blend
    lo, hi = RATIO_BUCKETS[bkt]
    ratio = rng.uniform(lo, hi)

    src = target_u8 if src_u8 is None else src_u8
    if src.shape[:2] != (H, W):
        src = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR)
    p = sample_photometric(rng)
    # SHARED_PHOTOMETRIC: one identical draw applied to the source and to real frames,
    # so fakes carry no global colour/brightness signature relative to reals.
    src_p = apply_photometric(src, p)
    tgt_p = apply_photometric(target_u8, p) if SHARED_PHOTOMETRIC else target_u8

    s = rng.uniform(0.90, 1.08)
    A = np.array([[s, 0.0, rng.uniform(-0.05, 0.05) * W],
                  [0.0, s, rng.uniform(-0.05, 0.05) * H]], np.float32)
    src_w = cv2.warpAffine(src_p, A, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    w = (m * ratio).astype(np.float32)[..., None]
    blended = np.clip(src_w.astype(np.float32) * w + tgt_p.astype(np.float32) * (1.0 - w),
                      0, 255).astype(np.uint8)
    B = np.clip(4.0 * m * (1.0 - m), 0.0, 1.0).astype(np.float32)
    region_bin = ((m * ratio) > 0.02).astype(np.float32)
    recipe = RECIPE_REGIONS[region] * len(RATIO_BUCKETS) + bkt
    return blended, m, B, region_bin, recipe


def intervene_fixed(arr_u8, cond):
    if cond == "clean":
        return arr_u8
    if cond == "jpeg30":
        ok, buf = cv2.imencode(".jpg", arr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else arr_u8
    if cond == "blur15":
        return cv2.GaussianBlur(arr_u8, (0, 0), 1.5)
    if cond == "resize50":
        h, w = arr_u8.shape[:2]
        small = cv2.resize(arr_u8, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"unknown condition {cond}")


def intervene_random(arr_u8, rng):
    kind = rng.randrange(4)
    if kind == 0:
        q = rng.choice([30, 50, 70])
        ok, buf = cv2.imencode(".jpg", arr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        return (cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else arr_u8), kind
    if kind == 1:
        return cv2.GaussianBlur(arr_u8, (0, 0), rng.uniform(0.5, 1.5)), kind
    if kind == 2:
        h, w = arr_u8.shape[:2]
        sc = rng.choice([0.5, 0.75])
        small = cv2.resize(arr_u8, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), kind
    p = dict(bright=rng.uniform(0.8, 1.2), contrast=1.0, noise=0.0, blur=0.0)
    return apply_photometric(arr_u8, p), kind


print("synthesis + intervention helpers ready")
''')

# --------------------------------------------------------------------------- 7
code(r'''
# ============================ DATASETS ============================
def to_tensor(arr_u8):
    t = torch.from_numpy(np.ascontiguousarray(arr_u8)).permute(2, 0, 1).float().div_(255.0)
    return IMNET(t)


def pool_region(region_bin):
    # soft fraction per cell: mean pooling handles partial coverage
    t = torch.from_numpy(np.ascontiguousarray(region_bin))[None, None]
    return F.adaptive_avg_pool2d(t, GRID)[0, 0]


def pool_boundary(B):
    # a seam is thin -> keep the max inside each cell, otherwise it disappears
    t = torch.from_numpy(np.ascontiguousarray(B))[None, None]
    return F.adaptive_max_pool2d(t, GRID)[0, 0]


def _read_cached(paths, idx):
    return _read_resize(paths[idx])


class ClipDataset(Dataset):
    # Emits one clip: (x, xi, region, boundary, y, recipe, is_fake, vid)
    #   x, xi            (T_FRAMES, 3, IMG, IMG)
    #   region, boundary (T_FRAMES, GRID, GRID)   <- one target per frame
    #   y, is_fake       scalar tensors; recipe a scalar long
    # Deterministic per (index, epoch salt). Training bumps the salt every epoch, which is
    # what turns 20 cached frames per video into effectively unlimited fresh supervision.
    def __init__(self, index, groups, hard=False, cond=None, two_views=True):
        self.groups = list(groups.keys())
        self.recs = {}
        for d in self.groups:
            keep = {r[0] for r in groups[d]}
            got = [r for r in index[d] if r["vid"] in keep]
            assert got, f"no cached frames for domain {d} (cache/vids mismatch)"
            self.recs[d] = got
        self.hard = hard
        self.cond = cond
        self.two_views = two_views
        self.salt = 0
        self.length = 2 * sum(len(v) for v in self.recs.values())

    def __len__(self):
        return self.length

    def set_epoch(self, e):
        self.salt = int(e)

    def __getitem__(self, i):
        rng = random.Random(SEED * 100003 + 7919 * self.salt + int(i))
        np.random.seed((SEED * 100003 + 7919 * self.salt + int(i)) % (2 ** 31 - 1))
        dom = self.groups[rng.randrange(len(self.groups))]
        rec = self.recs[dom][rng.randrange(len(self.recs[dom]))]
        is_fake = bool(rng.random() < 0.5)
        nf = len(rec["frames"])
        picks = [rng.randrange(nf) for _ in range(T_FRAMES)]

        # one target per frame: the model folds (batch, frames) into a single dimension, so
        # the losses need T_FRAMES region/boundary maps per clip, not one.
        xs, regs, bnds, recipe_full = [], [], [], 0
        if is_fake:
            for k, pi in enumerate(picks):
                tgt = _read_cached(rec["frames"], pi)
                src = None
                if nf > 1 and rng.random() < 0.5:          # cross-frame blend source
                    src = _read_cached(rec["frames"], (pi + rng.randrange(1, nf)) % nf)
                blend, m, B, reg, recipe = make_sbi(tgt, rec["lms"][pi] or None, src, rng,
                                                    hard=self.hard)
                xs.append(blend)
                regs.append(pool_region(reg))
                bnds.append(pool_boundary(B))
                if k == 0:
                    recipe_full = recipe
        else:
            for pi in picks:
                arr = _read_cached(rec["frames"], pi)
                xs.append(apply_photometric(arr, sample_photometric(rng)))
                regs.append(torch.zeros(GRID, GRID))
                bnds.append(torch.zeros(GRID, GRID))

        x = torch.stack([to_tensor(a) for a in xs])
        if self.two_views:
            xi = torch.stack([to_tensor(intervene_random(a, rng)[0] if self.cond is None
                                        else intervene_fixed(a, self.cond)) for a in xs])
        else:
            xi = x.clone()

        return (x, xi, torch.stack(regs), torch.stack(bnds),
                torch.tensor(1.0 if is_fake else 0.0),
                torch.tensor(recipe_full, dtype=torch.long),
                torch.tensor(1.0 if is_fake else 0.0),
                f"{dom}/{rec['vid']}")


class TestVideoDataset(Dataset):
    # Untouched test videos at T_TEST frames. No synthesis, no intervention.
    def __init__(self, records, t=None, size=IMG):
        self.records = records
        self.t = t or T_TEST
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        vid, fdir, _ = self.records[idx]
        frames, _ = pick_frame_files(fdir, self.t)
        arrs = [_read_resize(f, self.size) for f in frames]
        return torch.stack([to_tensor(a) for a in arrs]), vid


class InDatasetTestDataset(Dataset):
    # "Within dataset": held-out videos from the SAME pools the model trained on, pushed through
    # the SAME pixel pipeline. mode="synth" applies the training synthesis recipe (+ the same
    # photometric draw make_sbi uses internally); mode="real" applies only that photometric draw,
    # so real and fake sides of this test share the training distribution and no global-jitter
    # shortcut exists at evaluation time. mode="synth" also mirrors training's cross-frame blend
    # source (50% of samples), so the in-dataset fake distribution matches training exactly.
    def __init__(self, records, t=None, mode="synth", hard=False):
        assert mode in ("real", "synth"), mode
        self.records = records
        self.t = t or T_TEST
        self.mode = mode
        self.hard = hard

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        vid, fdir, ldir = self.records[idx]
        seed = SEED * 100003 + 104729 * idx + (0 if self.mode == "real" else 1)
        rng = random.Random(seed)
        np.random.seed(seed % (2 ** 31 - 1))
        frames, _ = pick_frame_files(fdir, self.t)
        arrs = []
        for k, fp in enumerate(frames):
            arr = _read_resize(fp)
            if self.mode == "real":
                arrs.append(apply_photometric(arr, sample_photometric(rng)))
                continue
            src = None
            if len(frames) > 1 and rng.random() < 0.5:  # cross-frame source, exactly as in training
                src = _read_resize(frames[(k + rng.randrange(1, len(frames))) % len(frames)])
            lm = ""
            if ldir:
                cand = Path(ldir) / (Path(fp).stem + ".npy")
                lm = str(cand) if cand.exists() else ""
            blend, _m, _B, _reg, _recipe = make_sbi(arr, lm or None, src, rng, hard=self.hard)
            arrs.append(blend)
        return torch.stack([to_tensor(a) for a in arrs]), vid


def make_loader(ds, bs=None, shuffle=False, workers=None):
    workers = NUM_WORKERS if workers is None else workers
    return DataLoader(ds, batch_size=bs or BATCH_VIDEOS, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None,
                      worker_init_fn=worker_init if workers > 0 else None)


print("datasets ready")
''')

# --------------------------------------------------------------------------- 8
code(r'''
# ============================ MODEL — EvidenceNet v1 ============================
class ForensicAdapter(nn.Module):
    # 1024 -> 384, two pre-norm transformer blocks. The caller L2-normalises the output
    # (GenD: hyperspherical geometry is what makes the boundary/consistency losses sane).
    def __init__(self, din=1024, d=D_MODEL, heads=ADAPTER_HEADS, layers=ADAPTER_LAYERS,
                 dropout=ADAPTER_DROPOUT):
        super().__init__()
        self.proj = nn.Linear(din, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=d * 4,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        z = self.proj(tokens)
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)


class PatchHead(nn.Module):
    def __init__(self, d=D_MODEL, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(),
                                 nn.Linear(h, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def lowlevel_residuals(x):
    # fixed, non-learned high-pass + local range: a 14x14 ViT patch cannot see a 1-3 px seam
    hp = x - F.avg_pool2d(x, 3, 1, 1)
    lr = F.max_pool2d(x, 3, 1, 1) + F.max_pool2d(-x, 3, 1, 1)
    return torch.cat([hp, lr], dim=1)


class LowLevelBranch(nn.Module):
    def __init__(self, out_dim=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), nn.GELU())

    def forward(self, x):
        f = self.net(lowlevel_residuals(x))
        f = F.adaptive_avg_pool2d(f, (GRID, GRID))
        return f.flatten(2).transpose(1, 2)


def pool_feats(s, mode):
    # s: (M, NPATCH) per-patch logits -> (M, 5) statistics of the evidence map.
    # mode="existence" is the i2 fix. mode="convex" reproduces i1's average pooling exactly
    # (every "max-like" slot collapses to the mean) for ablation A1.
    s = s.float()
    P = s.shape[-1]
    if mode == "existence":
        k = max(1, int(round(P * TOPK_FRAC)))
        a = s.topk(k, dim=-1).values.mean(-1)
        b = torch.logsumexp(LSE_TAU * s, dim=-1) / LSE_TAU - math.log(P) / LSE_TAU
        c = s.max(-1).values
    else:
        mu = s.mean(-1)
        a = b = c = mu
    return torch.stack([a, b, c, s.mean(-1), s.std(-1, unbiased=False)], -1)


class EvidenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = AutoModel.from_pretrained(TRUNK_NAME)
        self.trunk.requires_grad_(False)
        self.prefix = 1 + getattr(self.trunk.config, "num_register_tokens", 0)
        din = self.trunk.config.hidden_size

        self.adapter = ForensicAdapter(din)
        self.cls_proj = nn.Linear(din, D_MODEL)
        self.head_e = PatchHead()          # evidence: "is this patch forged?"
        self.head_b = PatchHead()          # boundary: "is this patch on the seam?"
        self.head_l = PatchHead()          # evidence from the residual branch
        self.head_global = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 256),
                                         nn.GELU(), nn.Linear(256, 1))
        self.head_sc = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 64), nn.GELU(),
                                     nn.Linear(64, 1))
        self.head_recipe = nn.Sequential(nn.LayerNorm(2 * D_MODEL),
                                         nn.Linear(2 * D_MODEL, 256), nn.GELU(),
                                         nn.Linear(256, N_RECIPE))
        self.lowlevel = LowLevelBranch()
        self.aggregator = nn.Sequential(nn.LayerNorm(AGG_DIM), nn.Linear(AGG_DIM, 64),
                                        nn.GELU(), nn.Linear(64, 1))
        if LN_TUNE:
            for m in self.trunk.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval()      # trunk dropout off even while training the head
        return self

    def forward(self, x, xi=None):
        B, T = x.shape[0], x.shape[1]
        N = B * T
        px = x.reshape(N, *x.shape[2:])
        if xi is not None:
            px = torch.cat([px, xi.reshape(N, *xi.shape[2:])], dim=0)

        with torch.no_grad():
            out = self.trunk(px)
        tok = out.last_hidden_state[:, self.prefix:, :]
        cls = out.last_hidden_state[:, 0, :]
        assert tok.shape[1] == NPATCH, f"expected {NPATCH} patch tokens, got {tok.shape[1]}"

        Z = F.normalize(self.adapter(tok), dim=-1)
        g = F.normalize(self.cls_proj(cls), dim=-1)

        e_logit = self.head_e(Z)
        b_logit = self.head_b(Z)
        b_in = b_logit if ABL["boundary"] else torch.zeros_like(b_logit)
        if ABL["lowlevel"]:
            l_logit = self.head_l(F.normalize(self.lowlevel(px), dim=-1))
        else:
            l_logit = torch.zeros_like(e_logit)

        stats = torch.stack([pool_feats(e_logit, ABL["pool"]),
                             pool_feats(b_in, ABL["pool"]),
                             pool_feats(l_logit, ABL["pool"])], dim=1).flatten(1)
        ev = torch.sigmoid(e_logit)
        extra = torch.stack([ev.mean(-1), ev.std(-1, unbiased=False)], -1)
        agg_in = torch.cat([stats, self.head_global(g), extra], -1)
        assert agg_in.shape[-1] == AGG_DIM, agg_in.shape
        bag_f = self.aggregator(agg_in).squeeze(-1)                     # per frame (N,)
        recipe_f = self.head_recipe(torch.cat([Z.mean(1), g], -1))       # per frame (N, 15)

        anchor = b_in.argmax(-1, keepdim=True)
        z_anchor = Z.gather(1, anchor[..., None].expand(-1, 1, Z.shape[-1])).squeeze(1)
        cos = (Z * z_anchor[:, None, :]).sum(-1)
        dif = (Z - z_anchor[:, None, :]).abs().mean(-1)
        sc = self.head_sc(torch.stack([cos, dif], -1)).squeeze(-1)

        # Patch-level outputs stay per frame (region / boundary targets are per frame). Only the
        # two DECISION outputs are folded to clip level, exactly as i1 did with
        # `Zv = Zf.reshape(B, T, -1).mean(dim=1)`, so `bag` / `recipe` line up with the clip-level
        # `y` and `recipe` labels instead of with the frame grid.
        frame_out = dict(bag=bag_f, recipe=recipe_f, e_logit=e_logit, b_logit=b_in,
                         l_logit=l_logit, g=g, Z=Z, sc=sc)
        if xi is None:
            return self._to_clip(frame_out, B, T)
        return (self._to_clip({k: v[:N] for k, v in frame_out.items()}, B, T),
                self._to_clip({k: v[N:] for k, v in frame_out.items()}, B, T))

    @staticmethod
    def _to_clip(o, B, T):
        # existence within a frame -> mean over the clip's frames (i1 / GenD video score)
        o["bag"] = o["bag"].reshape(B, T).mean(1)
        o["recipe"] = o["recipe"].reshape(B, T, -1).mean(1)
        return o


def unwrap(m):
    # eval/scoring paths use the unwrapped module: a batch smaller than the GPU count
    # would otherwise give DataParallel an empty replica chunk (crash inside BatchNorm)
    return m.module if isinstance(m, nn.DataParallel) else m


def strip_sd(sd):
    # checkpoints drop any "module." prefix so they reload into the unwrapped module
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


net = EvidenceNet()
n_total = sum(p.numel() for p in net.parameters())
n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"params: total={n_total/1e6:.1f}M frozen={(n_total-n_train)/1e6:.1f}M "
      f"trainable={n_train/1e6:.2f}M | agg_dim={AGG_DIM} | ln_tune={LN_TUNE}")
assert n_train > 1e6, "unexpectedly few trainable parameters"
''')

# --------------------------------------------------------------------------- 9
code(r'''
# ============================ OBJECTIVE + CURRICULUM ============================
# Three stages, every weight a fraction of total steps with a 5% ramp at each boundary.
# No absolute step thresholds: i1's v3.1 post-mortem showed absolute schedules can activate
# after training has ended, creating a train/eval graph mismatch.
PHASE_FRACTIONS = dict(locate_end=0.25, aggregate_end=0.55)
RAMP_FRAC = 0.05


def ramp(frac, start, width=RAMP_FRAC):
    return float(np.clip((frac - start) / max(1e-6, width), 0.0, 1.0))


def stage_of(frac):
    if frac < PHASE_FRACTIONS["locate_end"]:
        return 1
    return 2 if frac < PHASE_FRACTIONS["aggregate_end"] else 3


def stage_weights(frac):
    s2 = ramp(frac, PHASE_FRACTIONS["locate_end"])
    s3 = ramp(frac, PHASE_FRACTIONS["aggregate_end"])
    return dict(bag=1.0, region=1.0, boundary=1.0, dice=0.5,
                mil=0.5 * s2, cons=0.2 * s2, sc=0.1 * s2,
                align=0.2 * s3, unif=0.2 * s3, recipe=0.1 * s3)


def _bce(logits, target):
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())


def soft_dice(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    num = 2.0 * (p * target.float()).sum() + eps
    den = p.sum() + target.float().sum() + eps
    return 1.0 - num / den


def lse_pool(logits, tau=LSE_TAU):
    s = logits.float()
    P = s.shape[-1]
    return torch.logsumexp(tau * s, dim=-1) / tau - math.log(P) / tau


def to_frames(v, T=T_FRAMES):
    # clip-level (B, ...) -> frame-level (B*T, ...) to match the model's folded batch dim
    return (v.reshape(-1, *v.shape[1:]).unsqueeze(1)
            .expand(-1, T, *v.shape[1:]).reshape(-1, *v.shape[1:]))


def align_uniform(z, y):
    # Wang & Isola (ICML'20), used by GenD on the L2-normalised embedding
    d2 = (2.0 - 2.0 * (z @ z.t())).clamp_min(0.0)
    n = z.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=z.device)
    same = (y[:, None] == y[None, :]) & off
    l_align = d2[same].mean() if same.any() else z.sum() * 0.0
    l_unif = torch.log(torch.exp(-2.0 * d2[off]).mean().clamp_min(1e-12))
    return l_align, l_unif


def compute_losses(o, oi, region, boundary, y, recipe, is_fake):
    # region / boundary and every patch-level output: per frame (B*T_FRAMES, ...)
    # bag / recipe and every label: clip level (B, ...)          <- i1's convention
    y_f = to_frames(y).reshape(-1)                       # (B*T,) per-frame labels, view o
    y_ff = torch.cat([y_f, y_f], 0)                     # (2*B*T,) for o and oi concatenated
    fk = to_frames(is_fake).reshape(-1) > 0.5           # per-frame fake mask
    fk_c = is_fake.reshape(-1) > 0.5                    # clip-level fake mask
    y_s = y.reshape(-1).float() * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
    parts = {}

    parts["bag"] = 0.5 * (_bce(o["bag"], y_s) + _bce(oi["bag"], y_s))
    reg_g = region.reshape(-1, GRID * GRID).float()
    bnd_g = boundary.reshape(-1, GRID * GRID).float()
    parts["region"] = 0.5 * (_bce(o["e_logit"], reg_g) + _bce(oi["e_logit"], reg_g))
    parts["dice"] = 0.5 * (soft_dice(o["e_logit"], reg_g) + soft_dice(oi["e_logit"], reg_g))
    if ABL["boundary"]:
        parts["boundary"] = 0.5 * (_bce(o["b_logit"], bnd_g) + _bce(oi["b_logit"], bnd_g))
    else:
        parts["boundary"] = o["bag"].sum() * 0.0

    # MIL: a real clip has no forged patch anywhere (dense all-negative supervision); a fake
    # clip is positive if its strongest patch is positive (LSE soft-max over patches).
    parts["mil"] = o["bag"].sum() * 0.0
    if fk.any():
        lp = lse_pool(torch.cat([o["e_logit"][fk], oi["e_logit"][fk]], 0))
        parts["mil"] = parts["mil"] + _bce(lp, torch.ones_like(lp))
    if (~fk).any():
        rl = torch.cat([o["e_logit"][~fk], oi["e_logit"][~fk]], 0)
        parts["mil"] = parts["mil"] + _bce(rl, torch.zeros_like(rl))

    if ABL["consistency"]:
        pe_o, pe_i = torch.sigmoid(o["e_logit"].float()), torch.sigmoid(oi["e_logit"].float())
        pb_o, pb_i = torch.sigmoid(o["b_logit"].float()), torch.sigmoid(oi["b_logit"].float())
        parts["cons"] = ((pe_o - pe_i) ** 2).mean() + ((pb_o - pb_i) ** 2).mean()
    else:
        parts["cons"] = o["bag"].sum() * 0.0

    if fk.any():
        side = reg_g[fk] >= 0.5
        anchor = o["b_logit"][fk].argmax(-1, keepdim=True)
        parts["sc"] = _bce(o["sc"][fk], (side == side.gather(1, anchor)).float())
    else:
        parts["sc"] = o["bag"].sum() * 0.0

    if ABL["align"]:
        # g is per frame and the two views are concatenated, so the target is the per-frame
        # label repeated once per view (this is what makes the (2N, 2N) distance matrix square).
        parts["align"], parts["unif"] = align_uniform(torch.cat([o["g"], oi["g"]], 0), y_ff)
    else:
        parts["align"] = parts["unif"] = o["bag"].sum() * 0.0

    if ABL["recipe"] and fk_c.any():
        parts["recipe"] = F.cross_entropy(o["recipe"][fk_c].float(),
                                          recipe.reshape(-1)[fk_c].long())
    else:
        parts["recipe"] = o["bag"].sum() * 0.0
    return parts


def total_loss(parts, frac):
    w = stage_weights(frac)
    L = torch.zeros((), device=parts["bag"].device)
    for k, lam in w.items():
        if lam > 0.0 and k in parts:
            L = L + lam * parts[k]
    return L, w


print("objective ready | stages:", PHASE_FRACTIONS)
''')

# --------------------------------------------------------------------------- 10
code(r'''
# ============================ METRICS ============================
def eer_from_scores(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def ece_score(y, p, bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def bootstrap_auc(y, p, n=BOOTSTRAP_N, seed=SEED):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return dict(auc=float("nan"), lo=float("nan"), hi=float("nan"))
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return dict(auc=float(roc_auc_score(y, p)),
                lo=float(np.percentile(vals, 2.5)) if vals else float("nan"),
                hi=float(np.percentile(vals, 97.5)) if vals else float("nan"))


def classification_metrics(y, p, thr=0.5, prefix=""):
    keys = ["auc", "ap", "eer", "acc", "prec", "recall", "f1", "bacc", "ece", "brier",
            "tn", "fp", "fn", "tp"]
    if len(y) == 0 or len(set(np.asarray(y).tolist())) < 2:
        return {prefix + k: float("nan") for k in keys}
    hard = [int(v >= thr) for v in p]
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel().tolist()
    return {
        prefix + "auc": float(roc_auc_score(y, p)),
        prefix + "ap": float(average_precision_score(y, p)),
        prefix + "eer": eer_from_scores(y, p),
        prefix + "acc": float(accuracy_score(y, hard)),
        prefix + "prec": float(precision_score(y, hard, zero_division=0)),
        prefix + "recall": float(recall_score(y, hard, zero_division=0)),
        prefix + "f1": float(f1_score(y, hard, zero_division=0)),
        prefix + "bacc": float(balanced_accuracy_score(y, hard)),
        prefix + "ece": ece_score(y, p),
        prefix + "brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)),
        prefix + "tn": int(tn), prefix + "fp": int(fp),
        prefix + "fn": int(fn), prefix + "tp": int(tp),
    }


def localisation_metrics(m_prob, gt, max_points=200_000, seed=SEED):
    nan = dict(pixel_auc=float("nan"), iou=float("nan"), mask_f1=float("nan"))
    if not len(m_prob):
        return nan
    p = np.concatenate([a.ravel() for a in m_prob])
    t = (np.concatenate([a.ravel() for a in gt]) > 0.5).astype(np.int32)
    if t.sum() == 0 or t.sum() == len(t):
        return nan
    if len(p) > max_points:
        i = np.random.RandomState(seed).choice(len(p), max_points, replace=False)
        p, t = p[i], t[i]
    prec, rec, thr = precision_recall_curve(t, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    k = int(np.nanargmax(f1))
    best = float(thr[min(k, len(thr) - 1)]) if len(thr) else 0.5
    pred = p >= best
    inter = float((pred & (t > 0)).sum())
    union = float((pred | (t > 0)).sum())
    return dict(pixel_auc=float(roc_auc_score(t, p)), iou=inter / max(union, 1.0),
                mask_f1=float(np.nanmax(f1)))


def fit_temperature(z, y):
    # grid search: no optimizer dependency, fully reproducible
    z = np.asarray(z, float); y = np.asarray(y, float)
    grid = np.exp(np.linspace(-1.0, 1.5, 80))
    nlls = []
    for T in grid:
        p = 1.0 / (1.0 + np.exp(-z / T))
        nlls.append(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    i = int(np.argmin(nlls))
    return float(grid[i]), float(nlls[i])


@torch.no_grad()
def evaluate(ds, want_maps=False):
    # Runs on the UNWRAPPED module: a batch smaller than the GPU count would otherwise
    # hand DataParallel an empty replica chunk and crash inside BatchNorm.
    model = unwrap(net)
    model.eval()
    ld = DataLoader(ds, batch_size=BATCH_VIDEOS, shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=True)
    ys, ps, zs, doms, e_maps, region_gts = [], [], [], [], [], []
    for xb, _xib, rb, _bb, yb, _recb, fkb, domb in ld:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
            o = model(xb, None)
        z = o["bag"].float().reshape(-1)
        zs += z.cpu().tolist()
        ps += torch.sigmoid(z).cpu().tolist()
        ys += yb.reshape(-1).tolist()
        doms += list(domb)
        if want_maps:
            ev = torch.sigmoid(o["e_logit"].float()).reshape(-1, GRID, GRID).cpu().numpy()
            gt = rb.reshape(-1, GRID, GRID).numpy()
            fk = to_frames(fkb).reshape(-1).numpy().astype(bool)
            for i in range(len(ev)):
                if fk[i]:
                    e_maps.append(ev[i]); region_gts.append(gt[i])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(y=ys, p=ps, z=zs, dom=doms, e_maps=e_maps, region_gt=region_gts)


def full_validation(conds=None, want_maps=True, monitor_n=VAL_MONITOR_N):
    # Per-epoch monitoring runs on a subset so validation does not dominate the wall clock:
    # ~80 clips x 8 frames ~= 25 s, versus ~7 min for the complete val split.
    per_dom = max(1, monitor_n // max(1, len(val_vids)))
    vids = {d: v[:per_dom] for d, v in val_vids.items()}
    res, aucs = {}, {}
    for cond in (conds or ROBUST_CONDS):
        ds = ClipDataset(CACHE["val"], vids, hard=True, cond=cond, two_views=False)
        raw = evaluate(ds, want_maps=(want_maps and cond == "clean"))
        thr = youden_threshold(raw["y"], raw["p"]) if len(set(raw["y"])) > 1 else 0.5
        met = classification_metrics(raw["y"], raw["p"], thr=0.5)
        met.update(classification_metrics(raw["y"], raw["p"], thr=thr, prefix="cal_"))
        met["threshold"] = thr
        if cond == "clean":
            met.update({f"loc_{k}": v for k, v in
                        localisation_metrics(raw["e_maps"], raw["region_gt"]).items()})
            res["_raw"] = (raw["y"], raw["p"])
        res[cond] = met
        aucs[cond] = met["auc"]
    finite = [v for v in aucs.values() if np.isfinite(v)]
    res["_summary"] = dict(
        worst_auc=float(min(finite)) if finite else float("nan"),
        mean_auc=float(np.mean(finite)) if finite else float("nan"),
        clean_auc=aucs.get("clean", float("nan")),
        robust_gap=float(aucs.get("clean", np.nan) - min(finite)) if finite else float("nan"),
        per_cond=aucs)
    return res


print("metrics ready")
''')

# --------------------------------------------------------------------------- 11
code(r'''
# ============================ SELF-TEST + PREFLIGHT ============================
# A long unattended run must fail fast and loudly rather than 90 minutes into training.
_pf_start = time.time()
print("=" * 74)
print("SELF-TEST — aggregation, shapes, gradients, data")
print("=" * 74)

# --- the central claim: existence sees a sparse hotspot, convex averaging does not ------
_probe = torch.full((1, NPATCH), -6.0, device=device)
_probe[0, :24] = 6.0
_stat_ex = pool_feats(_probe, "existence")[0, :3]
_stat_cv = pool_feats(_probe, "convex")[0, :3]
print(f"sparse-hotspot probe: existence={[round(float(v), 2) for v in _stat_ex]} "
      f"convex={[round(float(v), 2) for v in _stat_cv]}")
assert float(_stat_ex.max()) > 3.0, "existence pooling failed to see the hotspot"
assert float(_stat_cv.max()) < 0.0, "convex pooling should dilute the hotspot (i1's bug)"
print("aggregation sanity ok: existence sees the hotspot, convex averaging does not")

# --- data pipeline ---------------------------------------------------------------------
_ds = ClipDataset(CACHE["train"], {d: v[:2] for d, v in train_vids.items()}, hard=False)
_b = _ds[0]
assert _b[0].shape == (T_FRAMES, 3, IMG, IMG), _b[0].shape
assert _b[1].shape == (T_FRAMES, 3, IMG, IMG), _b[1].shape
assert _b[2].shape == (T_FRAMES, GRID, GRID), _b[2].shape
assert _b[3].shape == (T_FRAMES, GRID, GRID), _b[3].shape
assert 0.0 <= float(_b[2].min()) and float(_b[2].max()) <= 1.0
assert 0.0 <= float(_b[3].min()) and float(_b[3].max()) <= 1.0
assert int(_b[5]) < N_RECIPE
print(f"dataset ok | x={tuple(_b[0].shape)} region_max={float(_b[2].max()):.2f} "
      f"seam_max={float(_b[3].max()):.2f} y={float(_b[4]):.0f} recipe={int(_b[5])}")

_ld = make_loader(_ds, bs=2, shuffle=True, workers=0)
_xb, _xib, _rb, _bb, _yb, _recb, _fkb, _db = next(iter(_ld))
assert _xb.shape == (2, T_FRAMES, 3, IMG, IMG), _xb.shape
assert _rb.shape == (2, T_FRAMES, GRID, GRID), _rb.shape
assert not torch.isnan(_xb).any() and not torch.isnan(_xib).any()
print(f"loader ok | batch={tuple(_xb.shape)} | {_db[0]}")

# --- model forward/backward (folded batch = videos x frames) ----------------------------
net = net.to(device)
if N_GPU > 1 and USE_DP:
    net = nn.DataParallel(net)
    print(f"DataParallel over {N_GPU} GPUs")

with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
    _o, _oi = net(_xb.to(device), _xib.to(device))
_parts = compute_losses(_o, _oi, _rb.to(device), _bb.to(device), _yb.to(device),
                        _recb.to(device), _fkb.to(device))
_L, _w = total_loss(_parts, 0.4)
_L.backward()

assert _o["bag"].shape == (2,), _o["bag"].shape              # clip-level (i1 convention)
assert _o["recipe"].shape == (2, N_RECIPE), _o["recipe"].shape
assert _o["e_logit"].shape == (2 * T_FRAMES, NPATCH), _o["e_logit"].shape
assert _o["g"].shape == (2 * T_FRAMES, D_MODEL), _o["g"].shape
assert np.isfinite(float(_L.detach())), "loss is not finite"
assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
           for p in net.parameters() if p.requires_grad), "no gradients reached any parameter"
print("forward/backward ok | folded batch=%d | loss=%.4f | active: %s"
      % (2 * T_FRAMES, float(_L.detach()),
         {k: round(v, 3) for k, v in _w.items() if v > 0}))
for _p in net.parameters():
    if _p.grad is not None:
        _p.grad = None

# --- evaluation path -------------------------------------------------------------------
val_ds = ClipDataset(CACHE["val"], val_vids, hard=True, cond="clean", two_views=False)
cal_ds = ClipDataset(CACHE["cal"], cal_vids, hard=True, cond="clean", two_views=False)
_smoke = evaluate(ClipDataset(CACHE["val"], {d: v[:2] for d, v in val_vids.items()},
                              hard=True, cond="clean", two_views=False))
assert len(_smoke["y"]) > 0 and all(0.0 <= v <= 1.0 for v in _smoke["p"])
print(f"evaluate() ok | n={len(_smoke['y'])} "
      f"p_range=({min(_smoke['p']):.3f},{max(_smoke['p']):.3f}) | calibration n="
      f"{len(evaluate(cal_ds)['y'])}")

net.train()
print("=" * 74)
print(f"PREFLIGHT PASSED ({time.time() - _pf_start:.0f}s) - safe to start {EPOCHS}-epoch run")
print("=" * 74)
''')

# --------------------------------------------------------------------------- 12
code(r'''
# ============================ TRAINING ============================
train_ds = ClipDataset(CACHE["train"], train_vids, hard=False)
steps_per_epoch = max(1, len(train_ds) // BATCH_VIDEOS)
if DRY_RUN:
    EPOCHS, steps_per_epoch = 2, 4
TOTAL_STEPS = EPOCHS * steps_per_epoch
print(f"epochs={EPOCHS} | steps/epoch={steps_per_epoch} | total optimiser steps={TOTAL_STEPS} "
      f"| images/step={BATCH_VIDEOS * T_FRAMES * 2}")

params = [p for p in net.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR_PEAK, weight_decay=WD)


def lr_at(opt_step, total):
    w = max(1, int(WARMUP_FRAC * total))
    if opt_step < w:
        return LR_PEAK * (opt_step + 1) / w
    t = (opt_step - w) / max(1, total - w)
    return LR_PEAK * (LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * t)))


class EMA:
    # shadows every non-trunk tensor (trainable params + BatchNorm running stats)
    def __init__(self, module, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()
                       if "trunk." not in k}
        self.backup = {}

    def update(self, module):
        with torch.no_grad():
            for k, v in module.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, module):
        self.backup = {k: v.detach().clone() for k, v in module.state_dict().items()
                       if k in self.shadow}
        module.load_state_dict(self.shadow, strict=False)

    def restore(self, module):
        module.load_state_dict(self.backup, strict=False)


ema = EMA(net, EMA_DECAY)
BEST = dict(select=-1.0, epoch=0)


def save_ckpt(path, epoch, extra=None):
    sd = strip_sd({k: v.detach().cpu() for k, v in net.state_dict().items()
                   if "trunk." not in k})
    payload = dict(model_trainable=sd, epoch=epoch, manifest_hash=MANIFEST["hash"],
                   best_select=BEST["select"])
    if extra:
        payload.update(extra)
    torch.save(payload, path)


metrics_rows = []
t_start = time.time()
global_step = 0
stop_reason = "completed"

for epoch in range(1, EPOCHS + 1):
    tr_ds = ClipDataset(CACHE["train"], train_vids, hard=False)
    tr_ds.set_epoch(epoch)
    tr_ld = make_loader(tr_ds, shuffle=True)
    run = collections.defaultdict(list)
    net.train()
    ep_t = time.time()
    it = 0
    tr_pred = collections.defaultdict(list)   # rolling window for the per-epoch train AUC
    for xb, xib, rb, bb, yb, recb, fkb, _ in tr_ld:
        frac = global_step / max(1, TOTAL_STEPS)
        for g in opt.param_groups:
            g["lr"] = lr_at(global_step, TOTAL_STEPS)
        xb = xb.to(device, non_blocking=True); xib = xib.to(device, non_blocking=True)
        rb = rb.to(device, non_blocking=True); bb = bb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True); recb = recb.to(device, non_blocking=True)
        fkb = fkb.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
            o, oi = net(xb, xib)
        parts = compute_losses(o, oi, rb, bb, yb, recb, fkb)
        loss, w = total_loss(parts, frac)
        (loss / ACCUM_STEPS).backward()

        if (it + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(net)
            global_step += 1

        with torch.no_grad():
            run["total"].append(float(loss.detach()))
            for k in ("bag", "region", "boundary", "dice", "mil", "cons", "sc",
                      "align", "unif", "recipe"):
                run[k].append(float(parts[k].detach()))
            # Train AUC over a rolling window of clip-level predictions: BATCH_VIDEOS = 2
            # gives only two labels per step, far too few to score on its own.
            tr_pred["p"].append(torch.sigmoid(o["bag"].float()).reshape(-1).cpu().numpy())
            tr_pred["y"].append(yb.reshape(-1).cpu().numpy())
            _pp = np.concatenate(tr_pred["p"][-64:])
            _yy = np.concatenate(tr_pred["y"][-64:])
            if len(np.unique(_yy)) > 1:
                run["tr_auc"].append(float(roc_auc_score(_yy, _pp)))
        it += 1

        if it % 10 == 0 or it == steps_per_epoch:
            el = time.time() - ep_t
            speed = it / max(el, 1e-6)
            eta_min = (steps_per_epoch - it) / max(speed, 1e-6) / 60
            tr_auc = float(np.mean(run["tr_auc"][-20:])) if run["tr_auc"] else float("nan")
            print(f"  ep{epoch}/{EPOCHS} step{it}/{steps_per_epoch} stage{stage_of(frac)} "
                  f"frac{frac:.2f} lr{lr_at(global_step, TOTAL_STEPS):.2e} "
                  f"loss{np.mean(run['total'][-10:]):.3f} train_auc{tr_auc:.3f} "
                  f"{speed:.2f} it/s eta{eta_min:.1f}m", flush=True)
        if global_step >= TOTAL_STEPS:
            break

    do_full = (epoch % 5 == 0) or (epoch == EPOCHS) or epoch <= 2
    frac_end = min(1.0, global_step / max(1, TOTAL_STEPS))
    val = full_validation(conds=ROBUST_CONDS if do_full else ["clean"])
    summ = val["_summary"]
    row = dict(epoch=epoch, step=global_step, minutes=(time.time() - t_start) / 60.0,
               stage=stage_of(frac_end), select=summ["worst_auc"],
               worst_auc=summ["worst_auc"], mean_auc=summ["mean_auc"],
               clean_auc=summ["clean_auc"], robust_gap=summ["robust_gap"],
               val_threshold=val["clean"]["threshold"],
               loc_pixel_auc=val["clean"].get("loc_pixel_auc", float("nan")),
               loc_iou=val["clean"].get("loc_iou", float("nan")),
               loc_mask_f1=val["clean"].get("loc_mask_f1", float("nan")),
               tr_auc=float(np.mean(run["tr_auc"][-20:])) if run["tr_auc"] else float("nan"))
    for c in ROBUST_CONDS:
        if c in val:
            row[f"{c}_auc"] = val[c]["auc"]
    for k in ("total", "bag", "region", "boundary", "dice", "mil", "cons", "sc",
              "align", "unif", "recipe"):
        row[f"L_{k}"] = float(np.mean(run[k][-20:])) if run[k] else float("nan")
    metrics_rows.append(row)

    improved = summ["worst_auc"] > BEST["select"]
    if improved:
        BEST = dict(select=float(summ["worst_auc"]), epoch=epoch)
        ema.apply_to(net)
        save_ckpt(Path(OUT_DIR, "checkpoint_best.pt"), epoch, extra=dict(val_raw=val["_raw"]))
        ema.restore(net)

    mins = (time.time() - t_start) / 60.0
    print(f"[epoch {epoch}] {mins:.1f}m | worst_auc={summ['worst_auc']:.4f} "
          f"clean={summ['clean_auc']:.4f} mean={summ['mean_auc']:.4f} "
          f"loc_pixel_auc={row['loc_pixel_auc']:.4f} loc_iou={row['loc_iou']:.4f} "
          f"{'*best*' if improved else ''}", flush=True)

    pd.DataFrame(metrics_rows).to_csv(Path(OUT_DIR, "progress.csv"), index=False)

    if mins > TIME_BUDGET_MIN:
        stop_reason = f"time budget {TIME_BUDGET_MIN} min reached at epoch {epoch}"
        print(f"SAFE STOP: {stop_reason}")
        break

TRAIN_DONE = dict(minutes=(time.time() - t_start) / 60.0, steps=global_step, epochs=epoch,
                  best=BEST, stop_reason=stop_reason, epochs_planned=EPOCHS)
print(f"\nTRAINING DONE: {TRAIN_DONE['minutes']:.1f} min | {global_step} optimiser steps | "
      f"best {SELECT_METRIC}={BEST['select']:.4f} at epoch {BEST['epoch']} | {stop_reason}")
Path(OUT_DIR, "metrics.json").write_text(json.dumps(
    dict(train=TRAIN_DONE, epochs=metrics_rows), indent=2, default=str))
''')

# --------------------------------------------------------------------------- 13
code(r'''
# ============================ REPORT, GATES, FIGURES ============================
mdf = pd.DataFrame(metrics_rows)
last = mdf.iloc[-1]
best = mdf.loc[mdf["select"].idxmax()]

G = {}
print("=" * 74); print(f"GATE CHECK — {RUN_NAME} ({ABLATION})"); print("=" * 74)


def gate(name, ok, detail):
    G[name] = (bool(ok), detail)
    return bool(ok)


gate("val_selection_auc", float(best["select"]) > 0.70,
     f"best worst-case val AUROC={float(best['select']):.4f} at epoch {int(best['epoch'])} "
     f"(i1: 0.578; chance 0.500)")
gate("localisation_quality", float(best["loc_pixel_auc"]) > 0.90 and float(best["loc_iou"]) > 0.55,
     f"pixel AUC={float(best['loc_pixel_auc']):.3f} IoU={float(best['loc_iou']):.3f} "
     f"(i1: 0.938 / 0.600 — must not regress)")
gate("threshold_usable", 0.02 < float(best["val_threshold"]) < 0.98,
     f"calibrated val threshold={float(best['val_threshold']):.3f} "
     f"(i1 classified every clean sample as real at 0.5)")
gate("robustness", float(best["robust_gap"]) < 0.15,
     f"clean-minus-worst={float(best['robust_gap']):.4f}")
gate("overfit_gap", float(last["tr_auc"]) < 0.98 or float(best["select"]) > 0.70,
     f"train AUC={float(last['tr_auc']):.3f} vs worst eval={float(best['select']):.3f} "
     f"(i1: 1.000 vs 0.578)")
gate("loss_decreased", float(mdf["L_total"].iloc[0]) > float(mdf["L_total"].iloc[-1]),
     f"total loss {float(mdf['L_total'].iloc[0]):.3f} -> {float(mdf['L_total'].iloc[-1]):.3f}")

for k, (ok, det) in G.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k:<22} {det}")
VERDICT = "GREEN" if all(ok for ok, _ in G.values()) else "YELLOW"
print(f"\nverdict: {VERDICT}")

# ---- efficiency (single device: per-clip latency, not throughput) ----------------------
model = unwrap(net)
model.eval()
_x1 = torch.randn(1, T_FRAMES, 3, IMG, IMG, device=device)
with torch.no_grad():
    for _ in range(2):
        model(_x1, None)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    _t = time.time()
    for _ in range(5):
        model(_x1, None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.time() - _t) / 5
model.train()
EFF = dict(params_total_M=n_total / 1e6, params_trainable_M=n_train / 1e6,
           clip_latency_s=lat, per_frame_ms=1000 * lat / T_FRAMES,
           peak_mem_GiB=(torch.cuda.max_memory_allocated() / 1024 ** 3)
           if device.type == "cuda" else 0.0)
print("efficiency: " + " | ".join(f"{k}={v:.4g}" for k, v in EFF.items()))


def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(Path(OUT_DIR, "figures", f"{stem}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


x = mdf["epoch"].to_numpy()
fig, ax = plt.subplots(figsize=(11, 6))
for c, col in zip(ROBUST_CONDS, ["#1565c0", "#ef6c00", "#2e7d32", "#8e24aa"]):
    if f"{c}_auc" in mdf.columns:
        ax.plot(x, mdf[f"{c}_auc"], marker="o", label=f"AUROC {c}", color=col)
ax.plot(x, mdf["select"], marker="s", lw=2.5, color="#000000", label=f"selection ({SELECT_METRIC})")
ax.set(xlabel="epoch", ylabel="clip AUROC", ylim=(0.4, 1.02),
       title="Validation AUROC (i1 was flat at ~0.50-0.60 here)")
ax.grid(alpha=.25); ax.legend(ncol=2); savefig(fig, "validation_auroc")

fig, ax = plt.subplots(figsize=(11, 6))
for col, lab in [("L_total", "total"), ("L_bag", "bag"), ("L_region", "region"),
                 ("L_boundary", "boundary"), ("L_mil", "MIL"), ("L_cons", "consistency"),
                 ("L_sc", "self-consistency"), ("L_recipe", "recipe"),
                 ("L_align", "align"), ("L_unif", "uniformity")]:
    if col in mdf.columns:
        ax.plot(x, mdf[col], marker="o", lw=1.3, label=lab)
ax.set(xlabel="epoch", ylabel="loss (mean of last 20 steps)", title="Objective components")
ax.set_yscale("symlog"); ax.grid(alpha=.25); ax.legend(ncol=3); savefig(fig, "training_losses")

fig, ax = plt.subplots(figsize=(11, 6))
for col, lab in [("loc_pixel_auc", "pixel AUC"), ("loc_iou", "IoU"), ("loc_mask_f1", "mask F1")]:
    if col in mdf.columns:
        ax.plot(x, mdf[col], marker="o", label=lab)
ax.set(xlabel="epoch", ylabel="score", ylim=(0, 1.02),
       title="Localization quality (val-hard synthesis)")
ax.grid(alpha=.25); ax.legend(); savefig(fig, "localisation")

ck = torch.load(Path(OUT_DIR, "checkpoint_best.pt"), map_location="cpu", weights_only=False)
vy, vp = ck["val_raw"]
if len(set(vy)) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    fpr, tpr, _ = roc_curve(vy, vp)
    axs[0].plot(fpr, tpr, color="#1565c0", lw=2, label=f"AUROC={roc_auc_score(vy, vp):.4f}")
    axs[0].plot([0, 1], [0, 1], "--", color="#888")
    axs[0].set(xlabel="FPR", ylabel="TPR", title="ROC (best epoch, clean)")
    pr, rc, _ = precision_recall_curve(vy, vp)
    axs[1].plot(rc, pr, color="#8e24aa", lw=2, label=f"AP={average_precision_score(vy, vp):.4f}")
    axs[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    cm = confusion_matrix(vy, [int(v >= 0.5) for v in vp], labels=[0, 1])
    im = axs[2].imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        axs[2].text(c, r, int(v), ha="center", va="center")
    axs[2].set(xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="actual",
               title="Confusion @0.5")
    axs[2].set_xticklabels(["real", "fake"]); axs[2].set_yticklabels(["real", "fake"])
    for a in axs[:2]:
        a.grid(alpha=.25); a.legend()
    fig.colorbar(im, ax=axs[2]); savefig(fig, "roc_pr_confusion")

# ---- evidence panels: input / GT region / GT seam / evidence / boundary / low-level -----
ema.apply_to(net); model = unwrap(net); model.eval()
panel_ds = ClipDataset(CACHE["val"], {d: v[:1] for d, v in val_vids.items()},
                       hard=True, cond="clean", two_views=False)
rows = []
with torch.no_grad():
    for idx in range(min(4, len(panel_ds))):
        xx, _, mm, bb, yy, _, _, _ = panel_ds[idx]
        with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
            o = model(xx[None].to(device), None)
        img = np.clip(xx[0].permute(1, 2, 0).numpy() * IMNET_STD + IMNET_MEAN, 0, 1)
        maps = [torch.sigmoid(o[k][0].float()).reshape(GRID, GRID).cpu().numpy()
                for k in ("e_logit", "b_logit", "l_logit")]
        rows.append((img, mm[0].numpy(), bb[0].numpy(), maps,
                     float(torch.sigmoid(o["bag"].float())[0]), int(yy[0])))
fig, axes = plt.subplots(len(rows), 6, figsize=(17, 3.0 * len(rows)))
axes = np.atleast_2d(axes)
for r, (img, gtm, gtb, maps, score, lab) in enumerate(rows):
    axes[r, 0].imshow(img)
    axes[r, 0].set_ylabel(f"{'fake' if lab else 'real'}\np={score:.2f}", fontsize=9)
    axes[r, 1].imshow(gtm, cmap="gray", vmin=0, vmax=1)
    axes[r, 2].imshow(gtb, cmap="gray", vmin=0, vmax=1)
    for c, (mp, nm) in enumerate(zip(maps, ["evidence", "boundary", "low-level"])):
        axes[r, c + 3].imshow(mp, cmap="inferno", vmin=0, vmax=1)
        if r == 0:
            axes[r, c + 3].set_title(nm)
    if r == 0:
        axes[r, 0].set_title("input"); axes[r, 1].set_title("GT region")
        axes[r, 2].set_title("GT seam")
    for a in axes[r]:
        a.set_xticks([]); a.set_yticks([])
savefig(fig, "evidence_panels")
model.train(); ema.restore(net)

REPORT = dict(run=RUN_NAME, ablation=ABLATION, verdict=VERDICT, gates=G, seed=SEED,
              manifest_hash=MANIFEST["hash"], training=TRAIN_DONE,
              steps=int(last["step"]), minutes=float(last["minutes"]),
              select_metric=SELECT_METRIC, best_select=float(best["select"]),
              best_epoch=int(best["epoch"]),
              best_epoch_metrics={k: (v if not isinstance(v, np.generic) else v.item())
                                  for k, v in best.to_dict().items()},
              efficiency=EFF, config=CFG,
              note="i2 EvidenceNet uses real videos + self-blended synthesis only. "
                   "Validation is open-loop; cross-dataset numbers come from the final "
                   "evaluation cell and were never used for selection.")
Path(OUT_DIR, "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print(f"\nreport -> {OUT_DIR}/report.json | figures -> {OUT_DIR}/figures/")
''')

# --------------------------------------------------------------------------- 14
code(r'''
# ============================ FINAL EVALUATION — runs once, after freeze ============================
# Two axes, and both are required:
#   A. WITHIN-DATASET — held-out videos from the SAME pools the model trained on, pushed through
#      the SAME pixel pipeline (training synthesis recipe for the fake side, the training
#      photometric draw for the real side). No domain shift, so this measures whether the task
#      was learnt at all. The pack's own real manipulations of those same held-out videos are
#      reported in the same group.
#   B. CROSS-DATASET — DFDCP and Celeb-DF-v2: other corpora and other generators, raw frames,
#      the published protocol.
# First and only read of the FF++ manipulated sequences and of DFDCP / Celeb-DF-v2.
# Nothing below may be used to change a hyper-parameter or to re-select a checkpoint.
model = unwrap(net)
ck_path = Path(OUT_DIR, "checkpoint_best.pt")
assert ck_path.exists(), "train first — no checkpoint_best.pt"
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
model.load_state_dict(strip_sd(ck["model_trainable"]), strict=False)
model.eval()
print(f"frozen checkpoint: epoch {ck['epoch']} | {SELECT_METRIC}={ck['best_select']:.4f} "
      f"| manifest {ck['manifest_hash']}")
assert ck["manifest_hash"] == MANIFEST["hash"], "checkpoint was trained on a different split"


@torch.no_grad()
def score_videos(records, cond=None, t=T_TEST, in_domain=False, mode="synth", hard=False):
    # video score = mean over chunks of the mean over frames. Cross-dataset scoring leaves the
    # pixels untouched (published protocol); in-dataset scoring runs the training pipeline.
    if in_domain:
        ds = InDatasetTestDataset(records, t, mode=mode, hard=hard)
    else:
        ds = TestVideoDataset(records, t)
    ld = DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS, pin_memory=True)
    out = []
    for xb, _ in ld:
        if cond is not None and cond != "clean":
            arr = (xb[0].permute(0, 2, 3, 1).numpy() * IMNET_STD + IMNET_MEAN)
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            xb = torch.stack([to_tensor(intervene_fixed(a, cond)) for a in arr])[None]
        chunks = []
        for ch in xb.split(T_FRAMES, dim=1):
            with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
                o = model(ch.to(device, non_blocking=True), None)
            chunks.append(float(torch.sigmoid(o["bag"].float()).mean()))
        out.append(float(np.mean(chunks)))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


EVAL, TABLES, IN, CROSS = {}, [], {}, {}


def pair(group, name, r, f, cond=None):
    # r / f are per-video scores; label 0 = real, 1 = fake
    y, p = [0] * len(r) + [1] * len(f), list(r) + list(f)
    m = classification_metrics(y, p, thr=0.5)
    m.update(bootstrap_auc(y, p))
    m.update(n_real=len(r), n_fake=len(f), condition=(cond or "clean"))
    TABLES.append(dict(group=group, benchmark=name, condition=(cond or "clean"), n=len(y),
                       auc=m["auc"], ap=m["ap"], eer=m["eer"], ci_lo=m["lo"], ci_hi=m["hi"]))
    print(f"  {group:<10s} {name:<40s} {str(cond or 'clean'):<9s} AUROC {m['auc']:.4f} "
          f"[{m['lo']:.4f}, {m['hi']:.4f}]  AP {m['ap']:.4f}  EER {m['eer']:.4f}")
    return m


print("=" * 100)
print(f"FINAL EVALUATION — within-dataset and cross-dataset ({T_TEST} frames/video, frozen model)")
print("=" * 100)
print(f"  checkpoint epoch {ck['epoch']} | {SELECT_METRIC}={ck['best_select']:.4f} "
      f"| manifest {MANIFEST['hash']}\n")

# ---- GROUP A: WITHIN-DATASET -------------------------------------------------------------------
ffpp_te = TEST_SETS.get("ffpp_test_real", [])
print("-" * 100)
print("GROUP A — WITHIN-DATASET: held-out videos from the training pools, training pixel pipeline")
print("-" * 100)
if ffpp_te:
    r_in = score_videos(ffpp_te, in_domain=True, mode="real")       # shared by every fake side
    IN["sbi"] = pair("in-dataset", "FF++ real vs SBI (training recipe)", r_in,
                     score_videos(ffpp_te, in_domain=True, mode="synth"))
    IN["sbi_hard"] = pair("in-dataset", "FF++ real vs SBI (hard recipe)", r_in,
                          score_videos(ffpp_te, in_domain=True, mode="synth", hard=True))
    IN["sbi_robustness"] = {}
    for cond in [c for c in ROBUST_CONDS if c != "clean"]:
        IN["sbi_robustness"][cond] = pair(
            "in-dataset", "FF++ real vs SBI",
            score_videos(ffpp_te, cond, in_domain=True, mode="real"),
            score_videos(ffpp_te, cond, in_domain=True, mode="synth"), cond)["auc"]
    # the real-manipulation sub-table uses RAW held-out reals: both sides raw, same corpus and
    # compression, so no jitter-vs-raw shortcut is introduced by the evaluation itself
    r_raw = score_videos(ffpp_te)
    fam = {}
    for key, recs in sorted(TEST_SETS.items()):
        if key.startswith("ffpp_") and key != "ffpp_test_real":
            manip = key[len("ffpp_"):]
            fam[manip] = pair("in-dataset", f"FF++ {manip} (real manipulation)",
                              r_raw, score_videos(recs))
    IN["ffpp_families"] = fam
    IN["ffpp_macro_auc"] = float(np.mean([v["auc"] for v in fam.values()])) if fam else float("nan")
    IN["headline_auc"] = float(IN["sbi"]["auc"])
    if fam:
        print(f"  in-dataset FF++ macro-average over {len(fam)} families: "
              f"AUROC {IN['ffpp_macro_auc']:.4f}")
else:
    IN["headline_auc"] = float("nan")
    print("  held-out FF++ real videos not found — group A skipped")

# ---- GROUP B: CROSS-DATASET --------------------------------------------------------------------
print("\n" + "-" * 100)
print("GROUP B — CROSS-DATASET: other corpora / other generators, raw frames, published protocol")
print("-" * 100)
if TEST_SETS.get("dfdcp_real") and TEST_SETS.get("dfdcp_fake"):
    r_d = score_videos(TEST_SETS["dfdcp_real"])
    f_d = score_videos(TEST_SETS["dfdcp_fake"])
    CROSS["dfdcp"] = pair("cross", "DFDCP / DFDC-preview (unseen corpus)", r_d, f_d)
    print("             published reference: Xception 0.699 | F3-Net 0.735 | Face X-ray 0.809")
    CROSS["dfdcp_robustness"] = {}
    for cond in [c for c in ROBUST_CONDS if c != "clean"]:
        CROSS["dfdcp_robustness"][cond] = pair(
            "cross", "DFDCP / DFDC-preview", score_videos(TEST_SETS["dfdcp_real"], cond),
            score_videos(TEST_SETS["dfdcp_fake"], cond), cond)["auc"]
else:
    print("  DFDCP not present — skipped")

if TEST_SETS.get("cdf_test_real") and TEST_SETS.get("cdf_test_fake"):
    r_c = score_videos(TEST_SETS["cdf_test_real"])
    f_c = score_videos(TEST_SETS["cdf_test_fake"])
    CROSS["cdf"] = pair("cross", "Celeb-DF-v2 (different generator)", r_c, f_c)
    print("             published reference: F3-Net 0.789 | Face X-ray 0.795 | SBI 0.932 | "
          "GenD 0.960")
    CROSS["cdf_robustness"] = {}
    for cond in [c for c in ROBUST_CONDS if c != "clean"]:
        CROSS["cdf_robustness"][cond] = pair(
            "cross", "Celeb-DF-v2", score_videos(TEST_SETS["cdf_test_real"], cond),
            score_videos(TEST_SETS["cdf_test_fake"], cond), cond)["auc"]
else:
    print("  Celeb-DF-v2 not present — skipped")

# ---- summary -----------------------------------------------------------------------------------
cross_aucs = [v["auc"] for v in (CROSS.get("dfdcp"), CROSS.get("cdf")) if v]
cross_worst = float(min(cross_aucs)) if cross_aucs else float("nan")
cross_mean = float(np.mean(cross_aucs)) if cross_aucs else float("nan")
cross_rob = [a for d in ("dfdcp_robustness", "cdf_robustness") for a in CROSS.get(d, {}).values()]
in_rob = list(IN.get("sbi_robustness", {}).values())
worst_rob = float(min(cross_rob)) if cross_rob else float("nan")
in_auc = float(IN.get("headline_auc", float("nan")))
gen_gap = ((in_auc - cross_worst) if (np.isfinite(in_auc) and np.isfinite(cross_worst))
           else float("nan"))
SUMMARY = dict(in_dataset_auc=in_auc,
               in_dataset_ffpp_macro=float(IN.get("ffpp_macro_auc", float("nan"))),
               in_dataset_robustness_worst=float(min(in_rob)) if in_rob else float("nan"),
               cross_dataset_worst=cross_worst, cross_dataset_mean=cross_mean,
               cross_dataset_robustness_worst=worst_rob, in_minus_cross_gap=gen_gap)

tdf = pd.DataFrame(TABLES)
if len(tdf):
    print("\n" + tdf.drop(columns=["group"]).to_string(index=False))
else:
    print("\n  no test sets found")

print("\n" + "=" * 100)
print("SUMMARY — within-dataset vs cross-dataset")
print("=" * 100)
print(f"  within-dataset AUROC (held-out, training recipe)    {SUMMARY['in_dataset_auc']:.4f}")
print(f"  within-dataset AUROC (FF++ real manipulations)      {SUMMARY['in_dataset_ffpp_macro']:.4f}")
print(f"  within-dataset worst robustness                     "
      f"{SUMMARY['in_dataset_robustness_worst']:.4f}")
print(f"  cross-dataset worst  AUROC                          {SUMMARY['cross_dataset_worst']:.4f}")
print(f"  cross-dataset mean   AUROC                          {SUMMARY['cross_dataset_mean']:.4f}")
print(f"  cross-dataset worst robustness                      "
      f"{SUMMARY['cross_dataset_robustness_worst']:.4f}")
print(f"  within-minus-cross gap                              {SUMMARY['in_minus_cross_gap']:.4f}")

# ---- acceptance gates from the design document --------------------------------------------------
ACC = dict(within_dataset_ge_0_90=bool(in_auc >= 0.90),
           cross_dfdpc_ge_0_80=bool(CROSS.get("dfdcp", {}).get("auc", float("nan")) >= 0.80),
           cross_celebdf_ge_0_85=bool(CROSS.get("cdf", {}).get("auc", float("nan")) >= 0.85),
           worst_robustness_ge_0_75=bool(worst_rob >= 0.75),
           generalisation_gap_le_0_30=bool(np.isfinite(gen_gap) and gen_gap <= 0.30))
print("\n" + "-" * 100)
for k, v in ACC.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"  {'PASS' if all(ACC.values()) else 'FAIL'}  OVERALL "
      f"(i1 measured no cross-dataset number at all)")

EVAL = dict(within_dataset=IN, cross_dataset=CROSS, summary=SUMMARY, acceptance=ACC,
            meta=dict(run=RUN_NAME, ablation=ABLATION, checkpoint_epoch=int(ck["epoch"]),
                      select_metric=SELECT_METRIC, select_value=float(ck["best_select"]),
                      manifest_hash=MANIFEST["hash"], t_test=T_TEST, verdict=VERDICT,
                      tests="within-dataset (held-out videos, training pixel pipeline) + "
                            "cross-dataset (DFDCP, Celeb-DF-v2)",
                      trained_on="real videos + self-blended synthesis only; "
                                 "no real manipulated video was ever trained on"))
Path(OUT_DIR, "evaluation", "final_results.json").write_text(
    json.dumps(EVAL, indent=2, default=str))
if len(tdf):
    tdf.to_csv(Path(OUT_DIR, "evaluation", "table1_main.csv"), index=False)
print(f"\n  results -> {OUT_DIR}/evaluation/final_results.json")
''')

# --------------------------------------------------------------------------- 15
code(r'''
# ============================ CALIBRATION + VALIDATED THRESHOLD ============================
# i1's finding #3: at threshold 0.5 it classified every recorded clean example as real.
# i1's next-step #3: validate one global threshold on a separate calibration split.
# The calibration split here is disjoint from train, val and every test set.
cal_raw = evaluate(cal_ds, want_maps=False)
cy = np.array(cal_raw["y"], float)
cp = np.array(cal_raw["p"], float)
cz = np.array(cal_raw["z"], float)

if len(set(cy.tolist())) > 1:
    T, nll = fit_temperature(cz, cy)
    p_cal = 1.0 / (1.0 + np.exp(-cz / T))
    thr_cal = youden_threshold(cy, p_cal)
    before = classification_metrics(cy, cp, thr=0.5)
    after_default = classification_metrics(cy, p_cal, thr=0.5)
    after_cal = classification_metrics(cy, p_cal, thr=thr_cal)
    CALIB = dict(temperature=T, nll=nll, threshold=thr_cal,
                 ece_before=before["ece"], ece_after=after_default["ece"],
                 brier_before=before["brier"], brier_after=after_default["brier"],
                 degenerate_at_default=bool(before["recall"] in (0.0, 1.0)),
                 acc_at_calibrated_threshold=after_cal["acc"],
                 bacc_at_calibrated_threshold=after_cal["bacc"],
                 f1_at_calibrated_threshold=after_cal["f1"])
    print("=" * 74); print("CALIBRATION (held-out split, never used for selection)"); print("=" * 74)
    print(f"  temperature             {T:.4f}")
    print(f"  ECE        {before['ece']:.4f} -> {after_default['ece']:.4f}   (i1 clean ECE 0.1508)")
    print(f"  Brier      {before['brier']:.4f} -> {after_default['brier']:.4f}   (i1 clean Brier 0.2670)")
    print(f"  validated threshold     {thr_cal:.4f}")
    print(f"  at threshold 0.5        acc={after_default['acc']:.3f} "
          f"recall={after_default['recall']:.3f}")
    print(f"  at validated threshold  acc={after_cal['acc']:.3f} bacc={after_cal['bacc']:.3f} "
          f"f1={after_cal['f1']:.3f}")
    print(f"  degenerate at 0.5?      {CALIB['degenerate_at_default']}  (i1: True)")
else:
    CALIB = dict(error="calibration split has only one class — check the split or DRY_RUN")
    print("CALIBRATION skipped:", CALIB["error"])

Path(OUT_DIR, "evaluation", "calibration.json").write_text(json.dumps(CALIB, indent=2, default=str))
Path(OUT_DIR, "evaluation", "calibration_raw.json").write_text(json.dumps(
    dict(y=cal_raw["y"], p=cal_raw["p"], z=cal_raw["z"]), indent=2))
print(f"\n  calibration -> {OUT_DIR}/evaluation/calibration.json")
print("\nNEXT: write record/i2-EvidenceNet-v1-{Pass,Fail}.md from "
      "evaluation/final_results.json + report.json, reporting default and calibrated "
      "thresholds separately, per record/rules.md.")
''')

# --------------------------------------------------------------------------- 16
md(r"""
## Running this notebook

**1. Validate first (a few minutes).** Set `DRY_RUN = True` in the config cell and run all
cells. Everything is exercised — synthesis, the frame cache, the model, the losses,
validation, the final evaluation and calibration — on a handful of videos. It must print
`PREFLIGHT PASSED`.

**2. The first run — short, one arm, full architecture.** Set `DRY_RUN = False`,
`REBUILD_CACHE = True` and leave `ABLATION = "A0"` (every component on; no ablation). The
shipped defaults are deliberately short: `EPOCHS = 8` and `TIME_BUDGET_MIN = 150`, which is
~80 min of training plus ~35 min of evaluation on 2×T4 (within-dataset and cross-dataset,
both required). Eight epochs is 2 480 optimiser steps — 2.7× i1's *entire* 900-step run — and
it reaches every curriculum stage (`locate_end = 0.25`, `aggregate_end = 0.55` of total
steps), so the pipeline is exercised end to end and the direction of the result is readable.
`RUN_NAME = "i2_evidence_v1_short"` keeps this arm's outputs separate from the full run.

**Treat a short run as a pilot, not a result.** The acceptance gates in the final evaluation
cell are written for the full schedule, so the cross-dataset gates are *expected to fail* at
8 epochs — that is the schedule, not a bug. Per `record/rules.md`, a `Pass`/`Fail` record with
real numbers needs the full run; a pilot is reported as a pilot.

**3. The full run, once the short one is clean.** Set `EPOCHS = 30`, `TIME_BUDGET_MIN = 420`
and `RUN_NAME = "i2_evidence_v1"`. At ~310 steps/epoch and ~1.5–2 s/step that is ~9–10
min/epoch, so 30 epochs is ~5 h. `CACHE_DIR` is keyed on the **split**, not on `RUN_NAME`, so
both runs reuse the same frame cache (`REBUILD_CACHE = False`).

**4. Ablations after that**, in order of value: `A1` (convex pooling — the direct test of the
central hypothesis), then `A2` (no boundary loss), then `A3` (no low-level branch). Each arm
writes to its own `OUT_DIR`, so runs cannot overwrite each other. `TIME_BUDGET_MIN` stops any
run cleanly before the session limit and still saves the best checkpoint.

### Reading the result

- `outputs/.../report.json` — gates, verdict, efficiency, config
- `outputs/.../evaluation/final_results.json` — within-dataset (SBI + the pack's own real
  manipulations) and cross-dataset (DFDCP, Celeb-DF-v2) results, each with confidence intervals
- `outputs/.../evaluation/calibration.json` — temperature, ECE/Brier, validated threshold
- `outputs/.../figures/evidence_panels.png` — input / GT region / GT seam / predicted
  evidence / predicted boundary / low-level map.

Check `evidence_panels.png` before trusting any AUROC. If the predicted evidence map is not
visibly localized on the fake row while the real row stays quiet, the run has not reproduced
i1's localization and the number should not be believed.

### Things that are deliberately NOT here

- **No token-level expert routing.** i1's router had entropy `1.606` against `ln 5 =
  1.609` — uniform, i.e. a no-op — and the reported "1 dead expert" was a consequence of
  `topi.clamp(max=3)`, which made the frequency expert undispatachable by construction.
  The hypothesis was tested and falsified, so it is not carried forward.
- **No LoRA.** i1's "LoRA" was an 11 k-parameter rank-8 bottleneck on the frozen trunk's
  output tokens; nothing inside the transformer was adapted. GenD's ablation shows LoRA
  rank-1 reaches 99.99 % train AUROC in two epochs while LayerNorm tuning generalizes best,
  so LN-tuning is exposed as flag `A7` rather than being on by default: it needs gradient
  checkpointing to fit a 16 GB T4 (design document §7.4).
- **No real manipulated training data.** That is the point of the experiment — it is the
  setting in which LAA-Net reaches 95.4 AUROC on Celeb-DF-v2. An i3 arm that trains on four
  FF++ families while holding out FaceShifter and DeepFakeDetection is the natural follow-up
  if this iteration falls short on cross-dataset transfer.
""")


# --------------------------------------------------------------------------- emit
def build():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"i2-cell-{i:02d}", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    return nb


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = root / "notebook" / "i2-EvidenceNet-v1.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1) + "\n")
    nc = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {out} | {len(CELLS)} cells ({nc} code)")
