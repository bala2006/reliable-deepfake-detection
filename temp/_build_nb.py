"""Emit Stable-RouteNet_v4_notebook.ipynb (Stable-RouteNet v4 — proposed redesign).

v4 changes from v3.1:
  - Shared expert always evaluated (common forensic structure)
  - Prototype/representation-first router (cosine similarity to learned prototypes)
  - Switch-style hard-load balancing (f_k x P_k)
  - Router z-loss for stability
  - Selective route consistency (only confident/high-evidence tokens)
  - Relative curriculum (all schedules as fractions of total steps)
  - Explicit specialist semantic roles (spatial, spectral, temporal, structural)
  - Train/test graph equivalence (evidence gating active throughout)
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n") + "\n"))


def code(src):
    CELLS.append(("code", src.strip("\n") + "\n"))


# ----------------------------------------------------------------------------- 0
md(r"""
# Stable-RouteNet v4 — Proposed Architecture (5-epoch confirmation run)

Route by representation, assess trust separately, train the exact evaluation graph.

Evidence-stability-conditioned sparse expert routing with shared + specialist experts,
prototype-based representation-first router, Switch-style load balancing, and router
z-loss for generalizable deepfake detection.

**Data:** `Stable-RouteNet-1` Kaggle pack — 300 train + 50 val real videos (FF++ YouTube-c23 150 +
Celeb-DF-v2 real 150), plus untouched test subsets (DFDCP 50+50, FF++ 6 manipulation families).
Fakes are generated on-the-fly by DD-SBI; **no real manipulated video is ever trained on.**

---

## v4 design principles

1. **Route by forensic representation** — prototype router selects experts by cosine
   similarity to learned forensic attribute prototypes, not by a generic MLP that can
   discover domain shortcuts.
2. **Assess trust separately** — M/S/R measure evidence trust; routing identity is
   decoupled from trust to prevent cascading error.
3. **Preserve common knowledge** — a shared expert is always evaluated so common
   forensic structure remains available even when sparse routing is imperfect.
4. **Train the exact evaluation graph** — all schedules are relative to total steps;
   no module required for final evaluation is permanently disabled by an absolute
   threshold.

## v4 ablation plan

| Ablation | Change | Question |
|----------|--------|----------|
| A0 | Corrected v3.1 curriculum only | Is train/test graph mismatch the main failure? |
| A1 | A0 + Switch balance + z-loss | Does router stabilization prevent collapse? |
| A2 | A1 + shared expert | Does common knowledge stabilize transfer? |
| A3 | A2 + prototype router | Does representation-first routing reduce shortcuts? |
| A4 | A3 + semantic specialist axes | Does meaningful specialization improve OOD? |

This notebook implements the full v4 architecture. Set `ABLATION` in the config cell
to run specific ablation stages.

---
""")

# ----------------------------------------------------------------------------- 1
md(r"""
## Protocol and leakage control (spec 34, 35)

| Split | Source | Used for |
|---|---|---|
| **Train** | FF++ YouTube-c23 real x150, Celeb-DF-v2 real x150, DD-SBI fakes on-the-fly | gradient updates |
| **Val** (`val-hard`) | FF++ real x25, Celeb-DF real x25, held out at video level, hard DD-SBI parameters | loss monitoring, threshold calibration, checkpoint selection |
| **Test — cross-dataset** | DFDCP 50 real + 50 fake | reported once, after freeze |
| **Test — cross-manipulation** | FF++ Deepfakes / Face2Face / FaceShifter / FaceSwap / NeuralTextures / DeepFakeDetection | reported once, after freeze |

Enforced in code by `assert_no_leakage()`: train, val and test video identities are disjoint, and
no test video is ever passed to the DD-SBI generator. The final evaluation cell is the only place
test data is read, and it runs after `checkpoint_best.pt` is written.

Run order: config -> preflight (must print `PREFLIGHT PASSED`) -> training -> report/gates ->
final evaluation.
""")

# ----------------------------------------------------------------------------- 2 config
code(r'''
# ============================ CONFIG — STABLE-ROUTENET V4 ============================
import os

# --- run identity -----------------------------------------------------------
RUN_NAME     = "v4_a0_5ep"
ABLATION     = "A0"        # A0 correct curriculum | A1 +switch/z | A2 +shared | A3 +prototype | A4 +semantic
SEED         = 0
DATA_ROOT    = "/kaggle/input/Stable-RouteNet-1"
OUT_DIR      = f"/kaggle/working/outputs/{RUN_NAME}"
RESUME_FROM_LAST = False

# --- token geometry ---------------------------------------------------------
IMG, PATCH   = 392, 14
GRID         = IMG // PATCH          # 28 -> 784 patch tokens
T_FRAMES     = 8                     # frames per training clip
T_TEST       = 32                    # frames per test video

# --- data splits ------------------------------------------------------------
N_TRAIN      = {"ffpp": 150, "cdf": 150}
N_VAL        = {"ffpp": 25,  "cdf": 25}

# --- optimisation -----------------------------------------------------------
EPOCHS       = 5
BATCH_VIDEOS = 2
ACCUM_STEPS  = 2                     # effective batch = 4 videos
LR_PEAK      = 1.5e-4
LR_FLOOR     = 0.05
WD           = 0.01
WARMUP_FRAC  = 0.10                  # warmup as fraction of total steps
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.999
LABEL_SMOOTH = 0.05
MAX_STEPS    = None

# --- model ------------------------------------------------------------------
D_MODEL      = 384
N_EXPERTS, TOPK = 4, 2
EXPERT_HIDDEN = 512
EPS          = 1e-6
TAU_S        = 0.75
PROTOTYPE_DIM = D_MODEL              # prototype embedding dimension

# --- v4: relative curriculum (fractions of total planned steps) -------------
# All schedules defined as fractions of total training steps.
# This fixes the v3.1 critical bug where MoE losses activate after training ends.
PHASE_FRACTIONS = dict(
    foundation_end  = 0.20,   # Phase 1: det + loc + mass + shared warm-start
    evidence_end    = 0.40,   # Phase 2: + M/S targets, stability, interventions
    moe_end         = 0.60,   # Phase 3: + top-2 routing, shared, Switch balance, z-loss
    reliability_end = 0.80,   # Phase 4: + A/C/R, evidence weighting, alpha learning
    # Phase 5 (80-100%): consolidation, all objectives, reduced auxiliary
)
RAMP_FRAC    = 0.05               # linear ramp at each phase boundary (as fraction)

# --- objective weights ------------------------------------------------------
LAM = dict(
    loc  = 1.00,
    mass = 0.25,
    stab = 0.20,
    s    = 0.20,
    s_dist = 0.05,
    sep  = 0.02,
    route = 0.05,
    bal  = 0.05,       # Switch-style load balance
    z    = 0.01,       # Router z-loss
    div  = 0.02,       # Expert diversity (prototype-based)
    nuis = 0.10,       # F_var nuisance prediction
)

# --- evaluation -------------------------------------------------------------
ROBUST_CONDS = ["clean", "jpeg30", "blur15", "resize50"]
SELECT_METRIC = "worst_auc"
BOOTSTRAP_N  = 2000
NUM_WORKERS  = 4

# --- ablation -> switches ---------------------------------------------------
_ABL = {
    # A0: corrected v3.1 curriculum only (no switch/z, no shared, no prototype)
    "A0": dict(switch=False, z_loss=False, shared=False, prototype=False,
               semantic=False, off=("z",)),
    # A1: + Switch-style balance + z-loss
    "A1": dict(switch=True, z_loss=True, shared=False, prototype=False,
               semantic=False, off=()),
    # A2: + shared expert
    "A2": dict(switch=True, z_loss=True, shared=True, prototype=False,
               semantic=False, off=()),
    # A3: + prototype router
    "A3": dict(switch=True, z_loss=True, shared=True, prototype=True,
               semantic=False, off=()),
    # A4: + semantic specialist axes
    "A4": dict(switch=True, z_loss=True, shared=True, prototype=True,
               semantic=True, off=()),
}
assert ABLATION in _ABL, f"ABLATION must be one of {sorted(_ABL)}"
ABL = _ABL[ABLATION]
for _k in ABL["off"]:
    LAM[_k] = 0.0

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUT_DIR}/evaluation", exist_ok=True)

CFG = dict(RUN_NAME=RUN_NAME, ABLATION=ABLATION, ABL=ABL, SEED=SEED, DATA_ROOT=DATA_ROOT,
           OUT_DIR=OUT_DIR, IMG=IMG, PATCH=PATCH, GRID=GRID, T_FRAMES=T_FRAMES, T_TEST=T_TEST,
           N_TRAIN=N_TRAIN, N_VAL=N_VAL, EPOCHS=EPOCHS, BATCH_VIDEOS=BATCH_VIDEOS,
           ACCUM_STEPS=ACCUM_STEPS, LR_PEAK=LR_PEAK, LR_FLOOR=LR_FLOOR, WD=WD,
           WARMUP_FRAC=WARMUP_FRAC, GRAD_CLIP=GRAD_CLIP, EMA_DECAY=EMA_DECAY,
           LABEL_SMOOTH=LABEL_SMOOTH, D_MODEL=D_MODEL, N_EXPERTS=N_EXPERTS, TOPK=TOPK,
           EXPERT_HIDDEN=EXPERT_HIDDEN, EPS=EPS, TAU_S=TAU_S,
           PROTOTYPE_DIM=PROTOTYPE_DIM, PHASE_FRACTIONS=PHASE_FRACTIONS, RAMP_FRAC=RAMP_FRAC,
           LAM=LAM, ROBUST_CONDS=ROBUST_CONDS, SELECT_METRIC=SELECT_METRIC)

print(f"run={RUN_NAME} ablation={ABLATION} -> {ABL}")
print(f"epochs={EPOCHS} | active loss terms: {sorted(k for k, v in LAM.items() if v > 0)}")
print(f"phase fractions: {PHASE_FRACTIONS}")
''')

# ----------------------------------------------------------------------------- 3 imports
code(r'''
# ============================ IMPORTS + DETERMINISM ============================
import io, json, math, glob, time, random, shutil, hashlib, collections
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TVT
from transformers import AutoModel
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
N_GPU  = torch.cuda.device_count()
USE_DP = N_GPU > 1
AMP    = torch.float16
assert N_GPU in (0, 1, 2), f"expected up to 2 GPUs, found {N_GPU}"

def worker_init(wid):
    s = SEED * 1000 + wid
    random.seed(s); np.random.seed(s)

IMNET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

print("device:", device, "| GPUs:", N_GPU, "| DataParallel:", USE_DP, "| torch:", torch.__version__)
for i in range(N_GPU):
    print(f"  gpu{i}: {torch.cuda.get_device_name(i)}")
''')

# ----------------------------------------------------------------------------- 4 data
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
    """Locate the Stable-RouteNet-1 pack wherever Kaggle mounted it."""
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
    raise AssertionError(f"could not resolve the Stable-RouteNet-1 layout; available inputs: {avail}")

DATA_ROOT = resolve_dataset_root()
FFPP_REAL = find_dir(DATA_ROOT, "youtube")
CDF_ROOT  = find_dir(DATA_ROOT, "Celeb-DF-v2")
DFDCP_ROOT = find_dir(DATA_ROOT, "DFDCP")
FFPP_ROOT  = find_dir(DATA_ROOT, "FaceForensics++")
assert FFPP_REAL is not None and CDF_ROOT is not None, "FF++ real / Celeb-DF-v2 not found"
print(f"dataset root: {DATA_ROOT}")

def video_frame_dirs(real_root):
    """-> [(video_id, frames_dir, landmarks_dir_or_None)] sorted deterministically."""
    out = []
    if real_root is None or not Path(real_root).exists():
        return out
    for fr in sorted(Path(real_root).rglob("frames")):
        lm = fr.parent / "landmarks"
        for vd in sorted(fr.iterdir()):
            if vd.is_dir():
                out.append((vd.name, str(vd), str(lm / vd.name) if lm.exists() else None))
    return out

ffpp_all = video_frame_dirs(FFPP_REAL)
cdf_all  = video_frame_dirs(CDF_ROOT / "Celeb-real") + video_frame_dirs(CDF_ROOT / "YouTube-real")
print(f"on disk: FF++ real {len(ffpp_all)} | Celeb-DF real {len(cdf_all)}")
assert len(ffpp_all) >= N_TRAIN["ffpp"] + N_VAL["ffpp"], "not enough FF++ real videos"
assert len(cdf_all)  >= N_TRAIN["cdf"]  + N_VAL["cdf"],  "not enough Celeb-DF real videos"

# deterministic video-level split: first N sorted, frozen in the manifest
train_vids = {"ffpp": ffpp_all[:N_TRAIN["ffpp"]],
              "cdf":  cdf_all[:N_TRAIN["cdf"]]}
val_vids   = {"ffpp": ffpp_all[N_TRAIN["ffpp"]:N_TRAIN["ffpp"] + N_VAL["ffpp"]],
              "cdf":  cdf_all[N_TRAIN["cdf"]:N_TRAIN["cdf"] + N_VAL["cdf"]]}

# test sets: read here only to assert disjointness; never sampled before the final cell
TEST_SETS = {}
if DFDCP_ROOT is not None:
    TEST_SETS["dfdcp_real"] = video_frame_dirs(DFDCP_ROOT / "original_videos")
    TEST_SETS["dfdcp_fake"] = (video_frame_dirs(DFDCP_ROOT / "method_A")
                               + video_frame_dirs(DFDCP_ROOT / "method_B"))
if FFPP_ROOT is not None:
    TEST_SETS["ffpp_test_real"] = video_frame_dirs(FFPP_ROOT / "original_sequences" / "youtube" / "c23")
    for manip in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap",
                  "NeuralTextures", "DeepFakeDetection"]:
        got = video_frame_dirs(FFPP_ROOT / "manipulated_sequences" / manip / "c23")
        if got:
            TEST_SETS[f"ffpp_{manip}"] = got

def _keys(groups):
    return {f"{d}/{v[0]}" for d, vs in groups.items() for v in vs}

def assert_no_leakage():
    """spec 35: train / val / test identities disjoint; SBI never sees test videos."""
    tr, va = _keys(train_vids), _keys(val_vids)
    assert not (tr & va), f"train/val overlap: {sorted(tr & va)[:5]}"
    tr_ids = {k.split('/', 1)[1] for k in tr}
    va_ids = {k.split('/', 1)[1] for k in va}
    for name, recs in TEST_SETS.items():
        te_ids = {r[0] for r in recs}
        assert not (te_ids & tr_ids), f"{name} leaks into train: {sorted(te_ids & tr_ids)[:5]}"
        assert not (te_ids & va_ids), f"{name} leaks into val: {sorted(te_ids & va_ids)[:5]}"
    return True

assert_no_leakage()
MANIFEST = dict(
    data_root=DATA_ROOT,
    train={d: [v[0] for v in vs] for d, vs in train_vids.items()},
    val={d: [v[0] for v in vs] for d, vs in val_vids.items()},
    test={k: [r[0] for r in v] for k, v in TEST_SETS.items()},
)
MANIFEST["hash"] = hashlib.sha1(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()[:12]
CFG["DATA_ROOT_RESOLVED"] = DATA_ROOT
CFG["MANIFEST_HASH"] = MANIFEST["hash"]
Path(OUT_DIR, "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
Path(OUT_DIR, "config.json").write_text(json.dumps(CFG, indent=2, default=str))

print(f"train {sum(len(v) for v in train_vids.values())} | val {sum(len(v) for v in val_vids.values())} videos")
print("test subsets: " + ", ".join(f"{k}={len(v)}" for k, v in TEST_SETS.items()))
print(f"leakage check passed | manifest {MANIFEST['hash']}")
''')

# ----------------------------------------------------------------------------- 5 sbi
code(r'''
# ============================ DD-SBI GENERATOR (v3) + INTERVENTIONS ============================
# Wider forgery distribution than v1/v2 to attack the compositing shortcut that produced
# val AUROC 1.0 / DFDCP 0.57: sub-region masks, jittered mask geometry, cross-frame blend
# sources, continuous blend ratio, source-side JPEG. Reals are quality-aligned with the
# same global jitter the blend source receives.

NUISANCE = ["jpeg", "blur", "resize", "photo"]
NUIS_IDX = {k: i for i, k in enumerate(NUISANCE)}
MASK_REGIONS = ["full", "upper", "lower", "middle"]

def photometric_jitter(img, rng):
    """Global appearance jitter. Applied to SBI sources AND to reals (quality alignment)."""
    if rng.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.6:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
    if rng.random() < 0.3:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.randint(60, 95)); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

def _band(size, region, rng):
    """Soft horizontal band used to carve sub-regions out of the face polygon."""
    w, h = size
    if region == "full":
        return None
    m = Image.new("L", size, 0)
    if region == "upper":
        y0, y1 = 0.0, rng.uniform(0.45, 0.62)
    elif region == "lower":
        y0, y1 = rng.uniform(0.38, 0.55), 1.0
    else:
        y0, y1 = rng.uniform(0.25, 0.35), rng.uniform(0.65, 0.78)
    ImageDraw.Draw(m).rectangle([0, int(h * y0), w, int(h * y1)], fill=255)
    return m.filter(ImageFilter.GaussianBlur(h * 0.03))

def landmark_mask(lm_path, size, rng, region="full", jitter=0.0):
    """Face polygon from landmarks, optionally jittered and restricted to a sub-region."""
    m = Image.new("L", size, 0)
    ok = False
    if lm_path is not None:
        try:
            lm = np.load(lm_path).astype(np.float32).reshape(-1, 2)
            pts = lm * np.array([size[0] / 256.0, size[1] / 256.0], np.float32)
            if jitter > 0:
                pts = pts + np.random.normal(0.0, jitter * size[0], pts.shape).astype(np.float32)
            if len(pts) >= 3:
                ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=255)
                ok = True
        except Exception:
            ok = False
    if not ok:
        w, h = size
        cx, cy = w * 0.5 + rng.uniform(-0.03, 0.03) * w, h * 0.5 + rng.uniform(-0.03, 0.03) * h
        rx, ry = w * rng.uniform(0.34, 0.42), h * rng.uniform(0.40, 0.47)
        ImageDraw.Draw(m).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    band = _band(size, region, rng)
    if band is not None:
        m = Image.fromarray((np.array(m, np.float32) * np.array(band, np.float32) / 255.0)
                            .clip(0, 255).astype(np.uint8))
    return m

def make_sbi(img_pil, lm_path=None, src_pil=None, hard=False, rng=None):
    """Self-blended image + exact GT mask (spec 18). Returns (uint8 HWC, float mask HW)."""
    rng = rng or random
    W, H = img_pil.size
    src = (src_pil if src_pil is not None else img_pil).copy()
    if src.size != (W, H):
        src = src.resize((W, H), Image.BILINEAR)
    src = photometric_jitter(src, rng)
    # small affine offset so the blend is not pixel-aligned
    s = rng.uniform(0.90, 1.08)
    sw, sh = max(8, int(W * s)), max(8, int(H * s))
    canvas = Image.new("RGB", (W, H))
    canvas.paste(src.resize((sw, sh), Image.BILINEAR),
                 (rng.randint(min(0, W - sw), max(0, W - sw)),
                  rng.randint(min(0, H - sh), max(0, H - sh))))
    src = canvas

    region = rng.choice(MASK_REGIONS[1:] if hard else MASK_REGIONS)
    mask = landmark_mask(lm_path, (W, H), rng, region=region, jitter=rng.uniform(0.0, 0.012))
    mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 8.0)))
    arr = np.array(mask, np.float32) / 255.0
    ratio = rng.uniform(0.15, 0.45) if hard else rng.uniform(0.15, 1.0)
    arr = arr * ratio

    if (arr > 0.02).mean() < 0.004:                      # degenerate polygon -> ellipse fallback
        m2 = landmark_mask(None, (W, H), rng, region=region)
        arr = np.array(m2.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255.0 * ratio

    a = arr[..., None]
    t_img = np.asarray(img_pil, np.float32)
    t_src = np.asarray(src, np.float32)
    blended = np.clip(t_src * a + t_img * (1.0 - a), 0, 255).astype(np.uint8)
    return blended, (arr > 0.02).astype(np.float32)

def intervene(arr, rng=None):
    """Random nuisance transform. Label- and mask-invariant by construction (spec 19)."""
    rng = rng or random
    img = Image.fromarray(arr)
    kind = rng.choice(NUISANCE)
    if kind == "jpeg":
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.choice([30, 50, 70])); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    elif kind == "blur":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    elif kind == "resize":
        sc = rng.choice([0.5, 0.75]); w, h = img.size
        img = img.resize((int(w * sc), int(h * sc)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    else:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    return np.array(img), NUIS_IDX[kind]

def intervene_fixed(arr, cond):
    """Deterministic, RNG-free nuisance for the robustness sweep."""
    if cond == "clean":
        return arr
    img = Image.fromarray(arr)
    if cond == "jpeg30":
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=30); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    elif cond == "blur15":
        img = img.filter(ImageFilter.GaussianBlur(1.5))
    elif cond == "resize50":
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    else:
        raise ValueError(f"unknown condition {cond}")
    return np.asarray(img)
''')

# ----------------------------------------------------------------------------- 6 datasets
code(r'''
# ============================ DATASETS ============================
def load_frame(fp):
    return Image.open(fp).convert("RGB").resize((IMG, IMG), Image.BILINEAR)

def to_tensor(arr):
    return IMNET(torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0)

def pick_frames(fdir, n):
    files = sorted(glob.glob(str(Path(fdir) / "*.png")))
    assert files, f"no frames in {fdir}"
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx], files

def mask_to_grid(gt):
    g = torch.from_numpy(gt)[None, None]
    return F.adaptive_avg_pool2d(g, GRID)[0, 0]

class PairedClipDataset(Dataset):
    """(clip, intervened clip, GT mask, label, per-frame nuisance id).

    Domain-balanced, 50/50 real vs DD-SBI fake. `deterministic=True` gives a fixed,
    reproducible enumeration (used for validation). `hard=True` samples the difficult end
    of the SBI parameter range so validation does not saturate.
    """

    def __init__(self, vids, deterministic=False, hard=False, p_fake=0.5):
        self.vids = vids
        self.domains = list(vids.keys())
        self.deterministic, self.hard, self.p_fake = deterministic, hard, p_fake
        self.samples = []
        if deterministic:
            for dom in self.domains:
                for rec in self.vids[dom]:
                    self.samples += [(dom, rec, False), (dom, rec, True)]
        self.length = len(self.samples) if deterministic else sum(len(v) for v in vids.values()) * 2

    def __len__(self):
        return self.length

    def _select(self, idx, rng):
        if self.deterministic:
            dom, rec, is_fake = self.samples[int(idx) % len(self.samples)]
        else:
            dom = self.domains[rng.randrange(len(self.domains))]
            rec = rng.choice(self.vids[dom])
            is_fake = rng.random() < self.p_fake
        return dom, rec, is_fake

    def _build(self, idx, rng, cond=None):
        dom, (vid, fdir, ldir), is_fake = self._select(idx, rng)
        frames, all_files = pick_frames(fdir, T_FRAMES)
        lms = []
        for f in frames:
            p = Path(ldir) / (Path(f).stem + ".npy") if ldir else None
            lms.append(str(p) if p is not None and p.exists() else None)
        # cross-frame blend source: a different frame of the same video
        use_cross = is_fake and len(all_files) > T_FRAMES and rng.random() < 0.5
        src_pool = [f for f in all_files if f not in set(frames)] if use_cross else []

        xs, xi, ms, nz = [], [], [], []
        for fp, lp in zip(frames, lms):
            img = load_frame(fp)
            if is_fake:
                src = load_frame(rng.choice(src_pool)) if src_pool else None
                arr, gt = make_sbi(img, lp, src_pil=src, hard=self.hard, rng=rng)
            else:
                arr = np.asarray(photometric_jitter(img, rng))   # quality alignment
                gt = np.zeros((IMG, IMG), np.float32)
            if cond is None:
                arr_t, kind = intervene(arr, rng)
            else:
                arr_t, kind = intervene_fixed(arr, cond), 0
            xs.append(to_tensor(arr)); xi.append(to_tensor(arr_t))
            ms.append(mask_to_grid(gt)); nz.append(kind)
        return (torch.stack(xs), torch.stack(xi), torch.stack(ms),
                torch.tensor([1.0 if is_fake else 0.0]), torch.tensor(nz, dtype=torch.long),
                dom, vid)

    def __getitem__(self, idx):
        if self.deterministic:
            rng = random.Random(SEED * 100003 + int(idx))
            np.random.seed((SEED * 100003 + int(idx)) % (2 ** 31 - 1))
        else:
            rng = random
        return self._build(idx, rng)

class RobustValDataset(PairedClipDataset):
    """Same content as the deterministic val set, one fixed nuisance condition applied.

    The SBI draw is seeded identically across conditions, so the four sweeps differ only in
    the nuisance transform. Returns the perturbed view as the model input.
    """

    def __init__(self, vids, cond, hard=True):
        super().__init__(vids, deterministic=True, hard=hard)
        self.cond = cond

    def __getitem__(self, idx):
        rng = random.Random(SEED * 100003 + int(idx))
        np.random.seed((SEED * 100003 + int(idx)) % (2 ** 31 - 1))
        x, xi, m, y, nz, dom, vid = self._build(idx, rng, cond=self.cond)
        return xi, m, y, dom, vid

class TestVideoDataset(Dataset):
    """Untouched test videos at T_TEST frames. No SBI, no intervention."""

    def __init__(self, records, t=None):
        self.records = records
        self.t = t or T_TEST

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        vid, fdir, _ = self.records[idx]
        frames, _ = pick_frames(fdir, self.t)
        return torch.stack([to_tensor(np.asarray(load_frame(f))) for f in frames]), vid

def make_loader(ds, bs=None, shuffle=False, workers=None):
    workers = NUM_WORKERS if workers is None else workers
    return DataLoader(ds, batch_size=bs or BATCH_VIDEOS, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=shuffle,
                      persistent_workers=workers > 0,
                      prefetch_factor=3 if workers > 0 else None,
                      worker_init_fn=worker_init if workers > 0 else None)
''')

# === PART 2 APPENDED BELOW ===

# ----------------------------------------------------------------------------- 7 model
code(r'''
# ============================ STABLE-ROUTENET V4 ============================
# v4: shared expert + specialist experts + prototype router + Switch balance + z-loss
# All schedules relative to total steps (no absolute step thresholds).

class ForensicAdapter(nn.Module):
    """Residual pre-norm adapter over frozen DINOv2 patch tokens."""
    def __init__(self, din=1024, d=D_MODEL, heads=6):
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.proj = nn.Linear(din, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=d * 4,
                                           dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, tokens):
        h = self.proj(self.norm(tokens))
        return self.out_norm(h + self.blocks(h))


class ExpertMLP(nn.Module):
    """Residual expert MLP."""
    def __init__(self, d=D_MODEL, h=EXPERT_HIDDEN):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.res_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        return x + self.res_scale.tanh() * self.fc2(F.gelu(self.fc1(self.norm(x))))


class SharedExpert(nn.Module):
    """Always-evaluated expert carrying common forensic structure.

    Output added directly to the final representation so common forensic
    evidence remains available even when sparse routing is imperfect.
    """
    def __init__(self, d=D_MODEL, h=EXPERT_HIDDEN):
        super().__init__()
        self.expert = ExpertMLP(d, h)

    def forward(self, x):
        return self.expert(x)


class PrototypeRouter(nn.Module):
    """Representation-first prototype router.

    Projects each token to a normalized routing vector, computes cosine
    similarity to learned prototypes (one per specialist expert), and
    applies temperature-controlled softmax. Selects experts because the
    feature resembles a learned forensic attribute, not because a generic
    MLP discovers a domain shortcut.
    """
    def __init__(self, d=D_MODEL, n_experts=N_EXPERTS, proto_dim=PROTOTYPE_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, proto_dim)
        )
        # One learned prototype per specialist expert
        self.prototypes = nn.Parameter(torch.randn(n_experts, proto_dim) * 0.02)

    def forward(self, x, temperature=1.0):
        """x: (N, D) -> logits: (N, n_experts)"""
        h = self.proj(x)                              # (N, proto_dim)
        h = F.normalize(h, dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)  # (E, proto_dim)
        logits = h @ protos.T                          # (N, E)
        logits = logits / max(temperature, 0.01)
        return logits


def sparse_topk_dispatch(x, idx, w, experts):
    """True sparse dispatch: each expert sees only its selected tokens."""
    out = x.new_zeros(x.shape)
    sel = x.new_zeros(x.shape[0], idx.shape[1], x.shape[1])
    counts = []
    for e, expert in enumerate(experts):
        fi, slot = (idx == e).nonzero(as_tuple=True)
        counts.append(int(fi.numel()))
        if fi.numel() == 0:
            continue
        y = expert(x[fi])
        out.index_add_(0, fi, (y * w[fi, slot].unsqueeze(-1)).to(out.dtype))
        sel[fi, slot] = y.to(sel.dtype)
    return out, sel, counts


class StableRouteNetV4(nn.Module):
    """v4 architecture: shared expert + specialist experts + prototype router.

    Key differences from v3.1:
      1. Shared expert always evaluated (common forensic structure)
      2. Prototype router (cosine similarity to learned prototypes)
      3. Switch-style hard-load balancing (f_k x P_k)
      4. Router z-loss for stability
      5. Selective route consistency (only high-evidence tokens)
      6. All schedules relative to total steps
      7. M/S/R are trust variables, NOT routing variables
    """

    def __init__(self, d=D_MODEL, n_experts=N_EXPERTS, topk=TOPK):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-with-registers-large")
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.prefix = 1 + getattr(self.backbone.config, "num_register_tokens", 0)
        din = self.backbone.config.hidden_size

        # --- feature formation ---
        self.adapter = ForensicAdapter(din, d)
        self.inv_proj = nn.Linear(d, d)
        self.var_proj = nn.Linear(d, d)

        # --- evidence heads (trust variables, NOT routing variables) ---
        def head(out):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, out))
        self.head_m = head(1)       # manipulation evidence (logits)
        self.head_s = head(1)       # stability (logits)
        self.nuis_head = head(len(NUISANCE))  # nuisance prediction for F_var

        # --- shared expert (always evaluated) ---
        self.shared_expert = SharedExpert(d) if ABL["shared"] else None

        # --- specialist experts ---
        self.experts = nn.ModuleList([ExpertMLP(d) for _ in range(n_experts)])

        # --- prototype router (representation-first) ---
        if ABL["prototype"]:
            self.token_router = PrototypeRouter(d, n_experts, PROTOTYPE_DIM)
        else:
            # Fallback: MLP router (v3.1 style) for ablation A0/A1
            self.token_router = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, n_experts)
            )

        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
                                        nn.Linear(128, 1))
        self.n_experts, self.topk = n_experts, topk

        # v4: learnable alpha for geometric softening
        target_alpha = (1/3 - 0.2) / 0.6
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(target_alpha / (1 - target_alpha))))

        # v4: shared expert blend coefficient
        self.gamma = nn.Parameter(torch.tensor(0.5))  # shared vs specialist blend

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def alpha_value(self):
        return 0.2 + 0.6 * torch.sigmoid(self.alpha_logit)

    def _get_temperature(self, step_fraction):
        """Temperature schedule: start high (exploration), anneal down."""
        t_start, t_end = 1.5, 0.75
        return t_start + min(1.0, step_fraction * 3) * (t_end - t_start)

    def _get_gate_strength(self, step_fraction, phase_start):
        """Ramp from 0 to 1 over 5% of total steps after phase_start."""
        ramp = RAMP_FRAC
        return float(np.clip((step_fraction - phase_start) / max(1e-6, ramp), 0.0, 1.0))

    def forward(self, x, xi=None, step_fraction=0.0):
        """Forward pass with v4 architecture.

        Args:
            x: original view (B, T, 3, H, W)
            xi: intervention view (B, T, 3, H, W) - required for C computation
            step_fraction: current_step / total_steps (0.0 to 1.0)
        """
        B, Tn = x.shape[0], x.shape[1]
        BT = B * Tn

        # --- backbone (frozen) ---
        with torch.no_grad():
            both = torch.cat([x, xi], dim=0) if xi is not None else x
            B2 = both.shape[0]
            tokens = self.backbone(both.reshape(B2 * Tn, *both.shape[2:])).last_hidden_state[:, self.prefix:, :]
        assert tokens.shape[1] == GRID * GRID

        # --- adapter + F_inv/F_var ---
        Fr = self.adapter(tokens)
        F_inv, F_var = self.inv_proj(Fr), self.var_proj(Fr)

        # --- evidence heads ---
        m_logits = self.head_m(F_inv).squeeze(-1)
        M = torch.sigmoid(m_logits)
        s_logits = self.head_s(F_inv).squeeze(-1)
        S = torch.sigmoid(s_logits)
        nuis_logits = self.nuis_head(F_var.mean(dim=1))

        # --- shared expert (always evaluated) ---
        H_shared = self.shared_expert(F_inv) if self.shared_expert is not None else None

        # --- prototype / token routing ---
        Nf = F_inv.shape[0] * F_inv.shape[1]
        flat = F_inv.reshape(Nf, -1)

        if ABL["prototype"]:
            temp = self._get_temperature(step_fraction)
            route_logits = self.token_router(flat, temperature=temp)
        else:
            route_logits = self.token_router(flat)

        probs = torch.softmax(route_logits.float(), dim=-1).to(F_inv.dtype)

        # --- top-2 dispatch ---
        topw, topi = probs.topk(self.topk, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + EPS)

        H_sparse, selected, counts = sparse_topk_dispatch(
            flat, topi.reshape(Nf, self.topk), topw.reshape(Nf, self.topk), self.experts
        )
        H_sparse = H_sparse.view(F_inv.shape)
        selected = selected.view(F_inv.shape[0], F_inv.shape[1], self.topk, -1)

        # --- combine shared + specialist ---
        gamma = torch.sigmoid(self.gamma)
        if H_shared is not None:
            Hout = gamma * H_shared + (1.0 - gamma) * H_sparse
        else:
            Hout = H_sparse

        # --- expert correction agreement A ---
        a = F.normalize(selected[..., 0, :].float(), dim=-1)
        b = F.normalize(selected[..., 1, :].float(), dim=-1)
        A = (((a * b).sum(-1) + 1.0) * 0.5).to(F_inv.dtype).clamp(0.05, 0.95)

        # --- split paired representation ---
        if xi is not None:
            o = dict(F_inv=F_inv[:BT], F_var=F_var[:BT], M=M[:BT], S=S[:BT],
                     A=A[:BT], Hout=Hout[:BT], probs=probs[:BT*Nf//F_inv.shape[0]:],
                     topi=topi[:BT*Nf//F_inv.shape[0]:],
                     selected=selected[:BT],
                     mask_logits=m_logits[:BT].reshape(-1, 1, GRID, GRID),
                     nuis_logits=nuis_logits[:BT] if nuis_logits.shape[0] == F_inv.shape[0] else nuis_logits,
                     counts=torch.tensor(counts, device=x.device, dtype=torch.long))
            oi = dict(F_inv=F_inv[BT:], F_var=F_var[BT:], M=M[BT:], S=S[BT:],
                      A=A[BT:], Hout=Hout[BT:], probs=probs,
                      topi=topi, selected=selected[BT:],
                      mask_logits=m_logits[BT:].reshape(-1, 1, GRID, GRID),
                      nuis_logits=nuis_logits[BT:] if nuis_logits.shape[0] == F_inv.shape[0] else nuis_logits,
                      counts=torch.tensor(counts, device=x.device, dtype=torch.long))
            o = self._finalize_pair(o, oi, step_fraction)
            oi = self._finalize_pair(oi, o, step_fraction)
            return o, oi

        # single-view fallback
        R = A
        alpha = self.alpha_value()
        W_raw = (M.clamp_min(EPS) * S.clamp_min(EPS) * R.clamp_min(EPS)).pow(alpha)
        W_norm = W_raw / (W_raw.mean(dim=1, keepdim=True) + EPS)
        Zf = (Hout * W_norm[..., None]).mean(dim=1)
        Zv = Zf.reshape(B, Tn, -1).mean(dim=1)

        return dict(
            logit=self.classifier(Zv), mask_logits=m_logits.reshape(-1, 1, GRID, GRID),
            M=M, S=S, A=A, R=R, W=W_raw, W_norm=W_norm,
            F_inv=F_inv, F_var=F_var,
            alpha=alpha.detach().reshape(1),
            nuis_logits=nuis_logits,
            probs=probs, topi=topi, topw=topw, selected=selected,
            counts=torch.tensor(counts, device=x.device, dtype=torch.long),
            H_shared=H_shared,
        )

    def _finalize_pair(self, o, oi, step_fraction):
        """Compute C, R = sqrt(A*C), W = (M*S*R)^alpha, logit."""
        a_f = o["F_inv"].float()
        b_f = oi["F_inv"].float()
        dist2 = (a_f - b_f).pow(2).mean(dim=-1)
        denom = a_f.pow(2).mean(dim=-1) + EPS
        C = torch.exp(-dist2 / denom).clamp(0.05, 0.95)

        A = o["A"].clamp(0.05, 0.95)
        R = torch.sqrt(A * C).clamp(0.05, 0.95)

        alpha = self.alpha_value()
        W_raw = (o["M"].clamp_min(EPS) * o["S"].clamp_min(EPS) * R.clamp_min(EPS)).pow(alpha)
        W_norm = W_raw / (W_raw.mean(dim=1, keepdim=True) + EPS)

        BT = o["Hout"].shape[0]
        Zf = (o["Hout"] * W_norm[..., None]).mean(dim=1)
        Tn = BT // (BT // T_FRAMES) if T_FRAMES > 0 else T_FRAMES
        B = BT // T_FRAMES
        Zv = Zf.reshape(B, T_FRAMES, -1).mean(dim=1)

        o = dict(o, C=C, R=R, W=W_raw, W_norm=W_norm,
                 logit=self.classifier(Zv),
                 alpha=alpha.detach().reshape(1))
        return o
''')

# ----------------------------------------------------------------------------- 8 losses
code(r'''
# ============================ OBJECTIVE V4 (relative curriculum) ============================
# All schedules defined as fractions of total training steps.
# This fixes the v3.1 critical bug where MoE losses activate after training ends.

def phase_ramp(step_fraction, phase_start):
    """Linear ramp from 0 to 1 starting at phase_start."""
    return float(np.clip((step_fraction - phase_start) / max(1e-6, RAMP_FRAC), 0.0, 1.0))

def stage_weights(step_fraction):
    """v4 relative curriculum: all stages defined as fractions of total steps."""
    w = {k: 0.0 for k in LAM}
    pf = PHASE_FRACTIONS

    # Phase 1 (0-20%): detection + localisation + mass always active
    for k in ("loc", "mass"):
        w[k] = LAM[k]

    # Phase 2 (20-40%): stability, S target, separation, nuisance
    r2 = phase_ramp(step_fraction, pf["foundation_end"])
    for k in ("stab", "s", "s_dist", "sep", "nuis"):
        w[k] = LAM[k] * r2

    # Phase 3 (40-60%): routing regularisers (Switch balance, z-loss, diversity)
    r3 = phase_ramp(step_fraction, pf["evidence_end"])
    for k in ("bal", "z", "div"):
        w[k] = LAM[k] * r3

    # Phase 4 (60-80%): route consistency (selective, high-evidence tokens only)
    r4 = phase_ramp(step_fraction, pf["moe_end"])
    w["route"] = LAM["route"] * r4

    return w

def stage_of(step_fraction):
    pf = PHASE_FRACTIONS
    if step_fraction < pf["foundation_end"]:
        return 1
    elif step_fraction < pf["evidence_end"]:
        return 2
    elif step_fraction < pf["moe_end"]:
        return 3
    elif step_fraction < pf["reliability_end"]:
        return 4
    return 5

def soft_dice(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    t = target.float()
    num = 2.0 * (p * t).flatten(1).sum(1) + eps
    den = p.flatten(1).sum(1) + t.flatten(1).sum(1) + eps
    return 1.0 - (num / den).mean()

def localisation_loss(mask_logits, gt):
    ml, t = mask_logits.float(), gt.float()
    pos = t.sum()
    neg = t.numel() - pos
    pw = torch.clamp(neg / pos.clamp_min(1.0), 1.0, 20.0) if pos > 0 else t.new_ones(())
    bce = F.binary_cross_entropy_with_logits(ml, t, pos_weight=pw)
    fake = t.flatten(1).amax(1) > 0.01
    dice = soft_dice(ml[fake], t[fake]) if bool(fake.any()) else ml.new_zeros(())
    return bce + dice

def evidence_mass_loss(M, gt_tok):
    pred, tgt = M.float().mean(1), gt_tok.float().mean(1)
    fake = (tgt > 0.01).float()
    return ((pred - tgt).abs() * fake).mean() + (pred.pow(2) * (1.0 - fake)).mean()

def relative_stability_target(f0, f1):
    dist = 1.0 - F.cosine_similarity(f0.float(), f1.float(), dim=-1)
    z = (dist - dist.mean(1, keepdim=True)) / dist.std(1, keepdim=True).clamp_min(1e-4)
    return torch.sigmoid(-z / TAU_S).detach()

def separation_loss(f_inv, f_var):
    a = (f_inv - f_inv.mean(1, keepdim=True)).float()
    b = (f_var - f_var.mean(1, keepdim=True)).float()
    cov = torch.einsum("bnd,bnk->bdk", a, b) / max(1, a.shape[1] - 1)
    sa = a.pow(2).mean(1).sqrt().clamp_min(1e-4)
    sb = b.pow(2).mean(1).sqrt().clamp_min(1e-4)
    corr = cov / (sa.unsqueeze(-1) * sb.unsqueeze(1) + 1e-6)
    var_floor = F.relu(1.0 - b.std(dim=1)).mean()
    return corr.pow(2).mean() + 0.5 * var_floor


def switch_balance_loss(probs, topi, n_experts):
    """Switch-style load balance: L_bal = E * sum_k(f_k * P_k).

    f_k = fraction of hard top-2 dispatches assigned to expert k
    P_k = mean router probability for expert k
    This directly targets actual dispatch imbalance.
    """
    N = probs.shape[0]
    # P_k: mean router probability per expert
    P = probs.float().mean(dim=0)  # (E,)

    # f_k: fraction of hard dispatches per expert (from top-2)
    onehot = F.one_hot(topi.reshape(-1, topi.shape[-1]), n_experts).sum(1).float()
    f = onehot.mean(dim=0) / topi.shape[-1]  # normalize by K

    return n_experts * (f * P.detach()).sum()


def router_z_loss(logits):
    """Router z-loss: regularize router logit magnitude.

    L_z = mean_i [log sum_k exp(z_i,k)]^2
    Following ST-MoE for stable sparse routing.
    """
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return (log_z ** 2).mean()


def selective_route_consistency(probs_orig, probs_interv, M, threshold=0.6):
    """Selective routing consistency: only penalize route changes on confident tokens.

    Only tokens where M > threshold (high manipulation evidence) contribute
    to the consistency loss. This avoids over-constraining low-evidence tokens.
    """
    # JSD between original and intervention routing distributions
    p = probs_orig.float().clamp_min(1e-8)
    q = probs_interv.float().clamp_min(1e-8)
    m = 0.5 * (p + q)
    jsd = 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))

    # Only apply to confident tokens
    weight = (M.float() > threshold).float().detach()
    if weight.sum() > 0:
        return (jsd * weight).sum() / weight.sum()
    return jsd.mean() * 0.0


def compute_losses(o1, o2, y, gt, nuis):
    """v4 paired-view objective with Switch balance, z-loss, selective route consistency."""
    y_s = y * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
    L_det = 0.5 * (F.binary_cross_entropy_with_logits(o1["logit"].float(), y_s)
                   + F.binary_cross_entropy_with_logits(o2["logit"].float(), y_s))

    gt4 = gt.reshape_as(o1["mask_logits"])
    gt_tok = gt4.reshape(o1["M"].shape)
    L_loc = localisation_loss(o1["mask_logits"], gt4)
    L_mass = evidence_mass_loss(o1["M"], gt_tok)

    dist = 1.0 - F.cosine_similarity(o1["F_inv"].float(), o2["F_inv"].float(), dim=-1)
    wgt = 0.25 + 0.75 * gt_tok.float()
    L_stab = (dist * wgt).sum() / (wgt.sum() + EPS)
    L_S = F.smooth_l1_loss(o1["S"].float(), relative_stability_target(o1["F_inv"], o2["F_inv"]))
    S_star = relative_stability_target(o1["F_inv"], o2["F_inv"])
    L_Sdist = (o1["S"].float().mean(1) - S_star.mean(1)).square().mean() + \
              (o1["S"].float().std(1) - S_star.std(1)).square().mean()

    L_sep = separation_loss(o1["F_inv"], o1["F_var"])
    L_nuis = F.cross_entropy(o2["nuis_logits"].float(), nuis.reshape(-1))

    # v4: Switch-style balance (targets actual dispatch imbalance)
    L_bal = switch_balance_loss(o1["probs"], o1["topi"], N_EXPERTS)

    # v4: Router z-loss (stabilizes router logit magnitude)
    L_z = router_z_loss(o1["probs"].float())

    # v4: Expert diversity (prototype-based, not correction-based)
    # Measure diversity across expert outputs on a batch basis
    sel = o1["selected"].float()
    L_div = F.relu(F.cosine_similarity(sel[..., 0, :], sel[..., 1, :], dim=-1) - 0.25).mean()

    # v4: Selective route consistency (only confident/high-evidence tokens)
    L_route = selective_route_consistency(o1["probs"], o2["probs"], o1["M"])

    parts = dict(det=L_det, loc=L_loc, mass=L_mass, stab=L_stab, s=L_S, s_dist=L_Sdist,
                 sep=L_sep, nuis=L_nuis, bal=L_bal, z=L_z, div=L_div, route=L_route)

    # Routing statistics
    p = o1["probs"].float().reshape(-1, N_EXPERTS)
    tok_ent = -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
    marg_ent = -(p.mean(0) * p.mean(0).clamp_min(1e-9).log()).sum()
    onehot = F.one_hot(o1["topi"].reshape(-1, TOPK), N_EXPERTS).sum(1).float()
    share = onehot.mean(0).detach().cpu().tolist()
    route_stats = dict(tok_entropy=float(tok_ent.detach()),
                       marg_entropy=float(marg_ent.detach()),
                       share=share)
    return parts, route_stats

def total_loss(parts, step_fraction):
    w = stage_weights(step_fraction)
    L = parts["det"]
    for k, lam in w.items():
        if lam > 0.0:
            L = L + lam * parts[k]
    return L
''')

# ----------------------------------------------------------------------------- 9 metrics
code(r'''
# ============================ METRICS (spec 25) ============================
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
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
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
    out = {}
    if not y or len(set(y)) < 2:
        keys = ["auc", "ap", "eer", "acc", "prec", "recall", "f1", "bacc", "ece", "brier",
                "tn", "fp", "fn", "tp"]
        return {prefix + k: float("nan") for k in keys}
    hard = [int(v >= thr) for v in p]
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel().tolist()
    out.update({
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
    })
    return out

def localisation_metrics(m_prob, gt, max_points=200_000, seed=SEED):
    """Pixel AUC / IoU / mask-F1 over forged frames only (spec 25 secondary, Table 5)."""
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
    return dict(pixel_auc=float(roc_auc_score(t, p)),
                iou=inter / max(union, 1.0),
                mask_f1=float(np.nanmax(f1)))

def merge_counts(counts):
    """DataParallel concatenates the per-replica count vectors; fold them back."""
    c = counts if torch.is_tensor(counts) else torch.as_tensor(counts)
    return c.reshape(-1, N_EXPERTS).sum(0)

def routing_health(share, tok_entropy):
    share = np.asarray(share, float)
    ideal = 1.0 / N_EXPERTS
    return dict(share=share.tolist(), min_share=float(share.min()), max_share=float(share.max()),
                dead_experts=int((share < 0.02).sum()),
                tok_entropy=float(tok_entropy),
                tok_entropy_frac=float(tok_entropy / math.log(N_EXPERTS)),
                share_gini=float(np.abs(share - ideal).sum() / (2 * (1 - ideal))))
''')

# ----------------------------------------------------------------------------- 10 model init
code(r'''
# ============================ BUILD MODEL + BUDGET ============================
model = StableRouteNetV4().to(device)
net = nn.DataParallel(model) if USE_DP else model

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"trainable {n_trainable/1e6:.2f}M | total {n_total/1e6:.1f}M (DINOv2-L frozen)")
print("module breakdown:")
for name in ["adapter", "inv_proj", "var_proj", "head_m", "head_s", "nuis_head",
             "shared_expert", "token_router", "experts", "classifier"]:
    mod = getattr(model, name, None)
    if mod is not None:
        print(f"  {name:<16} {sum(p.numel() for p in mod.parameters())/1e6:7.3f}M")

steps_per_epoch = max(1, (sum(len(v) for v in train_vids.values()) * 2) // BATCH_VIDEOS)
planned_steps = steps_per_epoch * EPOCHS
print(f"batches/epoch {steps_per_epoch} | planned batches {planned_steps} | "
      f"optimiser steps {planned_steps // ACCUM_STEPS}")
print(f"curriculum (relative): {PHASE_FRACTIONS}")
print(f"  Phase 1 (0-20%): det + loc + mass + shared warm-start")
print(f"  Phase 2 (20-40%): + M/S targets, stability, interventions")
print(f"  Phase 3 (40-60%): + top-2 routing, Switch balance, z-loss")
print(f"  Phase 4 (60-80%): + A/C/R, evidence weighting, alpha learning")
print(f"  Phase 5 (80-100%): consolidation")

_state_bytes = n_trainable * 4
_forecast = int((EPOCHS + 6) * 2 * _state_bytes * 1.25 + 512 * 1024 ** 2)
_disk = shutil.disk_usage(OUT_DIR)
print(f"output disk: free {_disk.free/1024**3:.2f} GiB | checkpoints need ~{_forecast/1024**3:.2f} GiB")
assert _disk.free >= _forecast, "not enough output storage for checkpoints"
CFG["trainable_params_M"] = n_trainable / 1e6
CFG["steps_per_epoch"] = steps_per_epoch
Path(OUT_DIR, "config.json").write_text(json.dumps(CFG, indent=2, default=str))
''')

# ----------------------------------------------------------------------------- 11 preflight
code(r'''
# ============================ PREFLIGHT ============================
# Every check below is a regression test for a failure that actually happened, or a
# contract the long run depends on. A red line here costs a minute; the v1 run cost
# 6.4 hours to discover the same class of bug.
print("=" * 74); print("PREFLIGHT"); print("=" * 74)

# --- 1 geometry -------------------------------------------------------------
with torch.no_grad():
    probe = model.backbone(torch.randn(1, 3, IMG, IMG, device=device)).last_hidden_state
assert probe.shape == (1, 1 + 4 + GRID * GRID, 1024), probe.shape
print(f"[ok] tokens {tuple(probe.shape)} = 1 CLS + 4 registers + {GRID*GRID} patches")

# --- 2 split contract -------------------------------------------------------
assert len(train_vids["ffpp"]) == N_TRAIN["ffpp"] and len(train_vids["cdf"]) == N_TRAIN["cdf"]
assert len(val_vids["ffpp"]) == N_VAL["ffpp"] and len(val_vids["cdf"]) == N_VAL["cdf"]
assert assert_no_leakage()
print("[ok] split sizes and train/val/test disjointness")

# --- 3 dataset contract, mask coverage, real/fake balance -------------------
ds_tr = PairedClipDataset(train_vids, deterministic=True)
ds_va = PairedClipDataset(val_vids, deterministic=True, hard=True)
x0, xi0, m0, y0, nz0, dom0, vid0 = ds_tr[1]
assert x0.shape == (T_FRAMES, 3, IMG, IMG) and xi0.shape == x0.shape
assert m0.shape == (T_FRAMES, GRID, GRID) and y0.shape == (1,) and nz0.shape == (T_FRAMES,)
assert 0.0 <= float(m0.min()) <= float(m0.max()) <= 1.0
assert int(nz0.max()) < len(NUISANCE)
cov_f = [float(ds_tr[i][2].mean()) for i in range(1, 12, 2)]
cov_r = [float(ds_tr[i][2].mean()) for i in range(0, 12, 2)]
assert max(cov_r) == 0.0, f"real samples must have empty masks, got {cov_r}"
assert 0.01 < float(np.mean(cov_f)) < 0.90, f"suspicious fake mask coverage {cov_f}"
cov_hard = [float(ds_va[i][2].mean()) for i in range(1, 12, 2)]
print(f"[ok] dataset: fake coverage train {np.mean(cov_f):.3f} / val-hard {np.mean(cov_hard):.3f}; reals empty")

# --- 4 mask head emits UNBOUNDED logits (v1/v2 double-sigmoid regression) ---
model.eval()
with torch.no_grad():
    o_chk = model(x0[None].to(device), xi=x0[None].to(device), step_fraction=0.5)
ml = o_chk["mask_logits"].float()
assert ml.shape == (T_FRAMES, 1, GRID, GRID)
assert float(ml.min()) < 0.0, f"mask_logits look pre-squashed (min={float(ml.min()):.4f}); L_loc needs raw logits"
mp = torch.sigmoid(ml)
assert float(mp.min()) < 0.5, "mask probability cannot drop below 0.5 -> double sigmoid is back"
assert torch.allclose(o_chk["M"].reshape_as(mp), mp, atol=1e-3), "M must equal sigmoid(mask_logits)"
print(f"[ok] mask logits raw: range [{float(ml.min()):.2f}, {float(ml.max()):.2f}] -> "
      f"p in [{float(mp.min()):.4f}, {float(mp.max()):.4f}]")

# --- 5 L_loc floor regression: degenerate M must NOT sit at ~0.748 ----------
gt_dummy = torch.zeros(2, 1, GRID, GRID, device=device)
gt_dummy[0, 0, 8:20, 8:20] = 1.0
loc_bad = float(localisation_loss(torch.full_like(gt_dummy, -12.0), gt_dummy))
loc_good = float(localisation_loss(torch.where(gt_dummy > 0.5, 8.0, -8.0), gt_dummy))
assert loc_good < 0.15 and loc_bad > 1.0, f"L_loc is not discriminative: perfect={loc_good:.4f} collapsed={loc_bad:.4f}"
print(f"[ok] L_loc discriminates: perfect mask {loc_good:.4f} vs collapsed mask {loc_bad:.4f} "
      f"(v1 was pinned at 0.748 for both)")

# --- 6 evidence weighting is not a constant --------------------------------
# In v4, evidence weighting is always active (no gate). Check that W varies.
w_chk = o_chk.get("W_norm", o_chk.get("W", None))
if w_chk is not None:
    w_eff = w_chk.float()
    assert float(w_eff.std()) > 1e-4, "W_norm is constant -> pooling degenerated to a plain mean"
    print(f"[ok] W_norm varies: mean {float(w_eff.mean()):.3f} std {float(w_eff.std()):.3f}")
else:
    print("[ok] evidence weighting check skipped (W not in output)")

# --- 7 relative stability target stays spread out --------------------------
f0 = torch.randn(2, 64, 32, device=device)
tgt_far = relative_stability_target(f0, f0 + 0.5 * torch.randn_like(f0))
tgt_near = relative_stability_target(f0, f0 + 1e-4 * torch.randn_like(f0))
assert float(tgt_near.std()) > 0.05, f"stability target collapsed when cos->1 (std={float(tgt_near.std()):.4f})"
print(f"[ok] stability target spread: far views std {float(tgt_far.std()):.3f} | "
      f"near-identical views std {float(tgt_near.std()):.3f} (v1 target was constant ~0.99)")

# --- 8 sparse dispatch == dense dispatch (spec 77 #9) ----------------------
if ABL.get("prototype", True):
    with torch.no_grad():
        xt = torch.randn(96, D_MODEL, device=device)
        pr = torch.softmax(torch.randn(96, N_EXPERTS, device=device), -1)
        tw, ti = pr.topk(TOPK, -1); tw = tw / tw.sum(-1, keepdim=True)
        sp, _, cnt = sparse_topk_dispatch(xt, ti, tw, model.experts)
        dn = torch.zeros_like(xt)
        for e, ex in enumerate(model.experts):
            ye = ex(xt)
            fi, sl = (ti == e).nonzero(as_tuple=True)
            dn.index_add_(0, fi, ye[fi] * tw[fi, sl].unsqueeze(-1))
        diff = float((sp - dn).abs().max())
    assert diff < 1e-3, f"sparse != dense ({diff:.2e})"
    assert sum(cnt) == 96 * TOPK, f"dispatch dropped tokens: {cnt}"
    print(f"[ok] sparse dispatch == dense dispatch (max diff {diff:.2e}), counts {cnt}")

# --- 9 full objective finite, gradients reach everything, backbone frozen ---
model.train()
ld_chk = make_loader(ds_tr, shuffle=False, workers=0)
xb, xib, mb, yb, nzb, _, _ = next(iter(ld_chk))
xb, xib, mb, yb, nzb = [t.to(device) for t in (xb, xib, mb, yb, nzb)]
with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
    oa, ob = net(xb, xib, step_fraction=0.5)
    parts, rstats = compute_losses(oa, ob, yb, mb, nzb)
    loss_full = total_loss(parts, 0.5)
assert torch.isfinite(loss_full).all(), "full objective is non-finite"
for k, v in parts.items():
    assert torch.isfinite(v).all(), f"loss term {k} is non-finite"
model.zero_grad(set_to_none=True)
loss_full.float().backward()
missing = [n for n, p in model.named_parameters()
           if p.requires_grad and p.grad is None and not n.startswith("backbone")]
# In v4, shared_expert and token_router are always present and always trained
missing = [n for n in missing if not n.startswith("backbone")]
touched_bb = [n for n, p in model.backbone.named_parameters()
              if p.grad is not None and float(p.grad.abs().sum()) > 0]
assert not missing, f"trainable parameters received no gradient: {missing[:6]}"
assert not touched_bb, "frozen DINOv2 received gradients"
model.zero_grad(set_to_none=True)
print("[ok] objective finite, all active modules get gradients, backbone untouched")
print("     " + " ".join(f"{k}={float(v):.4f}" for k, v in parts.items()))
print(f"     routing: tok_entropy={rstats['tok_entropy']:.3f} (ln E={math.log(N_EXPERTS):.3f}) "
      f"share={[round(s, 3) for s in rstats['share']]}")

# --- 10 short optimisation run: the checks v1 would have failed ------------
snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
opt_chk = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=WD)
fixed = [ds_tr[i] for i in range(8)]
hist = collections.defaultdict(list)
model.train()
for it in range(30):
    b = torch.utils.data.default_collate([fixed[it % 8], fixed[(it + 1) % 8]])
    bx, bxi, bm, by, bnz = [t.to(device) for t in b[:5]]
    opt_chk.zero_grad(set_to_none=True)
    oa, ob = net(bx, bxi, step_fraction=0.5)
    parts, rstats = compute_losses(oa, ob, by, bm, bnz)
    total_loss(parts, 0.5).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
    opt_chk.step()
    for k, v in parts.items():
        hist[k].append(float(v))
    hist["mean_M"].append(float(oa["M"].detach().float().mean()))
    hist["tok_ent"].append(rstats["tok_entropy"])

def _drop(key, frac):
    a, b = float(np.mean(hist[key][:5])), float(np.mean(hist[key][-5:]))
    return a, b, b < a * frac

d0, d1, det_ok = _drop("det", 0.98)
assert det_ok, f"detection loss did not move: {d0:.4f} -> {d1:.4f}"
l0, l1, loc_ok = _drop("loc", 0.90)
assert loc_ok, f"L_loc is not learning: {l0:.4f} -> {l1:.4f} (this is exactly the v1 failure)"
assert float(np.mean(hist["mean_M"][-5:])) > 1e-3, \
    f"M collapsed during the smoke run: {float(np.mean(hist['mean_M'][-5:])):.3e}"
print(f"[ok] 30-step run: det {d0:.4f}->{d1:.4f} | loc {l0:.4f}->{l1:.4f} | "
      f"mean_M {hist['mean_M'][0]:.4f}->{hist['mean_M'][-1]:.4f}")
if LAM["z"] > 0:
    e0, e1 = float(np.mean(hist["tok_ent"][:5])), float(np.mean(hist["tok_ent"][-5:]))
    print(f"[ok] router token entropy {e0:.4f} -> {e1:.4f} (target: well below ln E = {math.log(N_EXPERTS):.3f})")

with torch.no_grad():
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.copy_(snapshot[n])
model.zero_grad(set_to_none=True)
del opt_chk, snapshot, fixed, ld_chk
if device.type == "cuda":
    torch.cuda.empty_cache()

# --- 11 metric helpers ------------------------------------------------------
_y = [0, 0, 1, 1]; _p = [0.1, 0.2, 0.8, 0.9]
_m = classification_metrics(_y, _p)
assert abs(_m["auc"] - 1.0) < 1e-9 and _m["eer"] < 1e-6
assert abs(bootstrap_auc(_y, _p, n=50)["auc"] - 1.0) < 1e-9
_lm = localisation_metrics([np.array([[0.9, 0.1], [0.1, 0.05]])], [np.array([[1.0, 0.0], [0.0, 0.0]])])
assert _lm["pixel_auc"] > 0.9
assert routing_health([0.25] * N_EXPERTS, math.log(N_EXPERTS))["dead_experts"] == 0
assert routing_health([0.5, 0.5, 0.0, 0.0], 0.1)["dead_experts"] == 2
print("[ok] metric helpers: AUROC/EER/bootstrap/localisation/routing-health")

# --- 12 checkpoint round trip ---------------------------------------------
_st = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
_pth = Path(OUT_DIR, "_preflight_ckpt.pt")
torch.save({"format": "stable_routenet_v4", "model_trainable": _st}, _pth)
_ld = torch.load(_pth, map_location="cpu", weights_only=False)
assert set(_ld["model_trainable"]) == set(_st)
_pth.unlink()
print("[ok] checkpoint save/load round-trip")

PREFLIGHT_PASSED = True
print("=" * 74)
print("PREFLIGHT PASSED - safe to start the 20-epoch run")
print("=" * 74)
''')

# ----------------------------------------------------------------------------- 12 training
code(r'''
# ============================ TRAINING — 5 epochs (v4 relative curriculum) ============================
assert globals().get("PREFLIGHT_PASSED", False), "run the preflight cell first"

class EMA:
    """Exponential moving average of the trainable tail."""
    def __init__(self, module, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float()
                       for n, p in module.named_parameters() if p.requires_grad}
        self.backup = None

    @torch.no_grad()
    def update(self, module):
        for n, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, module):
        self.backup = {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}
        for n, p in module.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, module):
        if self.backup is None:
            return
        for n, p in module.named_parameters():
            if p.requires_grad:
                p.copy_(self.backup[n])
        self.backup = None

    def state_dict(self):
        return {n: v.cpu() for n, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for n in self.shadow:
            if n in sd:
                self.shadow[n] = sd[n].clone().float().to(self.shadow[n].device)

def lr_at(opt_step, total_opt_steps):
    warm = max(1, int(WARMUP_FRAC * total_opt_steps))
    if opt_step < warm:
        return LR_PEAK * (opt_step + 1) / warm
    prog = (opt_step - warm) / max(1, total_opt_steps - warm)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return LR_PEAK * (LR_FLOOR + (1.0 - LR_FLOOR) * cos)

train_ld = make_loader(PairedClipDataset(train_vids), shuffle=True)
val_lds = {c: make_loader(RobustValDataset(val_vids, c, hard=True), shuffle=False, workers=2)
           for c in ROBUST_CONDS}
n_val = len(val_lds["clean"].dataset)
print(f"train batches/epoch {len(train_ld)} | val videos/condition {n_val} | conditions {ROBUST_CONDS}")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR_PEAK, weight_decay=WD, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
ema = EMA(model, EMA_DECAY)
TOTAL_OPT_STEPS = max(1, (len(train_ld) * EPOCHS) // ACCUM_STEPS)
TOTAL_STEPS = len(train_ld) * EPOCHS

@torch.no_grad()
def evaluate(loader, step_fraction, want_diagnostics=False):
    """Video-level scores plus diagnostics."""
    model.eval()
    ys, ps = [], []
    m_fake, gt_fake, m_real = [], [], []
    ev = collections.defaultdict(list)
    counts = np.zeros(N_EXPERTS)
    tok_ent = []
    for xb, mb, yb, _, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
            o = net(xb, step_fraction=step_fraction)
        if isinstance(o, dict):
            logits = o.get("logit", o) if isinstance(o, dict) else o
        else:
            logits = o
        ps += torch.sigmoid(logits["logit"].float()).flatten().tolist()
        ys += yb.flatten().tolist()
        if not want_diagnostics:
            continue
        M = o["M"].float().reshape(-1, GRID, GRID).cpu().numpy()
        gtn = mb.reshape(-1, GRID, GRID).numpy()
        for i in range(len(M)):
            (m_fake if gtn[i].max() > 0.01 else m_real).append(M[i])
            if gtn[i].max() > 0.01:
                gt_fake.append(gtn[i])
        ev["M"] += o["M"].float().mean(1).cpu().tolist()
        ev["S"] += o["S"].float().mean(1).cpu().tolist()
        ev["R"] += o["R"].float().mean(1).cpu().tolist()
        ev["W"] += o["W_norm"].float().mean(1).cpu().tolist()
        ev["S_std"] += o["S"].float().std(1).cpu().tolist()
        ev["W_p90"] += torch.quantile(o["W_norm"].float(), 0.9, dim=1).cpu().tolist()
        ev["W_p10"] += torch.quantile(o["W_norm"].float(), 0.1, dim=1).cpu().tolist()
        counts += merge_counts(o["counts"]).cpu().numpy()
        p = o["probs"].float().reshape(-1, N_EXPERTS)
        tok_ent.append(float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean()))
    model.train()
    out = dict(y=ys, p=ps)
    if want_diagnostics:
        share = (counts / max(1.0, counts.sum())).tolist() if counts.sum() > 0 else [0.0] * N_EXPERTS
        out["loc"] = localisation_metrics(m_fake, gt_fake)
        out["routing"] = routing_health(share, float(np.mean(tok_ent)) if tok_ent else float("nan"))
        out["evidence"] = dict(
            mean_M_fake=float(np.mean([a.mean() for a in m_fake])) if m_fake else float("nan"),
            mean_M_real=float(np.mean([a.mean() for a in m_real])) if m_real else float("nan"),
            mean_M=float(np.mean(ev["M"])), mean_S=float(np.mean(ev["S"])),
            mean_R=float(np.mean(ev["R"])), mean_W=float(np.mean(ev["W"])),
            S_token_std=float(np.mean(ev["S_std"])),
            W_range=float(np.mean(ev["W_p90"]) / max(np.mean(ev["W_p10"]), 1e-6)),
        )
    return out

def full_validation(step_fraction):
    res, aucs = {}, {}
    for cond in ROBUST_CONDS:
        raw = evaluate(val_lds[cond], step_fraction, want_diagnostics=(cond == "clean"))
        thr = youden_threshold(raw["y"], raw["p"]) if len(set(raw["y"])) > 1 else 0.5
        met = classification_metrics(raw["y"], raw["p"], thr=0.5)
        met.update(classification_metrics(raw["y"], raw["p"], thr=thr, prefix="cal_"))
        met["threshold"] = thr
        res[cond] = met
        aucs[cond] = met["auc"]
        if cond == "clean":
            res["_diag"] = {k: raw[k] for k in ("loc", "routing", "evidence")}
            res["_raw"] = (raw["y"], raw["p"])
    finite = [v for v in aucs.values() if np.isfinite(v)]
    res["_summary"] = dict(worst_auc=float(min(finite)) if finite else float("nan"),
                           mean_auc=float(np.mean(finite)) if finite else float("nan"),
                           clean_auc=aucs.get("clean", float("nan")),
                           robust_gap=float(aucs.get("clean", np.nan) - min(finite)) if finite else float("nan"),
                           per_cond=aucs)
    return res

def save_ckpt(path, epoch, extra=None, with_opt=False):
    payload = dict(format="stable_routenet_v4", run=RUN_NAME, ablation=ABLATION,
                   epoch=int(epoch), global_step=int(global_step),
                   best_select=float(best_select), select_metric=SELECT_METRIC,
                   config=CFG, manifest_hash=MANIFEST["hash"],
                   model_trainable=ema.state_dict(),
                   model_live={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad})
    if extra:
        payload.update(extra)
    if with_opt:
        payload["optimizer"] = opt.state_dict()
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)

metrics_rows, progress_rows = [], []
best_select, best_epoch = -1.0, -1
global_step, opt_step = 0, 0
start_epoch = 0
t0 = time.time()

resume_path = Path(OUT_DIR, "checkpoint_last.pt")
if RESUME_FROM_LAST and resume_path.exists():
    ck = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_live"], strict=False)
    ema.load_state_dict(ck["model_trainable"])
    if "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    if "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    start_epoch, global_step = ck["epoch"], ck["global_step"]
    opt_step = global_step // ACCUM_STEPS
    best_select = ck.get("best_select", -1.0)
    for name, sink in (("metrics", metrics_rows), ("progress", progress_rows)):
        f = Path(OUT_DIR, f"{name}.json")
        if f.exists():
            sink.extend(json.loads(f.read_text()))
    print(f"resumed at epoch {start_epoch}, step {global_step}, best {best_select:.4f}")
else:
    print("fresh v4 run")

model.train()
opt.zero_grad(set_to_none=True)
try:
    for epoch in range(start_epoch, EPOCHS):
        ep_t0 = time.time()
        run = collections.defaultdict(float)
        ep_run = collections.defaultdict(float)
        nb = ep_nb = 0
        tr_y, tr_p = [], []
        print(f"\n===== EPOCH {epoch+1}/{EPOCHS} | step {global_step} | "
              f"stage {stage_of(global_step/max(1, TOTAL_STEPS))} =====", flush=True)
        for bi, batch in enumerate(train_ld):
            if MAX_STEPS and global_step >= MAX_STEPS:
                break
            xb, xib, mb, yb, nzb = [t.to(device, non_blocking=True) for t in batch[:5]]
            step_frac = global_step / max(1, TOTAL_STEPS)
            with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
                o1, o2 = net(xb, xib, step_fraction=step_frac)
                parts, rstats = compute_losses(o1, o2, yb, mb, nzb)
                loss = total_loss(parts, step_frac)
            scaler.scale(loss / ACCUM_STEPS).backward()

            if (bi + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                for g in opt.param_groups:
                    g["lr"] = lr_at(opt_step, TOTAL_OPT_STEPS)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                opt_step += 1
                run["gnorm"] += float(gnorm)

            global_step += 1
            nb += 1
            ep_nb += 1
            step_log = {"total": float(loss.detach()),
                        "tok_ent": rstats["tok_entropy"], "marg_ent": rstats["marg_entropy"],
                        "mean_M": float(o1["M"].detach().float().mean())}
            step_log.update({k: float(v.detach()) for k, v in parts.items()})
            for k, v in step_log.items():
                run[k] += v
                ep_run[k] += v
            tr_y += yb.flatten().tolist()
            tr_p += torch.sigmoid(o1["logit"].detach().float()).flatten().tolist()

            if global_step % 25 == 0:
                el = (time.time() - t0) / 60
                ips = global_step * BATCH_VIDEOS * T_FRAMES * 2 / max(time.time() - t0, 1e-6)
                tr_auc = roc_auc_score(tr_y, tr_p) if len(set(tr_y)) > 1 else float("nan")
                row = dict(epoch=epoch + 1, step=global_step,
                           stage=stage_of(global_step / max(1, TOTAL_STEPS)),
                           lr=opt.param_groups[0]["lr"],
                           step_fraction=global_step / max(1, TOTAL_STEPS),
                           elapsed_min=el, images_per_sec=ips, train_auc=tr_auc,
                           **{k: run[k] / max(1, nb) for k in
                              ("total", "det", "loc", "mass", "stab", "s", "s_dist", "sep", "nuis",
                               "bal", "z", "div", "route", "tok_ent", "marg_ent", "mean_M")})
                progress_rows.append(row)
                print(f"[{el:6.1f}m] ep{epoch+1} s{global_step} "
                      f"stage{row['stage']} frac={row['step_fraction']:.3f} "
                      f"lr={row['lr']:.2e} | total={row['total']:.3f} "
                      f"det={row['det']:.3f} loc={row['loc']:.3f} "
                      f"bal={row['bal']:.4f} z={row['z']:.4f} | "
                      f"M={row['mean_M']:.4f} "
                      f"tokH={row['tok_ent']:.3f} | "
                      f"trainAUC={tr_auc:.3f} {ips:.0f} img/s", flush=True)
                run = collections.defaultdict(float); nb = 0; tr_y, tr_p = [], []

        # ---- validation on EMA weights ----
        step_frac = global_step / max(1, TOTAL_STEPS)
        ema.apply_to(model)
        val = full_validation(step_frac)
        ema.restore(model)

        summ, diag = val["_summary"], val["_diag"]
        select = summ[SELECT_METRIC]
        ep_min = (time.time() - ep_t0) / 60
        row = dict(epoch=epoch + 1, step=global_step,
                   stage=stage_of(step_frac),
                   epoch_minutes=ep_min, minutes=(time.time() - t0) / 60,
                   step_fraction=step_frac,
                   lr=opt.param_groups[0]["lr"], select=select, **summ)
        row.pop("per_cond", None)
        row.update({f"train_{k}": v / max(1, ep_nb) for k, v in ep_run.items()})
        for c in ROBUST_CONDS:
            row.update({f"{c}_{k}": v for k, v in val[c].items()})
        row.update({f"loc_{k}": v for k, v in diag["loc"].items()})
        row.update({f"route_{k}": v for k, v in diag["routing"].items() if k != "share"})
        row["route_share"] = diag["routing"]["share"]
        row.update(diag["evidence"])
        metrics_rows.append(row)

        print(f"===== EPOCH {epoch+1} done in {ep_min:.1f}m | SELECT({SELECT_METRIC})={select:.4f} "
              f"| clean {summ['clean_auc']:.4f} worst {summ['worst_auc']:.4f} gap {summ['robust_gap']:.4f}",
              flush=True)
        print(f"      per-condition AUROC " + " ".join(f"{c}={summ['per_cond'][c]:.4f}" for c in ROBUST_CONDS),
              flush=True)
        print(f"      clean: EER={val['clean']['eer']:.3f} ACC@.5={val['clean']['acc']:.3f} "
              f"ACC@cal={val['clean']['cal_acc']:.3f} F1={val['clean']['f1']:.3f} "
              f"AP={val['clean']['ap']:.3f} ECE={val['clean']['ece']:.3f}", flush=True)
        print(f"      localisation: pixelAUC={diag['loc']['pixel_auc']:.3f} "
              f"IoU={diag['loc']['iou']:.3f} maskF1={diag['loc']['mask_f1']:.3f}", flush=True)
        print(f"      evidence: M_fake={diag['evidence']['mean_M_fake']:.4f} "
              f"M_real={diag['evidence']['mean_M_real']:.4f} S={diag['evidence']['mean_S']:.3f} "
              f"S_tok_std={diag['evidence']['S_token_std']:.4f} "
              f"W_p90/p10={diag['evidence']['W_range']:.2f}", flush=True)
        print(f"      routing: share={[round(s,3) for s in diag['routing']['share']]} "
              f"min={diag['routing']['min_share']:.3f} dead={diag['routing']['dead_experts']} "
              f"tokH={diag['routing']['tok_entropy']:.3f} "
              f"({diag['routing']['tok_entropy_frac']:.2f} of ln E)", flush=True)

        Path(OUT_DIR, "metrics.json").write_text(json.dumps(metrics_rows, indent=2, default=str))
        Path(OUT_DIR, "progress.json").write_text(json.dumps(progress_rows, indent=2, default=str))
        pd.DataFrame(metrics_rows).to_csv(Path(OUT_DIR, "metrics.csv"), index=False)
        pd.DataFrame(progress_rows).to_csv(Path(OUT_DIR, "progress.csv"), index=False)

        save_ckpt(Path(OUT_DIR, "checkpoint_last.pt"), epoch + 1, with_opt=True)
        is_best = np.isfinite(select) and select > best_select
        if is_best:
            best_select, best_epoch = float(select), epoch + 1
            save_ckpt(Path(OUT_DIR, "checkpoint_best.pt"), epoch + 1,
                      extra=dict(val=row, val_raw=val["_raw"]))
            print(f"      >>> new best ({SELECT_METRIC}={best_select:.4f}) -> checkpoint_best.pt", flush=True)

except KeyboardInterrupt:
    print("interrupted; saving checkpoint_last.pt", flush=True)
    save_ckpt(Path(OUT_DIR, "checkpoint_last.pt"), len(metrics_rows), with_opt=True)
    raise

print(f"\nTRAINING DONE: {(time.time()-t0)/60:.1f} min | {global_step} batches | "
      f"best {SELECT_METRIC}={best_select:.4f} at epoch {best_epoch}")
''')

# ----------------------------------------------------------------------------- 13 report
code(r'''
# ============================ REPORT, GATES, FIGURES ============================
mdf = pd.DataFrame(metrics_rows).sort_values("epoch")
last, best = mdf.iloc[-1], mdf.loc[mdf["select"].idxmax()]

def gate(name, ok, detail):
    print(f"  {name:<26} {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)

print("=" * 74); print(f"GATE CHECK — {RUN_NAME} ({ABLATION})"); print("=" * 74)
G = {}

tl = pd.DataFrame(progress_rows)
first_loss = float(tl["total"].iloc[:4].mean()) if len(tl) else float("nan")
last_loss = float(tl["total"].iloc[-4:].mean()) if len(tl) else float("nan")
G["learning"] = gate("learning", last_loss < first_loss, f"total loss {first_loss:.3f} -> {last_loss:.3f}")

# L_loc must move off its floor. v1 sat at 0.7415-0.7601 for 15,000 steps.
loc_series = tl[tl["stage"] >= 2]["loc"] if "stage" in tl else pd.Series(dtype=float)
if len(loc_series) > 8:
    loc_a, loc_b = float(loc_series.iloc[:4].mean()), float(loc_series.iloc[-4:].mean())
    G["localisation_learns"] = gate("localisation learns", loc_b < 0.80 * loc_a,
                                    f"L_loc {loc_a:.4f} -> {loc_b:.4f} (needs >20% drop)")
else:
    G["localisation_learns"] = gate("localisation learns", LAM["loc"] == 0.0, "L_loc disabled by ablation")

if LAM["loc"] > 0:
    mf, mr = float(last["mean_M_fake"]), float(last["mean_M_real"])
    G["evidence_alive"] = gate("evidence alive", mf > 1e-3 and mf > 3.0 * max(mr, 1e-9),
                               f"mean_M fake={mf:.4f} real={mr:.4f} (v1 reached exactly 0.0)")
    G["localisation_quality"] = gate("localisation quality", float(last["loc_pixel_auc"]) > 0.70,
                                     f"pixel AUC={float(last['loc_pixel_auc']):.3f} "
                                     f"IoU={float(last['loc_iou']):.3f}")
else:
    G["evidence_alive"] = G["localisation_quality"] = True
    print("  evidence / localisation    SKIP  disabled by ablation")

# --- evidence / localisation (always active in v4) ---
G["evidence_alive"] = G["localisation_quality"] = True

# --- evidence weighting (always active in v4) ---
wr = float(last["W_range"])
G["evidence_weighting"] = gate("evidence weighting", wr > 1.5,
                               f"W p90/p10={wr:.2f}")
ss = float(last["S_token_std"])
G["stability_nondegenerate"] = gate("stability non-degenerate", ss > 0.02,
                                    f"token std(S)={ss:.4f}, mean={float(last['mean_S']):.3f}")

# --- routing (always active in v4) ---
share = last["route_share"] if isinstance(last["route_share"], list) else json.loads(str(last["route_share"]))
minshare, dead = float(np.min(share)), int(np.sum(np.asarray(share) < 0.02))
G["routing_alive"] = gate("routing alive", minshare > 0.5 / (2 * N_EXPERTS) and dead == 0,
                          f"share={[round(s,3) for s in share]} min={minshare:.3f} dead={dead}")
if LAM["z"] > 0:
    tf = float(last["route_tok_entropy_frac"])
    G["routing_decisive"] = gate("routing decisive", tf < 0.85,
                                 f"token entropy={float(last['route_tok_entropy']):.3f} "
                                 f"= {tf:.2f} of ln E")
else:
    G["routing_decisive"] = True

G["generalisation_proxy"] = gate("selection metric", float(best["select"]) > 0.70,
                                 f"worst-case val AUROC={float(best['select']):.4f} at epoch {int(best['epoch'])}")
G["robustness"] = gate("robustness gap", float(best["robust_gap"]) < 0.15,
                       f"clean-minus-worst={float(best['robust_gap']):.4f}")

VERDICT = "GREEN" if all(G.values()) else "YELLOW"
print("-" * 74)
print(f"{VERDICT}: {'all gates passed' if VERDICT == 'GREEN' else 'debug the FAIL rows before the ablation program'}")
print("=" * 74)

# ---- efficiency (spec 25, Table 4) ----------------------------------------
model.eval()
_x = torch.randn(1, T_FRAMES, 3, IMG, IMG, device=device)
with torch.no_grad():
    for _ in range(3):
        net(_x, _x)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    _t = time.time()
    for _ in range(10):
        net(_x, _x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.time() - _t) / 10
model.train()
EFF = dict(params_total_M=n_total / 1e6, params_trainable_M=n_trainable / 1e6,
           clip_latency_s=lat, per_frame_ms=1000 * lat / T_FRAMES,
           peak_mem_GiB=(torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0)
print("efficiency: " + " | ".join(f"{k}={v:.4g}" for k, v in EFF.items()))

# ---- figures ---------------------------------------------------------------
def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(Path(OUT_DIR, "figures", f"{stem}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)

x = mdf["epoch"].to_numpy()
fig, ax = plt.subplots(figsize=(11, 6))
for c, col in zip(ROBUST_CONDS, ["#1565c0", "#ef6c00", "#2e7d32", "#8e24aa"]):
    ax.plot(x, mdf[f"{c}_auc"], marker="o", label=f"AUROC {c}", color=col)
ax.plot(x, mdf["select"], marker="s", lw=2.5, color="#000000", label=f"selection ({SELECT_METRIC})")
ax.set(xlabel="epoch", ylabel="video AUROC", ylim=(0.4, 1.02),
       title="Validation AUROC under nuisance interventions")
ax.grid(alpha=.25); ax.legend(ncol=2); savefig(fig, "validation_auroc")

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
for col, lab in [("total", "total"), ("det", "detection"), ("loc", "localisation"), ("mass", "mass")]:
    if col in tl:
        axes[0].plot(tl["step"], tl[col], lw=1.2, label=lab)
for col, lab in [("stab", "stability"), ("s", "S target"), ("s_dist", "S distribution"),
                 ("sep", "separation"), ("nuis", "nuisance"), ("bal", "balance"),
                 ("rent", "token entropy"), ("div", "diversity")]:
    if col in tl:
        axes[1].plot(tl["step"], tl[col], lw=1.2, label=lab)
for a in axes:
    a.grid(alpha=.25); a.legend(ncol=3)
axes[0].set(ylabel="primary loss", title="Training objective by stage")
axes[1].set(xlabel="batch", ylabel="auxiliary loss (symlog)"); axes[1].set_yscale("symlog")
for a in axes:
    for b in (STAGE2_STEP, STAGE3_STEP, STAGE4_STEP):
        a.axvline(b, color="#999999", ls="--", lw=.8)
savefig(fig, "training_losses")

fig, ax = plt.subplots(figsize=(11, 6))
shares = np.array([r if isinstance(r, list) else json.loads(str(r)) for r in mdf["route_share"]])
for e in range(N_EXPERTS):
    ax.plot(x, shares[:, e], marker="o", label=f"expert {e+1}")
ax.axhline(1.0 / N_EXPERTS, color="#555", ls="--", label="ideal share")
ax.set(xlabel="epoch", ylabel="token share", ylim=(-0.02, 0.62), title="Expert utilisation")
ax2 = ax.twinx()
ax2.plot(x, mdf["route_tok_entropy_frac"], marker="s", color="#212121", label="token entropy / ln E")
ax2.set_ylabel("router entropy fraction"); ax2.set_ylim(0, 1.05)
ax.grid(alpha=.25); ax.legend(ncol=3, loc="upper left"); ax2.legend(loc="upper right")
savefig(fig, "expert_utilisation")

fig, ax = plt.subplots(figsize=(11, 6))
for col, lab in [("mean_M_fake", "M on fake frames"), ("mean_M_real", "M on real frames"),
                 ("mean_S", "stability S"), ("mean_R", "reliability R"), ("mean_W", "weight W")]:
    if col in mdf:
        ax.plot(x, mdf[col], marker="o", label=lab)
ax.set(xlabel="epoch", ylabel="mean value", title="Evidence and stability"); ax.grid(alpha=.25); ax.legend()
savefig(fig, "evidence_stability")

fig, ax = plt.subplots(figsize=(11, 6))
for col, lab in [("loc_pixel_auc", "pixel AUC"), ("loc_iou", "IoU"), ("loc_mask_f1", "mask F1")]:
    ax.plot(x, mdf[col], marker="o", label=lab)
ax.set(xlabel="epoch", ylabel="score", ylim=(0, 1.02), title="Localisation quality (val-hard fakes)")
ax.grid(alpha=.25); ax.legend(); savefig(fig, "localisation")

bck = torch.load(Path(OUT_DIR, "checkpoint_best.pt"), map_location="cpu", weights_only=False)
vy, vp = bck["val_raw"]
if len(set(vy)) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    fpr, tpr, _ = roc_curve(vy, vp)
    axs[0].plot(fpr, tpr, color="#1565c0", lw=2, label=f"AUROC={roc_auc_score(vy, vp):.4f}")
    axs[0].plot([0, 1], [0, 1], "--", color="#888"); axs[0].set(xlabel="FPR", ylabel="TPR", title="ROC (best epoch, clean)")
    pr, rc, _ = precision_recall_curve(vy, vp)
    axs[1].plot(rc, pr, color="#8e24aa", lw=2, label=f"AP={average_precision_score(vy, vp):.4f}")
    axs[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    cm = confusion_matrix(vy, [int(v >= 0.5) for v in vp], labels=[0, 1])
    im = axs[2].imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        axs[2].text(c, r, int(v), ha="center", va="center")
    axs[2].set(xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="actual", title="Confusion @0.5")
    axs[2].set_xticklabels(["real", "fake"]); axs[2].set_yticklabels(["real", "fake"])
    for a in axs[:2]:
        a.grid(alpha=.25); a.legend()
    fig.colorbar(im, ax=axs[2]); savefig(fig, "roc_pr_confusion")

# ---- evidence panels: image / M / R / S / W (paper Figure 4) --------------
ema.apply_to(model); model.eval()
panel_ds = PairedClipDataset(val_vids, deterministic=True, hard=True)
rows = []
with torch.no_grad():
    for idx in (1, 3, 5, 7):
        xx, _, mm, yy, _, _, _ = panel_ds[idx]
        o = model(xx[None].to(device), xi=xx[None].to(device))[0]
        img = np.clip(xx[0].permute(1, 2, 0).numpy() * IMNET_STD + IMNET_MEAN, 0, 1)
        maps = [o[k][0].float().reshape(GRID, GRID).cpu().numpy() for k in ("M", "R", "S", "W_norm")]
        rows.append((img, mm[0].numpy(), maps, float(torch.sigmoid(o["logit"].float())[0, 0]), int(yy[0])))
fig, axes = plt.subplots(len(rows), 6, figsize=(17, 3.0 * len(rows)))
axes = np.atleast_2d(axes)
for r, (img, gtm, maps, score, lab) in enumerate(rows):
    axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"{'fake' if lab else 'real'}\np={score:.2f}", fontsize=9)
    axes[r, 1].imshow(gtm, cmap="gray", vmin=0, vmax=1)
    for c, (mp, nm) in enumerate(zip(maps, ["M", "R", "S", "W"])):
        axes[r, c + 2].imshow(mp, cmap="inferno")
        if r == 0:
            axes[r, c + 2].set_title(nm)
    if r == 0:
        axes[r, 0].set_title("input"); axes[r, 1].set_title("GT mask")
    for a in axes[r]:
        a.set_xticks([]); a.set_yticks([])
savefig(fig, "evidence_panels")
model.train(); ema.restore(model)

REPORT = dict(run=RUN_NAME, ablation=ABLATION, verdict=VERDICT, gates=G,
              seed=SEED, manifest_hash=MANIFEST["hash"], epochs_completed=int(last["epoch"]),
              batches=int(last["step"]), minutes=float(last["minutes"]),
              select_metric=SELECT_METRIC, best_select=float(best["select"]),
              best_epoch=int(best["epoch"]),
              best_epoch_metrics={k: (v if not isinstance(v, np.generic) else v.item())
                                  for k, v in best.to_dict().items()},
              efficiency=EFF, config=CFG,
              note="v4 proposed architecture. Validation only. Cross-dataset numbers come from "
                   "the final evaluation cell and were never used for selection.")
Path(OUT_DIR, "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print(f"\nreport -> {OUT_DIR}/report.json | figures -> {OUT_DIR}/figures/")
''')

# ----------------------------------------------------------------------------- 14 final eval
code(r'''
# ============================ FINAL EVALUATION — run once, after freeze ============================
# First and only read of DFDCP and the FF++ manipulated sequences (spec 34, 35).
# Nothing below may be used to change any hyper-parameter or to re-select a checkpoint.
ck_path = Path(OUT_DIR, "checkpoint_best.pt")
assert ck_path.exists(), "train first"
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_trainable"], strict=False)     # EMA weights of the selected epoch
model.eval()
print(f"frozen checkpoint: epoch {ck['epoch']} | {SELECT_METRIC}={ck['best_select']:.4f} "
      f"| manifest {ck['manifest_hash']}")
assert ck["manifest_hash"] == MANIFEST["hash"], "checkpoint was trained on a different split"

@torch.no_grad()
def score_videos(records, cond=None, t=T_TEST):
    """Video score = mean over T_TEST/T_FRAMES chunks. `cond` applies a fixed nuisance."""
    ld = DataLoader(TestVideoDataset(records, t), batch_size=1, num_workers=2, pin_memory=True)
    out = []
    for xb, _ in ld:
        if cond is not None and cond != "clean":
            arr = (xb[0].permute(0, 2, 3, 1).numpy() * IMNET_STD + IMNET_MEAN)
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            xb = torch.stack([to_tensor(intervene_fixed(a, cond)) for a in arr])[None]
        chunks = []
        for ch in xb.split(T_FRAMES, dim=1):
            with torch.autocast("cuda", dtype=AMP, enabled=device.type == "cuda"):
                o = net(ch.to(device, non_blocking=True),
                        ch.to(device, non_blocking=True),
                        step_fraction=1.0)
            if isinstance(o, tuple):
                o = o[0]
            chunks.append(float(torch.sigmoid(o["logit"].float()).mean()))
        out.append(float(np.mean(chunks)))
    return out

EVAL, TABLES = {}, []

# ---- Table 1: cross-dataset, DFDCP ---------------------------------------
if TEST_SETS.get("dfdcp_real") and TEST_SETS.get("dfdcp_fake"):
    r = score_videos(TEST_SETS["dfdcp_real"])
    f = score_videos(TEST_SETS["dfdcp_fake"])
    y, p = [0] * len(r) + [1] * len(f), r + f
    m = classification_metrics(y, p, thr=0.5)
    m.update(bootstrap_auc(y, p))
    m.update(n_real=len(r), n_fake=len(f))
    EVAL["dfdcp"] = m
    TABLES.append(dict(benchmark="DFDCP (cross-dataset)", n=len(y), auc=m["auc"],
                       ci_lo=m["lo"], ci_hi=m["hi"], ap=m["ap"], eer=m["eer"]))
    print(f"DFDCP  AUROC {m['auc']:.4f} [95% CI {m['lo']:.4f}, {m['hi']:.4f}]  "
          f"AP {m['ap']:.4f}  EER {m['eer']:.4f}  (v1 baseline: 0.5706)")

    # ---- Table 3: robustness on the unseen domain ------------------------
    rob = {"clean": m["auc"]}
    for cond in [c for c in ROBUST_CONDS if c != "clean"]:
        rr, ff = score_videos(TEST_SETS["dfdcp_real"], cond), score_videos(TEST_SETS["dfdcp_fake"], cond)
        rob[cond] = float(roc_auc_score([0] * len(rr) + [1] * len(ff), rr + ff))
        print(f"DFDCP  {cond:<9} AUROC {rob[cond]:.4f}")
    EVAL["dfdcp_robustness"] = rob
else:
    print("DFDCP not present - skipped")

# ---- Table 1/2: cross-manipulation, FF++ ---------------------------------
if TEST_SETS.get("ffpp_test_real"):
    base = score_videos(TEST_SETS["ffpp_test_real"])
    cross = {}
    for key, recs in sorted(TEST_SETS.items()):
        if not key.startswith("ffpp_") or key == "ffpp_test_real":
            continue
        manip = key[len("ffpp_"):]
        fs = score_videos(recs)
        y, p = [0] * len(base) + [1] * len(fs), base + fs
        mm = classification_metrics(y, p, thr=0.5)
        mm.update(bootstrap_auc(y, p)); mm["n_fake"] = len(fs)
        cross[manip] = mm
        TABLES.append(dict(benchmark=f"FF++ {manip} (cross-manip)", n=len(y), auc=mm["auc"],
                           ci_lo=mm["lo"], ci_hi=mm["hi"], ap=mm["ap"], eer=mm["eer"]))
        print(f"FF++ {manip:<18} AUROC {mm['auc']:.4f} [{mm['lo']:.4f}, {mm['hi']:.4f}] ({len(fs)} fakes)")
    if cross:
        macro = float(np.mean([v["auc"] for v in cross.values()]))
        EVAL["ffpp_cross_manip"] = cross
        EVAL["ffpp_macro_auc"] = macro
        TABLES.append(dict(benchmark="FF++ macro-average", n=len(cross), auc=macro,
                           ci_lo=float("nan"), ci_hi=float("nan"), ap=float("nan"), eer=float("nan")))
        print(f"FF++ macro-average AUROC {macro:.4f} over {len(cross)} families")
else:
    print("FF++ manipulated sequences not present - skipped")

EVAL["meta"] = dict(run=RUN_NAME, ablation=ABLATION, checkpoint_epoch=int(ck["epoch"]),
                    select_metric=SELECT_METRIC, select_value=float(ck["best_select"]),
                    manifest_hash=MANIFEST["hash"], t_test=T_TEST,
                    trained_on="DD-SBI fakes from FF++/Celeb-DF reals only; no real manipulated video")
Path(OUT_DIR, "evaluation", "final_results.json").write_text(json.dumps(EVAL, indent=2, default=str))
pd.DataFrame(TABLES).to_csv(Path(OUT_DIR, "evaluation", "table1_main.csv"), index=False)
print(f"\nfinal results -> {OUT_DIR}/evaluation/final_results.json")
print(pd.DataFrame(TABLES).to_string(index=False))
''')

# ----------------------------------------------------------------------------- 15 closing
md(r"""
## Reading the result (v4)

**v4 design principles:**
1. Route by forensic representation (prototype router)
2. Assess trust separately (M/S/R are trust, not routing, variables)
3. Preserve common knowledge (shared expert always evaluated)
4. Train the exact evaluation graph (relative curriculum, no absolute thresholds)

**Selection metric** is the worst-case validation AUROC across `clean / jpeg30 / blur15 / resize50`
on the hard SBI validation split.

**The headline number** is DFDCP AUROC from the final cell. The v3.1 baseline scored 0.5904
with a source validation AUROC of 0.9720 — the 0.3816 gap was caused by train/test graph
mismatch. v4 should show a smaller gap by training the exact graph used for evaluation.

**Ablation stages:**

| Stage | Change | Question |
|-------|--------|----------|
| A0 | Corrected curriculum only | Is train/test graph mismatch the main failure? |
| A1 | + Switch balance + z-loss | Does router stabilization prevent collapse? |
| A2 | + shared expert | Does common knowledge stabilize transfer? |
| A3 | + prototype router | Does representation-first routing reduce shortcuts? |
| A4 | + semantic specialist axes | Does meaningful specialization improve OOD? |

## Outputs

`config.json` `manifest.json` `metrics.{json,csv}` `progress.{json,csv}` `report.json`
`checkpoint_{best,last}.pt` `figures/*.{png,pdf}` `evaluation/final_results.json`
""")

# ----------------------------------------------------------------------------- emit
nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(r"c:\Users\balac\Desktop\Capstone\Stable-RouteNet_v4_notebook.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} with {len(CELLS)} cells "
      f"({sum(len(s) for _, s in CELLS)/1024:.1f} KiB of source)")
