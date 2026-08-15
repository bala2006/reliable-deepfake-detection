"""Generate tools/kaggle/train_temporal_region_kaggle.ipynb embedding kaggle_driver.py."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRIVER = (ROOT / "kaggle_driver.py").read_text(encoding="utf-8")
OUT = ROOT / "train_temporal_region_kaggle.ipynb"

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [src]})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.splitlines(keepends=True)})


md("""# RL-ROI-Net Autonomous Kaggle Training (BIG plan, v3)

Self-driving **1-seed-gate then full 3-seed** temporal-region training on the
full FF++ pool (~7,964 train / 1,220 val / 1,216 test sequences), **60 epochs**
per seed, with **early stopping** (patience 12) so each seed stops once val AUC
stops improving (~15-20 epochs), and a **cross-set evaluation** after seed 0.

## What it does automatically
1. Reads `rlroinet_kaggle.zip` from the mounted Kaggle dataset `/kaggle/input`
2. Extracts code + weights + FF++ manifest data into `/kaggle/working` (local, no download)
3. Installs dependencies (torch is already preinstalled on Kaggle)
4. Patches the runtime code to the current versions (trainer, temporal_region, cross-set eval)
5. Pulls prior checkpoints/state from a private Kaggle dataset (resume across the 12h session cap)
6. Detects GPU: **T4 => fp16** (native tensor cores), Ampere+ => bf16
7. Runs **seed 0** alone on GPU 0; every metrics table now has a **PASS/FAIL**
   column against the paper acceptance bar + an OVERALL acceptance row
8. **Gate:** seed-0 max val video AUC >= 0.80 => next step; else FAILED + stop
9. **Cross-set eval:** FF++ per-method breakdown + Celeb-DF (if present) -> saved
   to `outputs/seed0/eval_cross_set.json`, printed before escalating
10. **Seeds 1+2 run in PARALLEL**, one per GPU (both T4s used, not just GPU 0)
11. Full run => `run_summary.json` (mean +/- std) + human-readable `RESULTS.md`

## Required before running
- **Accelerator: GPU T4 x2** and **Internet ON** (right sidebar -> Settings)
- The private Kaggle dataset `rlroinet-kaggle` with `rlroinet_kaggle.zip` inside
  (add it via the **Add Input** button or it is already linked to this notebook)
- Optional overrides: `RLROINET_SEEDS` (default `0 1 2`), `RLROINET_EPOCHS` (60),
  `RLROINET_GATE_AUC` (0.80), `RLROINET_EARLY_STOP_PATIENCE` (12),
  `RLROINET_SNAPSHOT_MINUTES` (30), `RLROINET_BUDGET_MINUTES` (540 = 9h hard quota),
  `RLROINET_CROSS_QUICK` (100 = Celeb-DF videos/class sampled for the cross-set check)
""")

code('''# Cell 1 - Environment (run once per session)
import os, sys
from pathlib import Path

ZIP_NAME = os.environ.get("RLROINET_ZIP_NAME", "rlroinet_kaggle.zip")
DATASET_NAME = os.environ.get("RLROINET_DATASET", "rlroinet-kaggle")
WORK = Path(os.environ.get("RLROINET_WORK", "/kaggle/working"))

print("kaggle user:", os.environ.get("KAGGLE_USERNAME", "") or "(not set)")
print("torch will not be reinstalled (preinstalled on Kaggle)")''')
code('''# Cell 2 - Unpack the package from /kaggle/input (run once per session)
import shutil, os, zipfile
from pathlib import Path

WORK = Path(os.environ.get("RLROINET_WORK", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

INPUT = Path(os.environ.get("RLROINET_INPUT", "/kaggle/input"))
print("contents of", INPUT, ":")
if not INPUT.exists():
    print("  <does not exist>")
for p in sorted(INPUT.rglob("*"))[:50]:
    depth = len(p.parts) - len(INPUT.parts)
    print("  " * depth + p.name + ("/" if p.is_dir() else ""))

# Case 1: select the configured package only; never extract an arbitrary zip.
zip_candidates = [INPUT / DATASET_NAME / ZIP_NAME, INPUT / ZIP_NAME]
zip_candidates.extend(p for p in INPUT.rglob(ZIP_NAME) if p not in zip_candidates)
zip_path = next((p for p in zip_candidates if p.is_file()), None)
if zip_path is not None:
    print("found configured package zip:", zip_path)
    if not (WORK / "rlroinet" / "__init__.py").exists():
        print("extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(WORK)
        print("extracted entries:", len(zipfile.ZipFile(zip_path).namelist()))
    else:
        print("already extracted; skipping")
# Case 2: the tree was uploaded already extracted -> copy it from the configured dataset.
elif (WORK / "rlroinet" / "__init__.py").exists():
    print("rlroinet already in WORK; skipping")
else:
    init_files = list((INPUT / DATASET_NAME).rglob("rlroinet/__init__.py")) \\
        if (INPUT / DATASET_NAME).exists() else []
    if init_files:
        src_root = init_files[0].parent.parent
        print("found extracted tree at:", src_root)
        print("copying source tree and data into WORK...")
        for d in ("rlroinet", "outputs"):
            src = src_root / d
            if src.is_dir():
                shutil.copytree(src, WORK / d, dirs_exist_ok=True)
        for d in src_root.iterdir():
            if d.is_dir() and d.name in ("data",):
                shutil.copytree(d, WORK / d.name, dirs_exist_ok=True)
        for f in ("requirements.txt", "pyproject.toml"):
            src = src_root / f
            if src.is_file():
                shutil.copy2(src, WORK / f)
        print("copied tree entries:", len(list(WORK.rglob("*"))))
    else:
        raise SystemExit(
            f"no {ZIP_NAME!r} and no extracted rlroinet tree found under "
            f"{INPUT / DATASET_NAME}. Attach the {DATASET_NAME} dataset via Add Input."
        )

import rlroinet
print("rlroinet OK from", rlroinet.__file__)''')

code('''# Cell 3 - Repair + verify deps (idempotent; fixes a broken numpy if present)
import importlib, subprocess, sys

def _have(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False

def _pip(args, label):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--no-cache-dir", "--no-warn-conflicts", *args])
    if r.returncode != 0:
        raise SystemExit(f"{label} install failed (rc={r.returncode})")

# numpy/scipy/sklearn/cv2/matplotlib are ALL preinstalled and consistent on a
# FRESH Kaggle session. Only if one is broken do we uninstall the damaged stack
# and reinstall everything TOGETHER (no version pins) so the resolver picks a
# compatible matrix. The pip warnings about tpot/numba/umap etc. are for
# preinstalled packages we do not use -- safe to ignore.
STACK = ("numpy", "scipy", "sklearn", "cv2", "matplotlib")
MODS = ("numpy", "scipy", "sklearn", "cv2", "matplotlib")
need = [m for m in MODS if not _have(m)]
if need:
    print("broken/missing deps:", need)
    print("uninstalling the damaged stack, then reinstalling consistently...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "numpy", "scipy", "scikit-learn", "matplotlib",
                    "opencv-python", "opencv-python-headless"],
                   capture_output=True)
    _pip(["numpy", "scipy", "scikit-learn", "opencv-python-headless",
          "matplotlib"], "deps")
    still = [m for m in MODS if not _have(m)]
    if still:
        raise SystemExit(
            "Environment still broken after repair. Restart this session "
            "(top-right stop, then start again) to get a clean Kaggle image, "
            "then re-run cells 1-3."
        )
else:
    print("all deps healthy (numpy, scipy, sklearn, cv2, matplotlib); nothing to install")

# final verification
import cv2, sklearn, numpy
print("cv2", cv2.__version__, "| sklearn", sklearn.__version__, "| numpy", numpy.__version__)''')

import base64
driver_b64 = base64.b64encode(DRIVER.encode("utf-8")).decode("ascii")
code('# Cell 4 - Write the autonomous driver to disk (base64: no escaping issues)\n'
     'import base64\n'
     '_B64 = "' + driver_b64 + '"\n'
     'from pathlib import Path\n'
     'Path("/kaggle/working/kaggle_driver.py").write_bytes(base64.b64decode(_B64))\n'
     'print("driver written:", len(base64.b64decode(_B64)), "chars")')

import pathlib
fix_b64 = base64.b64encode(
    (pathlib.Path(__file__).resolve().parents[2] / "rlroinet" / "temporal_region.py")
    .read_text(encoding="utf-8").encode("utf-8")).decode("ascii")
code('# Cell 4b - Patch temporal_region.py to the fixed version if stale (idempotent)\n'
     'import base64\n'
     '_B64 = "' + fix_b64 + '"\n'
     'from pathlib import Path\n'
     'from importlib import import_module\n'
     '_p = Path("/kaggle/working/rlroinet/temporal_region.py")\n'
     'if "def _compose" in _p.read_text():\n'
     '    print("temporal_region.py already fixed; skipping")\n'
     'else:\n'
     '    _p.write_bytes(base64.b64decode(_B64))\n'
     '    print("patched temporal_region.py with the fixed _compose version")\n'
     'import importlib as _il\n'
     'import rlroinet.temporal_region as _tr\n'
     '_il.reload(_tr)\n'
     'print("_compose present:", hasattr(_tr.TemporalRegionAgent, "_compose"))')

root_rlroinet = pathlib.Path(__file__).resolve().parents[2] / "rlroinet"
train_b64 = base64.b64encode((root_rlroinet / "train_temporal_region.py")
                             .read_text(encoding="utf-8").encode("utf-8")).decode("ascii")
cross_b64 = base64.b64encode((root_rlroinet / "eval_cross_set.py")
                             .read_text(encoding="utf-8").encode("utf-8")).decode("ascii")
code('# Cell 4c - Patch trainer + cross-set eval to the current versions (idempotent)\n'
     'import base64\n'
     'from pathlib import Path\n'
     '_ROOT = Path("/kaggle/working/rlroinet")\n'
     '_TRAIN = "'
     + train_b64 + '"\n'
     '_CROSS = "'
     + cross_b64 + '"\n'
     '_tp = _ROOT / "train_temporal_region.py"\n'
     '_txt = _tp.read_text(encoding="utf-8") if _tp.exists() else ""\n'
     'if "def _compose" in _txt and "PASS_BAR" in _txt and "early_stop_patience" in _txt:\n'
     '    print("train_temporal_region.py already current; skipping")\n'
     'else:\n'
     '    _tp.write_bytes(base64.b64decode(_TRAIN))\n'
     '    print("patched train_temporal_region.py (pass/fail + early stopping)")\n'
     '_cp = _ROOT / "eval_cross_set.py"\n'
     '_ctxt = _cp.read_text(encoding="utf-8") if _cp.exists() else ""\n'
     'if "_CHECKPOINT_FORMAT" in _ctxt:\n'
     '    print("eval_cross_set.py already current; skipping")\n'
     'else:\n'
     '    _cp.write_bytes(base64.b64decode(_CROSS))\n'
     '    print("wrote eval_cross_set.py (cross-set evaluation)")\n')

code('''# Cell 5 - RUN (autonomous driver; walk away after starting this cell)
import sys, os, importlib
sys.path.insert(0, "/kaggle/working")
import kaggle_driver
importlib.reload(kaggle_driver)
kaggle_driver.main()''')

md("""## After the run
The driver ends with `DONE`. Download your results from `/kaggle/working/outputs`:
- `RESULTS.md` — human-readable, per-metric **PASS/FAIL** vs the acceptance bar + mean +/- std
- `run_summary.json` — machine-readable aggregate
- `seed0|1|2/test_metrics.json`, `decision_policy.json`, `run_config.json`, `metrics.json`
- `seed0/eval_cross_set.json` — FF++ per-method + Celeb-DF cross-set evaluation
Run the cell below to bundle them into one zip for download.""")
code('''# Cell 6 - Bundle results for download (run after the driver reports DONE)
import os, shutil, tempfile
from pathlib import Path

WORK = Path(os.environ.get("RLROINET_WORK", "/kaggle/working"))
source = WORK / "outputs"
out = WORK / "results_download.zip"
if out.exists():
    out.unlink()

# Explicit allowlist: no feature_cache, no arbitrary files, and no per-epoch
# checkpoint snapshots. Diagnostics are retained recursively for auditability.
root_files = {"RESULTS.md", "run_summary.json"}
seed_files = {"test_metrics.json", "decision_policy.json", "run_config.json",
              "metrics.json", "eval_cross_set.json"}
with tempfile.TemporaryDirectory(dir=WORK) as staging_name:
    staging = Path(staging_name) / "outputs"
    staging.mkdir()
    for name in root_files:
        src = source / name
        if src.is_file():
            shutil.copy2(src, staging / name)
    for src in sorted(source.glob("seed*")):
        if not src.is_dir():
            continue
        rel_seed = staging / src.name
        for path in src.rglob("*"):
            rel = path.relative_to(src)
            if path.is_file() and (
                rel.name in seed_files
                or (len(rel.parts) == 2 and rel.parts[0] == "checkpoints"
                    and rel.name in {"best.pt", "final.pt"})
                or "diagnostics" in rel.parts
            ):
                destination = rel_seed / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    diagnostics = source / "diagnostics"
    if diagnostics.is_dir():
        shutil.copytree(diagnostics, staging / "diagnostics", dirs_exist_ok=True)
    shutil.make_archive(str(out).with_suffix(""), "zip", staging)
print("results bundle:", out, f"({out.stat().st_size/1e6:.1f} MB) - download from the output panel")''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
