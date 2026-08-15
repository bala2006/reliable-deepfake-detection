"""Autonomous Kaggle driver for RL-ROI-Net temporal-region training.

A self-driving state machine: pull persisted state, resume the exact seed/
epoch where the previous session stopped, run seed 0, evaluate a health gate,
and only escalate to the full 3-seed run if the gate passes.  On gate failure
or completion it marks the state and ends the session (frees the GPU).

Runtime decisions (no manual input):
  * GPU type  -> capability >= 8.0 (Ampere+) uses bf16, Turing/older (T4/P100)
                uses fp16 (+GradScaler) so the tensor cores actually accelerate
  * CPU count -> --workers / --prefetch-factor derived from the session
  * GPUs      -> seed 0 runs alone, then seeds 1+2 run in PARALLEL, one per GPU
  * Resume    -> nearest snapshot of the current seed (the trainer --resume)
  * Gate      -> seed-0 max val video AUC >= RLROINET_GATE_AUC (default 0.80)
  * Cross-set -> after seed 0, run FF++ per-method + Celeb-DF (if present) eval
  * Budget    -> exit before the 12h session cap, pushing state first

Persistence: after every 30-min snapshot and at every state transition the
sync dir (checkpoints + metrics + state) is pushed to a private Kaggle
dataset via the ``kaggle`` CLI (pre-authenticated in notebooks through
KAGGLE_USERNAME / KAGGLE_KEY).  Restarting the notebook pulls that dataset,
inspects each seed's checkpoint dir (source of truth: final.pt / test_metrics),
and resumes.

Run directly (e.g. ``python kaggle_driver.py``) or as a notebook cell; all
config comes from environment variables so the same file drives both.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration (env-overridable)
# --------------------------------------------------------------------------

DRIVE_LINK = os.environ.get("RLROINET_DRIVE_LINK", "").strip()
ZIP_NAME = os.environ.get("RLROINET_ZIP_NAME", "rlroinet_kaggle.zip")
USERNAME = os.environ.get("KAGGLE_USERNAME", "").strip()
DATASET = os.environ.get("RLROINET_DATASET", "rlroinet-kaggle")
DATASET_REF = f"{USERNAME}/{DATASET}" if USERNAME else DATASET
GPU_IDS = [int(x) for x in os.environ.get("RLROINET_GPU_IDS", "0 1").split()]

WORK = Path(os.environ.get("RLROINET_WORK", "/kaggle/working"))
PULL = WORK / "pull"
SYNC = WORK / "sync"
STATE = SYNC / "state.json"
RESUME_MANIFEST = SYNC / "resume_manifest.json"

SEEDS = [int(x) for x in os.environ.get("RLROINET_SEEDS", "0 1 2").split()]
EPOCHS = int(os.environ.get("RLROINET_EPOCHS", "60"))
GATE_AUC = float(os.environ.get("RLROINET_GATE_AUC", "0.80"))
SNAPSHOT_MINUTES = float(os.environ.get("RLROINET_SNAPSHOT_MINUTES", "30"))
PUSH_INTERVAL_S = max(120.0, float(SNAPSHOT_MINUTES) * 60.0 * 0.9)
BUDGET_MINUTES = float(os.environ.get("RLROINET_BUDGET_MINUTES", "540"))  # hard 9h training budget

BATCH = int(os.environ.get("RLROINET_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("RLROINET_GRAD_ACCUM", "1"))
HEAD_LR = os.environ.get("RLROINET_HEAD_LR", "3e-4")
SEQUENCE_LENGTH = int(os.environ.get("RLROINET_SEQUENCE_LENGTH", "8"))
TRAIN_SAMPLE = int(os.environ.get("RLROINET_TRAIN_SAMPLE", "2400"))
VAL_SAMPLE = int(os.environ.get("RLROINET_VAL_SAMPLE", "600"))
TEST_SAMPLE = int(os.environ.get("RLROINET_TEST_SAMPLE", "600"))
WORKERS_MAX = int(os.environ.get("RLROINET_WORKERS_MAX", "8"))
EARLY_STOP_PATIENCE = int(os.environ.get("RLROINET_EARLY_STOP_PATIENCE", "12"))
# Cross-set eval runs automatically after seed 0; escalate to seeds 1+2 unless
# the cross-set AUC is below this floor (0 disables the floor -> always escalate).
CROSS_GATE_AUC = float(os.environ.get("RLROINET_CROSS_GATE_AUC", "0"))
CROSS_QUICK = int(os.environ.get("RLROINET_CROSS_QUICK", "100"))  # cap Celeb-DF videos/class

FFPP_DIR = WORK / "data" / "FaceForensics++"
CACHE_DIR = WORK / "outputs" / "feature_cache"
VERDICT = "honi05"


def log(msg: str) -> None:
    print(f"[driver] {msg}", flush=True)


# --------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------

def detect_amp() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(0)
            if major >= 8:      # Ampere/Hopper/Ada: native bf16 tensor cores
                return "bf16"
            return "fp16"       # Turing (T4) / Volta / Pascal: fp16 + GradScaler
    except Exception:
        pass
    return "fp16"


def detect_workers(parallel: bool = False) -> int:
    cpus = os.cpu_count() or 4
    w = max(2, min(WORKERS_MAX, cpus))
    return max(2, w // 2) if parallel else w


def _gpu_count() -> int:
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 1


# --------------------------------------------------------------------------
# Seed layout: each seed owns its own parent so metrics.json / test_metrics.json
# never collide across seeds.
# --------------------------------------------------------------------------

def seed_root(seed: int) -> Path:
    return WORK / f"outputs/seed{seed}"


def seed_ckpt_dir(seed: int) -> Path:
    return seed_root(seed) / "checkpoints"


def seed_test_metrics(seed: int) -> Path:
    return seed_root(seed) / "test_metrics.json"


def seed_metrics(seed: int) -> Path:
    return seed_root(seed) / "metrics.json"


def seed_done(seed: int) -> bool:
    """Source of truth: trainer writes final.pt + test_metrics.json on completion."""
    return seed_test_metrics(seed).exists()


def seed_has_work(seed: int) -> bool:
    ck = seed_ckpt_dir(seed)
    return ck.exists() and any(ck.glob("*.pt"))


def next_action() -> tuple:
    """Return (action, seed) deciding what to do next.

    Actions: 'fail_stop' | 'done_stop' | 'resume' | 'run'.
    """
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
            if st.get("status") == "failed":
                return "fail_stop", SEEDS[0]
            if st.get("status") == "done":
                return "done_stop", SEEDS[-1]
        except Exception:
            pass
    for seed in SEEDS:
        if seed_done(seed):
            continue
        if seed_has_work(seed):
            return "resume", seed
        return "run", seed
    return "done_stop", SEEDS[-1]


# --------------------------------------------------------------------------
# Trainer command
# --------------------------------------------------------------------------

def build_cmd(seed: int, resume: bool, workers: int) -> list:
    cmd = [
        sys.executable, "-m", "rlroinet.train_temporal_region",
        "--ffpp-dir", str(FFPP_DIR),
        "--checkpoint-dir", str(seed_ckpt_dir(seed)),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--grad-accum", str(GRAD_ACCUM),
        "--workers", str(workers),
        "--prefetch-factor", "2",
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--manifest-frames-per-video", "8",
        "--train-sample", str(TRAIN_SAMPLE),
        "--val-sample", str(VAL_SAMPLE),
        "--test-sample", str(TEST_SAMPLE),
        "--head-lr", HEAD_LR,
        "--lr-min-factor", "0.1",
        "--lr-warmup-epochs", "2",
        "--snapshot-minutes", str(SNAPSHOT_MINUTES),
        "--amp-dtype", detect_amp(),
        "--feature-cache", str(CACHE_DIR),
        "--no-train-augmentation",
        "--early-stop-patience", str(EARLY_STOP_PATIENCE),
        "--seed", str(seed),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


# --------------------------------------------------------------------------
# Kaggle dataset push / pull
# --------------------------------------------------------------------------

def _kaggle_cli() -> str:
    cli = shutil.which("kaggle")
    if cli is None:
        raise RuntimeError("kaggle CLI not found; run `!pip install -q kaggle` first")
    return cli


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_resume_artifacts() -> dict:
    """Mirror trainer outputs into the sync payload before every dataset push.

    The mirror intentionally includes feature caches and every checkpoint
    snapshot because they are needed to continue a run without recomputing or
    losing optimizer/scheduler/epoch state.  The results-download notebook
    bundle applies a separate, smaller release allowlist.
    """
    source = WORK / "outputs"
    staged = SYNC / "outputs"
    SYNC.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        shutil.rmtree(staged)
    if source.exists():
        shutil.copytree(source, staged, dirs_exist_ok=True)
        download_bundle = staged / "results_download.zip"
        if download_bundle.exists():
            download_bundle.unlink()

    files = {}
    if staged.exists():
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                rel = path.relative_to(staged).as_posix()
                files[rel] = _sha256(path)

    best_checkpoints = {}
    for path in sorted(staged.glob("seed*/checkpoints/best.pt")) if staged.exists() else []:
        seed_name = path.parts[-3]
        best_checkpoints[seed_name] = path.relative_to(staged).as_posix()
    manifest = {
        "version": 1,
        "files": files,
        "best_checkpoints": best_checkpoints,
    }
    RESUME_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _sync_outputs_root() -> Path | None:
    current = SYNC / "outputs"
    if current.is_dir():
        return current
    # Accept the pre-fix layout once, then rehydrate it into the canonical one.
    legacy_seeds = [p for p in SYNC.glob("seed*") if p.is_dir()]
    return SYNC if legacy_seeds else None


def verify_resume_artifacts(root: Path | None = None) -> bool:
    """Return False if any staged artifact is missing or changed after restore."""
    if not RESUME_MANIFEST.exists():
        return True
    try:
        manifest = json.loads(RESUME_MANIFEST.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    output_root = Path(root) if root is not None else WORK / "outputs"
    for rel, expected in files.items():
        path = output_root / Path(rel)
        if not path.is_file() or _sha256(path) != expected:
            return False
    return True


def rehydrate_resume_artifacts() -> bool:
    """Copy pulled sync artifacts into the paths consumed by the trainer."""
    source = _sync_outputs_root()
    if source is None:
        return False
    target = WORK / "outputs"
    target.mkdir(parents=True, exist_ok=True)
    if source == SYNC:
        for seed_dir in source.glob("seed*"):
            if seed_dir.is_dir():
                shutil.copytree(seed_dir, target / seed_dir.name, dirs_exist_ok=True)
    else:
        shutil.copytree(source, target, dirs_exist_ok=True)
    if not verify_resume_artifacts():
        raise RuntimeError("resume artifact integrity check failed after rehydration")
    return True


def _write_dataset_metadata() -> None:
    SYNC.mkdir(parents=True, exist_ok=True)
    meta = SYNC / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(json.dumps({
            "id": DATASET_REF,
            "title": DATASET.replace("-", " ").title(),
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8")


def push_sync(message: str) -> None:
    stage_resume_artifacts()
    if not USERNAME:
        log("no KAGGLE_USERNAME; skipping dataset push")
        return
    _write_dataset_metadata()
    cli = _kaggle_cli()
    for args in (
        [cli, "datasets", "version", "-p", str(SYNC), "-m", message],
        [cli, "datasets", "create", "-p", str(SYNC), "-m", message],
    ):
        log(f"push: {' '.join(args[2:])}")
        proc = subprocess.run(args, capture_output=True, text=True, timeout=180)
        if proc.returncode == 0:
            return
        log(f"push attempt failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    log("WARNING: dataset push failed (will retry at next snapshot)")


def pull_sync() -> None:
    if not USERNAME:
        rehydrate_resume_artifacts()
        return
    cli = _kaggle_cli()
    if PULL.exists():
        shutil.rmtree(PULL)
    PULL.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [cli, "datasets", "download", "-d", DATASET_REF, "-p", str(PULL), "--unzip"],
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        log("no prior dataset found; starting fresh")
        return
    for src in PULL.rglob("*"):
        if src.is_file():
            rel = src.relative_to(PULL)
            dst = SYNC / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    rehydrate_resume_artifacts()
    log("restored prior state + checkpoints + trainer artifacts")


# --------------------------------------------------------------------------
# Gate evaluation
# --------------------------------------------------------------------------

def evaluate_gate(seed: int) -> tuple:
    """Return (passed, summary) for the seed-0 health gate."""
    summary = {"seed": seed, "gate_auc": GATE_AUC}
    path = seed_test_metrics(seed)
    if not path.exists():
        return False, {**summary, "reason": "missing test_metrics.json"}
    try:
        tm = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, {**summary, "reason": f"unreadable test_metrics: {exc}"}
    best = tm.get("validation_best_auc", tm.get("auc"))
    summary["validation_best_auc"] = best
    summary["test_auc"] = tm.get("auc")
    summary["region_iou"] = tm.get("region_iou")
    summary["region_hit"] = tm.get("region_hit")
    for key in ("auc", "acc", "confident_false_positive_rate",
                "confident_false_negative_rate"):
        if not isinstance(tm.get(key), (int, float)):
            return False, {**summary, "reason": f"metric {key} missing/invalid"}
    if best is None or not isinstance(best, (int, float)) or best != best:
        return False, {**summary, "reason": "val AUC missing or NaN"}
    if float(best) < GATE_AUC:
        return False, {**summary, "reason": f"val AUC {best:.4f} < gate {GATE_AUC}"}
    return True, summary


# --------------------------------------------------------------------------
# Training run with snapshot-driven push + wall-clock budget
# --------------------------------------------------------------------------

def _start_trainer(seed: int, resume: bool, workers: int, gpu: int | None) -> dict:
    """Launch one trainer subprocess pinned to a single GPU (or default)."""
    cmd = build_cmd(seed, resume, workers)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK) + os.pathsep + env.get("PYTHONPATH", "")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    seed_root(seed).mkdir(parents=True, exist_ok=True)
    log_path = seed_root(seed) / f"train_seed{seed}.log"
    lf = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(WORK), env=env, stdout=lf, stderr=subprocess.STDOUT)
    handle = {
        "seed": seed, "proc": proc, "log_file": lf, "log_path": log_path,
        "last_push": time.monotonic(),
        "known": {p.name for p in seed_ckpt_dir(seed).glob("*.pt")}
                 if seed_ckpt_dir(seed).exists() else set(),
        "tail_offset": 0,
    }
    log(f"seed {seed} {'resume' if resume else 'fresh'} gpu={gpu if gpu is not None else 'any'}: "
        f"{' '.join(cmd)}")
    return handle


def _pump_trainer(handle: dict, budget_note: bool = False) -> None:
    """Stream log lines, detect new snapshots, and push at the sync interval."""
    seed, log_path = handle["seed"], handle["log_path"]
    try:
        with open(log_path, "r", encoding="utf-8") as rf:
            rf.seek(handle["tail_offset"])
            chunk = rf.read()
        if chunk:
            handle["tail_offset"] += len(chunk.encode("utf-8"))
            for line in chunk.splitlines():
                print(f"[train seed {seed}] {line}", flush=True)
    except OSError:
        pass
    now = time.monotonic()
    ck = seed_ckpt_dir(seed)
    fresh = {p.name for p in ck.glob("*.pt")} if ck.exists() else set()
    new_snapshots = sorted(fresh - handle["known"])
    if new_snapshots:
        handle["known"] |= fresh
        log(f"new checkpoints: {', '.join(new_snapshots)}")
    if (new_snapshots and now - handle["last_push"] >= PUSH_INTERVAL_S) \
            or now - handle["last_push"] >= PUSH_INTERVAL_S:
        push_sync(f"seed {seed} checkpoint")
        handle["last_push"] = now


def wait_trainers(handles: list, deadline: float) -> dict:
    """Wait for all trainers, streaming logs + pushing; budget-kills at deadline."""
    while True:
        alive = [h for h in handles if h["proc"].poll() is None]
        if not alive:
            break
        if time.monotonic() >= deadline:
            log("wall-clock budget reached; terminating trainers")
            for h in alive:
                h["proc"].terminate()
            try:
                for h in alive:
                    h["proc"].wait(timeout=60)
            except subprocess.TimeoutExpired:
                for h in alive:
                    h["proc"].kill()
            break
        time.sleep(20)
        for h in handles:
            _pump_trainer(h)
    for h in handles:
        h["log_file"].close()
    return {h["seed"]: h["proc"].returncode for h in handles}


def run_trainer(seed: int, resume: bool, deadline: float, workers: int | None = None,
                gpu: int | None = None) -> bool:
    """Run one seed; returns True if it completed, False if budget-killed."""
    workers = workers or detect_workers(parallel=False)
    handle = _start_trainer(seed, resume, workers, gpu)
    rcs = wait_trainers([handle], deadline)
    rc = rcs.get(seed, 1)
    log(f"seed {seed} exited rc={rc}")
    push_sync(f"seed {seed} finished rc={rc}")
    return rc == 0


def run_seeds_parallel(seeds: list, deadline: float) -> bool:
    """Run independent seeds concurrently, one process per configured GPU.

    Each trainer receives its own ``CUDA_VISIBLE_DEVICES`` value; this is
    process-level single-GPU concurrency, not DDP.  Refuse to reuse a GPU so a
    two-arm plan cannot silently oversubscribe one accelerator.
    """
    n_gpus = max(1, _gpu_count())
    gpu_ids = [gpu for gpu in GPU_IDS if 0 <= gpu < n_gpus]
    if len(gpu_ids) < len(seeds):
        raise RuntimeError(
            f"parallel arms require one GPU each: seeds={len(seeds)}, "
            f"configured usable GPUs={gpu_ids}"
        )
    handles = [_start_trainer(seed, seed_has_work(seed), detect_workers(parallel=True),
                              gpu_ids[i]) for i, seed in enumerate(seeds)]
    rcs = wait_trainers(handles, deadline)
    ok = all(rcs.get(seed, 1) == 0 for seed in seeds)
    for seed in seeds:
        log(f"seed {seed} exited rc={rcs.get(seed)}")
        push_sync(f"seed {seed} finished rc={rcs.get(seed)}")
    return ok


# --------------------------------------------------------------------------
# Cross-set evaluation (after seed 0, before escalating to seeds 1+2)
# --------------------------------------------------------------------------

def run_cross_set(seed: int) -> dict | None:
    """Evaluate the trained seed on unseen FF++ methods + Celeb-DF (if present).

    Writes ``outputs/seed<seed>/eval_cross_set.json``; returns the parsed
    report or None when the evaluation could not run.
    """
    out = seed_root(seed) / "eval_cross_set.json"
    cmd = [sys.executable, "-m", "rlroinet.eval_cross_set",
           "--checkpoint", str(seed_ckpt_dir(seed) / "best.pt"),
           "--ffpp-dir", str(FFPP_DIR),
           "--amp-dtype", detect_amp(),
           "--device", "cuda",
           "--out", str(out)]
    celebdf = WORK / "data" / "celebdf"
    if celebdf.exists():
        cmd += ["--celebdf-dir", str(celebdf)]
    if CROSS_QUICK > 0:
        cmd += ["--quick", str(CROSS_QUICK)]
    log(f"cross-set eval: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=str(WORK), env=env, capture_output=True, text=True, timeout=3600)
    for line in (proc.stdout or "").splitlines():
        print(f"[cross-set] {line}", flush=True)
    if proc.returncode != 0:
        log(f"cross-set eval failed rc={proc.returncode}: {(proc.stderr or '')[-500:]}")
        return None
    if not out.exists():
        log("cross-set eval exited 0 but produced no report; treating as unavailable")
        return None
    report = json.loads(out.read_text(encoding="utf-8"))
    log(f"CROSS-SET SUMMARY: {json.dumps(report.get('summary', report), default=str)}")
    return report


# --------------------------------------------------------------------------
# Aggregation for the full run
# --------------------------------------------------------------------------

AGG_METRICS = [
    "auc", "acc", "region_iou", "region_hit", "eer", "ece",
    "fp_rate", "fn_rate", "confident_false_positive_rate",
    "confident_false_negative_rate", "review_rate", "reliable_coverage",
]


def aggregate_seeds() -> dict:
    import statistics
    rows, summary = [], {}
    for seed in SEEDS:
        tm = json.loads(seed_test_metrics(seed).read_text(encoding="utf-8"))
        rows.append(tm)
    for key in AGG_METRICS:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            summary[key + "_mean"] = round(statistics.mean(vals), 4)
            summary[key + "_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    summary["n_seeds"] = len(rows)
    summary["seeds"] = SEEDS
    summary["verdict"] = VERDICT
    summary["epochs_per_seed"] = EPOCHS
    summary["amp_dtype"] = detect_amp()
    summary["gpu_count"] = _gpu_count()
    per_seed = {}
    for seed in SEEDS:
        tm = next((r for r in rows if r.get("seed") == seed), None) or {}
        entry = {"val_best_auc": tm.get("validation_best_auc"),
                 "test_auc": tm.get("auc"),
                 "region_iou": tm.get("region_iou"),
                 "conf_fp": tm.get("confident_false_positive_rate"),
                 "conf_fn": tm.get("confident_false_negative_rate")}
        xs = seed_root(seed) / "eval_cross_set.json"
        if xs.exists():
            try:
                entry["cross_set"] = json.loads(xs.read_text(encoding="utf-8")).get("summary", {})
            except Exception:
                entry["cross_set"] = None
        per_seed[str(seed)] = entry
    summary["per_seed"] = per_seed
    return summary


def write_results_markdown(summary: dict) -> Path:
    """Human-readable research summary written to ``WORK/outputs/RESULTS.md``."""
    def g(key, suffix):
        k = f"{key}_{suffix}"
        return summary.get(k)
    lines = ["# RL-ROI-Net temporal-region: run results", ""]
    lines.append(f"- seeds: {summary.get('seeds')} (n={summary.get('n_seeds')})")
    lines.append(f"- verdict backbone: {summary.get('verdict')} | epochs/seed: {summary.get('epochs_per_seed')} "
                 f"| amp: {summary.get('amp_dtype')} | GPUs: {summary.get('gpu_count')}")
    lines.append("")
    lines.append("## Primary metrics (mean +/- std across seeds)")
    lines.append("")
    lines.append("| metric | mean | std | acceptance bar |")
    lines.append("|--------|------|-----|----------------|")
    bars = {"auc": 0.85, "region_iou": 0.75, "region_hit": 0.90,
            "confident_false_positive_rate": 0.01, "confident_false_negative_rate": 0.01,
            "reliable_coverage": 0.50, "review_rate": 0.50}
    for key in AGG_METRICS:
        mean, std = g(key, "mean"), g(key, "std")
        if mean is None:
            continue
        if key in bars:
            if key in ("auc", "region_iou", "region_hit", "reliable_coverage"):
                meets = mean >= bars[key]
            else:
                meets = mean <= bars[key]
            bar = f"{bars[key]:.2f} {'PASS' if meets else 'FAIL'}"
        else:
            bar = "—"
        lines.append(f"| {key} | {mean:.4f} | {std:.4f} | {bar} |")
    lines.append("")
    lines.append("## Per-seed detail")
    lines.append("")
    for seed, entry in summary.get("per_seed", {}).items():
        lines.append(f"- seed {seed}: val best AUC={entry.get('val_best_auc')}, "
                     f"test AUC={entry.get('test_auc')}, regionIoU={entry.get('region_iou')}, "
                     f"confFP={entry.get('conf_fp')}, confFN={entry.get('conf_fn')}")
        if entry.get("cross_set"):
            cs = entry["cross_set"]
            lines.append(f"  - cross-set: FF++ unseen-method mean AUC={cs.get('per_method_mean_auc')}, "
                         f"Celeb-DF AUC={cs.get('celebdf_auc')} ({cs.get('celebdf_status')})")
    lines.append("")
    out = WORK / "outputs" / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out}")
    return out


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

def write_state(status: str, extra: dict | None = None) -> None:
    state = {"status": status, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra:
        state.update(extra)
    SYNC.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def stop_session(code: int, status: str, extra: dict | None = None) -> None:
    """Persist final state, push once more, then end the session (frees GPU)."""
    write_state(status, extra)
    push_sync(f"final state: {status}")
    log(f"FINAL STATE: {status}")
    log(f"{'DONE' if code == 0 else 'STOPPED'} — you may stop and close this notebook.")
    sys.exit(code)


def main() -> int:
    if not WORK.exists():
        raise SystemExit(f"work dir does not exist: {WORK} (are we on Kaggle?)")
    log(f"driver start: seeds={SEEDS} epochs={EPOCHS} gate_auc={GATE_AUC} "
        f"budget={BUDGET_MINUTES}min amp={detect_amp()} workers={detect_workers()} "
        f"gpus={_gpu_count()} early_stop={EARLY_STOP_PATIENCE}")
    pull_sync()

    action, seed = next_action()
    if action in ("fail_stop", "done_stop"):
        log(f"prior state says {action.split('_')[0]}; ending session without GPU work")
        stop_session(0 if action == "done_stop" else 1, action.split("_")[0])

    deadline = time.monotonic() + BUDGET_MINUTES * 60.0
    n_gpus = _gpu_count()
    usable_gpu_ids = [gpu for gpu in GPU_IDS if 0 <= gpu < max(1, n_gpus)]
    solo_gpu = 0 if n_gpus > 1 else None  # leave GPU 1 free for the parallel phase
    log(f"usable single-GPU arm ids: {usable_gpu_ids}")

    # ---- Seed 0: gate run (sequential, so its timing/metrics are clean) ----
    if not seed_done(SEEDS[0]):
        resume = seed_has_work(SEEDS[0])
        ok = run_trainer(SEEDS[0], resume, deadline, workers=detect_workers(False), gpu=solo_gpu)
        if not ok:
            stop_session(1, "resumable", {"stopped_at_seed": SEEDS[0]})
        if not seed_done(SEEDS[0]):
            log(f"seed {SEEDS[0]} trainer exited rc=0 but no test_metrics; treating as failure")
            stop_session(1, "failed", {"failed_seed": SEEDS[0], "reason": "no test_metrics"})
        passed, summary = evaluate_gate(SEEDS[0])
        summary["gate_threshold"] = GATE_AUC
        log(f"seed {SEEDS[0]} gate: passed={passed} {summary}")
        if not passed:
            stop_session(1, "failed", summary)
        write_state("gate_passed", summary)
    else:
        log(f"seed {SEEDS[0]} already complete; skipping gate run")

    # ---- Cross-set evaluation after seed 0, before spending GPU on 1+2 ----
    cross = run_cross_set(SEEDS[0])
    cross_summary = (cross or {}).get("summary", {})
    cross_auc = cross_summary.get("celebdf_auc")
    if CROSS_GATE_AUC > 0:
        if cross_auc is None:
            stop_session(1, "failed", {"reason": "cross-set Celeb-DF AUC unavailable",
                                       "cross_set": cross_summary})
        if float(cross_auc) < CROSS_GATE_AUC:
            stop_session(1, "failed", {"reason": "cross-set AUC below floor",
                                       "cross_set_floor": CROSS_GATE_AUC,
                                       "cross_set_auc": cross_auc})
    write_state("seed0_evaluated", {"cross_set": cross_summary})

    # ---- Seeds 1..N: parallel, one trainer per GPU ----
    remaining = [s for s in SEEDS if not seed_done(s)]
    if remaining:
        if len(remaining) == 1:
            ok = run_trainer(remaining[0], seed_has_work(remaining[0]), deadline,
                             workers=detect_workers(False), gpu=solo_gpu)
        elif len(usable_gpu_ids) >= len(remaining):
            ok = run_seeds_parallel(remaining, deadline)
        else:
            log("fewer usable GPUs than remaining arms; running arms sequentially")
            ok = True
            for remaining_seed in remaining:
                ok = run_trainer(remaining_seed, seed_has_work(remaining_seed), deadline,
                                 workers=detect_workers(False), gpu=solo_gpu) and ok
                if not ok:
                    break
        if not ok:
            stop_session(1, "resumable", {"stopped_at_seed": remaining})
        for s in remaining:
            if not seed_done(s):
                log(f"seed {s} trainer exited rc=0 but no test_metrics; treating as failure")
                stop_session(1, "failed", {"failed_seed": s, "reason": "no test_metrics"})

    summary = aggregate_seeds()
    (SYNC / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_markdown(summary)
    log(f"FULL RUN SUMMARY: {json.dumps(summary)}")
    stop_session(0, "done", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
