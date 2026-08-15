"""One-partition FaceSwap inversion diagnostic; inference only."""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from rlroinet.config import Config, default_config
from rlroinet.data import load_index
from rlroinet.data.maskdata import _read_png
from rlroinet.evaluate import load_region_agent
from rlroinet.predict import aggregate_video_confidence
from rlroinet.verdict import load_verdict

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
SOURCE_NAMES = ("frozen_honi05", "production_head", "spatial_adapter")


def _amp(cfg: Config, device: str):
    enabled = bool(getattr(cfg.train, "amp", True)) and device.startswith("cuda") and torch.cuda.is_available()
    return torch.autocast("cuda", enabled=enabled, dtype=torch.bfloat16)


def _enable_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def _score_from_output(out: dict) -> torch.Tensor:
    if "cls_logits" in out:
        return torch.sigmoid(out["cls_logits"]).reshape(-1)
    return out["verdict_prob"].reshape(-1)


def _score_items(model, items: list[dict], cfg: Config, device: str,
                 source: str, batch_size: int) -> np.ndarray:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            faces = torch.stack([_read_png(Path(item["face"]), cfg.data.face_size)
                                 for item in batch]).to(device)
            with _amp(cfg, device):
                if source == "frozen_honi05":
                    probs = model.verdict(faces)
                else:
                    probs = _score_from_output(model(faces))
            values.extend(float(x) for x in probs.float().cpu().numpy())
    return np.asarray(values, dtype=np.float64)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def _family_metrics(items: list[dict], scores: np.ndarray, cfg: Config,
                    video_level: bool) -> dict:
    rows: dict[str, dict] = {}
    groups: dict[str, dict] = {}
    for item, score in zip(items, scores):
        video = str(item["video"])
        groups.setdefault(video, {"label": int(item["label"]), "method": video.split("/", 1)[0],
                                 "scores": []})["scores"].append(float(score))
    if video_level:
        observations = [(row["method"], row["label"],
                         aggregate_video_confidence(np.asarray(row["scores"]), cfg))
                        for row in groups.values()]
    else:
        observations = [(str(item["video"]).split("/", 1)[0], int(item["label"]), float(score))
                        for item, score in zip(items, scores)]
    for family in (*METHODS, "overall"):
        selected = observations if family == "overall" else [x for x in observations
                                                               if x[0] == family or x[1] == 0]
        labels = np.asarray([x[1] for x in selected], dtype=np.int64)
        values = np.asarray([x[2] for x in selected], dtype=np.float64)
        real = values[labels == 0]
        fake = values[labels == 1]
        rows[family] = {
            "n": int(len(values)), "n_real": int(len(real)), "n_fake": int(len(fake)),
            "auc": _auc(labels, values),
            "real_mean": float(real.mean()) if len(real) else float("nan"),
            "fake_mean": float(fake.mean()) if len(fake) else float("nan"),
        }
    return rows


def _confound_stats(items: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[tuple[int, int, float, float]]] = defaultdict(list)
    for item in items:
        path = Path(item["face"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        luma = float(gray.mean())
        method = str(item["video"]).split("/", 1)[0]
        label_name = "fake" if int(item["label"]) else "real"
        grouped[(method, label_name)].append((w, h, sharpness, luma))
    result = {}
    for (method, label_name), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        shapes = Counter(f"{int(w)}x{int(h)}" for w, h, _, _ in values)
        result[f"{method}/{label_name}"] = {
            "n": int(len(values)),
            "mode_crop_size": shapes.most_common(1)[0][0],
            "crop_size_counts": dict(shapes),
            "mean_width": float(arr[:, 0].mean()), "mean_height": float(arr[:, 1].mean()),
            "mean_variance_of_laplacian": float(arr[:, 2].mean()),
            "mean_luma": float(arr[:, 3].mean()),
        }
    return result


def _label_sanity(items: list[dict]) -> list[dict]:
    checked = []
    for item in [x for x in items if str(x["video"]).startswith("FaceSwap/")][:3]:
        parts = Path(item["face"]).parts
        provenance = "FaceSwap" in parts and "manipulated_sequences" in parts
        checked.append({
            "video": str(item["video"]), "frame": str(item["frame"]),
            "label": int(item["label"]), "expected_label": 1,
            "face_path_contains_FaceSwap_manipulated_sequences": bool(provenance),
            "label_matches_provenance": bool(int(item["label"]) == 1 and provenance),
        })
    return checked


def _verdict(metrics: dict) -> str:
    frozen = metrics["frozen_honi05"]["frame"]["FaceSwap"]["auc"]
    production = metrics["production_head"]["frame"]["FaceSwap"]["auc"]
    if frozen < 0.5 and production < 0.5:
        return "INHERITED FROM FROZEN BACKBONE"
    if frozen >= 0.5 and production < 0.5:
        return "INTRODUCED BY HEAD"
    return ("UNDETERMINED (frame-level frozen and production FaceSwap AUCs do not "
            "separate the failure direction decisively)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpp-dir", default="/app/data/FaceForensics++")
    ap.add_argument("--production", default="/app/outputs/checkpoints/best.pt")
    ap.add_argument("--adapter", default="/app/outputs/experiments/spatial_adapter_official_split_5epoch/best.pt")
    ap.add_argument("--out", default="/app/outputs/diagnostics/faceswap_inversion_diagnostic_retry.json")
    ap.add_argument("--per-sample", default="/app/outputs/diagnostics/faceswap_inversion_diagnostic_retry_per_sample.jsonl")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()
    started = time.perf_counter()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable in this container")
    _enable_tf32(device == "cuda")

    cfg = default_config()
    cfg.data.source = "ffpp"
    cfg.data.ffpp_dir = args.ffpp_dir
    cfg.data.methods = METHODS
    cfg.data.compression = "c23"
    cfg.data.sample_train = 2400
    cfg.data.sample_val = 600
    cfg.data.sample_test = 600
    cfg.data.manifest_frames_per_video = 8
    cfg.data.frames_per_video = 32
    cfg.data.leakage_safe_split = True
    cfg.data.face_size = 288
    cfg.resolve_paths()
    items = load_index(cfg, "test")
    n_videos = len({str(x["video"]) for x in items})
    counts = Counter((int(x["label"]), str(x["video"]).split("/", 1)[0]) for x in items)
    if len(items) != 1132 or n_videos != 142:
        raise AssertionError(f"official test partition mismatch: records={len(items)} videos={n_videos}")

    batch_size = 16 if device == "cuda" else 4
    source_models = {
        "frozen_honi05": (load_verdict("honi05", device=device), cfg),
        "production_head": (load_region_agent(args.production, cfg, device), None),
        "spatial_adapter": (load_region_agent(args.adapter, cfg, device), None),
    }
    scores: dict[str, np.ndarray] = {}
    per_sample_path = Path(args.per_sample)
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    confounds = _confound_stats(items)
    metrics: dict[str, dict] = {}
    try:
        for source in SOURCE_NAMES:
            model, source_cfg = source_models[source]
            score_cfg = source_cfg or model.cfg
            scores[source] = _score_items(model, items, score_cfg, device, source, batch_size)
            metrics[source] = {
                "pooling": {"name": "robust", "top_fraction": float(cfg.eval.top_fraction)},
                "frame": _family_metrics(items, scores[source], cfg, False),
                "video": _family_metrics(items, scores[source], cfg, True),
            }
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        source_models.clear()

    with per_sample_path.open("w", encoding="utf-8") as handle:
        for i, item in enumerate(items):
            handle.write(json.dumps({
                "video": str(item["video"]), "frame": str(item["frame"]),
                "label": int(item["label"]), "method": str(item["video"]).split("/", 1)[0],
                "face": str(item["face"]),
                "frozen_honi05": float(scores["frozen_honi05"][i]),
                "production_head": float(scores["production_head"][i]),
                "spatial_adapter": float(scores["spatial_adapter"][i]),
            }, separators=(",", ":")) + "\n")

    report = {
        "schema": "faceswap_inversion_diagnostic_retry_v1",
        "partition": {"source": "FaceForensics++", "compression": "c23",
                      "official_pair_leakage_safe": True, "records": len(items),
                      "videos": n_videos, "counts_by_label_method": {
                          f"{label_name}/{method}": int(n)
                          for (label, method), n in sorted(counts.items())
                          for label_name in ("fake" if label else "real",)
                      }},
        "sources": metrics,
        "confounds": confounds,
        "label_sanity": _label_sanity(items),
        "verdict": _verdict(metrics),
        "runtime": {"device": device, "seconds": round(time.perf_counter() - started, 3),
                    "torch_cuda_available": bool(torch.cuda.is_available())},
        "per_sample_dump": str(per_sample_path),
        "assumptions": [
            "FF++ manifest contract is sample_train=2400, sample_val=600, sample_test=600, manifest_frames_per_video=8.",
            "Face-level scores use aligned PNG crops and the model's native preprocessing.",
            "Video scores use rlroinet.predict.aggregate_video_confidence with one common robust pool (top_fraction=0.25) for all sources.",
            "Family AUC combines that family's fake records with all official-test real records.",
            "The frozen source is load_verdict('honi05') and does not execute a trained region/classifier head.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    print(f"PARTITION records={len(items)} videos={n_videos} device={device}")
    print("source scope family auc real_mean fake_mean n_real n_fake")
    for source in SOURCE_NAMES:
        for scope in ("frame", "video"):
            for family in (*METHODS, "overall"):
                row = metrics[source][scope][family]
                print(f"{source} {scope} {family} {row['auc']:.6f} {row['real_mean']:.6f} "
                      f"{row['fake_mean']:.6f} {row['n_real']} {row['n_fake']}")
    print("CONFOUND method/label n crop_mode mean_laplacian mean_luma")
    for key, row in confounds.items():
        print(f"{key} {row['n']} {row['mode_crop_size']} "
              f"{row['mean_variance_of_laplacian']:.3f} {row['mean_luma']:.3f}")
    print("LABEL_SANITY " + json.dumps(_label_sanity(items), separators=(",", ":")))
    print("VERDICT " + report["verdict"])
    print("REPORT " + str(out_path))
    print("PER_SAMPLE " + str(per_sample_path))
    print(f"RUNTIME device={device} seconds={report['runtime']['seconds']}")


if __name__ == "__main__":
    main()
