import json

import numpy as np
import pytest
import torch

from rlroinet.config import default_config
from rlroinet.predict import aggregate_video_confidence
from rlroinet.snapshot import SnapshotScheduler
from rlroinet.train import configure_reproducibility, write_run_config


def test_robust_pool_matches_production_contract():
    cfg = default_config()
    cfg.eval.video_aggregation = "robust"
    cfg.eval.top_fraction = 0.25
    scores = np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    assert aggregate_video_confidence(scores, cfg) == pytest.approx(0.6375)


def test_zero_snapshot_interval_is_disabled(tmp_path):
    scheduler = SnapshotScheduler(tmp_path, interval_minutes=0)
    assert scheduler.disabled
    assert not scheduler.due()
    assert scheduler.maybe_save(lambda *_: {}, 1, {}) is None


def test_api_routes_adapter_checkpoint_to_region_loader(monkeypatch, tmp_path):
    import rlroinet.api as api

    checkpoint = tmp_path / "adapter.pt"
    torch.save({"checkpoint_format": "adapter-region-v1"}, checkpoint)
    sentinel = object()
    monkeypatch.setattr(api, "CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(api, "DEVICE", "cuda")
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api, "load_region_agent", lambda *args: sentinel)
    monkeypatch.setattr(api, "load_agent", lambda *args: pytest.fail("wrong loader"))
    monkeypatch.setattr(api, "_agent", None)

    assert api._get_agent() is sentinel


def test_deterministic_policy_seeds_cpu_and_disables_benchmarking():
    cfg = default_config()
    cfg.train.seed = 123
    cfg.train.deterministic = True
    configure_reproducibility(cfg)
    first = (random_value := torch.rand(3)).clone()
    configure_reproducibility(cfg)
    assert torch.equal(first, torch.rand(3))
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_run_config_contains_resolved_environment_and_manifest(tmp_path):
    cfg = default_config()
    cfg.data.source = "ffpp"
    cfg.data.ffpp_dir = str(tmp_path / "ffpp")
    cfg.resolve_paths()
    path = write_run_config(tmp_path / "run", cfg, trainer="test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cfg"]["data"]["ffpp_dir"] == cfg.data.ffpp_dir
    assert payload["seed"] == cfg.train.seed
    assert "torch_version" in payload
    assert "cuda_version" in payload
    assert payload["dataset_manifest"]["source"] == "ffpp"
    assert payload["trainer"] == "test"
