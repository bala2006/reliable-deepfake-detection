from copy import deepcopy

import pytest
import torch

from rlroinet.config import default_config
from rlroinet.evaluate import load_agent
from rlroinet.models import Agent
from rlroinet.predict import predict


def _write_checkpoint(path):
    cfg = default_config()
    cfg.model.backbone = "custom"
    cfg.model.pretrained = False
    cfg.data.face_size = 32
    cfg.train.amp = False
    agent = Agent(cfg)
    torch.save({
        "checkpoint_format": "supervised-v2",
        "model": agent.state_dict(),
        "cfg": cfg.to_dict(),
    }, path)
    return cfg, agent


def test_valid_checkpoint_round_trip_and_predict_compatibility(tmp_path):
    path = tmp_path / "valid.pt"
    cfg, original = _write_checkpoint(path)
    loaded = load_agent(path, cfg, device="cpu")
    assert loaded.cfg.schema_version == 2
    assert set(loaded.state_dict()) == set(original.state_dict())
    result = predict(loaded, torch.rand(2, 3, 32, 32), cfg, device="cpu")
    assert result["frames_per_video"] == 2
    assert result["compute_savings"] is None


@pytest.mark.parametrize("missing", ["model", "cfg"])
def test_malformed_checkpoint_is_rejected(tmp_path, missing):
    path = tmp_path / f"missing-{missing}.pt"
    cfg = default_config()
    payload = {"checkpoint_format": "supervised-v2", "cfg": cfg.to_dict()}
    payload["model"] = {}
    del payload[missing]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_agent(path, cfg, device="cpu")


def test_nested_checkpoint_configuration_is_rejected(tmp_path):
    path = tmp_path / "nested-invalid.pt"
    cfg = default_config()
    saved_cfg = deepcopy(cfg.to_dict())
    saved_cfg["model"]["unknown_field"] = True
    torch.save({"checkpoint_format": "supervised-v2", "model": {}, "cfg": saved_cfg}, path)
    with pytest.raises(ValueError, match="configuration fields in model"):
        load_agent(path, cfg, device="cpu")


def test_legacy_checkpoint_format_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"checkpoint_format": "legacy", "model": {}, "cfg": {}}, path)
    with pytest.raises(ValueError, match="not a supervised-v2 artifact"):
        load_agent(path, default_config(), device="cpu")


def test_incompatible_checkpoint_state_is_rejected(tmp_path):
    path = tmp_path / "incompatible.pt"
    cfg = default_config()
    cfg.model.backbone = "custom"
    cfg.model.pretrained = False
    torch.save({
        "checkpoint_format": "supervised-v2",
        "model": {"unexpected": torch.tensor(1)},
        "cfg": cfg.to_dict(),
    }, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_agent(path, cfg, device="cpu")
