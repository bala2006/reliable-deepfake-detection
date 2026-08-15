from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


DRIVER_PATH = Path(__file__).resolve().parents[1] / "tools" / "kaggle" / "kaggle_driver.py"


def _load_driver(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RLROINET_WORK", str(tmp_path / "working"))
    monkeypatch.setenv("KAGGLE_USERNAME", "resume-test-user")
    spec = importlib.util.spec_from_file_location("kaggle_driver_resume_test", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def _write_resume_outputs(driver) -> dict[str, bytes]:
    output = driver.WORK / "outputs"
    payloads = {
        "seed0/checkpoints/best.pt": b"best-model-state",
        "seed0/checkpoints/final.pt": b"final-model-state",
        "seed0/checkpoints/snapshot_epoch_3.pt": b"model optimizer scheduler epoch=3",
        "seed0/metrics.json": b"[{\"epoch\": 1}, {\"epoch\": 2}, {\"epoch\": 3}]",
        "seed0/test_metrics.json": b"{\"validation_best_auc\": 0.81}",
        "seed0/run_config.json": b"{\"seed\": 0, \"epoch\": 3}",
        "seed0/diagnostics/resume.json": b"{\"best_checkpoint\": \"best.pt\"}",
        "feature_cache/seed0/features.pt": b"cached-features",
    }
    for relative, content in payloads.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return payloads


def test_resume_stages_and_pull_rehydrates_trainer_paths(tmp_path, monkeypatch):
    driver = _load_driver(tmp_path, monkeypatch)
    payloads = _write_resume_outputs(driver)

    manifest = driver.stage_resume_artifacts()
    assert set(payloads) == set(manifest["files"])
    assert manifest["best_checkpoints"] == {
        "seed0": "seed0/checkpoints/best.pt"
    }
    for relative, content in payloads.items():
        assert (driver.SYNC / "outputs" / relative).read_bytes() == content

    remote = tmp_path / "remote_dataset"
    shutil.copytree(driver.SYNC, remote)
    shutil.rmtree(driver.WORK / "outputs")
    shutil.rmtree(driver.SYNC)

    def fake_download(args, **kwargs):
        assert args[1:3] == ["datasets", "download"]
        shutil.copytree(remote, driver.PULL, dirs_exist_ok=True)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(driver, "_kaggle_cli", lambda: "kaggle")
    monkeypatch.setattr(driver.subprocess, "run", fake_download)
    driver.pull_sync()

    for relative, content in payloads.items():
        assert (driver.WORK / "outputs" / relative).read_bytes() == content
    assert driver.verify_resume_artifacts()


def test_resume_integrity_check_fails_when_link_is_broken(tmp_path, monkeypatch):
    driver = _load_driver(tmp_path, monkeypatch)
    _write_resume_outputs(driver)
    driver.stage_resume_artifacts()

    (driver.WORK / "outputs" / "seed0" / "checkpoints" / "best.pt").unlink()

    assert not driver.verify_resume_artifacts()
    assert driver.RESUME_MANIFEST.exists()
    manifest = json.loads(driver.RESUME_MANIFEST.read_text(encoding="utf-8"))
    assert "seed0/checkpoints/best.pt" in manifest["files"]
