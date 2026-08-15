import numpy as np
import numpy as np
import torch

from rlroinet.config import default_config
from rlroinet.models import Agent
from rlroinet.predict import _face_crop, analyze, bbox_from_loc, predict


def _make_agent(cfg):
    return Agent(cfg).eval()


def _make_frames(cfg, t=4):
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, (t, 3, cfg.data.face_size, cfg.data.face_size))
    return torch.from_numpy(arr).float()


def _cfg():
    c = default_config()
    c.data.face_size = 96
    c.model.backbone = "custom"
    c.model.feat_dim = 64
    c.model.classifier_hidden = 32
    c.train.device = "cpu"
    c.eval.threshold = 0.5
    return c


def test_predict_schema():
    cfg = _cfg()
    agent = _make_agent(cfg)
    frames = _make_frames(cfg)
    result = predict(agent, frames, cfg, device="cpu")
    assert result["verdict"] in ("FAKE", "REAL", "REVIEW")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["frames_per_video"] == frames.shape[0]
    assert len(result["trace"]) == frames.shape[0]
    for t in result["trace"]:
        assert t["frame_index"] == t["step"] - 1
        assert t["roi"] in ("PERIOCULAR", "JAWLINE", "MOUTH", "HAIRLINE", "OTHER")


def test_predict_frames_per_frame_fields():
    from rlroinet.predict import predict_frames
    cfg = _cfg()
    agent = _make_agent(cfg)
    frames = _make_frames(cfg)
    out = predict_frames(agent, frames, cfg, device="cpu")
    assert len(out) == frames.shape[0]
    for a in out:
        assert a["mask"].shape == (cfg.data.face_size, cfg.data.face_size)
        assert "cx" in a and "cy" in a and "scale" in a
        assert "box_in_frame" in a and len(a["box_in_frame"]) == 4


def test_analyze_schema():
    cfg = _cfg()
    agent = _make_agent(cfg)
    frames = _make_frames(cfg)
    result = analyze(agent, frames, cfg, sample_idx=[0, 1, 2, 3], fps=30.0,
                     disp_w=128, disp_h=128)
    assert result["verdict"] in ("FAKE", "REAL", "REVIEW")
    assert "detections" in result and "trace" in result
    for d in result["trace"]:
        assert "bbox" in d and len(d["bbox"]) == 4
        assert "time_sec" in d and "video_frame" in d
    assert result["duration_sec"] is not None


def test_bbox_from_loc_bounds():
    x0, y0, x1, y1 = bbox_from_loc(0.5, 0.5, 0.3, 100, 100)
    assert x0 < x1 and y0 < y1
    assert x0 >= 0 and y0 >= 0 and x1 <= 100 and y1 <= 100



def test_face_crop_passes_bgr_to_detector(monkeypatch):
    import rlroinet.predict as predict_module

    frame_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    frame_rgb[..., 0] = 11
    frame_rgb[..., 1] = 22
    frame_rgb[..., 2] = 33
    seen = {}

    def fake_detect(frame_bgr):
        seen["frame"] = frame_bgr.copy()
        return (4, 5, 12, 10)

    monkeypatch.setattr(predict_module, "detect_face", fake_detect)
    face, _, box = _face_crop(frame_rgb, size=16)

    assert seen["frame"][0, 0].tolist() == [33, 22, 11]
    assert face.shape == (3, 16, 16)
    assert box == (3, 4, 17, 16)
