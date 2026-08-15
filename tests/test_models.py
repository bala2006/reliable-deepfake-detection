import pytest
import torch

from rlroinet.adapter_agent import SpatialForensicAdapter
from rlroinet.config import default_config
from rlroinet.models import Agent, FeatureExtractor, ClassifierHead, RegionHead, focal_loss


@pytest.fixture
def cfg():
    c = default_config()
    c.data.face_size = 96
    c.model.backbone = "custom"   # hermetic: no torchvision download in tests
    c.model.feat_dim = 64
    c.model.classifier_hidden = 32
    c.train.device = "cpu"
    return c


def test_backbone_forward(cfg):
    b = FeatureExtractor(feat_dim=cfg.model.feat_dim)
    x = torch.rand(2, 3, 96, 96)
    assert b(x).shape == (2, cfg.model.feat_dim)


def test_pretrained_backbone_forward_and_freeze():
    pytest.importorskip("torchvision")
    c = default_config()
    c.model.backbone = "efficientnet_b4"
    c.model.pretrained = False        # random init — no weight download in tests
    c.data.face_size = 96
    a = Agent(c)
    x = torch.rand(2, 3, 96, 96)
    out = a.forward(x)
    assert out["cls_logits"].shape == (2, 1)
    # decoder upsamples the 3x3 feature map: 3 -> 48 (2x per decoder stage)
    assert out["mask"].shape == (2, 1, 48, 48)
    # freezing blocks 0..freeze_blocks-1 leaves only later blocks + proj trainable
    n_train = sum(1 for p in a.backbone.parameters() if p.requires_grad)
    n_total = sum(1 for p in a.backbone.parameters())
    assert 0 < n_train < n_total


def test_pretrained_backbone_spatial_size(cfg):
    pytest.importorskip("torchvision")
    from rlroinet.models.backbone import PretrainedFeatureExtractor
    bb = PretrainedFeatureExtractor(name="efficientnet_b4", pretrained=False, feat_dim=64)
    assert bb.spatial_size(96) == 3
    assert bb.spatial_size(288) == 9


def test_classifier_forward(cfg):
    clf = ClassifierHead(feat_dim=cfg.model.feat_dim, hidden=32)
    feat = torch.rand(4, cfg.model.feat_dim)
    logits = clf(feat)
    assert logits.shape == (4, 1)
    prob = clf.probability(feat)
    assert prob.shape == (4,)
    assert torch.all((prob > 0.0) & (prob < 1.0))


def test_focal_loss_reduces():
    logits = torch.tensor([[3.0], [-3.0], [0.5]])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = focal_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_region_head_outputs(cfg):
    head = RegionHead(in_channels=64, out_channels=(32, 16), n_regions=4)
    feat = torch.rand(2, 64, 8, 8)
    mask, region = head(feat)
    # decoder upsamples 8 -> 16 -> 32 (2x per decoder stage)
    assert mask.shape == (2, 1, 32, 32)
    assert region.shape == (2, 4, 32, 32)
    assert torch.all((mask >= 0) & (mask <= 1))


def test_agent_forward_and_param_groups(cfg):
    a = Agent(cfg)
    x = torch.rand(2, 3, 96, 96)
    out = a.forward(x)
    assert out["cls_logits"].shape == (2, 1)
    groups = a.param_groups()
    assert len(groups) == 2
    assert groups[0]["lr"] == cfg.train.lr_backbone
    assert groups[1]["lr"] == cfg.train.lr_head


def test_spatial_forensic_adapter_preserves_feature_shape():
    adapter = SpatialForensicAdapter(channels=32, bottleneck=8)
    x = torch.rand(2, 32, 7, 7)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
