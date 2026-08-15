# ROI-Net v2 — Model & Head Architecture

The repository contains two model paths. The deployed `train.py --verdict honi05`
path uses the frozen `honi05` EfficientNet-B4 verdict backbone with trainable heads.
The `train.py` path without `--verdict` uses the configured ImageNet EfficientNet-B4
backbone with its configured frozen blocks. The diagram below describes that generic
`Agent` path; it must not be read as a description of the deployed checkpoint. Both
paths feed **two heads** — the binary classification head and the region (forgery-mask)
head. The four ROIs are fixed landmark-defined boxes and are localization evidence,
not causal attribution of how or where a manipulation was produced.

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        IN["Aligned face frame<br/>(B, 3, H, W)"]
    end

    subgraph BACKBONE["Backbone — PretrainedFeatureExtractor<br/>efficientnet_b4 (ImageNet), freeze_blocks=4"]
        FEAT["forward_features()<br/>EfficientNet-B4 features<br/>Feature map (B, 1792, H', W')"]
    end

    subgraph CLS_BRANCH["Classification Head — ClassifierHead"]
        GAP["AdaptiveAvgPool2d(1)<br/>→ (B, 1792)"]
        PROJ["Linear(1792 → 256)<br/>projected feature (B, 256)"]
        FC1["Linear(256 → 128)<br/>+ ReLU"]
        FC2["Linear(128 → 1)<br/>fake/real logit (B, 1)"]
        SIG["sigmoid()<br/>P(fake) ∈ [0, 1]"]
        LOSS1["Focal loss<br/>γ=2.0, α=0.75"]
    end

    subgraph REG_BRANCH["Region Head — RegionHead (thesis contribution)"]
        DEC1["Conv2d(1792 → 256) + BN + ReLU<br/>Upsample ×2 (bilinear)"]
        DEC2["Conv2d(256 → 128) + BN + ReLU<br/>Upsample ×2"]
        DEC3["Conv2d(128 → 64) + BN + ReLU<br/>Upsample ×2"]
        DEC4["Conv2d(64 → 32) + BN + ReLU<br/>Upsample ×2"]
        MASKH["Conv2d(32 → 1, 1×1)<br/>dense forgery logit map<br/>(B, 1, H', W')"]
        REGH["Conv2d(32 → 4, 1×1)<br/>per-ROI logits<br/>(B, 4, H', W')"]
        MASKP["sigmoid()<br/>forgery mask prob"]
        REGIONS["ROIs:<br/>PERIOCULAR · JAWLINE<br/>MOUTH · HAIRLINE"]
        LOSS2["Dice + BCE mask loss"]
        LOSS3["Per-ROI supervised loss"]
    end

    VERDICT["Verdict: FAKE / REAL"]
    MASK_OUT["Dense forgery mask<br/>(face-level, upsampled)"]
    ROI_OUT["Per-region forgery scores"]

    IN --> FEAT

    FEAT --> GAP --> PROJ --> FC1 --> FC2 --> SIG --> LOSS1
    SIG --> VERDICT

    FEAT --> DEC1 --> DEC2 --> DEC3 --> DEC4
    DEC4 --> MASKH --> MASKP --> LOSS2
    DEC4 --> REGH --> LOSS3
    REGIONS --> LOSS3
    MASKP --> MASK_OUT
    REGH --> ROI_OUT

    style INPUT fill:#2b2b2b,stroke:#888,color:#eee
    style BACKBONE fill:#1a2f4a,stroke:#4a90d9,color:#eee
    style CLS_BRANCH fill:#2d4a2d,stroke:#7ec87e,color:#eee
    style REG_BRANCH fill:#4a2d2d,stroke:#e07a7a,color:#eee
    style VERDICT fill:#3a3a1a,stroke:#d9c14a,color:#eee
    style MASK_OUT fill:#4a2d4a,stroke:#c97ec8,color:#eee
    style ROI_OUT fill:#4a2d4a,stroke:#c97ec8,color:#eee
    style LOSS1 fill:#5a5a5a,stroke:#aaa,color:#eee
    style LOSS2 fill:#5a5a5a,stroke:#aaa,color:#eee
    style LOSS3 fill:#5a5a5a,stroke:#aaa,color:#eee
```

## Tensor / module map

| Module | File | Purpose |
| --- | --- | --- |
| `PretrainedFeatureExtractor` | `models/backbone.py` | EfficientNet-B4 body, `forward_features()` gives `(B, 1792, 32, 32)` map |
| `ClassifierHead` | `models/classifier.py` | 2-layer MLP → binary fake/real logit (focal loss) |
| `RegionHead` | `models/region_head.py` | 4-step decoder → dense mask + 4 per-ROI logit maps |
| `Agent` | `models/agent.py` | Composes backbone + classifier + region head; separate LR groups |

## Flow at a glance

1. **Aligned face** (B, 3, H, W) → EfficientNet-B4 features → **feature map** (B, 1792, 32, 32).
2. **Classification path** (green): GAP → project to 256-dim → 2-layer MLP → sigmoid → frame verdict, trained with **focal loss** (down-weights easy frames).
3. **Region path** (red): decoder ladder `1792→256→128→64→32` with bilinear upsampling ×2 each step →
   - `mask_head` (1×1 conv) → **dense forgery mask**, supervised by DeepfakeBench masks via **Dice + BCE**;
   - `region_head` (1×1 conv) → **per-ROI logits** for PERIOCULAR / JAWLINE / MOUTH / HAIRLINE, supervised per-ROI against the GT mask restricted to each region.
4. Outputs: frame **verdict**, **dense mask**, and **per-region scores**.

> Mermaid flowchart renders in GitHub/GitLab/VSCode (Ctrl+Shift+V) and VS Code's built-in Mermaid preview.

## Temporal-region joint experiment head

The FF++ temporal-region experiments add a separate sequence path while retaining the frozen EfficientNet-B4/`honi05` feature extractor and the frame-level mask/ROI evidence path:

```mermaid
flowchart LR
    F["8 ordered aligned frames<br/>(B, T=8, 3, 288, 288)"] --> BB["Frozen honi05<br/>per-frame feature maps"]
    BB --> GAP2["GAP features<br/>(B, T, 1792)"]
    BB --> FRAME["Trainable frame classifier<br/>+ mask decoder + 4 ROI maps"]
    FRAME --> LOGIT["Frame fake/real logits"]
    GAP2 --> PROJ2["Temporal projection"]
    LOGIT --> PROJ2
    Q["Luminance / contrast / sharpness"] --> GATE["Quality/verdict-conditioned<br/>gated residual temporal fusion"]
    PROJ2 --> GATE
    GATE --> ATT["Learned attention pooling"]
    ATT --> VIDEO["Video fake/real logit"]
    FRAME --> MASK["Per-frame mask + ROI evidence"]
```

The current experimental temporal head uses depthwise temporal mixing, quality context, and a sigmoid residual gate conditioned on the projected frame representation, temporal candidate, and frozen-verdict/frame logit. It is jointly optimized with video focal loss, frame focal loss, dense mask Dice+BCE, and per-ROI BCE. The gated candidate is stored as `temporal-region-v2`; it is an experiment only and is not wired into the production API.

## DFDCP robustness-training path

DFDCP is an external robustness source, not a replacement for the FF++ mask-supervised source. Its manifest supplies official train/test assignments, labels, and source-video relationships; the extracted `frames/` trees supply aligned face crops. The loader groups all variants and their corresponding original by source identity, reserves validation groups from the official training pool, and keeps the official test groups untouched. Two source groups that crossed the supplied train/test tags were assigned to test and their train-side duplicates were excluded.

Because this extraction has landmarks but no verified forgery masks, fake DFDCP frames use the existing synthetic ROI mask fallback only to keep the region head trainable. DFDCP metrics therefore support classification robustness and head adaptation, but must not be reported as ground-truth localization evidence. The DFDCP continuation is written to an isolated experiment directory; `outputs/checkpoints/best.pt` remains the production artifact unless a later multi-dataset release review explicitly promotes a candidate.
