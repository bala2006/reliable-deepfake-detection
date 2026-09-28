# Reliable Deepfake Detection

A structured research project for building generalizable deepfake detectors using sparse expert routing and evidence-stability conditioning.

## Status

| Iteration | Architecture | Result | Worst AUC | Notes |
|-----------|-------------|--------|-----------|-------|
| i1 | Stable-RouteNet v4 | **FAIL** | 0.578 | Strong localization (0.938 pixel AUC), weak detection. Overfits: train 1.000 vs eval 0.578. 1 dead expert. |
| i2 | EvidenceNet v1 | pending | — | Existence aggregation replaces the convex average that diluted i1's evidence. Within-dataset **and** cross-dataset evaluation are both required. First run: full architecture (A0) on a short schedule, `EPOCHS = 8`. Design: `architecture/i2-EvidenceNet-v1.md`. |

## Project Structure

```
architecture/              Design specifications per iteration
  i1-Stable-RouteNet-v4.md
  i2-EvidenceNet-v1.md
  rules.md

notebook/                  Executable training notebooks
  i1-Stable-RouteNet-v4.ipynb
  i2-EvidenceNet-v1.ipynb
  rules.md

docs/                      Reference documents
  i1-Stable-RouteNet-v4.docx
  rules.md

record/                    Experiment results (Pass/Fail)
  i1-Stable-RouteNet-v4-Fail.md
  rules.md

temp/                      Build utilities
  _build_nb.py
  _build_nb_i2.py
```

## Dataset

`sekhar826/srn-v4b-data` (~5.6 GB, Kaggle ref `srn-v4b-data`) — a curated frames + landmarks
pack in three real pools (FF++ YouTube real, Celeb-DF Celeb-real, Celeb-DF YouTube-real; 200
videos each) plus the manipulated benchmark sets (DFDCP, Celeb-DF-v2 synthesis, six FF++
families). Discovery is by layout sniffing, so the notebook resolves wherever Kaggle mounts it.

## Iteration Convention

All artifacts for one experiment share the same identity `iN-Architecture-Name-vVersion`:

| Folder | File |
|--------|------|
| `architecture/` | `i1-Stable-RouteNet-v4.md` |
| `notebook/` | `i1-Stable-RouteNet-v4.ipynb` |
| `docs/` | `i1-Stable-RouteNet-v4.docx` |
| `record/` | `i1-Stable-RouteNet-v4-Fail.md` |
| `architecture/` | `i2-EvidenceNet-v1.md` |
| `notebook/` | `i2-EvidenceNet-v1.ipynb` |

New iterations create new `iN` files. Old iterations are never overwritten.

## Core Idea

Detectors overfit to unstable manipulation shortcuts. This project routes stable, reliable localized evidence through specialized sparse experts to improve generalization.

**Key components:**
- **Sparse expert routing** — 4 specialist experts + 1 shared expert + 1 frequency expert
- **Prototype-based router** — routes by forensic content, not domain
- **Trust scoring** — M (manipulation) x S (stability) x R (reliability)
- **Two-view training** — original + degraded views learn which evidence survives quality changes
- **5-phase curriculum** — gradual introduction of objectives

## i1 Failure Analysis

The i1 run (Stable-RouteNet v4, 5 epochs, 2x T4) failed the primary detection task:

- Train AUC reached 1.000 but worst evaluation AUC was only 0.578
- Localization was strong (pixel AUC 0.938, IoU 0.600)
- Routing was imbalanced — 1 expert received 51.8% of patches, 1 was dead
- Default threshold classified all clean examples as real

See `record/i1-Stable-RouteNet-v4-Fail.md` for detailed findings and next steps.

## i2 Direction

The routing hypothesis was tested and falsified — i1's router entropy was `1.606` against a
maximum of `ln 5 = 1.609`, i.e. uniform, and the "1 dead expert" was a guaranteed artefact of the
top-2 clamp rather than a learned outcome. i2 therefore keeps only what i1 proved works
(localization) and rebuilds the decision path around an **existence operator**: top-k / log-sum-exp
/ max statistics per score map instead of a convex average, which is what threw the localized
evidence away (fake–real mean score difference of `0.004`).

Evaluation is required on **both** axes: *within-dataset* (held-out videos from the training
pools, through the training pixel pipeline, plus the pack's own real manipulations) and
*cross-dataset* (DFDCP, Celeb-DF-v2, raw frames, the published protocol).

## Next Steps

1. Compare image-level pooling with localization pooling
2. Test whether SBI artifacts differ from real manipulation evidence
3. Validate a global threshold on a separate calibration split
4. Add routing-balance constraints or improve expert initialization
5. Ablate frequency expert, LoRA, trust weighting, and CRO loss
6. Evaluate on real manipulated videos separately from synthetic data
