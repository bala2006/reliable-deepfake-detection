# Reliable Deepfake Detection

A structured research project for building generalizable deepfake detectors using sparse expert routing and evidence-stability conditioning.

## Status

| Iteration | Architecture | Result | Worst AUC | Notes |
|-----------|-------------|--------|-----------|-------|
| i1 | Stable-RouteNet v4 | **FAIL** | 0.578 | Strong localization (0.938 pixel AUC), weak detection. Overfits: train 1.000 vs eval 0.578. 1 dead expert. |

## Project Structure

```
architecture/              Design specifications per iteration
  i1-Stable-RouteNet-v4.md
  rules.md

notebook/                  Executable training notebooks
  i1-Stable-RouteNet-v4.ipynb
  rules.md

docs/                      Reference documents
  i1-Stable-RouteNet-v4.docx
  rules.md

record/                    Experiment results (Pass/Fail)
  i1-Stable-RouteNet-v4-Fail.md
  rules.md

temp/                      Build utilities
  _build_nb.py
```

## Iteration Convention

All artifacts for one experiment share the same identity `iN-Architecture-Name-vVersion`:

| Folder | File |
|--------|------|
| `architecture/` | `i1-Stable-RouteNet-v4.md` |
| `notebook/` | `i1-Stable-RouteNet-v4.ipynb` |
| `docs/` | `i1-Stable-RouteNet-v4.docx` |
| `record/` | `i1-Stable-RouteNet-v4-Fail.md` |

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

## Next Steps

1. Compare image-level pooling with localization pooling
2. Test whether SBI artifacts differ from real manipulation evidence
3. Validate a global threshold on a separate calibration split
4. Add routing-balance constraints or improve expert initialization
5. Ablate frequency expert, LoRA, trust weighting, and CRO loss
6. Evaluate on real manipulated videos separately from synthetic data
