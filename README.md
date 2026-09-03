# Reliable Deepfake Detection

Evidence-stability-conditioned sparse expert routing for generalizable deepfake detection.

## Current Status

**Iteration 1 (i1) — Stable-RouteNet v4: FAIL**

| Metric | Value |
|---|---|
| Train AUC | 1.000 |
| Worst eval AUC | 0.578 |
| Mean eval AUC | 0.596 |
| Clean eval AUC | 0.602 |
| Localization pixel AUC | 0.938 |

The model overfits strongly. Train AUC is perfect but evaluation AUC is near chance. Localization works well, but the primary detection task fails. Routing is imbalanced (1 dead expert).

See `record/i1-Stable-RouteNet-v4-Fail.md` for full analysis and next steps.

## Repository Layout

```
architecture/          Architecture specifications
  i1-Stable-RouteNet-v4.md

notebook/              Kaggle training notebooks
  i1-Stable-RouteNet-v4.ipynb

record/                Experiment results (Pass/Fail)
  i1-Stable-RouteNet-v4-Fail.md
  rules.md             Record format rules

_build_nb.py           Notebook builder script
Stable-RouteNet_v4_notebook.ipynb   Source notebook
Stable-RouteNet_v4.docx            Reference document
```

## What This Project Does

Builds a deepfake detector that generalizes to unseen datasets and manipulation methods by:

1. **Sparse expert routing** — specialist experts focus on different forensic evidence types
2. **Prototype-based routing** — routes by forensic content, not domain shortcuts
3. **Shared safety-net expert** — always processes all patches to prevent knowledge loss
4. **Trust scoring** — M (manipulation) x S (stability) x R (reliability) weights evidence
5. **Two-view training** — original + degraded views learn which evidence is stable

## Next Steps (from i1 failure)

1. Compare image-level pooling with localization pooling
2. Test whether synthetic SBI artifacts differ from real manipulation evidence
3. Use a separate calibration split and validate one global threshold
4. Add routing-balance constraints or improve expert initialization
5. Run ablations for frequency expert, LoRA, trust weighting, and CRO loss
6. Evaluate on real manipulated videos separately from synthetic localization

## Quick Start (Kaggle)

1. Upload `data/` as private dataset; attach to notebook (T4 x2, internet on)
2. Run all — preflight must print `ALL V3 PRE-FLIGHT SMOKE TESTS PASSED`
3. Training: 5 epochs, AdamW, 4 experts top-2 sparse, T=8
4. Final cell evaluates on DFDCP screening
