# i1 — Stable-RouteNet v4 — FAIL

**Decision:** FAIL for the main real/fake detection goal. Training AUC reached `1.000`, but final evaluation AUC was only `0.578` worst-case and `0.596` mean. Localization was strong, but the detector did not generalize enough.

## 1. Run metadata

| Field | Value |
|---|---|
| Record | i1 |
| Architecture | Stable-RouteNet v4 |
| Architecture file | `architecture/i1-Stable-RouteNet-v4.md` |
| Notebook | `notebook/i1-Stable-RouteNet-v4.ipynb` |
| Reference document | `docs/i1-Stable-RouteNet-v4.docx` |
| Data / configuration | Run-specific split not provided |
| Training date | Not provided |
| Epochs / steps | 5 / 900 |
| Stages | 1 → 2 → 3 → 4 → 5 |
| Final learning rate | `7.502e-06` |
| Time in `metrics.json` | `443.48 min` (`~7h 23m`) |
| Final progress time | `432.79 min` (`~7h 13m`) |
| Source | Supplied `metrics.json` and `progress.json` |

## 2. Goal

- Detect real versus manipulated faces on evaluation data.
- Remain useful under JPEG-30, blur-15, and resize-50.
- Localize manipulated regions.
- Keep expert routing balanced.
- Produce usable scores at a practical threshold.

## 3. Key results

### Detection progress

| Epoch | Train AUC checkpoint | Worst AUC | Mean AUC | Clean AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.643 | 0.500 | 0.502 | 0.501 |
| 2 | 0.869 | 0.517 | 0.523 | 0.524 |
| 3 | 0.994 | 0.537 | 0.548 | 0.553 |
| 4 | 0.998 | 0.558 | 0.571 | 0.577 |
| 5 | **1.000** | **0.578** | **0.596** | **0.602** |

### Final evaluation — epoch 5

| Condition | AUC | AP | EER | Calibrated accuracy | Calibrated F1 |
|---|---:|---:|---:|---:|---:|
| Clean | 0.602 | 0.614 | 0.406 | 0.600 | 0.591 |
| JPEG-30 | 0.578 | 0.582 | 0.444 | 0.578 | 0.568 |
| Blur-15 | **0.603** | **0.617** | 0.422 | **0.611** | 0.573 |
| Resize-50 | 0.599 | 0.610 | 0.411 | 0.589 | 0.560 |

### Threshold check

- Default threshold `0.5`: clean `TN=90, FP=0, FN=90, TP=0`; accuracy `0.500`, recall `0.000`, F1 `0.000`.
- Recorded clean calibrated threshold: `0.349`; accuracy `0.600`, balanced accuracy `0.600`, F1 `0.591`.
- The calibrated result is only a limited improvement and is not evidence that one deployment threshold has been validated.

### Localization

| Metric | Final value |
|---|---:|
| Pixel AUC | **0.938** |
| IoU | **0.600** |
| Mask F1 | **0.750** |

## 4. Findings

1. **Main detector: weak.** Final mean AUC `0.596` and worst AUC `0.578` are only slightly above chance.
2. **Generalization: poor.** Train AUC `1.000` versus worst evaluation AUC `0.578` indicates strong overfitting or a train/evaluation mismatch.
3. **Threshold behavior: failed.** At `0.5`, the model classified every recorded clean example as real.
4. **Localization: promising.** Pixel AUC `0.938`, IoU `0.600`, and mask F1 `0.750` are the strongest results.
5. **Robustness: insufficient.** JPEG, blur, and resize AUCs remained around `0.58–0.60`.
6. **Routing: imbalanced.** One expert was unused and one received `51.8%` of routing share.

## 5. Next steps

1. Compare the current image-level pooling with localization pooling and a simple baseline.
2. Test whether synthetic self-blending artifacts differ from real manipulation evidence.
3. Use a separate calibration split and validate one global threshold.
4. Add routing-balance constraints or improve expert initialization.
5. Run ablations for the frequency expert, LoRA, trust weighting, and contractive-repulsive loss.
6. Evaluate on real manipulated videos separately from the synthetic localization task.

## Technical appendix

| Detail | Final value |
|---|---:|
| Robust gap | `0.024568` |
| Clean ECE / Brier | `0.150795 / 0.267031` |
| Final total / detection / localization loss | `1.207213 / 0.212001 / 0.692409` |
| Mean manipulation score: fake / real | `0.333593 / 0.329566` |
| Fake-real score difference | `0.004027` |
| Routing shares | `[0.166472, 0.060850, 0.254592, 0.518086, 0.000000]` |
| Dead experts | `1` |
| Routing entropy fraction / share Gini | `0.998123 / 0.465848` |

**Source note:** Values are summarized from the `metrics.json` and `progress.json` data supplied for `notebook/i1-Stable-RouteNet-v4.ipynb`. Raw logs remain the authoritative source.
