# Training and Evaluation Record

This is the measured history of the video-detector experiments. **BEST** is first because it is the deployed checkpoint. The other runs use different held-out splits and policies, so their scores are not a single directly comparable leaderboard. All values are proportions; higher is better except FP/FN/review rates. `—` means the deleted historical report did not preserve that metric.

## Production-readiness gate status

No numeric release-gate set is currently approved. Earlier proposals used conflicting values
(for example, AUC/IoU/REVIEW thresholds in this file and `next.md`); those proposals are
withdrawn pending approval. The measurements below are evidence, not pass/fail guarantees.
The prior capacity requirements of at least 1,000 test videos, at least 100 real test videos,
and two valid external datasets are unreachable with the data on disk: the FF++ official test
contains 142 videos, while valid external capacity is 605 source-video identities (Celeb-DF-v2
518 + DFDCP 87; DFDCP's 777 test records collapse to 87 source groups). Celeb-DF-v2 is also
contaminated for the honi05 backbone, so it cannot serve as an external test for the deployed path.
A revised gate set is pending approval.

| Evidence item | Current measured result | Status |
|---|---:|---:|---|
|---|---:|---|
| BEST report unit | 603 face samples; `unseen_videos` key has `video_level=false` | Not a video-level release result |
| BEST headline metrics | ACC 0.6965 / AUC 0.8104 / ECE 0.2171 / region IoU 0.5576 | Research evidence only |
| Official-pair production video test | 142 videos; honest retained production AUC 0.5592 | Research-only, capacity gate unreachable |
| External localization | Celeb-DF-v2 and DFDCP have zero mask files in this extraction | Not supported |
| Calibration | BEST ECE 0.2171 | Do not describe as still calibrated |

**Decision:** the current BEST checkpoint remains a local evidence-review prototype, not a
production-ready detector. Keep REVIEW and human assessment mandatory; do not use these results
for automated moderation, legal, safety, or identity decisions.

## BEST — deployed production checkpoint

**Checkpoint:** `outputs/checkpoints/best.pt` (epoch 16). The retained root `final.pt` is epoch 20, but is not deployed. **Distinct configuration:** FF++ c23; all four methods; 32 frames/video; train/test sample limits 1800/300; focal alpha 0.75; 20 epochs; batch 16; backbone/head learning rates 1e-4/3e-4.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected checkpoint | 603 FF++ face samples; metrics key `unseen_videos`, `video_level=false` | **0.6965** | **0.8104** | **0.7479** | **0.5914** | **0.0995** | **0.2040** | **0.5576** | **0.7409** |
| Epoch-20 endpoint (`final.pt`) | Held-out face samples / 603 | 0.7048 | 0.8055 | 0.7709 | 0.5814 | 0.0862 | 0.2090 | 0.5550 | 0.7542 |

**Lacks / not useful for:** The `0.2830` FaceSwap AUC came from `outputs/eval_generalization.json`, which is not the official-pair split and must not be treated as production model behavior. On the leakage-safe official-pair diagnostic (`outputs/diagnostics/faceswap_inversion_diagnostic_retry.json`), the production head has FaceSwap video AUC `0.5310` and overall video AUC `0.5615`; the honest retained official-pair comparison reports `0.5592` on 142 videos. The ECE `0.2171` and face-sample unit mean this is not safe for automatic enforcement without REVIEW and human assessment.

## Training 1 — balanced four-method candidate (`balanced_all4_v4`)

**Configuration that differs from BEST:** balanced FF++ sampling across all four fake methods and real origins; enhanced classifier. The raw checkpoint/report were intentionally removed, so exact epoch, batch size, learning rate, and split size are not reconstructed.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical held-out evaluation | Videos / 24 | 0.7917 | 0.8462 | 0.7222 | 1.0000 | — | — | 0.4895 | 0.7624 |

**Lacks / not useful for:** localization was weaker than the later diverse-video run, and 24 videos is too small to establish dependable unseen-video performance. It was not promoted.

## Training 2 — focal-alpha candidate (`balanced_all4_alpha50`)

**Configuration that differs from Training 1:** focal-loss alpha **0.50**. It retained the balanced all-four-method candidate setup. The raw checkpoint/report were intentionally removed.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical held-out evaluation | Videos / 24 | — | 0.8671 | 0.8000 | 0.6154 | — | — | — | — |

**Lacks / not useful for:** it achieved **FaceSwap recall 0.0000**. The aggregate AUC therefore does not make it useful for FaceSwap detection or broad deployment. It was not promoted.

## Training 3 — diverse-video alpha-0.60 candidate (`diverse_all4_alpha60`)

**Configuration that differs from earlier runs:** stratified FF++ targets 2400/600/600 for train/validation/test; 8 supervised manifest frames/source video and 32 inference frames/video; focal alpha **0.60**; video-level validation selection. It was selected at epoch 14 by validation video AUC 0.8139. Raw artifacts were intentionally removed.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical held-out evaluation | Videos / 152 | 0.7566 | 0.7961 | 0.7143 | 0.8553 | 0.1711 | 0.0724 | 0.6772 | 0.9309 |
| FaceSwap subset | Videos | — | 0.8130 | — | 0.8947 | — | — | 0.5806 | 0.8947 |

**Lacks / not useful for:** it had high abstention/review and unresolved calibration concerns despite strong localization. It was not promoted as the deployed model.

## Final — expanded-data continuation (`diverse_all4_alpha60_continuation`)

**Retained artifacts:** `outputs/checkpoints/diverse_all4_alpha60_continuation/` and `outputs/eval_diverse_all4_alpha60_continuation*.json`. **Configuration that differs from Training 3:** continued for 8 epochs from its epoch-14 checkpoint with a fresh optimizer, batch 64, and both learning rates 1e-4; 10,400-entry manifest (5,200 real/5,200 fake); 200 additional real plus 200 additional fake source videos were forced into training only; locked validation/test membership (153/152 videos). Selected at epoch 8 by validation video AUC 0.8226.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default robust policy | Locked unseen videos / 152 | 0.7368 | 0.8062 | 0.6837 | 0.8816 | 0.2039 | 0.0592 | 0.6625 | 0.9112 |
| FaceSwap subset | Videos | — | 0.8137 | — | 0.8947 | — | — | 0.6122 | 0.9211 |
| Calibrated policy | Locked unseen videos / 152 | 0.7368 | 0.8062 | 1.0000 reliable precision | — | 0.0000 confident FP | 0.1053 confident FN | 0.6625 | 0.9112 |

**Lacks / not useful for:** default-policy false positives rose to 0.2039. Calibration reduced confident false positives to zero, but confidently missed 10.53% of fakes and sent 64.47% of cases to REVIEW. It is not useful as a production replacement for BEST.

## One-epoch temporal-quality feasibility benchmark (`temporal_quality_fresh_1epoch`)

**Artifacts:** `outputs/experiments/temporal_quality_fresh_1epoch/`. This was a fresh head-only experiment, not a continuation and not a change to `outputs/checkpoints/best.pt`. **Design tested:** frozen per-frame verdict features, an 8-frame same-video sequence, learned quality-aware attention (luminance, contrast, and vertical-detail sharpness inputs), and a lightweight depthwise temporal convolution. It is a classification-only feasibility head; the existing mask/ROI localization head was deliberately not trained in this run.

**Configuration:** 10,400-entry balanced FF++ manifest; one deterministic 8-frame sequence per complete split video; 993 train, 152 validation, and 152 locked test sequences; one epoch; batch 8 videos; temporal hidden width 128; quality hidden width 16; fresh optimizer; LR 1e-4; focal alpha 0.55; no train-time augmentation in this feasibility epoch. The training pass took **429.91 seconds**.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit | REVIEW rate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| Validation after epoch 1 | Videos / 152 | 0.7434 | 0.8182 | 0.6947 | 0.8684 | 0.1908 | 0.0658 | Not trained | Not trained | 0.9342 |
| Locked test after epoch 1 | Videos / 152 | 0.7171 | 0.8058 | 0.6813 | 0.8158 | 0.1908 | 0.0921 | Not trained | Not trained | 0.9408 |

**Lacks / not useful for:** one epoch is only a feasibility benchmark, not a trained production candidate. It fails the production gates: FP/FN rates are 0.1908/0.0921, ECE is 0.1208, and only 0.0592 of videos fall outside REVIEW. It also has no localization score because its ROI/mask head was not trained. On the same 152-video test protocol, its AUC (0.8058) did not exceed the retained continuation (0.8062). Do not promote or deploy this checkpoint; the next candidate must train the temporal and ROI evidence paths jointly, add train-only augmentation, and then be evaluated externally.

## Temporal-quality continuation — five total epochs (`temporal_quality_5epoch_continuation`)

**Artifacts:** `outputs/experiments/temporal_quality_5epoch_continuation/`. This experiment continued the preceding temporal feasibility head from its epoch-1 `best.pt` through epochs 2–5. It retained the same frozen `honi05` verdict model, 8-frame same-video sequences, quality-aware attention, depthwise temporal mixing, 10,400-entry balanced FF++ manifest, 993/152/152 train/validation/locked-test sequences, batch 8, LR 1e-4, focal alpha 0.55, and no augmentation. As the epoch-1 artifact has no optimizer state, this continuation deliberately used a **fresh AdamW optimizer**. Both new checkpoints are `temporal-quality-v1`, have epoch 5, and record `resume: true`; production `outputs/checkpoints/best.pt` remains the separate epoch-16 `region-head-v2` checkpoint.

| Evaluation | Test unit / n | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit | ECE | REVIEW rate | Reliable coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| Best validation epoch (epoch 5) | Videos / 152 | 0.7763 | 0.8358 | 0.7188 | 0.9079 | 0.1776 | 0.0461 | Not trained | Not trained | 0.0867 | 0.5987 | 0.4013 |
| Locked test after epoch 5 | Videos / 152 | 0.7237 | 0.8148 | 0.6977 | 0.7895 | 0.1711 | 0.1053 | Not trained | Not trained | 0.0925 | 0.5987 | 0.4013 |

**Lacks / not useful for:** the locked-test AUC improved over the one-epoch temporal run (0.8058) and the retained region-head continuation (0.8062), but it still fails every applicable production gate: FP/FN rates are 0.1711/0.1053, ECE is 0.0925, and only 0.4013 of videos receive a non-REVIEW decision. Its strict policy still has confident false positives of 0.0395. Localization is **not trained**, so this is not a replacement for an evidence-localizing review model. With no external-dataset evidence and the fresh-optimizer caveat, it is **not promoted** and must not replace `outputs/checkpoints/best.pt`.

## Evidence and retention notes

`outputs/verdict_shootout.json` records an honi05 result on 200 videos (ACC 0.9450, AUC 0.9903) without dataset, split, disjointness, or preprocessing provenance, so it is **unattributable and not citable**. `outputs/eval_report.json` records an FF++ score (600 samples; AUC 0.9934, ACC 0.9667, region IoU 0.9302) without a checkpoint ID or configuration and is likewise **unattributable and not citable**. Both artifacts remain historical records; neither is evidence for the deployed model. Deleted `balanced_all4_v1`–`v3`, `enhanced_validation`, and the invalid `cross_method_faceswap` line have no safely attributable retained result/configuration. REVIEW remains mandatory human escalation; none of these evaluations establish universal deepfake-detection reliability.


## Joint temporal-region fresh FF++ run — retry (`temporal_region_joint_fresh_5epoch_retry`)

**Artifacts:** `outputs/experiments/temporal_region_joint_fresh_5epoch_retry/`; selected `run/best.pt`; endpoint `run/final.pt`; policy `decision_policy.json`; locked-test report `test_metrics.json`. A previous attempt is preserved separately at `outputs/experiments/temporal_region_joint_fresh_5epoch/` with only its epoch-1 `best.pt` and metrics; it is not used for the final result.

**Configuration:** FF++ c23 only; balanced 10,400-entry manifest; one complete 8-frame sequence per video; 993 train, 152 validation, and 152 locked-test videos; frozen `honi05`; jointly trained temporal classifier, frame classifier, mask decoder, and four-ROI head; sequence-consistent contrast/brightness, blur, and noise augmentation; 5 epochs; GPU micro-batch 2; gradient accumulation 2; effective batch 4; four CPU workers; prefetch factor 2; pinned memory; CUDA AMP; head LR 1e-4; focal alpha 0.55; validation video AUC checkpoint selection. Docker training uses a 2 GB shared-memory allocation to support the worker prefetch queue.

| Validation epoch | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit | ECE | REVIEW rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7303 | 0.8077 | 0.6636 | 0.9342 | 0.2368 | 0.0329 | 0.7981 | 1.0000 | 0.1119 | 0.9013 |
| 2 | 0.7237 | 0.8211 | 0.7656 | 0.6447 | 0.0987 | 0.1776 | 0.6219 | 0.9046 | 0.1168 | 0.7368 |
| 3 **selected** | 0.7566 | **0.8465** | 0.6970 | 0.9079 | 0.1974 | 0.0461 | 0.8109 | 1.0000 | 0.0909 | 0.6776 |
| 4 | 0.7697 | 0.8458 | 0.7303 | 0.8553 | 0.1579 | 0.0724 | 0.8142 | 1.0000 | 0.0722 | 0.5066 |
| 5 endpoint | 0.7237 | 0.8416 | 0.7833 | 0.6184 | 0.0855 | 0.1908 | 0.6722 | 0.9441 | 0.1378 | 0.6579 |

### Locked-test result for the validation-selected checkpoint

| Policy | Test videos | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit | ECE | REVIEW rate | Reliable coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 152 | 0.7368 | 0.8345 | 0.6800 | 0.8947 | 0.2105 | 0.0526 | 0.7966 | 1.0000 | 0.0885 | 0.7237 | 0.2763 |
| Validation-fitted calibrated policy | 152 | 0.7368 | 0.8345 | 0.6800 | 0.8947 | 0.2105 | 0.0526 | 0.7966 | 1.0000 | 0.0718 | 0.7500 | 0.2500 |

Per-ROI IoU on the locked test: **PERIOCULAR 0.9361**, **JAWLINE 0.8674**, **MOUTH 0.9410**, **HAIRLINE 0.5984**. The selected checkpoint is epoch 3 by validation video AUC `0.8465`; temperature fitting used validation only and produced `T=0.4874766`, with real threshold `0.072818` and fake threshold `0.834611`. On the locked test, the calibrated policy had confident FP `0.0132` and confident FN `0.0000`, while sending `75.00%` of cases to REVIEW.

**Decision:** **not promoted** and not wired into runtime/API inference. The joint model improves and trains localization successfully, but the locked-test default FP rate (`0.2105`), ECE (`0.0885`), precision (`0.6800`), and only `0.2763` reliable coverage fail the conservative release gates. The calibrated policy still has only `0.2500` coverage and confident FP `0.0132`; FF++-only evidence cannot establish external or unseen-method reliability. Production `outputs/checkpoints/best.pt` remains unchanged.

## Gated temporal-region precision iteration — alpha-0.45 (`temporal_region_gated_alpha45_2epoch`)

**Artifacts:** `outputs/experiments/temporal_region_gated_alpha45_2epoch/`; selected `run/best.pt`; endpoint `run/final.pt`; policy `decision_policy.json`; locked-test report `test_metrics.json`. This fresh two-epoch candidate used the same 993/152/152 video-disjoint FF++ c23 protocol and frozen `honi05` backbone as the preceding joint run. It added a small quality/verdict-conditioned gated residual temporal fusion head, explicitly enabled CUDA TF32 paths, and lowered focal alpha from `0.55` to `0.45` to reduce fake-class loss pressure. GPU micro-batch 2, accumulation 2, four workers, prefetch 2, AMP, sequence-consistent augmentation, and 2 GB Docker shared memory were retained. The checkpoint format is `temporal-region-v2` and is not runtime/API-compatible with the production checkpoint.

| Evaluation | Test videos | ACC | AUC | Precision | Recall | FP rate | FN rate | Region IoU | Region hit | ECE | REVIEW rate | Reliable coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation-selected epoch 2, default | 152 validation | 0.7039 | 0.8438 | 0.6867 | 0.7500 | 0.1711 | 0.1250 | 0.8015 | 1.0000 | 0.1711 | 0.7237 | 0.2763 |
| Locked test, default | 152 | 0.7566 | 0.8283 | 0.7532 | 0.7632 | 0.1250 | 0.1184 | 0.8052 | 1.0000 | 0.1316 | 0.7566 | 0.2434 |
| Locked test, validation-calibrated REVIEW policy | 152 | 0.7566 | 0.8283 | 0.7532 | 0.7632 | 0.1250 | 0.1184 | 0.8052 | 1.0000 | 0.0754 | 0.7105 | 0.2895 |

Compared with the preceding selected `temporal_region_joint_fresh_5epoch_retry` checkpoint on the identical locked split, default precision improved `0.6800 → 0.7532` and FP rate improved `0.2105 → 0.1250`, while recall declined `0.8947 → 0.7632`, FN rate worsened `0.0526 → 0.1184`, and AUC declined `0.8345 → 0.8283`. Localization was retained/slightly improved (`0.7966 → 0.8052` IoU; hit remained `1.0000`), but default ECE worsened `0.0885 → 0.1316`. The calibrated policy had confident FP `0.0526`, confident FN `0.0000`, and coverage `0.2895`; this does not meet the documented release gates.

**Decision:** retain for comparison only; **do not promote** and do not connect it to API/runtime inference. The precision gain is accompanied by an unacceptable recall/FN regression, weaker AUC, poor calibration, and low reliable coverage. Production `outputs/checkpoints/best.pt` remains unchanged. FF++-only evidence still does not establish external or unseen-method reliability.

## Leakage-safe official-pair spatial adapter benchmark — `spatial_adapter_official_split_5epoch`

**Artifacts:** `outputs/experiments/spatial_adapter_official_split_5epoch/`; candidate checkpoint `best.pt`; standard evaluator report `official_generalization.json`; production comparison report `best_official_eval.json`. The production checkpoint `outputs/checkpoints/best.pt` was not modified.

**Protocol:** FF++ c23 official `train.json`/`val.json`/`test.json` source-pair assignments; all configured manipulation variants and associated originals kept in one partition; unmatched manifest records assigned to training only; four methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures); 2,400/600/600 train/validation/test frame-record samples; 8 supervised frames per source video; video-level test aggregation over 142 videos; frozen `honi05`; five epochs; 64-channel spatial adapter; existing mask/four-ROI supervision retained. Cross-method pair mismatches and fake-to-original source mismatches were both zero. For the apples-to-apples comparison below, both checkpoints were rerun under one shared policy: robust aggregation, REAL threshold `0.30`, FAKE threshold `0.70`, REVIEW enabled, and minimum fake fraction `0.10`.

| Checkpoint | ACC | AUC | Precision | Recall | FP rate | FN rate | ECE | Region IoU | Region hit | REVIEW rate | Reliable coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Retained production `best.pt` | 0.5141 | 0.5592 | 0.7600 | 0.4000 | 0.0845 | 0.4014 | 0.3106 | 0.1906 | 0.4405 | 0.3028 | 0.6972 |
| Official-split adapter `best.pt` | **0.6408** | **0.6273** | 0.7200 | **0.7579** | 0.1972 | **0.1620** | **0.0939** | **0.6712** | **0.9431** | 0.7817 | 0.2183 |

Under this shared policy and exact official-pair test split, the adapter improves ACC, AUC, recall, ECE, localization, and REVIEW-aware false-negative behavior. The retained production checkpoint has lower FP rate, higher nominal precision, and substantially higher reliable coverage. Per-method adapter AUC was Deepfakes `0.7181`, Face2Face `0.5683`, FaceSwap `0.6364`, and NeuralTextures `0.5869`.

**Decision:** the adapter is a measured improvement over `best.pt` under this more trustworthy same-protocol comparison, but it is **not promoted**. It fails the conservative production gates: only 142 test videos, no external-dataset evidence, FP `0.1972`, FN `0.1620`, ECE `0.0939`, and reliable coverage `0.2183` (REVIEW `0.7817`). The standard evaluator reproduced the in-process training result exactly, and the common-policy rerun confirmed the relative improvement. Keep `outputs/checkpoints/best.pt` as the production checkpoint and retain REVIEW for human assessment.
## DFDCP external robustness continuation

DFDCP is now supported as a manifest-backed supervised source at `data/DFDCP`. The loader uses aligned extracted face frames, the official train/test tags, source-grouped train/validation/test selection, balanced class budgets, and conservative handling of the two source groups that cross the supplied partition tags. There are no verified DFDCP forgery masks in the extraction, so localization supervision is synthetic and is not valid localization evidence.

The bounded continuation uses the frozen `honi05` backbone and trains only the classifier/region heads from the retained checkpoint into a separate experiment directory. It is intended to measure whether DFDCP improves external robustness; it does not alter or promote the production checkpoint automatically. DFDCP test results remain research evidence until combined with the locked FF++ protocol, raw Celeb-DF-v2, a second external dataset, larger video-level coverage, and release-gate review.
