# Project Status

Updated 2026-08-15. All metrics below come from completed RTX 4050 runs; no estimate is presented as a measurement.

## Current retained artifacts

| Purpose | Location | Status |
|---|---|---|
| Production inference | `outputs/checkpoints/best.pt` | Retained; used by the API and `start.bat` |
| API compatibility checkpoint | `outputs/checkpoints/final.pt` | Retained |
| Production reports | `outputs/eval_generalization.json` | Retained historical report; not official-pair evidence |
| Unattributable historical reports | `outputs/verdict_shootout.json`, `outputs/eval_report.json` | Preserved history; not citable and not evidence for the deployed model |
| Official-pair diagnostic | `outputs/diagnostics/faceswap_inversion_diagnostic_retry.json` | Retained; 1,132 records / 142 videos |
| Latest continuation | `outputs/checkpoints/diverse_all4_alpha60_continuation/` | Retained experiment; not promoted |
| Continuation reports | `outputs/eval_diverse_all4_alpha60_continuation*.json` | Retained |
| Verdict weights | `outputs/models/` | Retained |

## Production boundary

Two training paths exist. `train.py --verdict honi05` uses the frozen Celeb-DF-trained `honi05` EfficientNet-B4 verdict backbone with trainable classification and region heads; this is the deployed production path. `train.py` without `--verdict` uses the ImageNet EfficientNet-B4 path with its configured partial-unfreeze setting; it is an alternate training path, not the deployed checkpoint. Both paths expose classification and region heads. The decision policy returns REAL, FAKE, or REVIEW; REVIEW is mandatory human escalation, not a low-confidence binary verdict.

The retained BEST report is a **603-face-sample** result, not a video-level result: ACC `0.6965`, AUC `0.8104`, ECE `0.2171`, regionIoU `0.5576`, and regionHit `0.7409`, stored under the misleading `unseen_videos` key with `video_level=false`. The honest official-pair production result is video AUC `0.5592` on 142 videos.

On the leakage-safe official-pair diagnostic (`outputs/diagnostics/faceswap_inversion_diagnostic_retry.json`, 1,132 records / 142 videos), there is **no FaceSwap inversion or collapse**. Production-head video AUC is Deepfakes `0.5488`, Face2Face `0.5833`, FaceSwap `0.5310`, NeuralTextures `0.5816`, overall `0.5615`; frozen honi05 is `0.5377` overall (real mean `0.145` vs fake `0.188`), and the spatial adapter is `0.6274` overall. The cited `0.951` / `0.927` / `0.283` spread came from `outputs/eval_generalization.json`, which is not the official-pair split and must not be used as production behavior.

The latest continuation was measured on a separate locked 152-video FF++ split. It reached AUC `0.8062`, recall `0.8816`, regionIoU `0.6625`, and FaceSwap regionIoU `0.6122`, but higher false positives and a calibrated confident false-negative rate of `0.1053` prevented promotion. `outputs/checkpoints/best.pt` remains unchanged.

## Data capacity and release-gate limits

The FF++ official-pair test contains 1,132 frame records across 142 videos, including 47 real videos. Valid external capacity is 605 source-video identities: Celeb-DF-v2 contributes 518 and DFDCP contributes 87 because its 777 official test records collapse to 87 source groups under the loader. FF++ has masks for 100% of its 5,200 fake records; Celeb-DF-v2 and DFDCP have zero mask files, so no external localization claim is supportable. Celeb-DF-v2 is contaminated for the `honi05` backbone.

Therefore the prior `>=1000`-test-video and `>=2`-valid-external-dataset gates are unreachable with the data on disk. No numeric release-gate set is approved; a revised gate set is pending approval and no replacement thresholds are asserted.

## Local review service

`start.bat` starts Docker Compose with `outputs/checkpoints/best.pt`, waits for local CUDA/checkpoint readiness, and opens the browser UI. The UI supports local upload review, cancellation, explicit service status, report download, and immediate deletion of generated review media. Default limits: 30 seconds, 250 MiB, 24-hour automatic retention, one concurrent GPU analysis.

## Cleanup completed

Removed historical non-production checkpoints/reports, empty failed-run directories, Python/pytest/model-download caches, smoke files, obsolete planning files, and stale documentation references. Raw datasets, model weights, root production artifacts, and the complete latest continuation were preserved.

## Remaining evidence gap

Raw Celeb-DF is unavailable and the `honi05` backbone is contaminated for that source. DFDCP is available only as a classification-only robustness source because it has no mask files. Cross-dataset and genuine unseen-method performance for the deployed path therefore remain unestablished. Do not claim zero-error detection or production-wide reliability.


## Joint temporal-region run completed

The FF++-only fresh joint temporal-region retry completed five epochs at `outputs/experiments/temporal_region_joint_fresh_5epoch_retry/`. It used 993/152/152 video-disjoint train/validation/locked-test sequences, frozen `honi05`, 8-frame sequences, jointly trained classification plus mask/four-ROI localization, sequence-consistent augmentation, GPU micro-batch 2 with accumulation 2, four workers, prefetch 2, CUDA AMP, and a 2 GB Docker shared-memory allocation.

Validation selected epoch 3 by video AUC `0.8465`. Locked test: default ACC `0.7368`, AUC `0.8345`, precision `0.6800`, recall `0.8947`, FP `0.2105`, FN `0.0526`, ECE `0.0885`, region IoU `0.7966`, region hit `1.0000`, REVIEW `0.7237`, reliable coverage `0.2763`. Validation-fitted calibration yielded temperature `0.4874766`, REVIEW `0.7500`, reliable coverage `0.2500`, confident FP `0.0132`, and confident FN `0.0000` on the locked test. Per-ROI IoU: periocular `0.9361`, jawline `0.8674`, mouth `0.9410`, hairline `0.5984`.

The candidate is **not promoted** because classification precision/false-positive behavior, calibration, and reliable coverage do not establish production readiness while the numeric release-gate set is pending approval. It is not connected to API/runtime inference. Checkpoint validation confirmed `temporal-region-v1` epoch 3/5 artifacts, while production `outputs/checkpoints/best.pt` remains `region-head-v2` epoch 16. FF++-only results do not establish external or unseen-method reliability.

## Gated temporal-region precision iteration completed

A fresh two-epoch FF++-only candidate was trained at `outputs/experiments/temporal_region_gated_alpha45_2epoch/` with a quality/verdict-conditioned gated residual temporal fusion head, focal alpha `0.45`, explicit CUDA TF32 paths, frozen `honi05`, 8-frame sequences, sequence-consistent augmentation, micro-batch 2 with accumulation 2, four workers, prefetch 2, AMP, and 2 GB Docker shared memory. It writes the new `temporal-region-v2` checkpoint format.

On the same locked 152-video split as the preceding joint run, default metrics were ACC `0.7566`, AUC `0.8283`, precision `0.7532`, recall `0.7632`, FP `0.1250`, FN `0.1184`, ECE `0.1316`, regionIoU `0.8052`, regionHit `1.0000`, REVIEW `0.7566`, and reliable coverage `0.2434`. Relative to the prior `v1` checkpoint, precision and FP improved, but AUC and recall declined, FN and ECE worsened, and the calibrated policy had confident FP `0.0526` with only `0.2895` coverage. This is a measured precision/recall tradeoff, not a reliable production improvement; it was not promoted and `outputs/checkpoints/best.pt` remains unchanged.



## Spatial forensic adapter experiments — measured and not promoted

A parameter-efficient spatial adapter candidate was added without changing the production checkpoint or API. It keeps the `honi05` verdict backbone frozen, applies a 64-channel bottleneck residual adapter to the 1792-channel spatial feature map, and trains the existing mask/four-ROI decoder plus an average+maximum pooled classifier with the frozen verdict logit. The evaluator supports the isolated `adapter-region-v1` format.

On the same 152-video FF++ protocol used by the recent temporal candidates, the all-four-method run at `outputs/experiments/spatial_adapter_5epoch/best.pt` reached AUC `0.8102`, ACC `0.6513`, precision `0.5891`, recall `1.0000`, FP `0.3487`, FN `0.0000`, ECE `0.1704`, regionIoU `0.7296`, regionHit `0.9868`, and reliable coverage `0.5724`. Validation calibration reduced confident FP/FN to `0.0132`/`0.0263`, but reliable coverage fell to `0.2961`, so the candidate was not promoted; the numeric release-gate set remains pending approval.

A separate leave-FaceSwap-out run at `outputs/experiments/spatial_adapter_holdout_faceswap_5epoch/best.pt` trained on Deepfakes, Face2Face, and NeuralTextures only, then evaluated FaceSwap-only held-out videos. It reached AUC `0.7814`, ACC `0.7152`, precision `0.7286`, recall `0.6800`, FP `0.1258`, FN `0.1589`, ECE `0.1093`, regionIoU `0.4948`, and regionHit `0.9117`. Validation-fitted conservative thresholds produced zero confident FP/FN but only `0.1523` reliable coverage and `0.8477` REVIEW. These results are evidence that the adapter improves ranking relative to the retained production FaceSwap result (`AUC 0.2830`), but not enough for the required gates; the candidate remains research-only and is not wired into runtime.



## Leakage-safe official-pair adapter comparison completed

The official FF++ source-pair split is now implemented as an opt-in protocol and was used to train and evaluate `outputs/experiments/spatial_adapter_official_split_5epoch/best.pt`. It keeps official source pairs, all configured manipulation variants, and associated originals together; unmatched records remain training-only. The resulting train/validation/test frame-record counts were 8,180/1,088/1,132, with zero cross-method pair mismatches and zero fake-to-original source mismatches. The video-level test contained 142 videos across Deepfakes, Face2Face, FaceSwap, and NeuralTextures.

Using the same standard evaluator, frame sampling, leakage-safe test split, and one shared policy—robust aggregation, REAL threshold `0.30`, FAKE threshold `0.70`, REVIEW enabled, and minimum fake fraction `0.10`—the retained production checkpoint `outputs/checkpoints/best.pt` measured ACC `0.5141`, AUC `0.5592`, ECE `0.3106`, FP `0.0845`, FN `0.4014`, regionIoU `0.1906`, regionHit `0.4405`, REVIEW `0.3028`, and reliable coverage `0.6972`. The spatial adapter measured ACC `0.6408`, AUC `0.6273`, ECE `0.0939`, FP `0.1972`, FN `0.1620`, regionIoU `0.6712`, and regionHit `0.9431`; its REVIEW rate was `0.7817` with reliable coverage `0.2183`. The persisted adapter report reproduced the training-time result, and the common-policy rerun confirmed the relative improvement.

This is evidence that the adapter is better than the retained checkpoint under an exact apples-to-apples official-pair protocol, especially for ranking and localization, but it is **not production-ready**: the test set is only 142 videos, valid external capacity is limited to 605 source-video identities, FP/FN and calibration do not establish production readiness while the numeric release-gate set is pending approval, and most adapter cases are REVIEW. The adapter was not promoted, the API remains on `outputs/checkpoints/best.pt`, and the production checkpoint is unchanged.
## DFDCP integration started

DFDCP is available under `data/DFDCP` with 5,250 manifest video records, extracted aligned frames, and landmarks. A grouped loader now preserves the official train/test assignments, creates a validation holdout from training groups, balances class budgets, and excludes train-side duplicates for source groups that cross the supplied partition boundary. DFDCP has no verified pixel masks in this extraction, so its region supervision is synthetic and its localization metrics are not release evidence. The bounded head-only continuation is isolated from `outputs/checkpoints/best.pt`.
