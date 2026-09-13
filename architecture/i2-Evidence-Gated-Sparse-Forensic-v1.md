# i2 — Evidence-Gated Sparse Forensic Network v1

**Status:** implementation prepared; not executed or benchmarked in this chat. No Pass/Fail result is claimed.

**Notebook:** [i2-Evidence-Gated-Sparse-Forensic-v1.ipynb](../notebook/i2-Evidence-Gated-Sparse-Forensic-v1.ipynb).  
**Design source:** [specialArchitecture.md](../specialArchitecture.md), from `design/lightweight-generalization-architecture` at `4acbff9b03ed0f5fed2b9e1c89e0fb825be26965`. This is the conservative MoE design, not the separate LiteGuard proposal.

## Goal

Detect manipulated face crops using one lightweight backbone and a complete shared prediction path. Routed specialists add bounded corrections instead of controlling the entire decision. Evaluate held-out source identities, video ancestry, and quality conditions; do not mistake synthetic validation accuracy for unseen-generator generalization.

## Implemented graph

```text
Pre-extracted aligned RGB face crops, 224×224, up to 12 distinct frames
  ├─ EfficientNet-B0 features → [1280,7,7] → RGB projection, 192 channels
  ├─ fixed Sobel/Laplacian residuals + optional true 8×8 DCT statistics
  │    + chroma residuals → forensic projection, 32 channels
  └─ nearby-frame RGB feature differences → temporal projection, 32 channels
                 ↓
       256-channel normalized forensic adapter
                 ↓
       shared residual MLP + conservative top-1 MoE
                 ↓
       M / Q / U token heads; unweighted shared path is retained
                 ↓
       masked spatial evidence pooling + nearby-frame gated pooling
                 ↓
       shared video logit + bounded evidence correction
                 ↓
       held-out temperature calibration + real/fake/uncertain thresholds
```

The notebook is self-contained: configuration, manifest validation, synthetic training data, architecture, preflight tests, training, checkpoint selection, calibration, and separately enabled final testing are all included. It uses PyTorch, torchvision, NumPy, Pillow and scikit-learn, already present in the i1 code. No new framework or shell installation command is required. It records installed versions. The standard torchvision EfficientNet-B0 ImageNet weights must be cached or downloadable; it fails explicitly rather than silently training a random backbone.

### Features

EfficientNet-B0 replaces DINOv2-L. All input normalization occurs inside the model. RGB images and fixed forensic transforms use raw [0,1] pixels. Three fixed high-pass kernels use reflection padding. Optional orthonormal 8×8 DCT filters produce low-band energy, high-band energy and their ratio, excluding DC from low-band energy. Two chroma residual channels complete an 8-channel forensic descriptor. Frequency convolutions are vectorized, not Python loops over image tokens.

The adapter concatenates 192 RGB + 32 forensic + 32 temporal channels. Temporal differences only connect valid frames whose original frame indices differ by at most two. Widely separated evaluation samples are not falsely treated as consecutive video frames. This conservative adjacency condition may disable most temporal edges under uniform sampling; the notebook does not claim continuous-video temporal coverage.

### Conservative experts

Four 256→128→256 expert MLPs receive normalized features, never metadata. Top-1 routes always map to the intended expert. Per-frame capacity is `ceil(1.25 * tokens / 4)`. Overflow uses the shared path; selection is confidence-ranked separately inside each frame, so evaluation results do not depend on other videos in the batch. Expert dropout is 5% during training.

Expert feature corrections use `0.30 * sigmoid(gate) * tanh(expert_output)`, initially bounded by 0.15 per component. Selected router probabilities retain gradients. Balance loss uses detached hard assignment fractions and differentiable mean router probabilities. Requested utilization, accepted dispatches, overflow and entropy are recorded separately. The four semantic roles from the design are aspirations, not guaranteed learned specialization; no fabricated expert-role supervision is used.

The global classifier always sees the unweighted shared representation. The evidence path adds at most 0.5 in absolute video-logit units, using `0.5*tanh(local_logit)`. Both branches receive direct classification supervision. This explicitly bounds correction magnitude, unlike an unconstrained sigmoid coefficient multiplying unbounded logits.

### Evidence and uncertainty

M is supervised by approximate synthetic manipulation masks. Q learns agreement of M under a moderate paired degradation. U learns a detached token mask-error target where masks exist. **Q and U are learned proxies, not calibrated epistemic uncertainty or guarantees of correctness.** The video probability is calibrated separately.

Spatial weights use M, and in A6 also Q and U. Normalized weighted features are mixed equally with an unweighted mean. Temporal gates use only valid nearby neighbors; absent neighbors receive a neutral gate. All-invalid clips are rejected. Maps are returned in face-crop coordinates and are suspicious-region estimates, not full-video pixel-accurate explanations.

## Data contract and leakage controls

The supplied repository contains no trustworthy identity/ancestry manifest. The notebook therefore requires an explicit JSON manifest rather than invent identity-disjoint splits from filenames. Set `MANIFEST_PATH` (or the `I2_MANIFEST` environment variable).

The JSON root is a list. Each row requires:

| Field | Meaning |
|---|---|
| `split` | `train`, `dev`, `calibration`, or `test` |
| `dataset` | Dataset name, consistent across its real/fake rows |
| `video_id` | Unique ID within a dataset |
| `frames_dir` | Directory of chronologically named PNG/JPEG aligned face crops; relative paths resolve against the manifest |
| `label` | Integer 0 real or 1 fake |
| `identity_ids` | Nonempty list of globally canonical subject IDs; include source and target subjects |
| `source_ids` | Nonempty list of globally canonical original-video ancestry IDs; include all swap ancestors |
| `method` | Optional manipulation-family name; defaults to real/unknown |
| `masks_dir` | Optional per-frame PNG region masks with matching filename stems |

The validator rejects overlapping subject/ancestry IDs across splits, duplicate video IDs or reused crop directories, missing frames, missing masks, insufficient splits, and any training row labeled fake. It computes connected identity/ancestry groups for bootstrap resampling. Metadata must honestly describe identities and ancestry: it cannot infer identity or detect all visually near-duplicate crops. Manual/independent duplicate auditing is still necessary. Face detection/alignment and metadata preparation are prerequisites, not implemented by this crop-training notebook.

Training uses real FF++/Celeb-DF-style source videos and generates one real and one synthetic example per source. Domain-balanced sampling preserves class balance in expectation. The synthetic generator varies soft elliptical/subregion masks, blend strength, small source shifts, color and resampling. Its geometry is consistent across a clip. These are approximate region labels, not proof of exact changed pixels. It is a compact SBI-style augmentation, not a faithful reproduction of a published SBI implementation.

By default, `SYNTHETIC_DEV_CAL=True`: dev and calibration contain held-out real source rows expanded into real/synthetic pairs. This permits the current real-only research protocol, but thresholds calibrated on synthetic fakes must not be described as deployment-calibrated. Setting this to false requires actual labeled real/fake development and calibration rows; that is a distinct protocol. Test always uses actual labels and is never expanded into synthetic examples.

Dev, calibration and test are separate. The configured unseen dataset (`dfdcp` by default, case-insensitive) must occur only in test and contain both labels. Other test datasets may be partially seen domains; report them accordingly. Manifest labels are inspected for split validation but test images are not loaded by training/model selection.

## Training and loss

Defaults: A6 full design, 224×224, 12 frames, batch 2 videos, accumulation 2, five-epoch smoke experiment, AdamW, FP16 on CUDA, head LR 5e-4, backbone LR 1e-4, weight decay 1e-4, cosine schedule, 5% update warm-up, gradient clip 1.0. Backbone BatchNorm statistics remain frozen. Warm up the tail for one epoch, then unfreeze the last three EfficientNet feature blocks. All inference modules remain present from the beginning; curriculum ramps auxiliary losses, never switches to a different inference graph.

```
L = classification on shared and combined video logits, both views
  + 0.20 * mask BCE + Dice
  + 0.15 * nearby-frame temporal consistency
  + 0.10 * paired evidence-quality supervision
  + 0.02 * differentiable router load balance
  + 0.05 * (video Brier + supervised token-error proxy)
```

Temporal consistency excludes invalid/non-neighbor pairs and pairs with substantially different known mask coverage. Quality and temporal weights ramp after 20% of updates; balance ramps after 40%. Classification/localization anchor training from the start. Paired views add training compute but never add a second inference view. Train losses do not use dataset/method names as model inputs. Per-module additions follow A0–A6 from the design; disabled modules have no training objective and their parameters are frozen.

A0 is the required baseline. A6 is configured to make the full proposed design available, not because it is proven superior. Five epochs are a smoke experiment, not an accuracy promise. The output directory must be new; reruns do not silently overwrite previous experiments. Checkpoints contain model, optimizer, scaler, configuration, manifest hash and version information. Automatic resume is not implemented.

## Validation, calibration and final test

Selection computes the minimum development-dataset/quality score: `AUROC - 0.25*ECE - 0.25*FPR`, with FPR at the fixed diagnostic threshold 0.5. Conditions: clean, JPEG-30, blur sigma/radius 1.5 via Pillow, half-size down/up sampling, and quarter-size down/up sampling. This is a development heuristic, not a deployment objective. No test results participate.

Calibration uses only a separately held-out calibration split and the chosen checkpoint. It fits scalar temperature on video logits, then empirical clear/fake thresholds with declared FPR, false-clear-rate and minimum-fake-recall constraints. If no useful fake threshold satisfies the constraints, it emits an explicit **all-uncertain, unvalidated** policy rather than invent a threshold or claim precision. It records calibration scope and counts. Thresholds are fixed across final conditions. Small calibration sets cannot certify very low false-positive rates; no confidence-bound guarantee is claimed.

Final testing requires explicitly setting `RUN_FINAL_TEST=True`. It uses the reloaded checkpoint and calibration, records a final-test marker, and refuses an accidental rerun in the same output directory. Per-dataset and per-method (against that dataset's reals) reports include AUROC, AP, EER, accuracy, balanced accuracy, precision, recall, F1, confusion counts, ECE, Brier, selective risk, coverage and uncertain counts. Group-bootstrap AUROC intervals resample connected identity/source groups. Both classes are required for overall dataset evaluation. Classifier-only crop latency/throughput and peak CUDA allocated memory are recorded; face detection and video decoding are excluded and no end-to-end speed claim is made.

Raw per-video scores and routing diagnostics are saved. Where masks are known, fixed-0.5 mask IoU/Dice are reported rather than thresholds fitted on test pixels. Pixel AUC and pointing-game metrics are not implemented in this first notebook; unseen-method localization requires externally supplied masks. No Pass/Fail experiment record is created before a real run.

## Runtime artifacts

A new `OUT_DIR` receives `config.json`, `manifest.json`, `history.json`, `best.pt`, `last.pt`, `calibration.json`, and, only after explicit final testing, `test_scores.json`, `test_metrics.json`, `final_test_started.json`. Model tensors and maps remain accessible through the notebook functions. Nothing writes a fabricated success report into `record/`.

## Preflight and limitations

Embedded tests cover DCT orthonormality, feature/output shapes, finite outputs, sparse capacities, balance-loss gradients, bounded correction, padding invariance, missing-frame rejection, single-view equivalence, and finite training gradients. They run without pretrained downloads using a random-weight diagnostic model; actual training explicitly loads pretrained weights.

The implementation was reviewed statically but could not be executed here. Run every preflight before spending training compute. No dataset, GPU, Python execution tool, pretrained cache or GitHub write connection was available. Accuracy, numerical stability, GPU memory and speed remain unmeasured. Expert utilization warnings do not alone establish collapse or usefulness. Neither synthetic localization nor agreement-based Q/U establishes generalization.

Only the two requested i2 files are added. i1 and the design proposal are preserved.
