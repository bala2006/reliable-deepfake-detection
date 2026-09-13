# Special Architecture: LiteGuard-Face v1

**Status:** proposed design, not implemented or benchmarked.  
**Candidate next experiment:** i2-LiteGuard-Face-v1.  
**Review date:** 2026-09-12.  
**Repository baseline:** `0f1ebad5543ac5d31420c2f319aac1cc91ab9509`.

## 1. Recommendation

Use **one pretrained MobileNetV3-Large image backbone at 224×224**, a directly supervised global classifier, an optional small local/residual evidence head, and simple video pooling. Learn moderate compression invariance during training, not through a second inference view. Add calibrated abstention and adaptive frame sampling only after validating the fixed-budget detector.

The best first experiment is the **RGB-only baseline**, not every proposed component at once. Promote the residual head or adaptive policy only if it improves the accuracy–latency trade-off on held-out development data. Keep the final system on that trade-off frontier rather than pursuing architectural complexity for its own sake.

This is a defensible low-compute starting point, **not a proven universal best model**. High accuracy, high precision, high recall, low latency, and generalization are competing objectives. No architecture can guarantee detection of all unseen generators or recover forensic evidence destroyed by heavy compression. Abstention improves reliability only when its coverage and errors are measured honestly.

**Scope:** face-manipulation detection in still images and videos. This does not cover audio deepfakes, arbitrary synthetic scenes, or establish authenticity of an entire video from a few sampled faces.

## 2. Current directory structure

The complete recursive tree returned at review time contains:

```text
reliable-deepfake-detection/
├── .gitignore
├── .vscode/
│   └── settings.json
├── README.md
├── architecture/
│   ├── i1-Stable-RouteNet-v4.md
│   └── rules.md
├── docs/
│   ├── i1-Stable-RouteNet-v4.docx
│   └── rules.md
├── notebook/
│   ├── i1-Stable-RouteNet-v4.ipynb
│   └── rules.md
├── record/
│   ├── i1-Stable-RouteNet-v4-Fail.md
│   └── rules.md
└── temp/
    └── _build_nb.py
```

This proposal adds `specialArchitecture.md` at the repository root, as requested. No existing experiment artifact is overwritten. No implementation, new notebook, or measured-results record is created by this change.

**Review coverage:** README, full Markdown architecture, failure record, folder conventions, configuration files, visible notebook configuration/model/loss/evaluation cells and outputs, and selected builder sections. The notebook is serialized as a very large JSON line; its response was truncated. This is a targeted source review, not a complete notebook/output audit. The DOCX was listed but not parsed. No dataset, weights, runtime, or dependency lockfile was available for execution. The builder and executed notebook differ; the builder must not be assumed to reproduce the executed v4b run.

## 3. What the existing evidence actually shows

The [i1 failure record](record/i1-Stable-RouteNet-v4-Fail.md) reports training AUROC 1.000, worst evaluation AUROC 0.578, mean 0.596, clean 0.602, and localization pixel AUROC 0.938. At threshold 0.5 its clean confusion matrix predicts every example as real. Reported training time is about 7.4 hours.

Important qualification: the visible notebook's `full_validation()` evaluates `RobustValDataset`, derived from real videos plus synthetic SBI fakes. The 90 real / 90 fake confusion counts match this validation construction. These figures must not be relabeled as established cross-dataset test performance. Synthetic mask localization is not proof of localization on real manipulations. Its IoU/F1 also use a threshold optimized on the evaluated localization data, rather than an independently frozen threshold.

### Source-level findings

| Finding in visible code | Consequence / design response |
|---|---|
| Executed notebook prints 311.3M total parameters, 6.98M trainable; uses frozen DINOv2-L at 392×392 with 784 spatial tokens | Frozen weights still incur forward compute and model memory. Replace the deployment backbone, not just its trainable adapters. |
| Actual printed split is 180 training videos and 90 validation videos; prose advertises several larger counts | Generate run summaries from the saved manifest, not hard-coded prose. |
| Five router logits are top-k selected, then indices are clamped to four spatial experts | Index 4 is redirected to expert 3, possibly producing duplicate expert selections and inflated agreement. This is not a valid sparse-routing fix. |
| Frequency branch runs separately, while count reporting pads its dispatch count with zero | The reported fifth “dead expert” is partly a bookkeeping consequence, not proof that the frequency branch never executes. |
| `switch_balance_loss` multiplies non-differentiable hard counts by `P.detach()` | That balance term supplies no router gradient. A corrected comparison must leave mean probabilities differentiable. |
| `router_z_loss` is called on probabilities despite a logits-presence guard | The guard does not fix the wrong input. |
| Paired training uses `R = sqrt(A*C)`; single-view inference uses `R = A` | Train/inference evidence weighting differs. Use one single-view classifier graph in both modes. |
| The frequency expert computes token-neighbor differences in nested Python loops | It is not a pixel DCT. Replace with vectorized fixed filters only if useful. |
| Low-rank layers operate on frozen output tokens | This is feature adaptation, not attention-layer LoRA fine-tuning inside DINOv2. |
| Real/SBI evaluation is the model-selection signal; real manipulation results are not established by the failure summary | First repair validation methodology and compare a simple classifier. |
| Leakage checks compare directory IDs, not identity and manipulation ancestry | Passing those assertions does not establish identity- or source-video-disjoint evaluation. |

These are implementation observations and plausible contributors, **not experimentally isolated causes**. Do not conclude that all transformers or mixture-of-experts models fail. Conversely, do not carry forward the old document's predicted “94–97% AUC” or additive percentage improvements as measured evidence. Prototype routing does not mathematically prevent domain shortcuts.

## 4. Why this design instead of the alternatives?

| Candidate | Strength | Trade-off | Recommendation |
|---|---|---|---|
| MobileNetV3-Small | Very small CPU/mobile footprint | Less capacity for subtle evidence | Ultra-low-compute baseline |
| **MobileNetV3-Large** | Compact, established convolutional implementation; pretrained features | Fine forensic detail can disappear at late strides | Default starting point; optional stride-16 local head |
| EfficientNet-B0 | Strong compact accuracy baseline | More compute than MobileNetV3-Large under common 224 configurations | Mandatory compact challenger if budget permits |
| DINOv2-S/14 | Strong pretrained representation candidate | Substantially heavier than the mobile student | Research/teacher challenger, not required |
| Current DINOv2-L + experts | High representation capacity | Large forward cost, dispatch complexity, fragile objectives | Diagnostic reference, not the lightweight deployment design |
| Two full RGB/frequency backbones or 3D transformer | Potential complementary/temporal evidence | Multiplies cost and training demands | Exclude from first iteration |

Choose using measured hardware latency and held-out quality, not ImageNet accuracy or parameter count alone. A larger backbone can win generalization; if it does, report the trade-off rather than declare the smallest detector “best.”

## 5. Proposed inference architecture

```text
Video / image
   │
   ├─ Decode sampled timestamps; detect and track faces
   ├─ Validate crop quality, preserve moderate face margin
   │
   └─ Face RGB crop [3,224,224]
          │
          ├─ MobileNetV3-Large features (one shared backbone)
          │      ├─ stride 32 [960,7,7] → GAP → global classifier g
          │      └─ stride 16 [112,14,14] → 1×1 → [32,14,14]
          │                                               │
          └─ OPTIONAL fixed luminance residual filters    │
                    → tiny depthwise CNN [32,14,14] ──────┤
                                                          ↓
                                            local evidence map [1,14,14]
                                                          ↓
                                      frame logit z = g + bounded local correction
                                                          ↓
                                      mean frame logit for each face track
                                                          ↓
                                      calibrated score + quality/coverage checks
                                                          ↓
                              manipulation detected / no evidence detected / uncertain
```

### 5.1 Preprocessing is part of the system

Use cached face crops for initial model experiments so detector variability does not obscure the classifier comparison. For deployment, benchmark an actual licensed face detector on the target device; selection is not finalized without hardware requirements. Detect on sampled frames and propagate boxes using tracking when valid. Never assume face detection/decoding is free.

Use a moderate crop margin, initially about 20% of the face-box dimensions, clipped to the image. Do not stretch arbitrary rectangular images to a square; preserve aspect ratio through the defined crop/padding procedure. Use identical RGB channel order, interpolation, crop policy and normalization in training and deployment. Generate residuals before ImageNet normalization.

Do not use a whole-frame fallback as though it were an ordinary face crop. Missing face, severely occluded/tiny face, track discontinuity, or failed decode must produce an explicit unsupported/uncertain state. Keep separate track results for multiple people. A video containing one manipulated track should not be declared clean by averaging it with many other people; calibrate any video-level combination separately because more tracks create more false-positive opportunities.

### 5.2 Global RGB path: mandatory baseline

Use standard-width, non-dilated `torchvision.models.mobilenet_v3_large` feature layers and the appropriate pretrained weight preprocessing. Torchvision is already imported in the repository, but the installed version/weight availability must be checked during implementation.

For 224×224 inputs, retain final convolutional features `[B,960,7,7]`. Replace the original ImageNet classifier with:

- global average pool → `[B,960]`;
- linear 960→128, Hardswish, dropout 0.2;
- linear 128→1, yielding real-valued logit `g`.

Fake is label 1. This RGB-only model is A0 and the fallback if all extensions fail. Supervise classification directly; never require a learned manipulation mask to unlock the global path.

### 5.3 Local evidence path: optional

Tap the standard backbone's stride-16 112-channel feature map. Confirm the actual implementation's tensor shape with assertions rather than depending on an undocumented layer index. Project 112→32 with a 1×1 convolution.

Optionally complement it with a small residual branch:

1. Convert raw RGB in [0,1] to luminance `Y = 0.299R + 0.587G + 0.114B`.
2. Apply three fixed 3×3 filters: horizontal Sobel, vertical Sobel, and Laplacian, using reflection padding. Normalize each kernel by its absolute-weight sum and retain signed responses. These are **high-pass residuals**, not DCT coefficients.
3. 3×3 convolution, stride 2, 3→8 channels: 112×112.
4. Depthwise 3×3 stride 2 + pointwise 8→16: 56×56.
5. Depthwise 3×3 stride 2 + pointwise 16→24: 28×28.
6. Depthwise 3×3 stride 2 + pointwise 24→32: 14×14.

Use normalization/nonlinearity compatible with the chosen export backend. Use fused BatchNorm at deployment; during training use sufficiently large real microbatches or frozen backbone BatchNorm statistics. Gradient accumulation alone does not improve BatchNorm statistics.

Fuse local RGB features with `a * residual_features`, with `a = 0.25*sigmoid(a_raw)`, initially a=0.05. When the residual branch is disabled, use RGB local features alone. Follow with depthwise 3×3, pointwise 32→32, Hardswish, and a 1×1 evidence-map classifier.

Pool local logits with the mean of the largest 20% of the 196 positions (40 positions). This top-k pooling is a hypothesis for localized manipulation, not a calibrated explanation map. Compare against plain mean pooling because top-k can overreact to isolated artifacts.

Combine as `z = g + beta*local_logit`, where `beta = 0.3*sigmoid(beta_raw)`, initially beta=0.05. Bound weights to temper branch influence, but note this does **not** bound the correction's magnitude when local logits are large; monitor branch logit scales. Do not call this a formal reliability guarantee. Supervise the global classifier separately so it remains useful if the local branch overfits.

No segmentation supervision, trust multiplication, expert router, frequency transformer, or second full backbone is required.

### 5.4 Video pooling and compute allocation

First ship a **fixed 8-frame baseline**: sample temporally distributed unique frames, compute one logit per frame, and average logits per track. Average logits, not independent-frame posterior odds; correlated frames do not create independent evidence. Training uses the same pooling function with a variable number of frames.

An optional budget-aware policy uses **4 → 8 → 16 unique frames**, caching previous predictions. Construct a deterministic nested sequence that covers the duration early. Route ambiguous scores, high frame disagreement, and insufficient temporal coverage to the next budget. If the budget is exhausted or quality is inadequate, abstain rather than invent confidence. Low-quality early frames may trigger a search for better frames, not an automatic “real.”

Calibrate and evaluate the **whole stopping policy**, including early exits, on held-out data. A confident model can be confidently wrong; confidence alone is not evidence that unobserved intervals are clean. For long videos, use bounded windows across the duration or return a clearly labeled sampled-coverage result. A 16-frame cap cannot guarantee detection of a brief manipulated interval in an arbitrarily long video.

For short videos use the number of distinct usable frames actually available; do not duplicate frames and count them as new evidence. Keep a fixed-budget fallback if adaptive sampling loses recall or calibration.

## 6. Data and generalization: the main investment

### Separate two scientifically different protocols

**R: strict real-only training.** Train on real faces plus SBI and varied synthetic blends, respecting the existing project premise. Do not use actual manipulated examples to select checkpoints or thresholds if claiming a strict real-only development protocol. Select using held-out synthetic development data and acknowledge that this is an imperfect proxy for real-fake generalization. Real-only anomaly calibration may control an in-distribution real rejection rate, but cannot establish fake recall or deployment precision.

**M: practical mixed-source training, recommended when permitted.** Train with authentic videos, diverse genuine manipulations, and synthetic examples. An initial label-balanced minibatch target is 50% real, 25% actual manipulated, 25% synthetic fake; adjust only on development data. Balance identity/source/generator exposure within available label groups. This is a different experiment from R, not an improvement under an identical protocol. If actual manipulated data are currently reserved for testing, do not quietly reuse them: define a new disjoint training/development corpus and preserve the final test set.

SBI alone is not sufficient evidence of universality. Include multiple manipulation mechanisms and realistic benign edits, codecs, cameras, resizing pipelines, makeup, lighting, and motion. Confirm each class appears across acquisition conditions where feasible, so dataset origin is not a label shortcut.

### Split before extracting frames or generating fakes

Create a canonical manifest containing dataset, original-video identifier, subject identities where available, source/target manipulation ancestry, generator, compression, and split. All descendants, alternate encodes and near-duplicates of an original must remain in the same split. For swaps, group both source and target ancestry. Audit connected groups and duplicate content, not just filenames. If identity information is unavailable, state that limitation.

Use disjoint **train / development / calibration / final test** groups. Development selects architecture and stopping rules; calibration fits final scores and thresholds after architecture freeze. With little data, use grouped nested development folds rather than pretend 90 videos support precise low-FPR certification.

Leave entire datasets and generator families out for external tests. Celeb-DF is not a fully unseen dataset if its real videos were used during training: report that as a partially seen-domain evaluation. Report strict unseen-dataset results separately. Do not select the best checkpoint by final-test AUROC or reuse failed final tests for endless tuning without declaring them development data.

### Cheap, controlled augmentation

Apply crop jitter and moderate JPEG/re-encoding, blur, resizing, color and illumination changes to **both labels under the same distribution**. Include unmodified examples. Initial development ranges: JPEG quality 40–100, resize factor 0.5–1.0, modest Gaussian blur; reserve harder/composed conditions for separately reported stress tests. Specify exact implementation and radius/sigma semantics.

Do not assume arbitrarily strong degradation preserves observable evidence merely because the semantic label stays unchanged. Mask dubious samples out of consistency loss. Inspect SBI masks and blends, ensure transformations do not create class-specific black borders, and do not present the alpha mask as exact pixel-change ground truth after Poisson blending.

If temporal models are later tested, synchronize synthetic transforms across adjacent frames where appropriate; otherwise synthetic flicker can become a trivial shortcut. Dense temporal networks are outside the first iteration.

## 7. Training specification

Use the existing PyTorch/torchvision stack; no new model framework is needed. Pin versions and pretrained-weight identifiers during implementation.

1. **Sanity run:** train RGB-only A0 on a small fixed set and verify that it can fit labels; perform label-shuffle and preprocessing checks. Do not use memorization as a generalization result.
2. **Warm-up:** train the new head with frozen backbone for approximately one epoch.
3. **Fine-tune:** unfreeze later backbone blocks; initial AdamW learning rates 1e-4 for the backbone and 5e-4 for heads, weight decay 1e-4, cosine decay, 5% optimizer-step warm-up, gradient norm cap 1.0. These are starting settings, not established optima. Try full-backbone fine-tuning only as a controlled comparison.
4. **Sampling:** start with two frames per source video per update, diversify videos before adding many correlated frames, and introduce variable 1/2/4/8-frame groups for pooled-video training. Mask padded frames. Benchmark effective batches of 32–64 crops subject to memory limits.
5. **Budget:** run a short 5-epoch screen, then a capped 20-epoch development experiment with early stopping after five non-improving evaluations. Compare optimizer updates, unique videos, and GPU-hours; epochs across different data pipelines are not equal compute.

Use FP16 mixed precision on T4, with loss/scaler handling appropriate to PyTorch version. Start on one device: a small CNN may not benefit from two-device overhead. If multi-GPU training is useful, benchmark DDP rather than assume DataParallel scales well. Sample statistics and preprocessing time matter as much as network time.

### Minimal objective

For branch-enabled models, initially use:

`L_cls = 0.5 * BCEWithLogits(mean_t(z_t), y) + 0.5 * BCEWithLogits(mean_t(g_t), y)`

Use true labels with no smoothing initially; keep label balancing and final deployment-prior calibration explicit. For RGB-only A0 this reduces to ordinary pooled classification BCE. Fully manipulated clips may use frame-level auxiliary classification as a separate ablation; do not assign the video label to every frame when manipulations are only intermittent.

**Optional consistency:** for about 25% of training groups, create a moderate second view using matching geometry. Run the same single-view model independently on each view. Add its supervised classification loss with normalization by the number of evaluated views, plus `0.1 * JSD(Bernoulli(p_original), Bernoulli(p_augmented))` on valid pairs. Compute probabilities/logs stably. Gradients reach both predictions; BCE anchors labels. Start with zero consistency weight, then ramp to 0.1. Validate that fake recall under compression does not fall.

This adds roughly 25% more crop forwards than single-view training at that pairing rate, excluding augmentation/backprop details. It adds **no extra inference view**. Never place view-comparison variables inside the deployment classifier.

**Optional distillation:** only after a teacher beats the student on development data. Distill task-relevant video logits or pooled features; generic DINO features are not automatically forensic targets. Do not distill the failed detector merely because it is large. Teacher use is a separate, explicitly higher-training-compute experiment; no teacher is shipped. Cache targets only for the exact corresponding crop/transformation.

## 8. Reliability, precision and abstention

Fit a scalar temperature on disjoint calibration video logits after model selection. If deployment priors differ, validate an intercept/prior correction on representative data; temperature alone does not fix class-prior shift. Calibration is empirical and can fail under domain shift.

Select two thresholds on calibration data:

- above `tau_fake`: manipulation detected;
- below `tau_clear`: no manipulation evidence detected in the analyzed faces/coverage;
- in between, or unsupported quality/coverage: uncertain.

`tau_clear < tau_fake`. “No evidence detected” is not “authenticated real.” Set thresholds to a declared operating objective, such as bounded false accusations with a minimum fake-recall/coverage constraint. If no threshold satisfies the objective, report failure rather than select an empty-positive operating point. Freeze thresholds across final datasets; per-dataset oracle thresholds may appear only as explicitly labeled diagnostics.

At fake prevalence pi:

`precision = (TPR*pi) / (TPR*pi + FPR*(1-pi))`.

For example, TPR 90%, FPR 1%, and fake prevalence 1% give only about **47.6% precision**. Balanced benchmark accuracy is therefore not a deployment precision claim. Report precision for declared prevalence scenarios and measure real-domain FPR carefully.

A zero-error result on a small sample does not certify a tiny FPR: with zero false positives on n independent representative real videos, the approximate one-sided 95% upper bound is 3/n. Around 3,000 independent real videos with zero errors are needed just to approach a 0.1% upper bound. Correlated subjects/videos reduce effective evidence.

Report risk-versus-coverage, abstention rate by class/domain, automatic-decision precision/recall, and all-input errors with uncertainty handled explicitly. Never hide rejected examples from the denominator. Frame disagreement is a warning signal, not a calibrated posterior uncertainty estimate. No dropout ensemble or test-time augmentation is mandatory.

## 9. Compute and size targets—not measurements

| Item | Planning estimate / target |
|---|---|
| RGB-only model | About 3.1M parameters with replacement head; verify instantiated count |
| Local + residual extension | Target total below 3.5M parameters |
| 224×224 classifier compute | Target below 0.30 GMAC/frame; common full MobileNetV3-Large reference is about 0.22 GMAC |
| FP32 weight payload at 3.5M | About 14 MB decimal, excluding runtime metadata |
| FP16 weight payload at 3.5M | About 7 MB decimal |
| INT8 weight payload at 3.5M | About 3.5 MB decimal; not a guarantee of actual file/RAM size |
| Fixed 8-frame crop inference | Target below 2.4 GMAC/track, excluding decode/detection |
| Adaptive policy | 4, 8 or 16 frames; distribution must be measured |

These are engineering budgets, not profiler outputs. MAC and FLOP conventions differ; if multiplication and addition count separately, 1 MAC is approximately 2 FLOPs. Compare models using one profiler and the same convention.

For the adaptive policy, expected classifier work is `C_frame * (4 + 4*p_reach8 + 8*p_reach16)`. Include preprocessing, detection, tracking, I/O and all tracks in total system cost. Relative to the notebook's 311.3M total parameters, a 3.1–3.5M detector is roughly 90–100× smaller in parameter count, **not automatically 90–100× faster**.

No target FPS is asserted because deployment CPU/GPU, video duration/resolution, batching and number of faces are not specified. Measure warmed-up batch-1 p50/p95 latency and throughput separately, plus peak host/GPU memory and end-to-end seconds per input minute. Report hardware, software versions, precision, threads, number of tracks and sample coverage. A low-MAC model can still be slow on an unsuitable kernel backend.

Export only after FP32/FP16 validation. Test ONNX parity if that is the chosen serving format. Attempt representative-data INT8 calibration, then quantization-aware training only if necessary. Residual filters, normalization and logits may need higher precision. Refit score calibration and retest after quantization; if the backend leaves many float operators, the hypothetical INT8 footprint does not apply.

## 10. Experiments and release decisions

Run ablations under the same split, seeds and training budget:

| ID | Change | Question |
|---|---|---|
| A0 | MobileNetV3-Large RGB, mean video pooling | Is a direct compact classifier already better? |
| A1 | A0 + matched moderate augmentations and sparse paired-view consistency | Does robustness improve without suppressing weak fake evidence? |
| A2 | A1 + local RGB head | Does local evidence add value beyond global pooling? |
| A3 | A2 + residual branch | Does high-pass evidence generalize, or learn codec shortcuts? |
| A4 | Best fixed-budget model + calibrated adaptive sampling | Can average compute fall at matched recall/FPR/coverage? |
| A5 | Best model + optional teacher distillation | Is extra training compute justified? |
| C1/C2 | MobileNetV3-Small / EfficientNet-B0 under matched protocol | Is the proposed backbone actually on the measured trade-off frontier? |

Compare local mean versus top-k pooling, 160/192/224 resolution where budget permits, and 4/8/16 fixed frames. Do not expand every axis at once. Use at least three seeds for finalists and confidence intervals for paired video-level differences; bootstrap source/identity groups when videos share ancestry. Do not bootstrap frames as independent videos.

For each held-out domain/generator/quality condition report AUROC, average precision, EER, accuracy, balanced accuracy, precision, recall, F1, confusion counts, TPR at declared FPRs, Brier score, calibration plots/ECE and coverage. Report low-FPR uncertainty and sample counts. Evaluate benign edits and real demographic/quality subgroups where consent and labels permit. For unknown or one-class slices, mark undefined metrics rather than silently omit the slice.

**Selection:** prioritize worst predeclared development-domain performance with acceptable low-FPR recall; among statistically comparable models choose lower end-to-end p95 latency/memory. Avoid selecting on a tiny noisy worst slice without uncertainty. Final external test is an assessment, not another selection loop.

**Promotion gates:**

- No known train/development/calibration/test ancestry overlap.
- Classification and reliability improve versus A0 under a predeclared operating point, not only training/localization metrics.
- Added modules earn their compute; otherwise remove them.
- Quantization/adaptive policy preserve acceptable low-FPR recall, calibration and coverage.
- No release-grade claim until representative real-manipulation evaluation exists.

Absolute acceptable accuracy/FPR/latency targets require the deployment use case. Do not invent a universal “Pass at 95%” threshold.

## 11. Tests to implement before the next training run

- Label direction and confusion-count sanity; shuffled labels should not generalize.
- Group/ancestry split validation; alternate-encoding and near-duplicate detection.
- Crop/RGB/normalization equivalence between training, evaluation and export.
- Exact feature shapes and finite losses/gradients for batch sizes 1 and larger.
- Variable frame counts, padding masks, single-frame inputs and no-face/error handling.
- In evaluation mode, paired-view execution must match independent single-view execution for each view; no hidden companion-view dependency.
- Local path disabled must reproduce global-only inference; no dependence on synthetic masks at test time.
- Fixed residual kernels stay frozen; boundary padding does not create label-specific artifacts.
- Adaptive sampler reuses predictions, never counts duplicate frames, obeys budgets and reports coverage.
- Calibration never sees test labels; bootstrap units are videos/groups, not frames.
- Export parity and post-quantization operating-point comparison.
- Regression benchmarks include video decode and face detection, not only prepared tensors.

## 12. What could be novel—and what is not

MobileNetV3, high-pass residuals, local pooling, consistency regularization, distillation, calibration and early exiting all have substantial prior art. Their combination alone is not evidence of publishable novelty.

A research hypothesis worth testing is **coverage-aware progressive forensic inference**: a compact directly supervised classifier plus a bounded local evidence path allocates additional temporal samples using quality, evidence disagreement and coverage, then abstains when the budget cannot support the desired operating point. Compare this against confidence-only stopping and uniform sampling at matched average compute and recall/FPR constraints.

A convincing contribution would be measured robustness and precision–coverage–compute improvements across truly unseen generators, with source-disjoint evaluation and a failure analysis. Do not use “first,” “SOTA,” “universal,” or “best generalization” without a systematic literature search and reproducible comparisons.

## 13. Research grounding and limitations

Live internet browsing was not available during this review. The following are established references recalled for design grounding, **not freshly retrieved results or an exhaustive 2026 literature review**. Verify details and newer work before making publication claims. No numerical performance prediction is borrowed from these papers.

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244): efficient mobile convolutional architecture; motivates the deployment backbone, not a claim about deepfake accuracy.
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946): compact challenger for the accuracy–compute comparison.
- [Detecting Deepfakes with Self-Blended Images](https://arxiv.org/abs/2204.08376): synthetic self-blending training; motivates an SBI baseline, not immunity to unseen generators.
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193): pretrained representation reference and optional teacher/backbone comparison.
- [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174): motivates testing pretrained representation transfer; general synthetic-image detection is not identical to compressed face-video detection.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599): temperature scaling; calibration under domain shift still needs testing.

Repository evidence: [README](README.md), [existing architecture](architecture/i1-Stable-RouteNet-v4.md), [executed notebook](notebook/i1-Stable-RouteNet-v4.ipynb), [failure record](record/i1-Stable-RouteNet-v4-Fail.md), and [builder](temp/_build_nb.py). See review-coverage limitations above.

## Bottom line

**Spend compute on better data diversity and validation before spending it on experts.** Start with a roughly 3M-parameter MobileNetV3-Large detector, direct global classification and fixed 8-frame pooling. Test consistency, then local/residual evidence, then calibrated adaptive sampling independently. Retain only measured improvements. This is a lightweight, falsifiable design with a credible path to reliability—not a promise of unmeasured accuracy.
