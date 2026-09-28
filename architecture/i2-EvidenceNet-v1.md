# EvidenceNet v1 — Localization-First Detection with Existence Aggregation

**Iteration:** `i2` · **Architecture:** EvidenceNet · **Version:** v1
**Supersedes:** the Stable-RouteNet line (`i1-Stable-RouteNet-v4`), whose token-routing
hypothesis was tested and falsified in i1.
**Companion artifacts:** `notebook/i2-EvidenceNet-v1.ipynb` (executable),
`record/i2-EvidenceNet-v1-{Pass,Fail}.md` (to be written after the run).

---

## 1. Plain-language goal

Build a detector that decides whether a face in a video is real or manipulated, and
generalizes to datasets and manipulation methods it has never seen.

The v4 family tried to do this by routing patches to specialist experts and weighting
them by a trust score. That failed, and we can now say precisely why. EvidenceNet keeps
the part that worked — the model learned *where* the manipulation was (pixel AUC 0.938 on
i1) — and rebuilds the part that did not: the step that turns a localization map into a
real/fake decision.

**One-sentence difference from i1:** i1 averaged its patch evidence into a single vector
and then asked a small classifier what that average looked like; EvidenceNet asks
"*does any patch contain artifact evidence?*" using an existence operator, which is the
correct question for a signal that is inherently sparse and localized.

---

## 2. Why i1 failed — measured, not guessed

The i1 record reports `train AUC 1.000` vs `worst eval AUC 0.578`. Below is the
mechanism, read directly from `notebook/i1-Stable-RouteNet-v4.ipynb`.

### 2.1 The decision path was a convex combination (the primary fault)

After the expert blend, i1 computed the image score as:

```python
W_norm = W_raw / (W_raw.mean(dim=1, keepdim=True) + EPS)   # normalized to mean 1
Zf = (Hout * W_norm[..., None]).mean(dim=1)                # mean over 784 patches
Zv = Zf.reshape(B, T, -1).mean(dim=1)                      # mean over 8 frames
logit = self.classifier(Zv)
```

Because `W_norm` is normalized to mean 1, `Zf` is a **convex combination** of patch
features: it can reweight contributions but can never amplify them. The classifier
therefore never sees "is there a forged patch"; it sees "what does the average of the
whole face look like".

That is fatal for a sparse signal. Let per-patch evidence be `m[p] ∈ [0,1]`, with a forged
region covering a fraction `f` of the face and `m = 1` inside it, `m = 0` outside. Then:

| operator | value on a fake | value on a real | fake−real contrast |
|---|---|---|---|
| convex mean (i1) | `f` | `0` | **`f`** |
| max / LSE (existence) | `≈ 1` | `≈ 0` | **`≈ 1`** |

The detector's usable signal is proportional to the mask area fraction `f`. For
high-quality localized manipulations `f ≈ 0.02–0.05`, so **i1 threw away 95–98 % of its
own evidence before the classifier ever saw it.**

The measurements agree exactly. i1's own diagnostics:

| quantity | value | reading |
|---|---|---|
| pixel AUC (localization) | **0.938** | patch-level evidence is strong and correctly ranked |
| IoU / mask F1 | 0.600 / 0.750 | the map is genuinely accurate |
| mean M on fake frames | 0.3336 | ≈ same as real |
| mean M on real frames | 0.3296 | ≈ same as fake |
| fake − real difference | **0.004027** | 1.2 % of the mean → indistinguishable |

So the model could rank patches *within* an image but produced essentially the same
image-level statistic for real and fake. The information existed and was destroyed by the
aggregator. This is the textbook failure of mean pooling on sparse targets, and the reason
multiple-instance-learning literature separates *instance* scores from *bag* scores with
max/noisy-OR/LSE operators rather than averages.

### 2.2 A globally learnable shortcut made in-domain training trivial

`make_sbi()` applied `photometric_jitter()` to the blend source but only a light
quality-alignment jitter to real frames, then blended with `ratio ~ U(0.15, 1.0)` over a
raw landmark polygon plus a Gaussian blur of 2–8 px. A `ratio` of 1.0 replaces the whole
face region, and the exclusive source-side jitter changes global colour/brightness
statistics. Fakes were therefore separable from reals by *global* statistics alone.

That is consistent with `train AUC 1.000` in 5 epochs on 180 videos, and with the
localization maps being huge (a full-face polygon is easy to locate but its boundary is
not informative).

### 2.3 The "LoRA" in i1 was not LoRA

The architecture document describes "LoRA adaptation in DINOv2 backbone". The code is:

```python
self.lora_up   = nn.Linear(1024, 8, bias=False)    # on the *output* tokens
self.lora_down = nn.Linear(8, 384, bias=False)
Fr = Fr + self.lora_down(self.lora_up(tokens)) * self.lora_scale
```

That is an 11 k-parameter **rank-8 linear bottleneck applied to the frozen trunk's output
tokens**. Nothing inside the transformer was adapted; the backbone features are exactly
the frozen pretrained ones. So i1 never tested LoRA, and the "DFF-Adapter style" and
"LoRA" claims in the v4 document are inaccurate. Separately, the same document's "Total
params: ~350M (backbone frozen, adapter + experts + router trainable)" conflates the
frozen 300 M trunk with the ~7–8 M trainable head.

### 2.4 The router was a no-op, and the "dead expert" was a measurement artifact

i1 reported `routing entropy fraction 0.998` and `token entropy 1.606`. Since
`ln(5) = 1.609`, the router's output distribution was **uniform to within 0.2 %** — it
learned nothing. Meanwhile dispatch shares were `[0.166, 0.061, 0.255, 0.518, 0.000]`.
Both facts are explained by one line:

```python
topi = topi.clamp(max=len(self.experts) - 1)   # hard clamp of index 4 -> 3
```

`topi` was clamped to `[0,3]` for the 4 spatial experts, so the frequency expert
(index 4) could *never* be dispatched. The reported "1 dead expert" is therefore a
guaranteed consequence of the clamp, not a finding about routing collapse — the v4
document and the i1 record both misdescribe it. With near-uniform router probabilities,
top-k selection is decided by tiny logit differences, which is why one expert absorbed
51.8 % while the routing distribution itself was flat.

### 2.5 The cross-dataset evaluation never ran

In the saved notebook, cell 14 (`FINAL EVALUATION — run once, after freeze`) has **no
outputs**. Every number in `record/i1-Stable-RouteNet-v4-Fail.md` is a *validation*
number (clean / JPEG-30 / blur-15 / resize-50), summarized from `metrics.json` and
`progress.json`. **i1 has no measured DFDCP, Celeb-DF-v2 or FF++ cross-manipulation
result at all.** The record's own "Next steps" call for a validated threshold, and the
notebook's calibration was fit on the same validation set used for selection.

### 2.6 The data pipeline, not the GPU, was the bottleneck

i1 ran 900 batches in 443.5 min = **29.6 s/batch** for 32 images/batch ≈ 1.1 images/s.
The trunk forward is under `torch.no_grad()`, so the GPU work is a single ViT-L/14
forward: at 392² that is a few hundred GFLOPs per image, i.e. **25–40 images/s** on 2×T4
(§7.2). i1 was therefore **~30× data-pipeline bound** — `make_sbi()` calls
`cv2.seamlessClone` (Poisson blending, tens of ms), `ImageFilter.GaussianBlur(2–8)` on a
392² mask, and a PIL `photometric_jitter` chain that JPEG-round-trips the image, all
inside `__getitem__`, for every one of 8 frames per clip.

Consequence for i2: epochs are cheap on the GPU and expensive in the loader. Fixing the
loader is worth more than any architectural tweak.

---

## 3. What the literature says to do instead

All numbers are as published, or quoted from the survey table in [GenD] where noted.

| Work | Venue | Mechanism | Reported cross-dataset AUROC |
|---|---|---|---|
| Xception [Rossler 2019] | ICCV'19 | plain CNN classifier | CDF-v2 81.7 · DFDCP 69.9 |
| F3-Net [Qian 2020] | ECCV'20 | two-stream frequency-aware | CDF-v2 78.9 · DFDC 71.8 · DFDCP 73.5 |
| Face X-ray [Li 2020] | CVPR'20 | **predicts the blending boundary map** (arXiv:1912.13458) | CDF-v2 79.5 · DFDCP 80.9 |
| SPSL [Liu 2021] | CVPR'21 | phase-spectrum shallow learning | CDF-v2 79.9 · DFDC 66.2 · DFDCP 75.9 |
| RECCE [Cao 2022] | CVPR'22 | reconstruction-classification | CDF-v2 82.3 · DFDC 69.6 · DFDCP 71.5 |
| SBI [Shiohara 2022] | CVPR'22 | self-blended synthetic fakes + EFN-B4 | CDF-v2 93.2 · DFD 82.7 · DFDC 72.4 · FFIW 84.8 |
| LAA-Net [Nguyen 2024] | CVPR'24 | **explicit vulnerable-point heatmap + self-consistency attention**; trained on real data only | CDF-v2 95.4 · DFD 98.4 · DFDC 86.9 |
| NPR [Tan 2024] | CVPR'24 | neighboring-pixel-relationship residual, ResNet-50 | strong cross-*generator* generalization |
| ForAda [2025] | CVPR'25 | CLIP ViT-L/14 + parallel forensics adapter | CDF-v2 95.7 · DFD 97.2 · DFDC 87.2 |
| Effort [2025] | ICML'25 | orthogonal (SVD) fine-tuning of CLIP | CDF-v2 95.6 · DFD 96.5 · DFDC 84.3 |
| GenD [Yermakov 2025] | arXiv:2508.06248 | CLIP-L + **LayerNorm-only tuning** (0.03 % params) + L2-norm with alignment/uniformity losses | CDF-v2 96.0 · DFD 97.0 · DFDC 87.1 · **mean 91.2 over 14 datasets** |
| DFF-Adapter [Zhang 2026] | AAAI'26 | multi-head LoRA adapters in every DINOv2 block + forgery-type aux branch | best overall on DF40 cross-manipulation |

### 3.1 Five findings that drive the design

**(a) Adapting the trunk: LN-tuning beats LoRA and full fine-tuning.** GenD's ablation on
identical data is unambiguous (14-dataset mean AUROC):

| tuning strategy | mean AUROC |
|---|---|
| frozen + linear probe | 76.0 |
| **+ LayerNorm tuning** | **90.3** |
| + L2-norm with alignment/uniformity | 91.2 |

and in their words: *"LoRA with rank one … allowed the model to reach a near-perfect
training AUROC (99.99 %) in two epochs, resulting in rapid overfitting"*, while *"FFT leads
to rapid overfitting"*. LN-tuning modifies 0.03 % of parameters and is the only strategy
that both improves validation AUC and resists overfitting. → **i2 uses LN-tuning, not
LoRA.**

**(b) Train on paired real–fake data from the same source video.** GenD's primary
conclusion: *"training on paired real-fake data from the same source video is essential
for mitigating shortcut learning"*. SBI already satisfies this by construction (the fake
is blended from the real video itself), and so does i2.

**(c) Explicit localization supervision is what makes synthetic-only training work.**
LAA-Net trains **using real data only** plus blending synthesis, and reaches 95.4 / 86.9 on
CDF-v2 / DFDC. Its explicit targets are the *vulnerable points* — the pixels where the
foreground and background contributions are most equal. It defines the boundary target as

```
B = 4 · M ⊙ (1 − M)
```

which peaks exactly where the blend mask `M ≈ 0.5`, i.e. at the seam. This is the same
insight as Face X-ray: supervise **the boundary**, not the whole edited region. i1
supervised only the binary region mask.

**(d) The decision should read localized artifact evidence, not a global average.**
LAA-Net's framing of the failure mode is i1's failure mode: methods *"based on a supervised
binary classifier coupled with an implicit attention mechanism … do not generalize well"*,
and *"through successive convolutions, localized features across layers gradually fade"*.
Its fixes are explicit attention on vulnerable points and an Enhanced-FPN that *"spreads
discriminative low-level features into the final feature output"*.

**(e) Low-level pixel residuals carry honest, transferable signal.** NPR (CVPR'24) shows
neighboring-pixel-relationship statistics discriminate synthetic from real imagery across
generators. A ViT patch is 14×14 px, so a 1–3 px blending seam is largely invisible to the
trunk; a high-pass/NPR residual branch sees it directly and costs almost nothing.

---

## 4. Architecture

### 4.1 Full pipeline (vertical flow)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT: one face clip, 392×392 × 8 frames  (28×14 → 784 patch tokens)    │
│  clean view  x   +   intervention view  xi (JPEG / blur / resize / photo) │
│  plus, for synthetic fakes only: soft blend mask m, blend weight w,       │
│  binary region mask, and synthesis-recipe label                           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FROZEN TRUNK — DINOv2-L/14-with-registers        [no_grad, 300 M]       │
│  strips 1 CLS + 4 register tokens → 784 patch tokens @1024               │
│  (LN-tuning is a flag-gated stage-2 option; default OFF for this run)     │
└──────────────┬───────────────────────────────┬───────────────────────────┘
               │ patch tokens (N,784,1024)     │ CLS token (N,1024)
               ▼                               ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────┐
│ FORENSIC ADAPTER (trainable) │   │ GLOBAL PATH (trainable)              │
│ 1024→384, 2 transformer blks │   │ Linear(1024→384) → L2-normalize      │
│ then L2-normalize            │   │ → global logit + alignment/uniformity│
└──────────┬───────────────────┘   └──────────────┬───────────────────────┘
           │ Z (N,784,384), unit-norm patches      │ g (N,384), unit-norm
           ▼                                       │
┌──────────────────────────────────────┐           │
│ THREE PATCH HEADS (each MLP→1)       │           │
│  • evidence head   e[p]  "fake here?"│           │
│  • boundary head   b[p]  "seam here?"│           │
│  • self-consistency head  c[p]       │           │
└──────────┬───────────────────────────┘           │
           │                                       │
           ▼                                       │
┌──────────────────────────────────────────────────┴───────────────────────┐
│  LOW-LEVEL RESIDUAL BRANCH (trainable, cheap, from raw pixels)            │
│  fixed high-pass  x − avgpool3×3(x)  ⊕  local range  max3 − min3          │
│  → small CNN (stride 2) → adaptive pool to 28×28 → CNN patch head l[p]    │
│  (this is where 1–3 px compositing seams are actually visible)            │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EXISTENCE AGGREGATION  ← THE FIX.  No convex averaging anywhere.        │
│  for s ∈ {e, b, l}:                                                      │
│     topk_mean(s, k=24 of 784)   ← smooth-max over the strongest patches  │
│     LSE_τ(s) = logsumexp(τs)/τ − log P/τ                                 │
│     max(s), mean(s), std(s)     ← shape statistics of the evidence map   │
│  ⊕ global logit ⊕ evidence-map summaries                                 │
│  → 17-dim vector → MLP(17→64→1) → BAG LOGIT                              │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LOSSES (3-stage relative curriculum, all schedules fraction-of-total)    │
│   bag BCE · MIL patch · region · boundary · consistency · alignment/      │
│   uniformity · recipe classification                                     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CALIBRATION (held-out split, disjoint from train and val)                │
│   temperature scaling → ECE/Brier before & after → validated threshold    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FINAL EVALUATION — runs once, after freeze, and DOES run this time       │
│   DFDCP · Celeb-DF-v2 · FF++ 6 manipulation families · robustness sweep   │
│   video score = mean of bag probabilities over 32 frames in 8-frame chunks│
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component notes

**Trunk.** `facebook/dinov2-with-registers-large`, frozen, `no_grad`. Input 392² = 28×14,
so the 784 patch tokens form a clean 28×28 spatial grid — the same geometry i1 used, so
the existing frames, landmarks and masks are reused unchanged. The CLS token is kept
separately for the global path (i1 discarded it).

**Adapter.** 1024→384 then two `nn.TransformerEncoderLayer` blocks (6 heads, gelu,
pre-norm), then **L2-normalization of every patch embedding** (GenD). Unit-norm patch
features make cosine geometry identical to Euclidean geometry, which is what the
boundary/consistency losses assume.

**Three patch heads.** Each is `LayerNorm → Linear(384,256) → GELU → Linear(256,1)`.
The *evidence* head answers "is this patch forged?", the *boundary* head answers "is this
patch on the compositing seam?", and the *self-consistency* head answers "does this patch
belong to the same side of the seam as the most seam-like patch?" (LAA-Net's
self-consistency branch, simplified to a single predicted anchor).

**Low-level residual branch.** Fixed, non-learned filters — a 3×3 high-pass residual
`x − avgpool₃ₓ₃(x)` and the local range `maxpool₃ₓ₃ − minpool₃ₓ₃` — concatenated to 6
channels, then a 4-block stride-2 CNN, then `adaptive_avg_pool2d` to 28×28 so its output
aligns patch-for-patch with the trunk grid. Cheap (≈1–2 GFLOPs/image) and it is the only
part of the model that can see sub-patch artifacts. Motivated by NPR and by LAA-Net's
E-FPN ("spread low-level features into the final output").

**Existence aggregation.** For each of the three per-patch score maps the aggregator
computes `topk_mean` (k = 24 of 784 ≈ 3 %), a normalized log-sum-exp (a smooth maximum),
`max`, `mean` and `std`, then concatenates the global logit and two evidence-map
summaries. This is what converts "locate the artifact" into "decide there is an artifact".

*Why LSE and not just max:* max is non-differentiable in a way that gives gradient to one
patch only. LSE is a smooth maximum whose normalized form is a principled global pooling
operator, and top-k mean is the standard robust MIL aggregator. Keeping all three
(plus mean/std, which encode "concentrated hotspot" vs "uniform noise") lets the small MLP
learn the right mixture instead of us guessing.

**Recipe classification head.** Because i2 still never trains on real manipulated video,
a "6 forgery families" head would be supervised on a constant label — dead parameters,
exactly as in i1 (`N_FORGERY_TYPES = 7` while no manipulated video was ever seen). The
head is therefore repurposed to classify the *synthesis recipe*: `5 mask regions × 3 blend
ratio buckets = 15 always-instantiated classes`. The auxiliary task forces sensitivity to
artifact variety, which is the same mechanism DFF-Adapter uses forgery-type labels for.

---

## 5. Data, protocol, splits

### 5.1 Protocol — deliberately unchanged from i1

Training uses **real videos only** plus on-the-fly blending synthesis. **No real
manipulated video is ever used for training.** This was i1's research stance and it is
also the setting in which LAA-Net reaches 95.4 CDF-v2, so it is not the reason i1 failed.
Every test set remains fully unseen.

### 5.2 Sources (existing Kaggle pack `srn-v4b-data`)

| split | source | count | purpose |
|---|---|---|---|
| train | FF++ YouTube-c23 real / Celeb-DF Celeb-real / Celeb-DF YouTube-real | 200 + 120 + 120 | gradient updates |
| val-hard | held-out videos, video-level disjoint, hard SBI parameters | 90 | monitoring, checkpoint selection |
| **calibration** | **held-out videos + fresh SBI draws, disjoint from train and val** | **60** | **temperature + threshold fitting (new in i2)** |
| test | DFDCP 50+50, Celeb-DF-v2 50+50, FF++ 6 families ×50 | 9 sets | reported once, after freeze |

The calibration split is new and directly answers i1's finding #3 ("no recorded clean
example was classified correctly at 0.5") and its next-step #3 ("validate a global
threshold on a separate calibration split"). `assert_no_leakage()` is extended to cover
it: train, val, calibration and every test set must be disjoint at video level, and no
test video may ever be passed to the SBI generator.

### 5.3 Harder, less-shortcuttable synthesis

Changes to `make_sbi()`, each aimed at a specific i1 shortcut:

| change | i1 | i2 | why |
|---|---|---|---|
| blend ratio | `U(0.15, 1.0)` | `U(0.15, 0.60)` | ratio 1.0 replaces the whole face → global cue |
| source-side photometrics | applied to source only | **same jitter distribution applied to source and to real frames** | removes the global colour-statistics shortcut (§2.2) |
| mask edge | `GaussianBlur(2–8)` | `GaussianBlur(1.5–5)` | keeps the seam narrow, so localization must be precise |
| Poisson blending (`seamlessClone`) | 30 % of samples, tens of ms each | **off by default** (flag), numpy alpha blend | seam-free blends delete the very cue we supervise; also the main loader cost |
| mask geometry | 4 regions + ellipse fallback | 5 regions (full/upper/lower/middle/ellipse) with per-sample jitter | more artifact variety, and gives the recipe head real classes |
| frame selection | 8 fixed evenly-spaced frames | **random frames per epoch** | §7.3: 10–30× more effective data |

### 5.4 Loader rework (the 30× speedup)

1. **One-off frame cache.** Decode and resize every training/val/calibration video frame
   once into a PNG cache on `/kaggle/working`. Per-epoch cost becomes a PNG decode
   (~3–5 ms) instead of repeated `glob` + `pick_frames` + decode + resize.
2. **Numpy-only forging.** `make_sbi()` becomes array arithmetic: a precomputed soft
   polygon mask, bilinear affine for the source, one multiply-add. Target ≤3 ms/frame
   (vs i1's tens-of-ms *plus* `seamlessClone`).
3. **Random frame sampling** per epoch (`pick_frames(..., mode="random")`).
4. `num_workers = 3`, `pin_memory`, `persistent_workers`, `prefetch_factor = 4`.

Target: **≤2 s/step** (vs i1's 29.6 s), i.e. ~200 steps/epoch.

---

## 6. Training curriculum and objective

Three stages, every weight a fraction of total steps, with a 5 % linear ramp at each
boundary. i1's relative-schedule idea is kept; its four *auxiliary trust* objectives are
replaced.

| Stage | Fraction | Active |
|---|---|---|
| **S1 — locate** | 0–25 % | bag BCE + region + boundary. The model must first learn *where* the artifact is. |
| **S2 — aggregate** | 25–55 % | + MIL patch supervision, + low-level branch, + two-view consistency |
| **S3 — tighten** | 55–100 % | + alignment/uniformity, + recipe head, LR decay to floor |

| loss | target | term |
|---|---|---|
| `L_bag` | clip label, clean **and** intervention view | BCE on the 17-dim aggregator output |
| `L_region` | SBI binary region mask, mean-pooled to 28×28 (soft target) | BCE on `σ(e[p])` |
| `L_boundary` | **`B = 4·m⊙(1−m)`** from the raw soft mask, max-pooled to 28×28 | BCE on `σ(b[p])` |
| `L_mil` | real clips: every patch negative; fake clips: LSE pool positive | BCE — supplies gradient to the aggregator for real manipulations we lack masks for |
| `L_loc_dice` | region mask | soft Dice, balances the heavy real/background imbalance |
| `L_cons` | clean vs intervention evidence and boundary maps | MSE (i1's stability idea, demoted from the decision path to a regularizer) |
| `L_align` + `L_unif` | L2-normalized global features | GenD: pull same-class together, spread all features on the sphere |
| `L_sc` | same-side-of-seam as the predicted anchor patch, fake clips only | BCE |
| `L_recipe` | synthesis recipe, 15 classes, synthetic clips only | cross-entropy |

Optimizer: AdamW, `lr_peak 1.5e-4`, `wd 0.01`, 10 % warmup, cosine to `lr_floor 0.05×`,
`grad_clip 1.0`, `EMA 0.999`, `label_smoothing 0.05`, fp16 autocast. Effective batch: 2
videos × 8 frames × 2 views = 32 image-tokens per step.

---

## 7. Theoretical calculations for the 2×T4 budget

### 7.1 The dilution argument, quantified against i1

For i1's own measured numbers: mean M = 0.3336 (fake) / 0.3296 (real), and patch-level
localization is correct (pixel AUC 0.938). Take a typical forged frame at i1's mask
statistics: a landmark polygon blurred by 2–8 px covers roughly `f ≈ 0.30–0.40` of the
frame. Per the table in §2.1 the pooled contrast should then be `f·Δ_m` where `Δ_m` is the
per-patch separation. i1 measured a pooled contrast of `0.0040`, implying a per-patch
separation of only `Δ_m ≈ 0.011` at the *image* level — because the mask is huge and soft,
even the patch-level signal is weak, and the convex mean then divides it by the effective
patch count. An existence operator (`max`, `LSE`, `topk_mean`) reads the top 24 of 784
patches directly, i.e. it measures the artifact rather than the average face.

**Predicted effect of the aggregation change alone:** the bag-level contrast should rise
from `f·Δ_m` to `≈Δ_m`, a **3–25× increase** depending on mask area, with no change to the
trunk or the data. This is the single highest-expected-value change in i2 and it is
tested directly by ablation A1.

### 7.2 Compute budget

Per-image forward cost of the trunk (24 blocks, d = 1024, 789 tokens):

| component | per layer | ×24 layers |
|---|---|---|
| MLP (2 × 1024 × 4096 × 789) | 13.2 GFLOPs | 317 |
| attention QKVO (2 × 4 × 1024² × 789) | 6.6 GFLOPs | 159 |
| attention matrix (2 × 789² × 1024) | 1.3 GFLOPs | 31 |
| **total** | | **≈ 500 GFLOPs/image** |

T4 fp16 dense peak is 65 TFLOPS; a realistic 30–45 % efficiency for a memory-bound ViT-L
gives **~20–30 TFLOPS per T4, ~40–60 TFLOPS for two**. So the trunk alone is
`40e12 / 500e9 ≈` **80 images/s theoretical, 25–40 images/s realistic**.

Per training step (32 images = 2 videos × 8 frames × 2 views):

| item | cost |
|---|---|
| trunk forward, `no_grad` | 32 × 500 = 16 TFLOPs |
| head forward+backward (adapter, heads, CNN, aggregator ≈ 3 GFLOPs/img fwd) | ≈ 0.3 TFLOPs |
| **GPU time per step** | **≈ 0.8–1.2 s** |
| data per step (32 frames × ≤3 ms, 3 workers) | ≈ 0.3–0.5 s |
| **realistic step** | **≈ 1.5–2 s** |

Budget: the training set is 310 real videos and `ClipDataset` exposes `2 × 310 = 620` clip
slots, so `steps/epoch = 620 / BATCH_VIDEOS = 310` — **≈ 9–10 min/epoch**, plus a validation
pass on 40 videos every 5 epochs. `EPOCHS = 30` is therefore ≈ 4.5–5 h of training, inside the
`TIME_BUDGET_MIN = 420` min safe-stop so a Kaggle session cannot be overrun. The budget leaves
room for the both-axis final evaluation (~35 min), the figures and one ablation arm. i1 managed
5 epochs in 7.4 h; i2 targets the same wall clock for 6× the epochs.

**Staged schedule.** The first run is the full architecture (A0, no ablation) on a deliberately
short budget — `EPOCHS = 8`, `TIME_BUDGET_MIN = 150`, `RUN_NAME = "i2_evidence_v1_short"` — which
is ≈ 80 min of training, 2 480 optimiser steps (2.7× i1's *entire* 900-step run) and still
reaches every curriculum stage. It is a **pilot**: the cross-dataset gates are expected to fail on
that schedule and it must not be written up as a `Pass`/`Fail` record. The full run then uses
`EPOCHS = 30`, `TIME_BUDGET_MIN = 420`, `RUN_NAME = "i2_evidence_v1"`. `CACHE_DIR` is keyed on the
split rather than the run name, so both schedules share one frame cache.

### 7.3 Supervision volume

| | i1 | i2 | factor |
|---|---|---|---|
| effective real+forged frame-passes | 7 200 | ~148 800 (30 epochs × 620 clips × 8 frames) | **20.7×** |
| fresh SBI draws per epoch | 0 (fixed per sample) | yes, regenerated | — |
| per-patch supervision points | 0 (region BCE only, one soft label per cell) | 784 per frame × dense region + boundary + MIL | — |
| distinct labels | clip label + coarse mask | clip + region + boundary + recipe + consistency | — |

### 7.4 Memory

Trunk runs under `no_grad`, so no activation graph is retained. Retained per step: token
output (32 × 789 × 1024 × 2 B = 52 MB), head graph (a 2-block transformer over
32 × 784 × 384 fp32 ≈ 39 MB per saved tensor, order 10–20 tensors → < 1 GB), plus the
CNN branch. **Peak < 6 GB per T4**, comfortably inside 16 GB. `BATCH_VIDEOS = 2` with
`DataParallel` over 2 T4s, matching i1's working regime.

*If stage-2 LN-tuning is enabled* the trunk graph must be kept: ~18 GB of activations for
32 images at 392², which does **not** fit a T4. Stage 2 is therefore off by default and,
when enabled, must use gradient checkpointing (try `gradient_checkpointing_enable` with
`use_reentrant=False`) or `TRUNK_MICRO` micro-batching, at ≥30 % time cost. This is
documented as a deliberate default rather than an oversight.

### 7.5 Parameters

| part | params | trainable |
|---|---|---|
| DINOv2-L trunk | 300 M | no (LN-tuning flag) |
| forensic adapter (1024→384 + 2 blocks) | ~4.0 M | yes |
| global path (1024→384) | 0.4 M | yes |
| patch heads ×3 (+ recipe, aggregator) | ~0.6 M | yes |
| low-level residual CNN | ~0.5 M | yes |
| **total trainable** | **≈ 5.5 M** | |

i1's document claimed "~350 M (backbone frozen, adapter + experts + router trainable)";
the actual i1 trainable count was ≈7–8 M. Reporting both correctly matters, because the
300 M frozen trunk is not free parameters — it is a fixed feature extractor.

---

## 8. Evaluation plan, gates and acceptance

### 8.1 Two test axes — within-dataset and cross-dataset

Both axes are required, and they answer different questions. Reporting only one of them is how a
"generalizable" detector gets accidentally claimed.

| axis | real side | fake side | question it answers |
|---|---|---|---|
| **within-dataset** | 50 held-out FF++ real videos, never trained on, training photometric draw | self-blended fakes generated from **those same 50 videos** with the training synthesis recipe (ratio 0.15–0.60, 5 mask regions, same cross-frame-source rate) | was the task learnt at all, with no domain shift? |
| **within-dataset — real manipulation** | the same 50 held-out real videos, **raw** | the pack's own FF++ Deepfakes / Face2Face / FaceShifter / FaceSwap / NeuralTextures / DeepFakeDetection | does a synthetic seam transfer to a real manipulation of the same corpus? |
| **cross-dataset** | DFDCP and Celeb-DF-v2 real, **raw** | DFDCP and Celeb-DF-v2 fake, **raw** | does it survive a different corpus and a different generator? |

Pairing real and fake from the **same source video** is deliberate: GenD's second conclusion is
that same-source pairing is what removes shortcut learning, so the within-dataset test is built
that way. Because the real side would otherwise differ from the fake side only by the photometric
draw that `make_sbi` applies internally, the real side receives an independent draw from the same
distribution — the evaluation reproduces the training pixel pipeline on both sides. The
real-manipulation sub-table instead uses **raw** frames on both sides, so no jitter-versus-raw
artefact is introduced by the evaluation itself. Cross-dataset uses raw frames on both sides,
which is the published protocol and what the numbers in §3 are comparable to.

### 8.2 Metrics (per `record/rules.md` §4)

- Video-level **AUROC** with bootstrap 95 % CI (2000 resamples) — primary, comparable to
  the literature table in §3.
- **AP, EER, thresholded accuracy / recall / F1 / balanced accuracy**, reported at
  **both** the default 0.5 threshold **and** the calibrated threshold, always separated
  (i1's record was right to insist on this).
- **Calibration:** ECE and Brier before/after temperature scaling on the held-out
  calibration split.
- **Localization:** pixel AUC, IoU, mask F1, boundary-map AUC.
- **Robustness:** clean / JPEG-30 / blur-15 / resize-50, reported separately for the
  within-dataset pair (FF++ real vs SBI), DFDCP and Celeb-DF-v2.
- `worst_auc` across conditions is the **selection metric**, as in i1.

### 8.3 Reporting rules (the i1 gap that must be closed)

1. The final evaluation cell **must produce outputs on both axes**: the within-dataset pair
   (SBI built from the held-out videos, plus the 6 real FF++ families) and the cross-dataset
   pair (DFDCP, Celeb-DF-v2), each with `n_real`, `n_fake`, CIs and every robustness
   condition. A record may not be written from validation numbers alone again.
2. Test conditions are read **once**, after `checkpoint_best.pt` is frozen; nothing
   downstream may change a hyper-parameter or re-select a checkpoint.
3. `manifest_hash` must match between checkpoint and evaluation, so results cannot be
   reported against a different split.

### 8.4 Acceptance gates (Pass/Fail)

| gate | threshold | rationale |
|---|---|---|
| **sanity — within-dataset AUROC** | **≥ 0.90** | No domain shift. A detector that cannot pass this has not learnt the task, whatever the cross-dataset number says. i1's in-domain number was inflated by the ratio→1.0 shortcut, so this gate sits on the *hard* recipe (ratio ≤ 0.60). |
| **primary — DFDCP AUROC** | **≥ 0.80** | Face X-ray reaches 0.809 with the same core idea (blending-boundary supervision); a method that claims to do this well should at least match it. |
| Celeb-DF-v2 AUROC | ≥ 0.85 | SBI 0.932, F3-Net 0.789. |
| worst robustness condition | ≥ 0.75 | i1: 0.578. |
| train vs eval gap | < 0.25 | i1: 0.42. |
| within-minus-cross gap | ≤ 0.30 | how much in-domain skill survives the domain shift; the honest cost of generalisation. |
| threshold usable at 0.5 | not all-one-class | i1 failed this outright. |
| localization | pixel AUC ≥ 0.90 and IoU ≥ 0.55 | i1: 0.938 / 0.600 — must not regress. |

`Pass` requires the primary detection gate. Localization quality cannot rescue a failed
detector (i1's record already states this rule, correctly).

**Honest expectation.** Published cross-dataset numbers in §3 use FF++ with 720 real
videos plus real manipulated training data, larger backbones and much bigger budgets. i2
has ~440 real videos, synthetic-only fakes, a frozen ViT-L and one overnight run on two
T4s. Tiered targets: *minimum* DFDCP 0.80; *expected* 0.82–0.88 on DFDCP and 0.85–0.92 on
Celeb-DF-v2; *stretch* > 0.90. The primary comparison is nevertheless i1 (0.578, chance
level), not the literature.

---

## 9. Ablation grid

Each stage is a single flag, so an overnight run can cover the primary config plus one or
two arms. **A0 is the only arm run first** (full architecture, short schedule — see §7.2); every
other row below is a follow-up, in the priority order listed. A single-flag ablation isolates one
variable but cannot see interactions, so an absent effect is not proof of an absent effect.

| id | change vs primary | question |
|---|---|---|
| **A0** | primary config | does the full i2 design clear the gates? |
| **A1** | aggregation → i1's convex mean | **direct test of §2.1**: is the pooling fix sufficient? Highest value. |
| **A2** | boundary loss off (region only) | is the Face X-ray/LAA-Net boundary target load-bearing? |
| **A3** | low-level residual branch off | do sub-patch cues add anything over the trunk? |
| **A4** | two-view consistency off | is the intervention view worth 2× trunk cost? |
| **A5** | recipe head off | does the auxiliary recipe task help? |
| **A6** | ratio `U(0.15,1.0)` + exclusive source jitter (= i1 synthesis) | how much of i1's in-domain inflation was the shortcut? |
| **A7** | LN-tuning stage 2 on | does trunk adaptation add over a frozen trunk, per GenD? |
| **A8** | alignment/uniformity off | GenD's contribution on this data size |

---

## 10. Assumptions, risks, changes from the previous iteration

### Changes from i1

| # | i1 | i2 | evidence |
|---|---|---|---|
| 1 | convex-average pooling of patch features | existence aggregation (top-k / LSE / max / mean / std) | §2.1 |
| 2 | rank-8 linear bottleneck mislabelled "LoRA" | LN-tuning available as a flag-gated stage 2; frozen trunk by default | §2.3, §3.1(a) |
| 3 | region-mask supervision only | + boundary target `4m(1−m)` | §3.1(c) |
| 4 | token-level prototype router, top-2 of 5 | **removed** (falsified: entropy 1.606 ≈ ln 5) | §2.4 |
| 5 | trust weighting M×S×R on the decision path | demoted to a two-view consistency regularizer | §2.1 |
| 6 | 0.5 threshold; calibration on the selection split | held-out calibration split, temperature scaling, both thresholds reported | §2.5 |
| 7 | cross-dataset evaluation never executed | evaluation is a required, output-producing cell, run on **both** axes (within-dataset and cross-dataset) | §2.5, §8.1 |
| 8 | 30× data-pipeline bound | frame cache + numpy forging + random frames, target ≤2 s/step | §2.6, §5.4 |
| 9 | 8 fixed frames/video | random frames per epoch | §7.3 |
| 10 | forgery head supervised on a constant label | recipe head, 15 real classes | §4.2 |
| 11 | accuracy claims in the design doc not marked as unvalidated | this document separates *reported by others* (§3) from *predicted* (§7–8) | i1 doc §"94–97 % AUC" |

### Assumptions

- The Kaggle pack layout (`frames/` + `landmarks/`, `manipulated_sequences/*/c23`) is
  stable and matches what i1 resolved; discovery is by layout sniffing, as in i1.
- DINOv2-L/14 weights are available in the Kaggle session (they were for i1).
- A 9-hour GPU session is available; the notebook prints a per-epoch ETA and a "safe stop"
  line so the run can end cleanly before the limit.
- Two T4s with `DataParallel` behave as in i1.

### Risks, ranked

1. **Existence aggregation over-triggers** (a single false-positive patch drives the bag
   logit). Mitigated by keeping `mean`/`std` in the aggregator feature vector and by
   threshold calibration; LAA-Net's heatmap supervision is the structural counterweight.
2. **Frozen trunk feature quality.** GenD's linear probe scored only 76.0 mean. Our
   adapter is far more expressive than a linear probe, but if A7 (LN-tuning) shows a large
   gap, stage 2 becomes the priority for i3, with gradient checkpointing.
3. **Still no real manipulated training data.** Cross-dataset numbers will be limited by
   this. That is the point of the experiment, but it is the most likely reason to fall
   short of the stretch targets. An arm training on 4 FF++ families and holding out
   FaceShifter + DeepFakeDetection is the natural i3 follow-up.
4. **Seam supervision on synthetic blends only.** Real manipulations may not leave a
   `4m(1−m)`-like seam (GAN-inpainting and face-reenactment seams differ from a
   copy-paste affine blend). Boundary AUC is therefore also measured on real test data
   where no GT exists, as an indirect check via detection performance.
5. **Stage-2 LN-tuning memory.** Documented in §7.4; default OFF for safety.
6. **Loader still not fast enough.** If measured throughput is >5 s/step the run must
   reduce `T_FRAMES` to 6 or `FRAMES_PER_EPOCH`, both exposed as flags.

---

## 11. References

1. Shiohara, Yamasaki. *Detecting Deepfakes with Self-Blended Images.* CVPR 2022.
2. Li et al. *Face X-ray for More General Face Forgery Detection.* CVPR 2020. arXiv:1912.13458.
3. Nguyen et al. *LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection.* CVPR 2024. arXiv:2401.13856.
4. Yermakov, Cech, Matas, Fritz. *Deepfake Detection that Generalizes Across Benchmarks* (GenD). arXiv:2508.06248.
5. Zhang et al. *Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection* (DFF-Adapter). AAAI 2026. arXiv:2511.12107.
6. Tan et al. *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection* (NPR). CVPR 2024. arXiv:2312.10461.
7. Qian et al. *Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues* (F3-Net). ECCV 2020.
8. Liu et al. *Spatial-Phase Shallow Learning.* CVPR 2021.
9. Cao et al. *End-to-End Reconstruction-Classification Learning for Face Forgery Detection* (RECCE). CVPR 2022.
10. Rossler et al. *FaceForensics++.* ICCV 2019.
11. Wang & Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020.
12. Wang et al. *LogAvgExp Provides a Principled and Performant Global Pooling Operator.* arXiv:2111.01742.
13. Wang et al. *A Comparison of Five Multiple Instance Learning Pooling Functions.* ICASSP 2019.
14. Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
15. Yan et al. *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection.* NeurIPS 2023 (protocols/preprocessing).

Numbers for ForAda, Effort, GenD, LAA-Net (CDF-v2/DFD/DFDC) are quoted from the
comparison table in [4]; the Xception/F3-Net/FaceX-ray/SPSL/RECCE DFDCP column is quoted
from [5]. All are marked "reported by others" and none of them were reproduced here.
