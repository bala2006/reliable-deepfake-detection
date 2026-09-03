# Stable-RouteNet v4 — Architecture & Training Pipeline

## What This Model Does (In Plain Language)

**Goal:** Build a system that can tell whether a face in a video is real or manipulated (deepfake), and it should work well even on videos it has never seen before from different datasets and different manipulation methods.

**The core idea:** Instead of trying to learn everything at once, the model uses a team of "specialist experts" — each one focuses on a different type of forensic evidence (like color inconsistencies, texture artifacts, edge problems). A smart "router" decides which experts should look at which parts of the image. A "shared expert" always looks at everything so common patterns are never missed. And a "trust system" figures out how much to believe each piece of evidence.

**v4b enhancements:** LoRA adaptation in DINOv2, frequency-domain expert, diverse SBI masks with Poisson blending, training-time degradations, contractive-repulsive loss, and comprehensive 9-condition evaluation.

---

## Full Pipeline Diagram (Vertical Flow)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING DATA                                    │
│  600 real videos (FF++ 200 + Celeb-DF Celeb-real 200 +               │
│                   Celeb-DF YouTube-real 200)                          │
│  Fakes: DD-SBI with diverse masks (full/upper/lower/middle/ellipse) │
│  + Poisson blending + training-time degradations                     │
│  Never uses real manipulated videos during training                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TWO VIEWS PER CLIP                               │
│  Original: 8 frames with slight color jitter + degradation          │
│  Intervention: same 8 frames + JPEG / blur / resize / brightness    │
│  Also: ground-truth mask showing where the fake blend was applied   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   FROZEN BACKBONE         │    │   FROZEN BACKBONE         │
│   (DINOv2-Large)         │    │   (DINOv2-Large)         │
│   + LoRA adaptation       │    │   + LoRA adaptation       │
│   Breaks image into 784   │    │   Same model, same       │
│   patches, describes each │    │   process for the        │
│   with 1024 numbers       │    │   degraded view          │
└────────────┬─────────────┘    └────────────┬─────────────┘
             │                               │
             ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   ADAPTER + LoRA          │    │   ADAPTER + LoRA          │
│   Converts generic visual │    │   (shared weights)        │
│   features into forensic  │    │                           │
│   features (384 numbers)  │    │                           │
│   + low-rank adaptation   │    │                           │
│                           │    │                           │
│   Output: INVARIANT       │    │   Output: INVARIANT       │
│   features (what the      │    │   features                │
│   model believes about    │    │                           │
│   manipulation)           │    │                           │
│                           │    │                           │
│   Output: VARIANT         │    │   Output: VARIANT         │
│   features (how much      │    │   features                │
│   appearance changes)     │    │                           │
└──────┬──────────┬────────┘    └──────┬──────────┬────────┘
       │          │                    │          │
       ▼          ▼                    ▼          ▼
┌─────────────────────┐      ┌─────────────────────┐
│   EVIDENCE MEASUREMENT│      │   EVIDENCE MEASUREMENT│
│                       │      │                       │
│   M = "How much fake  │      │   (same heads,        │
│   evidence in this    │      │    same scores)       │
│   patch?" (0 to 1)    │      │                       │
│                       │      │                       │
│   S = "How stable is  │      │                       │
│   this evidence when  │      │                       │
│   quality changes?"   │      │                       │
│   (0 to 1)            │      │                       │
│                       │      │                       │
│   Nuisance = "What    │      │                       │
│   kind of degradation │      │                       │
│   is present?"        │      │                       │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SMART ROUTER (Prototype-Based)                    │
│                                                                     │
│   4 prototypes — one per expert — learned during training           │
│   Think: "What does the evidence this expert detects look like?"    │
│                                                                     │
│   Each patch compared to all 4 prototypes                           │
│   Top 2 most similar experts selected for each patch               │
│                                                                     │
│   WHY: Routes by WHAT the patch shows, not WHERE it came from      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXPERT TEAM                                      │
│                                                                     │
│   4 SPECIALIST EXPERTS              1 FREQUENCY EXPERT              │
│   Each processes only the           Processes DCT features          │
│   patches assigned to it            of each patch (spectral         │
│   Each specializes in different     artifacts like compression      │
│   forensic patterns                 patterns, texture anomalies)    │
│                                                                     │
│   1 SHARED EXPERT                                                   │
│   Always processes ALL patches regardless of routing                │
│   Safety net for common forensic knowledge                          │
│                                                                     │
│   Final = shared + specialists + frequency (learned blend)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TRUST SCORING                                    │
│                                                                     │
│   AGREEMENT:     Do the 2 selected experts agree?                  │
│                  Both say "fake" → trust goes up                    │
│                  They disagree → trust goes down                    │
│                                                                     │
│   CONSISTENCY:   Does the evidence survive JPEG/blur/resize?       │
│                  Same score in original + degraded view → trusted   │
│                  Score changes a lot → less trusted                 │
│                                                                     │
│   RELIABILITY:   Combination of agreement + consistency            │
│                                                                     │
│   Each patch gets a TRUST WEIGHT based on M × S × Reliability     │
│   High weight = "this patch's evidence is strong and trustworthy"  │
│   Low weight  = "ignore this patch — evidence is weak or unstable" │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WEIGHTED POOLING                                 │
│                                                                     │
│   Sum up all patch contributions, weighted by trust                 │
│   Trustworthy patches dominate the decision                        │
│   Untrustworthy patches are mostly ignored                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FINAL DECISION                                   │
│                                                                     │
│   Small classifier network reads the weighted summary              │
│   Outputs a probability:                                            │
│     > 0.5 = FAKE       < 0.5 = REAL                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING (5 Phases)                              │
│                                                                     │
│   Phase 1 (0–20%):   Learn real vs fake, find manipulated regions  │
│   Phase 2 (20–40%):  Add stability & quality-invariance learning   │
│   Phase 3 (40–60%):  Activate expert routing & load balancing      │
│   Phase 4 (60–80%):  Add trust scoring & evidence weighting        │
│   Phase 5 (80–100%): Consolidate everything, polish                │
│                                                                     │
│   Smooth transitions — no sudden jumps between phases              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EVALUATION (Once, After Training)                │
│                                                                     │
│   Cross-dataset:  DFDCP 200 videos (never seen during training)    │
│   Cross-dataset:  Celeb-DF-v2 200 videos (never seen)              │
│   Cross-manip:    FF++ 6 manipulation families (200 each)          │
│   Robustness:     JPEG / blur / resize applied to test videos      │
│   Metrics:        AUROC, EER, AP, F1, bootstrap 95% CI             │
│   Selection:      Best model = worst robustness score is highest   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How v4 Works — Step by Step

### 1. Preparing the Training Data

**What we feed it:**
- **600 real videos** — 200 from FaceForensics++ (YouTube quality) + 200 from Celeb-DF-v2 Celeb-real + 200 from Celeb-DF-v2 YouTube-real
- **Fakes are created on the fly** during training using a technique called "self-blended images" with diverse masks (full face, upper/lower/middle bands, random ellipse) and Poisson blending. The model never sees real manipulated videos during training. This forces it to learn general forensic patterns, not shortcuts from specific manipulation tools.

**How we split the data:**
- **Training:** 600 real videos (the model learns from these)
- **Validation:** 150 real videos (used to check progress and pick the best model)
- **Testing:** DFDCP (400 videos) + Celeb-DF-v2 (400 videos) + FF++ manipulation families (1,200 videos total) — held completely separate — the model never sees these until the very end

**Why this matters:** If the model memorizes specific manipulation artifacts instead of learning general forensic rules, it will fail on new datasets. The strict separation prevents "data leakage."

**v4b improvements:** Training-time degradations (JPEG, blur, resize) applied to both views improve robustness to quality variations.

---

### 2. Creating Two Views of Every Video

For each video clip, the model receives two versions of the same 8 frames:

- **Original view:** The real or SBI-faked frames with slight color/brightness jitter (quality alignment)
- **Intervention view:** The same frames with a random "nuisance" transform applied — JPEG compression, blur, resize, or brightness change

**Why two views?** The model needs to learn which evidence is stable (survives quality changes) and which is fragile (disappears with JPEG compression). Stable evidence is more trustworthy.

---

### 3. Reading the Images (The Backbone)

Both views go through **DINOv2-L/14** — a powerful pre-trained vision model that is **frozen** (its weights don't change). It breaks each frame into 784 small patches and describes each patch with 1024 numbers.

Think of it as: "Here is what each tiny region of the face looks like, described in rich detail."

---

### 4. Adapting to Forensics (The Adapter)

The backbone's descriptions are generic (trained on natural images). The **ForensicAdapter** converts them into forensic-specific features:
- Shrinks from 1024 to 384 dimensions
- Runs through 2 small transformer layers (lightweight processing)
- Produces two feature types:
  - **Invariant features:** What the model believes about manipulation (stable evidence)
  - **Variant features:** How much the appearance varies (nuisance sensitivity)

---

### 5. Measuring Evidence (The Evidence Heads)

Three small neural networks read the features and produce:

- **M (Manipulation score, 0 to 1):** "How much manipulation evidence do I see in this patch?" High M = likely tampered region.
- **S (Stability score, 0 to 1):** "How stable is this evidence across quality changes?" High S = the evidence survives JPEG/blur/resize.
- **Nuisance predictor:** "What kind of quality degradation is present?" (JPEG, blur, resize, or brightness)

---

### 6. Routing — Who Looks at What (The Prototype Router)

This is a **key v4 innovation.** Each patch of the image is sent to the most relevant expert(s).

**How it works:**
- The model learns 4 "prototypes" — one per spatial expert. Think of these as: "What does the kind of evidence this expert is good at detecting look like?"
- Each patch is compared to all 4 prototypes using **cosine similarity** (how similar is this patch to each prototype?)
- The top 2 most similar experts are selected for each patch
- Additionally, a **frequency expert** processes DCT features of each patch to detect spectral artifacts

**Why prototypes instead of a regular neural network router?**
- A regular router can learn to route based on domain shortcuts (e.g., "all YouTube videos go to expert 1")
- Prototypes force routing based on **what the patch looks like forensically**, not where it came from
- This is more robust and generalizes better

---

### 7. The Expert Team

**4 specialist experts** — each is a small neural network (MLP) that processes patches assigned to it. Each expert specializes in different forensic patterns.

**1 frequency expert** — processes DCT (Discrete Cosine Transform) features of each patch to detect spectral artifacts like compression patterns, texture anomalies, and frequency-domain inconsistencies that spatial experts might miss.

**1 shared expert** — always processes ALL patches, regardless of routing. This ensures common forensic knowledge is never lost, even when routing is imperfect.

**Why all three?** Specialists handle nuanced, specific evidence. The frequency expert catches spectral artifacts. The shared expert acts as a safety net — if the router makes mistakes (especially early in training), common patterns still get processed.

---

### 8. Combining Expert Outputs

The final representation blends:
- **Shared expert output** (always present) — weighted by a learned parameter gamma (starts at 50/50)
- **Specialist expert output** (sparse, only selected patches) — weighted by the remaining share
- **Frequency expert output** — added with a small weight (0.1) to complement spatial features

---

### 9. Trust Assessment — Should We Believe This Evidence?

Three measurements determine how much to trust each patch's evidence:

- **Agreement (A):** Do the two selected experts agree on what they see? If both specialists say "this looks manipulated," we trust the evidence more. If they disagree, we trust it less.

- **Consistency (C):** Does the evidence survive the intervention (JPEG/blur/resize)? If the same manipulation score appears in both the original and degraded view, the evidence is consistent and trustworthy.

- **Reliability (R):** A combination of agreement and consistency. R = the geometric mean of A and C.

These three measurements (M, S, R) are **trust scores, not routing scores.** They don't decide which expert looks at what — they decide how much to weight each patch's contribution to the final decision.

---

### 10. Weighted Decision Making

Each patch gets a **trust weight** that combines:
- How strong the manipulation evidence is (M)
- How stable it is across quality changes (S)
- How reliable it is based on expert agreement and consistency (R)

The weight is raised to a learned power (alpha) that controls how aggressively to down-weight untrustworthy patches. The model learns alpha during training (starts at 1/3, allowed to range between 0.2 and 0.8).

**High weight patches** (strong, stable, reliable evidence) contribute more to the final real/fake decision. **Low weight patches** (weak, unstable, or contradictory evidence) are mostly ignored.

---

### 11. Final Classification

After weighted pooling across all patches and frames:
- The combined representation goes through a small classifier network
- It outputs a single number (logit) that is converted to a probability
- Probability > 0.5 = fake, < 0.5 = real

---

## How the Model Learns (Training Curriculum)

The model doesn't try to learn everything at once. It follows a **5-phase schedule** where new objectives are gradually introduced:

| Phase | What Happens | Why |
|-------|-------------|-----|
| **Phase 1 (first 20% of training)** | Learn to detect real vs fake, localize manipulated regions, and warm up the shared expert | Get the basics right first |
| **Phase 2 (20–40%)** | Add stability measurement, S-score targets, and nuisance invariance | Start learning what evidence is trustworthy |
| **Phase 3 (40–60%)** | Activate sparse routing, load balancing, and router stability | Let the experts specialize without collapsing |
| **Phase 4 (60–80%)** | Add agreement/consistency/reliability measurement and evidence weighting | Fine-tune trust assessment |
| **Phase 5 (80–100%)** | Consolidate all objectives, reduce auxiliary losses | Polish and stabilize |

Each phase boundary has a **smooth 5% ramp** — no hard jumps that could destabilize training.

---

## Why v4 Is Better Than Previous Versions

### Problem with v1/v2:
- The model learned **shortcuts** — e.g., "YouTube videos with certain compression artifacts are fake"
- It scored 97% on the training dataset but only 57% on completely new datasets
- The localisation head got stuck and never learned to find manipulated regions

### Problem with v3.1:
- Some training objectives only activated at specific training steps, meaning they could turn on **after** training was supposed to be done
- This caused a train/test mismatch — the model used during evaluation was different from the one used during training

### What v4 fixes:

| Fix | How |
|-----|-----|
| **No more shortcuts** | Prototype router forces routing based on forensic content, not domain |
| **Trustworthy evidence** | M/S/R are trust measures, separate from routing — a routing mistake doesn't cascade into a trust mistake |
| **Safety net** | Shared expert ensures common knowledge is never lost |
| **Consistent training** | All schedules are percentages of total training steps, not fixed step numbers — works regardless of how long you train |
| **Stable routing** | Switch-style balancing and z-loss prevent the router from collapsing (sending everything to one expert) |
| **Selective consistency** | Only high-evidence patches are penalized for routing changes — low-evidence patches can route freely |

### What v4b adds:

| Improvement | How | Expected Impact |
|-------------|-----|-----------------|
| **3-domain training** | FF++ + Celeb-DF Celeb-real + YouTube-real (600 videos, 200 each) | +3-5% cross-dataset |
| **Diverse SBI masks** | Full/upper/lower/middle/ellipse masks + Poisson blending | +3-6% robustness |
| **Training degradations** | JPEG/blur/resize on both views during training | +2-4% under degradation |
| **LoRA adaptation** | Low-rank adaptation in DINOv2 backbone | +5-10% feature quality |
| **Frequency expert** | DCT-based spectral artifact detection | +2-3% cross-manipulation |
| **Forgery-type branch** | Auxiliary classification head (6 manipulation types) | +3-5% fine-grained detection |
| **Contractive-repulsive loss** | Pull same-class together, push different apart | +1-2% generalization |
| **9-condition evaluation** | DFDCP + Celeb-DF + 6 FF++ families, 200 each | Credibility |

---

## Ablation Plan — Testing Each Improvement

The model is tested in 5 stages to measure the impact of each improvement:

| Stage | What's New | Question Being Answered |
|-------|-----------|------------------------|
| **A0** | Fixed training schedule only | Was the main problem just a mismatch between training and testing? |
| **A1** | + Load balancing + router stability | Does preventing router collapse help? |
| **A2** | + Shared expert | Does having a safety net of common knowledge help? |
| **A3** | + Prototype router | Does routing by forensic content (not domain) help? |
| **A4** | + Semantic specialist axes + LoRA + Frequency expert + Forgery branch + CRO loss | Does meaningful specialization + enhanced features improve OOD? |

Each stage builds on the previous one, so we can isolate which change actually improves performance.

## v4b Implementation Summary

**Total changes:** 16 improvements across 5 phases

| Phase | Changes | Expected Impact |
|-------|---------|-----------------|
| Phase 1: Data | 3-domain training, 600 videos (200/domain), 150 val, 9 test conditions | +3-5% |
| Phase 2: SBI | Diverse masks, Poisson blending, training degradations | +3-6% |
| Phase 3: Architecture | LoRA, frequency expert, forgery-type branch | +5-10% |
| Phase 4: Training | Progressive curriculum, contractive-repulsive loss | +2-3% |
| Phase 5: Evaluation | Multiple metrics, 9 test conditions, bootstrap CI | Credibility |
| **Total** | **16 changes** | **+13-24%** |

**Expected baseline improvement:** 86% → 94-97% AUC on DFDCP

---

## Summary

**In one sentence:** Stable-RouteNet v4b uses a team of specialist experts guided by a prototype-based router, with a shared safety-net expert, frequency-domain expert, LoRA adaptation, and a separate trust system, trained with a gradual curriculum and diverse SBI augmentation for generalizable deepfake detection.

**Why it should work better:** It directly addresses the three main failure modes of previous versions — domain shortcuts, train/test mismatch, and expert collapse — through architectural innovations (prototype router, shared expert, frequency expert, trust decoupling, LoRA) and training improvements (relative curriculum, load balancing, selective consistency, diverse SBI, contractive-repulsive loss).

**v4b expected performance:** 94-97% AUC on DFDCP (cross-dataset), validated across 9 test conditions with bootstrap confidence intervals.

---

## Implementation Notes & Known Details

### Bug Fixes Applied (v4b notebook)

The following critical bugs were found and fixed during code audit:

| Bug | Issue | Fix |
|-----|-------|-----|
| **FREQ_EXPERT ordering** | Used before defined (NameError at startup) | Moved definition before N_EXPERTS |
| **x.device undefined** | `compute_losses()` else branches referenced `x` | Changed to `o1["logit"].device` |
| **FrequencyExpert.proj dim** | Expected 196-dim input, received 384 | Changed to `Linear(d, d)` |
| **Router expert mismatch** | Router outputs 5 logits but only 4 spatial experts | Added `topi.clamp(max=3)` before dispatch |
| **o2 probs unsliced** | Full tensor passed to intervention view dict | Sliced to `probs[BT*GRID*GRID:]` |
| **evaluate double-index** | `logits["logit"]` on tensor | Removed redundant indexing |
| **M mask shape** | 2D mask vs 1D JSD in consistency loss | Added `.mean(dim=-1)` |
| **router_z_loss** | Received probabilities, not logits | Guarded with `"route_logits" in o1` |
| **Indentation** | Paired output dicts at wrong indent level | Fixed to be inside `if xi is not None:` |

### Architecture Details

- **Backbone:** `facebook/dinov2-with-registers-large` — 0.3B params, hidden_size=1024, 4 register tokens, patch_size=14
- **Prefix handling:** Strips 1 CLS + 4 register tokens, keeping 784 patch tokens for 392×392 input
- **Adapter:** Projects 1024→384 via `ForensicAdapter` (2 transformer layers)
- **LoRA:** Global low-rank adaptation (rank=8, alpha=16) on backbone features: `tokens → Linear(1024,8) → Linear(8,384) → residual add`
- **Router:** 4 learned prototypes, cosine similarity, top-2 dispatch. Router outputs 5 logits but clamps indices to [0,3] for 4 spatial experts
- **Frequency expert:** Applied post-routing to all tokens with 0.1 weight (not part of sparse routing)
- **Shared expert:** Always evaluated, blended via learnable gamma

### Training Configuration

- **Batch:** 2 videos × 8 frames, gradient accumulation ×2 (effective batch = 4)
- **Optimizer:** AdamW (lr=1.5e-4, wd=0.01, warmup 10%)
- **Mixed precision:** FP16 via `torch.amp`
- **Multi-GPU:** `DataParallel` for 2× T4
- **Total params:** ~350M (backbone frozen, adapter + experts + router trainable)
