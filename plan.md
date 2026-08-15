# RL-ROI-Net v2 — Publication Plan

Goal: publish an **honest, reproducible deepfake forensic triage paper**.
The contribution framing remains a hypothesis pending validation; no novelty count is asserted.

Status baseline (2026-08-15, from retained artifacts):

| Item | Value | Source |
|---|---|---|
| Production checkpoint | `outputs/checkpoints/best.pt` (frozen honi05 EfficientNet-B4 + head-only classifier + 4-ROI region head) | README/progress |
| BEST report headline | ACC 0.6965, AUC 0.8104, ECE 0.2171, regionIoU 0.5576 | `outputs/checkpoints/metrics.json` |
| BEST report unit | **603 face samples**, under a key named `unseen_videos`; `video_level` is `false` | `outputs/checkpoints/metrics.json` |
| Official-pair production head | Video AUC 0.5592 on 142 videos; not a release result | `outputs/experiments/spatial_adapter_official_split_5epoch/best_official_eval.json` |
| Same-partition diagnostic | Production head per-family video AUC: Deepfakes 0.5488, Face2Face 0.5833, FaceSwap 0.5310, NeuralTextures 0.5816; overall 0.5615 | `outputs/diagnostics/faceswap_inversion_diagnostic_retry.json` |
| Frozen / adapter diagnostic | Frozen honi05 overall video AUC 0.5377; spatial adapter overall video AUC 0.6274 | `outputs/diagnostics/faceswap_inversion_diagnostic_retry.json` |
| Unattributable 0.99 result | **DO NOT CITE** — `verdict_shootout.json` records ACC 0.9450 / AUC 0.9903 for 200 videos but no dataset, split, disjointness, or preprocessing; `eval_report.json` is likewise unattributed | `outputs/verdict_shootout.json`, `outputs/eval_report.json` |
| External data capacity | 605 source-video identities available: Celeb-DF-v2 518 + DFDCP 87; DFDCP's 777 test records collapse to 87 source groups | retained data audit / manifests |
| Compute | Single RTX 4050 (8 GB), Docker CUDA, torch 2.5.1, AMP | config/README |

---

## 1. Paper concept (the story)

**Title (working):** *"Don't Guess, Escalate: A Deepfake Evidence-Review System with Fixed-ROI Localization Evidence"*

The paper is a **forensic triage system**: the deployed path keeps the honi05 verdict
backbone frozen and trains small classification/region heads. The fixed landmark-defined
ROIs provide review evidence; they do not establish causal attribution. The output is a
three-way decision — REAL / FAKE / **REVIEW** — where REVIEW is a **mandatory human escalation
contract**. Calibration quality is not established: the retained BEST report has ECE 0.2171.
No FLOPs/VRAM comparison, chain-of-custody claim, or reusable-any-detector study is established.

Target venue (see §7): a **media-forensics workshop** (CVPRW/ICCVW Media Forensics
(WMF), IEEE SPS conferences) or an **open preprint**. Do not aim for a main-conference
SOTA paper with current numbers.

---

## 2. Contribution hypotheses (not established novelty claims)

Each item below is a framing hypothesis to validate, not a measured novelty claim. The current evidence does not support a fixed novelty count.

1. **"Say 'I'm not sure' on purpose."** The system has a real third answer, REVIEW,
   that *forces a person to look*. It is not a fallback classifier and it is not a
   soft binary score. → Technical: implemented three-way decision policy with a REVIEW
   band and mandatory escalation; calibration quality is unverified rather than established
   (BEST ECE 0.2171).

2. **"Point at review evidence, don't just guess."** Instead of a bare yes/no, the tool
   renders fixed landmark-box ROIs — around the eyes, jawline, mouth, or hairline — on
   actual frames. → Technical: mask-supervised region heads over four fixed ROIs
   (PERIOCULAR, JAWLINE, MOUTH, HAIRLINE) with region IoU/hit metrics. These boxes
   are localization evidence, not causal attribution; external localization is unsupported.

3. **"Lock the verdict backbone, train the evidence heads."** The deployed path freezes
   the honi05 verdict backbone while trainable heads are added. → Technical: frozen-verdict
   head-only training with Focal + Dice/BCE + per-ROI BCE losses. No FLOPs/VRAM comparison
   or reusable-any-detector study has been measured; the alternate ImageNet EfficientNet-B4
   path also exists in `train.py` without `--verdict`.

4. **"One private, local evidence desk."** Everything runs on one machine; the video
   never leaves the machine, the flagged frames and reasoning are saved as a report,
   and the evidence can be wiped after review (24 h retention, DELETE endpoint). →
   Technical: self-contained FastAPI/Docker evidence-review workflow with local report
   and deletion behavior. This is not a chain-of-custody claim.

5. **"Built-in honesty meter."** It reports stratified per-method accuracy, region
   IoU, calibration (ECE), and *confidently-wrong* rates (confident false negatives,
   per-method AUC) instead of one headline number. → Technical: evaluation protocol
   reporting ECE, per-method AUC, regionIoU/hit, and confident-error rates, aligned
   with forensic reproducibility requirements.

6. **"Use a fixed frame schedule and measure its cost."** Robust video aggregation
   (mean + top-fraction of frame evidence) uses a fixed 32-frame sample. → Technical:
   robust video-level aggregation; low-compute efficiency and calibration benefit are
   unverified because no FLOPs/VRAM comparison exists and BEST ECE is 0.2171.

7. **"Fails honestly, reports where it fails."** The official-pair diagnostic found no
   FaceSwap collapse: production video AUC is 0.5310 for FaceSwap and 0.5615 overall,
   while the spatial adapter reaches 0.6274 overall. REVIEW routes ambiguous cases to
   people. → Technical: per-method analysis and confident-error auditing, with protocol
   and split provenance required before interpreting a family difference.

8. **"Built to be re-checked."** A human can open the exact frames, see the flagged
   ROI boxes, and audit the reasoning; decisions are not black boxes. →
   Technical: ROI-box rendering on real frames + machine-readable JSON report,
   positioned against explainability gaps found in forensic reviews (WJARR 2026).

9. **"Make the measurements re-checkable."** The reporting records split, sample unit,
   metric definitions, and error rates where retained. → Technical: seed-controlled
   splits, locked test sets, and documented limitations. This does not establish legal
   admissibility or compliance with a forensic standard.

---

## 3. MUST-FIX before submission (blockers)

These are "no issues" requirements — reviewers will catch all of them.

1. **Document, do not overclaim Celeb-DF-v2.** The extraction contains frames and landmarks but
   zero mask files. The honi05 backbone was trained on Celeb-DF-v2, so a honi05-based Celeb-DF
   result is contaminated and cannot be used as an external test of the deployed path. Any future
   external classification run requires a disjoint backbone, explicit provenance, and a separate
   report; it cannot support an external localization claim.

2. **Quarantine unattributable reports.** `outputs/verdict_shootout.json` records ACC 0.9450 /
   AUC 0.9903 on 200 videos without dataset, split, disjointness, or preprocessing; `outputs/eval_report.json`
   is likewise unattributed. Preserve both as historical artifacts, but do not cite either.

3. **Decide the trained/held-out method split and say it.** Current training uses
   all 4 methods. For a cross-method novelty claim, hold out one method (e.g.,
   FaceSwap) and show the system's REVIEW rate rises on it. `generalize.py --eval-methods`
   supports this.

4. **Add seeds to locked splits.** Run at least 3 seeds and report mean ± std for the
   headline metrics. Reviewers in forensics want error bars.

5. **Publish data-availability and license statements.**
   - FF++ is research/non-commercial. Celeb-DF v2 is research-only. DeepfakeBench
     preprocessing and the `deepfakebench_xception` weights are CC BY-NC 4.0.
   - The paper only *cites* these. If you release code, do **not** bundle FF++/Celeb-DF
     data or the NC weights in the repo.
   - Verify the honi05 `best_model.pt` provenance/license in `outputs/models/` before
     any code release; the code comment says MIT, confirm it.
   - Reproducibility statement: pinned `requirements.txt` + `pyproject.toml`,
     torch 2.5.1, and the frozen-verdict design already help.

---

## 4. Required experiments (compute budget: single RTX 4050)

Use the available RTX 4050 runtime deliberately; resource efficiency is not a claim until measured. Recommended order:

| # | Experiment | Command / note | Purpose |
|---|---|---|---|
| E1 | Cross-dataset Celeb-DF (raw) | `docker compose run --rm verdict` then `generalize --celebdf-dir data/Celeb-DF-v2` after the path fix | strongest generalization claim |
| E2 | Cross-method hold-out | `generalize --checkpoint ... --eval-methods FaceSwap` (retrain w/o FaceSwap first) | shows REVIEW band handles weak methods |
| E3 | Calibration + REVIEW rate | `generalize --calibrate` (writes `decision_policy.json`) | ECE, review-rate, confident-FN numbers |
| E4 | Seed study (3 seeds) | run training/eval with `--seed 0/1/2`, report mean±std | error bars |
| E5 | Frame-count ablation | eval at 8/16/32 frames per video | measures frame-schedule sensitivity |
| E6 | Compute measurement | time inference + VRAM/FLOPs at 32 frames; compare vs full-video | required before any resource-efficiency claim |
| E7 | Region localization | report per-ROI IoU/hit (already produced) + a small qualitative figure of fixed ROI boxes on frames | supports bounded review evidence, not causal attribution |
| E8 | Robust aggregation ablation | mean vs mean+top-fraction | defends aggregation choice |

Every run must be committed to a results JSON under `outputs/` with the checkpoint
path, config, and seed recorded (existing reports already follow this pattern).

---

## 5. Results table to include (honest framing)

- BEST headline report: ACC 0.6965, AUC 0.8104, ECE 0.2171, regionIoU 0.5576 — **603 face samples**, despite the `unseen_videos` key; `video_level=false`.
- Official-pair production head: honest retained video AUC 0.5592 on 142 videos. The same-partition diagnostic reports Deepfakes 0.5488, Face2Face 0.5833, FaceSwap 0.5310, NeuralTextures 0.5816, overall 0.5615.
- Frozen honi05 diagnostic: video AUC 0.5377 overall (real mean 0.145 vs fake mean 0.188). Spatial adapter diagnostic: video AUC 0.6274 overall.
- ECE 0.2171 is measured before any approved calibration claim; report calibration only with its split, policy, and coverage.
- Region localization: regionIoU 0.558, regionHit 0.741 on the 603-face-sample report; external localization is unsupported because Celeb-DF-v2 and DFDCP have zero mask files in this extraction.
- Cross-dataset Celeb-DF: no citable honi05 result; the backbone is contaminated for Celeb-DF-v2 and the available external data cannot satisfy the prior capacity gate.

Never report a number without: checkpoint path, config, seed, dataset split, and
metric definition.

---

## 6. Writing deliverables

Create a `paper/` directory with:

- `paper/abstract.md` — 250-word abstract from §2 novelty list (non-technical lead).
- `paper/contributions.md` — numbered contributions mapped 1:1 to the 9 novelties.
- `paper/related_work.md` — comparison table against: DiffusionFF (CVPR 2026),
  Schwarcz & Chellappa parts-based (CVPRW 2021), LESB (WACV 2025W), Segmentation-Guided
  Spatial Indexing (arXiv 2606.00098), DFD-FCG (CVPR 2025), GenD (WACV 2026),
  three-way synthetic detection (MediaEval 2025), DeepfakeBench (NeurIPS 2023),
  "Don't Guess, Escalate" position paper (arXiv 2512.16614), WJARR 2026 forensic
  review. For each: "same" vs "different vs RL-ROI-Net."
- `paper/method.md`, `paper/experiments.md`, `paper/limitations.md`, `paper/ethics.md`
  (licensing + no legal/sole-evidence claims).
- `paper/latex/` — skeleton (IEEE/ACM template) ready to fill.

Guidelines:
- Use the phrase **"review assistant"** and "do not use as sole evidence" prominently
  (mirrors README and WJARR conclusions).
- Report **protocol-qualified findings explicitly**: near-chance official-pair production ranking,
  the measured adapter comparison, ECE before any approved calibration policy, and external-data
  limitations. Do not use the non-official-pair FaceSwap 0.283 result as model behavior.

---

## 7. Venue & submission path

- Primary: **CVPRW / ICCVW Media Forensics (WMF)**, or **IEEE MLSP / ICASSP
  workshops**, or **EUSIPCO / WIFS** — system + protocol papers fit these.
- Fallback: preprint (arXiv, MIT-licensed code repo) + a capstone report.
- Checklist before any submission:
  1. §3 blockers resolved.
  2. §4 experiments run, JSON reports saved.
  3. All licenses verified; no NC weights or datasets bundled in any release.
  4. Reproducibility statement + pinned env.
  5. Honest numbers only (§5).

---

## 8. Order of execution (roadmap)

1. **Fix the Celeb-DF path** (§3.1) — highest value. Verify with the existing
   `test_celebdf.py` tests.
2. **E1 + E3**: cross-dataset run + calibration (fills the largest evidence gaps).
3. **E2**: cross-method hold-out (FaceSwap) to demonstrate REVIEW band.
4. **E4, E5, E6, E8**: ablations and compute measurement (cheap on one GPU).
5. **Write** the paper deliverables (§6) using §2 novelties and §5 honest numbers.
6. **Ethics/license pass** (§3.5) then submit (§7).
7. Update `progress.md` and this plan as results land.
