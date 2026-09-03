# Stable-RouteNet — v3.1 Architecture (Kaggle 2xT4)

Evidence-stability-conditioned sparse expert routing for generalizable deepfake detection.
Implements `Stable-RouteNet_v3_1_Architecture_Document.md`.

**Central hypothesis:** detectors overfit to unstable manipulation/domain shortcuts; stable,
reliable localized evidence routed through specialized sparse experts generalizes better.

**Core formula:**

\[
W_i = (M_i \cdot S_i \cdot R_i)^{\alpha}, \quad 0.2 \leq \alpha \leq 0.8, \quad \alpha_0 = \tfrac13
\]

where \(R_i = \sqrt{A_i \cdot C_i}\) is derived from expert-correction agreement \(A_i\) and
intervention consistency \(C_i\) — **no free neural reliability head**.

## Contents

| File | Purpose |
|---|---|
| `Stable-RouteNet_v3_1_Architecture_Document.md` | **Authoritative v3.1 architecture & research specification** |
| `Stable-RouteNet_v3_1_5epoch_2xT4_CUDA_CONTEXT_FIXED_FINAL.ipynb` | 5-epoch confirmation run notebook (2x T4) |
| `Stable-RouteNet_Q1_Focused_Build_Specification_v5.md` | Full Q1 build specification (20-epoch, ablations) |
| `_build_nb.py` | Builder that generates the Q1 notebook from source (re-run to rebuild) |
| `data/` | FF++, Celeb-DF-v2, DFDCP (preprocessed frames + landmarks) |

## Quick start (Kaggle)

1. Upload `data/` as private dataset `Stable-RouteNet-1`; attach to notebook (T4 x2, internet on).
2. Run all — preflight must print `ALL V3 PRE-FLIGHT SMOKE TESTS PASSED`.
3. Training: 5 epochs, AdamW, 4 experts top-2 sparse, `T=8`, frozen DINOv2-L/14 with registers.
4. Final cell evaluates **once** on DFDCP 100-video screening (50 real + 50 fake, no leakage).

Train: FF++ (150) + Celeb-DF-v2 (150) reals via DD-SBI (50/50 real/fake) · Val: 25+25 · Test: DFDCP screening.

## Architecture (v3.1)

`M` = local manipulation evidence, `S` = intervention stability, `A` = expert-correction agreement,
`C` = intervention consistency, `R = sqrt(A*C)`, `W = (M*S*R)^alpha`.
See the architecture document for the full specification, loss contract, curriculum, and evaluation protocol.
