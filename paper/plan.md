# HIGAN paper — master plan

> **Target**: CVPR 26 main track (deadline ~ Nov 2025) or ICCV 26 (deadline ~ Mar 2026).
> **Lead**: solo first-author (안지홍). Time horizon: 12–18 months, fail-tolerant.
> **Status**: draft 0. Started 2026-05-09.

## 1. Thesis

Trained GAN latent spaces admit a clean **differential-geometric**
interpretation: the generator is a smooth map G : W+ → I from a Euclidean
latent space to an image manifold, and many "interpretation" tasks
(saliency, attribute editing, disentanglement) are *exactly* the
classical geometric operations on this map — pushforward of tangent
vectors, Hessian-vector products, Lie brackets of vector fields.

We propose **forward-mode autodifferentiation (JVP)** as a unified
computational tool to estimate these geometric quantities efficiently
on a frozen pre-trained generator, and we present six empirical claims
that the resulting picture makes coherent, with **compositional editing
prediction** as the applied payoff.

## 2. Six theoretical claims

| # | Claim | Empirical handle (from prior work / current repo) |
|---|---|---|
| **C1** | The image manifold has measurable extrinsic curvature along every latent direction, estimable via composed forward-mode JVP in one combined pass. | §19 of current HIGAN report — ∂²I/∂α² ratios 0.5–23.2 across attributes. |
| **C2** | Curvature magnitude predicts **topological** transitions: directions with high ∂²I/∂α² introduce/remove image-space objects (windows, doors), while low curvature directions only re-paint pixels. | §19 view (ratio 23.2) vs lighting/wood (~0.5) + §23 saliency-morph qualitative. |
| **C3** | Attribute boundary directions are **layer-localised tangent vectors**: the same 512-d vector applied to a non-canonical W+ layer produces large but semantically incoherent perturbations. The right interpretation is a *section* of a layer-indexed fiber bundle. | §08 (8 attr × 14 layer matrix). |
| **C4** | Compositional editing failure is predicted by the **Lie bracket** of attribute vector fields: [X_a, X_b] ≠ 0 ⇒ non-linear composition. | §13 (corr 0.55 for view+texture, 0.97 for texture+texture) ↔ §19 (view curvature 23×) — these two are *linked*. |
| **C5** | An encoder approximates a **coordinate chart** on the manifold, and its quality is best measured not by reconstruction error but by *agreement of derivative structure* with the true wp. | §18 of current report — saliency-vs-GT correlation 1k→40k: 0.008 → 0.359 (45×) while recon MSE only 18% lower. |
| **C6** | Random tangent directions cluster (via K-means on their pushforward magnitude) into a **stratification** of the image manifold that rediscovers human-labelled attribute axes. CLIP zero-shot then assigns names to these strata, completing unsupervised attribute discovery. | §16 + §17 — cluster 2 (n=37, layer 1) auto-identified as "a view through a window". |

## 3. What needs to exist that doesn't yet

### 3.1 Domains
Currently only bedroom. For main-track:
- **FFHQ faces** (StyleGAN2 or StyleGAN3 + InterFaceGAN boundaries: smile, age, glasses, pose).
- **LSUN church** (HiGAN or StyleGAN2 + at least 2–3 boundaries).
- (Stretch) cats / cars.

Each domain must replicate all 6 claims with the same pipeline.

### 3.2 Baselines (on same data, same metrics)
- **GAN Dissection** (Bau 2019) — segmentation-classifier-based per-unit saliency.
- **GANSpace** (Härkönen 2020) — PCA on intermediate activations to find unsupervised directions.
- **SeFa** (Shen & Zhou 2021) — closed-form eigendecomposition of the affine layer.
- **StyleSpace** (Wu 2021) — per-channel style manipulation.
- **Classifier-based Grad-CAM** — train a small attribute classifier (or use CLIP score) and run real Grad-CAM through it. Direct head-to-head with our classifier-free saliency.

### 3.3 Quantitative metrics
- **Saliency–segmentation IoU**: pretrained Mask2Former (e.g., COCO-stuff) → semantic masks → IoU(top-k% saliency, mask of relevant class). Apply per (attribute, region) pair.
- **Compositional interference prediction**: scatter of (∂² curvature) vs (compositional non-linearity from §13). Spearman / Pearson + plot.
- **Edit precision**: for saliency-guided local edits (§10), measure pixel change inside vs outside saliency mask. Higher inside/outside ratio = more precise.
- **Direction recovery**: do random-direction clusters recover known boundaries (precision/recall)?
- **Compute**: forward-mode JVP wall time vs reverse-mode equivalents vs finite difference (matched accuracy).

### 3.4 User study
- 60–100 participants via Prolific (~$2/response × ~10 min = ~$300 budget).
- Pairwise edit quality rating: global vs local (saliency-masked) edit on a held-out set of 30 image-attribute pairs.
- Goal: validate that saliency-guided local edit (§10) is *preferred* over global edit.
- Secondary: subjective coherence rating of taxonomy cluster names (§16-17).

### 3.5 Theory writeup
Math expansions needed:
- Pushforward of tangent vectors through smooth maps (G_*v formalism).
- Composed JVP = HVP derivation, with assumptions (smoothness).
- Lie bracket of latent vector fields → image-space interpretation.
- Sample-complexity bound for the Monte-Carlo expectation estimators.

### 3.6 Code
- Refactor `higan_dev/` core into a domain-agnostic library `manifold_probe/`.
- Plug each domain in via a `Generator` interface.
- Reproducibility: pinned versions, deterministic seeds, exact CLI.

## 4. Repository layout (this directory)

```
paper/
├── plan.md                        # this file
├── progress.md                    # weekly status updates
├── paper/                         # LaTeX source
│   ├── main.tex
│   ├── sections/{intro,method,...}.tex
│   ├── figures/
│   └── references.bib
├── theory/                        # mathematical writeups in markdown
│   ├── 00_overview.md
│   ├── 01_manifold_view.md
│   ├── 02_jvp_pushforward.md
│   ├── 03_hvp_curvature.md
│   ├── 04_lie_bracket_composition.md
│   ├── 05_encoder_as_chart.md
│   ├── 06_stratification_discovery.md
│   └── claims.md                  # one-pager per claim, with evidence
├── experiments/
│   ├── domains/
│   │   ├── bedroom/               # references ../higan_dev (existing)
│   │   ├── ffhq/                  # new
│   │   └── church/                # new
│   ├── baselines/{gan_dissection,ganspace,sefa,stylespace,classifier_gradcam}/
│   ├── metrics/{seg_iou,composition_pred,edit_precision,...}/
│   ├── user_study/{protocol,data,analysis}/
│   └── ablations/
├── figures/                       # source files (PIL, matplotlib) for paper figures
├── related_work/                  # PDFs + per-paper notes (.md)
├── data/                          # symlinks to /mnt/20t large data
└── scratch/                       # one-off experiments, sandbox
```

## 5. Timeline (12-month version, padded for solo)

| Month | Block | Concrete deliverables |
|------|-------|----------------------|
| **1** | Foundation | This plan + theory v1 + LaTeX skeleton + Related work corpus (40+) + FFHQ generator + first cross-domain saliency figure |
| **2-3** | Multi-domain replication | All 6 claims verified on FFHQ and church. Comparison tables. |
| **4-5** | Baselines | All 5 baselines running, comparison tables for IoU + speed + direction recovery |
| **6** | User study | Pilot → full deployment → analysis |
| **7-8** | Theory rigour | Formal definitions, derivations, sample-complexity, edge cases. Math section in LaTeX. |
| **9** | Paper draft v1 | 8 pg main + supplementary. Internal review (find one external eyeball). |
| **10** | Revision | Polish figures, tighten claims, address feedback |
| **11** | Submission | CVPR 26 (Nov deadline) — if missed, ICCV 26 (Mar) |
| **12+** | Review cycle | Rebuttal, possible re-submission |

## 6. Failure modes (be honest)

1. **"yet another GAN saliency"** — 가장 큰 reject risk. Mitigation: lead with second-order curvature + compositional prediction, not first-order saliency.
2. **Solo first-author distrust** — Mitigation: cite parent JKIIT paper, position as principled spin-off.
3. **One-domain narrowness** — fully addressed by multi-domain extension (mandatory).
4. **Missing baselines** — fully addressed by Month 4-5 block.
5. **Insufficient math** — Mitigation: Month 7-8 dedicated rigour pass; pre-print on arXiv for community check before submission.
6. **No user study** — Mitigation: Month 6 dedicated block, modest budget.

## 7. Definition of done

A first version is "done" when:
- 6 claims each have ≥ 1 quantitative result + ≥ 1 cross-domain replication
- All 5 baselines benchmarked on at least 2 domains
- User study reports significance for at least the edit-precision claim
- Math section is self-contained
- Submitted to one of CVPR 26 / ICCV 26 / NeurIPS 26

Reject is acceptable. Iteration based on reviews is expected.

## 8. How to use Claude Code

- **TodoWrite** for macro plan tracking.
- **Subagent (Explore)** for related-work paper search & summarisation in parallel.
- **Subagent (Plan)** when architectural decisions need multiple alternatives.
- **Background bash** for long-running experiments (baseline replication, encoder retrains on new domains).
- **Worktree isolation** for independent baseline implementations (each in its own worktree to avoid conflicts).
- **Memory** for cross-session continuity (already set up).

Hard rules:
- Never auto-write LaTeX paper prose without human review.
- Never publish without checking against `claims.md` factual table.
- Always run reproducibility test (`make repro`) before any commit.
