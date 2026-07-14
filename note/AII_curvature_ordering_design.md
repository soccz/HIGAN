# ⓐ-ii Experiment — Does the naive /δ² estimator INVERT a curvature-based ordering?

**Date:** 2026-06-19. **Goal:** test whether the consequence needed to keep 9732 in TMLR
exists: the naive single-anchor `/δ²` second-difference curvature estimator (what a
practitioner adds when measuring generator curvature) **reverses a curvature-based ORDERING**
that the exact composed-JVP instrument reveals on a real generator. If yes → TMLR (a present
decision flips); if no → honest fallback to a numerical-analysis venue (ⓑ).

## Why this is the honest form of ⓐ (not "flip a published number")

Triangulated 2026-06-19 (canonical survey 0/7; diffusion frontier all analytic/autodiff;
Koulakis 2606.15760 verified grid-FD truncation-safe): **no published generative-geometry
pipeline computes curvature by the catastrophic `/δ²` second-difference**, so "flip paper X's
number" is not available. What IS available and legible:

- **Anchor (present-tense audience):** Koulakis et al., *The Data Manifold under the
  Microscope* (arXiv 2606.15760, June 2026, code public) order a generative manifold's layers
  by curvature ("deeper layers → more curved → class separation"). Curvature-ORDERING of a
  generative model is a **live 2026 practice**. A practitioner adding this measurement on a
  generator, reaching for the obvious `/δ²` second difference instead of nested AD, is the
  at-risk reader — present tense, not a future bet.
- **Mechanism (ours):** Proposition 1 — central `/δ²` second difference of a deep generator
  in fp32 has an irreducible relative-error floor ~√ε_mach (≥45% on StyleGAN, §5b); the
  first-order central difference of the same code is safe (≤2.4%, the control).

## Hypothesis

H1 (per-layer): The exact instrument orders the 14 StyleGAN synthesis layers by curvature
intensity **monotonically** (coarse→fine decay; the claimed §5a Spearman(layer,intensity)
∈[−0.96,−0.80]). The naive `/δ²` estimator, dominated by the floor where true curvature is
small (fine layers), **destroys or reverses** that ordering — Spearman(layer, c_FD) loses its
magnitude/sign, and the rank agreement between the exact ordering and the FD ordering is low,
**robustly across every step in the grid** (no step recovers the ordering).

H2 (per-attribute): The exact ranking of the 8 annotated bedroom attributes by curvature is
reordered by the `/δ²` estimator (a different "which attribute is most curved" answer).

## Method (reuses `29_shao_fd_vs_exact.py` primitives)

- Generator: StyleGAN-bedroom-256 (fp32, 8 GB), L=14, D=512. Base latents: N samples (seeded).
- Directions: each of the 8 attribute boundaries placed in a **single** layer ℓ via
  `_layered_direction(b, L, D, dev, only_layer=ℓ)` (unit 512-vector in slot ℓ, zero else).
- Per (attr, layer ℓ, sample): curve `z(α)=wp + α v_ℓ`; compute
  - exact: `c*(ℓ) = mean|d²/dα² G|` via composed JVP (`exact_second`, step-free);
  - 1st-order control: `mean|dG/dα|` exact vs central-FD (must agree ≤~2.4% → pipeline sound);
  - naive FD: `c_FD(ℓ,δ) = mean|(G(z+δv)−2G(z)+G(z−δv))/δ²|` for δ in a grid.
- Aggregate per layer (mean over attrs×samples): `c_exact[ℓ]`, `c_FD[δ][ℓ]`.
- **Decision metrics:**
  - `ρ_exact = Spearman(ℓ, c_exact[ℓ])` (expect ≈ −0.9 — the clean ordering).
  - `ρ_FD[δ] = Spearman(ℓ, c_FD[δ][ℓ])` for each δ (test).
  - `τ[δ] = Spearman(rank c_exact, rank c_FD[δ])` — rank agreement exact↔FD (test).
  - per-attribute: same on the 8-attribute rankings + list the two orderings explicitly.

## Expected signal (H1 true) vs null (H1 false → ⓑ)

| metric | H1 TRUE (flip → TMLR) | NULL (no flip → ⓑ) |
|---|---|---|
| ρ_exact | ≈ −0.85…−0.96 (monotonic) | (same; exact is the truth either way) |
| ρ_FD[δ] | |ρ| small or sign-flipped at **every** δ | tracks ρ_exact (≈ −0.9) |
| τ[δ] exact↔FD | low (≲0.5), all δ | high (≳0.8) |
| 1st-order control | ≤ ~2.4% (sound) | ≤ ~2.4% (sound) |

**Decision rule (pre-registered):** H1 confirmed iff the exact ordering is clean
(|ρ_exact|≥0.8) AND the FD ordering is destroyed (|ρ_FD| materially smaller, or sign-flipped,
or τ<0.5) **at the practitioner's best fixed step and robustly across the grid**, with the
1st-order control passing. Otherwise report NULL honestly and recommend ⓑ. We will not
cherry-pick a step that flips; the claim requires robustness across the grid.

## Honest-scope firewall (kept)

We do NOT claim Koulakis et al. or any paper is wrong — they use a different, safe (grid-FD)
route. We claim: the obvious single-anchor `/δ²` shortcut a practitioner would reach for
inverts a curvature-based ordering on a real generator, so the estimator choice is decisive
for a conclusion people draw today. ρ here is the layer/attribute curvature intensity — a
standard quantity (not a self-defined ratio).

## Output
`higan_dev/out/curvature_ordering_flip/metrics.json` + `ordering_flip.pdf` (per-layer exact vs
FD curves + rank-agreement panel). Pilot first (small N) to validate, then large-N run.
