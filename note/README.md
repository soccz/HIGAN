# Second-Order Finite Differences Invert the Geometry of Generative Models

TMLR submission draft + reproducible code.

**Paper:** `tmlr_main.tex` → `tmlr_main.pdf` (build: `tectonic tmlr_main.tex`, or
`pdflatex` ×2 with `natbib`).

## What it shows
A concrete, reproducible numerical hazard in the differential geometry of
generative models, and the exact instrument that removes it.

- **Sign inversion (§2).** On Stable Diffusion, a second-order geometric
  measurement (Spearman of the Jacobian singular value vs a curvature ratio)
  comes out with **opposite sign** depending only on the numerical method: exact
  composed-JVP gives a consistent −0.73; finite-difference-of-JVP of the *same*
  quantity is sign-unstable and averages −0.18.
- **Why (§3).** Central finite differences are **non-recoverable** for a *second*
  derivative — truncation vs cancellation leave a √ε_mach error floor, large in
  the single precision generators use — while *first*-order FD is well conditioned
  (the asymmetry we exploit as a validation control).
- **Quantified (§4).** On StyleGAN, the naive FD curvature estimate misses the
  exact value by **≥45% at every step**; the first-order FD of the same pipeline
  agrees with exact JVP to **2.4%**, proving the instrument correct.
- **Instrument (§5).** Composed forward-mode AD: exact higher-order pushforward,
  constant memory, any order, generator-agnostic.
- **Survey (§6).** Where the trap is and isn't in published practice (the
  catastrophic /ε² regime is mostly avoided — but it is the natural first thing a
  practitioner reaches for).

## Honest scope
We do not claim any published result is wrong; the survey shows most methods
route around the catastrophic regime. The sign-inversion quantity is one we define
to expose method dependence. This is a numerical-methods + tooling contribution,
not a downstream task win.

## Reproduce
- Toy analytic referee: `toy_fd_groundtruth.py` (→ `toy_fd_groundtruth.json`).
- StyleGAN FD-vs-exact curvature (§4): `../higan_dev/scripts/29_shao_fd_vs_exact.py`
  (→ `../higan_dev/out/shao_fd_vs_exact/metrics.json`).
- Stable Diffusion sign-inversion (§2): `../paper/experiments/diffusion/run_park_fd_vs_exact.py`.

All experiments run on a single 8 GB GPU. `main.tex` is the earlier short-note
version, superseded by `tmlr_main.tex`.
