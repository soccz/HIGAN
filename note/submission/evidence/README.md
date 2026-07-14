# Evidence Manifest

This directory contains the fixed-seed artifacts behind the manuscript numbers.
All JSON files are generated outputs. `compute_trilemma_table.py` is a
standalone re-aggregation script for the submitted per-record evidence. The
FFHQ Python files record the GPU protocols used to generate the robustness
outputs; they require the public checkpoints, InterFaceGAN boundaries, and
repository generator wrappers. The fixed JSON files let reviewers verify the
submitted tables without rerunning generator experiments.

## Main Tables

- Table 1, analytic toy:
  - `toy_table_groundtruth.json`
  - Key numbers: composed-JVP second-order relative error
    `2.5e-17`; best finite-difference second-order relative error
    `8.6e-09`.

- Figure 1 and the magnitude axis of Table 2:
  - `stylegan_fd_vs_exact_table2.json`
  - `stylegan_fp64_precision_control.json`
  - Key numbers: StyleGAN-bedroom seed 31 second-order magnitude-best
    step `3.0`, mean relative error `45.2%`; fp32/fp64 matched control
    both approximately `47%`.

- Tables 2 and 3, StyleGAN-bedroom 16-step trilemma:
  - `stylegan_fd_vs_exact_table2.json`
  - `compute_trilemma_table.py`
  - `trilemma_table.json`
  - Regenerate with:
    `python evidence/compute_trilemma_table.py`
  - Key numbers: second-order magnitude/bias/rank best steps
    `3.0 / 2.0 / 0.2`; first-order stable step `0.02`.

- Table 4, robustness summary:
  - `fd_trilemma_robustness_summary.json`
  - Per-record source files:
    - `bedroom_seed32_fd_vs_exact_metrics.json`
    - `bedroom_seed33_fd_vs_exact_metrics.json`
    - `ffhq_lod0_seed31_fd_vs_exact_metrics.json`
    - `ffhq_lod0_seed32_fd_vs_exact_metrics.json`
    - `ffhq_lod0_seed33_fd_vs_exact_metrics.json`
    - `ffhq_lod1_fd_vs_exact_metrics.json`
    - `ffhq_lod2_fd_vs_exact_metrics.json`
  - Per-sweep trilemma tables:
    - `bedroom_seed32_trilemma_table.json`
    - `bedroom_seed33_trilemma_table.json`
    - `ffhq_lod0_seed31_trilemma_table.json`
    - `ffhq_lod0_seed32_trilemma_table.json`
    - `ffhq_lod0_seed33_trilemma_table.json`
    - `ffhq_lod1_trilemma_table.json`
    - `ffhq_lod2_trilemma_table.json`
  - Key qualitative result: the second-order magnitude, signed-bias, and
    rank optima separate on bedroom seeds 31--33 and full-resolution FFHQ
    seeds 31--33. Lower-detail FFHQ lod 1 and 2 are included as controls,
    serving as lower-detail controls.

## Replication Audit

- Single-anchor curvature/nonlinearity result:
  - `prediction_single_anchor_gonogo.json`
  - Key numbers: raw curvature/nonlinearity correlation about `0.47`;
    partial correlation given first-order displacement about `0.45`.

- Multi-anchor audit:
  - `prediction_multianchor_table.json`
  - Key numbers: partial correlations
    `[0.098, 0.193, -0.151, 0.373, 0.330]`; mean `0.168` (`+0.17`);
    `n_anchors_CI_excludes_0 = 0`. The `+0.17` cross-anchor figure is the
    unweighted mean of the five per-anchor partials with a Student-t CI
    on n=5 (no random-effects model is fit).

- Section 6 confound check (finite-difference vs. exact ranking and decision recovery):
  - `30_curvature_ordering_flip.py`, `curvature_ordering_flip_metrics.json`
  - `31_curvature_decision_groundtruth.py`, `curvature_decision_gt_metrics.json`
  - Key numbers: a fixed-step finite-difference curvature recovers the exact
    direction ranking (Spearman `0.815` at delta=3.0 to `0.963` at delta=0.1)
    and two of the three exact top directions at small steps (one of three at
    delta=3.0); against the estimator-free nonlinearity, rho_FD ranges
    `0.19`-`0.35` across steps vs rho_exact `0.47`.
  - Stored verdict `verdict_consequence_demonstrated = false`: on this
    rank-level question the exact instrument does not overturn the audit
    verdict. The instrument's necessity is on the magnitude and sign axes
    (Table 2), not on this ranking question.

## FFHQ Runners

- `run_ffhq_fd_vs_exact.py`
  - Records the FFHQ finite-difference-vs-exact trilemma sweep protocol and
    can re-run it inside the full repository with the public InterFaceGAN
    checkpoint and semantic boundaries.
  - The manuscript uses full-resolution lod `0` seeds 31--33 and lower-detail
    lod `1`/`2` controls.

- `run_ffhq_resolution_invariance.py`
  - Records the exact-JVP FFHQ resolution audit protocol and can re-run it
    inside the full repository with the same public FFHQ assets.
  - `ffhq_resolution_metrics.json` reports rank stability across lod `0`, `1`,
    and `2`; all pairwise Spearman correlations are `1.0`.

## Files Intentionally Absent

The archive intentionally excludes old drafts, excluded/dead-line evidence,
and identity-bearing local files. Only the files listed in this manifest are
part of the submission evidence.
