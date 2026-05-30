# WORKPLAN — paper_refutation/ (live status board)

Working dir for the v2 (refutation / TMLR-primary) rewrite. The locked spec is
`SPINE.md`. The stale CVPR skeleton lives in `../paper/` and is NOT used for
drafting — only mined for reusable figures/numbers.

**Thesis (one line, v5 FLOOR):** Edit magnitude operationally DOMINATES latent
curvature ρ for edit-risk; ρ is NOT an actionable per-direction risk signal — its
apparent residual is an attribute-difficulty/magnitude-axis artifact (≈0 within the
operational unit: +0.07/+0.16/−0.08; reverses under LPIPS control: −0.19/−0.38/−0.11).
Matched-pair edit-selection benchmarks silently leak realized magnitude (low-ρ wins
0.765 small / 0.287 large) — the cleanest contribution. Plus the forward-mode
instrument + protocol. (BANNED: "sufficient statistic", "flat", "real-but-weak
ordinal residual".)

**Venue:** TMLR primary (~65-78%); ICLR 2027 / NeurIPS 2027 upside (~25-35%),
same manuscript. CVPR/ICCV/ECCV dropped.

---

## Experiments: FROZEN
- No blocking experiments remain.
- One OPTIONAL bonus (ffhq+LPIPS incremental CV-R², n=10→≥100) — **expected to
  fail**; never block TMLR on it.

## Steps (status)

| # | Step | GPU | Blocking | Status |
|---|------|-----|----------|--------|
| 0 | Verify SPINE (5 adversarial workflows) → converged v5 FLOOR | 0 | yes | DONE |
| 0b | Clean fixed-seed re-runs (deterministic_set false on lead aggregate) | tiny | yes | pending |
| 1 | Rewrite intro: drop dead C4 payoff + single-seed hook; magnitude-dominance + C4b hook | 0 | yes | pending |
| 2 | Demote C3 honestly: per-config null pass-rate table + Benjamini-Hochberg FDR | 0 | yes | pending |
| 3 | Audit + name magnitude baseline (leakage) | 0 | yes | DONE (leakage_audit, residual_conservative) |
| 4 | Harden hooks: bootstrap 92.3% clustering; C2a survives partialling \|Δ\| | mins | no | pending |
| 5 | Fix FD anomaly (eps-sweep; rel_err 91.6% @ eps=1e-3) | mins | no | pending |
| 6 | C5 magnitude-matched + bootstrap CI on per-bin win-rates (non-monotone) | 0 | no | DONE (c5_magnitude_matched); bootstrap-CI pending |
| 7 | Release artifact (README/setup + protocol harness + magnitude-baseline eval) | 0 | no | pending |
| 8 | Dual-submit: TMLR primary + ICLR upside; apply kill criteria | 0 | no | pending |

## Section files (sections/)
01_introduction · 02_related · 03_method · 04_signal_real ·
05_no_beat_magnitude · 06_no_control · 07_mechanism ·
08_baseline_protocol · 09_conclusion

Prose is written only AFTER Step 0 verification passes (avoid building on a flawed
spine — the lesson from v1, where an adversarial audit caught blockers a manual
check missed).

## Kill criteria (route solely to TMLR if any fire)
1. After FDR, zero C3 cells survive.
2. Bootstrapped clustering < ~80% or CI overlaps permutation null.
3. C2a encoder-invariance explained away by per-attr magnitude.
4. Optional n≥100 scale-up still fails operational AUROC (expected).
