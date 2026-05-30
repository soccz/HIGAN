> ⚠️ **v5 DID NOT CONVERGE — it over-corrected toward the null. (6th audit, verified.)**
> Three "floor" pillars are artifacts, NOT findings:
> - **C4b "magnitude leak" (0.765/0.287) is a TAUTOLOGY** (independently reproduced):
>   a RANDOM selector gives 0.730–0.745/0.251–0.265; low-ρ is the smaller-edit side
>   only 51.9% (coin flip); mean magnitude flow to low-ρ = −0.0001. **DELETE C4b.**
> - **"ρ reverses sign under LPIPS control" (−0.19/−0.38/−0.11) is CIRCULAR**:
>   dmg_lpips correlates +0.92/+0.99/+0.92 with dmg_id → partialling the outcome on a
>   copy of itself. Under TRUE magnitude controls the sign STAYS POSITIVE
>   (partial(ρ,dmgID|attr_mag)=+0.27/+0.48/+0.20; |abs_alpha=+0.46/+0.72/+0.57). **DELETE.**
> - **within-(seed,attr) collapse (+0.07/+0.16/−0.08)** is n=6/group noise.
>
> **HONEST POSITION (v6, pending framing decision):** ρ is a **weak, magnitude-
> confounded, but NON-ZERO** per-direction edit-risk signal — positive partial-Spearman
> **6/6 over clean latent-step magnitude (+0.46…+0.75)** — that **does NOT dominate**
> edit magnitude in variance-ranking. Whether ρ "beats magnitude" is
> **baseline-definition-dependent** (beats abs_alpha; not the outcome-coupled realized
> Δ). Neither "ρ predicts" (v1) nor "ρ not actionable" (v5) holds. The 6-workflow
> oscillation IS the finding: the signal is weak and the question is fragile.

# TARGET LOCK v4 (CONVERGED) — TMLR (primary) + ICLR/NeurIPS (upside) [SUPERSEDED, see banner]

**Re-locked 2026-05-30 after FOUR adversarial workflows. v1 ("ρ predicts damage")
over-claimed; v2/v3 ("magnitude is a sufficient statistic / 0-41") over-claimed
the other way (a baseline-leakage artifact). The 4th workflow CONVERGED (reconciled
the contradiction with new analysis) rather than flipping. This v4 is the
data-true position. Single source of truth; if a sentence does not serve the
through-line below, it is out.**

> CONVERGENCE NOTE (v1→v5 monotonically tightened; each audit removed a layer of
> artifact; "magnitude dominates" survived all five. This is the FLOOR — the
> residual cannot be claimed weaker than "no robust per-direction signal". All
> numbers reproduced from committed scripts in `paper_refutation/`.):
> - "0/41 ρ beats baseline" and "ρ beats clean abs_alpha 6/6" reconcile as
>   metric/threshold artifacts (0/41 = per-seed 80%-folds threshold over a leaky
>   4-feature MAX; 6/6 = pooled partial-Spearman over one clean baseline).
> - **The "6/6 ordinal residual" was itself a pooling artifact** (`residual_conservative.json`):
>   the positive pooled partial-Spearman(ρ, dmgID | attr-mag) (bedroom +0.27,
>   church +0.48, ffhq +0.20) is a **between-attribute attribute-difficulty proxy**;
>   WITHIN the operational (seed,attr) selection unit it collapses to
>   **+0.07 / +0.16 / −0.08**, and controlling for **total image movement (LPIPS)**
>   it **REVERSES sign in all three domains (−0.19 / −0.38 / −0.11)**. ρ is not a
>   robust per-direction edit-risk signal.
> - **Matched-pair benchmarks leak realized magnitude** (`residual_conservative.json`):
>   low-ρ wins **0.765** when it applies the SMALLER realized edit vs **0.287** when
>   larger (pooled n=2489 / 2311). At tightly magnitude-matched edits the win-rate is
>   ~0.55 (z≈+2.7) at rel≤0.05, collapsing to NS (z≈+1.9) at rel≤0.02.
>
> THESIS: **edit magnitude OPERATIONALLY DOMINATES ρ for edit-risk; ρ's apparent
> signal is an attribute-difficulty / magnitude-axis artifact (≈0 within the
> operational unit, sign-reversing under total-movement control); and matched-pair
> edit-selection benchmarks silently re-encode realized magnitude — a
> measurement-validity caution that is the paper's cleanest contribution.**

---

## 0. Venue (no more chasing a nonexistent positive)

- **PRIMARY: TMLR** (rolling). Bar = "claims supported by clear evidence" + "of
  interest to some audience"; **no novelty/SOTA gate**. Exactly fits a rigorous
  honest null. Est. accept **65–78%** after the work below (conditional on FD
  anomaly fixed, `deterministic_set=true` re-runs, and C2a surviving partialling).
- **UPSIDE (same manuscript, ~free): ICLR 2027** (abstract ~Sep 2026, full ~Oct
  2026) and/or **NeurIPS 2027** (~May 2027), via the **refutation** framing.
  Est. accept **25–35%**. A real minority shot, not a primary bet.
- **DROPPED: CVPR / ICCV / ECCV.** All four strategy panels agree this is an
  analysis/characterization paper, not method-beats-benchmark. Wrong family.
  *(User's MEMORY still lists CVPR 2027 — superseded by this decision.)*
- **Do NOT** spend the 4 months chasing geometry>magnitude on GPU. 5 independent
  HIGH-confidence probes say it does not exist in the data. Compute cannot make it.

---

## 1. THE SPINE (refutation through-line — every section serves this)

> Using an **exact forward-mode differential-geometry instrument** for generator
> latent spaces, we show that **edit magnitude operationally dominates latent
> curvature ρ for edit-risk**: ρ's apparent edit-risk signal is an
> **attribute-difficulty / magnitude-axis artifact** — within the operational
> (seed,attr) selection unit it is ≈0 (+0.07 / +0.16 / −0.08) and it **reverses
> sign under total-image-movement (LPIPS) control** (−0.19 / −0.38 / −0.11), so it
> is **not an actionable per-direction risk signal**. We further show **matched-pair
> edit-selection benchmarks silently re-encode realized edit magnitude** (low-ρ wins
> 0.765 on smaller edits, 0.287 on larger) — a measurement-validity caution. We
> deliver the instrument, the audited leakage-free baseline, and a reusable protocol.

Arc (each step earns the next):
1. **Instrument** — forward-mode JVP geometry: exact ρ in one pass, cross-arch.
2. **ρ is a real geometric signal** — encodes semantics (encoder-correlated,
   magnitude-confounded; hardening pending Step 4).
3. **ρ does not beat magnitude per-direction** — the apparent residual is a
   between-attribute difficulty proxy (≈0 within the operational unit; reverses
   under LPIPS control); C3 null-sensitive. "0/41" and "6/6" reconcile as artifacts.
4. **The matched-pair benchmark leaks realized magnitude** — the apparent
   edit-selection signal re-encodes how big the edit was (low-ρ 0.765 small / 0.287
   large); the residual collapses to NS under tight magnitude matching.
5. **Why** — pooled/between-attribute correlation ≠ operational per-direction value;
   magnitude dominance + the benchmark confound.
6. **The contribution** — instrument + audited leakage-free baseline + the
   measurement-validity finding + protocol; honest limits.

**Anti-collapse rule**: main text carries the corrected ledger numbers below ONLY.
The 95 control_v* versions live in a Robustness Appendix that defends, never narrates.

---

## 2. CLAIM LEDGER v4 (converged; every number adversarially re-verified + reproduced)

| # | Claim | Evidence (verified) | Strength | Notes / fixes applied |
|---|-------|---------------------|----------|-----------------------|
| **M** | Exact per-pixel pushforward in one forward pass; composed higher-order JVP; on StyleGAN + Diffusion | JVP 45.6 ms/img **(bedroom@256 only)**; determinism bit-identical 17 dp | **Strong (tool)** | **2 architecture families** (StyleGAN, Diffusion), 4 generators — NOT "3 architectures". FIX FD anomaly (rel_err 91.6% @ eps=1e-3) before release; commit a determinism artifact; `deterministic_set` currently false |
| **C2a** | ρ encodes semantics, **encoder-correlated** (NOT yet "invariant") | multi-CLIP B/L/H Pearson 0.97–0.99; DINOv2 0.94/0.98 | **Moderate (LIVE kill-criterion)** | DOWNGRADED: Pearson is **outlier-driven** (drop 'view'/'pose' → ~0.10); bedroom Spearman 0.29–0.43 **NS**. "encoder-invariant" is a CLAIM PENDING Step-4 hardening (partial out per-attr \|Δ\| + shuffle-null), not established fact. If it dies to partialling → kill-criterion 3 fires. |
| **C2b** | Structure persists at higher order & in diffusion | **exact 2nd-order: pose 14.5× vs smile 0.96×**; SD signal across 9 timesteps (`sd_c1_c2_full_t`) | **Moderate** | LEAD with the **exact** 2nd-order contrast. 368× (3rd) is **FD-on-JVP, n=8, no CI** → demote to descriptive. SD = signal **exists**; **gap untested** in diffusion (do not imply it generalizes). |
| **C2c** | Curvature signature transfers across domains | k=2 clustering **92.3%** (12/13, bedroom+ffhq) | **Weak/Suggestive** | FIX: was 92.86%/13-14 (wrong; n=13, misassigns ffhq-eyeglasses). **Single-seed, no CI.** Bootstrap ≥100 + ARI vs label-permutation null. Drops to 77.8% incl SD. |
| **C3** | ρ predicts damage rank (population), null-sensitive | FFHQ-only Spearman +0.38±0.08 (CLIP)/+0.32±0.09 (ArcFace); 3-domain CLIP ~+0.41, std ~0.15, spread bedroom +0.28→church +0.58; **null-sensitive 6/21** | **Weak, regime-dependent** | FIX label: the ±0.08 set is **FFHQ-only** (ArcFace=face). Report FDR over the ~125-test family; state which cells survive. Frame (per floor): ρ adds **no actionable per-direction value** — apparent residual is a between-attribute proxy that vanishes within-unit and reverses under LPIPS (see BASELINE row). |
| **C4** | ρ matched-pair win-rate **collapses under tight magnitude matching** | tight-match (`matched_pair_magnitude_residual.json`): win ≈0.55, z≈+2.7 at rel≤0.05 (n≈646); **→ z≈+1.9 NS at rel≤0.02** (n≈247) | **Moderate (collapses)** | Raw 0.587 was magnitude-confounded (C4b). The residual that remains at moderate matching reverses under LPIPS control (BASELINE) → not actionable. Robust to pairing method. |
| ~~C4b~~ | ~~Matched-pair benchmark leaks realized magnitude~~ **DISPROVEN — TAUTOLOGY** | random selector reproduces 0.765/0.287 (0.730–0.745); low-ρ smaller-edit side only 51.9%; magflow −0.0001 | **DELETED** | Independently confirmed selector-independent. Remove from thesis/intro/§7/§8. The only defensible coupling is Spearman(ρ, realized-Δ)=+0.07/+0.41/+0.28 (modest, church/ffhq only). |
| **C5** | ρ-gap stratification is **non-monotone even at matched magnitude** | DONE magnitude-matched (`c5_magnitude_matched.json`, rel≤0.05, n=646): Q1 0.56 → Q3 0.60 (peak) → **Q4 0.50/0.45 collapse** at extreme ρ-gap; Spearman(ρ-gap,win) −0.19. (unmatched was 0.551/0.551/0.598/0.540) | **Supporting (confirmed real)** | The inverted-U **survives** magnitude matching → not a pure artifact; at extreme ρ-separation low-ρ even *loses*. Consistent with **no actionable per-direction signal**. `deterministic_set=false` still needs a clean re-run. |
| **BASELINE** | Magnitude operationally dominates; ρ's apparent residual is an **attribute-difficulty / magnitude-axis artifact** | within-(seed,attr) partial Spearman(ρ,dmgID\|attr-mag) = **+0.07 / +0.16 / −0.08**; **reverses under LPIPS control −0.19 / −0.38 / −0.11**; pooled looked +0.27/+0.48/+0.20 (between-attr proxy). incr CV-R² over realized-mag: large 1/6 (bedroom/LPIPS +0.43), **vanishes ffhq/ID (−0.010)** (`residual_conservative.json`, `rho_incremental_over_realized.json`) | **Strong — THE RECONCILED FLOOR** | Honest claim: **ρ is NOT an actionable per-direction risk signal**; magnitude dominates. Report the pooled-vs-within-unit gap + the LPIPS sign-flip as the reconciliation. |

**Honest limitations (own in §8):** ρ effect is weak & magnitude-confounded; no
controller shipped (the point); FFHQ encoder C5 transfer fails (~0.07); Park
NeurIPS23 reproduction fails (ρ≈0.04 — anchor for refutation); baselines
memory-constrained (K≤50, B=2); 743 audited runs from a **git-dirty** tree.

---

## 3. SECTION MAP (refutation; ~9 pp ICLR / TMLR flexible)

| § | Title | Headline used | Appendix backup |
|---|-------|---------------|-----------------|
| 1 | Intro: magnitude dominates geometry for edit risk | BASELINE + C4b leak hook | — |
| 2 | Related: latent-geometry-for-editing & its omitted baseline | Park, Arvanitidis, Haas24, Lobashev25, Cobb25 | — |
| 3 | Method: forward-mode JVP geometry instrument | M | derivations, FD eps-sweep |
| 4 | ρ is a real geometric signal (encodes semantics) | C2a (hardened), C2b (exact 2nd-order) | multi-CLIP/DINO, partial-Δ |
| 5 | ρ's residual over magnitude (population) | BASELINE (6/6 ordinal, 1/6 operational) + C3 FDR null | 0/41-vs-6/6 reconciliation, per-config null |
| 6 | ρ does not control (instance) — CORE | **C4 magnitude-matched residual** + C5 non-monotone (n=646 matched) | rho-gap bins, C4b leak, 95 versions |
| 7 | Why: population corr ≠ instance value | mechanism (magnitude dominance) | — |
| 8 | The magnitude baseline + protocol + limitations | BASELINE + released harness | full limitation log |
| 9 | Conclusion | the omitted-baseline caution | — |
| App | Robustness appendix | 95 control_v*, audits (743 runs, 0 fail) | — |

---

## 4. WORK PLAN (≈zero GPU; lifts BOTH venues identically)

Ordered by leverage. Steps 1–3 are BLOCKING (paper is self-contradictory until done).

0. **[BLOCKING, mins, CPU/tiny-GPU] Clean fixed-seed re-runs of the 4 load-bearing files** BEFORE any prose freezes: `control_rho_incremental_value_table_v1`, `control_predictive_diagnostic_utility_table_v1`, `control_predictive_permutation_null_table_v1`, `control_predictive_n20/rho_gap_stratification`. All currently `git_dirty=true`; the C5 lead file has `deterministic_set=false`. Re-run with a fixed seed on a clean tree so the lead numbers are reproducible.
1. **[BLOCKING, days, 0 GPU] Fix intro↔spine contradiction.** `paper/sections/01_introduction.tex` still sells dead C4 as "Contribution 3: compositional editing payoff" (L65–70) and leads with single-seed 92.86% (L98). Rewrite contributions to: (1) instrument, (2) ρ-encodes-semantics, (3) **magnitude-dominance** refutation + protocol (NOT "sufficient statistic"). Recast/cut `06_application.tex` compositional claim. **Single most certain reject trigger; free to fix.**
2. **[BLOCKING, hours, CPU] Demote C3 + report residual honestly.** Add per-config permutation-null pass-rate table to main text (6/21; 30/135=22%). Report β_ρ-with-magnitude (significant in 5 cells, e.g. ffhq/LPIPS +0.320 p=0.002) as the *weak residual* — this is why the claim is "dominance" not "sufficiency". Apply Benjamini-Hochberg FDR over the ~125-test family; report survivors.
3. **[DONE] Audit + name the magnitude baseline honestly.** Done in `leakage_audit.py` + `rho_incremental_over_realized.json`: 0/41 does NOT survive a clean baseline (ρ beats clean abs_alpha 6/6), so 0/41 was a leaky-4-feature-MAX artifact. Over the FAIR covariate (realized magnitude): ρ is ordinal-positive 6/6 but operationally weak (incr CV-R² large 1/6, vanishes 1/6). Reframed in BASELINE row. STILL TODO: bake C4 composition confound diagnostics (partial +0.235 NS; no_view flip −0.166) into a committed metrics.json.
4. **[mins GPU] Harden the two hooks.** Bootstrap 92.3% clustering ≥100 → mean±CI + ARI vs permutation null. Verify C2a survives partialling per-attr \|Δ\| and a shuffled-attribute null. (Kill criteria 2 & 3 if either fails.)
5. **[mins GPU] Fix FD anomaly.** `walltime` rel_err_vs_jvp=91.6% @ eps=1e-3 contradicts "FD≈JVP". Run eps-sweep, report honest agreement.
6. **[DONE] Recompute C5 ρ-gap stratification within TIGHT realized-magnitude bands.** Done in `c5_magnitude_matched.json` (rel≤0.05, n=646): the inverted-U **survives** matching (Q1 0.56 → Q3 0.60 → Q4 0.50/0.45; Spearman −0.19) → not a pure magnitude artifact. STILL TODO: bootstrap CI on the per-bin rates + a clean fixed-seed re-run (`deterministic_set` currently false).
6b. **[DONE] Measurement-validity subsection (C4b).** Matched-pair magnitude leak (low-ρ wins **0.765 small / 0.287 large**, n=2489/2311 — reproducible). Committed: `residual_conservative.json` (+ `matched_pair_magnitude_residual.json`, `rho_incremental_over_realized.json`, `c5_magnitude_matched.json`, `leakage_audit.json`). Leakage-free baseline = latent-step × probe-gain; realized Δ is a post-hoc covariate, never controlled-by-construction.
7. **[days, 0 GPU] Release artifact.** No README/setup yet. Ship: JVP geometry lib + generator wrappers; the protocol harness (95 configs are a real asset); a "is-this-latent-signal-actionable" eval with the audited magnitude baseline as reference. Honest: torch.func.jvp + wrappers, not algorithmic novelty. **Scope note:** the predictive/control universe uses only GANSpace/SeFa/random/prompt directions (InterFaceGAN/LatentCLR/DisCo absent) → state the claim is direction-agnostic, do not imply per-method editor coverage.
8. **[decision gate] Dual-submit.** TMLR whenever Steps 0–7 clean (~65–78%). ICLR 2027 only if FDR leaves ≥1 surviving C3 regime AND bootstrapped clustering ≥80% AND C2a survives partialling. Optional bonus GPU: scale least-bad cell (ffhq+LPIPS incremental CV-R², n=10→≥100) — **expected to fail**; never block TMLR on it.

---

## 5. KILL CRITERIA (route solely to TMLR if any fire)

1. After FDR, **zero** C3 cells survive → pure null + fragile predictor → ICLR reject (TMLR still fine).
2. Bootstrapped clustering < ~80% or CI overlaps permutation null → hook not Strong.
3. C2a encoder-invariance explained away by per-attr magnitude → the one clean positive is gone.
4. Optional n≥100 scale-up still fails operational AUROC → positive-core revival foreclosed (expected).

**Do NOT** keep buying GPU hoping geometry beats magnitude — decisive negative, 5 HIGH-confidence probes.

---

## 6. HONEST PROBABILITY (so we never chase a phantom again)

- TMLR after plan: **65–78%** (primary), conditional on FD anomaly fixed +
  `deterministic_set=true` re-runs + C2a surviving partialling.
- ICLR/NeurIPS after plan: **25–35%** (upside, same manuscript, ~free). The
  measurement-validity finding (C4b) is what lifts it toward the top of that range,
  NOT a stronger positive — the structural cap stays.
- "~40%" from before was **illusory** — it priced in a geometry>magnitude win the data lacks.
- The ICLR cap is **structural** (C3 null-sensitive; killed controller was our own
  hypothesis, not field-held; single-seed C2c hook). **No compute raises it.** Only
  framing + rigor do.
