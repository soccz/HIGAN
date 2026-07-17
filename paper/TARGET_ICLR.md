# TARGET LOCK v2 — TMLR (primary) + ICLR/NeurIPS (upside) — REFUTATION framing

**Re-locked 2026-05-30 after two adversarial workflows (13-agent claim audit +
10-agent strategy panel). Supersedes the "prediction–control gap" v1 spine,
which an adversarial audit broke: ρ never beats edit-magnitude (0/41 configs),
so there is no "prediction" to gap against. This document is the single source
of truth. If a draft sentence does not serve the through-line below, it is out.**

> Why v1 died (verified, do not re-litigate): `control_failure_boundary_table_v4`
> shows ρ-AUROC minus best simple (edit-magnitude / step-size) baseline = **0/41
> positive**. C4 composition: bedroom geometry Spearman +0.480 but raw-magnitude
> **−0.658 (stronger)**; partialling magnitude → +0.235 **NS**. Permutation null:
> only **30/135 (22%)** rows separate; readiness = `predictive_permutation_null_sensitive`.
> Conclusion: **magnitude is the sufficient statistic; geometry adds nothing for
> edit risk.** This is now the paper's thesis, not its failure.

---

## 0. Venue (no more chasing a nonexistent positive)

- **PRIMARY: TMLR** (rolling). Bar = "claims supported by clear evidence" + "of
  interest to some audience"; **no novelty/SOTA gate**. Exactly fits a rigorous
  honest null. Est. accept **70–80%** after the work below.
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
> latent spaces, we show that **edit magnitude is a sufficient statistic for
> perceptual/identity edit damage**: latent **curvature ρ, though it cleanly and
> encoder-invariantly encodes semantic structure, provides no incremental
> predictive or control value over trivial edit magnitude** — across
> architectures, encoders, and identity metrics (2221 matched pairs). We deliver
> the **omitted magnitude baseline** that latent-geometry-for-editing methods must
> beat, plus a **reusable evaluation protocol**.

Arc (each step earns the next):
1. **Instrument** — forward-mode JVP geometry: exact ρ in one pass, cross-arch.
2. **ρ is a real geometric signal** — encodes semantics, encoder-invariant (C2a).
3. **Yet ρ does not beat magnitude (population)** — 0/41 AUROC; C3 null-sensitive.
4. **And ρ does not control (instance)** — flat ρ-gap stratification, n=2221.
5. **Why** — population correlation ≠ instance value; magnitude dominance.
6. **The contribution** — magnitude baseline + protocol; honest limitations.

**Anti-collapse rule**: main text carries the corrected ledger numbers below ONLY.
The 95 control_v* versions live in a Robustness Appendix that defends, never narrates.

---

## 2. CLAIM LEDGER v2 (corrected; every number adversarially re-verified)

| # | Claim | Evidence (verified) | Strength | Notes / fixes applied |
|---|-------|---------------------|----------|-----------------------|
| **M** | Exact per-pixel pushforward in one forward pass; composed higher-order JVP; on StyleGAN + Diffusion | JVP 45.6 ms/img **(bedroom@256 only)**; determinism bit-identical 17 dp | **Strong (tool)** | **2 architecture families** (StyleGAN, Diffusion), 4 generators — NOT "3 architectures". FIX FD anomaly (rel_err 91.6% @ eps=1e-3) before release; commit a determinism artifact; `deterministic_set` currently false |
| **C2a** | ρ encodes semantics, **encoder-invariant** | multi-CLIP B/L/H Pearson 0.97–0.99; DINOv2 0.94/0.98 | **Moderate** | DOWNGRADED: Pearson is **outlier-driven** (drop 'view'/'pose' → ~0.10); bedroom Spearman 0.29–0.43 **NS**. Must **partial out per-attr \|Δ\|** and report Spearman + shuffle-null. This is the load-bearing positive — harden it. |
| **C2b** | Structure persists at higher order & in diffusion | **exact 2nd-order: pose 14.5× vs smile 0.96×**; SD signal across 9 timesteps (`sd_c1_c2_full_t`) | **Moderate** | LEAD with the **exact** 2nd-order contrast. 368× (3rd) is **FD-on-JVP, n=8, no CI** → demote to descriptive. SD = signal **exists**; **gap untested** in diffusion (do not imply it generalizes). |
| **C2c** | Curvature signature transfers across domains | k=2 clustering **92.3%** (12/13, bedroom+ffhq) | **Weak/Suggestive** | FIX: was 92.86%/13-14 (wrong; n=13, misassigns ffhq-eyeglasses). **Single-seed, no CI.** Bootstrap ≥100 + ARI vs label-permutation null. Drops to 77.8% incl SD. |
| **C3** | ρ predicts damage rank (population) but **does not beat magnitude** | FFHQ-only Spearman +0.38±0.08 (CLIP)/+0.32±0.09 (ArcFace); 3-domain CLIP ~+0.41, std ~0.15, spread bedroom +0.28→church +0.58; **null-sensitive 6/21** | **Weak, regime-dependent** | FIX label: the ±0.08 set is **FFHQ-only** (ArcFace=face). Report FDR over the ~125-test family; state which cells survive. Frame: ρ correlates but **adds 0 over magnitude (0/41 AUROC)**. |
| **C4** | ρ is **not a usable instance selector** | low-ρ win-rate 0.587 (CLIP)/0.570 (ArcFace) | **Strong (weak-but-sig, not coin flip)** | FIX: 0.587 is **~9 SE > 0.5, p<1e-4** — weak-but-significant POSITIVE, *not* "coin flip". Frame as "transfers only weakly; <0.59 ⇒ not operationally usable". |
| **C5** | **The gap is fundamental** — ρ-gap gives no edit-selection skill at any separation | **FLAT** stratification: Q1 0.551 → Q4 0.540, Spearman **−0.09**, **n=2221 pairs** | **Strong — LEAD RESULT** | This is the strongest, best-powered, least-attackable result. **Lead the negative with this**, add bootstrap CI on the slope. |
| **BASELINE** | **Edit magnitude is the sufficient statistic** geometry must beat | magnitude CV-R²≈+0.54; **0/41** ρ-beats-baseline; magnitude dominates C4 composition (−0.658) | **Strong — THE CONTRIBUTION** | Promote magnitude to the paper's explicit reference baseline. "We provide the baseline the subfield omits." |

**Honest limitations (own in §8):** ρ effect is weak & magnitude-confounded; no
controller shipped (the point); FFHQ encoder C5 transfer fails (~0.07); Park
NeurIPS23 reproduction fails (ρ≈0.04 — anchor for refutation); baselines
memory-constrained (K≤50, B=2); 743 audited runs from a **git-dirty** tree.

---

## 3. SECTION MAP (refutation; ~9 pp ICLR / TMLR flexible)

| § | Title | Headline used | Appendix backup |
|---|-------|---------------|-----------------|
| 1 | Intro: magnitude is all you need for edit risk | BASELINE + C5 hook | — |
| 2 | Related: latent-geometry-for-editing & its omitted baseline | Park, Arvanitidis, Haas24, Lobashev25, Cobb25 | — |
| 3 | Method: forward-mode JVP geometry instrument | M | derivations, FD eps-sweep |
| 4 | ρ is a real geometric signal (encodes semantics) | C2a (hardened), C2b (exact 2nd-order) | multi-CLIP/DINO, partial-Δ |
| 5 | ρ does not beat magnitude (population) | C3 + 0/41 AUROC + FDR null table | per-config null pass-rates |
| 6 | ρ does not control (instance) — CORE | **C5 flat n=2221** + C4 weak | rho-gap bins, 95 versions |
| 7 | Why: population corr ≠ instance value | mechanism (magnitude dominance) | — |
| 8 | The magnitude baseline + protocol + limitations | BASELINE + released harness | full limitation log |
| 9 | Conclusion | the omitted-baseline caution | — |
| App | Robustness appendix | 95 control_v*, audits (743 runs, 0 fail) | — |

---

## 4. WORK PLAN (≈zero GPU; lifts BOTH venues identically)

Ordered by leverage. Steps 1–3 are BLOCKING (paper is self-contradictory until done).

1. **[BLOCKING, days, 0 GPU] Fix intro↔spine contradiction.** `paper/sections/01_introduction.tex` still sells dead C4 as "Contribution 3: compositional editing payoff" (L65–70) and leads with single-seed 92.86% (L98). Rewrite contributions to: (1) instrument, (2) ρ-encodes-semantics, (3) magnitude-sufficient-statistic refutation + protocol. Recast/cut `06_application.tex` compositional claim. **Single most certain reject trigger; free to fix.**
2. **[BLOCKING, hours, CPU] Demote C3 honestly.** Add per-config permutation-null pass-rate table to main text (6/21; 30/135=22%). Apply Benjamini-Hochberg FDR over the ~125-test family; report survivors. Converts the reviewer's own-null attack into a pre-empted strength.
3. **[BLOCKING, hours, CPU] Make magnitude the explicit baseline.** Bake magnitude-confound diagnostics into committed metrics.json (C4 partial +0.235 NS; no_view flip −0.166; 0/41 AUROC; magnitude CV-R² +0.54). Reframe "our signal fails" → "magnitude is the baseline the subfield must beat".
4. **[mins GPU] Harden the two hooks.** Bootstrap 92.3% clustering ≥100 → mean±CI + ARI vs permutation null. Verify C2a survives partialling per-attr \|Δ\| and a shuffled-attribute null. (Kill criterion if either fails.)
5. **[mins GPU] Fix FD anomaly.** `walltime` rel_err_vs_jvp=91.6% @ eps=1e-3 contradicts "FD≈JVP". Run eps-sweep, report honest agreement.
6. **[days, 0 GPU] Strengthen C5 mechanism.** Bootstrap CI on flat-slope; connect population-corr vs instance-noise. Frame as measurement-validity / benchmarking methodology.
7. **[days, 0 GPU] Release artifact.** No README/setup yet. Ship: JVP geometry lib + 3 generator wrappers; the protocol harness (95 configs are a real asset); a "is-this-latent-signal-actionable" eval with magnitude as reference baseline. Position as released instrument (honestly torch.func.jvp + wrappers), not algorithmic novelty.
8. **[decision gate] Dual-submit.** TMLR whenever Steps 1–7 clean (~70–80%). ICLR 2027 only if FDR leaves ≥1 surviving C3 regime AND bootstrapped clustering ≥80%. Optional bonus GPU: scale least-bad cell (ffhq+LPIPS incremental CV-R², n=10→≥100) — **expected to fail**; never block TMLR on it.

---

## 5. KILL CRITERIA (route solely to TMLR if any fire)

1. After FDR, **zero** C3 cells survive → pure null + fragile predictor → ICLR reject (TMLR still fine).
2. Bootstrapped clustering < ~80% or CI overlaps permutation null → hook not Strong.
3. C2a encoder-invariance explained away by per-attr magnitude → the one clean positive is gone.
4. Optional n≥100 scale-up still fails operational AUROC → positive-core revival foreclosed (expected).

**Do NOT** keep buying GPU hoping geometry beats magnitude — decisive negative, 5 HIGH-confidence probes.

---

## 6. HONEST PROBABILITY (so we never chase a phantom again)

- TMLR after plan: **70–80%** (primary).
- ICLR/NeurIPS after plan, refutation framing: **25–35%** (upside, same manuscript, ~free).
- "~40%" from before was **illusory** — it priced in a geometry>magnitude win the data lacks.
- The ICLR cap is **structural** (C3 null-sensitive; killed controller was our own hypothesis, not field-held; single-seed hook). **No compute raises it.** Only framing + rigor do.
