# TSFM Contamination Audit — STATUS: explored, no clean finding, stepped back

**2026-05-30. Honest close-out. Do not re-grind without reading this.**

## What this was
Problem-first pivot after HIGAN: audit how much open TSFM zero-shot "SOTA" is
pretraining contamination vs generalization, via a publication-date / strict
logical negative control. Plan in `provenance/chronos_provenance.md`.

## What actually happened (Day 1–3, honest)
1. **Provenance locked (real, useful):** Chronos's own paper admits the zero-shot
   temporal-disjointness rule "was not strictly enforced"; exact Benchmark I
   (in-domain, 15) / II (zero-shot, 27) lists captured; ETT is in the ZERO-SHOT
   set. This part is solid and reusable if anyone revisits.
2. **Pipeline works, validated:** Chronos-t5-small 8GB inference + leakage-free
   harness. Validated on ETTh1 (Chronos MASE 0.58, DLinear 0.61 — both in the
   published ballpark), so the harness is trustworthy.
3. **The contamination hook collapsed under its own evidence:**
   - First 3-stratum run showed a big "Chronos crushes baseline" gap, but it was
     a **baseline-starvation artifact** — short eval series (m4_hourly ~1k pts)
     starve the trained DLinear; on long series (ETT ~17k) DLinear is competitive.
   - With a *validated* baseline, Chronos beats trained DLinear by only ~0.03–0.08
     MASE on ETT. There is **no large zero-shot advantage for contamination to
     inflate** — the hook's premise is gone.
   - The within-series **recall index** (error-curve flatness vs the non-memorizing
     reference) showed **no memorization signal** (tiny, negative, wrong-direction
     across strata).
   - The synthetic-floor contrast is **character-confounded** (synthetic = linear-
     friendly), so it cannot separate contamination from genuine nonlinear skill.

## The one suggestive-but-unconfirmed observation
On ETTh1 the Chronos−DLinear gap shrinks as DLinear gets more training data
(K=300 → full: gap −0.40 → −0.05), hinting that reported TSFM advantages are
inflated by under-trained baselines. BUT my DLinear is numerically **unstable
across training budgets** (non-monotone; K=1200 spikes), so this is NOT cleanly
established. Confirming it would need a stable, properly-tuned baseline — i.e.
more grinding, which we deliberately chose not to do.

## Decision (user, 2026-05-30): BANK + STEP BACK
- No clean publishable result here. Not a paper.
- This is the **second** consecutive project (after HIGAN) where the strong signal
  was an artifact and the real effect was thin. Recognized structural pattern:
  **adversarial measurement-validity auditing reliably deflates claimed effects →
  TMLR-tier, deflationary results**, not the top-venue positive the ambition wanted.
- The one real banked artifact from the whole arc = the **HIGAN short methods note**
  (`/mnt/20t/study/HIGAN/note/`): forward-mode JVP instrument + FD-non-recoverability.
  Clean, committed, reproducible.

## If anyone revisits TSFM later
The honest path is NOT the contamination angle (dead). It would be: a stable,
properly-tuned baseline suite + a clean demonstration of the baseline-starvation
inflation across many datasets. That is a real but known-adjacent critique
(DLinear-paper lineage). TMLR-tier at best. Decide if it's worth it before starting.

## Code/results (kept for reproducibility)
- `code/`: synthetic_surrogates, eval_harness, baselines (DLinear/NLinear),
  load_strata, recall_harness, run_3stratum, run_recall, bank_starvation.
- `results/`: run_3stratum.json, run_recall.json, bank_starvation.json.
- `provenance/chronos_provenance.md`.
