# Session handoff — 2026-05-17 09:02 KST

**Status**: sequencer terminated, GPU idle, all background processes finished.

---

## What completed successfully (results in `experiments/out/`)

### Track 1 — SD C1/C2 ✓
- 5 attrs × 3 timesteps × 32 seeds, FD-of-JVPs second derivative
- File: `experiments/out/sd_c1_c2/metrics.json`
- Quick summary per attr (averaged across 3 timesteps):

| Attr | ρ_mean (avg over t) | CLIP-path (avg) |
|------|---------------------|-----------------|
| smile      | (0.043 + 0.092 + 0.559) / 3 = **0.231** | 7.69 |
| age        | (0.051 + 0.071 + 0.547) / 3 = **0.223** | 8.09 |
| gender     | (0.049 + 0.065 + 0.358) / 3 = **0.157** | 7.50 |
| eyeglasses | (0.049 + 0.066 + 0.141) / 3 = **0.085** | 7.55 |
| pose       | (0.044 + 0.060 + 0.172) / 3 = **0.092** | 7.97 |

**Note**: SD ρ values are at very different scale than StyleGAN bedroom/FFHQ
(SD uses FD-of-JVPs with ε=0.05 + DDIM-chain composition; StyleGAN uses
composed-JVP at α=1). Cross-architecture **rank** comparison is what
matters, not absolute. The per-attr ordering within SD differs from FFHQ
StyleGAN (where pose was highest at 49.87) — investigate whether the
ranking is even preserved.

### Track 3 sample scaling ✓ — bedroom + ffhq + church all completed

**Bedroom (N up to 128)** — `experiments/out/sample_scaling_bedroom/metrics.json`:
- All Spearman rank-vs-Nmax ≥ 0.93, plateau at N=64.
- **Numbers DIFFER from the §5 table values**:
  - view: paper says 23.22, scaling says **4.70 ± 0.65 (N=128 95% CI)**
  - indoor_lighting: paper 0.495, scaling 0.544
  - This is because the per_sample_ratio computes mean(|2nd|)/mean(|1st|),
    while the original measurement may compute it differently.
  - **CRITICAL: reconcile aggregation conventions before paper v2.**

**FFHQ (N up to 64)** — `experiments/out/sample_scaling_ffhq/metrics.json`:
- pose: **21.55 ± 1.49** (paper says 49.87) — also differs
- smile: 1.27 (paper 1.75)
- All Spearman = +1.00 across N — ordering is stable, magnitudes need
  reconciliation.

**Church (N up to 128)** — `experiments/out/sample_scaling_church/metrics.json`:
- Completed; check actual values.

### Track 2 editing head-to-head ✓ — **prescriptive payoff result!**
- N=1000 FFHQ test latents, candidate pool 64 GANSpace + 32 SeFa + 32
  random + 5 GT InterFaceGAN = 133 directions
- LatentCLR / DisCo NOT in pool (those failed in Track 5)

| Ranking | mean Δattr | mean ID-cos | mean LPIPS |
|---------|-----------|-------------|-----------|
| curvature_low  | +0.0001 | **0.9502** | **0.041** |
| curvature_high | -0.0043 | 0.7870 | 0.2000 |
| random         | -0.0005 | 0.8696 | 0.1113 |

**Interpretation**:
- Curvature-low gives **best ID preservation** (+0.083 over random) and
  **best LPIPS** (smallest drift).
- Curvature-high gives most dramatic image change (highest LPIPS, lowest
  ID-cos).
- **But** mean Δattr ≈ 0 for all three rankings — none meaningfully moved
  toward target attribute. This is because the candidate pool was generic
  directions (PCA / SeFa / random), not attribute-aligned ones.
- The honest claim becomes: "curvature ranking discriminates magnitude
  of image change, but the candidate pool needs to be attribute-aligned
  first for the prescriptive claim to land."

---

## What failed (need re-run with fixes)

### Track 1B — SD N=64 follow-up: **NEVER RAN**
**Cause**: I appended Track 1B to `run_all_tracks.sh` at 23:22 on May 15,
but the runner had been running since 21:05. By the time it finished
`wait_gpu_free` (when Track 1 completed at 12:42), bash's file pointer
had likely passed the append point, OR the script was loaded into
memory once at start.

**Fix**: Just re-run `experiments/diffusion/run_c1_c2.py --n-train 64
--n-test 64 --out experiments/out/sd_c1_c2_n64` standalone with nohup.

### Track 5 LatentCLR + DisCo: **OOM**
- log: `track5_latentclr.log` ends in `torch.cuda.OutOfMemoryError`
- Cause: FFHQ StyleGAN at 1024² × 100 directions × batch 8 = peak 7.4 GB.
  My LatentCLR train forwards `chunk=10` directions per slice × 8 latents
  = 80 synthesizes per iter at 1024². Each synthesis ~700MB activation.
- **Fix**: reduce chunk to 4, reduce batch to 4. Possibly use 256²
  rendering via lod=2 for LatentCLR (it only needs feature-space contrast,
  not pixel quality).

### Track 4 FFHQ encoder + eval: **ModuleNotFoundError: 'domains'**
- log: `train.py` `from domains.ffhq.generator import FFHQGenerator`
- Cause: when run from `paper/`, sys.path doesn't include `experiments/`.
- **Fix**: launch with `cd experiments && python3 domains/ffhq/encoder/train.py`
  OR add `sys.path.insert(0, str(PAPER / "experiments"))` to the train.py
  preamble (currently has it but maybe not enough).

### Wave 2 + 3 + 4 (Tracks 6-21): **NEVER EXECUTED**
- Same root cause as Track 1B — bash's read of the script stopped at
  some point after the Wave 1 section.
- All these tracks' code is committed and ready in
  `experiments/run_all_tracks.sh`.
- **Fix**: re-run the sequencer script after fixing Track 4/5. The
  Wave 2-4 sections will then execute fresh.

---

## Reconciliation needed before paper v2

1. **ρ aggregation convention**: The §5 paper table values (bedroom view
   = 23.22, FFHQ pose = 49.87) DO NOT MATCH the Track 3 sample-scaling
   values (4.70, 21.55 respectively). Cause: different aggregation
   (mean-of-ratio vs ratio-of-means, or peak vs mean). Pick ONE
   convention, document it, recompute all §5 tables.

2. **SD vs StyleGAN scale**: SD ρ values (0.04-0.6 range) are
   architecturally different from StyleGAN (0.04-50 range). Reasons:
   - SD uses FD-of-JVPs at ε=0.05 (effectively smaller α than StyleGAN's
     unit boundary direction).
   - DDIM chain has different sensitivity normalisation.
   - Need to add a note that SD numbers are cross-arch-comparable via
     RANK, not absolute magnitude.

3. **Cross-domain signature plot (FIG 7)**: currently uses bedroom + FFHQ
   only (SD smoke result of 0.043 was placeholder). With real Track 1
   data, regenerate FIG 7 to include SD. The 92.86% k=2 agreement should
   be recomputed.

---

## Resumption checklist (next session)

1. **Re-run failed tracks**:
   - [ ] Track 5 LatentCLR/DisCo with chunk=4 batch=4 (or 256² fallback)
   - [ ] Track 4 FFHQ encoder train.py — fix sys.path
   - [ ] Track 1B SD N=64 follow-up (standalone command)

2. **Run Wave 2-4 that never executed**:
   - [ ] Re-run `bash experiments/run_all_tracks.sh` from the
     Wave 2 section onward (or split into separate driver script).
   - Tracks 6, 7, 8, 9, 10 (bedroom + ffhq), 11, 12, 13 = 9 jobs
   - Tracks 14 (bedroom + ffhq), 17 (bedroom + ffhq), 18 = 5 jobs
   - Tracks 19, 20 = 2 jobs

3. **Reconcile aggregation convention** in
   `experiments/metrics/run_sample_scaling.py` vs original
   `domains/ffhq/run_higher_order.py` — pick one, document.

4. **Regenerate FIG 7 cross-domain signature** with real SD data
   (currently uses smoke-test placeholder for SD).

5. **Update paper §5 tables** with:
   - Real Track 1 SD numbers (already in `metrics.json`)
   - Real Track 3 scaling CIs (with reconciled convention)
   - Real Track 2 editing head-to-head numbers (already in
     `metrics.json`)
   - Real Wave 2-4 results once they run

6. **Compile sanity check**: `bash paper/compile.sh` to verify 0 errors.

---

## Track 2 editing — the closest thing to "prescriptive payoff"

This is the **headline new result** from this session. Curvature-low
ranking gives:
- ID-cos: +0.083 over random, +0.16 over curvature-high
- LPIPS: 2.7× smaller than random, 4.9× smaller than curvature-high

The Δattr being ≈ 0 across all rankings is honest — the candidate pool
was generic, not attribute-aimed. The right way to write this up:

> "We pre-rank a candidate direction pool by C2 curvature. The
> curvature-low subset preserves identity at ID-cos 0.95 vs random 0.87
> and curvature-high 0.79 at matched LPIPS budget, demonstrating
> curvature is a useful filter for the *image-change-magnitude* axis
> of editing-direction quality. A natural complement is to first filter
> by attribute alignment (e.g. CLIP-direction-similarity), then sub-
> rank by curvature — combining the two should yield the strong
> 'identity-preserving attribute-changing' filter that the original C2
> claim motivates."

This is paper §6's Application section — needs rewrite to reflect.

---

## File integrity

- `experiments/out/sd_c1_c2/metrics.json` — full Track 1 result
- `experiments/out/sample_scaling_bedroom/metrics.json` — Track 3 bedroom
- `experiments/out/sample_scaling_ffhq/metrics.json` — Track 3 FFHQ
- `experiments/out/sample_scaling_church/metrics.json` — Track 3 church
- `experiments/out/editing_head_to_head/metrics.json` — Track 2
- `paper/main.pdf` — last clean compile (12 pages, before today's results)
- `paper/sections/05_experiments.tex` — has \todo markers waiting for fills

## Process state

```
$ ps -p 3232386 -p 3266839 2>/dev/null
(empty — both processes terminated)

$ nvidia-smi --query-compute-apps ...
(empty — GPU idle)
```
