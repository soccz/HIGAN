# Progress log

Weekly status updates. Most recent first.

---

## Week 0 · 2026-05-09

**Goal**: bootstrap paper repo from scratch, get to a state where every
subsequent week can pick a single experiment / theory item and finish it.

**Done**
- Created repo structure under `/mnt/20t/study/HIGAN/paper/`:
  - `plan.md` — master 12-month plan with 6 claims + 5 baselines + 3 domains
  - `theory/00_overview.md` through `06_stratification_discovery.md` + `claims.md`
    (5 substantive chapters + 1 evidence ledger)
  - `related_work/RELATED_WORK.md` — 59-paper survey across 12 categories
    (collected via Explore subagent)
  - `paper/main.tex` + 7 `sections/*.tex` skeleton files
  - `paper/references.bib` with 22 starter entries (citations used in skeleton)
- Started TodoWrite tracking on the macro plan.
- Memory: existing `user_collaboration_style.md` and
  `site_design_system.md` carry over.

**Not done / known gaps**
- CVPR style file (`cvpr.sty`) not yet committed — needs to be downloaded
  from the official CVPR 2026 template page.
- No experiment code yet under `experiments/`. Bedroom code lives in the
  existing `higan_dev/` package; FFHQ and church directories are empty.
- Baselines all empty stubs.

**Next week (Week 1)**
1. Download CVPR style template + place under `paper/`. Compile the
   skeleton to confirm LaTeX builds end-to-end (will print a lot of
   `\todo` red flags, that's expected).
2. Bootstrap `experiments/domains/ffhq/` — load StyleGAN2 FFHQ
   pretrained, attach InterFaceGAN boundaries (smile, age, pose,
   glasses), patch synthesis for \jvp compatibility.
3. Reproduce the first-order saliency figure (§06 of report) on FFHQ
   as a sanity check.
4. Push first commit to GitHub (private repo or new public — TBD).

**Calibration**
- The Month 1 plan said: foundation + theory v1 + LaTeX skeleton + related
  work corpus + first FFHQ figure. We are on track for everything except
  the FFHQ figure, which is now next week's target.

---

<!-- Future weeks add entries above this line -->
