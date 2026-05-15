# Progress log

Weekly status updates. Most recent first.

---

## Week 1 · 2026-05-15

**Goal**: bootstrap FFHQ domain, run first cross-domain saliency
figures, replicate C2 (∂²I/∂α² non-linearity) on FFHQ. Total wall
time today: ~2 h.

**Done**
- FFHQ generator wrapper (`experiments/domains/ffhq/generator.py`)
  built on top of genforce/interfacegan StyleGAN1 FFHQ
  (1024², 18 layers). JVP-safe synthesis patch ported with
  InterFaceGAN-specific layer indexing.
- First-order saliency (`run_saliency.py`) on 5 attributes
  (smile, age, pose, gender, eyeglasses), 8 base latents.
  Sanity checks pass: smile→mouth, eyeglasses→eyes, pose→contour.
- **C2 cross-domain validated** (`run_higher_order.py`):
  - smile 1.75, age 7.6, gender 8.7, eyeglasses 22.8, pose 49.9.
  - Pattern matches bedroom: texture ≪ 10, structural ≈ 20+.
  - **eyeglasses ratio ≈ bedroom view ratio** — same curvature
    signature for object-insertion attributes across domains.

**Key finding**
The non-linearity-ratio threshold (~20) for separating "structural /
topological" from "textural" attributes is **transferable across
generative domains**. This is much stronger than the bedroom-only
result and the central empirical lever for the paper's C2 claim.

**Not done / known gaps**
- LSUN church domain (planned next).
- Pair-wise disentanglement matrix on FFHQ (5×5 instead of bedroom's
  8×8).
- Quantitative saliency–segmentation IoU.

**Next**
1. FFHQ 5×5 disentanglement matrix.
2. LSUN church setup (genforce has stylegan_church_outdoor +
  HiGAN-trained boundaries).
3. Combined three-domain figure for the paper's "C1+C2 replicate"
  composite.

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
