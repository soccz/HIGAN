# TMLR paper — design doc (literature-first, honest scope)

**Working title:** *Finite Differences Cannot Recover Curvature: An Exact Forward-Mode
Instrument for Higher-Order Geometry of Generative Models*

**Status:** design locked 2026-06-13. Supersedes the "edit-risk ρ controller" thesis
(shelved, fragile) and the "Challenge A overturns Park" framing (RETRACTED — see §0).

---

## 0. What we claim and what we DO NOT claim (overreach firewall)

We have been burned three times by strong-looking signals that were artifacts or did
not attach to a real external claim. This section is the firewall.

**We DO claim (each fully supported):**
1. Central/forward finite differences cannot recover a second-order pushforward
   (curvature) term at *any* step size, because truncation error and floating-point
   cancellation trade off with no usable window — proven on an analytic toy with known
   ground truth. (THEOREM + toy.)
2. An exact forward-mode **composed-JVP** computes the same second-order term to machine
   precision with no step size, at O(1) forward passes and no graph retention. (INSTRUMENT.)
3. The differential-geometry-of-generative-models literature is **split** in how it
   obtains second-order/geodesic quantities: some published methods use finite differences
   (Shao et al. 2018; Saito & Matsubara 2025), others use analytic/autodiff
   (Arvanitidis 2018; Yang/Arvanitidis 2018; Yu 2025), others are purely first-order
   (Park 2023; Chen 2018; Asyrp 2023). The FD-based ones are exposed to (1). (SURVEY, cited.)
4. On real generators the gap is operational: the exact instrument recovers stable
   second-order structure that the FD estimate destroys/renders unstable. (DEMONSTRATION.)

**CORRECTION 2026-06-13 (caught while designing the conclusion-flip experiment):**
Shao et al.'s "second difference" $g_{i+1}-2g_i+g_{i-1}$ is NOT a $/\delta^2$ curvature
estimate. Deriving the gradient of their discrete energy $E=\sum\|g(z_{i+1})-g(z_i)\|^2$
gives $\partial E/\partial z_j=-2J_g(z_j)^\top[g_{j+1}-2g_j+g_{j-1}]$ — the second
difference is the EXACT gradient of the discrete energy (a discrete Laplacian), with NO
division by $\delta^2$, hence NO cancellation catastrophe (refine $T$ safely, truncation
only). So **Shao is NOT exposed to our non-recoverability theorem.** Saito's score-JVP is
first-order FD ($/h$), well-conditioned (floor $\sqrt{\eps}$), also not catastrophically
exposed. NET: no surveyed published method uses the catastrophic $/\delta^2$ FD curvature.
Consequences: (a) Pillar C's $(g_+\!-2g_0\!+g_-)/\delta^2$ is the *naive curvature
estimate a practitioner would write*, NOT "Shao's primitive" — relabel. (b) The survey's
"exposed" column collapses; reframe as "the catastrophic regime is avoidable and mostly
avoided; the instrument is the principled tool when higher-order quantities ARE needed."
(c) A "flip a published conclusion" experiment is NOT honestly available and is dropped.
THESIS PIVOT: contribution = the exact instrument + a positive geometric finding it
enables that FD cannot resolve; non-recoverability is the motivation, not the headline.

**We DO NOT claim:**
- That any specific published numerical result is *wrong*. We show the *method* is
  unreliable and *which papers are exposed*; we do not re-run their exact experiments to
  assert their headline number is false (except where we explicitly reproduce, §5b).
- That Park NeurIPS23 is overturned. Park is **purely first-order** (Jacobian-SVD local
  basis via autodiff power iteration); it computes NO second-order quantity. Our earlier
  "ρ vs σ flips Park's conclusion" was a MISATTRIBUTION — the σ↔ρ relation is *our own
  2nd-order extension*, not Park's claim. RETRACTED. The SD σ↔ρ result survives only as a
  *self-labeled demonstration* (§5c), not as a published-conclusion flip.
- That FD-of-first-derivative (Saito's score-JVP) is as catastrophic as FD-of-second-
  derivative. It is not (it has an optimal h ~ √ε_machine). We are explicit: the
  non-recoverability theorem bites hardest on TRUE second-order FD (Shao's 2nd difference).

---

## 1. Problem & motivation

Latent-geometry analyses of generative models increasingly compute second-order
quantities: curvature of the generated manifold, geodesics (whose ODE contains the metric
derivative / Christoffel symbols), and second-order pushforward terms. How these are
*numerically obtained* is rarely scrutinized. We show it matters: the naive route (finite
differences) is provably non-recoverable for second-order, and a cheap exact instrument
exists. We map who in the field is exposed and demonstrate the operational gap.

## 2. Theory — FD non-recoverability (HAVE, in note/main.tex)

For a smooth f, the central-difference second derivative
`(f(α+ε) − 2f(α) + f(α−ε))/ε²` has error `= O(ε²)` (truncation) `+ O(ε_machine/ε²)`
(cancellation). The two terms cannot be simultaneously small: the optimal ε gives relative
error `~ ε_machine^{1/2}` for the *second* derivative — and in single precision (what GANs
/ diffusion models run in) this floor is large. The exact composed-JVP has no ε.

- Toy (analytic, known d²f): JVP rel-err 2.5e-17 vs best-FD 8.6e-9 (double); single
  precision FD floor ~1e7× worse. Table in note/.

## 3. Instrument — exact composed-JVP (HAVE)

`d²/dα² f(x+αv)|_{α=0}` via `jvp(λα: jvp(f,(α,),(1,))[1], (0,), (1,))`. Forward-mode,
two composed pushforwards, no Hessian materialization, no graph retention; O(1) forward
passes, memory ≈ a few activations. Verified exact on the toy.

## 4. Survey — how the field computes second-order (HAVE, agents 2026-06-13)

| Paper | venue | 2nd-order quantity | how obtained | exposed to FD-non-recov? |
|---|---|---|---|---|
| Shao, Kumar, Fletcher | CVPR-W'18 (1711.08014) | geodesic, discards Christoffel | **FD 2nd-difference** g_{i+1}−2g_i+g_{i−1}, T=10 | **YES (true 2nd-order FD)** |
| Saito & Matsubara | 2025 (2510.05509) | geodesic | **FD JVP** (s(x+hv)−s(x))/h | **YES (1st-deriv FD; diffusion)** |
| Saito & Matsubara | 2025 (2504.20288) | geodesic velocities | **FD** (explicit) | partial |
| Arvanitidis et al. | ICLR'18 (1710.11379) | geodesic ODE w/ Christoffel | **analytic** ∂g, numeric ODE integ (bvp5c) | no (metric closed-form) |
| Yang/Arvanitidis et al. | 2018 (1809.04747) | geodesic energy | **autodiff** (TF) | no |
| Yu et al. | 2025 (2504.06675) | geodesic, Euler–Lagrange | score + **RK4** | no |
| Park, Kwon, Choi, Jo, Uh | NeurIPS'23 (2307.12868) | **none — 1st-order** Jacobian-SVD | autodiff power-iter | n/a (first-order) |
| Chen et al. | AISTATS'18 (1711.01204) | none — 1st-order metric | n/a | n/a |
| Kwon/Jeong/Uh (Asyrp) | ICLR'23 (2210.10960) | none — 1st-order | CLIP-grad + MLP | n/a |

Takeaway: FD-second-order is a *live, published* practice (Shao classic; Saito modern/
diffusion), not a straw man — and a clean autodiff/analytic alternative also exists in the
same literature. That is exactly the setting where a non-recoverability result + a drop-in
exact instrument is "of interest."

## 5. Demonstrations

**(a) StyleGAN curvature structure via the exact instrument (HAVE, note/).**
Exact composed-JVP curvature decays monotonically across synthesis layers; 8/8 annotated
bedroom attributes, Spearman ∈ [−0.96, −0.80]. The instrument *reveals* structure.

**(b) Shao-style FD vs exact on StyleGAN — the real-method demonstration (DONE 2026-06-13, VALIDATED).**
Isolated Shao's core numerical primitive — the FD second difference of the generator along
a latent curve, `(g(z+δv) − 2g(z) + g(z−δv))/δ²` — and measured its accuracy vs the exact
composed-JVP `d²/dα² g(z+αv)` on StyleGAN-bedroom (256px, fp32), over 6 directions
(3 boundary attrs + 3 random) × 16 samples, sweeping δ.
- RESULT (out/shao_fd_vs_exact/metrics.json):
  - exact curvature (ground truth, step-free): mean 1.17.
  - FD second-difference error FLOOR = **45.2% at the best step (δ=3)**, bracketed on both
    sides (δ=8 → 74.5%, δ=5 → 57.3%; δ<3 rises monotonically). **No step recovers the
    curvature better than ~45%.**
  - At Shao's discretization regime (δ~0.3): **266.9%** error.
  - As δ→0: cancellation explodes (δ=0.01 → 2.5e4%, δ=0.001 → 2.0e6%).
  - SANITY (the adversarial check): the *first*-order central FD agrees with the exact
    JVP first-order magnitude to **2.4%** at δ=0.02 (clean U-curve). So the pipeline and the
    exact instrument are correct; the second-order blow-up is genuine non-recoverability,
    not a bug.
- HONEST SCOPE (firewall, §0): this proves Shao's *primitive* is non-recoverable on a real
  generator. It does NOT by itself prove Shao's *near-flat verdict* is false — that verdict
  is read off geodesic distances/MDS eigenvalues, which would need the full geodesic
  pipeline (FD-gradient vs exact-gradient geodesics → distances → MDS) to re-test. Claim =
  "the numerical primitive any such second-order conclusion rests on is unreliable; here is
  the exact fix." Optional upgrade (§5b+) = reproduce the geodesic pipeline to test verdict-flip.
- WHY strongest: TRUE second-order FD ⇒ the §2 theorem bites hardest; reproduces a
  *published numerical method* on a real generator; validated by the first-order sanity.

**(c) SD h-space σ↔ρ (HAVE, finegrain run 2026-06-13) — SUPPORTING, self-labeled.**
In a realistic SD h-space pipeline, the exact composed-JVP second-order ratio ρ correlates
strongly & consistently with the first-order singular value σ (Spearman mean −0.73, 5/6
seeds ≤ −0.82), while the FD-of-JVP estimate of the *same* ρ is sign-unstable (mean −0.18,
signs flip across seeds): FD destroys recoverable second-order structure. **Labeled as our
own 2nd-order construction, NOT a Park conclusion.**

**(d) OPTIONAL — Saito diffusion score-metric (NEW, modern second example).**
Their metric G = J^T J, J = ∇_x s_θ, with J·v by FD (s(x+hv)−s(x))/h. Compare to exact
`jvp(s_θ, x, v)` on SD; show the induced metric / geodesic-length element differs and the
"which path is shorter / interpolation metric" verdict is method-dependent in fp. Lower
priority because 1st-deriv FD is less catastrophic than (b)'s 2nd-deriv FD.

## 6. Honest scope statement (goes in the paper verbatim-ish)

We provide a numerical-analysis result and a tool. We do not assert that Shao et al. or
Saito & Matsubara reached false conclusions; we show their second-order quantities are
computed by a method that is provably non-recoverable, identify the exposed vs clean
sub-literature, and supply an exact drop-in instrument with reproducible evidence that the
choice changes the measured geometry on real generators.

## 7. Reproducibility package
Code (toy, StyleGAN, SD), configs, seeds, the exact-instrument module, all metrics.json.
TMLR values this; we ship it.

## 8. Build plan / TODO
- [x] §2 theorem + toy (note/)
- [x] §3 instrument (note/ + paper/experiments)
- [x] §4 survey (agents)
- [x] §5a StyleGAN curvature (note/)
- [x] §5c SD σ↔ρ finegrain (run 2026-06-13)
- [x] §5b Shao-style FD-vs-exact curvature on StyleGAN — DONE, VALIDATED (45% floor, 267%@Shao, 1st-order sanity 2.4%)
- [ ] §5d (optional) Saito score-metric FD-vs-exact on SD  ← NEXT (diffusion breadth)
- [ ] §5b+ (optional upgrade) geodesic-pipeline verdict-flip test
- [ ] §6 scope paragraph into main.tex
- [ ] expand note/main.tex → full TMLR draft (intro, survey table, demos, repro)
- [ ] reproducibility package + arXiv + TMLR submission
