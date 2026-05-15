# Design — Park-NeurIPS23 Riemannian-metric reproduction + 2nd-order extension

**Wave 2, Track 7.** Reproduce the first-order pullback Riemannian
metric of Park, Kwon et al. (NeurIPS 2023) on SD v1.5 h-space, then
show our second-fundamental-form measurement provides information
their first-order analysis does not.

## Hypothesis

**H8 (reproduction).** Their reported result — that the pullback
metric $g(t) = J^\top J$ on h-space (with $J = \partial x_t/\partial h_t$)
has a coarse-to-fine evolution across DDIM steps and that the top
right-singular vectors of $J$ define a meaningful editing basis — holds
in our reimplementation.

**H9 (our delta).** When we compute the **second**-order tangent
$d^2 G(b, b)$ along the same top-SVD direction $b$ at the same step,
its magnitude (a) varies non-trivially across $b$ and (b) is
**negatively correlated** with the singular value of $b$ — i.e. the
*highest-energy* first-order direction is **not** the *highest-curvature*
direction. This would establish that first-order and second-order
geometric structures carry complementary information.

## Prior art

- **Park et al. NeurIPS 2023** [arXiv:2307.12868](https://arxiv.org/abs/2307.12868)
  — the closest precedent paper; constructs $g(t) = J^\top J$ via
  FD-of-JVP at multiple $t$, takes SVD, edits along top singular
  vectors. They use SD v1.5 too. Code:
  https://github.com/byungjun-kwon/Riemannian-Diffusion (or similar
  org/team page, to verify).
- **Stanczuk et al. ICML 2024** [arXiv:2212.12611](https://arxiv.org/abs/2212.12611)
  — second-order structure of the score, but in pixel/noise space
  (not h-space). Different but related.

## Method

1. **First-order reproduction.** For each of 8 noise seeds:
   - Set $t_{\text{edit}} = 25$ (= 0.5T on a 50-step schedule).
   - Build a probe basis $\{e_i\}_{i=1}^{32}$ of random unit
     directions in h-space.
   - Compute $J e_i = (x_0(\epsilon \cdot e_i) - x_0(-\epsilon \cdot e_i)) / (2\epsilon)$
     via two first-order JVPs (Park's exact construction).
   - Stack into $J \in \mathbb{R}^{3HW \times 32}$ (rectangular).
   - SVD: $J = U \Sigma V^T$. Top right-singular vectors = editing basis.

2. **Second-order extension (ours).** For the top-k=4 right-singular
   vectors $v_1, ..., v_4$:
   - Compute $d^2 G(v_i, v_i)$ via composed JVP at h-space (single-step,
     not full chain — for memory).
   - Report $\rho_i = |d^2 G(v_i, v_i)| / \sigma_i$ where $\sigma_i$ is
     the singular value.
   - Test rank correlation between $\sigma$ and $\rho$ across 32 random
     bases × 8 seeds = 256 (basis, seed) pairs.

3. **Visual qualitative check.** Render $\alpha$-sweeps along
   $v_1$ at the 8 seeds; compare visually to Park et al. Fig. 3 / 4.

## Expected signal

- **Reproduction**: singular-value spectrum of $J$ has effective rank
  ~10-30, matching Park's reported numbers.
- **Top SVD direction $v_1$**: visually corresponds to a global
  semantic edit (consistent with Park's "homogeneous" claim).
- **Curvature ranking**: Spearman correlation between $\sigma$ and
  $\rho$ is **negative** $\leq -0.3$ — high-energy directions have
  *lower* normalised curvature because the rectangle dimensions
  cancel out in the right way.

## Failure modes

1. Their exact algorithm uses Lanczos-style iterative SVD on the
   full $\mathbb{R}^{3HW} \times \mathbb{R}^{1280 \cdot 64}$ implicit
   matrix. Our rectangular 32-probe approach is a low-rank
   approximation; we should expect modest spectral disagreement.
2. SVD of $J$ in fp32 at 512² involves $\sim 6\times 10^5 \times 32$
   matrix — fine, ~75 MB.
3. If the spectrum doesn't match qualitatively, that's a failed
   reproduction — report as such, do not paper over.

## Compute budget

- 8 seeds × 32 probes × 2 JVPs (~26 s each) ≈ 110 min for the JVP pass.
- SVD: seconds.
- Second-order on top-4 SVD: 8 × 4 × ~30 s (composed JVP single step,
  not full chain) = 16 min.
- **Total ~2.5 h.** Sequential after Track 1 finishes.

## Deliverables

- `experiments/diffusion/run_park_repro.py`
- `experiments/out/sd_park_repro/` — singular-value plot, top-k
  editing-direction visualisations, $\sigma$-vs-$\rho$ scatter.
- §5.exp-c2 paragraph "Comparison to first-order metric" added.
