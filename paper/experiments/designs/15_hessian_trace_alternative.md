# Design — Hessian-trace as alternative C1 metric (Stanczuk 2024 precedent)

**Wave 3, Track 15.** The current C1 metric is the directional ratio
$\bar\rho_b = E[|d^2 G(b,b)|]/E[|dG b|]$ along *one specific direction* $b$.
Stanczuk et al. ICML 2024 measure curvature globally via the score
Jacobian's eigenvalue spectrum. Define a third metric: **Hutchinson-trace
estimator** of $tr(H_b)$ where $H_b$ is the Hessian along $b$, integrated
over a random Gaussian basis.

## Hypothesis

**H17.** The Hessian-trace estimator, computed via Hutchinson estimator
$tr(H) \approx \mathbb{E}_{v \sim N(0, I)}[v^T H v]$, is rank-correlated
with our directional $\bar\rho$ (Spearman r ≥ 0.7), indicating the two
are alternative geometric measurements of the same underlying signal.

## Prior art

- **Stanczuk et al. ICML 2024** [arXiv:2212.12611](https://arxiv.org/abs/2212.12611)
  uses the score-Jacobian spectrum to recover intrinsic data dimension —
  precedent for Hutchinson-trace-style measurements.
- **Hutchinson** 1990 estimator: $tr(A) = \mathbb{E}_v[v^T A v]$ for
  $v$ standard Gaussian.
- **Pearlmutter 1994 HVP** [paper](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf):
  HVP via two passes of fwd-bwd; we use composed JVP instead.

## Method

For each (attribute, domain), 8 noise seeds:

For each seed:
- 16 Gaussian probes $v_i \sim N(0, I)$ in the unit sphere of $\W$.
- Compute $v_i^T d^2 G(b, b) v_i$ via composed JVP --- but this is
  scalar so doesn't make sense. Reformulate as:
  $\text{trace}(H_b) \approx \frac{1}{K} \sum_k \|d^2 G(b, b)\|^2_{R^{3HW}}$
  where the norm is the L2 over pixels.

Actually for a single fixed $b$, the "Hessian" is $d^2 G/d\alpha^2$
along $b$, a vector in $R^{3HW}$, not a matrix. So the trace concept
needs to be reformulated:

**Reformulation.** Compute the *total directional curvature*
$\|d^2 G(b, b)\|_2$ summed over all pixels, normalised by
$\|dG b\|_2$. This is an alternative scalar to our mean-ratio.

Compare this normalised L2-curvature scalar to the mean-ratio
$\bar\rho$ across all (attribute, domain) pairs.

## Expected signal

L2-curvature and mean-curvature ratios should be rank-correlated
at $r \geq 0.95$ (they're two summary statistics of the same map).
This is a sanity check more than a discovery — establishes that
$\bar\rho$ isn't artefacted by the mean vs L2 choice.

## Compute budget

Reuse second-order maps already computed in Track 1 / Track 3. Only
the aggregation differs. <5 minutes.

## Deliverables

- `experiments/metrics/run_hessian_trace.py` (post-processing of
  saved JVP arrays).
- §5.exp-c1 robustness sentence.
