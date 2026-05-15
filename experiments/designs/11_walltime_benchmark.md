# Design — Forward-mode vs reverse-mode wall-clock benchmark

**Wave 2, Track 11.** Formalise the "JVP is the right tool" claim
into a head-to-head wall-clock table on a fixed task at fixed
accuracy. Reviewer-anticipated.

## Hypothesis

**H13.** At matched per-pixel accuracy ($\epsilon < 10^{-4}$ relative
error vs the JVP reference), per-pixel reverse-mode saliency via
`vmap+vjp` is **$\geq 10^3 \times$ slower** than `jvp` on FFHQ
StyleGAN at 1024².

For finite-difference $(G(wp + \epsilon b) - G(wp - \epsilon b)) / 2\epsilon$:
$\sim 2\times$ slower than JVP (two forward passes vs one composed),
with truncation error $O(\epsilon^2)$ that we must tune.

## Prior art

- **PyTorch func module** Yi & Wei NeurIPS 2022
  [arXiv:2210.09360](https://arxiv.org/abs/2210.09360) discusses
  forward vs reverse cost.
- **JAX documentation** has the canonical analysis: forward AD costs
  $O(n_\text{input})$, reverse AD costs $O(n_\text{output})$.
  For per-pixel saliency $n_\text{input} = 1$, $n_\text{output} =
  3HW$, the ratio is $3HW \approx 10^5$ in our favour.
- **GENIE** Dockhorn et al. NeurIPS 2022 makes the analogous
  argument for higher-order diffusion solvers.

## Method

Single direction $b$, single seed, FFHQ generator at 1024². Compute
the *same per-pixel saliency map* via three methods:

1. **Forward JVP** (reference): `torch.func.jvp(G.synthesize, (wp,), (b,))`.
2. **Per-pixel reverse mode** (vmap+vjp): for each output pixel
   $(c, h, w)$, compute $\partial G_{c,h,w}/\partial wp$ via VJP,
   take the sum over (c, h, w) of the appropriate slice.
   Implementation: `vmap` over the unit-vector basis of the
   output space.
3. **Finite-difference**: $(G(wp + \epsilon b) - G(wp - \epsilon b)) / 2\epsilon$
   with $\epsilon$ chosen so that the result matches JVP to
   $10^{-4}$ rel error.

Report:
- wall-clock per call (mean of 5 calls, exclude first warm-up call)
- peak GPU memory
- numerical agreement with JVP reference

## Expected signal

- JVP: ~40 ms (bedroom 256²) / ~250 ms (FFHQ 1024²) per call.
- FD: ~1.5× JVP (two forward passes).
- vmap+vjp at full per-pixel: at FFHQ 1024², the output has 3M
  scalars; vjp is $O(n_\text{output})$. Even with vmap-batching of
  ~256 pixels at a time, this is 12k batches × backward pass.
  Expected: hours, possibly OOM.

## Failure modes

1. The full per-pixel vmap+vjp may OOM at 1024². Mitigation: cap
   benchmark to bedroom 256² for the head-to-head, extrapolate
   theoretically to FFHQ. Or report partial-pixel benchmark (e.g.
   "first 1024 pixels' Jacobians take X seconds").
2. Finite-difference truncation error depends on $\epsilon$. We
   sweep $\epsilon \in \{10^{-3}, 10^{-4}, 10^{-5}\}$ and report
   the optimum.

## Compute budget

- 3 methods × 2 domains × 5 timed calls = 30 measurements,
  most of which are seconds; the slow vmap+vjp dominates. Likely
  a few hours total.

## Deliverables

- `experiments/method/run_walltime_benchmark.py`
- `experiments/out/walltime/` — table (ms, GB, agreement)
- §4 (Method) and §5.exp-walltime get the actual numbers.
