# 02 — Forward-mode autodifferentiation and the pushforward

## What `torch.func.jvp` computes

For a smooth function $f: \mathbb{R}^n \to \mathbb{R}^m$ and a primal
$x \in \mathbb{R}^n$ with a tangent $v \in \mathbb{R}^n$, the call

```python
y, ẏ = jvp(f, (x,), (v,))
```

returns the pair $(y, \dot y) = (f(x), \, df_x \cdot v)$. The forward-mode
machinery carries *dual numbers* $(x + \varepsilon v)$ through every
elementary operation of $f$, accumulating $\dot y$ by the chain rule
without ever materialising the full Jacobian.

Cost: one extra forward pass worth of compute, no extra memory beyond
storing intermediate primals + tangents (no backward graph).

## Why exactly forward mode for our problem

Define $f_b : \mathbb{R} \to \mathbb{R}^{3HW}$ by
$f_b(\alpha) := G(wp + \alpha b)$.
- Input dimension: $1$ (scalar $\alpha$).
- Output dimension: $3HW \approx 2 \times 10^5$.

Forward mode gives all $\partial f_b / \partial \alpha$ in *one* call.
Reverse mode would need $3HW$ separate backward calls (one per output
component) to recover the same information, or one call to compute the
inner product against a chosen image-space vector.

For per-pixel saliency (our quantity of interest), forward mode wins by
$\sim 10^5$ over a naive reverse approach. Functorch's `vmap` over reverse
mode can recover some of this, but with $O(\log)$ memory growth, whereas
forward mode is constant.

## Composing JVPs for higher-order derivatives

If `jvp(f, ...)` returns a function $f'$, then `jvp(f', ...)` returns
$f''$. For our scalar-input case:

```python
inner = lambda α: jvp(f_b, (α,), (1.,))[1]          # = f'(α)
img0, second = jvp(inner, (0.,), (1.,))             # img0 = f'(0), second = f''(0)
```

A subtlety: the first `jvp` returns `(y, dy)` but `dy` itself is the
*tangent* output of an autodiff-augmented forward, so wrapping it in an
outer `jvp` triggers a second-order tangent computation. PyTorch
implements this correctly via dual-tensor stacking.

In our practice, second-order JVP costs $\approx 2.2 \times$ a first-order
forward, and uses no more memory than first-order JVP.

## Practical gotchas

### Pre-trained generators rarely tolerate composed JVP out of the box

Forward-mode requires every elementary op to define its tangent rule and
to *avoid* operations that escape autodiff (e.g., conversion to numpy,
in-place writes, `.item()` reads). We discovered (and patched) one such
escape in genforce HiGAN's StyleGAN synthesis:

```python
# Before — kills dual tensor
def forward(self, w):
    lod = self.lod.cpu().tolist()    # → JVP fails
    ...

# After — cache as Python float at wrapper init
cached_lod = float(synth.lod.detach().cpu().item())
def jvp_safe_forward(self, w):
    lod = cached_lod                  # → just a Python number
    ...
```

This is a 5-line monkey-patch in `higan_dev/generator.py`. Similar patches
were needed for FFHQ and church generators when we ported them.

### Memory accounting

Forward mode stores the same activations as a plain forward, doubled by
the tangent. For a $256\times256$ StyleGAN bedroom, peak GPU memory for
one composed (second-order) JVP at batch 4 is $\approx 3$ GB; the
practical batch limit on an 8 GB card is around 8.

### Numerical noise vs finite difference

A finite-difference estimator $\hat S^{(1)} \approx (G(wp + \delta b) - G(wp - \delta b))/(2\delta)$
also approximates the pushforward. Compared to exact JVP:
- For *first order*, FD with $\delta = 1$ is within a few percent of JVP
  on smooth attributes. Curvature directions show larger error.
- For *second order*, FD is dramatically noisier: it requires
  $G(wp \pm \delta b)$ at two scales and subtracts cancellation, producing
  error $O(\delta^2)$ on the value and $O(1/\delta)$ on the derivative.
  JVP avoids this entirely.

A comparison table will appear in the experiments section.

## Connection to classical interpretability

Forward-mode autodifferentiation has appeared in deep-learning
interpretability in three places:

1. **Hessian-vector products** (Pearlmutter 1994 reformulated; modern
   tools like JAX/PyTorch `jvp+grad`): used in second-order optimisation
   (K-FAC, natural gradient, Shampoo). Our use is interpretation, not
   optimisation, but the mechanism is identical.
2. **Sensitivity analysis** of model outputs: standard, but typically
   reverse-mode for the scalar-output regime of classifiers.
3. **GAN inversion gradient flow**: e.g., Image2StyleGAN optimises the
   latent code via reverse mode through the generator. We use the same
   generator but with forward-mode for per-pixel sensitivities at fixed
   latent.

We are not aware of a prior paper that applies *composed forward-mode
JVP* to GAN latent space interpretation, particularly for measuring
extrinsic curvature. This is the closest thing to a methodological
"new bit" in our work, and we will defend it explicitly in related work.
