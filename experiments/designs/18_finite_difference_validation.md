# Design — Finite-difference validation of composed JVP

**Wave 3, Track 18.** Cross-check composed-JVP second-derivative
computation against a *third* independent method: finite-difference of
finite-differences. If all three match, our C1 numbers are not the
artefact of any one estimator.

## Hypothesis

**H20.** $|d^2 G(b, b)|$ computed by:
1. Composed JVP: `jvp(jvp(...))` --- our default,
2. FD of JVP: $(jvp|_{+\epsilon} - jvp|_{-\epsilon}) / (2\epsilon)$,
3. Second FD: $(G(wp + \epsilon b) - 2 G(wp) + G(wp - \epsilon b)) / \epsilon^2$,

agree to relative error $\leq 10^{-3}$ at $\epsilon \in [10^{-3}, 10^{-2}]$.
At smaller $\epsilon$, methods 2 and 3 degrade due to floating-point
cancellation (well-known).

## Prior art

Standard finite-difference numerical analysis. Pearlmutter 1994 (HVP
via composed AD) is the established forward-mode approach we use.

## Method

For each (attribute, domain) at 8 seeds:
- Compute the second derivative pixel map via all three methods at
  $\epsilon \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$ for
  the FD methods.
- Report per-pixel relative-error vs the composed-JVP reference,
  averaged over the map.

## Expected signal

- At $\epsilon = 10^{-2}$: rel error < $10^{-3}$ between all three.
- At $\epsilon = 10^{-5}$: cancellation errors dominate; FD methods
  diverge.

This validates the composed-JVP method as the correct reference and
shows the engineering tradeoffs of the FD alternatives.

## Failure modes

If even the largest $\epsilon$ shows > 1% disagreement between
composed-JVP and 2nd-FD, that would indicate a bug in our composed
JVP wrapper. Probability low (would have surfaced earlier), but the
check is cheap.

## Compute budget

3 methods × 5 ε values × 8 seeds × few attrs = ~30 min on bedroom.

## Deliverables

- `experiments/method/run_fd_validation.py`
- `experiments/out/fd_validation/` — error-vs-ε plots.
- §4 method appendix.
