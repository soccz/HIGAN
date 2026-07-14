# Cover Letter — TMLR Submission

## Prior-submission disclosure

An earlier version of this work was previously submitted to TMLR and desk-rejected
(title: "When Finite Differences Flip the Sign of Second-Order Generative Geometry"). We
disclose this up front, and summarize below the substantive changes that make the present
manuscript a responsive revision rather than a resubmission of the same line.

## What changed since the prior version

1. **The headline claim was retracted and replaced.** The prior version's marquee result was a
   seed-dependent sign-inversion of a curvature ratio (n=6 seeds, sensitive to a near-zero
   outlier). That claim did not hold up and has been removed entirely; its evidence is not part
   of this submission. The contribution is now an *empirical step-selection trilemma*: across
   StyleGAN, the finite-difference step that best preserves curvature magnitude, the step that
   is least biased, and the step that best preserves the ranking of directions are three
   different steps, so no single step is defensible. This is a deterministic-plus-measured
   result, not a fragile point estimate.

2. **A downstream consequence is now closed.** The prior version documented a measurement
   problem without showing what decision it changes. The present version (i) names a published
   method, the Hessian Penalty (Peebles et al., 2020), whose per-sample estimate is exactly the
   non-recoverable second difference, and (ii) shows, using the exact instrument, that a
   plausible single-anchor curvature finding (partial correlation +0.45) collapses to a
   non-transferring cross-anchor effect (+0.17, 95% CI spanning zero) once the estimator
   confound is removed — a conclusion that changes when the measurement is fixed.

3. **Robustness was broadened.** The trilemma replicates across two additional StyleGAN-bedroom
   seeds, three full-resolution StyleGAN-FFHQ seeds, and lower-detail FFHQ controls. The
   specific step values are generator-dependent; the conflict between magnitude, bias, and rank
   is not.

4. **The self-deflating framing was removed.** Statements implying that the field is already
   safe, or that nothing depends on the result, have been deleted. The contribution is stated
   first-class, and the recommendations (use an exact or step-validated estimator; report the
   step axis if finite differences are used; replicate direction-level claims across anchors)
   are operational.

## Fit to TMLR's two criteria

The contribution — an exact, step-free instrument for second-order generator geometry, the
non-recoverability of the finite-difference alternative on real generators, and an
instrument-enabled replication audit — is supported by deterministic proof, an analytic
ground-truth check, and multi-generator empirical sweeps whose every number is reproducible
from the released artifacts. The at-risk audience is present-tense: practitioners who today
estimate generator curvature for geodesic interpolation, curvilinear editing, and Hessian
disentanglement priors.
