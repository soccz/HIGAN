"""Signed finite responses; raw tensors are retained before scalar summaries.

The square-observer split is an exact, finite counterfactual decomposition for
phi(y)=y**2. It is not the general local Hessian chain-rule decomposition.
"""

from __future__ import annotations

import math

import torch
from torch.func import jvp


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean().sqrt()


def components(base, single_a, single_b, joint):
    values = (base, single_a, single_b, joint)
    if base.numel() == 0 or any(v.shape != base.shape for v in values):
        raise ValueError("Nonempty, equal-shaped observations are required")
    if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in values):
        raise ValueError("Observations must be finite floating-point tensors")
    if any(v.device != base.device for v in values):
        raise ValueError("Observations must share a device")
    # This improves subtraction/reduction accuracy, not generator accuracy.
    base, single_a, single_b, joint = (v.to(torch.float64) for v in values)
    a, b, total = single_a - base, single_b - base, joint - base
    return base, a, b, total - a - b


def summarize(base, single_a, single_b, joint, *, atol=1e-6, rtol=0.1):
    if not math.isfinite(atol) or atol <= 0 or not math.isfinite(rtol) or rtol < 0:
        raise ValueError("atol must be positive and rtol nonnegative, both finite")
    y0, a, b, c = components(base, single_a, single_b, joint)
    additive = a + b
    total = additive + c
    a_size, b_size, c_size = (float(rms(v)) for v in (a, b, c))
    scale = a_size + b_size
    joint_size = float(rms(total))
    normalization_resolved = scale > 10 * atol
    responsive = max(scale, joint_size, c_size) > 10 * atol
    overlap = float((a * b).mean())
    cosine = overlap / (a_size * b_size) if min(a_size, b_size) > atol else None
    if cosine is not None:
        cosine = max(-1.0, min(1.0, cosine))

    # ||A+B+C||^2 - ||A||^2 - ||B||^2 = 2<A,B> + 2<A+B,C> + ||C||^2.
    squared_distance_gap = total.square().mean() - a.square().mean() - b.square().mean()
    overlap_term = 2 * (a * b).mean()
    nonlinear_term = 2 * (additive * c).mean() + c.square().mean()

    # Counterfactual observer response if the generator were additive at these
    # four points, plus the difference caused by the actual finite C.
    observer_only = 2 * a * b
    generator_contrast = 2 * (y0 + additive) * c + c.square()
    observed_c = (
        (y0 + total).square() - (y0 + a).square() - (y0 + b).square() + y0.square()
    )
    return {
        "a_rms": a_size,
        "b_rms": b_size,
        "joint_rms": joint_size,
        "interaction_rms": c_size,
        "interaction_relative": c_size / scale if normalization_resolved else None,
        "normalization_resolved": normalization_resolved,
        "responsive": responsive,
        "additive_at_tolerance": c_size <= atol + rtol * scale,
        "signed_cosine": cosine,
        "absolute_cancellation_gap_rms": float(rms(a.abs() + b.abs() - additive.abs())),
        "squared_distance_gap": float(squared_distance_gap),
        "squared_distance_overlap_term": float(overlap_term),
        "squared_distance_nonlinear_term": float(nonlinear_term),
        "squared_distance_closure_error": float(
            (squared_distance_gap - overlap_term - nonlinear_term).abs()
        ),
        "square_observer_interaction_rms": float(rms(observed_c)),
        "square_observer_only_rms": float(rms(observer_only)),
        "square_generator_contrast_rms": float(rms(generator_contrast)),
        "square_observer_closure_rms": float(
            rms(observed_c - observer_only - generator_contrast)
        ),
    }


def local_observer_split(function, observer, point, direction_a, direction_b):
    """C2-point chain-rule terms; no claim about an interval or activation kink."""
    y, dy_a = jvp(function, (point,), (direction_a,))
    _, dy_b = jvp(function, (point,), (direction_b,))
    _, mixed = jvp(
        lambda p: jvp(function, (p,), (direction_a,))[1],
        (point,),
        (direction_b,),
    )
    _, model_term = jvp(observer, (y,), (mixed,))
    _, observer_term = jvp(lambda v: jvp(observer, (v,), (dy_a,))[1], (y,), (dy_b,))
    return model_term, observer_term
