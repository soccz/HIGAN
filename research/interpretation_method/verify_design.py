"""Analytic counterexamples for the interpretation-method design; CPU/stdlib only.

These tests check definitions and assumptions. They do not evaluate a trained
model, establish semantic faithfulness, or establish research novelty.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import unittest


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def residual(path, step, tangent):
    return tuple(
        end - start - step * velocity
        for end, start, velocity in zip(path(step), path(0.0), tangent)
    )


def interaction(function, alpha, beta):
    return tuple(
        joint - first - second + base
        for joint, first, second, base in zip(
            function(alpha, beta),
            function(alpha, 0.0),
            function(0.0, beta),
            function(0.0, 0.0),
        )
    )


def normalize(values):
    maximum = max(values)
    return tuple(value / maximum for value in values)


def correlation(first, second):
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    x = tuple(value - mean_first for value in first)
    y = tuple(value - mean_second for value in second)
    return sum(a * b for a, b in zip(x, y)) / (norm(x) * norm(y))


class DesignChecks(unittest.TestCase):
    measurements = {}

    def test_linear_absolute_saliency_is_not_interaction(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (2.0, -4.0, 1.0, -5.0)

        def affine(s, t):
            return tuple(7.0 + s * x + t * y for x, y in zip(a, b))

        combined = normalize(tuple(abs(x + y) for x, y in zip(a, b)))
        abs_a = normalize(tuple(abs(x) for x in a))
        abs_b = normalize(tuple(abs(x) for x in b))
        expected = normalize(tuple(x + y for x, y in zip(abs_a, abs_b)))
        old_corr = correlation(combined, expected)
        self.assertLess(old_corr, 0.99)
        self.assertEqual(interaction(affine, 0.5, -2.0), (0.0,) * 4)
        self.measurements["linear_counterexample"] = {
            "absolute_saliency_correlation": old_corr,
            "finite_interaction_norm": 0.0,
        }

    def test_bilinear_interaction_matches_closed_form(self):
        def function(s, t):
            return (s + t, s * t)

        max_error = 0.0
        for alpha in (-2.0, -0.25, 0.0, 0.5, 3.0):
            for beta in (-1.5, 0.0, 0.25, 2.0):
                actual = interaction(function, alpha, beta)
                error = norm((actual[0], actual[1] - alpha * beta))
                max_error = max(max_error, error)
                self.assertLessEqual(error, 1e-12)
        self.measurements["bilinear_max_error"] = max_error

    def test_constant_direction_order_and_interaction_differ(self):
        def function(s, t):
            return (s * s + t, s * t)

        start = (1.0, -2.0)
        a = (0.5, 0.0)
        b = (0.0, 2.0)
        ab = tuple((z + x) + y for z, x, y in zip(start, a, b))
        ba = tuple((z + y) + x for z, x, y in zip(start, a, b))
        self.assertEqual(function(*ab), function(*ba))
        self.assertEqual(interaction(function, 0.5, 2.0), (0.0, 1.0))
        self.measurements["constant_directions"] = {
            "order_difference": 0.0,
            "finite_interaction_norm": 1.0,
        }

    def test_direction_units_preserve_matched_intervention(self):
        def path(t):
            return (t, t * t)

        step = 0.5
        reference = residual(path, step, (1.0, 0.0))
        ratios = {}
        for scale in (0.25, 1.0, 4.0):
            def scaled_path(t):
                return path(scale * t)

            self.assertEqual(scaled_path(step / scale), path(step))
            self.assertEqual(
                residual(scaled_path, step / scale, (scale, 0.0)), reference
            )
            ratios[str(scale)] = 2.0 * scale * scale / abs(scale)
        self.assertEqual(len(set(ratios.values())), 3)
        self.measurements["direction_rescaling"] = {
            "raw_second_over_first": ratios,
            "matched_residual_norm": norm(reference),
        }

    def test_nonlinear_coordinates_require_path_transport(self):
        def transform(x, y):
            return (x, y + x * x)

        def inverse(u, v):
            return (u, v - u * u)

        for step in (-1.0, -0.25, 0.0, 0.5, 1.0):
            original = (step, 0.0)
            self.assertEqual(inverse(*transform(*original)), original)
        naive_straight = inverse(0.5, 0.0)
        self.assertNotEqual(naive_straight, (0.5, 0.0))
        self.measurements["coordinate_transport"] = {
            "transported_endpoint": inverse(*transform(0.5, 0.0)),
            "new_straight_endpoint": naive_straight,
        }

    def test_zero_local_second_derivative_does_not_bound_finite_error(self):
        def hinge_path(t):
            return (t + max(0.0, t - 0.5),)

        self.assertEqual(residual(hinge_path, 0.25, (1.0,)), (0.0,))
        self.assertEqual(residual(hinge_path, 1.0, (1.0,)), (0.5,))
        self.measurements["hinge"] = {
            "local_second_derivative": 0.0,
            "residual_at_1": 0.5,
        }

    def test_sampled_agreement_does_not_certify_interval(self):
        def path(t):
            return (t + math.sin(4.0 * math.pi * t) ** 2,)

        errors = [norm(residual(path, t, (1.0,))) for t in (0.25, 0.5, 0.75, 1.0)]
        unseen_error = norm(residual(path, 0.125, (1.0,)))
        self.assertLess(max(errors), 1e-12)
        self.assertAlmostEqual(unseen_error, 1.0)
        self.measurements["finite_grid"] = {
            "max_sampled_residual": max(errors),
            "unsampled_residual": unseen_error,
        }

    def test_rank_deficiency_requires_output_or_subspace_condition(self):
        def generator(x, y):
            return x

        def encoder(output):
            return (output, 0.0)

        self.assertEqual(generator(*encoder(generator(2.0, 3.0))), 2.0)
        for direction in ((0.0, 1.0), (1.0, 2.0)):
            output_tangent = direction[0]
            pulled_tangent = (output_tangent, 0.0)
            self.assertNotEqual(pulled_tangent, direction)
            self.assertEqual(pulled_tangent[0], output_tangent)
        self.measurements["rank_deficiency"] = {
            "full_latent_identity_possible": False,
            "output_tangent_preserved": True,
        }

    def test_reconstruction_and_absolute_response_hide_direction_sign(self):
        original_z = 1.0
        reconstructed_z = -math.sqrt(original_z * original_z)
        original_response = 2.0 * original_z
        fixed_direction_response = 2.0 * reconstructed_z
        self.assertEqual(original_z ** 2, reconstructed_z ** 2)
        self.assertEqual(abs(original_response), abs(fixed_direction_response))
        self.assertNotEqual(original_response, fixed_direction_response)
        transported_direction = -0.5 * original_response  # DE(1) DG(1) b
        transported_response = 2.0 * reconstructed_z * transported_direction
        self.assertEqual(transported_response, original_response)
        self.measurements["encoder_branch"] = {
            "original_response": original_response,
            "same_numeric_direction_response": fixed_direction_response,
            "transported_direction_response": transported_response,
        }

    def test_observer_can_create_nonlinearity(self):
        # F(s,t)=s+t is linear, but observing phi(F)=F**2 creates interaction.
        raw = interaction(lambda s, t: (s + t,), 0.5, 2.0)
        observed = interaction(lambda s, t: ((s + t) ** 2,), 0.5, 2.0)
        self.assertEqual(raw, (0.0,))
        self.assertEqual(observed, (2.0,))
        self.measurements["observer"] = {
            "raw_interaction": raw,
            "squared_observation_interaction": observed,
        }

    def test_path_acceleration_is_not_curve_curvature(self):
        # q(t)=(exp(t),0) lies on a straight line; at zero q'=q''=(1,0).
        velocity = (1.0, 0.0)
        acceleration = (1.0, 0.0)
        projection = sum(a * v for a, v in zip(acceleration, velocity)) / norm(velocity) ** 2
        normal = tuple(a - projection * v for a, v in zip(acceleration, velocity))
        raw_ratio = norm(acceleration) / norm(velocity)
        curve_curvature = norm(normal) / norm(velocity) ** 2
        self.assertEqual(raw_ratio, 1.0)
        self.assertEqual(curve_curvature, 0.0)
        self.measurements["straight_curve_variable_speed"] = {
            "second_over_first": raw_ratio,
            "curve_curvature": curve_curvature,
        }


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DesignChecks)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    source = Path(__file__)
    report = {
        "scope": "analytic design checks only; no trained-model or novelty validation",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "checks_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "design_sha256": hashlib.sha256(source.with_name("DESIGN.md").read_bytes()).hexdigest(),
        "measurements": DesignChecks.measurements,
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
