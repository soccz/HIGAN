"""Closed-form and adversarial checks of the actual measurement implementation."""

import unittest

import torch

from measure import components, local_observer_split, summarize


class MeasurementTests(unittest.TestCase):
    def tensor(self, values):
        return torch.tensor(values, dtype=torch.float64)

    def observe(self, f, alpha=1.0, beta=1.0):
        return tuple(
            self.tensor(f(s, t))
            for s, t in [(0, 0), (alpha, 0), (0, beta), (alpha, beta)]
        )

    def test_affine_crossloading_is_additive(self):
        result = summarize(*self.observe(lambda s, t: [s + t, s + 2 * t]))
        self.assertEqual(result["interaction_rms"], 0)
        self.assertTrue(result["additive_at_tolerance"])
        self.assertGreater(result["signed_cosine"], 0.9)
        self.assertEqual(result["squared_distance_gap"], 3)
        self.assertEqual(result["squared_distance_nonlinear_term"], 0)

    def test_affine_cancellation_is_not_nonlinearity(self):
        result = summarize(*self.observe(lambda s, t: [s - t, 2 * s - 2 * t]))
        self.assertAlmostEqual(result["signed_cosine"], -1)
        self.assertEqual(result["joint_rms"], 0)
        self.assertGreater(result["absolute_cancellation_gap_rms"], 0)
        self.assertEqual(result["interaction_rms"], 0)
        self.assertTrue(result["responsive"])

    def test_additive_nonlinear_function_has_no_mixed_effect(self):
        result = summarize(*self.observe(lambda s, t: [s * s, t * t]))
        self.assertGreater(result["a_rms"], 0)
        self.assertEqual(result["interaction_rms"], 0)

    def test_bilinear_signed_closed_form(self):
        for alpha in [-2.0, -0.25, 0.0, 0.5, 3.0]:
            for beta in [-1.0, 0.0, 0.25, 2.0]:
                _, _, _, mixed = components(
                    *self.observe(lambda s, t: [s + t, s * t], alpha, beta)
                )
                torch.testing.assert_close(mixed, self.tensor([0, alpha * beta]))

    def test_scale_reparameterization(self):
        original = self.observe(lambda s, t: [s + t, s * t], 0.5, -0.5)
        scaled = self.observe(lambda s, t: [4 * s + 0.25 * t, s * t], 0.125, -2.0)
        for a, b in zip(original, scaled):
            torch.testing.assert_close(a, b)

    def test_square_observer_creates_interaction_in_affine_model(self):
        result = summarize(*self.observe(lambda s, t: [s + t]))
        self.assertEqual(result["interaction_rms"], 0)
        self.assertEqual(result["square_observer_interaction_rms"], 2)
        self.assertEqual(result["square_observer_only_rms"], 2)
        self.assertEqual(result["square_generator_contrast_rms"], 0)

    def test_observer_decomposition_closes_with_real_interaction(self):
        for a, b in [(1.0, 1.0), (-1.0, 0.5), (0.25, -0.25)]:
            result = summarize(
                *self.observe(lambda s, t: [1 + s + t + 3 * s * t, s - t + s * t], a, b)
            )
            self.assertLess(result["square_observer_closure_rms"], 1e-12)
            self.assertLess(result["squared_distance_closure_error"], 1e-12)

    def test_local_chain_rule_against_independent_polynomial(self):
        f = lambda p: 1 + p[0] + p[1] + 3 * p[0] * p[1]
        model, observer = local_observer_split(
            f,
            lambda y: y.square(),
            self.tensor([0, 0]),
            self.tensor([1, 0]),
            self.tensor([0, 1]),
        )
        self.assertEqual(float(model), 6)
        self.assertEqual(float(observer), 2)
        # Coefficient of s*t in (1+s+t+3*s*t)^2 is independently 8.
        self.assertEqual(float(model + observer), 8)

    def test_hinge_boundary_finite_effect_survives_zero_local_mixed(self):
        result = summarize(
            *self.observe(lambda s, t: [max(0.0, s + t - 0.5)], 0.5, 0.5)
        )
        self.assertEqual(result["interaction_rms"], 0.5)
        self.assertFalse(result["additive_at_tolerance"])
        self.assertTrue(result["responsive"])
        self.assertFalse(result["normalization_resolved"])

    def test_no_response_has_no_relative_or_cosine_claim(self):
        result = summarize(*self.observe(lambda s, t: [5.0, 7.0]))
        self.assertFalse(result["responsive"])
        self.assertIsNone(result["interaction_relative"])
        self.assertIsNone(result["signed_cosine"])

    def test_invalid_observations_rejected(self):
        z = self.tensor([0.0, 1.0])
        for bad in [
            self.tensor([0]),
            self.tensor([float("nan"), 1]),
            self.tensor([float("inf"), 1]),
        ]:
            with self.assertRaises(ValueError):
                summarize(z, z, z, bad)
        with self.assertRaises(ValueError):
            summarize(z, z, z, z, atol=0)


if __name__ == "__main__":
    unittest.main()
