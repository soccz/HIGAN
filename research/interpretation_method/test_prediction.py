"""Analytic, leakage, and decision checks for held-out joint prediction."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.func import jvp

from prediction import evaluate_response, fit_model, single_features, vector_predictions
from run_prediction import run, validate_protocol


class PredictionTests(unittest.TestCase):
    def test_bilinear_signed_extrapolation_and_mixed_jvp(self):
        x = torch.zeros(2, dtype=torch.float64)
        a, b = torch.eye(2, dtype=torch.float64)
        f = lambda z: torch.stack([z[0] + z[1], 3 * z[0] * z[1]])
        mixed = jvp(lambda z: jvp(f, (z,), (a,))[1], (x,), (b,))[1]
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                p, t = 0.01, 0.1
                probe = (
                    f(x + p * sa * a + p * sb * b)
                    - f(x + p * sa * a)
                    - f(x + p * sb * b)
                    + f(x)
                )
                target = (
                    f(x + t * sa * a + t * sb * b)
                    - f(x + t * sa * a)
                    - f(x + t * sb * b)
                    + f(x)
                )
                pred = vector_predictions(probe, mixed, t, p, (sa, sb), 100)
                for method in ("local_mixed", "probe_quadratic", "probe_calibrated"):
                    torch.testing.assert_close(pred[method], target)

    def test_unseen_activation_boundary_defeats_all_zero_probe_methods(self):
        z = torch.zeros(1, dtype=torch.float64)
        vectors = vector_predictions(z, z, 0.1, 0.01, (1, 1), 10)
        # max(0, s+t-0.15): singles and 0.01 joint do not expose the target kink.
        outcome = evaluate_response(
            torch.tensor([0.05]), vectors, {k: 0 for k in vectors}, 0.2, 1e-6, 0.1
        )
        self.assertFalse(outcome["actual_additive"])
        self.assertTrue(all(outcome["adopt"].values()))

    def test_vector_fit_minimizes_normalized_squared_error(self):
        rows = [
            {
                "seed": 1,
                "scale": 0.1,
                "normalized_probe_target_dot": 6.0,
                "normalized_probe_squared": 2.0,
                "single_features": [0, 0, 0, 1],
                "actual_relative": 0.2,
            }
        ]
        protocol = {
            "development_seeds": [1],
            "evaluation_seeds": [2],
            "target_scales": [0.1],
            "ridge_penalty": 1.0,
        }
        self.assertAlmostEqual(
            fit_model(rows, protocol)["by_scale"]["0.1"]["coefficient"], 3
        )
        rows[0]["normalized_probe_target_dot"] = -6
        self.assertEqual(fit_model(rows, protocol)["by_scale"]["0.1"]["coefficient"], 0)

    def test_evaluation_anchor_cannot_train_model(self):
        with self.assertRaises(ValueError):
            fit_model(
                [{"seed": 2}], {"development_seeds": [1], "evaluation_seeds": [2]}
            )

    def test_no_false_success_from_zero_correction(self):
        actual = torch.tensor([1.0, -1.0])
        pred = {"additive": torch.zeros(2), "perfect": actual}
        result = evaluate_response(
            actual, pred, {"additive": 0, "perfect": 0.5}, 2, 1e-6, 0.1
        )
        self.assertEqual(result["relative_errors"]["additive"], 0.5)
        self.assertEqual(result["relative_errors"]["perfect"], 0)
        self.assertTrue(result["adopt"]["additive"])
        self.assertFalse(result["actual_additive"])

    def test_cancellation_features_do_not_use_joint_output(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        denominator, features = single_features(a, -a, 0.1)
        self.assertGreater(denominator, 0)
        self.assertAlmostEqual(features[1], -1)
        self.assertEqual(features[3], 1)

    def test_unresolved_response_and_invalid_extrapolation_rejected(self):
        z = torch.zeros(2)
        with self.assertRaises(ValueError):
            single_features(z, z, 0.1)
        with self.assertRaises(ValueError):
            vector_predictions(z, z, 0.01, 0.1, (1, 1), 1)

    def test_complete_runner_freezes_predictions_before_target_calls(self):
        class Toy:
            def synthesize(self, w):
                return torch.stack([w[0] + w[1], w[0] - w[1], 3 * w[0] * w[1]])

        p = json.loads(
            Path(__file__).with_name("protocol_prediction_v1.json").read_text()
        )
        p.update(
            attributes=["a", "b"],
            development_seeds=[1],
            evaluation_seeds=[2],
            expected_training_rows=12,
            expected_evaluation_rows=12,
            expected_training_calls=33,
            expected_evaluation_forward_calls=33,
            expected_evaluation_mixed_jvps=1,
        )
        validate_protocol(p)
        directions = dict(zip(p["attributes"], torch.eye(2, dtype=torch.float64)))
        with tempfile.TemporaryDirectory() as tmp, patch(
            "run_prediction.sample",
            return_value=torch.tensor([0.2, 0.3], dtype=torch.float64),
        ):
            train, test = Path(tmp) / "train", Path(tmp) / "test"
            train.mkdir()
            test.mkdir()
            run(Toy(), directions, p, train)
            model = json.loads((train / "model.json").read_text())
            result = run(Toy(), directions, p, test, model)
            self.assertEqual(result["forward_calls"], 33)
            events = [
                json.loads(s) for s in (test / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [e["event"] for e in events],
                [
                    "predictions_locked",
                    "target_joint_observation_begins",
                    "anchor_complete",
                ],
            )
            self.assertEqual(events[0]["forward_calls"], 21)
            self.assertEqual(events[1]["forward_calls"], 21)
            rows = [
                json.loads(s) for s in (test / "rows.jsonl").read_text().splitlines()
            ]
            for row in rows:
                self.assertLess(row["relative_errors"]["probe_quadratic"], 1e-10)
                self.assertLess(row["relative_errors"]["local_mixed"], 1e-10)
                self.assertLess(row["relative_errors"]["probe_calibrated"], 1e-10)


if __name__ == "__main__":
    unittest.main()
