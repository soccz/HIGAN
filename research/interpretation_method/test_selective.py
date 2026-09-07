"""Closed-form, nonidentification, phase leakage, and end-to-end toy checks."""

import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from run_selective import collect, validate
from selective import (
    METHODS,
    calibrate,
    features,
    fit,
    fixed_selection,
    predict,
    report,
    configure,
)
from verify_selective import close


class SelectiveTests(unittest.TestCase):
    def protocol(self):
        p = json.loads(
            Path(__file__).with_name("protocol_selective_v2.json").read_text()
        )
        p.update(
            attributes=["a", "b"],
            training_seeds=[1, 2],
            calibration_seeds=[3, 4],
            evaluation_seeds=[5, 6],
            confirmation_seeds=[7, 8],
            expected_calls_per_anchor=41,
            expected_rows_per_anchor=12,
            bootstrap_repetitions=100,
        )
        return p

    def test_smooth_hidden_bump_defeats_finite_probe_certification(self):
        # F0(s,t)=s+t and F1=F0+amplitude*bump agree at every allowed observation.
        # The target joint (0.08,0.08) is outside that finite set.
        probes, targets = [0.01, 0.03], [0.06, 0.08, 0.1]
        observed = {(0.0, 0.0)}
        for scale in probes + targets:
            for sign in [-1, 1]:
                observed.update([(sign * scale, 0), (0, sign * scale)])
        for scale in probes:
            observed.update(
                (a * scale, b * scale) for a, b in itertools.product([-1, 1], repeat=2)
            )
        target = (0.08, 0.08)
        radius = min(math.dist(target, x) for x in observed) / 2

        def bump(point):
            r2 = (math.dist(point, target) / radius) ** 2
            return math.exp(1 - 1 / (1 - r2)) if r2 < 1 else 0.0

        self.assertTrue(all(bump(x) == 0 for x in observed))
        self.assertEqual(bump(target), 1)
        for amplitude in [1.0, 1000.0]:
            f = lambda point: sum(point) + amplitude * bump(point)
            c = f(target) - f((0.08, 0)) - f((0, 0.08)) + f((0, 0))
            self.assertAlmostEqual(c, amplitude)

    def test_known_uniform_mixed_lipschitz_bound(self):
        # F(s,t)=st(s+t), H_st=2(s+t): coordinate Lipschitz constant L=2.
        for p, t in [(0.01, 0.03), (0.03, 0.1)]:
            actual = abs(2 * t**3 - (t / p) ** 2 * 2 * p**3)
            bound = 2 * t * t * (t - p)
            self.assertAlmostEqual(actual, bound)

    def test_features_track_direction_separately_from_norm(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        cp = torch.tensor([0.01, 0.02], dtype=torch.float64)
        same = features(a, a, cp, 2 * cp, 0.1)
        reverse = features(a, a, cp, -2 * cp, 0.1)
        self.assertEqual(same["probe_relative_norms"], reverse["probe_relative_norms"])
        self.assertAlmostEqual(same["probe_cosine"], 1)
        self.assertAlmostEqual(reverse["probe_cosine"], -1)

    def test_calibration_cannot_use_training_rows(self):
        with self.assertRaises(ValueError):
            calibrate([{"seed": 1}], self.protocol())

    def test_overlapping_anchor_phases_rejected(self):
        p = self.protocol()
        p["calibration_seeds"] = [2, 3]
        with self.assertRaises(ValueError):
            validate(p)

    def test_fixed_selection_ignores_target_labels(self):
        rows = [
            {
                "seed": 1,
                "a": "a",
                "b": "b",
                "scale": 0.1,
                "sign_a": a,
                "sign_b": b,
                "scores": {m: float(i) for m in METHODS},
                "actual_relative": 0.0,
            }
            for i, (a, b) in enumerate(itertools.product([-1, 1], repeat=2))
        ]
        first = fixed_selection(rows, "two_direction", 0.5)
        for row in rows:
            row["actual_relative"] = 1e100
        self.assertEqual(
            first.tolist(), fixed_selection(rows, "two_direction", 0.5).tolist()
        )
        self.assertEqual(int(first.sum()), 2)

    def test_full_toy_training_calibration_evaluation(self):
        configure()
        p = self.protocol()
        validate(p)

        class Toy:
            def synthesize(self, w):
                return torch.stack([w[0] + w[1], w[0] - w[1], 3 * w[0] * w[1]])

        directions = dict(zip(p["attributes"], torch.eye(2, dtype=torch.float64)))
        with tempfile.TemporaryDirectory() as tmp, patch(
            "run_selective.sample",
            side_effect=lambda g, seed: torch.tensor(
                [0.1 * seed, 0.2], dtype=torch.float64
            ),
        ):
            root = Path(tmp)
            for name in ("train", "calibrate", "evaluate"):
                (root / name).mkdir()
            train = collect(Toy(), directions, p, root / "train", "train")
            model = fit(train, p)
            calibration = collect(
                Toy(), directions, p, root / "calibrate", "calibrate", model
            )
            thresholds = calibrate(calibration, p)
            evaluation = collect(
                Toy(), directions, p, root / "evaluate", "evaluate", model
            )
            result = report(evaluation, p, thresholds)
            self.assertEqual(result["overall"]["independent_anchors"], 2)
            self.assertEqual(result["overall"]["n"], 24)
            events = [
                json.loads(s)
                for s in (root / "evaluate/events.jsonl").read_text().splitlines()
            ]
            for seed in p["evaluation_seeds"]:
                group = [e for e in events if e["seed"] == seed]
                self.assertEqual(
                    [e["event"] for e in group],
                    [
                        "predictions_locked",
                        "target_joint_observation_begins",
                        "anchor_complete",
                    ],
                )
                self.assertEqual(
                    group[2]["forward_calls"] - group[0]["forward_calls"], 12
                )
            record = evaluation[0]
            scores = predict(record, model, p)
            poisoned = {**record, "actual_relative": 1e200}
            self.assertEqual(scores, predict(poisoned, model, p))
            for method in METHODS:
                self.assertEqual(
                    result["overall"]["methods"][method]["fixed_coverage"], 0.5
                )
            json.dumps(result, allow_nan=False)

    def test_small_reduction_differences_do_not_break_audit(self):
        self.assertTrue(close({"x": 0.1}, {"x": 0.1 + 1e-17}))
        self.assertFalse(close({"x": 0.1}, {"x": 0.101}))
        self.assertFalse(close({"x": float("nan")}, {"x": float("nan")}))


if __name__ == "__main__":
    unittest.main()
