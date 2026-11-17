#!/usr/bin/env python3
"""Numerical contract tests, independent of real GIZMO outputs."""
import json
import unittest

import numpy as np

from local_live_math import propose_step


class LocalLiveMathTests(unittest.TestCase):
    @staticmethod
    def coupled_problem():
        z = np.full(12, 0.2)
        z[:2] = [3.6, 3.4]
        jac = np.zeros((12, 2))
        jac[:2] = [[1.0, 1.0], [1.0, -1.0]]
        return z, jac, np.array([2.0, 2.0])

    def test_coupled_minimax_and_physical_bounds(self):
        z, jac, bounds = self.coupled_problem()
        out = propose_step(z, jac, bounds)
        # The sum of the first two residuals is >= 7 - 4 = 3.
        # Setting delta=(-2,-0.1) attains max residual 1.5 exactly.
        self.assertAlmostEqual(out["minimax_optimum"], 1.5, places=8)
        np.testing.assert_allclose(out["delta"], [-2.0, -0.1], atol=1e-6)
        self.assertTrue(out["ready_for_live_validation"])
        np.testing.assert_allclose(out["projected_normalized_residual"],
                                   z + jac @ out["delta"], atol=1e-12)
        self.assertTrue(np.all(np.abs(out["delta"]) <= bounds + 1e-9))
        json.dumps(out, allow_nan=False)

    def test_uncontrolled_bad_component_prevents_ready(self):
        z, jac, bounds = self.coupled_problem()
        z[11] = 3.4
        out = propose_step(z, jac, bounds)
        self.assertAlmostEqual(out["minimax_optimum"], 3.4)
        self.assertFalse(out["ready_for_live_validation"])
        self.assertIn("predicted_max_does_not_reach_target", out["reasons"])
        self.assertEqual(out["projected_normalized_residual"][11], 3.4)

    def test_physical_unit_conversion_invariance(self):
        z, jac, bounds = self.coupled_problem()
        original = propose_step(z, jac, bounds)
        conversion = np.array([1000.0, 0.001])
        converted = propose_step(z, jac / conversion, bounds * conversion)
        np.testing.assert_allclose(np.array(converted["delta"]) / conversion,
                                   original["delta"], atol=1e-9)
        np.testing.assert_allclose(converted["projected_normalized_residual"],
                                   original["projected_normalized_residual"], atol=1e-9)
        self.assertAlmostEqual(converted["condition_number"], original["condition_number"])

    def test_nonfinite_or_invalid_input_rejected(self):
        z, jac, bounds = self.coupled_problem()
        for value in [np.nan, np.inf, -np.inf]:
            bad = z.copy()
            bad[0] = value
            with self.assertRaises(ValueError):
                propose_step(bad, jac, bounds)
            bad_jac = jac.copy()
            bad_jac[5, 0] = value
            with self.assertRaises(ValueError):
                propose_step(z, bad_jac, bounds)
        for bad_bound in ([0, 1], [-1, 1], [1, np.nan]):
            with self.assertRaises(ValueError):
                propose_step(z, jac, bad_bound)
        with self.assertRaises(ValueError):
            propose_step(z[:6], jac[:6], bounds)
        with self.assertRaises(ValueError):
            propose_step(z, jac, bounds, max_scaled_step=0)

    def test_near_collinear_columns_diagnostic_only(self):
        z = np.full(12, 0.2)
        z[:2] = [3.1, 3.1]
        jac = np.zeros((12, 2))
        jac[:2] = [[1, 1], [1, 1 + 1e-5]]
        out = propose_step(z, jac, [1, 1])
        self.assertEqual(out["rank"], 2)
        self.assertGreater(out["condition_number"], 1000)
        self.assertFalse(out["ready_for_live_validation"])
        self.assertIn("response_columns_ill_conditioned", out["reasons"])

    def test_rank_deficient_columns_diagnostic_only(self):
        z = np.full(12, 0.2)
        z[:2] = [3.1, 3.1]
        jac = np.zeros((12, 2))
        jac[:2] = [[1, 1], [1, 1]]
        out = propose_step(z, jac, [1, 1])
        self.assertEqual(out["rank"], 1)
        self.assertIsNone(out["condition_number"])
        self.assertFalse(out["ready_for_live_validation"])
        self.assertIn("response_columns_rank_deficient", out["reasons"])
        json.dumps(out, allow_nan=False)

    def test_step_multiplier_and_no_false_success(self):
        z, jac, bounds = self.coupled_problem()
        out = propose_step(z, jac, bounds, max_scaled_step=0.1)
        self.assertTrue(np.all(np.abs(out["delta"]) <= 0.2 + 1e-9))
        self.assertFalse(out["ready_for_live_validation"])
        self.assertGreater(out["projected_max_abs"], 3)

    def test_minimum_step_does_not_move_unnecessary_direction(self):
        z = np.full(12, 0.2)
        z[0] = 3.6
        jac = np.zeros((12, 2))
        jac[0, 0] = 1
        jac[1, 1] = 1
        out = propose_step(z, jac, [1, 1], regularization=0)
        self.assertAlmostEqual(out["delta"][0], -1.0, places=6)
        self.assertAlmostEqual(out["delta"][1], 0, places=6)


if __name__ == "__main__":
    unittest.main()
