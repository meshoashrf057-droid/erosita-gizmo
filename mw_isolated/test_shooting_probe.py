"""Run: python -m unittest -v test_shooting_probe.py"""
from __future__ import annotations

import unittest
import numpy as np

from shooting_probe import COORDINATES, make_probe_step, probe_static_diagnostic


class ProbeStepTests(unittest.TestCase):
    def test_every_coordinate_has_exactly_one_unchanged_signed_step(self):
        for i, name in enumerate(COORDINATES):
            for sign in (-1, 1):
                with self.subTest(coordinate=name, sign=sign):
                    value = sign * (1.0 if i % 6 < 3 else 2.5)
                    step, audit = make_probe_step(name, value)
                    expected = np.zeros(12)
                    expected[i] = value
                    np.testing.assert_array_equal(step, expected)
                    self.assertEqual(audit["probe_coordinate_index"], i)
                    self.assertEqual(audit["probe_step_unit"], "kpc" if i % 6 < 3 else "km/s")
                    self.assertEqual(audit["purpose"], "live-response-measurement")
                    self.assertFalse(audit["alpha_applied"])
                    self.assertEqual(audit["importer_increment_scale"], 1.0)

    def test_hard_limits_accept_boundary_reject_excess(self):
        for name, limit in (("LMC_x", 2.0), ("SMC_vy", 5.0)):
            for sign in (-1, 1):
                with self.subTest(name=name, sign=sign):
                    make_probe_step(name, sign*limit)
                    with self.assertRaises(ValueError):
                        make_probe_step(name, sign*np.nextafter(limit, np.inf))

    def test_bad_coordinate(self):
        for name in ("MW_y", "LMC_Y", "SMC_vr", "", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_probe_step(name, 1.0)

    def test_bad_step(self):
        for value in (0.0, np.nan, np.inf, -np.inf, True, [1.0], "invalid", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_probe_step("LMC_y", value)

    def test_no_alpha_argument_or_implicit_rescaling(self):
        with self.assertRaises(TypeError):
            make_probe_step("LMC_y", 1.0, alpha=0.3)
        step, audit = make_probe_step("LMC_y", 1.0)
        self.assertEqual(step[1], 1.0)
        self.assertEqual(audit["applied_step_norms"]["LMC"]["position_kpc"], 1.0)
        self.assertEqual(audit["applied_step_norms"]["SMC"]["velocity_kms"], 0.0)


class StaticDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.base = np.arange(12, dtype=float)
        self.step, _ = make_probe_step("LMC_y", 1.0)
        self.residual = np.ones(12)
        self.sigma = np.ones(12)

    def test_two_diagnostic_calls_preserve_inputs_and_allow_worse_prediction(self):
        calls = []
        def forward(value):
            calls.append(value.copy())
            return 2*value
        before = [v.copy() for v in (self.base, self.step, self.residual, self.sigma)]
        result = probe_static_diagnostic(self.base, self.step, self.residual, self.sigma, forward)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], self.base)
        np.testing.assert_array_equal(calls[1], self.base+self.step)
        np.testing.assert_array_equal(result["predicted_residual_12d"], self.residual+2*self.step)
        self.assertGreater(result["predicted_residual_norm"], result["current_residual_norm"])
        self.assertEqual(result["static_diagnostic_status"], "ok")
        for original, current in zip(before, (self.base, self.step, self.residual, self.sigma)):
            np.testing.assert_array_equal(original, current)

    def test_static_failure_prevents_incomplete_metadata(self):
        def broken(value):
            raise RuntimeError("synthetic integration failure")
        with self.assertRaisesRegex(ValueError, "synthetic integration failure"):
            probe_static_diagnostic(self.base, self.step, self.residual, self.sigma, broken)
        self.assertEqual(self.step[1], 1.0)

    def test_nonfinite_static_endpoint_never_becomes_prediction(self):
        calls = []
        def forward(value):
            calls.append(value)
            return value if len(calls) == 1 else np.full(12, np.nan)
        with self.assertRaisesRegex(ValueError, "静态诊断积分失败"):
            probe_static_diagnostic(self.base, self.step, self.residual, self.sigma, forward)

    def test_invalid_input_or_sigma_rejected_before_forward(self):
        def unused(value):
            self.fail("invalid inputs must not reach forward")
        bad_vectors = (np.ones(11), np.full(12, np.nan), np.ones((12, 1)))
        for slot in range(4):
            for bad in bad_vectors:
                values = [self.base, self.step, self.residual, self.sigma]
                values[slot] = bad
                with self.subTest(slot=slot, shape=bad.shape), self.assertRaises(ValueError):
                    probe_static_diagnostic(*values, unused)
        with self.assertRaises(ValueError):
            probe_static_diagnostic(self.base, self.step, self.residual, np.zeros(12), unused)


if __name__ == "__main__":
    unittest.main()
