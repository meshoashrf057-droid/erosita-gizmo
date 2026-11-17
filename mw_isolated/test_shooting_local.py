"""Numerical surrogate tests only; no test here runs live GIZMO."""
import unittest
import numpy as np
import tg19_backend as b
from shooting_local import forward_state, propose_local_step


class LocalShootingTests(unittest.TestCase):
    def test_linear_coupled_map_correct_sign_and_local_offset(self):
        matrix = np.eye(12)
        matrix[0, 3] = 0.7
        matrix[6, 9] = -0.3
        matrix[3, 6] = 0.2
        constant = np.arange(12) / 3
        forward = lambda x: matrix @ x + constant
        inverse = lambda y: np.linalg.solve(matrix, y - constant)
        base, residual = np.arange(12, dtype=float), np.linspace(-2, 3, 12)
        step, info = propose_local_step(base, residual, inverse, forward, np.ones(12), 0.4)
        np.testing.assert_allclose(step, -0.4 * np.linalg.solve(matrix, residual), atol=1e-13)
        np.testing.assert_allclose(info["predicted_residual_12d"], 0.6 * residual, atol=1e-13)
        self.assertEqual(info["applied_lambda"], 1.0)

    def test_nonlinear_correction_uses_current_model_endpoint(self):
        # The frozen inverse at target=1 would use sqrt(1-r)-1. At base=2,
        # the correct local inverse acts around S(base)=4.
        def forward(x):
            y = x.copy(); y[0] = x[0] ** 2
            return y
        def inverse(y):
            x = y.copy(); x[0] = np.sqrt(y[0])
            return x
        base, residual = np.zeros(12), np.zeros(12)
        base[0], residual[0] = 2.0, 0.8
        step, info = propose_local_step(base, residual, inverse, forward, np.ones(12), 1.0)
        frozen_step = np.sqrt(1.0 - 0.8) - 1.0
        self.assertAlmostEqual(step[0], np.sqrt(4.0 - 0.8) - 2.0)
        self.assertGreater(abs(frozen_step - step[0]), 0.3)
        self.assertLess(info["predicted_residual_norm"], 1e-12)

    def test_single_scale_caps_all_satellite_blocks(self):
        residual = np.zeros(12)
        residual[0], residual[3], residual[6], residual[9] = 100., 40., 20., 80.
        step, info = propose_local_step(np.zeros(12), residual, lambda x:x, lambda x:x,
                                       np.ones(12), 1., 20., 30.)
        np.testing.assert_allclose(step, -0.2 * residual)
        self.assertEqual(info["cap_lambda"], 0.2)
        self.assertLess(info["predicted_residual_norm"], info["current_residual_norm"])

    def test_zero_residual_skips_integrations(self):
        def forbidden(_):
            raise AssertionError("zero residual must not require an integration")
        step, info = propose_local_step(np.ones(12), np.zeros(12), forbidden, forbidden,
                                       np.ones(12), 0.5)
        np.testing.assert_array_equal(step, np.zeros(12))
        self.assertEqual(info["status"], "zero_residual")

    def test_bad_inputs_and_inconsistent_inverse_rejected(self):
        for alpha in (0., -0.1, np.nan, np.inf, 1.1):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                propose_local_step(np.zeros(12), np.ones(12), lambda x:x, lambda x:x,
                                   np.ones(12), alpha)
        with self.assertRaises(ValueError):
            propose_local_step(np.zeros(12), np.full(12, np.nan), lambda x:x, lambda x:x,
                               np.ones(12), 0.5)
        with self.assertRaises(ValueError):
            propose_local_step(np.zeros(12), np.ones(12), lambda x:x, lambda x:x,
                               np.zeros(12), 0.5)
        with self.assertRaisesRegex(RuntimeError, "closure"):
            propose_local_step(np.zeros(12), np.ones(12), lambda x:x+0.02, lambda x:x,
                               np.ones(12), 0.5)

    def test_backtracking_checks_model_prediction(self):
        # Exact positive inverse of x**2. A heavily curved step crossing the
        # origin after capping would need checking; use a synthetic inverse
        # with exact base closure but a deliberately bad candidate response.
        def forward(x):
            y = x.copy(); y[0] = x[0] + 10*x[0]**2
            return y
        residual = np.zeros(12); residual[0] = 1.
        step, info = propose_local_step(np.zeros(12), residual, lambda x:x, forward,
                                       np.ones(12), 1.)
        self.assertGreater(info["backtracks"], 0)
        self.assertLess(info["predicted_residual_norm"], 1.)
        self.assertLess(abs(step[0]), 1.)

    def test_no_descent_rejected(self):
        residual = np.ones(12)
        with self.assertRaisesRegex(RuntimeError, "No bounded shooting step"):
            propose_local_step(np.zeros(12), residual, lambda x:-x, lambda x:x,
                               np.ones(12), 0.5, max_backtracks=2)

    def test_backend_forward_inverse_units_df_and_recoil(self):
        radii = np.geomspace(0.001, 1000, 1000)
        profile = b.HostProfile(radii, 8e11 * radii**2/(radii+25)**2,
                                radii, 2e7/(1+radii/15)**3)
        class TestSatellite:
            def __init__(self, mass, scale):
                self.live_mass, self.scale = mass, scale
            def menc(self, radius):
                radius = np.asarray(radius)
                return self.live_mass * radius**2/(radius+self.scale)**2
        target = np.array([10., -70., -40., -60., -150., 160.,
                           30., -35., -55., 30., -80., 90.])
        endpoints = []
        for df in (False, True):
            for recoil in (False, True):
                kwargs = dict(profile=profile, lmc=TestSatellite(1.3e11, 12),
                              smc=TestSatellite(1.3e10, 4), lookback_gyr=0.02,
                              use_dynamical_friction=df, include_host_recoil=recoil,
                              coulomb_log=3., orbit_samples=5)
                result = b.integrate_magellanic_orbit(**kwargs,
                    lmc_target_position=target[:3], lmc_target_velocity=target[3:6],
                    smc_target_position=target[6:9], smc_target_velocity=target[9:12])
                initial = result[0].copy()
                initial[[3,4,5,9,10,11]] /= b.KMS_TO_KPC_GYR
                endpoint = forward_state(initial, kwargs)
                np.testing.assert_allclose(endpoint, target, rtol=0, atol=2e-5)
                np.testing.assert_allclose(endpoint-target, result[1], rtol=0, atol=2e-7)
                endpoints.append(forward_state(target, kwargs))
        self.assertGreater(np.linalg.norm(endpoints[0]-endpoints[1]), 0.01)
        self.assertGreater(np.linalg.norm(endpoints[0]-endpoints[2]), 0.01)


if __name__ == "__main__":
    unittest.main()
