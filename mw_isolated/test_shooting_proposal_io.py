"""Synthetic identity/linear-response tests; no GIZMO executable required."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shooting_proposal_io import ORDER, load_proposal_step


class ProposalImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.base_file = root / "base.hdf5"
        self.base_file.write_bytes(b"synthetic base identity bytes; not a real HDF5")
        self.path = root / "proposal.json"
        base = np.arange(12, dtype=float)
        residual = np.array([1., 4., -2., 1., -2., 1., -1., -3.5, 1., 2., 1., -2.])
        sigma = np.ones(12)
        increment = -.3*residual / 1.1
        prior = residual + increment
        predicted = residual + 1.1*increment
        norms = [float(np.linalg.norm(increment[i:i+3])) for i in range(0, 12, 3)]
        norm = float(np.linalg.norm(predicted))
        maximum = float(np.max(np.abs(predicted)))
        selected = dict(id="candidate_01", accepted=True, ridge_relative=.001,
            passes_all_ridge_acceptance_criteria=True, increment_12d=increment.tolist(),
            candidate_initial_12d=(base+increment).tolist(), applied_block_norms=norms,
            predicted_residual_12d=predicted.tolist(), predicted_residual_norm=norm,
            max_abs_normalized=maximum, static_predicted_residual_12d=prior.tolist(),
            static_predicted_residual_norm=float(np.linalg.norm(prior)),
            static_max_abs_normalized=float(np.max(np.abs(prior))),
            cross_ridge_predictions=[dict(ridge_relative=.001,
                predicted_residual_12d=predicted.tolist(), predicted_residual_norm=norm,
                max_abs_normalized=maximum, acceptance_criteria_met=True)],
            worst_across_ridge_residual_norm=norm,
            worst_across_ridge_max_abs_normalized=maximum)
        signature = dict(algorithm="toy-fixed", tracer_policy="original-stars")
        base_sha = hashlib.sha256(self.base_file.read_bytes()).hexdigest()
        key_inputs = dict(source_sha256=["mw", "lmc", "smc"], lookback=1.)
        diag = dict(base_initial_12d=base.tolist(), residual_12d=residual.tolist(),
            sigma_12d=sigma.tolist(), initial_scale_12d=np.ones(12).tolist(),
            alpha=.3, max_position_step_kpc=5., max_velocity_step_kms=10.,
            current_residual_norm=float(np.linalg.norm(residual)),
            current_max_abs_normalized=float(np.max(np.abs(residual))),
            currently_failing_component_indices=[1,7],
            defect_models=[dict(ridge_relative=.001,
                correction_scaled_matrix=(.1*np.eye(12)).tolist())])
        self.proposal = dict(schema_version=1, purpose="candidate_only_not_live_convergence",
            coordinate_order=ORDER, base_ic_sha256=base_sha, model_key="model-a",
            source_sha256=["mw", "lmc", "smc"], model_key_inputs=key_inputs,
            target_time_code=1.6, measurement_signature=signature,
            code_sha256=dict(backend="backend-hash", merger="OLD-merger-hash"),
            base_initial_12d=base.tolist(), live_residual_12d=residual.tolist(),
            sigma_12d=sigma.tolist(), result=dict(status="candidate_requires_live_validation",
                selected=copy.deepcopy(selected), candidates=[copy.deepcopy(selected)], diagnostics=diag))
        self.proposal["result"]["selected"]["selection_note"] = "Selected copy has this extra field."
        self.kwargs = dict(path=self.path, base_ic=self.base_file, base_initial=base,
            residual=residual, sigma=sigma, model_key="model-a", source_sha256=["mw","lmc","smc"],
            model_key_inputs=key_inputs, target_time_code=1.6,
            residual_provenance=dict(status="verified", reference_ic_sha256=base_sha,
                model_key="model-a", target_covered=True, target_time_code=1.6,
                measurement_signature=signature),
            backend_sha256="backend-hash", forward=lambda x: x.copy(), alpha=.3,
            max_position_step_kpc=5., max_velocity_step_kms=10.)

    def call(self, **changes):
        self.path.write_text(json.dumps(self.proposal), encoding="utf-8")
        kwargs = dict(self.kwargs, **changes)
        return load_proposal_step(**kwargs)

    def test_increment_applied_once_not_alpha_twice(self):
        expected = np.array(self.proposal["result"]["selected"]["increment_12d"])
        step, diag = self.call()
        np.testing.assert_array_equal(step, expected)
        self.assertFalse(np.allclose(step, .3*expected))
        self.assertEqual(diag["applied_lambda"], 1.)
        self.assertFalse(diag["alpha_applied_by_importer"])
        self.assertEqual(diag["proposal_report"]["code_sha256"]["merger"], "OLD-merger-hash")
        json.dumps(diag, allow_nan=False)

    def test_wrong_base_bytes_rejected(self):
        self.base_file.write_bytes(b"another IC")
        with self.assertRaisesRegex(ValueError, "base_ic_sha256"):
            self.call()

    def test_selection_note_present_in_both_copies(self):
        self.proposal["result"]["candidates"][0]["selection_note"] = (
            self.proposal["result"]["selected"]["selection_note"])
        step, _ = self.call()
        np.testing.assert_array_equal(step,
            self.proposal["result"]["selected"]["increment_12d"])

    def test_residual_changed_rejected(self):
        with self.assertRaisesRegex(ValueError, "live_residual"):
            self.call(residual=self.kwargs["residual"]+.01)

    def test_tampered_selected_rejected(self):
        self.proposal["result"]["selected"]["increment_12d"][0] += .1
        with self.assertRaisesRegex(ValueError, "Selected candidate"):
            self.call()

    def test_wrong_alpha_or_caps_rejected_not_rescaled(self):
        for changes in (dict(alpha=.5), dict(max_position_step_kpc=20.),
                        dict(max_velocity_step_kms=30.)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(**changes)

    def test_step_exceeding_declared_caps_rejected(self):
        self.proposal["result"]["diagnostics"]["max_position_step_kpc"] = .1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.call(max_position_step_kpc=.1)

    def test_candidate_state_mismatch_rejected(self):
        for candidate in (self.proposal["result"]["selected"], self.proposal["result"]["candidates"][0]):
            candidate["candidate_initial_12d"][0] += .5
        with self.assertRaisesRegex(ValueError, "candidate_initial"):
            self.call()

    def test_changed_static_response_rejected(self):
        with self.assertRaisesRegex(ValueError, "static prediction"):
            self.call(forward=lambda x: 2*x)

    def test_small_static_roundoff_allowed(self):
        base = self.kwargs["base_initial"]
        def forward(x):
            return x + (1e-5 if np.linalg.norm(x-base) > .1 else 0.)
        step, diag = self.call(forward=forward)
        self.assertTrue(np.all(np.isfinite(step)))
        self.assertLess(max(diag["static_replay_errors"]["static"].values()), .01)

    def test_defect_matrix_tamper_rejected(self):
        self.proposal["result"]["diagnostics"]["defect_models"][0]["correction_scaled_matrix"][1][1] += 1
        with self.assertRaisesRegex(ValueError, "ridge"):
            self.call()

    def test_nonfinite_increment_rejected(self):
        for candidate in (self.proposal["result"]["selected"], self.proposal["result"]["candidates"][0]):
            candidate["increment_12d"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.call()

    def test_wrong_source_model_backend_and_signature_rejected(self):
        variants = [dict(source_sha256=["wrong", "lmc", "smc"]), dict(model_key="wrong"),
                    dict(backend_sha256="wrong"), dict(target_time_code=2.),
                    dict(residual_provenance=dict(self.kwargs["residual_provenance"],
                         measurement_signature=dict(algorithm="wrong"))),
                    dict(residual_provenance=dict(self.kwargs["residual_provenance"], status="legacy"))]
        for change in variants:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.call(**change)

    def test_reordered_coordinates_rejected(self):
        self.proposal["coordinate_order"] = ORDER[::-1]
        with self.assertRaisesRegex(ValueError, "coordinate_order"):
            self.call()

    def test_cross_ridge_nonimprovement_rejected(self):
        # Forge internally consistent saved predictions for a worsening model;
        # runtime acceptance must reject even when recorded flags say True.
        result = self.proposal["result"]
        correction = -2*np.eye(12)
        result["diagnostics"]["defect_models"][0]["correction_scaled_matrix"] = correction.tolist()
        residual = self.kwargs["residual"]
        step = np.array(result["selected"]["increment_12d"])
        worse = residual-step
        for candidate in (result["selected"], result["candidates"][0]):
            candidate["predicted_residual_12d"] = worse.tolist()
            candidate["predicted_residual_norm"] = float(np.linalg.norm(worse))
            candidate["max_abs_normalized"] = float(np.max(np.abs(worse)))
            candidate["worst_across_ridge_residual_norm"] = float(np.linalg.norm(worse))
            candidate["worst_across_ridge_max_abs_normalized"] = float(np.max(np.abs(worse)))
            record = candidate["cross_ridge_predictions"][0]
            record.update(predicted_residual_12d=worse.tolist(),
                predicted_residual_norm=float(np.linalg.norm(worse)),
                max_abs_normalized=float(np.max(np.abs(worse))))
        with self.assertRaisesRegex(ValueError, "does not improve"):
            self.call()


if __name__ == "__main__":
    unittest.main()
