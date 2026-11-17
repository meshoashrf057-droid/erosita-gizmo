#!/usr/bin/env python3
"""对符号、来源、累计位移和预测的独立合成测试；无需真实快照。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shooting_joint_y import COORDINATE_ORDER, load_joint_y_step


class JointYTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "proposal.json"
        self.anchor = np.arange(12, dtype=float) * 7
        self.sig = np.arange(1, 13, dtype=float)
        z = np.zeros(12); z[1] = -4; z[7] = 4
        self.anchor_residual = z * self.sig
        j = np.zeros((12, 2)); j[1] = [1, .5]; j[7] = [.5, -1]
        j[0] = [.1, -.1]
        self.base = self.anchor.copy(); self.base[1] += 1
        self.residual = self.anchor_residual + self.sig * j[:, 0]
        step = np.zeros(12); step[1] = .13; step[7] = 2.72
        offsets = np.array([1.13, 2.72])
        evidence = []
        for label, coord, amount, col, digest in (
            ("trial08", 1, 1, 0, "b" * 64),
            ("trial09", 7, -1, 1, "c" * 64),
        ):
            delta = np.zeros(12); delta[coord] = amount
            evidence.append({"trial": label, "ic_sha256": digest,
                             "applied_initial_12d": (self.anchor + delta).tolist(),
                             "residual_12d": (self.anchor_residual + self.sig * j[:, col] * amount).tolist(),
                             "source_step_12d": delta.tolist()})
        self.proposal = {
            "schema": "measured-joint-y-proposal-v1", "coordinate_order": COORDINATE_ORDER,
            "base_trial": "trial08", "fit_anchor_trial": "trial06",
            "base_ic_path": "/old/mw_trial08.hdf5", "base_ic_sha256": "b" * 64,
            "source_residual_csv_path": "/old/present08.csv", "source_residual_csv_sha256": "d" * 64,
            "shooting_model_key": "fixture-model", "base_initial_12d": self.base.tolist(),
            "current_residual_12d": self.residual.tolist(), "sigma_12d": self.sig.tolist(),
            "proposed_step_12d": step.tolist(),
            "predicted_residual_12d": (self.anchor_residual + self.sig * (j @ offsets)).tolist(),
            "measured_response_normalized_12x2": j.tolist(),
            "fit_anchor_initial_12d": self.anchor.tolist(),
            "fit_anchor_residual_12d": self.anchor_residual.tolist(),
            "total_y_offsets_from_fit_anchor_kpc": offsets.tolist(),
            "probe_evidence": evidence, "fit_anchor_base_ic_sha256": "a" * 64,
        }
        self.kwargs = dict(base_initial=self.base, residual=self.residual, sigma=self.sig,
                           base_ic_path="/moved/mw_trial08.hdf5", base_ic_sha256="b" * 64,
                           source_residual_path="/moved/present08.csv",
                           source_residual_sha256="d" * 64, model_key="fixture-model")

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, proposal=None, **kwargs):
        self.path.write_text(json.dumps(self.proposal if proposal is None else proposal))
        return load_joint_y_step(self.path, **(self.kwargs | kwargs))

    def test_exact_once_and_negative_probe_sign(self):
        step, diag = self.load()
        np.testing.assert_array_equal(step, self.proposal["proposed_step_12d"])
        np.testing.assert_allclose(diag["candidate_initial_12d"], self.base + step)
        self.assertEqual(diag["importer_increment_scale"], 1)
        self.assertFalse(diag["alpha_applied_by_importer"])
        self.assertLess(diag["predicted_normalized_norm"], diag["current_normalized_norm"])
        self.assertEqual(diag["measured_response_normalized_12x2"][7][1], -1)

    def test_reject_stale_fingerprints_and_names(self):
        for key, value in (("base_ic_sha256", "e" * 64),
                           ("source_residual_sha256", "e" * 64),
                           ("base_ic_path", "wrong07.hdf5"),
                           ("source_residual_path", "wrong07.csv"),
                           ("model_key", "other-model")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(**{key: value})

    def test_reject_changed_current_state(self):
        for key, values in (("base_initial", self.base), ("residual", self.residual), ("sigma", self.sig)):
            changed = values.copy(); changed[3] += .01
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(**{key: changed})

    def test_reject_bad_sigma_and_nonfinite(self):
        for value in (0, -1, float("nan"), float("inf")):
            sig = self.sig.copy(); sig[4] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(sigma=sig)
        proposal = copy.deepcopy(self.proposal); proposal["proposed_step_12d"][1] = float("nan")
        with self.assertRaises(ValueError): self.load(proposal)

    def test_reject_dimensions_and_coordinate_order(self):
        for key, value in (("coordinate_order", COORDINATE_ORDER[::-1]),
                           ("proposed_step_12d", [0] * 11),
                           ("measured_response_normalized_12x2", [[0, 0]] * 11)):
            proposal = copy.deepcopy(self.proposal); proposal[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.load(proposal)

    def test_reject_non_y_or_oversized_steps(self):
        for idx, value in ((0, .01), (3, .01), (1, 3.01), (7, -3.01)):
            proposal = copy.deepcopy(self.proposal); proposal["proposed_step_12d"][idx] = value
            with self.subTest(idx=idx, value=value), self.assertRaises(ValueError): self.load(proposal)

    def test_reject_duplicate_or_wrong_probe(self):
        for mutate in (
            lambda p: p["probe_evidence"].pop(),
            lambda p: p["probe_evidence"][1].update(trial="trial08"),
            lambda p: p["probe_evidence"][1]["source_step_12d"].__setitem__(7, 1),
            lambda p: p["probe_evidence"][0].update(ic_sha256="f" * 64),
            lambda p: p["probe_evidence"][1].update(ic_sha256="a" * 64),
            lambda p: p["probe_evidence"][1]["applied_initial_12d"].__setitem__(1, 123),
            lambda p: p["probe_evidence"][0]["residual_12d"].__setitem__(1, 123),
        ):
            proposal = copy.deepcopy(self.proposal); mutate(proposal)
            with self.assertRaises(ValueError): self.load(proposal)

    def test_reject_wrong_sign_response_or_prediction(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["measured_response_normalized_12x2"][7][1] *= -1
        with self.assertRaises(ValueError): self.load(proposal)
        proposal = copy.deepcopy(self.proposal)
        proposal["predicted_residual_12d"][0] += .01
        with self.assertRaises(ValueError): self.load(proposal)

    def test_reject_double_application_and_total_cap(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["proposed_step_12d"][1] = 1.13
        with self.assertRaises(ValueError): self.load(proposal)
        proposal = copy.deepcopy(self.proposal)
        proposal["total_y_offsets_from_fit_anchor_kpc"][0] = 3.1
        with self.assertRaises(ValueError): self.load(proposal)

    def test_reject_nonimproving_consistent_proposal(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["proposed_step_12d"] = [0.0] * 12
        proposal["total_y_offsets_from_fit_anchor_kpc"] = [1.0, 0.0]
        proposal["predicted_residual_12d"] = self.residual.tolist()
        with self.assertRaises(ValueError): self.load(proposal)


if __name__ == "__main__":
    unittest.main()
