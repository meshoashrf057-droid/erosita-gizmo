"""Reviewed-build propagation into the merger; small files, no GIZMO runs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pipeline"))
import merge_relaxed_mw_lmc_smc_local_live_v1 as merger
import local_live_build_review as build


class ReviewedBuildMergerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.review_file = self.root / "review.json"
        self.review_file.write_text(json.dumps(build.REVIEW_DATA))
        self.anchor = {"ic_sha256": build.ANCHOR_IC_SHA256,
                       "gizmo_sha256": build.ANCHOR_BINARY_SHA256}
        self.review = build.load_review(self.review_file, self.anchor)
        self.exe = {"path": "/synthetic/GIZMO", "sha256": build.EXECUTION_BINARY_SHA256}
        self.run = self.root / "lmcre"
        self.run.mkdir()
        self.snapshot = self.run / "snapshot_007.hdf5"
        with h5py.File(self.snapshot, "w") as f:
            f.create_group("Header").attrs["Time"] = .6136
        self.units = {"UnitLength_in_cm": 3.08568e21, "UnitMass_in_g": 1.989e43,
                      "UnitVelocity_in_cm_per_s": 1e5}
        self.manifest = dict(system="LMC", version=merger.VERSION,
            gizmo={"path": "/historical/GIZMO", "sha256": build.ANCHOR_BINARY_SHA256},
            raw_units=self.units, target_end_code=.6136)
        self.manifest_path = self.run / "relax_manifest.json"
        self.write_manifest()
        self.ic = self.root / "base_000.hdf5"
        self.audit = {"trial_label": "trial10", "gizmo": self.manifest["gizmo"]}
        self.write_ic()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def write_ic(self):
        with h5py.File(self.ic, "w") as f:
            f.create_group("MergeProvenance").create_dataset("AuditJSON", data=json.dumps(self.audit),
                dtype=h5py.string_dtype())
            f.create_group("MagellanicOrbitReference").create_dataset("TimeFromPresent_Gyr", data=[-1., 0.])

    def args(self, **updates):
        result = dict(reviewed_build=str(self.review_file), _reviewed_build_from_next=True,
            shooting_method="live-probe", shooting_base_ic=str(self.ic),
            require_shooting_provenance=True, inherit_base_rotations=True)
        result.update(updates)
        return SimpleNamespace(**result)

    def read_end(self, review=None, executable=None):
        return merger.end_snapshot(self.run, "LMC", self.snapshot, executable or self.exe,
                                   self.units, review)

    def test_manifest_old_new_pair_requires_review(self):
        with self.assertRaises(ValueError):
            self.read_end()
        before = self.manifest_path.read_bytes()
        path, manifest = self.read_end(self.review)
        self.assertEqual(path, self.snapshot)
        self.assertEqual(manifest["gizmo"]["sha256"], build.ANCHOR_BINARY_SHA256)
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_exact_manifest_build_remains_supported_without_review(self):
        path, _ = self.read_end(executable=self.manifest["gizmo"])
        self.assertEqual(path, self.snapshot)

    def test_review_does_not_authorize_other_manifest_hash(self):
        self.manifest["gizmo"]["sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.read_end(self.review)

    def test_review_does_not_authorize_other_current_hash(self):
        with self.assertRaises(ValueError):
            self.read_end(self.review, dict(self.exe, sha256="1" * 64))

    def test_review_keeps_time_and_unit_checks(self):
        for key, value in (("target_end_code", .7), ("raw_units", {**self.units, "UnitVelocity_in_cm_per_s": 2e5})):
            original = copy.deepcopy(self.manifest)
            self.manifest[key] = value
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.read_end(self.review)
            self.manifest = original
        self.write_manifest()

    def test_review_file_change_rejected_at_manifest_check(self):
        self.review_file.write_text(self.review_file.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.read_end(self.review)

    def test_guard_no_review_is_default(self):
        self.assertIsNone(merger.checked_build_review(self.args(reviewed_build=None), self.exe))

    def test_guard_forbids_direct_merge_and_generic_methods(self):
        cases = [{"_reviewed_build_from_next": False}, {"shooting_method": "local-defect"},
                 {"require_shooting_provenance": False}, {"inherit_base_rotations": False}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                merger.checked_build_review(self.args(**case), self.exe)

    def test_guard_does_not_accept_another_trial(self):
        self.audit["trial_label"] = "trial06"
        self.write_ic()
        with self.assertRaises(ValueError):
            merger.checked_build_review(self.args(), self.exe)

    def test_guard_checks_actual_ic_hash_not_just_trial_name(self):
        # This is a small synthetic file labelled trial10, not the user's real IC.
        with self.assertRaises(ValueError):
            merger.checked_build_review(self.args(), self.exe)

    def test_guard_supplies_actual_ic_and_original_binary_hash(self):
        with patch.object(merger, "load_review", return_value=self.review) as loader:
            result = merger.checked_build_review(self.args(), self.exe)
        self.assertEqual(result, self.review)
        loader.assert_called_once_with(str(self.review_file),
            {"ic_sha256": merger.sha256(self.ic), "gizmo_sha256": build.ANCHOR_BINARY_SHA256})

    def test_next_forwards_review_and_internal_origin(self):
        self.audit.update(source_files=["mw.hdf5", "lmc.hdf5", "smc.hdf5"],
            source_relaxation_manifests=[None, {"output_dir": "/lmc/output"}, {"output_dir": "/smc/output"}],
            model_key_inputs={"coulomb_log": 3., "targets": np.zeros((4,3)).tolist(), "df": True, "recoil": True},
            lookback_gyr=1., stage2_start_code=.6136)
        self.write_ic()
        args = self.args(base_param="base.param", gizmo_executable="GIZMO", output="trial11_000.hdf5",
            shooting_residual_csv="residual.csv", shooting_alpha=.3,
            shooting_max_position_step_kpc=20., shooting_max_velocity_step_kms=30.,
            trial_label="trial11", mpi_tasks=6, shooting_proposal_json=None,
            probe_coordinate="LMC_x", probe_step=.5, host_reference_ic=None)
        with patch.object(merger, "main") as main:
            merger.next_trial(args)
        cmd = main.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--reviewed-build") + 1], str(self.review_file))
        self.assertEqual(main.call_args.kwargs, {"reviewed_from_next": True})


if __name__ == "__main__":
    unittest.main()
