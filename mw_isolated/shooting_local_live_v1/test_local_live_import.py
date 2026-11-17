#!/usr/bin/env python3
"""Synthetic importer tests. These do not run GIZMO or claim live convergence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import local_live_import as imp
from local_live_evidence import pair_diagnostics, normalized_params, paths, validate_common


class TestLocalLiveImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        self.base_initial = np.arange(12, dtype=float)
        self.sigma = np.ones(12)
        self.residual = np.zeros(12)
        self.residual[[0, 1, 7]] = [3.1, -3.2, 3.0]
        self.response = np.zeros((12, 3))
        self.response[[0, 1, 7], [0, 1, 2]] = 2.0
        self.evidence = {}
        param = self.root / "base.param"
        param.write_text("TimeBegin 0.6\nMaxSizeTimestep 0.001\nSofteningGas 0.005\n"
                         "InitMetallicity 0.3\nCritPhysDensity 1000\n"
                         "UnitLength_in_cm 3.08568e21\nUnitMass_in_g 1.989e43\n"
                         "UnitVelocity_in_cm_per_s 100000\n")
        runtime = paths(self.root, "trial10")["runtime_param"]
        runtime.parent.mkdir(parents=True)
        runtime.write_text(param.read_text() + "TimeMax 1.6\nTimeOfFirstSnapshot 0.6\n"
                           "TimeBetSnapshot 0.1\nICFormat 3\nSnapFormat 3\nOutputListOn 0\n"
                           "NumFilesPerSnapshot 1\nNumFilesWrittenInParallel 1\nInitGasTemp 0\n"
                           "InitCondFile anchor\nOutputDir output\n"
                           "SnapshotFileBase snapshot\nRestartFile restart\n")
        gizmo = self.root / "GIZMO"; gizmo.write_bytes(b"synthetic binary"); gizmo.chmod(0o755)
        common = dict(model_key="test-model", source_sha256=["a"*64, "b"*64, "c"*64],
            measurement_signature={"tracer_policy": "original-stars"},
            units={"length_kpc": 1.0, "mass_msun":1e10, "velocity_kms":1., "time_gyr":1.},
            base_param_path=str(param), base_param_sha256=imp._file_hash(param),
            gizmo_path=str(gizmo), gizmo_sha256=imp._file_hash(gizmo),
            runtime_param_normalized=normalized_params(runtime),runtime_param_path=str(runtime),
            runtime_param_sha256=imp._file_hash(runtime),
            sigma_12d=self.sigma.tolist(), observed_12d=np.zeros(12).tolist(),
            rotation_matrices=[np.eye(3).tolist()]*3, target_time_code=1.6, stage2_start_code=0.6)
        for index, trial in enumerate(("trial10", "trial11", "trial12", "trial13")):
            ic = self.root / f"mw_lmc_smc_relaxed_{trial}_000.hdf5"
            ic.write_bytes((trial + " synthetic IC").encode())
            csv = self.root / f"{trial}.csv"
            csv.write_text(trial + " synthetic CSV")
            record = dict(common, trial=trial, work_dir=str(self.root),
                validation_subdir="magellanic_validation_fixed", ic_path=str(ic),
                ic_sha256=imp._file_hash(ic), residual_csv_path=str(csv),
                residual_csv_sha256=imp._file_hash(csv))
            if index == 0:
                record.update(applied_initial_12d=self.base_initial.tolist(),
                    residual_12d=self.residual.tolist(),
                    audit={"cumulative_offset_12d": np.zeros(12).tolist()})
            else:
                anchor = self.evidence["trial10"]
                coord = imp.COORDINATE_INDICES[index - 1]
                step = np.zeros(12); step[coord] = 0.5
                residual = self.residual + 0.5 * self.response[:, index - 1]
                record.update(applied_initial_12d=(self.base_initial + step).tolist(),
                    residual_12d=residual.tolist(),
                    audit=dict(shooting_base_ic=anchor["ic_path"], previous_offset_12d=np.zeros(12).tolist(),
                        this_iteration_increment_12d=step.tolist(), manual_increment_12d=np.zeros(12).tolist(),
                        shooting_alpha=None,
                        shooting_residual_provenance=dict(reference_ic_sha256=anchor["ic_sha256"],
                            residual_csv_sha256=anchor["residual_csv_sha256"],
                            measurement_signature=anchor["measurement_signature"]),
                        shooting_diagnostics=dict(method="live-probe", alpha_applied=False,
                            alpha_applied_by_importer=False, base_initial_12d=self.base_initial.tolist(),
                            residual_12d=self.residual.tolist(), sigma_12d=self.sigma.tolist())))
            self.evidence[trial] = record
        anchor = self.evidence["trial10"]
        step = np.zeros(12); step[[0, 1, 7]] = [-0.2, 0.25, -0.2]
        self.proposal = dict(schema=imp.SCHEMA, coordinate_order=imp.COORDINATE_ORDER,
            base_trial="trial10", shooting_model_key="test-model", alpha_applied=False,
            target_max=2.8, step_caps_kpc=[0.5]*3,
            solver_report=dict(ready_for_live_validation=True, delta=step[[0,1,7]].tolist(),
                projected_normalized_residual=(self.residual+self.response @ step[[0,1,7]]).tolist()),
            history_dependency=False,
            param_comparison_policy="effective-production-v1",
            execution_base_param_path=str(param), execution_base_param_sha256=imp._file_hash(param),
            execution_gizmo=dict(path=str(gizmo),sha256=imp._file_hash(gizmo)),
            base_ic_path=anchor["ic_path"], base_ic_sha256=anchor["ic_sha256"],
            source_residual_csv_path=anchor["residual_csv_path"],
            source_residual_csv_sha256=anchor["residual_csv_sha256"],
            base_initial_12d=self.base_initial.tolist(), current_residual_12d=self.residual.tolist(),
            sigma_12d=self.sigma.tolist(), proposed_step_12d=step.tolist(),
            candidate_initial_12d=(self.base_initial+step).tolist(),
            predicted_residual_12d=(self.residual+self.response @ step[[0,1,7]]).tolist(),
            measured_response_normalized_12x3=self.response.tolist(),
            anchor_evidence=copy.deepcopy(anchor),
            probe_evidence=[copy.deepcopy(self.evidence[t]) for t in ("trial11", "trial12", "trial13")])
        self.path = self.root / "proposal.json"
        self.anchor_patch = patch.object(imp, "ANCHOR_IC_SHA256", anchor["ic_sha256"])
        self.anchor_patch.start(); self.addCleanup(self.anchor_patch.stop)
        self.reader_patch = patch.object(imp, "read_evidence", side_effect=self.fresh)
        self.reader_mock = self.reader_patch.start(); self.addCleanup(self.reader_patch.stop)

    def fresh(self, work_dir, trial, validation_subdir):
        self.assertEqual(work_dir, str(self.root))
        self.assertEqual(validation_subdir, "magellanic_validation_fixed")
        record = copy.deepcopy(self.evidence[trial])
        for prefix in ("ic", "residual_csv"):
            record[prefix+"_sha256"] = imp._file_hash(record[prefix+"_path"])
        return record

    def run_import(self):
        self.path.write_text(json.dumps(self.proposal))
        anchor = self.evidence["trial10"]
        return imp.load_local_live_step(self.path, self.base_initial, self.residual, self.sigma,
            anchor["ic_path"], anchor["ic_sha256"], anchor["residual_csv_path"],
            anchor["residual_csv_sha256"], "test-model")

    def test_single_unscaled_increment_and_four_fresh_reads(self):
        before = {p: imp._file_hash(p) for p in self.root.iterdir() if p.is_file()}
        step, diag = self.run_import()
        np.testing.assert_array_equal(step, self.proposal["proposed_step_12d"])
        self.assertEqual(self.reader_mock.call_count, 4)
        self.assertFalse(diag["alpha_applied"])
        self.assertEqual(diag["importer_increment_scale"], 1.0)
        self.assertAlmostEqual(diag["predicted_max_abs_normalized_residual"], 2.7)
        self.assertEqual(diag["status"], "joint_candidate_requires_live_validation")
        for path, digest in before.items():
            self.assertEqual(imp._file_hash(path), digest)

    def test_old_param_policy_rejected(self):
        self.proposal.pop("param_comparison_policy")
        with self.assertRaisesRegex(ValueError, "参数核对策略"):
            self.run_import()

    def test_mutated_probe_file_rejected_after_fresh_read(self):
        Path(self.evidence["trial12"]["ic_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "证据文件"):
            self.run_import()

    def test_parent_fingerprint_mismatch_rejected(self):
        self.evidence["trial12"]["audit"]["shooting_residual_provenance"]["reference_ic_sha256"] = "a"*64
        self.proposal["probe_evidence"][1] = copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError, "parent IC/CSV"):
            self.run_import()

    def test_no_history_file_needed(self):
        self.assertFalse((self.root / "history.json").exists())
        self.assertNotIn("history_json_path", self.proposal)
        step, diag = self.run_import()
        np.testing.assert_array_equal(step, self.proposal["proposed_step_12d"])
        self.assertFalse(diag["history_dependency"])

    def test_old_schema_rejected(self):
        self.proposal["schema"] = "local-live-proposal-v1"
        with self.assertRaisesRegex(ValueError, "schema"):
            self.run_import()

    def test_history_dependency_rejected(self):
        for value in (True, None):
            self.proposal["history_dependency"] = value
            with self.assertRaisesRegex(ValueError, "history_dependency"):
                self.run_import()
        self.proposal["history_dependency"] = False
        self.proposal["history_json_path"] = str(self.root / "not-required-history.json")
        with self.assertRaisesRegex(ValueError, "不得依赖旧历史"):
            self.run_import()

    def test_prediction_tamper_rejected(self):
        self.proposal["predicted_residual_12d"][0] -= 0.1
        with self.assertRaisesRegex(ValueError, "预测残差"):
            self.run_import()

    def test_response_tamper_rejected(self):
        self.proposal["measured_response_normalized_12x3"][0][0] = 3
        with self.assertRaisesRegex(ValueError, "实测响应矩阵"):
            self.run_import()

    def test_alpha_and_mutated_target_rejected(self):
        self.proposal["alpha_applied"] = True
        with self.assertRaisesRegex(ValueError, "alpha"):
            self.run_import()
        self.proposal["alpha_applied"] = False
        self.proposal["target_max"] = 3.0
        with self.assertRaisesRegex(ValueError, "2.8"):
            self.run_import()

    def test_extra_coordinate_rejected(self):
        self.proposal["proposed_step_12d"][3] = 0.01
        with self.assertRaisesRegex(ValueError, "只允许"):
            self.run_import()

    def test_limit_and_nonfinite_rejected(self):
        self.proposal["proposed_step_12d"][0] = -0.501
        self.proposal["solver_report"]["delta"][0] = -0.501
        with self.assertRaisesRegex(ValueError, "0.5 kpc"):
            self.run_import()
        self.proposal["proposed_step_12d"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "有限数值"):
            self.run_import()

    def test_predicted_pass_margin_required(self):
        step = np.zeros(12); step[[0,1,7]] = [-0.1,0.15,-0.1]
        self.proposal.update(proposed_step_12d=step.tolist(),
            candidate_initial_12d=(self.base_initial+step).tolist(),
            predicted_residual_12d=(self.residual+self.response@step[[0,1,7]]).tolist())
        self.proposal["solver_report"].update(delta=step[[0,1,7]].tolist(),
            projected_normalized_residual=self.proposal["predicted_residual_12d"])
        with self.assertRaisesRegex(ValueError, "2.8 sigma"):
            self.run_import()

    def test_changed_units_rejected(self):
        self.evidence["trial13"]["units"] = {"length_kpc":2.0}
        self.proposal["probe_evidence"][2] = copy.deepcopy(self.evidence["trial13"])
        with self.assertRaisesRegex(ValueError, "units"):
            self.run_import()

    def test_wrong_anchor_rejected(self):
        with patch.object(imp, "ANCHOR_IC_SHA256", "f"*64):
            with self.assertRaisesRegex(ValueError, "trial10 基准"):
                self.run_import()

    def test_changed_binary_rejected(self):
        Path(self.evidence["trial10"]["gizmo_path"]).write_bytes(b"changed binary")
        with self.assertRaises(ValueError):
            self.run_import()

    def use_backup_binary(self):
        original=Path(self.evidence["trial10"]["gizmo_path"])
        backup=self.root/"GIZMO_t10_backup"
        backup.write_bytes(original.read_bytes());backup.chmod(0o755)
        self.proposal["execution_gizmo"]={"path":str(backup),"sha256":imp._file_hash(backup)}
        for j,trial in enumerate(("trial11","trial12","trial13")):
            self.evidence[trial]["gizmo_path"]=str(backup)
            self.proposal["probe_evidence"][j]=copy.deepcopy(self.evidence[trial])
        return original,backup

    def test_exact_backup_allows_overwritten_original_binary(self):
        original,backup=self.use_backup_binary()
        original.write_bytes(b"new AGN build, not this calibration")
        with patch("subprocess.run",side_effect=AssertionError("must not execute GIZMO")):
            step,diag=self.run_import()
        np.testing.assert_array_equal(step,self.proposal["proposed_step_12d"])
        self.assertEqual(diag["execution_gizmo"]["path"],str(backup))
        self.assertEqual(self.proposal["anchor_evidence"]["gizmo_path"],str(original))

    def test_exact_backup_allows_missing_original_binary(self):
        original,_=self.use_backup_binary();original.unlink()
        self.run_import()

    def test_changed_backup_rejected_even_with_updated_proposal_hash(self):
        _,backup=self.use_backup_binary()
        backup.write_bytes(b"new build")
        self.proposal["execution_gizmo"]["sha256"]=imp._file_hash(backup)
        with self.assertRaises(ValueError):self.run_import()

    def test_probe_execution_binary_path_must_match(self):
        original,_=self.use_backup_binary()
        self.evidence["trial12"]["gizmo_path"]=str(original)
        self.proposal["probe_evidence"][1]=copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError,"GIZMO 路径"):
            self.run_import()

    def test_nonexecutable_backup_rejected(self):
        _,backup=self.use_backup_binary();backup.chmod(0o644)
        with self.assertRaises(ValueError):self.run_import()

    def test_execution_binary_metadata_required(self):
        self.proposal.pop("execution_gizmo")
        with self.assertRaisesRegex(ValueError,"execution_gizmo"):
            self.run_import()

    def test_unexpected_probe_build_review_rejected(self):
        self.evidence["trial12"]["audit"]["build_review"] = {"unrequested": True}
        self.proposal["probe_evidence"][1] = copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError, "构建审查记录"):
            self.run_import()

    def use_reviewed_rebuild(self):
        """Synthetic bytes exercise real review loading/selection, never GIZMO.

        Only the narrowly reviewed identities are replaced by fixture hashes;
        the ordinary file, record, probe and importer checks remain active.
        """
        import local_live_build_review as br
        anchor = self.evidence["trial10"]
        old_sha = anchor["gizmo_sha256"]
        binary = Path(anchor["gizmo_path"])
        binary.write_bytes(b"synthetic reviewed rebuild")
        new_sha = imp._file_hash(binary)
        data = copy.deepcopy(br.REVIEW_DATA)
        data.update(anchor_ic_sha256=anchor["ic_sha256"],
                    anchor_binary_sha256=old_sha, execution_binary_sha256=new_sha)
        for name, value in (("ANCHOR_IC_SHA256", anchor["ic_sha256"]),
                            ("ANCHOR_BINARY_SHA256", old_sha),
                            ("EXECUTION_BINARY_SHA256", new_sha),
                            ("REVIEW_DATA", data)):
            active = patch.object(br, name, value)
            active.start(); self.addCleanup(active.stop)
        review_path = self.root / "reviewed_build.json"
        review_path.write_text(json.dumps(data))
        record = br.load_review(review_path, anchor)
        self.proposal["build_review"] = copy.deepcopy(record)
        self.proposal["execution_gizmo"] = {"path": str(binary), "sha256": new_sha}
        for j, trial in enumerate(("trial11", "trial12", "trial13")):
            self.evidence[trial]["gizmo_sha256"] = new_sha
            self.evidence[trial]["audit"]["build_review"] = copy.deepcopy(record)
            self.proposal["probe_evidence"][j] = copy.deepcopy(self.evidence[trial])
        return binary, review_path, record, old_sha

    def test_reviewed_rebuild_all_three_probes_and_candidate_bound(self):
        binary, _, record, old_sha = self.use_reviewed_rebuild()
        with patch("subprocess.run", side_effect=AssertionError("must not execute GIZMO")):
            step, diag = self.run_import()
        np.testing.assert_array_equal(step, self.proposal["proposed_step_12d"])
        self.assertEqual(diag["execution_gizmo"]["sha256"], imp._file_hash(binary))
        self.assertEqual(diag["build_review"], record)
        self.assertEqual(self.proposal["anchor_evidence"]["gizmo_sha256"], old_sha)
        self.assertIn("not been measured", diag["model_uncertainty"])
        self.assertEqual(diag["baseline_reuse_assumption"],
                         "reviewed-rebuild-endpoint-equivalence-unmeasured")
        candidate = copy.deepcopy(self.evidence["trial11"])
        candidate["trial"] = "trial14"
        validate_common(self.evidence["trial10"], candidate, build_review=record)
        with self.assertRaises(ValueError):
            validate_common(self.evidence["trial10"], candidate)

    def test_reviewed_rebuild_without_review_is_rejected(self):
        self.use_reviewed_rebuild()
        self.proposal.pop("build_review")
        with self.assertRaises(ValueError):
            self.run_import()

    def test_reviewed_rebuild_requires_all_probes_use_execution_hash(self):
        _, _, _, old_sha = self.use_reviewed_rebuild()
        self.evidence["trial12"]["gizmo_sha256"] = old_sha
        self.proposal["probe_evidence"][1] = copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError, "GIZMO 指纹"):
            self.run_import()

    def test_reviewed_rebuild_requires_probe_review_provenance(self):
        self.use_reviewed_rebuild()
        self.evidence["trial12"]["audit"].pop("build_review")
        self.proposal["probe_evidence"][1] = copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError, "构建审查记录"):
            self.run_import()

    def test_reviewed_rebuild_does_not_allow_a_third_build(self):
        binary, _, _, _ = self.use_reviewed_rebuild()
        binary.write_bytes(b"third unreviewed build")
        self.proposal["execution_gizmo"]["sha256"] = imp._file_hash(binary)
        with self.assertRaises(ValueError):
            self.run_import()

    def test_reviewed_rebuild_record_file_cannot_be_changed(self):
        _, review_path, _, _ = self.use_reviewed_rebuild()
        review_path.write_text(review_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "冻结后已改变"):
            self.run_import()

    def test_reviewed_rebuild_record_payload_cannot_be_changed(self):
        self.use_reviewed_rebuild()
        self.proposal["build_review"]["data"]["endpoint_numerical_equivalence_measured"] = True
        with self.assertRaisesRegex(ValueError, "冻结后已改变"):
            self.run_import()

    def use_backup_param(self):
        original = Path(self.evidence["trial10"]["base_param_path"])
        backup = self.root / "base_t10_backup.param"
        backup.write_bytes(original.read_bytes())
        self.proposal.update(execution_base_param_path=str(backup),
                             execution_base_param_sha256=imp._file_hash(backup))
        for j, trial in enumerate(("trial11", "trial12", "trial13")):
            self.evidence[trial]["base_param_path"] = str(backup)
            self.proposal["probe_evidence"][j] = copy.deepcopy(self.evidence[trial])
        return original, backup

    def freeze_execution_param_for_probes(self, param):
        digest=imp._file_hash(param)
        self.proposal.update(execution_base_param_path=str(param),execution_base_param_sha256=digest)
        for j,trial in enumerate(("trial11","trial12","trial13")):
            self.evidence[trial]["base_param_path"]=str(param)
            self.evidence[trial]["base_param_sha256"]=digest
            self.proposal["probe_evidence"][j]=copy.deepcopy(self.evidence[trial])

    def test_comment_only_new_param_hash_is_accepted(self):
        original,backup=self.use_backup_param()
        backup.write_text("% regenerated comment, physical parameters unchanged\n"+backup.read_text())
        self.freeze_execution_param_for_probes(backup)
        self.assertNotEqual(imp._file_hash(backup),self.evidence["trial10"]["base_param_sha256"])
        step,diag=self.run_import()
        np.testing.assert_array_equal(step,self.proposal["proposed_step_12d"])
        self.assertTrue(diag["execution_param_compatibility"]["compatible"])

    def test_physical_change_rejected_even_with_updated_execution_hash(self):
        _,backup=self.use_backup_param()
        backup.write_text(backup.read_text().replace("SofteningGas 0.005","SofteningGas 0.05"))
        self.freeze_execution_param_for_probes(backup)
        with self.assertRaisesRegex(ValueError,"SofteningGas"):
            self.run_import()

    def test_probe_execution_param_sha_must_match_actual(self):
        self.evidence["trial12"]["base_param_sha256"]="d"*64
        self.proposal["probe_evidence"][1]=copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError,"实际 param 指纹"):
            self.run_import()

    def test_same_directory_backup_allows_changed_original(self):
        original, backup = self.use_backup_param()
        original.write_text("TimeBegin 0.0\nHighResolutionOtherSetting 1\n")
        step, diag = self.run_import()
        np.testing.assert_array_equal(step, self.proposal["proposed_step_12d"])
        self.assertEqual(diag["execution_base_param_path"], str(backup))

    def test_same_directory_backup_allows_missing_original(self):
        original, _ = self.use_backup_param()
        original.unlink()
        self.run_import()

    def test_changed_execution_backup_rejected(self):
        _, backup = self.use_backup_param()
        backup.write_text("changed actual execution param")
        with self.assertRaisesRegex(ValueError, "实际执行 param 已改变"):
            self.run_import()

    def test_probe_different_execution_param_rejected(self):
        original, _ = self.use_backup_param()
        self.evidence["trial12"]["base_param_path"] = str(original)
        self.proposal["probe_evidence"][1] = copy.deepcopy(self.evidence["trial12"])
        with self.assertRaisesRegex(ValueError, "param 路径与本批实际执行 param 不同"):
            self.run_import()

    def test_execution_param_moved_to_other_directory_rejected(self):
        original, backup = self.use_backup_param()
        other = self.root / "other"
        other.mkdir()
        moved = other / "base.param"
        moved.write_bytes(backup.read_bytes())
        self.proposal["execution_base_param_path"] = str(moved)
        with self.assertRaisesRegex(ValueError, "位于同一目录"):
            self.run_import()

    def test_rank_deficient_actual_response_rejected(self):
        self.response[:, 2] = self.response[:, 0]
        self.evidence["trial13"]["residual_12d"] = (self.residual + 0.5*self.response[:,2]).tolist()
        self.proposal["probe_evidence"][2] = copy.deepcopy(self.evidence["trial13"])
        self.proposal["measured_response_normalized_12x3"] = self.response.tolist()
        with self.assertRaisesRegex(ValueError, "秩不足或病态"):
            self.run_import()

    def test_trial11_response_recomputed_from_actual_evidence(self):
        self.evidence["trial11"]["residual_12d"][0] += 0.01
        self.proposal["probe_evidence"][0] = copy.deepcopy(self.evidence["trial11"])
        with self.assertRaisesRegex(ValueError, "实测响应矩阵"):
            self.run_import()


class PairDiagnosticsTests(unittest.TestCase):
    def test_vector_and_scalar_differences(self):
        observed = np.array([0.,0.,0., 1.,2.,3., 3.,4.,0., 4.,6.,3.])
        residual = np.zeros(12); residual[6:9] = [-3.,1.,0.]
        residual[9:12] = [-3.,1.,0.]
        result = pair_diagnostics(observed, residual)
        self.assertEqual(result["target_separation_kpc"], 5.)
        self.assertEqual(result["simulated_separation_kpc"], 5.)
        self.assertEqual(result["separation_residual_kpc"], 0.)
        self.assertEqual(result["relative_position_residual_kpc"], [-3.,1.,0.])
        self.assertEqual(result["relative_speed_residual_kms"], 0.)
        self.assertEqual(result["relative_velocity_residual_kms"], [-3.,1.,0.])
        self.assertFalse(result["independent_observation"])

    def test_common_translation_and_boost_cancel(self):
        observed = np.arange(12, dtype=float)
        residual = np.array([1.,2.,3.,4.,5.,6.]*2)
        result = pair_diagnostics(observed, residual)
        self.assertEqual(result["separation_residual_kpc"], 0.)
        self.assertEqual(result["relative_speed_residual_kms"], 0.)
        self.assertEqual(result["relative_position_residual_kpc"], [0.]*3)

    def test_invalid_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "有限数值"):
            pair_diagnostics(np.zeros(12), np.full(12, np.nan))


if __name__ == "__main__":
    unittest.main()
