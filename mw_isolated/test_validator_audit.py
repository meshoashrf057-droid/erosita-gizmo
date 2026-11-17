#!/usr/bin/env python3
"""Synthetic diagnostics tests; these do not validate GIZMO dynamics."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

import validation_dashboard_sat_6d_gaswake_v3 as validator


class ValidatorAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tg19_validator_audit_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.ic = self.folder / "reference_000.hdf5"
        self.end = self.folder / "snapshot_011.hdf5"
        self.write_fixture(self.ic, 1.0)
        self.write_fixture(self.end, 2.0)
        self.ref = validator.load_reference(self.ic)
        self.row = validator.inspect_snapshot(self.end, self.ref, 8.0, 3.0, 1e-6)

    @staticmethod
    def write_fixture(path, time):
        rng = np.random.default_rng(91)
        ids, positions, velocities = [], [], []
        # Each satellite includes stars formed from gas IDs and original stars.
        for old_start, gas_start, center in ((120_000_000, 100_000_000, [50., 0., 0.]),
                                             (210_000_000, 220_000_000, [80., 0., 0.])):
            ids.extend(np.arange(old_start, old_start+80))
            positions.extend(np.asarray(center)+rng.normal(0., .15, (80, 3)))
            velocities.extend(np.zeros((80, 3)))
            ids.extend(np.arange(gas_start, gas_start+40))
            positions.extend(np.asarray(center)+[1., 0., 0.]+rng.normal(0., .1, (40, 3)))
            velocities.extend(np.tile([30., 0., 0.], (40, 1)))
        ids.extend(np.arange(1, 201))
        positions.extend(rng.normal(0., .1, (200, 3)))
        velocities.extend(np.zeros((200, 3)))
        with h5py.File(path, "w") as f:
            f.create_group("Header").attrs["Time"] = time
            for ptype, pid, pos, vel in (
                (4, ids, positions, velocities),
                (1, [110_000_000, 200_000_000], [[50., 0., 0.], [80., 0., 0.]], np.zeros((2, 3))),
                (5, [90_000_000], [[0., 0., 0.]], [[0., 0., 0.]]),
            ):
                group = f.create_group(f"PartType{ptype}")
                group["ParticleIDs"] = np.asarray(pid, dtype=np.uint64)
                group["Coordinates"] = np.asarray(pos, dtype=float)
                group["Velocities"] = np.asarray(vel, dtype=float)
                group["Masses"] = np.ones(len(pid))
            gas = f.create_group("PartType0")
            gas["TimeStep"] = np.array([.01])
            model = f.create_group("MagellanicModelReference")
            model.attrs.update(SATELLITE_MODEL_KEY="test-model-key", TRIAL_LABEL="trial01",
                               SATELLITE_STRUCTURE_MODEL="tg19-independent-relax")
            f.create_group("MergeProvenance").attrs["VERSION"] = "tg19-independent-relax-v1"
            orbit = f.create_group("MagellanicOrbitReference")
            orbit.attrs.update(Lookback_Gyr=1., InsertionTime_Code=1., PresentTargetTime_Code=2.,
                               UnitTime_Gyr=1., MW_BH_ID=90_000_000,
                               ReferenceCurveIncludesAppliedShootingOffsets=0)
            orbit["TimeFromPresent_Gyr"] = [-1., 0.]
            for sat, gas_lo, dm_lo, old_lo, center in (
                ("LMC", 100_000_000, 110_000_000, 120_000_000, [50., 0., 0.]),
                ("SMC", 220_000_000, 200_000_000, 210_000_000, [80., 0., 0.]),
            ):
                for key, value in {"STAR_ID_START": min(gas_lo, old_lo), "STAR_ID_END": max(gas_lo+9999, old_lo+79),
                                   "DM_ID_START": dm_lo, "DM_ID_END": dm_lo,
                                   "ID_START": 100_000_000 if sat == "LMC" else 200_000_000,
                                   "ID_END": 199_999_999 if sat == "LMC" else 299_999_999,
                                   "TotalMass_Msun": 100.}.items():
                    orbit.attrs[f"{sat}_{key}"] = value
                orbit[f"{sat}_Position_kpc"] = np.tile(center, (2, 1))
                orbit[f"{sat}_Velocity_kms"] = np.zeros((2, 3))
                orbit[f"{sat}_Current_PositionSigma_kpc"] = np.ones(3)
                orbit[f"{sat}_Current_VelocitySigma_kms"] = np.ones(3)

    def evaluate(self, rows):
        return validator.evaluate_present(rows, self.ref, 3., False, .5, .3, .001, False)

    def test_single_exact_target_is_covered(self):
        evaluation = self.evaluate([self.row])
        self.assertTrue(evaluation["target_time_bracketed"])
        self.assertEqual(evaluation["evaluation_method"], "exact target snapshot")
        self.assertEqual(validator.target_snapshot_sources([self.row], 2.)[0]["weight"], 1.)

    def test_audit_is_additive_and_detects_extra_stars(self):
        row_before = copy.deepcopy(self.row)
        before_path = self.folder / "before.csv"
        after_path = self.folder / "after.csv"
        validator.write_present_table(self.evaluate([self.row]), before_path)
        audit = validator.build_stellar_tracer_audit(self.ic, [self.row], self.ref, 8., 3.)
        validator.write_present_table(self.evaluate([self.row]), after_path)
        self.assertEqual(before_path.read_bytes(), after_path.read_bytes())
        self.assertEqual(self.row, row_before)
        self.assertFalse(audit["baseline_measurement_changed"])
        for sat in ("lmc", "smc"):
            report = audit["reference_ic"]["satellites"][sat]
            self.assertEqual(report["legacy_non_original_star_count"], 40)
            self.assertGreater(report["delta_velocity_norm_kms"], 5.)
            self.assertGreater(audit["target_old_minus_legacy"][sat]["delta_position_norm_kpc"], .1)

    def test_provenance_binds_csv_and_rejects_uncovered_target(self):
        row = copy.deepcopy(self.row)
        row["time_code"] = 1.8
        evaluation = self.evaluate([row])
        table = self.folder / "magellanic_present_6d.csv"
        validator.write_present_table(evaluation, table)
        sidecar = validator.write_present_provenance(table, self.ic, self.ref, [row], evaluation, 8., 3.)
        data = json.loads(sidecar.read_text())
        self.assertFalse(data["target_covered"])
        self.assertFalse(data["shooting_usable"])
        self.assertEqual(data["reference_ic_sha256"], validator.sha256_file(self.ic))
        self.assertEqual(data["residual_csv_sha256"], validator.sha256_file(table))
        self.assertEqual(data["model_key"], "test-model-key")
        self.assertEqual(data["measurement_signature"], validator.measurement_signature(8., 3.))

    def test_original_star_ranges_not_guessed_for_unrecognized_model(self):
        with h5py.File(self.ic, "r+") as f:
            f["MagellanicModelReference"].attrs["SATELLITE_STRUCTURE_MODEL"] = "another-model"
        with self.assertRaisesRegex(RuntimeError, "cannot identify"):
            validator.load_reference(self.ic)
        reference = validator.load_reference(self.ic, "legacy")
        self.assertIsNone(reference["old_star_id_ranges"])
        audit = validator.build_stellar_tracer_audit(self.ic, [self.row], reference, 8., 3.)
        self.assertEqual(audit["status"], "not_available")

    def append_star(self, path, particle_id, position=(81., 0., 0.), velocity=(30., 0., 0.), child=0):
        with h5py.File(path, "r+") as f:
            group = f["PartType4"]
            for name in list(group):
                values = group[name][:]
                extra = {"ParticleIDs": particle_id, "Coordinates": position,
                         "Velocities": velocity, "Masses": 1.,
                         "ParticleChildIDsNumber": child, "ParticleIDGenerationNumber": 0}[name]
                del group[name]
                group[name] = np.concatenate((values, np.asarray([extra], dtype=values.dtype)))

    def test_newborn_growth_does_not_move_fixed_center(self):
        for index in range(80):
            self.append_star(self.end, 220_000_100+index)
        row = validator.inspect_snapshot(self.end, self.ref, 8., 3., 1e-6)
        self.assertEqual(row["smc_tracer_fraction"], 1.)
        for component in ("x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms"):
            self.assertEqual(row[f"smc_{component}"], self.row[f"smc_{component}"])
        legacy = validator.load_reference(self.ic, "legacy")
        legacy_row = validator.inspect_snapshot(self.end, legacy, 8., 3., 1e-6)
        self.assertAlmostEqual(legacy_row["smc_tracer_fraction"], 200/120)
        self.assertGreater(legacy_row["smc_vx_kms"], row["smc_vx_kms"]+10.)

    def test_new_star_cannot_hide_missing_original_identity(self):
        with h5py.File(self.end, "r+") as f:
            ids = f["PartType4/ParticleIDs"][:]
            ids[np.flatnonzero(ids == 210_000_000)[0]] = 220_000_100
            f["PartType4/ParticleIDs"][:] = ids
        row = validator.inspect_snapshot(self.end, self.ref, 8., 3., 1e-6)
        self.assertAlmostEqual(row["smc_tracer_fraction"], 79/80)
        self.assertFalse(self.evaluate([row])["satellite_structure_pass"])
        legacy = validator.load_reference(self.ic, "legacy")
        self.assertEqual(validator.inspect_snapshot(self.end, legacy, 8., 3., 1e-6)["smc_tracer_fraction"], 1.)

    def test_initial_aperture_retention_is_one(self):
        row = validator.inspect_snapshot(self.ic, self.ref, 8., 3., 1e-6)
        self.assertEqual(row["lmc_mass_r30_retention"], 1.)
        self.assertEqual(row["smc_mass_r15_retention"], 1.)

    def test_composite_identity_excludes_new_child_and_rejects_missing_field(self):
        for path in (self.ic, self.end):
            with h5py.File(path, "r+") as f:
                group = f["PartType4"]
                group["ParticleChildIDsNumber"] = np.zeros(len(group["ParticleIDs"]), dtype=np.uint32)
                group["ParticleIDGenerationNumber"] = np.zeros(len(group["ParticleIDs"]), dtype=np.uint32)
        ref = validator.load_reference(self.ic)
        self.append_star(self.end, 210_000_000, child=1)
        row = validator.inspect_snapshot(self.end, ref, 8., 3., 1e-6)
        self.assertEqual(row["smc_tracer_fraction"], 1.)
        self.assertEqual(row["smc_vx_kms"], self.row["smc_vx_kms"])
        with h5py.File(self.end, "r+") as f:
            del f["PartType4/ParticleChildIDsNumber"]
        with self.assertRaisesRegex(RuntimeError, "identity fields differ"):
            validator.inspect_snapshot(self.end, ref, 8., 3., 1e-6)

    def test_duplicate_complete_identity_is_rejected(self):
        self.append_star(self.end, 210_000_000)
        with self.assertRaisesRegex(RuntimeError, "duplicate complete"):
            validator.inspect_snapshot(self.end, self.ref, 8., 3., 1e-6)

    def test_explicit_old_and_all_star_metadata_preserves_legacy(self):
        legacy_before = validator.load_reference(self.ic, "legacy")
        with h5py.File(self.ic, "r+") as f:
            g = f["MagellanicOrbitReference"]
            for sat, bounds in validator.OLD_STAR_BLOCKS.items():
                for suffix, value in zip(("START", "END"), bounds):
                    g.attrs[f"{sat.upper()}_OLD_STAR_ID_{suffix}"] = value
                    g.attrs[f"{sat.upper()}_ALL_STAR_ID_{suffix}"] = g.attrs[f"{sat.upper()}_STAR_ID_{suffix}"]
                    g.attrs[f"{sat.upper()}_STAR_ID_{suffix}"] = value
        legacy_after = validator.load_reference(self.ic, "legacy")
        a = validator.inspect_snapshot(self.end, legacy_before, 8., 3., 1e-6)
        b = validator.inspect_snapshot(self.end, legacy_after, 8., 3., 1e-6)
        for key, value in a.items():
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(b[key]), key)
            else:
                self.assertEqual(value, b[key], key)
        audit = validator.stellar_tracer_audit(self.end, validator.load_reference(self.ic), 8., 3.)
        self.assertEqual(audit["satellites"]["smc"]["legacy_non_original_star_count"], 40)

    def test_legacy_uses_original_v3_values_and_signature_is_distinct(self):
        legacy = validator.load_reference(self.ic, "legacy")
        row = validator.inspect_snapshot(self.end, legacy, 8., 3., 1e-6)
        # These are the original v3 mass-weighted range-based estimator outputs.
        self.assertAlmostEqual(row["smc_vx_kms"], 10.)
        self.assertEqual(row["smc_total_original_nstar"], 120)
        self.assertEqual(row["smc_tracer_fraction"], 1.)
        self.assertNotEqual(validator.measurement_signature(8., 3.),
                            validator.measurement_signature(8., 3., "legacy"))

    def test_time_panels_do_not_share_category_axis(self):
        fig = validator.plot_dashboard([self.row], self.ref, self.evaluate([self.row]), self.folder/"dashboard.png")
        axes = fig.axes
        for index in (0, 1, 2, 3, 4):
            self.assertLess(axes[index].get_xlim()[1], 1.1)
            self.assertFalse(axes[index].get_shared_x_axes().joined(axes[index], axes[5]))
        self.assertEqual([label.get_text() for label in axes[5].get_xticklabels()], ["x", "y", "z", "vx", "vy", "vz"])

    def test_legacy_matches_uploaded_v3_when_available(self):
        original_path = Path(__file__).parents[2]/"upload"/"validation_dashboard_sat_6d_gaswake_v3(1).py"
        if not original_path.exists():
            self.skipTest("original uploaded v3 is not part of the distributed suite")
        spec = importlib.util.spec_from_file_location("original_validator_v3", original_path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        old_ref = original.load_reference(self.ic)
        old_row = original.inspect_snapshot(self.end, old_ref, 8., 3., 1e-6)
        new_ref = validator.load_reference(self.ic, "legacy")
        new_row = validator.inspect_snapshot(self.end, new_ref, 8., 3., 1e-6)
        for key, value in old_row.items():
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(new_row[key]), key)
            else:
                self.assertEqual(new_row[key], value, key)
        old_eval = original.evaluate_present([old_row, dict(old_row, time_code=1.9)], old_ref, 3., False, .5, .3, .001, False)
        new_eval = validator.evaluate_present([new_row, dict(new_row, time_code=1.9)], new_ref, 3., False, .5, .3, .001, False)
        for sat in ("lmc", "smc"):
            np.testing.assert_array_equal(old_eval[f"{sat}_residual_6d"], new_eval[f"{sat}_residual_6d"])


if __name__ == "__main__":
    unittest.main()
