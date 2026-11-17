"""Effective production parameter equality and read-only recovery tests."""
import copy
from pathlib import Path
import tempfile
import unittest

from local_live_evidence import normalized_params, sha256
import local_live_params as p


class ParamSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'mw_isolated.param'
        self.base.write_text('''InitCondFile old_ic
OutputDir old_output
ICFormat 3
SnapFormat 3
OutputListOn 0
NumFilesPerSnapshot 1
NumFilesWrittenInParallel 1
SnapshotFileBase snapshot
RestartFile restart
TimeBegin 0
TimeMax 0.6136
InitGasTemp 0
TimeOfFirstSnapshot 0
TimeBetSnapshot 0.1
SofteningGas 0.005
InitMetallicity 0.3
CritPhysDensity 1000
UnitMass_in_g 1.989e43
MaxSizeTimestep 0.001
MinSizeTimestep 4e-6
CoolingTable ./tables/table.dat
''')
        self.original_hash = sha256(self.base)
        self.anchor = dict(base_param_path=str(self.base), base_param_sha256=self.original_hash,
                           stage2_start_code=0.6136, target_time_code=1.636312026336,
                           trial='trial10', work_dir=str(self.root))
        self.runtime = p._anchor_runtime_path(self.anchor)
        self.runtime.parent.mkdir(parents=True)
        lines = self.base.read_text().splitlines()
        overrides = p.production_overrides(self.anchor)
        out = []
        for line in lines:
            key, val = line.split(maxsplit=1)
            out.append(f'{key} {overrides.get(key, val)}')
        self.runtime.write_text('\n'.join(out) + '\n')
        self.anchor.update(runtime_param_sha256=sha256(self.runtime),
                           runtime_param_normalized=normalized_params(self.runtime))

    def test_original_template_generates_identical_production(self):
        report = p.assert_base_param_compatible(self.anchor, self.base)
        self.assertTrue(report['compatible'])
        self.assertTrue(report['byte_identical'])

    def test_comments_spacing_and_numeric_spelling_accepted(self):
        self.base.write_text(self.base.read_text().replace('SofteningGas 0.005',
                             'SofteningGas     5e-3 # same softening') + '\n% comment\n')
        report = p.assert_base_param_compatible(self.anchor, self.base)
        self.assertFalse(report['byte_identical'])
        self.assertEqual(report['differences'], [])

    def test_true_physics_changes_are_named_and_rejected(self):
        original = self.base.read_text()
        for old, new, key in [('SofteningGas 0.005', 'SofteningGas 0.01', 'SofteningGas'),
                             ('InitMetallicity 0.3', 'InitMetallicity 0.1', 'InitMetallicity'),
                             ('CritPhysDensity 1000', 'CritPhysDensity 100', 'CritPhysDensity'),
                             ('UnitMass_in_g 1.989e43', 'UnitMass_in_g 1.989e40', 'UnitMass_in_g'),
                             ('MaxSizeTimestep 0.001', 'MaxSizeTimestep 0.01', 'MaxSizeTimestep')]:
            with self.subTest(key=key):
                self.base.write_text(original.replace(old, new))
                self.assertEqual([d['key'] for d in p.assess_base_param(self.anchor, self.base)['differences']], [key])
                with self.assertRaisesRegex(ValueError, key):
                    p.assert_base_param_compatible(self.anchor, self.base)

    def test_generator_overridden_values_do_not_block(self):
        self.base.write_text(self.base.read_text().replace('TimeMax 0.6136', 'TimeMax 0.8')
                             .replace('TimeBegin 0\n', 'TimeBegin 0.7\n')
                             .replace('TimeBetSnapshot 0.1', 'TimeBetSnapshot 0.05')
                             .replace('InitCondFile old_ic', 'InitCondFile new_ic'))
        self.assertTrue(p.assess_base_param(self.anchor, self.base)['compatible'])

    def test_unknown_and_missing_keys_are_not_ignored(self):
        original = self.base.read_text()
        self.base.write_text(original + 'UnknownPhysicsFlag 1\n')
        self.assertFalse(p.assess_base_param(self.anchor, self.base)['compatible'])
        self.base.write_text(original.replace('CoolingTable ./tables/table.dat\n', ''))
        self.assertFalse(p.assess_base_param(self.anchor, self.base)['compatible'])

    def test_duplicate_and_nonfinite_rejected(self):
        original = self.base.read_text()
        self.base.write_text(original + 'SofteningGas 0.005\n')
        with self.assertRaisesRegex(ValueError, '重复'): p.assess_base_param(self.anchor, self.base)
        self.base.write_text(original.replace('TimeMax 0.6136', 'TimeMax nan'))
        with self.assertRaisesRegex(ValueError, '非有限'): p.assess_base_param(self.anchor, self.base)

    def test_recovery_does_not_require_old_base(self):
        self.base.unlink()
        source_before = self.runtime.read_bytes()
        output = self.root / 'shooting_t10.param'
        report = p.recover_base_param(self.anchor, output)
        self.assertTrue(report['compatible'])
        self.assertFalse(report['reused_existing'])
        self.assertEqual(output.read_bytes(), source_before)
        self.assertEqual(self.runtime.read_bytes(), source_before)
        self.assertFalse(self.base.exists())
        self.assertTrue(p.recover_base_param(self.anchor, output)['reused_existing'])

    def test_recovery_does_not_overwrite_existing_or_original(self):
        output = self.root / 'shooting_t10.param'
        output.write_text('keep me')
        with self.assertRaisesRegex(ValueError, '覆盖'): p.recover_base_param(self.anchor, output)
        self.assertEqual(output.read_text(), 'keep me')
        with self.assertRaisesRegex(ValueError, '副本'): p.recover_base_param(self.anchor, self.base)
        self.assertEqual(sha256(self.base), self.original_hash)

    def test_modified_runtime_rejected(self):
        self.runtime.write_text(self.runtime.read_text() + '% altered\n')
        with self.assertRaisesRegex(ValueError, '指纹'): p.recover_base_param(self.anchor, self.root / 'new.param')
        self.assertFalse((self.root / 'new.param').exists())

    def test_runtime_normalized_freeze_is_also_checked(self):
        self.anchor['runtime_param_normalized']['SofteningGas'] = 0.02
        with self.assertRaisesRegex(ValueError, '有效参数'): p.recover_base_param(self.anchor, self.root / 'new.param')

    def test_other_directory_rejected(self):
        other = self.root / 'other'; other.mkdir()
        candidate = other / 'copy.param'; candidate.write_bytes(self.base.read_bytes())
        with self.assertRaisesRegex(ValueError, '同一目录'): p.assess_base_param(self.anchor, candidate)
        with self.assertRaisesRegex(ValueError, '同一目录'): p.recover_base_param(self.anchor, other / 'recovery.param')

    def test_symlink_recovery_destination_rejected(self):
        output = self.root / 'link.param'; output.symlink_to(self.base)
        with self.assertRaisesRegex(ValueError, '符号链接'): p.recover_base_param(self.anchor, output)


if __name__ == '__main__':
    unittest.main()
