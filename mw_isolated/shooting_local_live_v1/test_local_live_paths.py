"""GIZMO HDF5 前缀、运行目录和跨轮路径回归测试；不执行 GIZMO。"""
from pathlib import Path
import tempfile
import unittest
import local_live_evidence as ev
import live_shooting_round_v1  # sets the private pipeline import path
from merge_relaxed_mw_lmc_smc_local_live_v1 import create_stage2_files


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.ic=self.root/"mw_trial10_000.hdf5";self.ic.touch()
        self.prefix=self.ic.with_suffix("")
        self.output=self.root/"trial10_run/production/output";self.output.mkdir(parents=True)
        self.param=self.output.parent/"stage2.param"

    def write(self,value,*,output=None,fmt=3):
        self.param.write_text(f"InitCondFile {value}\nICFormat {fmt}\nOutputDir {output or self.output}\nSnapshotFileBase snapshot\nRestartFile restart\n")

    def check(self,**kwargs):
        return ev.check_runtime_paths(self.param,self.ic,self.output,self.root,**kwargs)

    def test_absolute_hdf5_prefix(self):
        self.write(self.prefix);self.check(require_hdf5_basename=True)

    def test_relative_prefix_uses_running_cwd(self):
        self.write("./"+self.prefix.name,output="./trial10_run/production/output/")
        self.check(require_hdf5_basename=True)
        with self.assertRaisesRegex(ValueError,"运行目录="):
            ev.check_runtime_paths(self.param,self.ic,self.output,self.param.parent)

    def test_legacy_full_path_accepted_only_for_history(self):
        self.write(self.ic);self.check()
        with self.assertRaisesRegex(ValueError,"去掉 .hdf5"):
            self.check(require_hdf5_basename=True)

    def test_wrong_trial_and_same_basename_elsewhere_rejected(self):
        for value in (self.root/"mw_trial11_000",self.root/"elsewhere"/self.prefix.name):
            with self.subTest(value=value):
                Path(str(value)+".hdf5").parent.mkdir(exist_ok=True)
                Path(str(value)+".hdf5").touch()
                self.write(value)
                with self.assertRaisesRegex(ValueError,"本轮期望="):self.check()

    def test_output_directory_is_not_suffix_normalized(self):
        self.write(self.prefix,output=str(self.output)+".hdf5")
        with self.assertRaisesRegex(ValueError,"OutputDir"):self.check()

    def test_binary_formats_do_not_add_hdf5(self):
        for fmt in (1,2):
            with self.subTest(fmt=fmt):
                self.write(self.prefix,fmt=fmt)
                with self.assertRaisesRegex(ValueError,"InitCondFile"):self.check()

    def test_split_file_shadow_is_rejected(self):
        self.write(self.prefix)
        Path(str(self.prefix)+".0.hdf5").touch()
        with self.assertRaisesRegex(ValueError,"分片"):self.check()

    def test_double_extension_shadow_is_rejected(self):
        self.write(self.ic)
        Path(str(self.ic)+".hdf5").touch()
        with self.assertRaisesRegex(ValueError,"歧义"):self.check()

    def test_generator_writes_prefix_for_both_modes(self):
        base=self.root/"base.param";base.write_text("MinSizeTimestep 4e-6\n")
        work=create_stage2_files(self.ic,base,base.read_text().splitlines(),
            {"MinSizeTimestep":"4e-6"},{"path":"/not/executed/GIZMO","sha256":"a"*64},
            .6136,1.,{"time_gyr":.97779222168},6)
        for mode in ("smoke","production"):
            p=work/mode/"stage2.param"
            self.assertEqual(ev.read_params(p)["InitCondFile"],str(self.prefix))
            ev.check_runtime_paths(p,self.ic,work/mode/"output",self.root,require_hdf5_basename=True)
            self.assertIn(" 2 2>&1",(work/mode/"run.sh").read_text())


if __name__=="__main__":unittest.main()
