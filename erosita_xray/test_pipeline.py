#!/usr/bin/env python3
"""Synthetic I/O integration tests. These do not run or simulate GIZMO physics."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import h5py
import numpy as np
import satellite_relax_common as common
import merge_relaxed_mw_lmc_smc as merger

ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="tg19_pipeline_test_")
        cls.work = Path(cls.temp.name)
        cls.base = cls.work/"mw.param"
        cls.base.write_text("""UnitLength_in_cm 3.08568e21
UnitMass_in_g 1.989e43
UnitVelocity_in_cm_per_s 1e5
ComovingIntegrationOn 0
HubbleParam 1
BoxSize 1000
TimeBegin 0
TimeMax 0.6136
InitGasTemp 0
InitMetallicity 0.3
InitStellarAge 4
BlackHoleAccretionFactor 0
BlackHoleEddingtonFactor 0
BlackHoleFeedbackFactor 0
MaxSizeTimestep 0.001
MinSizeTimestep 4e-6
CritPhysDensity 1000
SfEffPerFreeFall 1
""")
        cls.exe = cls.work/"FAKE_GIZMO_TEST_ONLY"
        cls.exe.write_text("#!/usr/bin/env bash\nexit 99\n")
        cls.exe.chmod(0o755)
        cls.raw,cls.units = common.param_units(common.read_param(cls.base)[2])
        for system in ("lmc","smc"):
            run = cls.work/(system+"re")
            cls.call(f"make_{system}_relax_ic.py","--base-param",cls.base,
                "--gizmo-executable",cls.exe,"--run-dir",run,"--relax-gyr","0.6136",
                "--gas-particle-mass","1e7","--dm-particle-mass","1e8","--star-particle-mass","1e7")
            manifest = json.loads((run/"relax_manifest.json").read_text())
            final = run/"output/snapshot_007.hdf5"
            shutil.copyfile(run/f"{system}_ic.hdf5",final)
            cls.complete_synthetic_snapshot(final,manifest["target_end_code"])
            with h5py.File(final,"r+") as f:
                # A gas-born star retains a gas parent ID; identities are per type.
                f["PartType4/ParticleIDs"][0] = f["PartType0/ParticleIDs"][0]
                f["PartType4/ParticleChildIDsNumber"][0] = 1
                f["PartType4/ParticleIDGenerationNumber"][0] = 1
        cls.mw = cls.work/"mw_snapshot_007.hdf5"
        shutil.copyfile(cls.work/"lmcre/output/snapshot_007.hdf5",cls.mw)
        with h5py.File(cls.mw,"r+") as f:
            f["Header"].attrs["Time"] = .7136
            for p in (0,1,4):
                g = f[f"PartType{p}"]
                g["ParticleIDs"][:] = np.arange(1+p*10000,1+p*10000+len(g["ParticleIDs"]),dtype=np.uint32)
                if p in (0,4):
                    g["Metallicity"][:] = np.tile(common.backend.stellar_metallicity_vector(.3),(len(g["ParticleIDs"]),1))
            # MW systemic frame differs from the satellites' standalone frames.
            f["PartType4/Velocities"][:] += np.array([11.,-7.,3.])
            f["PartType4/StellarFormationTime"][:] = -.75
            bh=f.create_group("PartType5")
            for key,value in dict(Coordinates=[[500.,500.,500.]],Velocities=[[11.,-7.,3.]],
                Masses=[.0004],BH_Mass=[.0004],ParticleIDs=np.array([90000000],dtype=np.uint32),
                ParticleChildIDsNumber=np.array([0],dtype=np.uint32),ParticleIDGenerationNumber=np.array([0],dtype=np.uint32),
                Potential=[-123.],Acceleration=[[1.,2.,3.]]).items():
                bh.create_dataset(key,data=value)
            counts=f["Header"].attrs["NumPart_ThisFile"];counts[5]=1
            f["Header"].attrs["NumPart_ThisFile"]=counts
            f["Header"].attrs["NumPart_Total"]=counts
        cls.out = cls.work/"trial00_000.hdf5"
        cls.merge_command(cls.out,"--lmc-position-offset=1,2,3","--smc-velocity-offset=4,5,6")
        cls.report = json.loads(cls.out.with_suffix(".orbit.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def call(cls,script,*args,success=True):
        result=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],capture_output=True,text=True)
        if success and result.returncode:
            raise AssertionError(result.stdout+"\n"+result.stderr)
        if not success and result.returncode==0:
            raise AssertionError("Expected failure but command succeeded")
        return result

    @classmethod
    def merge_command(cls,out,*extra,success=True):
        return cls.call("merge_relaxed_mw_lmc_smc.py","merge","--mw-snapshot",cls.mw,
            "--lmc-run-dir",cls.work/"lmcre","--smc-run-dir",cls.work/"smcre",
            "--base-param",cls.base,"--gizmo-executable",cls.exe,
            "--output",out,"--lookback-gyr","0.02","--orbit-samples","21",*extra,success=success)

    @classmethod
    def complete_synthetic_snapshot(cls,path,time):
        with h5py.File(path,"r+") as f:
            f["Header"].attrs["Time"]=time
            config=f.create_group("Config")
            for key in ("HYDRO_MESHLESS_FINITE_MASS","METALS","GALSF","GALSF_FB_MECHANICAL","COOLING","COOL_LOW_TEMPERATURES"):
                config.attrs[key]=""
            for p in (0,1,4):
                g=f[f"PartType{p}"];n=len(g["ParticleIDs"])
                g.create_dataset("Potential",data=np.full(n,-123.))
                g.create_dataset("Acceleration",data=np.tile([1.,2.,3.],(n,1)))
                if p==0:
                    for key,value in dict(Density=np.linspace(.001,.1,n),SmoothingLength=np.full(n,.4),
                            ElectronAbundance=np.full(n,.15),Temperature=np.full(n,1e4),
                            Vorticity=np.tile([1.,2.,3.],(n,1)),CustomScalar=np.linspace(1.,2.,n)).items():
                        g.create_dataset(key,data=value)
                    # Preserve nonuniform enrichment, not only an initial uniform Z.
                    z=g["Metallicity"][...];z[:,0]*=np.linspace(1,1.2,n);g["Metallicity"][:]=z
            # Simulate two particles with the same parent but distinct child identities.
            gas=f["PartType0"]
            gas["ParticleIDs"][1]=gas["ParticleIDs"][0]
            gas["ParticleChildIDsNumber"][1]=1

    def test_01_physical_duration_and_param_inheritance(self):
        for system in ("lmc","smc"):
            manifest=json.loads((self.work/(system+"re")/"relax_manifest.json").read_text())
            self.assertAlmostEqual(manifest["target_end_gyr"],.6136,places=12)
            self.assertAlmostEqual(manifest["target_end_code"]*self.units["time_gyr"],.6136,places=12)
            params=common.read_param(self.work/(system+"re")/f"{system}.param")[2]
            self.assertEqual(params["CritPhysDensity"],"1000")
            self.assertEqual(float(params["InitStellarAge"]),4.)
            self.assertEqual(float(params["InitMetallicity"]),.2 if system=="lmc" else .1)
            self.assertNotIn("GALSF_EFFECTIVE_EQS",params)

    def test_02_mass_metals_ages_and_scalar_state_preserved(self):
        with h5py.File(self.out,"r") as out:
            self.assertEqual(out["Header"].attrs["Flag_Metals"],11)
            for idx,path in enumerate(self.report["source_files"]):
                with h5py.File(path,"r") as src:
                    for p in (0,1,4):
                        dest=out[f"PartType{p}"];g=src[f"PartType{p}"]
                        mask=out[f"MergeParticleOrigins/PartType{p}"][...]==idx
                        for key in ("Masses",*common.ID_FIELDS,"Metallicity","InternalEnergy","Density","SmoothingLength","ElectronAbundance","CustomScalar"):
                            if key in g: np.testing.assert_array_equal(dest[key][...][mask],g[key][...])
                        if p==4:
                            oldage=float(src["Header"].attrs["Time"])-g["StellarFormationTime"][...]
                            newage=float(out["Header"].attrs["Time"])-dest["StellarFormationTime"][...][mask]
                            np.testing.assert_allclose(newage,oldage,rtol=1e-14,atol=1e-14)
            np.testing.assert_array_equal(out["PartType5/BH_Mass"][...],[.0004])
            self.assertLess(out["MagellanicOrbitReference"].attrs["LMC_STAR_ID_START"],120000000)

    def test_03_rigid_transform_and_gravity_reinitialization(self):
        with h5py.File(self.out,"r") as out:
            for idx,path in enumerate(self.report["source_files"]):
                with h5py.File(path,"r") as src:
                    for p in (0,1,4):
                        mask=out[f"MergeParticleOrigins/PartType{p}"][...]==idx
                        x=out[f"PartType{p}/Coordinates"][...][mask]
                        v=out[f"PartType{p}/Velocities"][...][mask]
                        xs=src[f"PartType{p}/Coordinates"][...]
                        vs=src[f"PartType{p}/Velocities"][...]
                        np.testing.assert_allclose(np.linalg.norm(x-x[0],axis=1),np.linalg.norm(xs-xs[0],axis=1),rtol=1e-10,atol=1e-10)
                        np.testing.assert_allclose(np.linalg.norm(v-v[0],axis=1),np.linalg.norm(vs-vs[0],axis=1),rtol=1e-10,atol=1e-10)
                        if idx==0:
                            np.testing.assert_array_equal(x,xs);np.testing.assert_array_equal(v,vs)
                        self.assertTrue(np.all(out[f"PartType{p}/Potential"][...][mask]==0))

    def test_04_cumulative_shooting_and_no_reorientation(self):
        csv=self.work/"residual.csv"
        csv.write_text("satellite,component,residual\n"+"\n".join(f"{s},{c},{.2 if c=='x' else 0}" for s in ("LMC","SMC") for c in ("x","y","z","vx","vy","vz"))+"\n")
        out=self.work/"trial01_000.hdf5"
        self.merge_command(out,"--shooting-base-ic",self.out,"--shooting-residual-csv",csv,"--trial-label","trial01")
        r=json.loads(out.with_suffix(".orbit.json").read_text())
        np.testing.assert_array_equal(r["previous_offset_12d"],self.report["cumulative_offset_12d"])
        np.testing.assert_allclose(np.array(r["previous_offset_12d"])+r["this_iteration_increment_12d"],r["cumulative_offset_12d"])
        np.testing.assert_array_equal(r["rotation_matrices"],self.report["rotation_matrices"])
        result=self.merge_command(self.work/"bad_000.hdf5","--shooting-residual-csv",csv,success=False)
        self.assertIn("shooting-base-ic",result.stderr)

    def test_05_missing_state_unit_mismatch_and_unknown_vectors_rejected(self):
        sources=[common.read_snapshot(path,self.raw) for path in self.report["source_files"]]
        del sources[1]["parts"][0]["ElectronAbundance"]
        with self.assertRaisesRegex(ValueError,"ElectronAbundance"):
            merger.assert_compatible(sources)
        copyfile=self.work/"bad_units.hdf5"
        shutil.copyfile(self.mw,copyfile)
        with h5py.File(copyfile,"r+") as f:f["Header"].attrs["UnitMass_in_g"]=1e40
        with self.assertRaisesRegex(ValueError,"单位"):
            common.read_snapshot(copyfile,self.raw)
        src=common.read_snapshot(self.report["source_files"][1],self.raw)
        src["parts"][0]["UnknownVector"]=np.ones((len(src["parts"][0]["Masses"]),3))
        with self.assertRaisesRegex(ValueError,"未识别"):
            merger.transform_parts(src,"LMC",self.units,np.zeros(3),np.zeros(3),np.eye(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),.7)

    def test_06_verify_detects_metal_reset_despite_reordering(self):
        snap=self.work/"smoke.hdf5";shutil.copyfile(self.out,snap)
        with h5py.File(snap,"r+") as f:
            for p in (0,1,4,5):
                for ds in f[f"PartType{p}"].values():ds[:]=ds[...][::-1]
        result=self.call("merge_relaxed_mw_lmc_smc.py","verify","--reference-ic",self.out,"--snapshot",snap)
        self.assertIn("PASS",result.stdout)
        with h5py.File(snap,"r+") as f:f["PartType0/Metallicity"][:]=0.
        self.call("merge_relaxed_mw_lmc_smc.py","verify","--reference-ic",self.out,"--snapshot",snap,success=False)

    def test_07_shell_and_stage2_times(self):
        work=self.out.with_suffix("").with_name(self.out.stem+"_run")
        for path in (self.work/"lmcre/run_relax.sh",self.work/"smcre/run_relax.sh",work/"smoke/run.sh",work/"production/run.sh"):
            subprocess.run(["bash","-n",str(path)],check=True)
        p=common.read_param(work/"production/stage2.param")[2]
        self.assertAlmostEqual(float(p["TimeBegin"]),.7136)
        self.assertAlmostEqual((float(p["TimeMax"])-float(p["TimeBegin"]))*self.units["time_gyr"],.02)
        self.assertIn(" 2 2>&1",(work/"production/run.sh").read_text())


if __name__=="__main__":
    unittest.main(verbosity=2)
