"""实际小 HDF5/CSV 的流程测试；不运行 GIZMO。"""
import copy
import csv
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import h5py
import numpy as np
import local_live_evidence as ev
import local_live_import as imp
import live_shooting_round_v1 as cli


class FileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.param=self.root/"mw_isolated.param";self.param.write_text("MaxSizeTimestep 0.001\n")
        self.binary=self.root/"GIZMO";self.binary.write_bytes(b"synthetic binary, never execute")
        self.binary.chmod(0o755)
        self.q=np.arange(12,dtype=float);self.r=np.zeros(12);self.r[[0,1,7]]=[-3.1,-1.1,3.2]
        self.J=np.zeros((12,3));self.J[[0,1,7],[0,1,2]]=[4,1,-4]
        self.write_trial("trial10",self.q,self.r)
        self.a=ev.read_evidence(self.root,"trial10")
        for trial,index,h in cli.PROBES:
            delta=np.zeros(12);delta[index]=h
            col=[0,1,7].index(index)
            self.write_trial(trial,self.q+delta,self.r+self.J[:,col]*h,self.a,delta)
        self.state=self.root/"state.json"
        self.s=dict(schema="local-live-round-state-v2",anchor_trial="trial10",work_dir=str(self.root),
            anchor_evidence=self.a,history_dependency=False,
            base_param=str(self.param),base_param_sha256=ev.sha256(self.param),
            param_comparison_policy="effective-production-v1",
            execution_gizmo=dict(path=str(self.binary),sha256=ev.sha256(self.binary)),
            mpi_tasks=6,step_caps_kpc=[.5]*3,target_max=2.8,
            probes=[dict(trial=t,coordinate=ev.COORDS[i],index=i,step=h) for t,i,h in cli.PROBES],
            candidate_trial="trial14")
        self.state.write_text(json.dumps(self.s))
        for module,name,value in [(cli,"ANCHOR_SHA",self.a["ic_sha256"]),
            (imp,"ANCHOR_IC_SHA256",self.a["ic_sha256"])]:
            p=patch.object(module,name,value);p.start();self.addCleanup(p.stop)
        self.proposal=self.root/"proposal.json"

    def write_trial(self,trial,q,r,anchor=None,delta=None,method="live-probe",proposal=None,base_param=None,gizmo_path=None,build_review=None):
        p=ev.paths(self.root,trial);p["csv"].parent.mkdir(parents=True,exist_ok=True)
        base_param=Path(base_param) if base_param else self.param
        gizmo_path=Path(gizmo_path) if gizmo_path else self.binary
        audit=dict(trial_label=trial,model_key="synthetic",model_key_inputs=dict(targets=np.zeros((4,3)).tolist()),
            source_sha256=["a"*64,"b"*64,"c"*64],rotation_matrices=[np.eye(3).tolist()]*3,
            applied_initial_12d=q.tolist(),nominal_initial_12d=self.q.tolist(),
            cumulative_offset_12d=(q-self.q).tolist(),stage2_start_code=.6,stage2_target_code=1.6,
            units=dict(length_kpc=1.,velocity_kms=1.,mass_msun=1e10,time_gyr=1.),
            base_param=str(base_param),base_param_sha256=ev.sha256(base_param),
            gizmo=dict(path=str(gizmo_path),sha256=ev.sha256(gizmo_path)))
        if build_review is not None:
            audit["build_review"]=build_review
        if anchor:
            audit.update(shooting_base_ic=anchor["ic_path"],previous_offset_12d=anchor["audit"]["cumulative_offset_12d"],
                this_iteration_increment_12d=delta.tolist(),manual_increment_12d=[0.]*12,shooting_alpha=None,
                shooting_residual_provenance=dict(reference_ic_sha256=anchor["ic_sha256"],
                    residual_csv_sha256=anchor["residual_csv_sha256"],measurement_signature=ev.SIGNATURE),
                shooting_diagnostics=dict(method=method,alpha_applied=False,alpha_applied_by_importer=False,
                    base_initial_12d=anchor["applied_initial_12d"],residual_12d=anchor["residual_12d"],sigma_12d=[1.]*12))
        with h5py.File(p["ic"],"w") as f:
            f.create_group("Header").attrs["Time"]=.6
            f.create_group("MergeProvenance").create_dataset("AuditJSON",data=json.dumps(audit),dtype=h5py.string_dtype())
            g=f.create_group("MagellanicOrbitReference")
            for i,s in enumerate(("LMC","SMC")):
                g.create_dataset(s+"_AppliedInitialPosition_kpc",data=q[6*i:6*i+3])
                g.create_dataset(s+"_AppliedInitialVelocity_kms",data=q[6*i+3:6*i+6])
        orbit=dict(audit,output_ic_sha256=ev.sha256(p["ic"]));p["orbit"].write_text(json.dumps(orbit))
        with p["csv"].open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["satellite","component","simulated","observed","residual","sigma","zscore","unit"]);w.writeheader()
            for i,c in enumerate(ev.COORDS):
                sat,component=c.split("_")
                w.writerow(dict(satellite=sat,component=component,simulated=r[i],observed=0.,residual=r[i],sigma=1.,zscore=r[i],unit="kpc" if i%6<3 else "km/s"))
        snap=p["output"]/"snapshot_011.hdf5"
        with h5py.File(snap,"w") as f:f.create_group("Header").attrs["Time"]=1.6
        provenance=dict(schema_version=1,trial_label=trial,reference_ic=str(p["ic"]),
            reference_ic_sha256=orbit["output_ic_sha256"],residual_csv_sha256=ev.sha256(p["csv"]),
            model_key="synthetic",measurement_signature=ev.SIGNATURE,stellar_tracer_policy="original-stars",
            target_covered=True,shooting_usable=True,target_time_code=1.6,evaluation_method="exact target snapshot",
            target_snapshot_sources=[dict(file=str(snap),time_code=1.6,weight=1.)])
        p["csv"].with_suffix(".provenance.json").write_text(json.dumps(provenance))
        p["runtime_param"].write_text(f"InitCondFile {p['ic'].with_suffix('')}\nICFormat 3\nSnapFormat 3\nOutputListOn 0\nNumFilesPerSnapshot 1\nNumFilesWrittenInParallel 1\nInitGasTemp 0\nTimeOfFirstSnapshot 0.6\nTimeBetSnapshot 0.1\nOutputDir {p['output']}\nSnapshotFileBase snapshot\nRestartFile restart\nTimeBegin 0.6\nTimeMax 1.6\nMaxSizeTimestep 0.001\n")
        for mode in ("smoke","production"):
            script=p["run"]/mode/"run.sh";script.parent.mkdir(exist_ok=True);script.write_text("#!/bin/bash\nexit 0\n")
        return p

    def test_read_actual_files_and_signed_response(self):
        before={p:ev.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        for j,(trial,index,h) in enumerate(cli.PROBES):
            e=ev.read_evidence(self.root,trial)
            np.testing.assert_allclose(ev.validate_probe(self.a,e,index,h),self.J[:,j],atol=1e-12)
        self.assertEqual(before,{p:ev.sha256(p) for p in before})

    def test_propose_import_review_real_evidence(self):
        cli.propose(SimpleNamespace(state=self.state,output=self.proposal))
        proposal=json.loads(self.proposal.read_text())
        self.assertTrue(proposal["solver_report"]["ready_for_live_validation"])
        step,_=imp.load_local_live_step(self.proposal,self.q,self.r,np.ones(12),
            self.a["ic_path"],self.a["ic_sha256"],self.a["residual_csv_path"],self.a["residual_csv_sha256"],"synthetic")
        pred=np.array(proposal["predicted_residual_12d"])
        self.write_trial("trial14",self.q+step,pred,self.a,step,method="local-live-joint")
        out=self.root/"review.json"
        cli.review(SimpleNamespace(state=self.state,proposal=self.proposal,output=out))
        result=json.loads(out.read_text())
        self.assertTrue(result["orbit_componentwise_3sigma_pass"])
        self.assertAlmostEqual(result["actual_to_predicted_max_reduction_ratio"],1.)
        with self.assertRaisesRegex(ValueError,"覆盖"):
            cli.propose(SimpleNamespace(state=self.state,output=self.proposal))

    def test_wrong_target_frame_rejected(self):
        p=ev.paths(self.root,"trial12")
        with h5py.File(p["output"]/"snapshot_011.hdf5","r+") as f:f["Header"].attrs["Time"]=1.5
        with self.assertRaisesRegex(ValueError,"Header/Time"):
            ev.read_evidence(self.root,"trial12")

    def test_runtime_physics_drift_rejected(self):
        p=ev.paths(self.root,"trial12")
        with p["runtime_param"].open("a") as f:f.write("BlackHoleAccretionFactor 1\n")
        e=ev.read_evidence(self.root,"trial12")
        with self.assertRaisesRegex(ValueError,"runtime_param_normalized"):
            ev.validate_probe(self.a,e,1,.5)

    def test_csv_and_orbit_mutations_rejected(self):
        p=ev.paths(self.root,"trial12")
        with p["csv"].open("a") as f:f.write("\n")
        with self.assertRaisesRegex(ValueError,"CSV 指纹"):
            ev.read_evidence(self.root,"trial12")
        p=ev.paths(self.root,"trial13");orbit=json.loads(p["orbit"].read_text());orbit["applied_initial_12d"][0]+=.1;p["orbit"].write_text(json.dumps(orbit))
        with self.assertRaisesRegex(ValueError,"AuditJSON"):
            ev.read_evidence(self.root,"trial13")

    def test_branch_from_wrong_parent_rejected(self):
        e=ev.read_evidence(self.root,"trial12")
        e["audit"]["shooting_base_ic"]="mw_lmc_smc_relaxed_trial11_000.hdf5"
        with self.assertRaisesRegex(ValueError,"同一个基点"):
            ev.validate_probe(self.a,e,1,.5)

    def test_wrong_runtime_input_output_paths_rejected(self):
        for key in ("InitCondFile","OutputDir"):
            p=ev.paths(self.root,"trial12");original=p["runtime_param"].read_text()
            bad="\n".join(key+" /tmp/another_trial" if line.startswith(key+" ") else line for line in original.splitlines())+"\n"
            p["runtime_param"].write_text(bad)
            with self.assertRaisesRegex(ValueError,key):ev.read_evidence(self.root,"trial12")
            p["runtime_param"].write_text(original)

    def make_run_files(self):
        from satellite_relax_common import run_script
        p=ev.paths(self.root,"trial12")
        production=p["runtime_param"].read_text()
        for mode in ("smoke","production"):
            folder=p["run"]/mode;(folder/"output").mkdir(exist_ok=True)
            if mode=="smoke":
                (folder/"stage2.param").write_text(production.replace(str(p["output"]),str(folder/"output")))
            run_script(folder/"run.sh",self.a["audit"]["gizmo"],folder/"stage2.param",self.root,2,folder/"gizmo_run.log",6)
        return p

    def test_flag0_script_edit_rejected(self):
        p=self.make_run_files();cli.verify_run_files(self.s,p)
        script=p["run"]/"production/run.sh"
        script.write_text(script.read_text().replace(" 2 2>&1"," 0 2>&1"))
        with self.assertRaisesRegex(ValueError,"已改变"):
            cli.verify_run_files(self.s,p)

    def test_failed_smoke_verify_prevents_production(self):
        p=self.make_run_files()
        shutil.rmtree(p["output"]);p["output"].mkdir()
        calls=[]
        def fake_run(command,env=None):
            calls.append(list(map(str,command)))
            if len(calls)==1:
                with h5py.File(p["run"]/"smoke/output/snapshot_000.hdf5","w") as f:
                    f.create_group("Header").attrs["Time"]=.60000001
            else:
                self.assertIn("verify",calls[-1])
                raise ValueError("synthetic smoke field failure")
        with patch.object(cli,"run_cmd",side_effect=fake_run):
            with self.assertRaisesRegex(ValueError,"smoke field failure"):
                cli.run_trial(SimpleNamespace(state=self.state,trial="trial12",proposal="unused"))
        self.assertEqual(len(calls),2)

    def test_nonfinite_param_rejected(self):
        p=self.root/"invalid.param";p.write_text("TimeMax nan\n")
        with self.assertRaisesRegex(ValueError,"非有限"):
            ev.normalized_params(p)

    def remove_probes_and_state(self):
        for trial,_,_ in cli.PROBES:
            p=ev.paths(self.root,trial)
            p["ic"].unlink();p["orbit"].unlink();shutil.rmtree(p["run"])
        self.state.unlink()

    def prepare_args(self,resume=False):
        return SimpleNamespace(work_dir=self.root,base_param=self.param,mpi_tasks=6,
                               state=self.state,resume=resume)

    def fake_merge(self,command,env=None):
        cmd=list(map(str,command))
        trial=cmd[cmd.index("--trial-label")+1]
        coord=cmd[cmd.index("--probe-coordinate")+1]
        h=float(cmd[cmd.index("--probe-step")+1])
        self.assertEqual(cmd[cmd.index("--shooting-base-ic")+1],self.a["ic_path"])
        index=ev.COORDS.index(coord);step=np.zeros(12);step[index]=h
        review=None
        if "--reviewed-build" in cmd:
            from local_live_build_review import load_review
            review=load_review(cmd[cmd.index("--reviewed-build")+1],self.a)
        self.write_trial(trial,self.q+step,self.r+self.J[:,[0,1,7].index(index)]*h,
                         self.a,step,base_param=cmd[cmd.index("--base-param")+1],
                         gizmo_path=cmd[cmd.index("--gizmo-executable")+1],build_review=review)

    def setup_reviewed_rebuild(self):
        import local_live_build_review as br
        self.remove_probes_and_state()
        self.binary.write_bytes(b"synthetic rebuilt binary, never execute")
        data=copy.deepcopy(br.REVIEW_DATA)
        data.update(anchor_ic_sha256=self.a["ic_sha256"],
                    anchor_binary_sha256=self.a["gizmo_sha256"],
                    execution_binary_sha256=ev.sha256(self.binary))
        for name,value in (("ANCHOR_IC_SHA256",self.a["ic_sha256"]),
                           ("ANCHOR_BINARY_SHA256",self.a["gizmo_sha256"]),
                           ("EXECUTION_BINARY_SHA256",ev.sha256(self.binary)),
                           ("REVIEW_DATA",data)):
            p=patch.object(br,name,value);p.start();self.addCleanup(p.stop)
        review_path=self.root/"reviewed.json"
        review_path.write_text(json.dumps(data))
        args=self.prepare_args();args.reviewed_build=review_path
        return args

    def test_reviewed_rebuild_prepare_propose_import_and_review(self):
        args=self.setup_reviewed_rebuild()
        old_ic_hash=ev.sha256(self.a["ic_path"])
        with self.assertRaisesRegex(ValueError,"GIZMO"):
            cli.prepare(self.prepare_args())
        with patch.object(cli,"run_cmd",side_effect=self.fake_merge) as run:
            cli.prepare(args)
        self.assertEqual(run.call_count,3)
        state,anchor=cli.read_state(self.state)
        self.assertEqual(anchor["gizmo_sha256"],self.a["gizmo_sha256"])
        self.assertNotEqual(state["execution_gizmo"]["sha256"],anchor["gizmo_sha256"])
        self.assertEqual(state["execution_gizmo"]["sha256"],ev.sha256(self.binary))
        self.assertIn("--reviewed-build",cli.merge_command(state,"trial14","local-live-joint"))
        cli.propose(SimpleNamespace(state=self.state,output=self.proposal))
        proposal=json.loads(self.proposal.read_text())
        self.assertEqual(proposal["build_review"],state["build_review"])
        step,diag=imp.load_local_live_step(self.proposal,self.q,self.r,np.ones(12),
            self.a["ic_path"],self.a["ic_sha256"],self.a["residual_csv_path"],self.a["residual_csv_sha256"],"synthetic")
        self.assertEqual(diag["execution_gizmo"],state["execution_gizmo"])
        self.assertEqual(diag["build_review"],state["build_review"])
        pred=np.asarray(proposal["predicted_residual_12d"])
        self.write_trial("trial14",self.q+step,pred,self.a,step,method="local-live-joint",build_review=state["build_review"])
        out=self.root/"review.json"
        cli.review(SimpleNamespace(state=self.state,proposal=self.proposal,output=out))
        self.assertTrue(json.loads(out.read_text())["orbit_componentwise_3sigma_pass"])
        self.assertEqual(ev.sha256(self.a["ic_path"]),old_ic_hash)

    def test_reviewed_rebuild_is_frozen_after_prepare(self):
        args=self.setup_reviewed_rebuild()
        with patch.object(cli,"run_cmd",side_effect=self.fake_merge):cli.prepare(args)
        data=json.loads(self.state.read_text());data.pop("build_review")
        self.state.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"GIZMO"):cli.read_state(self.state)

    def test_reviewed_rebuild_changed_again_is_rejected(self):
        args=self.setup_reviewed_rebuild()
        with patch.object(cli,"run_cmd",side_effect=self.fake_merge):cli.prepare(args)
        self.binary.write_bytes(b"another unreviewed rebuild")
        with self.assertRaisesRegex(ValueError,"GIZMO"):cli.read_state(self.state)

    def test_prepare_starts_with_only_t10_and_generates_all_three(self):
        self.remove_probes_and_state()
        before={p:ev.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(cli,"run_cmd",side_effect=self.fake_merge) as run:
            cli.prepare(self.prepare_args())
        self.assertEqual(run.call_count,3)
        state=json.loads(self.state.read_text())
        self.assertFalse(state["history_dependency"])
        self.assertNotIn("history_json_path",state)
        self.assertNotIn("x_probe_evidence",state)
        for trial,index,h in cli.PROBES:
            ev.validate_probe(self.a,ev.read_evidence(self.root,trial),index,h)
        self.assertEqual(before,{p:ev.sha256(p) for p in before})
        with patch.object(cli,"run_cmd") as run:
            cli.prepare(self.prepare_args(resume=True))
        run.assert_not_called()

    def test_prepare_preflights_all_destinations_before_writing(self):
        self.remove_probes_and_state()
        p=ev.paths(self.root,"trial13");p["ic"].write_bytes(b"keep")
        with patch.object(cli,"run_cmd") as run:
            with self.assertRaisesRegex(ValueError,"覆盖"):
                cli.prepare(self.prepare_args())
        run.assert_not_called()
        self.assertFalse(self.state.exists())
        self.assertFalse(ev.paths(self.root,"trial11")["ic"].exists())
        self.assertEqual(p["ic"].read_bytes(),b"keep")

    def test_legacy_state_rejected(self):
        old=dict(self.s,schema="local-live-round-state-v1")
        self.state.write_text(json.dumps(old))
        with self.assertRaises(ValueError):cli.read_state(self.state)

    def test_probe_definition_in_state_cannot_change(self):
        changed=copy.deepcopy(self.s);changed["probes"][0]["step"]=1.0
        self.state.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):cli.read_state(self.state)

    def test_incomplete_validation_is_not_overwritten(self):
        p=ev.paths(self.root,"trial10")
        p["csv"].with_suffix(".provenance.json").unlink()
        old=p["csv"].read_bytes()
        with patch.object(cli,"run_cmd") as run:
            with self.assertRaises(ValueError):cli.ensure_validation(self.root,"trial10")
        run.assert_not_called()
        self.assertEqual(p["csv"].read_bytes(),old)

    def test_recovered_production_param_runs_whole_calibration_chain(self):
        self.remove_probes_and_state()
        self.param.write_text("MaxSizeTimestep 0.0001\nSofteningGas 0.002\n")
        current_contents=self.param.read_bytes()
        output=self.root/"shooting_t10_runtime.param"
        cli.recover_param(SimpleNamespace(work_dir=self.root,output=output))
        self.assertEqual(self.param.read_bytes(),current_contents)
        self.assertNotEqual(ev.sha256(output),self.a["base_param_sha256"])
        args=self.prepare_args();args.base_param=output
        with patch.object(cli,"run_cmd",side_effect=self.fake_merge):cli.prepare(args)
        cli.propose(SimpleNamespace(state=self.state,output=self.proposal))
        proposal=json.loads(self.proposal.read_text())
        step,_=imp.load_local_live_step(self.proposal,self.q,self.r,np.ones(12),
            self.a["ic_path"],self.a["ic_sha256"],self.a["residual_csv_path"],self.a["residual_csv_sha256"],"synthetic")
        self.write_trial("trial14",self.q+step,np.array(proposal["predicted_residual_12d"]),
                         self.a,step,method="local-live-joint",base_param=output)
        review=self.root/"recovered-review.json"
        cli.review(SimpleNamespace(state=self.state,proposal=self.proposal,output=review))
        self.assertTrue(json.loads(review.read_text())["orbit_componentwise_3sigma_pass"])

    def test_effective_prediction_matches_real_parameter_generator(self):
        from local_live_params import assess_base_param
        from merge_relaxed_mw_lmc_smc_local_live_v1 import create_stage2_files
        from satellite_relax_common import read_param
        base,lines,values=read_param(self.param)
        run=create_stage2_files(self.root/"parity_000.hdf5",base,lines,values,
                               self.a["audit"]["gizmo"],.6,1.,{"time_gyr":1.},6)
        generated=ev.normalized_params(run/"production/stage2.param")
        self.assertEqual(generated,assess_base_param(self.a,self.param)["effective_normalized"])

    def test_frozen_execution_param_mutation_rejected_after_prepare(self):
        with self.param.open("a") as f:f.write("# later change\n")
        with self.assertRaisesRegex(ValueError,"prepare 后已改变"):
            cli.read_state(self.state)

    def test_identical_binary_backup_replaces_path_without_touching_old_audit(self):
        self.remove_probes_and_state()
        backup=self.root/"GIZMO_t10_backup";shutil.copy2(self.binary,backup)
        self.binary.write_bytes(b"new build for a different experiment")
        original=Path(self.a["ic_path"]);old_digest=ev.sha256(original)
        args=self.prepare_args();args.gizmo_executable=backup
        with patch.object(cli,"run_cmd",side_effect=self.fake_merge):cli.prepare(args)
        state=json.loads(self.state.read_text())
        self.assertEqual(state["execution_gizmo"]["path"],str(backup))
        cli.read_state(self.state)
        from satellite_relax_common import run_script
        p=ev.paths(self.root,"trial12")
        for mode in ("smoke","production"):
            folder=p["run"]/mode
            if mode=="smoke":
                (folder/"stage2.param").write_text(p["runtime_param"].read_text().replace(str(p["output"]),str(folder/"output")))
            run_script(folder/"run.sh",state["execution_gizmo"],folder/"stage2.param",self.root,2,folder/"gizmo_run.log",6)
        cli.verify_run_files(state,p)
        cli.propose(SimpleNamespace(state=self.state,output=self.proposal))
        proposal=json.loads(self.proposal.read_text())
        step,_=imp.load_local_live_step(self.proposal,self.q,self.r,np.ones(12),
            self.a["ic_path"],self.a["ic_sha256"],self.a["residual_csv_path"],self.a["residual_csv_sha256"],"synthetic")
        self.write_trial("trial14",self.q+step,np.array(proposal["predicted_residual_12d"]),
                         self.a,step,method="local-live-joint",gizmo_path=backup)
        review=self.root/"binary-review.json"
        cli.review(SimpleNamespace(state=self.state,proposal=self.proposal,output=review))
        self.assertTrue(json.loads(review.read_text())["orbit_componentwise_3sigma_pass"])
        self.assertEqual(ev.sha256(original),old_digest)
        self.assertEqual(self.binary.read_bytes(),b"new build for a different experiment")

    def test_new_build_cannot_be_used_for_probe_without_new_baseline(self):
        self.remove_probes_and_state()
        other=self.root/"GIZMO_new";other.write_bytes(b"different build");other.chmod(0o755)
        args=self.prepare_args();args.gizmo_executable=other
        with patch.object(cli,"run_cmd") as run:
            with self.assertRaises(ValueError):cli.prepare(args)
        run.assert_not_called();self.assertFalse(self.state.exists())


if __name__=="__main__":unittest.main()
