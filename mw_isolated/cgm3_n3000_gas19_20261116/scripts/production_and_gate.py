"""Start the authorized production once; immediately record the true 12D gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_trial import utc, read_json, write_status
from stage_safety import ROOT, new_json, require

def stage(name,trial,extra=None):
    d=read_json(ROOT/"campaign.json")
    d.update(stage=name,trial=trial.name,active_run=str(trial),updated_utc=utc(),status_path=str(trial/"production/status.json"))
    d.update(extra or {})
    write_status(ROOT/"campaign.json",d)

def main(trial):
    require(os.isatty(0) and os.isatty(1) and trial.parent==ROOT,"attached campaign terminal required")
    require(not (ROOT/"accepted_orbit.json").exists(),"an orbit already passed")
    require(not (trial/"production/status.json").exists(),"previous production attempt; inspect instead of resubmitting")
    require(read_json(trial/"production_ready.json")["status"]=="pass","direct production input readiness is missing")
    clearance=read_json(trial/"production_launch_clearance.json")
    from run_trial import sha256
    require(clearance["status"]=="pass" and clearance["ready_sha"]==sha256(trial/"production_ready.json") and clearance["ic_sha256"]==read_json(trial/"production_ready.json")["ic_sha256"],"production clearance missing or stale")
    new_json(trial/"production_followthrough.lock",{"started":utc(),"pid":os.getpid(),"tty":os.ttyname(0)})
    stage("production_starting",trial)
    rc=subprocess.run(["bash",str(trial/"start_production.sh")],cwd=trial).returncode
    if rc:
        stage("production_requires_review",trial,{"production_wrapper_exit":rc})
        return rc
    stage("endpoint_gate_running",trial)
    # The verified runner has already generated the canonical dashboard with this exact IC.
    cmd=[sys.executable,"-B","-u",str(ROOT/"scripts/gate_endpoint.py"),"--run-dir",str(trial)]
    with (trial/"endpoint_gate_terminal.log").open("x") as log:
        gate=subprocess.Popen(cmd,cwd=trial,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in gate.stdout:
            print(line,end="",flush=True);log.write(line);log.flush()
        rc=gate.wait()
    if rc not in (0,3):
        stage("endpoint_evidence_requires_review",trial,{"gate_exit":rc})
        return rc
    path=trial/"endpoint_review/gate_result.json"
    report=read_json(path)
    require(report["total_components"]==12 and (rc==0)==(report["passed_component_count"]==12),"gate/status mismatch")
    result={"trial":trial.name,"checked_utc":utc(),"gate":str(path),"D12":report["D12"],"max_abs_sigma":report["max_abs_sigma"],"passed_component_count":report["passed_component_count"],"initial_ic_sha256":report["initial_ic_sha256"],"endpoint":report["endpoint_snapshot"],"residual_csv":report["residual_csv"],"dashboard":str(trial/"orbit_validation/magellanic_6d_dashboard.png"),"normalized_residual_12d":report["normalized_residual_12d"]}
    if rc==0:
        new_json(ROOT/"accepted_orbit.json",result)
        stage("orbit_passed_awaiting_user_review",trial,{"accepted_trial":str(trial),"active_run":None,"latest_gate":str(path),"next_stage":"Present evidence and remove active follow-up. Wait for user review. No AGN or jet."})
    else:
        extra={"latest_gate":str(path),"most_recent_measured":str(trial),"next_stage":"Review actual residuals against the measured parent and best new-family result; design one independent new correction. Do not use old response matrices."}
        if trial.name=="calibration01":
            extra.update(best_actual_run=str(trial),best_D12=report["D12"],best_max_abs_sigma=report["max_abs_sigma"])
        stage("measured_orbit_needs_correction",trial,extra)
    new_json(trial/"orbit_outcome.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return rc

if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("--trial-dir",required=True,type=Path)
    sys.exit(main(arg.parse_args().trial_dir.resolve()))
