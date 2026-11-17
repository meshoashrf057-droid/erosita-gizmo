"""Independent pre-production review of the actual completed calibration smoke receipt."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from stage_safety import ROOT, require, new_json
from run_trial import sha256, utc, read_json
from make_trial_runtime import parse_params

def audit(trial):
    require(trial.parent == ROOT,"trial outside this orbit campaign")
    receipt = read_json(trial/"production_ready.json")
    endpoint = read_json(trial/"smoke_endpoint_review.json")
    status = read_json(trial/"smoke/status.json")
    require(receipt["status"] == endpoint["status"] == "pass" and status["simulation_returncode"] == 0, "smoke did not pass")
    require(sha256(trial/"orbit_000.hdf5")==receipt["ic_sha256"]==endpoint["ic_sha256"],"wrong merged IC")
    log=(trial/"smoke/gizmo_run.log").read_text(errors="replace")
    rows=re.findall(r"Largest Allocation = ([.0-9]+) Mbyte \(on task=([0-9]+)\)",log)
    require(rows,"allocator report missing")
    peak,rank=max((float(v),int(index)) for v,index in rows)
    require(peak<float(parse_params((trial/"production/stage2.param").read_text())["MaxMemSize"]),"GIZMO internal allocation too large")
    heads=[row for row in log.splitlines() if "#000:" in row]
    duplicate=sum("H5Dcreate1(): unable to create dataset" in row for row in heads)
    require(duplicate==log.count("name already exists"),"unexplained HDF5 create error")
    require(all("H5Dcreate1(): unable to create dataset" in row or "H5Dwrite(): not a dataset" in row or "H5Dclose(): not a dataset" in row for row in heads),"unreviewed HDF5 error")
    cfg=read_json(trial/"run_configuration.json")
    snapshot_check=read_json(trial/"smoke_initial_output_check.json")
    require(snapshot_check["status"]=="pass" and 0 < snapshot_check["snapshot_time"]-cfg["production_start_code"] < 1e-7,"bad first frame")
    data={"status":"pass","checked_utc":utc(),"ic_sha256":receipt["ic_sha256"],"ready_sha":sha256(trial/"production_ready.json"),"smoke_status_sha":sha256(trial/"smoke/status.json"),"initial":snapshot_check,"end":endpoint,"max_recorded_internal_allocation_mib":peak,"rank_at_peak":rank,"duplicate_cooling_rate_create_stacks":duplicate,"physical_equivalence":"Production inherits n3000 host CritPhysDensity3000, MaxKernelRadius1e56, MinSizeTimestep1e-12; remaining physical parameters match original CPP orbit. Smoke tests initialization, resources and short evolution with MaxSizeTimestep1e-6 versus production0.001; both use MinSizeTimestep1e-12.","production_param_sha":cfg["param_sha256"]["production"]}
    new_json(trial/"production_launch_clearance.json",data)
    print(f"smoke verified; {peak} MiB peak internal allocator on rank {rank}; {endpoint['readable_fields']} endpoint fields; final time {endpoint['time']}",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--trial-dir",required=True,type=Path)
    audit(ap.parse_args().trial_dir.resolve())
