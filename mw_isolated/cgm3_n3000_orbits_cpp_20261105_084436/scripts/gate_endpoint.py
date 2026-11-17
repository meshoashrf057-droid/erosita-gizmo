#!/usr/bin/env python3
"""Verify one CGM3 orbit at its actual endpoint. Exit 0 pass, 2 pending, 3 12D fail, 4 invalid evidence."""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MW = ROOT.parent
PYTHON = MW.parent / ".venv/bin/python"
VALIDATOR = MW / "validation_dashboard_sat_6d_gaswake_v3.py"
SEMANTICS = MW / "shooting_cpp_20261031_082656/check_orbit_gate.py"
VALIDATOR_SHA = "28a98850094c1bf97ba06af13ca55fa76b059b0b3e1728520de8795f0b7fc799"
SEMANTICS_SHA = "755b300704b453d91690d4fef11d4c9e1a617c4e82781a05378b17271485427a"
BINARY_SHA = "07154d9677d2e897a7dc557e37f9565d0be835a5e612943870cc5021b137f343"
HOST_SHA = "1f38047d72b5c6bdfd8a324489f0988c8fbb8250b4415a47e7ed902c94fb89ac"
SATELLITE_SHA = ["7cfbbd4c284d6259806f80143d9ab8ead3db58bf11740a699032a98670c52a8f",
                 "91db5535acc7c08399314414fb94742c4df22d3383d5aa7d261e97cad066b916"]
TARGET = np.float64(1.6363392421373844)
START = 0.6136272158015191
OBS = {"lmc": [-.8,-41.5,-26.9,-57.,-226.,221.], "smc":[15.3,-36.9,-43.3,19.,-153.,153.]}
SIGMA = {"lmc":[.39645665,1.89267257,1.28905242,13.,15.,19.], "smc":[1.0799982,1.75861938,2.0393135,18.,21.,17.]}
TRACERS = {"lmc":5000,"smc":1600}
NAMES = ["magellanic_present_6d.csv", "magellanic_6d_history.csv", "magellanic_present_6d.provenance.json",
         "magellanic_validation_summary.json", "magellanic_6d_dashboard.png"]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def stamp(path):
    stat = path.stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")

def same_time(actual, wanted):
    return math.isclose(float(actual),wanted,abs_tol=1e-10,rel_tol=0)

def semantic_functions():
    require(sha(SEMANTICS)==SEMANTICS_SHA,"Pinned semantic checker changed")
    spec = importlib.util.spec_from_file_location("fixed_orbit_gate",SEMANTICS)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module

def command(trial, destination):
    return [str(PYTHON),str(VALIDATOR),"--snapshot-dir",str(trial/"production/output"),
            "--reference-ic",str(trial/"orbit_000.hdf5"),"--output-dir",str(destination)]

def bind_run(trial, status):
    campaign = read(ROOT/"campaign.json")
    ic = trial/"orbit_000.hdf5"
    sidecar = ic.with_suffix(".orbit.json")
    audit = read(sidecar)
    cfg = read(trial/"run_configuration.json")
    require(campaign["host_sha256"]==HOST_SHA and campaign["jet_authorized"] is False,
            "Unexpected campaign host or scope")
    expected_sources = [HOST_SHA, *SATELLITE_SHA]
    require(audit["source_sha256"]==expected_sources,"IC sidecar source family differs")
    require(Path(audit["source_files"][0]).resolve()==Path(campaign["host_endpoint"]).resolve(),"Wrong host source")
    require(same_time(campaign["target_time_code"],TARGET) and same_time(campaign["start_time_code"],START),
            "Campaign epoch differs")
    require(same_time(audit["stage2_target_code"],TARGET),"Reference audit epoch differs")
    ic_sha = sha(ic)
    require(ic_sha==audit["output_ic_sha256"]==cfg["ic_sha256"]==status["ic_sha256"],"IC hash mismatch")
    require(sha(sidecar)==cfg["sidecar_sha256"]==status["sidecar_sha256"],"Sidecar hash mismatch")
    param = trial/"production/stage2.param"
    require(sha(param)==cfg["param_sha256"]["production"]==status["param_sha256"],"Production parameter hash mismatch")
    params = {}
    for line in param.read_text().splitlines():
        fields=line.split("%",1)[0].split()
        if len(fields)==2:
            params[fields[0]]=fields[1]
    param_ic = (param.parent/params["InitCondFile"]).resolve()
    if param_ic.suffix!=".hdf5":
        param_ic=Path(str(param_ic)+".hdf5")
    require(param_ic==ic and (param.parent/params["OutputDir"]).resolve()==trial/"production/output",
            "Runtime input/output differs from this trial")
    require(same_time(params["TimeBegin"],START) and same_time(params["TimeMax"],TARGET),"Parameter epoch differs")
    require(float(params["CritPhysDensity"])==3000.0 and float(params["MaxKernelRadius"])==1e56
            and float(params["MinSizeTimestep"])==1e-12,"Chosen n3000 host physical/numeric settings differ")
    require(math.isclose((TARGET-START)*(float(params["UnitLength_in_cm"])/float(params["UnitVelocity_in_cm_per_s"])/(31557600.0*1e9)),1.0,rel_tol=0,abs_tol=1e-14),
            "Infall is not exactly one physical Gyr")
    require(status["binary_sha256"]==cfg["binary"]["sha256"]==BINARY_SHA,"Unpinned binary")
    require(Path(status["cwd"]).resolve()==trial/"production","Wrong production cwd")
    require(status["command"][-3:]==[cfg["binary"]["path"],"stage2.param","2"],"Wrong GIZMO invocation")
    with h5py.File(ic,"r") as h:
        require(same_time(h["Header"].attrs["Time"],START),"IC start mismatch")
        for system,source_hash in zip(("MW","LMC","SMC"),expected_sources):
            require(h["MergeProvenance/"+system].attrs["SOURCE_SHA256"]==source_hash,"Embedded source mismatch: "+system)
        orbit=h["MagellanicOrbitReference"]
        model=h["MagellanicModelReference"].attrs
        require(model["SATELLITE_MODEL_KEY"]==audit["model_key"] and model["TRIAL_LABEL"]==audit["trial_label"],
                "Embedded model or trial label mismatch")
        require(same_time(orbit.attrs["PresentTargetTime_Code"],TARGET),"Embedded target epoch differs")
        reference={"target_time_code":TARGET,"unit_time_gyr":float(orbit.attrs["UnitTime_Gyr"])}
        for sat in ("lmc","smc"):
            cap=sat.upper()
            obs=np.r_[orbit[cap+"_Position_kpc"][-1],orbit[cap+"_Velocity_kms"][-1]]
            sigma=np.r_[orbit[cap+"_Current_PositionSigma_kpc"][:],orbit[cap+"_Current_VelocitySigma_kms"][:]]
            require(np.allclose(obs,OBS[sat],rtol=0,atol=1e-12) and np.allclose(sigma,SIGMA[sat],rtol=0,atol=1e-12),
                    "Changed observational target/sigma: "+sat)
            reference[sat]={"observed":obs.tolist(),"sigma":sigma.tolist()}
    return ic_sha,audit,reference

def snapshot_evidence(trial,status):
    output=trial/"production/output"
    snapshots=[]
    for path in sorted(output.glob("snapshot_*.hdf5")):
        require(re.fullmatch(r"snapshot_[0-9]+\.hdf5",path.name) and path.resolve().parent==output,"Unexpected snapshot path")
        with h5py.File(path,"r") as f:
            header=f["Header"].attrs
            require(int(header.get("NumFilesPerSnapshot",1))==1,"Expected complete single-file snapshots")
            t=float(header["Time"])
            require(np.isfinite(t) and t>=START-1e-10,"Unexpected snapshot time")
        snapshots.append((t,path,stamp(path)))
    require(snapshots,"No snapshots")
    snapshots.sort()
    require(all(a[0]<b[0] for a,b in zip(snapshots,snapshots[1:])),"Duplicate output time")
    final_time,final,_=snapshots[-1]
    require(same_time(final_time,TARGET) and same_time(status["target_time"],TARGET)
            and same_time(status["final_snapshot_time"],final_time) and Path(status["final_snapshot"]).resolve()==final,
            "Latest actual snapshot is not the bound target")
    return snapshots

def verify_products(trial,folder,reference,ic_sha,audit,snapshots,semantic):
    paths={n:folder/n for n in NAMES}
    require(all(p.is_file() and p.resolve().parent==folder for p in paths.values()),"Missing/redirected validator products")
    with Image.open(paths[NAMES[4]]) as img:
        img.load()
    summary=read(paths[NAMES[3]])
    provenance=read(paths[NAMES[2]])
    require(summary["failed_snapshot_count"]==0 and summary["snapshot_count"]==len(snapshots),"Incomplete validation")
    ic=trial/"orbit_000.hdf5"
    require(Path(summary["reference_ic"]).resolve()==ic and Path(provenance["reference_ic"]).resolve()==ic and
            provenance["reference_ic_sha256"]==ic_sha,"Validator reference differs")
    require(Path(summary["shooting_provenance"]).resolve()==paths[NAMES[2]] and
            Path(provenance["residual_csv"]).resolve()==paths[NAMES[0]] and sha(paths[NAMES[0]])==provenance["residual_csv_sha256"],
            "CSV/provenance binding differs")
    require(provenance["model_key"]==audit["model_key"] and provenance["trial_label"]==audit["trial_label"] and
            provenance["reference_tracer_counts"]==TRACERS,"Unexpected model/trial or reference stellar population")
    sources=provenance["target_snapshot_sources"]
    t,final,final_stamp=snapshots[-1]
    require(len(sources)==1 and Path(sources[0]["file"]).resolve()==final and same_time(sources[0]["time_code"],t)
            and sources[0]["weight"]==1,"Target does not use the single true endpoint")
    require(max(final_stamp[3:])<=min(p.stat().st_mtime_ns for p in paths.values()),"Endpoint changed after measurement")
    with paths[NAMES[1]].open(newline="") as stream:
        history=list(csv.DictReader(stream))
    require(len(history)==len(snapshots),"Incomplete history CSV")
    for row,(time,path,_) in zip(history,snapshots):
        require(Path(row["file"]).resolve()==path and same_time(row["time_code"],time)
                and row["stellar_tracer_policy"]=="original-stars","Wrong history input/identity")
    require(history[-1]["host_center_method"]=="MW BH position + inner-star velocity","Endpoint host-frame fallback")
    decision=semantic.validate_12d(summary,provenance,reference)
    semantic.validate_csv(paths[NAMES[0]],decision)
    z=decision["normalized_residual_12d"]
    d12=math.sqrt(sum(x*x for x in z))
    require(math.isclose(d12,summary["present_epoch"]["diagonal_scaled_residual_l2"],rel_tol=1e-10),"D12 mismatch")
    return paths,decision,d12

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--fresh-validation",action="store_true",help="Regenerate independent products in the new evidence directory")
    args=parser.parse_args()
    trial=args.run_dir.resolve(strict=True)
    require(trial.parent==ROOT and trial.is_dir(),"Trial must be a direct child of this CGM3 campaign")
    status_path=trial/"production/status.json"
    if not status_path.is_file():
        print("waiting: production has no status")
        return 2
    st=read(status_path)
    if st.get("state") in ("starting","running","validating","exited"):
        print("waiting:",st["state"])
        return 2
    require(st.get("simulation_returncode")==0 and st.get("state") in ("completed","simulation_completed"),
            "Production is not a successful target run")
    if st["state"]=="completed":
        require(st.get("returncode")==0 and st.get("wrapper_returncode",0)==0 and
                st.get("orbit_validation_returncode",0)==0,"Runtime/validation failure")
    ic_sha,audit,reference=bind_run(trial,st)
    snapshots=snapshot_evidence(trial,st)
    require(sha(VALIDATOR)==VALIDATOR_SHA,"Pinned validator changed")
    semantic=semantic_functions()
    evidence=trial/"endpoint_review"
    folder=trial/"orbit_validation"
    generate=args.fresh_validation or not folder.exists()
    if generate:
        require(os.isatty(0) and os.isatty(1),"Reconstruct only in a visible terminal")
    else:
        require(st.get("orbit_validation_returncode")==0 and st.get("validation_command")==command(trial,folder),
                "Existing canonical validator invocation is unverified; use --fresh-validation")
    evidence.mkdir(exist_ok=False)
    try:
        if generate:
            folder=evidence/"validation" if args.fresh_validation else folder
            folder.mkdir(exist_ok=False)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(evidence/"mpl"))
            with (evidence/"validator.log").open("x") as log:
                process=subprocess.Popen(command(trial,folder),cwd=trial,env=env,stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,text=True,errors="replace")
                for line in process.stdout:
                    print(line,end="",flush=True)
                    log.write(line)
                require(process.wait()==0,"Validator failed; inspect validator.log")
        paths,decision,d12=verify_products(trial,folder,reference,ic_sha,audit,snapshots,semantic)
        z=decision["normalized_residual_12d"]
        t,end,_=snapshots[-1]
        endpoint_hash=sha(end)
        require(all(stamp(p)==initial for _,p,initial in snapshots) and sha(trial/"orbit_000.hdf5")==ic_sha,
                "Input changed during endpoint review")
        report=dict(state="pass" if decision["orbit_pass"] else "orbit_failed",
                    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    run_directory=str(trial),family_input_sha256=audit["source_sha256"],model_key=audit["model_key"],
                    reference_ic=str(trial/"orbit_000.hdf5"),reference_ic_sha256=ic_sha,
                    endpoint_snapshot={"path":str(end),"sha256":endpoint_hash,"time_code":t},
                    actual_D12=d12,max_abs_sigma=decision["max_abs_sigma"],passed_components=sum(abs(v)<=3 for v in z),
                    passed_component_count=sum(abs(v)<=3 for v in z),total_components=12,D12=d12,
                    initial_ic_sha256=ic_sha,residual_csv=str(paths[NAMES[0]]),residual_csv_sha256=sha(paths[NAMES[0]]),
                    residual_12d=[value for sat in ("lmc","smc") for value in decision["values"][sat]["residual"]],
                    normalized_residual_12d=z,coordinate_order=decision["coordinate_order"],
                    failed_coordinates=decision["failed_coordinates"],threshold=3,component_values=decision["values"],
                    measurement_signature=semantic.SIGNATURE,validation_command=command(trial,folder),
                    product_hashes={str(p):sha(p) for p in paths.values()},validator_sha256=VALIDATOR_SHA,
                    semantic_functions_sha256=SEMANTICS_SHA,production_status_sha256=sha(status_path),
                    validated_frame_count=len(snapshots),after_pass="Stop for user review.")
        with (evidence/"endpoint_components.csv").open("xb") as stream:
            stream.write(paths[NAMES[0]].read_bytes())
        save(evidence/"gate_result.json",report)
        print(json.dumps({k:report[k] for k in ("state","actual_D12","max_abs_sigma","passed_components")}))
        return 0 if decision["orbit_pass"] else 3
    except Exception as error:
        save(evidence/"invalid_evidence.json",{"error":str(error)})
        raise

if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,KeyError,ValueError,TypeError,RuntimeError) as error:
        print(f"invalid evidence: {error}",file=sys.stderr)
        sys.exit(4)
