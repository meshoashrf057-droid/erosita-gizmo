#!/usr/bin/env python3
"""Verify a real flag=2 first output before authorizing its isolated full orbit."""
from pathlib import Path
import argparse, datetime, json, os, re, sys
import h5py
import numpy as np
PROJECT=Path("/home/xingdi/gizmo-runs")
sys.path.insert(0,str(PROJECT/"mw_isolated"))
from satellite_relax_common import sha256,particle_keys

def check(trial_dir):
    r=Path(trial_dir).resolve(strict=True)
    cfg=json.loads((r/"run_configuration.json").read_text())
    status=json.loads((r/"smoke/status.json").read_text())
    if status.get("state")!="smoke_completed" or status.get("simulation_returncode")!=0:
        raise ValueError("Smoke did not finish successfully")
    for k,wanted in (("ic_sha256",cfg["ic_sha256"]),("param_sha256",cfg["param_sha256"]["smoke"]),("binary_sha256",cfg["binary"]["sha256"])):
        if status.get(k)!=wanted:raise ValueError("Smoke evidence mismatch: "+k)
    if sha256(r/"orbit_000.hdf5")!=cfg["ic_sha256"]:raise ValueError("IC changed")
    if sha256(r/"production/stage2.param")!=cfg["param_sha256"]["production"]:raise ValueError("Production parameters changed")
    if sha256(r/"smoke/stage2.param")!=cfg["param_sha256"]["smoke"]:raise ValueError("Smoke parameters changed")
    if sha256(r/"orbit_000.orbit.json")!=cfg["sidecar_sha256"]:raise ValueError("IC audit changed")
    paths=[]
    for f in (r/"smoke/output").iterdir():
        if re.fullmatch(r"snapshot_[0-9]+\.hdf5",f.name):
            with h5py.File(f,"r") as h:paths.append((float(h["Header"].attrs["Time"]),f))
    if not paths:raise ValueError("No genuine smoke snapshots")
    from stage_safety import check_smoke_endpoint
    check_smoke_endpoint(r, paths)
    stamp,first=min(paths)
    report={"status":"pass","snapshot":str(first),"snapshot_time":stamp,"snapshot_sha256":sha256(first),"fields":{}}
    with h5py.File(r/"orbit_000.hdf5","r") as a,h5py.File(first,"r") as b:
        start=float(a["Header"].attrs["Time"])
        if not 0<stamp-start<1e-7:raise ValueError("Smoke first output is not the intended near-initial check")
        if not np.array_equal(a["Header"].attrs["NumPart_Total"],b["Header"].attrs["NumPart_Total"]):raise ValueError("Particle counts changed at initialization")
        for name in a:
            if not name.startswith("PartType"):continue
            ga,gb=a[name],b[name]
            def keys(g):return particle_keys({k:g[k][...] for k in ("ParticleIDs","ParticleChildIDsNumber","ParticleIDGenerationNumber") if k in g})
            ka,kb=keys(ga),keys(gb)
            ia,ib=np.argsort(ka),np.argsort(kb)
            if not np.array_equal(ka[ia],kb[ib]):raise ValueError("Particle composite identities changed: "+name)
            item={"count":len(ka),"composite_identity_preserved":True}
            for field in ("Masses","Metallicity","StellarFormationTime"):
                if field in ga:
                    if field not in gb or not np.array_equal(ga[field][...][ia],gb[field][...][ib]):
                        raise ValueError("Evolved initialization field changed: "+name+"/"+field)
                    item[field]="exactly_preserved"
            for field in ("Coordinates","Velocities"):
                if not np.isfinite(gb[field][...]).all():raise ValueError("Nonfinite phase space")
            report["fields"][name]=item
        for field in ("KernelMaxRadius","Density","InternalEnergy"):
            values=b["PartType0"][field][...]
            if not np.isfinite(values).all() or np.any(values<=0):raise ValueError("Invalid gas field: "+field)
        if any(name.startswith(("Sink_","BH_")) for name in b["PartType5"]):
            raise ValueError("Unexpected active sink state in an AGN-off orbit experiment")
    report_path=r/"smoke_initial_output_check.json"
    with report_path.open("x") as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
    ready={"status":"pass","checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ic_sha256":cfg["ic_sha256"],"sidecar_sha256":cfg["sidecar_sha256"],
        "production_param_sha256":cfg["param_sha256"]["production"],"smoke_param_sha256":cfg["param_sha256"]["smoke"],
        "binary_sha256":cfg["binary"]["sha256"],"smoke_returncode":0,"smoke_checks":str(report_path),
        "ic_verification":str(r/"ic_verification.json"),"preserve_evolved_metallicity_and_birth_times":True}
    verified=json.loads((r/"ic_verification.json").read_text())
    if verified.get("status")!="pass" or verified.get("output_ic_sha256")!=cfg["ic_sha256"]:raise ValueError("IC transformation verification missing")
    with (r/"production_ready.json").open("x") as f:json.dump(ready,f,indent=2);f.write("\n")
    print(json.dumps(ready,indent=2))
    return ready
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--trial-dir",required=True);a=p.parse_args();check(a.trial_dir)
