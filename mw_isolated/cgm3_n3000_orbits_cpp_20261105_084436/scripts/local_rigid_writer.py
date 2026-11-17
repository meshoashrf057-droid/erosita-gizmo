#!/usr/bin/env python3
"""Apply one audited rigid 12D orbit increment to an owned copy of an initial IC."""
from pathlib import Path
import argparse, copy, datetime, hashlib, json, os, shutil, sys
import h5py
import numpy as np
PROJECT = Path("/home/xingdi/gizmo-runs")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "mw_isolated"))
from gizmo_ic_compat import validate_generated_ic
from gizmo_units import resolve_units, physical_units
from satellite_relax_common import sha256, particle_keys, plain
from merge_relaxed_mw_lmc_smc import coordinate_extent_report

def vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (12,) or not np.isfinite(x).all():
        raise ValueError(name+" must be a finite 12-vector")
    return x

def apply(proposal_path, trial_dir, campaign_path, label=None):
    proposal_path=Path(proposal_path).resolve(strict=True)
    q=json.loads(proposal_path.read_text())
    campaign_path=Path(campaign_path).resolve(strict=True)
    campaign=json.loads(campaign_path.read_text())
    from local_refine import ROOT,verify_proposal
    if campaign_path!=ROOT/'campaign.json':
        raise ValueError('Only the CGM3 orbit campaign is accepted')
    verify_proposal(q, proposal_path, Path(trial_dir))
    base=Path(q["base_ic"]).resolve(strict=True)
    before_hash=sha256(base)
    if before_hash != q["base_ic_sha256"]:
        raise ValueError("Base initial IC changed")
    current=vector(q["base_initial_12d"],"base")
    step=vector(q["step_12d"],"increment")
    proposed=vector(q["proposed_initial_12d"],"proposed")
    if not np.array_equal(current+step, proposed):
        raise ValueError("Proposed state is not base plus exactly one increment")
    if not np.any(step):
        raise ValueError("Refusing a zero-step duplicate full simulation")
    family=campaign["family_key"]
    family_inputs=campaign["family_inputs"]
    expected_family="cpp-shooting-family:"+hashlib.sha256(
        json.dumps(family_inputs,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if family != expected_family:
        raise ValueError("Campaign family fingerprint does not match its frozen inputs")
    if q.get("family_key",family) != family:
        raise ValueError("Proposal belongs to another family")
    with h5py.File(base,"r") as f:
        audit=json.loads(f["MergeProvenance/AuditJSON"].asstr()[()])
        if audit.get("family_key") != family:
            if str(base)!=campaign["baseline_ic"] or before_hash!=campaign["baseline_ic_sha256"]:
                raise ValueError("Base is outside this frozen campaign")
        if audit.get("family_key") == family and audit.get("model_key_inputs") != family_inputs:
            raise ValueError("Embedded frozen family inputs changed")
        if audit["source_sha256"] != family_inputs["source_sha256"]:
            raise ValueError("Relaxed source realization changed")
        actual_units,_=resolve_units(f)
        for key,expected in family_inputs["raw_units"].items():
            if (not np.isclose(float(audit["raw_units"][key]),float(expected),rtol=1e-12,atol=0)
                    or not np.isclose(actual_units[key],float(expected),rtol=1e-12,atol=0)):
                raise ValueError("Physical units changed: "+key)
        converted=physical_units(actual_units)
        for key in ("length_kpc","velocity_kms"):
            if not np.isclose(float(audit["units"][key]),converted[key],rtol=1e-12,atol=0):
                raise ValueError("Audit conversion differs from HDF5 units: "+key)
        for key in ("stage2_start_code","stage2_target_code"):
            values=np.asarray([audit[key],family_inputs[key]],dtype=float)
            if not np.isfinite(values).all() or abs(values[0]-values[1])>1e-10:
                raise ValueError("Campaign epoch changed: "+key)
        start=float(f["Header"].attrs["Time"])
        if not np.isfinite(start) or abs(start-float(family_inputs["stage2_start_code"]))>1e-10:
            raise ValueError("Initial Header/Time differs from campaign")
        if q["binary_sha256"] != campaign["family_inputs"]["gizmo_sha256"] or audit["gizmo"]["sha256"]!=q["binary_sha256"]:
            raise ValueError("Executable differs from campaign")
        if not np.array_equal(current, vector(audit["applied_initial_12d"],"embedded base")):
            raise ValueError("Proposal does not use actual initial state")
        if not np.array_equal(audit["rotation_matrices"],campaign["family_inputs"]["rotation_matrices"]):
            raise ValueError("Satellite rotation changed")
        if q["measurement_signature"] != campaign["family_inputs"]["measurement_signature"]:
            raise ValueError("Orbit measurement definition changed")
        target=float(q["target_time_code"])
        if not np.isfinite(target) or abs(float(audit["stage2_target_code"])-target)>1e-10:
            raise ValueError("Target epoch changed")
        orbit=f["MagellanicOrbitReference"]
        embedded=np.concatenate([orbit[s+"_AppliedInitial"+v][...]
            for s in ("LMC","SMC") for v in ("Position_kpc","Velocity_kms")])
        if not np.array_equal(current,embedded):
            raise ValueError("Embedded orbit datasets disagree with actual base")
        for name in f:
            if not name.startswith("PartType"):continue
            g=f[name];origin=f["MergeParticleOrigins"][name][...]
            if origin.shape!=(len(g["ParticleIDs"]),) or not np.isin(origin,[0,1,2]).all():
                raise ValueError("Invalid origin map for "+name)
            for field in ("Coordinates","Velocities"):
                if g[field].dtype.kind!="f" or g[field].dtype.itemsize!=8 or not np.isfinite(g[field][...]).all():
                    raise ValueError("Require finite double-precision phase space")
            particle_keys({k:g[k][...] for k in ("ParticleIDs","ParticleChildIDsNumber","ParticleIDGenerationNumber") if k in g})
    for name,expected in ((q["residual_csv"],q["residual_csv_sha256"]),):
        if sha256(name)!=expected:raise ValueError("Residual evidence changed")
    run=Path(trial_dir).resolve()
    if run.parent != campaign_path.parent:
        raise ValueError("Trial must be a direct child of its campaign workspace")
    if run != Path(q["target_run"]).resolve():
        raise ValueError("Local probe trial directory differs from its frozen target run")
    run.mkdir(exist_ok=False)
    label=label or run.name
    new=copy.deepcopy(audit)
    for key in ("output_ic_sha256","output_compatibility"):new.pop(key,None)
    if "coordinate_extent_diagnostics" in new:
        new["parent_coordinate_extent_diagnostics"]=new.pop("coordinate_extent_diagnostics")
    nominal=vector(new["nominal_initial_12d"],"historical nominal")
    cumulative=proposed-nominal
    new.update(version="cpp-orbit-correction-v1",trial_label=label,model_key=family,
        family_key=family,model_key_inputs=campaign["family_inputs"],
        scientific_role="Rigid initial orbit correction of fixed evolved C++ galaxies",
        parent_ic=str(base),parent_ic_sha256=before_hash,
        shooting_base_ic=str(base),shooting_method="local-live-"+q["local_role"],
        shooting_alpha=None,
        shooting_increment_12d=step.tolist(),shooting_diagnostics=q["diagnostics"],
        shooting_residual_csv=q["residual_csv"],shooting_proposal=str(proposal_path),
        shooting_proposal_sha256=sha256(proposal_path),
        applied_initial_12d=proposed.tolist(),cumulative_offset_12d=cumulative.tolist(),
        nominal_and_offsets_role="Historical nominal retained only as an additive coordinate ledger; no reintegration",
        campaign=str(campaign_path),created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    local_fields = ("local_role", "alpha_applied", "anchor_run", "anchor_ic_sha256",
                    "probe_coordinate", "probe_step_value","probe_step_unit","local_group_id",
                    "plan_sha256","anchor_gate_sha256","code_sha256")
    new.update({key:copy.deepcopy(q[key]) for key in local_fields})
    tmp=run/"orbit_000.hdf5.partial";output=run/"orbit_000.hdf5"
    with base.open("rb") as src,tmp.open("xb") as dst:shutil.copyfileobj(src,dst,8*1024*1024)
    length=float(audit["units"]["length_kpc"]);velocity=float(audit["units"]["velocity_kms"])
    if not np.isfinite([length,velocity]).all() or min(length,velocity)<=0:raise ValueError("Invalid physical units")
    with h5py.File(tmp,"r+") as f:
        for name in f:
            if not name.startswith("PartType"):continue
            origins=f["MergeParticleOrigins"][name][...]
            for field,offset,unit in (("Coordinates",0,length),("Velocities",3,velocity)):
                values=f[name][field][...]
                for i in (1,2):
                    values[origins==i]+=step[6*(i-1)+offset:6*(i-1)+offset+3]/unit
                f[name][field][...]=values
        extent_parts={};extent_origins={}
        for name in f:
            if not name.startswith("PartType"):continue
            ptype=int(name[len("PartType"):]);g=f[name]
            if not len(g["ParticleIDs"]):continue
            mass=(g["Masses"][...] if "Masses" in g else
                  np.full(len(g["ParticleIDs"]),float(f["Header"].attrs["MassTable"][ptype])))
            extent_parts[ptype]={"Coordinates":g["Coordinates"][...],
                                 "Velocities":g["Velocities"][...],"Masses":mass}
            extent_origins[ptype]=f["MergeParticleOrigins"][name][...]
        new["coordinate_extent_diagnostics"]=plain(coordinate_extent_report(
            extent_parts,extent_origins,float(f["Header"].attrs["BoxSize"]),audit["units"]))
        del extent_parts,extent_origins
        model=f["MagellanicModelReference"];orbit=f["MagellanicOrbitReference"]
        model.attrs["SATELLITE_MODEL_KEY"]=family;model.attrs["TRIAL_LABEL"]=label
        orbit.attrs["Reference"]="Historical curve retained as background; corrected actual initial state recorded explicitly"
        orbit.attrs["ReferenceCurveRole"]="Historical nominal curve only; not the corrected live trajectory or prediction"
        for i,s in enumerate(("LMC","SMC")):
            pos=proposed[6*i:6*i+3];vel=proposed[6*i+3:6*i+6]
            dp=cumulative[6*i:6*i+3];dv=cumulative[6*i+3:6*i+6]
            for suffix,v in (("_INITIAL_POSITION_RELATIVE_MW_kpc",pos),("_INITIAL_VELOCITY_RELATIVE_MW_km_s",vel),
                             ("_SHOOTING_POSITION_OFFSET_kpc",dp),("_SHOOTING_VELOCITY_OFFSET_km_s",dv)):
                model.attrs[s+suffix]=v
            for suffix,v in (("_AppliedInitialPosition_kpc",pos),("_AppliedInitialVelocity_kms",vel),
                             ("_ShootingPositionOffset_kpc",dp),("_ShootingVelocityOffset_kms",dv)):
                orbit[s+suffix][...]=v
        f["MergeProvenance"].attrs["VERSION"]="cpp-orbit-correction-v1"
        f["MergeProvenance/AuditJSON"][()]=json.dumps(new,ensure_ascii=False)
    compatibility=validate_generated_ic(tmp,require_unique_ids=False)
    if compatibility.get("status") in ("fail","failed"):raise ValueError("Generated IC failed validation")
    verification={"status":"pass","base_ic_sha256":before_hash,"changed_particle_fields":[],
                  "preserved_particle_field_count":0,"step_12d":step.tolist(),"initial_12d":proposed.tolist()}
    with h5py.File(base,"r") as a,h5py.File(tmp,"r") as b:
        for name in a:
            if not name.startswith("PartType"):continue
            origins=a["MergeParticleOrigins"][name][...]
            for field in a[name]:
                old=a[name][field][...];actual=b[name][field][...]
                if field in ("Coordinates","Velocities"):
                    expected=old.copy();offset=0 if field=="Coordinates" else 3;unit=length if offset==0 else velocity
                    for i in (1,2):expected[origins==i]+=step[6*(i-1)+offset:6*(i-1)+offset+3]/unit
                    if not np.array_equal(actual,expected):raise ValueError("Incorrect rigid transform "+name+"/"+field)
                    if not np.array_equal(actual[origins==0],old[origins==0]):raise ValueError("MW phase space changed")
                    verification["changed_particle_fields"].append(name+"/"+field)
                else:
                    if not np.array_equal(actual,old,equal_nan=True):raise ValueError("Internal particle field changed "+name+"/"+field)
                    verification["preserved_particle_field_count"]+=1
            if not np.array_equal(origins,b["MergeParticleOrigins"][name][...]):raise ValueError("Origins changed")
        checked=np.concatenate([b["MagellanicOrbitReference"][s+"_AppliedInitial"+v][...] for s in ("LMC","SMC") for v in ("Position_kpc","Velocity_kms")])
        if not np.array_equal(checked,proposed):raise ValueError("Written initial metadata inconsistent")
    if sha256(base)!=before_hash:raise ValueError("Base input changed during copy")
    new["output_compatibility"]=compatibility
    new["output_compatibility"]["path"]=str(output)
    with h5py.File(tmp,"r+") as f:f["MergeProvenance/AuditJSON"][()]=json.dumps(new,ensure_ascii=False)
    os.link(tmp,output);tmp.unlink()
    new["output_ic_sha256"]=sha256(output)
    (run/"orbit_000.orbit.json").write_text(json.dumps(new,ensure_ascii=False,indent=2)+"\n")
    verification["output_ic_sha256"]=new["output_ic_sha256"]
    (run/"ic_verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"pass","output":str(output),"family_key":family,"initial_12d":proposed.tolist(),"preserved_fields":verification["preserved_particle_field_count"]},indent=2))
    return output

# Invoked by local_refine.py apply under the existing visible-terminal compute lock.
