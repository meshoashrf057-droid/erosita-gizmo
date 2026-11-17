#!/usr/bin/env python3
"""Replay an audited orbit on explicitly selected evolved snapshots.
This creates a new experiment; it does not approve or amend a historical shooting build.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import h5py

CAMPAIGN = Path(__file__).resolve().parents[1]
MW = CAMPAIGN.parent
ROOT = MW.parent
ISOLATED = MW / 'cgm3_isolated_20261103_155517'
SOURCE_FREEZE = ISOLATED / 'provenance/future_satellite_sources.json'
ACCEPTED = MW / 'shooting_cpp_20261031_082656/accepted_orbit.json'
sys.path.insert(0, str(MW))
sys.path.insert(0, str(ROOT))
import merge_relaxed_mw_lmc_smc as m
from satellite_relax_common import (
    read_snapshot, read_param, param_units, check_agn_off, executable_info,
    sha256, plain, header_attrs, particle_keys, write_json,
)
from gizmo_ic_compat import finalize_new_ic, validate_generated_ic

SYSTEMS = ("MW", "LMC", "SMC")

def replay(args):
    run = Path(args.run_dir).expanduser().resolve()
    if run != CAMPAIGN/'calibration01' or not run.is_dir():
        raise RuntimeError("Require the pre-created calibration01 directory")
    products = ("input_manifest_before.json","orbit_000.hdf5","orbit_000.hdf5.partial","orbit_000.orbit.json")
    if any((run / name).exists() for name in products):
        raise FileExistsError("A prior merge attempt or product exists")
    freeze=json.loads(SOURCE_FREEZE.read_text())
    accept=json.loads(ACCEPTED.read_text())
    campaign=json.loads((CAMPAIGN/"campaign.json").read_text())
    if campaign['jet_authorized'] or not os.environ.get("TMUX_PANE") or not Path("/dev/tty").exists():
        raise ValueError("Require orbit-only campaign in attached terminal")
    with open('/dev/tty','rb',buffering=0) as console:
        if not os.isatty(console.fileno()): raise RuntimeError("Missing controlling tty")
    reference = Path(args.orbit_ic).expanduser().resolve(strict=True)
    audit = m.read_merge_audit(reference)
    expected_hash = json.loads(reference.with_suffix(".orbit.json").read_text()).get("output_ic_sha256")
    reference_hash = sha256(reference)
    if reference_hash != accept['initial_ic_sha256'] or str(reference) != accept['initial_ic']:
        raise ValueError("Exact accepted initial IC is required")
    if not expected_hash or reference_hash != expected_hash:
        raise ValueError("Historical orbit IC differs from its recorded hash")
    applied = np.asarray(audit["applied_initial_12d"], dtype=float)
    rotations = np.asarray(audit["rotation_matrices"], dtype=float)
    if applied.shape != (12,) or not np.isfinite(applied).all():
        raise ValueError("Invalid historical applied 12D state")
    if rotations.shape != (3,3,3) or not np.isfinite(rotations).all():
        raise ValueError("Invalid historical rotations")
    if not np.array_equal(rotations[0], np.eye(3)):
        raise ValueError("Replay requires an unrotated MW")
    for rotation in rotations:
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0):
            raise ValueError("Historical transform is not a proper rotation")
    with h5py.File(reference, "r") as ref:
        embedded = np.concatenate([ref["MagellanicOrbitReference"][s+"_AppliedInitial"+q][...]
                                   for s in SYSTEMS[1:] for q in ("Position_kpc", "Velocity_kms")])
        if not np.array_equal(applied, embedded):
            raise ValueError("Historical audit and orbit datasets disagree")
    if not np.array_equal(rotations,freeze['model_key_inputs']['rotation_matrices']) or audit['source_sha256'] != freeze['source_sha256']:
        raise ValueError("Accepted historical family differs from frozen satellite provenance")
    base, _, values = read_param(args.base_param)
    if sha256(base) != 'cfba684bd55b2acc093175106a6a55af879eae532876937eb48666b6f6dd4105':
        raise ValueError("Historical physical parameter template hash changed")
    raw_units, units = param_units(values)
    check_agn_off(values)
    for key, wanted in audit["raw_units"].items():
        if not np.isclose(raw_units[key], wanted, rtol=1e-10, atol=0):
            raise ValueError("Historical orbit and new parameter units differ")
    executable = executable_info(args.gizmo_executable)
    paths = [Path(getattr(args, s.lower()+"_snapshot")).expanduser().resolve(strict=True) for s in SYSTEMS]
    input_manifest = []
    for path in paths:
        st = path.stat()
        input_manifest.append(dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha256(path)))
    write_json(run/"input_manifest_before.json", input_manifest)
    if [str(x) for x in paths] != [campaign['host_endpoint'], *freeze['source_files'][1:]]:
        raise ValueError('Wrong new snapshot inputs')
    if [i['sha256'] for i in input_manifest] != [campaign['host_sha256'], *freeze['source_sha256'][1:]]:
        raise ValueError('Input checksum mismatch')
    if executable['sha256'] != freeze['model_key_inputs']['gizmo_sha256']:
        raise ValueError('GIZMO changed')
    if sha256(args.host_reference_ic) != '8bf6e31f2d1dd55a0f93ab57ed6076e7364666991b078f3607dd0d9265245304':
        raise ValueError('New host IC reference metadata does not match its frozen initial IC')
    sources = [read_snapshot(path, raw_units) for path in paths]
    m.assert_compatible(sources)
    for source in sources:
        if int(source["header"].get("GIZMO_version", 0)) != 2026:
            raise ValueError("This replay requires the explicitly selected GIZMO 2026 snapshots")
    start = float(sources[0]["header"]["Time"])
    expected_source_times = [campaign["start_time_code"], *freeze["source_times_code"][1:]]
    actual_source_times = [float(s["header"]["Time"]) for s in sources]
    if not np.allclose(actual_source_times, expected_source_times, rtol=0, atol=1e-13):
        raise ValueError("Source times differ from individually frozen epochs")
    epoch_offsets = [start-time for time in actual_source_times]
    # The satellites are not evolved for this offset. transform_parts shifts all
    # stellar birth times by this same dt, preserving each star's physical age.
    if any(abs(dt) > 3e-5 for dt in epoch_offsets):
        raise ValueError("Unreviewed source epoch difference")
    lookback = float(audit["lookback_gyr"])
    if not np.isfinite(lookback) or lookback != 1.0:
        raise ValueError("Require exactly one physical Gyr infall")
    models = freeze['source_models']
    if models != audit['source_models']: raise ValueError('Satellite core models differ from accepted run')
    with h5py.File(paths[0], "r") as f:
        hc, hv = m.b.find_host_center_and_velocity(f)
    hc, hv = hc*units["length_kpc"], hv*units["velocity_kms"]
    centers, bulks = [hc], [hv]
    for source, system, model in zip(sources[1:], SYSTEMS[1:], models[1:]):
        center, bulk, _ = m.core_frame(source, system, units, model)
        centers.append(center); bulks.append(bulk)
    before = [m.diagnostics(s,c,units,model) for s,c,model in zip(sources,centers,models)]
    original_keys = [{p: particle_keys(g) for p,g in s["parts"].items()} for s in sources]
    actions = []
    for i, (source, system) in enumerate(zip(sources, SYSTEMS)):
        pos = np.zeros(3) if i == 0 else applied[6*(i-1):6*(i-1)+3]
        vel = np.zeros(3) if i == 0 else applied[6*(i-1)+3:6*(i-1)+6]
        actions.append(m.transform_parts(source,system,units,centers[i],bulks[i],rotations[i],hc,hv,pos,vel,start))
        for p,g in source["parts"].items():
            if not np.array_equal(original_keys[i][p], particle_keys(g)):
                raise ValueError("Particle identity changed during transform")
    parts, origins = {}, {}
    for p in sorted(set().union(*(s["parts"] for s in sources))):
        selected = [(i,s["parts"][p]) for i,s in enumerate(sources) if p in s["parts"]]
        fields = set().union(*(g for _,g in selected))
        parts[p] = {}
        for field in sorted(fields):
            template = next(g[field] for _,g in selected if field in g)
            parts[p][field] = np.concatenate([g.get(field,np.zeros((len(g["ParticleIDs"]),)+template.shape[1:],dtype=template.dtype)) for _,g in selected])
        origins[p] = np.concatenate([np.full(len(g["ParticleIDs"]),i,dtype=np.uint8) for i,g in selected])
        particle_keys(parts[p])
    counts = [len(parts[p]["ParticleIDs"]) if p in parts else 0 for p in range(6)]
    box = float(sources[0]["header"]["BoxSize"])
    target = start + lookback/units["time_gyr"]
    key_inputs = json.loads(json.dumps(freeze["model_key_inputs"]))
    key_inputs["source_sha256"] = [x["sha256"] for x in input_manifest]
    if key_inputs["production_parameters"]["CpuTimeBetRestartFile"] != 28800:
        raise ValueError("Unexpected historical checkpoint cadence")
    key_inputs["production_parameters"].update(
        CpuTimeBetRestartFile=900.0, CritPhysDensity=3000.0,
        MaxKernelRadius=1e56, MinSizeTimestep=1e-12,
        TimeBegin=start, TimeOfFirstSnapshot=start, TimeMax=target)
    key_inputs["stage2_start_code"] = start
    key_inputs["stage2_target_code"] = target
    if not np.isclose(target, campaign["target_time_code"], rtol=0, atol=1e-13) or start!=campaign["start_time_code"]:
        raise ValueError("New epochs mismatch")
    model_key = "cpp-shooting-family:"+hashlib.sha256(json.dumps(key_inputs,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if model_key == audit['model_key']: raise ValueError('Historical family key reused')
    fpath = CAMPAIGN/'provenance/cgm3_family_inputs.json'
    with fpath.open('x') as f: json.dump(key_inputs,f,indent=2,allow_nan=False); f.write(chr(10))
    report = dict(version="cgm3-calibration-initial-merge-v1", family_key=model_key, trial_label=args.label, target_format="cpp",
        scientific_role="Same historical applied orbit and rotation matrices; new evolved galaxies; no new shooting correction",
        replay_orbit_ic=str(reference), replay_orbit_ic_sha256=reference_hash,
        historical_trial_label=audit["trial_label"], historical_model_key=audit["model_key"],
        source_files=[str(x) for x in paths], source_sha256=[x["sha256"] for x in input_manifest],
        source_times_code=actual_source_times,
        source_birth_time_offsets_code=epoch_offsets,
        source_epoch_alignment="Reference epoch relabel only; each satellite birth time += host_time-source_time, conserving stellar ages, phase space, masses, metals and thermodynamics; no added evolution.",
        source_centers_kpc=centers, source_bulk_kms=bulks, rotation_matrices=rotations,
        diagnostics_before_merge=before, raw_units=raw_units, units=units, counts=counts,
        stage2_start_code=start, stage2_target_code=target, lookback_gyr=lookback,
        model_key=model_key, model_key_inputs=key_inputs, applied_initial_12d=applied,
        nominal_initial_12d=applied, cumulative_offset_12d=np.zeros(12),
        shooting_increment_12d=np.zeros(12), alpha_applied=False,
        nominal_and_offsets_role="New family uses the old accepted actual initial 12D as its nominal ledger origin. No correction yet; no old residual/Jacobian reused.",
        source_relaxation_manifests_role="Historical initial models used only to define the old-star core aperture",
        source_models=models, field_actions=actions, gizmo=executable,
        base_param=str(base), base_param_sha256=sha256(base),
        coordinate_extent_diagnostics=m.coordinate_extent_report(parts,origins,box,units),
        build_provenance_limit="All three headers identify GIZMO 2026 and schemas match. No Config or source-binary SHA in snapshots; old satellite manifests/logs are stale and are not rewritten or treated as proof of the new build.",
        initial_metallicity_policy="Preserve evolved values with RestartFlag=2; no global reset",
        warning="A new C++ experiment. Historical t13 convergence/acceptance is not inherited.")
    output = run/"orbit_000.hdf5"
    tmp = run/"orbit_000.hdf5.partial"
    with h5py.File(tmp,"x") as f, h5py.File(paths[0],"r") as host, h5py.File(reference,"r") as ref:
        for name in host:
            if not name.startswith("PartType") and name != "Header":
                host.copy(name,f)
        for name in ("MagellanicModelReference","MagellanicOrbitReference"):
            if name in f: raise ValueError("Input MW already contains satellite orbit metadata")
            ref.copy(name,f)
        if args.host_reference_ic:
            with h5py.File(args.host_reference_ic,"r") as hostref:
                for name in ("MWModelReference","ReservedParticleIDs","ModelNotes"):
                    if name in hostref and name not in f: hostref.copy(name,f)
        header = f.create_group("Header")
        header.attrs.update(sources[0]["header"])
        header.attrs.update(header_attrs(counts,box,start,raw_units))
        for p,g in parts.items():
            group=f.create_group(f"PartType{p}")
            template=next(s for s in sources if p in s["parts"])
            group.attrs.update(template["group_attrs"][p])
            for name,value in g.items():
                ds=group.create_dataset(name,data=value)
                for source in sources:
                    if name in source["field_attrs"].get(p,{}):
                        ds.attrs.update(source["field_attrs"][p][name]);break
        origin_group=f.create_group("MergeParticleOrigins")
        origin_group.attrs["Legend"]="0=MW, 1=LMC, 2=SMC; reference IC row labels, not automatically output by GIZMO"
        for p,value in origins.items():origin_group.create_dataset(f"PartType{p}",data=value)
        provenance=f.create_group("MergeProvenance")
        provenance.attrs.update(VERSION="cgm3-calibration-initial-merge-v1",GIZMO_SHA256=executable["sha256"],BOUNDARY_ASSUMPTION="isolated_nonperiodic")
        for source,system,item in zip(sources,SYSTEMS,input_manifest):
            group=provenance.create_group(system)
            group.attrs.update(SOURCE_PATH=item["path"],SOURCE_SHA256=item["sha256"],SOURCE_TIME_CODE=float(source["header"]["Time"]))
            with h5py.File(item["path"],"r") as original:
                for name in original:
                    if not name.startswith("PartType"):original.copy(name,group)
        model=f["MagellanicModelReference"];orbit=f["MagellanicOrbitReference"]
        model.attrs.update(SATELLITE_MODEL_KEY=model_key,TRIAL_LABEL=args.label,
            INSERTION_TIME_CODE=start,PRESENT_TARGET_TIME_CODE=target,LOOKBACK_GYR=lookback,
            SOURCE_ORBIT_IC=str(reference),BINARY_PRE_EVOLUTION_INCLUDED=0)
        orbit.attrs.update(InsertionTime_Code=start,PresentTargetTime_Code=target,
            Reference="Legacy nominal static orbit shown only for historical context; actual launch relative state from accepted C++ initial IC",
            ReferenceCurveRole="Historical nominal static curve only; not recomputed and not a prediction for this new run",
            ReferenceCurveIncludesAppliedShootingOffsets=0)
        for i,system in enumerate(SYSTEMS[1:]):
            for suffix,dataset_suffix in [('SHOOTING_POSITION_OFFSET_kpc','ShootingPositionOffset_kpc'),('SHOOTING_VELOCITY_OFFSET_km_s','ShootingVelocityOffset_kms')]:
                model.attrs[system+'_'+suffix]=np.zeros(3)
                orbit[system+'_'+dataset_suffix][...]=np.zeros(3)
            model.attrs[system+"_INDEPENDENT_RELAXATION_GYR"]=float(sources[i+1]["header"]["Time"])*units["time_gyr"]
            orbit.attrs[system+"_TotalMass_Msun"]=sum(before[i+1]["mass_msun"].values())
            orbit.attrs[system+"_GasMass_Msun"]=before[i+1]["mass_msun"][0]
            orbit.attrs[system+"_GasRmax_kpc"]=before[i+1]["gas_radius_number_percentiles_kpc"][-1]
            for component,p in (("GAS",0),("DM",1),("STAR",4)):
                ids=sources[i+1]["parts"][p]["ParticleIDs"]
                if component=="STAR":
                    orbit.attrs[system+"_ALL_STAR_ID_START"]=int(ids.min())
                    orbit.attrs[system+"_ALL_STAR_ID_END"]=int(ids.max())
                    lo=int(getattr(m.b,system+"_STAR_ID_START"))
                    ids=ids[(ids>=lo)&(ids<lo+m.b.ID_BLOCK_SIZE)]
                orbit.attrs[system+"_"+component+"_ID_START"]=int(ids.min())
                orbit.attrs[system+"_"+component+"_ID_END"]=int(ids.max())
        if "ModelNotes" in f:
            f["ModelNotes"].attrs.update(LMC_SMC_PRESENT=1,
                REQUIRED_RUNTIME_PHYSICS="New CGM3 n3000 host and original frozen satellites; CritPhysDensity3000, MaxKernelRadius1e56, MinSizeTimestep1e-12 inherited from chosen host; all other physical parameters preserved; AGN off",
                FIRST_RELAXATION_PHYSICS_OFF="Historical host initial conditions only; current input is already evolved")
        finalize_new_ic(f,"cpp",raw_units)
        provenance.create_dataset("AuditJSON",data=json.dumps(plain(report),ensure_ascii=False),dtype=h5py.string_dtype())
    report["output_compatibility"]=validate_generated_ic(tmp,require_unique_ids=False)
    report["output_compatibility"]["path"]=str(output)
    with h5py.File(tmp,"r+") as f:
        f["MergeProvenance/AuditJSON"][()]=json.dumps(plain(report),ensure_ascii=False)
    os.link(tmp,output);tmp.unlink()
    report["output_ic_sha256"]=sha256(output)
    write_json(output.with_suffix(".orbit.json"),report)
    for item in input_manifest:
        st=Path(item["path"]).stat()
        if st.st_size!=item["size"] or st.st_mtime_ns!=item["mtime_ns"] or sha256(item["path"])!=item["sha256"]:
            raise RuntimeError("Input changed during merge")
    print(json.dumps(plain(dict(output=str(output),counts=counts,applied_initial_12d=applied,start=start,target=target)),indent=2))
    return output

def main():
    freeze=json.loads(SOURCE_FREEZE.read_text())
    accepted=json.loads(ACCEPTED.read_text())
    cfg=json.loads((CAMPAIGN/'campaign.json').read_text())
    return replay(argparse.Namespace(
        run_dir=CAMPAIGN/'calibration01',orbit_ic=accepted['initial_ic'],
        mw_snapshot=cfg['host_endpoint'],lmc_snapshot=freeze['source_files'][1],
        smc_snapshot=freeze['source_files'][2],host_reference_ic=ISOLATED/'mw_cgm3_initial.hdf5',
        base_param=Path(accepted['run'])/'production/stage2.param',
        gizmo_executable='/home/xingdi/gizmo-public-gizmo-cpp/GIZMO',
        label='cgm3_n3000_calibration01'))

if __name__ == "__main__": main()
