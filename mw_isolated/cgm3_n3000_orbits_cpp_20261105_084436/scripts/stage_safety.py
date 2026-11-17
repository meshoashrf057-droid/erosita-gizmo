"""Scope, actual runtime parameters and safety checks for the new live orbit."""
from pathlib import Path
import json, math, os, re, shutil
import h5py
import numpy as np
from run_trial import read_json, require_hash, sha256, utc
from make_trial_runtime import parse_params

ROOT = Path(__file__).resolve().parent.parent
FAMILY_KEY = "cpp-shooting-family:bb9b2bbcbcef2fddb4e66c4084fe218a9ecadb983331db587f7ed2fa0fef854f"
FAMILY_SHA = "90e187bfe2c3f27d270aee247eb27029f88450d006a2e7a7f2aea12f07feaeef"
CONFIG = Path("/home/xingdi/gizmo-public-gizmo-cpp/Config.sh")
CONFIG_SHA = "62490ddc581aa9648afd7338ecacdc4d0385b2dd76544bba5e097918d6d16180"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def new_json(p, record):
    with p.open("x") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)
        file.write("\n")

def memory():
    wanted = {"MemAvailable", "SwapTotal", "SwapFree"}
    return {line.split(":")[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.split(":")[0] in wanted}

def compute_processes():
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            name = (proc/"comm").read_text().strip()
            if name.lower() not in {"gizmo", "mpirun", "orterun", "mpiexec", "mpiexec.hydra"}:
                continue
            row = {"pid":int(proc.name), "name":name}
            try:
                row["cwd"] = os.readlink(proc/"cwd")
                row["cmd"] = (proc/"cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
                status = (proc/"status").read_text().splitlines()
                row["rss_kib"] = next((int(s.split()[1]) for s in status if s.startswith("VmRSS:")), 0)
            except PermissionError:
                row["detail"] = "permission denied"
            found.append(row)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found

def expected_parameters(family, trial):
    params = parse_params((trial/"production/stage2.param").read_text())
    require(params.pop("OutputDir") == "./output/" and params.pop("InitCondFile") == "../orbit_000","wrong run routing")
    normalized = {}
    for k,v in params.items():
        try: normalized[k] = float(v)
        except ValueError: normalized[k] = v
    expected = family["production_parameters"]
    require(normalized == expected, "production parameters differ from the new frozen family: "+str({k:(normalized.get(k),expected.get(k)) for k in normalized.keys()|expected.keys() if normalized.get(k)!=expected.get(k)}))

def verify_before_run(trial, mode):
    require(trial.parent == ROOT and mode in {"smoke","production"},"invalid new-family trial")
    require(os.isatty(0) and os.ttyname(0)==os.environ.get("GIZMO_VISIBLE_TTY"),"required visible Ubuntu TTY missing")
    require(not compute_processes(),"MPI or GIZMO already present")
    require(shutil.disk_usage(trial).free > 20*2**30,"insufficient free space; review retention first")
    require(memory()["MemAvailable"] > 6*1024**2,"insufficient free memory for 6 ranks")
    require_hash(CONFIG, CONFIG_SHA, "C++ build configuration")
    family_path = ROOT/"provenance/cgm3_family_inputs.json"
    require_hash(family_path,FAMILY_SHA,"orbit family inputs")
    family = read_json(family_path)
    expected_parameters(family,trial)
    frozen = read_json(ROOT/"provenance/launch_hashes.json")
    for p,h in frozen["sha256"].items(): require_hash(p,h,"frozen launch dependency")
    audit = read_json(trial/"orbit_000.orbit.json")
    require(audit["family_key"] == FAMILY_KEY and audit["source_sha256"] == family["source_sha256"],"wrong source family")
    static = read_json(trial/"ic_verification.json")
    require(static["status"]=="pass" and static["output_ic_sha256"]==audit["output_ic_sha256"],"missing or stale IC approval")
    if mode=="production":
        evidence = read_json(trial/"smoke_endpoint_review.json")
        require(evidence["status"]=="pass" and evidence["ic_sha256"]==audit["output_ic_sha256"],"smoke endpoint evidence missing")
        snap=Path(evidence["endpoint"]).resolve(strict=True)
        require(snap.parent==(trial/"smoke/output").resolve() and snap.name.startswith("snapshot_"),"invalid smoke endpoint reference")
        require_hash(snap,evidence["sha256"],"smoke endpoint")
        with h5py.File(snap,"r") as h:
            require(abs(float(h["Header"].attrs["Time"])-float(evidence["target"]))<1e-12,"smoke time mismatch")
    new_json(trial/mode/"model_launch_review.json",{"checked_utc":utc(),"status":"pass","family_key":FAMILY_KEY,"ic_sha256":audit["output_ic_sha256"],"mode":mode,"visible_terminal":os.ttyname(0),"config_sha256":CONFIG_SHA,"production_params_semantically_frozen":True,"free_bytes":shutil.disk_usage(trial).free,"meminfo_kib":memory(),"global_mpi_gizmo_before":[]})

def check_smoke_endpoint(trial, paths):
    params = parse_params((trial/"smoke/stage2.param").read_text())
    stamp, endpoint = max(paths)
    target = float(params["TimeMax"])
    require(abs(stamp-target)<1e-12 and len(paths)>=2,"smoke exited before its true endpoint")
    resources=[json.loads(line) for line in (trial/"smoke/resource_samples.jsonl").read_text().splitlines()]
    six=[s for s in resources if sum(p["name"]=="GIZMO" and p.get("cwd")==str(trial/"smoke") for p in s["processes"])==6]
    require(six,"no sample with six real ranks in smoke cwd")
    minavail=min(s["meminfo_kib"]["MemAvailable"] for s in six)
    require(minavail>1024**2,"smoke memory margin below 1 GiB; manual review needed")
    peak_rss=max(sum(p.get("rss_kib",0) for p in s["processes"] if p["name"]=="GIZMO") for s in six)
    log=(trial/"smoke/gizmo_run.log").read_text(errors="replace")
    require("reaching time-limit. stopping." not in log.lower(),"smoke walltime stop requires review")

    end_times=re.findall(r"Final time=([0-9.eE+-]+) reached[.] Simulation ends[.]",log)
    require(bool(end_times) and abs(float(end_times[-1])-target)<1e-10,"missing GIZMO true final-time marker")
    allocations=[float(val) for val in re.findall(r"presently allocated=([0-9.]+) MB",log)]
    require(bool(allocations) and max(allocations)<1000,"GIZMO allocator requires resource review")
    baseline = read_json(ROOT/"provenance/known_runtime_warnings_baseline.json")
    warnings=[l.strip() for l in log.splitlines() if "is being ignored" in l]
    require(set(warnings)==set(baseline["ignored_parameter_lines"]),"unexpected ignored-parameter set: "+str(set(warnings)^set(baseline["ignored_parameter_lines"])))
    dataset_fields=read_json(ROOT/"provenance/expected_snapshot_fields.json")["fields"]
    roots=[l for l in log.splitlines() if "#000:" in l]
    require(all("H5Dcreate1(): unable to create dataset" in l or "H5Dwrite(): not a dataset" in l or "H5Dclose(): not a dataset" in l for l in roots),"new HDF5 diagnosis in smoke")
    fields=0
    with h5py.File(endpoint,"r") as h:
        require(set(dataset_fields)=={k for k in h if k.startswith("PartType")},"endpoint particle group missing")
        counts=h["Header"].attrs["NumPart_Total"]
        require(np.array_equal(counts,h["Header"].attrs["NumPart_ThisFile"]),"incomplete single-file snapshot")
        for group_name,names in dataset_fields.items():
            group=h[group_name]
            require(set(group)==set(names),"endpoint fields differ: "+group_name)
            require(all(len(group[name])==int(counts[int(group_name[8:])]) for name in names),"dataset length disagrees with Header")
        for name in ("Density","InternalEnergy","Temperature","KernelMaxRadius"):
            require((h["PartType0"][name][...] > 0).all(),"invalid endpoint gas field: "+name)
        for group_name,group in h.items():
            if not group_name.startswith("PartType"): continue
            for field,dataset in group.items():
                for offset in range(0,len(dataset),262144):
                    a=dataset[offset:offset+262144]
                    if a.dtype.kind=="f": require(np.isfinite(a).all(), f"nonfinite {group_name}/{field}")
                fields+=1
    cfg=read_json(trial/"run_configuration.json")
    new_json(trial/"smoke_endpoint_review.json",{"status":"pass","checked_utc":utc(),"time":stamp,"target":target,"endpoint":str(endpoint),"sha256":sha256(endpoint),"readable_fields":fields,"float_fields_finite":True,"ic_sha256":cfg["ic_sha256"],"normal_end_marker":end_times[-1],"peak_reported_allocator_mib":max(allocations),"observed_six_rank_samples":len(six),"peak_aggregate_rank_rss_kib":peak_rss,"min_available_kib":minavail,"ignored_parameters_match_reference":True})
