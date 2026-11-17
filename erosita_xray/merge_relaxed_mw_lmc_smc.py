#!/usr/bin/env python3
"""Merge three independently evolved snapshots; K13 orbit guess; flag2 input.
No particle thermodynamic state is synthesized. Unknown/missing state fails closed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from pathlib import Path
import sys
for _root in Path(__file__).resolve().parents:
    if (_root/"gizmo_units.py").is_file():
        sys.path.insert(0,str(_root))
        break
from gizmo_units import resolve_units, require_noncosmological
from gizmo_schema import normalize_fields, FIELD_ALIASES
from gizmo_ic_compat import finalize_new_ic, validate_generated_ic, checked_ids

import os
import h5py
import numpy as np
import tg19_backend as b
from satellite_relax_common import (VERSION, OFFSETS, ID_FIELDS, cli_run, fail,
    sha256, plain, write_json, read_param, write_param, param_units, check_agn_off,
    executable_info, run_script, header_attrs, particle_keys, read_snapshot)

MERGER_REVISION = "v1.1-nonperiodic-boundary-2026-10-09"

# Quantities recomputed by GIZMO after changing the gravitational environment.
RESET_FIELDS = {"Potential", "Acceleration", "GravitationalAcceleration"}
VECTOR_FIELDS = {"Vorticity", "HydroAcceleration", "HydrodynamicAcceleration",
                 "DensityGradient", "PressureGradient", "InternalEnergyGradient"}
TENSOR_FIELDS = {"VelocityGradient"}
LABEL_FIELDS = {"GasComponent", "StellarComponent", "SourceSystem"}


def end_snapshot(run_dir, system, explicit, executable, raw_units):
    run = Path(run_dir).expanduser().resolve(strict=True)
    manifest = json.loads((run/"relax_manifest.json").read_text(encoding="utf-8"))
    if manifest["system"] != system or manifest["version"] != VERSION:
        fail(f"{run}: 并非本套脚本的 {system} 弛豫目录。")
    if manifest["gizmo"]["sha256"] != executable["sha256"]:
        fail(f"{system}: 当前 GIZMO 与生成弛豫输入时记录的二进制不同。")
    if any(not np.isclose(manifest["raw_units"][k], v, rtol=1e-10, atol=0) for k,v in raw_units.items()):
        fail(f"{system}: 弛豫与合并阶段单位不同。")
    if explicit:
        selected = Path(explicit).expanduser().resolve(strict=True)
    else:
        candidates = []
        for path in sorted((run/"output").glob("snapshot_*.hdf5")):
            if not re.fullmatch(r"snapshot_\d+\.hdf5", path.name):
                continue
            with h5py.File(path, "r") as f:
                candidates.append((float(f["Header"].attrs["Time"]), path))
        if not candidates:
            fail(f"{system}: {run/'output'} 中没有单文件演化快照；先完成弛豫。")
        selected = max(candidates)[1]
    with h5py.File(selected, "r") as f:
        time = float(f["Header"].attrs["Time"])
    if abs(time-manifest["target_end_code"]) > max(1e-7, abs(time)*1e-6):
        fail(f"{system}: 最新快照 Time={time:.12g}，目标={manifest['target_end_code']:.12g}；弛豫尚未完成或取错快照。")
    return selected, manifest


def core_frame(source, system, units, initial_model):
    """Shrinking old-star core, then mass-weighted inner bulk/spin."""
    stars = source["parts"][4]
    ids = stars["ParticleIDs"]
    lo = int(getattr(b, f"{system}_STAR_ID_START"))
    old = (ids >= lo) & (ids < lo+b.ID_BLOCK_SIZE)
    if old.sum() < 32:
        fail(f"{system}: 老恒星核心少于 32 粒子。")
    x = stars["Coordinates"][old].astype(float)*units["length_kpc"]
    v = stars["Velocities"][old].astype(float)*units["velocity_kms"]
    m = stars["Masses"][old].astype(float)
    c = np.median(x, axis=0)
    mask = np.ones(len(x), dtype=bool)
    for _ in range(12):
        radius = np.linalg.norm(x-c, axis=1)
        cut = np.quantile(radius[mask], .8)
        new = radius <= cut
        if new.sum() < max(32, int(.1*len(x))):
            break
        mask = new
        c = np.average(x[mask], weights=m[mask], axis=0)
    inner = np.linalg.norm(x-c, axis=1) < max(initial_model["star_rd"], .5)
    if inner.sum() < 32:
        inner = mask
    bulk = np.average(v[inner], weights=m[inner], axis=0)
    spin_region = np.linalg.norm(x-c, axis=1) < 3*initial_model["star_rd"]
    spin = np.sum(np.cross(x[spin_region]-c, v[spin_region]-bulk)*m[spin_region,None], axis=0)
    if np.linalg.norm(spin) < 1e-15:
        fail(f"{system}: 无法确定恒星盘角动量。")
    return c, bulk, spin/np.linalg.norm(spin)


class MeasuredSatellite:
    def __init__(self, source, center, units):
        radii, masses = [], []
        for g in source["parts"].values():
            radii.append(np.linalg.norm(g["Coordinates"]*units["length_kpc"]-center, axis=1))
            masses.append(g["Masses"]*units["mass_msun"])
        r, mass = np.concatenate(radii), np.concatenate(masses)
        order = np.argsort(r)
        r, cumulative = r[order], np.cumsum(mass[order], dtype=float)
        unique, counts = np.unique(r, return_counts=True)
        self.radius = unique
        self.mass = cumulative[np.cumsum(counts)-1]
        self.live_mass = float(self.mass[-1])

    def menc(self, r):
        return np.interp(r, self.radius, self.mass, left=0., right=self.live_mass)


def host_profile(source, center, units):
    measured = MeasuredSatellite(source, center, units)
    dm = source["parts"][1]
    radius = np.linalg.norm(dm["Coordinates"]*units["length_kpc"]-center, axis=1)
    positive = radius[radius > 0]
    if len(positive) < 20:
        fail("MW 暗晕粒子过少。")
    edges = np.geomspace(max(.1, np.percentile(positive, .05)), max(400, positive.max()*1.05), 121)
    mass = np.histogram(radius, bins=edges, weights=dm["Masses"]*units["mass_msun"])[0]
    rho = mass/(4*np.pi/3*np.diff(edges**3))
    rr = np.sqrt(edges[:-1]*edges[1:])
    valid = rho > 0
    if valid.sum() < 5:
        fail("MW 暗晕无法提供动力学摩擦密度剖面。")
    rho = np.exp(np.interp(np.log(rr), np.log(rr[valid]), np.log(rho[valid])))
    return b.HostProfile(measured.radius, measured.mass, rr, rho)


def assert_compatible(sources):
    known_configs = [s["config"] for s in sources if s["config"]]
    if known_configs and any(c != known_configs[0] for c in known_configs[1:]):
        fail("三个快照的 Config 不一致，不能保证 flag2 字段和物理兼容。")
    for source in sources:
        # This pipeline describes isolated galaxies. GRAVITY_NOT_PERIODIC alone
        # does not disable periodic fluid boundaries (BOX_PERIODIC/PERIODIC).
        box_flags = [key for key in source["config"] if key in ("BOX_PERIODIC", "PERIODIC")
                     or key.startswith(("BOX_SHEARING", "BOX_REFLECT", "BOX_OUTFLOW"))
                     or key == "BOX_DEFINED_SPECIAL_XYZ_BOUNDARY_CONDITIONS_ARE_ACTIVE"]
        if box_flags:
            fail(f"该孤立星系合并流程要求非周期、无特殊盒边界；Config 中检测到 {box_flags}。GRAVITY_NOT_PERIODIC 不会关闭周期流体边界。")
    for p in (0, 1, 4):
        names = set().union(*(s["parts"][p] for s in sources))
        for name in names:
            present = [s["parts"][p].get(name) for s in sources]
            if any(x is None for x in present) and name not in LABEL_FIELDS:
                fail(f"PartType{p}/{name} 并非三个快照都有；不自动猜补物理字段。")
            present = [x for x in present if x is not None]
            if any(x.shape[1:] != present[0].shape[1:] for x in present):
                fail(f"PartType{p}/{name} 列数不同。")
            if any(x.dtype != present[0].dtype for x in present):
                fail(f"PartType{p}/{name} dtype 不同，需统一 GIZMO 编译/输出精度。")
    for source, system in zip(sources, ("MW", "LMC", "SMC")):
        if system != "MW" and set(source["parts"]) != {0,1,4}:
            fail(f"{system}: 独立卫星应只含气体、DM、恒星，无 BH 等其他类型。")
        for p, g in source["parts"].items():
            ids = g["ParticleIDs"].astype(np.uint64)
            if system == "MW":
                invalid = (ids >= b.LMC_ID_START) & (ids <= b.SMC_ID_END)
            else:
                invalid = (ids < getattr(b,system+"_ID_START")) | (ids > getattr(b,system+"_ID_END"))
            if invalid.any():
                fail(f"{system}/PartType{p}: ID 不符合预留区域，拒绝静默重编号。")


def transform_parts(source, system, units, center, bulk, rotation, host_center, host_bulk,
                    target_position, target_velocity, merge_time):
    dt = merge_time-float(source["header"]["Time"])
    actions = {}
    for p, g in source["parts"].items():
        actions[p] = {}
        for key in list(g):
            value = g[key]
            if key == "Coordinates" and system != "MW":
                g[key] = ((value*units["length_kpc"]-center)@rotation.T + host_center + target_position)/units["length_kpc"]
                actions[p][key] = "rigid rotation + translation"
            elif key == "Velocities" and system != "MW":
                g[key] = ((value*units["velocity_kms"]-bulk)@rotation.T + host_bulk + target_velocity)/units["velocity_kms"]
                actions[p][key] = "rigid rotation + boost"
            elif key == "StellarFormationTime":
                g[key] = value.astype(np.float64) + dt
                actions[p][key] = f"birth-time shift {dt:.16g} code; age preserved"
            elif key in RESET_FIELDS:
                g[key] = np.zeros_like(value)
                actions[p][key] = "zero placeholder; GIZMO recomputes combined gravity"
            elif system != "MW" and key in VECTOR_FIELDS:
                if value.shape[1:] != (3,):
                    fail(f"向量 {key} 应为 N×3。")
                g[key] = value@rotation.T
                actions[p][key] = "vector rotation"
            elif system != "MW" and key in TENSOR_FIELDS:
                if value.shape[1:] != (3,3):
                    fail(f"张量 {key} 应为 N×3×3。")
                g[key] = np.einsum("ai,nij,bj->nab", rotation, value, rotation)
                actions[p][key] = "tensor rotation"
            elif system != "MW" and value.shape[1:] in ((3,), (3,3)) and key not in ("Coordinates", "Velocities"):
                fail(f"未识别的矢量/张量字段 {key}，请明确物理含义后添加变换规则。")
            else:
                actions[p][key] = "preserved"
        if p == 4 and np.any(g["StellarFormationTime"] > merge_time+1e-7):
            fail(f"{system}: 存在负年龄恒星，不能把 age 数值当 birth time。")
    return actions


def diagnostics(source, center, units, model=None):
    result = {"counts": {p: len(g["ParticleIDs"]) for p,g in source["parts"].items()},
              "mass_msun": {p: float(g["Masses"].sum()*units["mass_msun"]) for p,g in source["parts"].items()}}
    gas = source["parts"][0]
    r = np.linalg.norm(gas["Coordinates"]*units["length_kpc"]-center, axis=1)
    z = gas["Metallicity"][:,0]/.02
    result.update(gas_radius_number_percentiles_kpc=np.percentile(r,[50,90,99,100]),
        gas_z_mass_weighted_zsun=float(np.average(z,weights=gas["Masses"])),
        gas_z_range_zsun=[float(z.min()),float(z.max())])
    if "Temperature" in gas:
        result["cold_gas_mass_T_lt_2e4K_msun"] = float(gas["Masses"][gas["Temperature"].reshape(-1)<2e4].sum()*units["mass_msun"])
    if model:
        result["gas_mass_over_initial"] = result["mass_msun"][0]/model["cgm_mass_rmax"]
        result["note"] = "Uses all retained particles; does not excise unbound outflow or restore consumed gas."
    return result


def coordinate_extent_report(parts, origins, box, units):
    """Describe nominal-box excursions without changing isolated-system particles.

    Coordinates are intentionally neither wrapped, clipped nor recentered. A
    finite Header/BoxSize is bookkeeping here, not an isolated-system wall.
    A zero BoxSize means no nominal cube is defined, so only extents are logged.
    """
    if not np.isfinite(box) or box < 0:
        fail("Header/BoxSize 必须为非负有限值。")
    result = dict(boundary_assumption="isolated_nonperiodic_no_special_box_boundaries",
                  policy="diagnostic_only_preserve_all_coordinates_and_particles",
                  nominal_box_size_code=box, nominal_box_size_kpc=box*units["length_kpc"],
                  nominal_cube_defined=bool(box > 0), part_types={})
    for p, g in parts.items():
        xyz = np.asarray(g["Coordinates"])
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(g["Velocities"])):
            fail(f"合并后 PartType{p} 坐标或速度出现 NaN/Inf。")
        outside = np.any((xyz < 0) | (xyz > box), axis=1) if box > 0 else None
        mass = g["Masses"] * units["mass_msun"]
        item = dict(count=len(xyz), coordinate_min_code=xyz.min(axis=0),
                    coordinate_max_code=xyz.max(axis=0),
                    coordinate_min_kpc=xyz.min(axis=0)*units["length_kpc"],
                    coordinate_max_kpc=xyz.max(axis=0)*units["length_kpc"],
                    outside_count=None if outside is None else int(outside.sum()),
                    outside_mass_msun=None if outside is None else float(mass[outside].sum()),
                    outside_mass_fraction=None if outside is None else float(mass[outside].sum()/mass.sum()),
                    by_source={})
        for index, name in enumerate(("MW", "LMC", "SMC")):
            selected = origins[p] == index
            if not selected.any():
                continue
            item["by_source"][name] = dict(count=int(selected.sum()),
                outside_count=None if outside is None else int((outside & selected).sum()),
                outside_mass_msun=None if outside is None else float(mass[outside & selected].sum()))
        result["part_types"][f"PartType{p}"] = item
        if outside is not None and outside.any():
            counts = ", ".join(f"{name}={data['outside_count']}" for name,data in item["by_source"].items())
            print(f"信息 PartType{p}: {int(outside.sum())}/{len(xyz)} 粒子位于名义 [0, BoxSize] 外"
                  f"（{counts}）；质量占比={item['outside_mass_fraction']:.3e}。非周期模型保留原坐标，继续写出。")
            print(f"  坐标范围(code): min={item['coordinate_min_code']}, max={item['coordinate_max_code']}")
    return result


def create_stage2_files(output, base, lines, values, executable, start, lookback, units, mpi_tasks):
    work = output.with_suffix("").with_name(output.stem+"_run")
    if work.exists():
        fail(f"拒绝覆盖运行目录 {work}")
    work.mkdir()
    shared = dict(InitCondFile=str(output), ICFormat=3, SnapFormat=3, OutputListOn=0,
        NumFilesPerSnapshot=1, NumFilesWrittenInParallel=1, SnapshotFileBase="snapshot",
        RestartFile="restart", TimeBegin=format(start,".16g"), InitGasTemp=0)
    for mode in ("smoke", "production"):
        sub = work/mode
        (sub/"output").mkdir(parents=True)
        overrides = dict(shared, OutputDir=str(sub/"output")+"/")
        if mode == "smoke":
            duration = max(1e-4, 8*float(values.get("MinSizeTimestep", 4e-6)))
            overrides.update(TimeMax=format(start+duration,".16g"),
                TimeOfFirstSnapshot=format(start+1e-8,".16g"), TimeBetSnapshot=format(duration,".16g"),
                MaxSizeTimestep=format(min(float(values.get("MaxSizeTimestep",.001)), duration/4),".16g"))
        else:
            overrides.update(TimeMax=format(start+lookback/units["time_gyr"],".16g"),
                TimeOfFirstSnapshot=format(start,".16g"), TimeBetSnapshot="0.1")
        param = sub/"stage2.param"
        write_param(param, lines, overrides)
        run_script(sub/"run.sh", executable, param, base.parent, 2, sub/"gizmo_run.log", mpi_tasks)
    return work


def merge(args):
    base, lines, values = read_param(args.base_param)
    raw_units, units = param_units(values)
    check_agn_off(values)
    executable = executable_info(args.gizmo_executable)
    if not np.isfinite(args.lookback_gyr) or args.lookback_gyr <= 0:
        fail("lookback-gyr 必须为正。")
    if args.orbit_samples < 2 or not np.isfinite(args.coulomb_log) or args.coulomb_log < 0:
        fail("orbit-samples 至少为 2；coulomb-log 必须非负且有限。")
    if args.shooting_residual_csv and not args.shooting_base_ic:
        fail("使用残差 CSV 必须同时给 --shooting-base-ic，防止丢失上一轮累计修正。")
    if not 0 < args.shooting_alpha <= 1:
        fail("shooting-alpha 应在 (0,1]。")
    output = Path(args.output).expanduser().resolve()
    if not re.search(r"_\d+\.hdf5$", output.name):
        fail("flag2 输入文件名必须以 _数字.hdf5 结尾，例如 trial00_000.hdf5。")
    sidecar = output.with_suffix(".orbit.json")
    work = output.with_suffix("").with_name(output.stem+"_run")
    if output.exists() or sidecar.exists() or work.exists():
        fail("输出 IC、orbit.json 或配套运行目录已存在，请使用新的 trial 文件名。")
    satellite_inputs = [end_snapshot(args.lmc_run_dir,"LMC",args.lmc_snapshot,executable,raw_units),
                        end_snapshot(args.smc_run_dir,"SMC",args.smc_snapshot,executable,raw_units)]
    paths = [Path(args.mw_snapshot).expanduser().resolve(strict=True)] + [item[0] for item in satellite_inputs]
    sources = [read_snapshot(path,raw_units) for path in paths]
    assert_compatible(sources)
    manifests = [None] + [item[1] for item in satellite_inputs]
    for system,source in zip(("MW","LMC","SMC"),sources):
        if source["missing_units"]:
            print(f"提示 {system}: 快照未存完整单位，采用明确给定的 base-param；请确认 MW 的原运行也采用这些单位。")
        if not source["config"]:
            print(f"提示 {system}: 缺少 Config 元数据，无法从快照证明编译项相同；字段结构已逐项核对。")
    start = float(sources[0]["header"]["Time"]) if args.stage2_start_code is None else args.stage2_start_code
    if not np.isfinite(start) or start < 0:
        fail("合并时间必须为非负有限值。")
    with h5py.File(paths[0],"r") as f:
        hc,hv = b.find_host_center_and_velocity(f)
    hc,hv = hc*units["length_kpc"], hv*units["velocity_kms"]
    centers, bulks, normals = [hc], [hv], [np.array([0.,0.,1.])]
    for source,system,manifest in zip(sources[1:],("LMC","SMC"),manifests[1:]):
        c,v,n = core_frame(source,system,units,manifest["model"])
        centers.append(c); bulks.append(v); normals.append(n)
    profiles = [MeasuredSatellite(source,c,units) for source,c in zip(sources[1:],centers[1:])]
    host = host_profile(sources[0],hc,units)
    targets = [getattr(args,name) for name in ("lmc_target_position","lmc_target_velocity","smc_target_position","smc_target_velocity")]
    orbit_kwargs = dict(profile=host,lmc=profiles[0],smc=profiles[1],lookback_gyr=args.lookback_gyr,
        use_dynamical_friction=not args.no_dynamical_friction,include_host_recoil=not args.no_host_recoil,
        coulomb_log=args.coulomb_log,orbit_samples=args.orbit_samples)
    def integrate(target):
        result = b.integrate_magellanic_orbit(**orbit_kwargs,lmc_target_position=target[0],lmc_target_velocity=target[1],
            smc_target_position=target[2],smc_target_velocity=target[3])
        initial = result[0].copy()
        initial[[3,4,5,9,10,11]] /= b.KMS_TO_KPC_GYR
        return initial, result[1], result[2], result[3]
    print("按弛豫后的质量分布反演轨道；这仍是静态球平均势初猜，需要 live shooting。", flush=True)
    nominal,closure,reference_time,reference_states = integrate(targets)
    rotations = [np.eye(3)]
    for i,system in enumerate(("lmc","smc")):
        normal = getattr(args,system+"_disk_normal")
        if normal is None:
            normal = np.cross(nominal[6*i:6*i+3], nominal[6*i+3:6*i+6])
        if np.linalg.norm(normal) < 1e-12:
            fail(f"{system}: 盘面法向无效。")
        target_rotation = b.rotation_from_z_to_vector(normal.copy())
        source_rotation = b.rotation_from_z_to_vector(normals[i+1].copy())
        rotations.append(target_rotation@source_rotation.T)
    source_hashes = [sha256(p) for p in paths]
    key_inputs = dict(version=VERSION,source_sha256=source_hashes,lookback=args.lookback_gyr,
        targets=targets,rotations=rotations,df=not args.no_dynamical_friction,
        recoil=not args.no_host_recoil,coulomb_log=args.coulomb_log,
        backend_sha256=sha256(Path(__file__).with_name("tg19_backend.py")))
    model_key = VERSION+":"+hashlib.sha256(json.dumps(plain(key_inputs),sort_keys=True).encode()).hexdigest()
    previous = np.zeros(12)
    if args.shooting_base_ic:
        previous = np.concatenate(b.load_base_shooting_offsets(args.shooting_base_ic,model_key))
    increment = np.zeros(12)
    residual = None
    if args.shooting_residual_csv:
        residual = b.load_present_residuals(args.shooting_residual_csv)
        corrected_targets = [t-args.shooting_alpha*r for t,r in zip(targets,residual)]
        corrected,_,_,_ = integrate(corrected_targets)
        increment = corrected-nominal
    manual = np.concatenate([getattr(args,name) for name in ("lmc_position_offset","lmc_velocity_offset","smc_position_offset","smc_velocity_offset")])
    offset = previous+increment+manual
    applied = nominal+offset
    before = [diagnostics(s,c,units,m["model"] if m else None) for s,c,m in zip(sources,centers,manifests)]
    actions = []
    for i,(source,system) in enumerate(zip(sources,("MW","LMC","SMC"))):
        pos = np.zeros(3) if i==0 else applied[6*(i-1):6*(i-1)+3]
        vel = np.zeros(3) if i==0 else applied[6*(i-1)+3:6*(i-1)+6]
        actions.append(transform_parts(source,system,units,centers[i],bulks[i],rotations[i],hc,hv,pos,vel,start))
    parts, origins = {}, {}
    box = float(sources[0]["header"]["BoxSize"])
    for p in sorted(set().union(*(s["parts"] for s in sources))):
        selected = [(i,s["parts"][p]) for i,s in enumerate(sources) if p in s["parts"]]
        fields = set().union(*(g for _,g in selected))
        parts[p] = {}
        for field in sorted(fields):
            template = next(g[field] for _,g in selected if field in g)
            values_to_join = [g.get(field,np.zeros((len(g["ParticleIDs"]),)+template.shape[1:],dtype=template.dtype)) for _,g in selected]
            parts[p][field] = np.concatenate(values_to_join,axis=0)
        origins[p] = np.concatenate([np.full(len(g["ParticleIDs"]),i,dtype=np.uint8) for i,g in selected])
        particle_keys(parts[p])
    coordinate_extents = coordinate_extent_report(parts,origins,box,units)
    counts = [len(parts[p]["ParticleIDs"]) if p in parts else 0 for p in range(6)]
    report = dict(version=VERSION,merger_revision=MERGER_REVISION, target_format=getattr(args,"target_format","c"),
        coordinate_extent_diagnostics=coordinate_extents,
        scientific_role="TG19-inspired independently relaxed galaxies; no binary pre-evolution",
        binary_pre_evolution_gyr=0,source_files=paths,source_sha256=source_hashes,
        source_times_code=[float(s["header"]["Time"]) for s in sources],
        source_relaxation_manifests=manifests,diagnostics_before_merge=before,
        source_centers_kpc=centers,source_bulk_kms=bulks,rotation_matrices=rotations,
        raw_units=raw_units,units=units,stage2_start_code=start,
        stage2_target_code=start+args.lookback_gyr/units["time_gyr"],lookback_gyr=args.lookback_gyr,
        model_key=model_key,model_key_inputs=key_inputs,nominal_initial_12d=nominal,
        applied_initial_12d=applied,cumulative_offset_12d=offset,previous_offset_12d=previous,
        this_iteration_increment_12d=increment,manual_increment_12d=manual,
        residual_csv=args.shooting_residual_csv,residual_csv_sha256=sha256(args.shooting_residual_csv) if args.shooting_residual_csv else None,
        shooting_base_ic=args.shooting_base_ic,shooting_alpha=args.shooting_alpha,
        static_numerical_closure_12d=closure,field_actions=actions,counts=counts,
        gizmo=executable,trial_label=args.trial_label,
        warning="Snapshot restart is not a binary restart. Gravity/hydrodynamic neighborhoods are rebuilt; some subgrid state may be reinitialized by GIZMO.")
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp = output.with_suffix(".hdf5.partial")
    if tmp.exists():
        fail(f"临时文件已存在: {tmp}")
    try:
        with h5py.File(tmp,"x") as f, h5py.File(paths[0],"r") as host_file:
            skip = {"Header","MagellanicModelReference","MagellanicOrbitReference","MergeProvenance","MergeParticleOrigins"}
            for name in host_file:
                if not name.startswith("PartType") and name not in skip:
                    host_file.copy(name,f)
            if args.host_reference_ic:
                with h5py.File(args.host_reference_ic,"r") as ref:
                    for name in ("MWModelReference","ModelNotes","ReservedParticleIDs"):
                        if name in ref and name not in f:
                            ref.copy(name,f)
            if "ModelNotes" in f:
                # Original notes describe the old EQS experiment, not this run.
                note = f["ModelNotes"].attrs
                for key in ("REQUIRED_RUNTIME_PHYSICS", "FIRST_RELAXATION_PHYSICS_OFF"):
                    if key in note:
                        note["ORIGINAL_HOST_"+key] = note[key]
                note["LMC_SMC_PRESENT"] = 1
                note["REQUIRED_RUNTIME_PHYSICS"] = "Same GIZMO build as the three evolved inputs; user's explicit SF/feedback setup; no GALSF_EFFECTIVE_EQS"
                note["FIRST_RELAXATION_PHYSICS_OFF"] = "AGN accretion/jet off; cooling and explicit stellar feedback inherited from actual run"
                note["HOST_ANALYTIC_MODEL_METADATA"] = "Historical IC settings; actual input is evolved MW snapshot. Metadata is not a runtime switch."
            if "Parameters" in f:
                f["Parameters"].attrs["TimeBegin"] = start
                f["Parameters"].attrs["TimeMax"] = report["stage2_target_code"]
                f["Parameters"].attrs["InitCondFile"] = str(output)
            h = f.create_group("Header")
            h.attrs.update(sources[0]["header"])
            h.attrs.update(header_attrs(counts,box,start,raw_units))
            origin_group = f.create_group("MergeParticleOrigins")
            origin_group.attrs["Legend"] = "0=MW, 1=LMC, 2=SMC; IC row labels only, not automatically output by GIZMO"
            for p,g in parts.items():
                group = f.create_group(f"PartType{p}")
                template_source = next(s for s in sources if p in s["parts"])
                group.attrs.update(template_source["group_attrs"][p])
                for name,value in g.items():
                    ds = group.create_dataset(name,data=value)
                    for s in sources:
                        if p in s["field_attrs"] and name in s["field_attrs"][p]:
                            ds.attrs.update(s["field_attrs"][p][name]); break
                origin_group.create_dataset(f"PartType{p}",data=origins[p])
            provenance = f.create_group("MergeProvenance")
            provenance.attrs.update(VERSION=VERSION,MERGER_REVISION=MERGER_REVISION,
                BOUNDARY_ASSUMPTION="isolated_nonperiodic",GIZMO_SHA256=executable["sha256"])
            for source,system,path in zip(sources,("MW","LMC","SMC"),paths):
                g = provenance.create_group(system)
                g.attrs.update(SOURCE_PATH=str(path),SOURCE_SHA256=sha256(path),SOURCE_TIME_CODE=float(source["header"]["Time"]))
                with h5py.File(path,"r") as original:
                    for name in original:
                        if not name.startswith("PartType"):
                            original.copy(name,g)
            model = f.create_group("MagellanicModelReference")
            model.attrs.update(SATELLITE_MODEL_KEY=model_key,SATELLITE_STRUCTURE_MODEL="tg19-independent-relax",
                SATELLITE_GAS_MODEL="tg19-extended-disks-after-isolated-relaxation",
                INSERTION_TIME_CODE=start,PRESENT_TARGET_TIME_CODE=report["stage2_target_code"],
                LOOKBACK_GYR=args.lookback_gyr,TRIAL_LABEL=args.trial_label,
                BINARY_PRE_EVOLUTION_INCLUDED=0,UNIT_TIME_GYR=units["time_gyr"])
            for i,key in enumerate(OFFSETS):
                model.attrs[key] = offset[3*i:3*i+3]
            orbit = f.create_group("MagellanicOrbitReference")
            orbit.attrs.update(Reference="Inherited K13 endpoints; measured relaxed masses; static spherical shooting guess",
                Potential="MW measured spherical profile + extended satellite masses + optional dynamical friction/recoil",
                CoordinateFrame="relative MW centre; original Galactocentric axes; MW itself not rotated",
                Lookback_Gyr=args.lookback_gyr,InsertionTime_Code=start,
                PresentTargetTime_Code=report["stage2_target_code"],UnitTime_Gyr=units["time_gyr"],
                BinaryPreEvolution_Gyr=0,MW_BH_ID=int(b.MW_BH_ID),
                ForwardClosurePositionVelocityMax=float(np.max(np.abs(closure))),
                ReferenceCurveIncludesAppliedShootingOffsets=0)
            orbit.create_dataset("TimeFromPresent_Gyr",data=reference_time)
            for i,system in enumerate(("LMC","SMC")):
                model.attrs[system+"_INDEPENDENT_RELAXATION_GYR"] = manifests[i+1]["target_end_gyr"]
                model.attrs[system+"_INITIAL_POSITION_RELATIVE_MW_kpc"] = applied[6*i:6*i+3]
                model.attrs[system+"_INITIAL_VELOCITY_RELATIVE_MW_km_s"] = applied[6*i+3:6*i+6]
                for component,p in (("GAS",0),("DM",1),("STAR",4)):
                    ids = sources[i+1]["parts"][p]["ParticleIDs"]
                    orbit.attrs[system+"_"+component+"_ID_START"] = int(ids.min())
                    orbit.attrs[system+"_"+component+"_ID_END"] = int(ids.max())
                orbit.attrs[system+"_ID_START"] = int(getattr(b,system+"_ID_START"))
                orbit.attrs[system+"_ID_END"] = int(getattr(b,system+"_ID_END"))
                orbit.attrs[system+"_TotalMass_Msun"] = profiles[i].live_mass
                orbit.attrs[system+"_GasMass_Msun"] = before[i+1]["mass_msun"][0]
                orbit.attrs[system+"_GasRmax_kpc"] = before[i+1]["gas_radius_number_percentiles_kpc"][-1]
                orbit.attrs[system+"_GasProfile"] = "independently_relaxed_tg19_disk"
                for suffix,data in (
                    ("Position_kpc",reference_states[:,6*i:6*i+3]),("Velocity_kms",reference_states[:,6*i+3:6*i+6]),
                    ("Current_PositionSigma_kpc",getattr(b,"K13_"+system+"_POSITION_SIGMA_KPC")),
                    ("Current_VelocitySigma_kms",getattr(b,"K13_"+system+"_VELOCITY_SIGMA_KMS")),
                    ("AppliedInitialPosition_kpc",applied[6*i:6*i+3]),("AppliedInitialVelocity_kms",applied[6*i+3:6*i+6]),
                    ("ShootingPositionOffset_kpc",offset[6*i:6*i+3]),("ShootingVelocityOffset_kms",offset[6*i+3:6*i+6])):
                    orbit.create_dataset(system+"_"+suffix,data=data)
            provenance.create_dataset("AuditJSON",data=json.dumps(plain(report),ensure_ascii=False),dtype=h5py.string_dtype())
            finalize_new_ic(f, getattr(args,"target_format","c"), raw_units)
        validation = validate_generated_ic(tmp, require_unique_ids=False)
        validation["path"] = str(output)
        report["output_compatibility"] = validation
        with h5py.File(tmp, "r+") as checked_output:
            checked_output["MergeProvenance/AuditJSON"][()] = json.dumps(plain(report),ensure_ascii=False)
        # Atomic no-overwrite publication of our own temporary output.
        os.link(tmp, output)
        tmp.unlink()
    except Exception:
        if tmp.exists(): tmp.unlink()
        raise
    report["output_ic_sha256"] = sha256(output)
    write_json(sidecar,report)
    work = create_stage2_files(output,base,lines,values,executable,start,args.lookback_gyr,units,args.mpi_tasks)
    print(f"完成: {output}\ncounts={counts}\nStage2: {start:.12f} → {report['stage2_target_code']:.12f} code")
    for i,system in enumerate(("LMC","SMC")):
        gas = sources[i+1]["parts"][0]
        near = np.min(np.linalg.norm(gas["Coordinates"]*units["length_kpc"]-hc,axis=1))
        print(f"{system}: 弛豫后气体 {before[i+1]['mass_msun'][0]:.6e} Msun；初始 r={np.linalg.norm(applied[6*i:6*i+3]):.3f} kpc，|v|={np.linalg.norm(applied[6*i+3:6*i+6]):.3f} km/s；最近气体距 MW={near:.3f} kpc")
    print(f"同一近似势数值闭合最大残差={np.max(np.abs(closure)):.3e}（非 live 轨道验证）")
    print(f"先读入检查: bash {work/'smoke/run.sh'}\n正式演化: bash {work/'production/run.sh'}")


def verify(args):
    """Compare the near-zero-time flag2 output to the assembled input by compound IDs."""
    results = {}
    with h5py.File(args.reference_ic,"r") as a, h5py.File(args.snapshot,"r") as c:
        dt = float(c["Header"].attrs["Time"])-float(a["Header"].attrs["Time"])
        if dt < -1e-10 or dt > 1e-6:
            fail(f"读入检查需要近零时刻快照，当前 elapsed={dt:.9g} code；请选择 smoke 的第一帧。")
        for p in range(6):
            name = f"PartType{p}"
            if (name in a) != (name in c):
                fail(f"{name}: 类型数量变化，无法当作纯读入检查。")
            if name not in a: continue
            ga,gc = a[name],c[name]
            ka,kc = particle_keys(ga),particle_keys(gc)
            ia,ic = np.argsort(ka),np.argsort(kc)
            if not np.array_equal(ka[ia],kc[ic]):
                fail(f"{name}: 粒子身份不同，可能错误 flag / IDs 重置 / 已发生转星。")
            for key in ("Metallicity","StellarFormationTime","BH_Mass"):
                if key not in ga: continue
                if key not in gc: fail(f"{name}: 输出缺少 {key}")
                x,y = ga[key][...][ia],gc[key][...][ic]
                if x.shape != y.shape: fail(f"{name}/{key}: 输出形状改变。")
                equal = np.isclose(x,y,rtol=1e-6,atol=5e-11)
                results[name+"/"+key] = dict(max_absolute_difference=float(np.max(np.abs(x-y))),
                    changed_entries=int(np.count_nonzero(~equal)),entries=int(x.size))
        bad = {k:v for k,v in results.items() if v["changed_entries"]}
    print(json.dumps(dict(elapsed_code=dt,fields=results),ensure_ascii=False,indent=2))
    if bad:
        fail("近零时刻存在变化；请检查是否使用 flag2，并结合 feedback 日志区分读入覆盖与真实演化。此结果不自动等同于 I/O 错误。")
    print("PASS：粒子身份、金属丰度、恒星形成时间及 BH 质量在读入后保持一致（给定容差内）。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("merge",help="合并三个真实演化末态，输出 flag2 IC 和运行文件")
    p.add_argument("--mw-snapshot",required=True)
    p.add_argument("--lmc-run-dir",default="lmcre")
    p.add_argument("--smc-run-dir",default="smcre")
    p.add_argument("--lmc-snapshot")
    p.add_argument("--smc-snapshot")
    p.add_argument("--host-reference-ic")
    p.add_argument("--base-param",required=True)
    p.add_argument("--gizmo-executable",required=True)
    p.add_argument("--output",default="mw_lmc_smc_relaxed_trial00_000.hdf5")
    p.add_argument("--target-format", choices=("c","cpp"), default="c", help="IC field naming; no change to trajectory or restart mode")
    p.add_argument("--lookback-gyr",type=float,default=1.)
    p.add_argument("--stage2-start-code",type=float,help="默认沿用 MW Header/Time；若重置，所有 stellar birth times 同步平移")
    p.add_argument("--no-dynamical-friction",action="store_true")
    p.add_argument("--no-host-recoil",action="store_true")
    p.add_argument("--coulomb-log",type=float,default=3.)
    p.add_argument("--orbit-samples",type=int,default=501)
    p.add_argument("--shooting-base-ic")
    p.add_argument("--shooting-residual-csv")
    p.add_argument("--shooting-alpha",type=float,default=.5)
    p.add_argument("--trial-label",default="trial00")
    p.add_argument("--mpi-tasks",type=int,default=6)
    for system in ("lmc","smc"):
        p.add_argument(f"--{system}-target-position",type=b.vector_argument,default=getattr(b,"K13_"+system.upper()+"_POSITION_KPC").copy())
        p.add_argument(f"--{system}-target-velocity",type=b.vector_argument,default=getattr(b,"K13_"+system.upper()+"_VELOCITY_KMS").copy())
        p.add_argument(f"--{system}-position-offset",type=b.vector_argument,default=np.zeros(3),help="本次新增修正，非覆盖累计值；使用 =x,y,z")
        p.add_argument(f"--{system}-velocity-offset",type=b.vector_argument,default=np.zeros(3))
        p.add_argument(f"--{system}-disk-normal",type=b.vector_argument,help="目标盘面法向；默认名义初始轨道角动量方向")
    p.set_defaults(func=merge)
    p = sub.add_parser("verify",help="对比 flag2 smoke 第一帧与合并 IC 的 Z/恒星时间/IDs")
    p.add_argument("--reference-ic",required=True)
    p.add_argument("--snapshot",required=True)
    p.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli_run(main)
