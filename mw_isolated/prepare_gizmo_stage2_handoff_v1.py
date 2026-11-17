#!/usr/bin/env python3
r"""GIZMO Stage-2 handoff: relaxed MW + newly inserted satellites.

No GIZMO source edits. Requires Python 3.9+, numpy and h5py.
For NON-COSMOLOGICAL MFM, 11-column metals, PartType4 stars, standard
cooling (including COOL_MOLECFRAC=5), and a pre-existing BH with AGN OFF.
This is a snapshot handoff, NOT a lossless binary-checkpoint restart.

Workflow (keep the same GIZMO executable/configuration throughout):

  python prepare_gizmo_stage2_handoff_v1.py capture \
    --host-snapshot snapshot_006.hdf5 \
    --combined-ic mw_relaxed_lmc_smc_tg19_trial00.hdf5 \
    --param mw_isolated.param --gizmo-executable ~/gizmo-public/GIZMO \
    --output-prefix tg19_handoff --duration-gyr 1.0 \
    --satellite-time-frame host

  mpirun -np 6 ~/gizmo-public/GIZMO tg19_handoff.bootstrap.param 0

  python prepare_gizmo_stage2_handoff_v1.py restore \
    --state tg19_handoff.state.hdf5 \
    --bootstrap-snapshot tg19_handoff_bootstrap_output/snapshot_000.hdf5

  mpirun -np 6 ~/gizmo-public/GIZMO tg19_handoff.smoke.param 2

  python prepare_gizmo_stage2_handoff_v1.py verify \
    --state tg19_handoff.state.hdf5 \
    --snapshot tg19_handoff_smoke_output/snapshot_001.hdf5

  mpirun -np 6 ~/gizmo-public/GIZMO tg19_handoff.production.param 2

capture saves ALL groups/datasets/attributes of the original host and combined
IC in an independent archive. It also creates a bootstrap IC with TEMPORARY
globally unique IDs. Only run flag=0 on THAT bootstrap IC, not the old combined
IC: flag=0 discards child/generation IDs, which otherwise makes split particles
ambiguous. restore restores original parent/child/generation IDs before flag=2.

The bootstrap snapshot MUST be at the exact insertion time, with no physical
evolution. The short bootstrap run may continue, but only its first, zero-elapsed
snapshot is used. Coordinates, velocities, masses and gas U are checked before
restoring anything. No input is modified and existing outputs are never replaced.

Host physical state comes from the relaxed host, in the combined IC frame.
Coordinates/velocities come from the combined IC after checking that its host
differs ONLY by a uniform translation and velocity boost. Rotations, rescaling,
periodic wrapping and changed host particle populations are not supported.
Density, smoothing length, gravity/derivative diagnostics come from the combined
bootstrap and are recalculated during the next GIZMO initialization/evolution.
Old host values of these fields remain available in the full archive.

Host formation times shift by t_insertion - t_host, preserving stellar AGES.
--satellite-time-frame host matches the uploaded v7 generator, which copies the
host median birth time without rebasing. Use stage2 only if the satellite birth
times in your generator are ALREADY in the new clock. This option never changes
the handling of host stars. Optionally choose --satellite-star-age-gyr explicitly.

Unknown host state fields are preserved where possible, not silently zeroed.
If a new satellite needs an undefined value for such a field, restore stops;
--satellite-default FIELD=JSON supplies an explicit value if physically justified.
Nonstandard absolute-time or frame-dependent state fields cause a clear error.
MHD, explicit radiation, cosmic rays, dust/CHIMES and non-equilibrium molecular
extensions are outside the audited scope and are rejected when identifiable.

The output is an INITIALIZATION FILE. Temperature/cooling diagnostics in it may
still reflect bootstrap abundances; analyze fresh flag=2 outputs instead.
This utility cannot undo an earlier evolution with incorrect metallicity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import h5py
import numpy as np

VERSION = "1.0"
ID_FIELDS = ("ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber")
CORE = {"Coordinates", "Velocities", "Masses", *ID_FIELDS}
TIME_FIELDS = {"StellarFormationTime", "BH_FormationTime"}
ELEMENTS = ("Z", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe")
DERIVED = {
    "Density", "SmoothingLength", "KernelRadius", "Potential", "Acceleration",
    "HydroAcceleration", "GravAcceleration", "TidalTensor", "Vorticity",
    "VelocityDivergence", "VelocityCurl", "VelocityGradient", "Pressure",
    "SoundSpeed", "Temperature", "Entropy", "TimeStep", "TimeBin",
    "StarFormationRate", "RateOfChangeOfInternalEnergy", "BH_Mdot",
    "BH_Density", "BH_SmoothingLength", "ConditionNumber", "NumNgb",
    "DensityAroundStar", "DensityAroundStarParticle", "SNe_ThisTimeStep",
}
LABELS = {"GasComponent", "StellarComponent", "ParticleOrigin", "OriginTag"}
UNIT_KEYS = ("UnitLength_in_cm", "UnitVelocity_in_cm_per_s", "UnitMass_in_g")
YEAR_S = 31557600.0


def need(condition, message):
    if not condition:
        raise ValueError(message)


def text_value(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    h = root["Header"].attrs
    need(int(h["NumFilesPerSnapshot"]) == 1, "仅支持单文件 HDF5 快照/IC。")
    need(not np.any(h.get("NumPart_Total_HighWord", np.zeros(6))), "粒子计数高位非零。")
    counts = np.zeros(6, dtype=np.int64)
    for p in range(6):
        name = f"PartType{p}"
        if name not in root:
            continue
        group = root[name]
        need("ParticleIDs" in group, f"{name} 缺少 ParticleIDs。")
        counts[p] = len(group["ParticleIDs"])
        for field, ds in group.items():
            need(isinstance(ds, h5py.Dataset) and ds.ndim >= 1 and ds.shape[0] == counts[p],
                 f"{name}/{field} 不是按粒子排列的数据集。")
    for name in ("NumPart_ThisFile", "NumPart_Total"):
        need(np.array_equal(h[name], counts), f"{name} 与实际粒子数不一致。")
    need(counts[0] > 0 and counts[4] > 0, "需要气体和 PartType4 恒星。")
    need(counts[2] == counts[3] == 0, "本工具要求恒星全部位于 PartType4。")
    for name in ("Omega0", "Omega_Matter", "OmegaLambda", "Omega_Lambda", "Redshift"):
        need(abs(float(h.get(name, 0))) < 1e-12, "只支持非宇宙学模型。")
    need(np.isfinite(float(h["Time"])), "Header/Time 非法。")
    return counts


def integer_column(group, name):
    n = len(group["ParticleIDs"])
    if name not in group:
        return np.zeros(n, dtype=np.uint64)
    a = group[name][:]
    need(a.shape == (n,) and a.dtype.kind in "iu" and np.all(a >= 0),
         f"{group.name}/{name} 必须是一维非负整数。")
    return a.astype(np.uint64)


def identity(group):
    key = np.empty(len(group["ParticleIDs"]), dtype=[(n, "<u8") for n in ID_FIELDS])
    for n in ID_FIELDS:
        key[n] = integer_column(group, n)
    ordered = np.sort(key)
    need(len(key) < 2 or np.all(ordered[1:] != ordered[:-1]),
         f"{group.name}: (父ID, 子ID, 代数) 仍有重复；无法无歧义匹配。")
    return key


def subset_indices(needle, haystack, label):
    """Return indices into haystack for each needle, with no float ID conversion."""
    order = np.argsort(haystack)
    sorted_keys = haystack[order]
    loc = np.searchsorted(sorted_keys, needle)
    need(np.all(loc < len(sorted_keys)), f"{label}: 找不到全部宿主粒子。")
    need(np.array_equal(sorted_keys[loc], needle), f"{label}: ID/子ID/代数集合不匹配。")
    return order[loc]


def masses(root, p):
    g = root[f"PartType{p}"]
    if "Masses" in g:
        a = g["Masses"][:]
    else:
        a = np.full(len(g["ParticleIDs"]), float(root["Header"].attrs["MassTable"][p]))
    need(a.shape == (len(g["ParticleIDs"]),) and np.all(np.isfinite(a)) and np.all(a > 0),
         f"PartType{p} 质量缺失或非法。")
    return np.asarray(a, dtype=np.float64)


def equal(a, b, label, rtol=1e-8, atol=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    need(a.shape == b.shape, f"{label}: 形状不一致。")
    need(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{label}: 有 NaN/Inf。")
    delta = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
    need(np.allclose(a, b, rtol=rtol, atol=atol),
         f"{label}: 差异超限，最大绝对差={delta:.9g}；拒绝把演化或变形当作初始化。")
    return delta


def metal_array(group):
    a = group["Metallicity"][:]
    need(a.shape == (len(group["ParticleIDs"]), 11), f"{group.name} 需要 N×11 金属数组。")
    need(np.all(np.isfinite(a)) and np.all(a >= 0), "金属数组有负数或 NaN/Inf。")
    need(np.all(a[:, 0] + a[:, 1] <= 1.0 + 1e-10), "总 Z 加 He 超过 1。")
    return a


def derived(name):
    return name in DERIVED or name.startswith(("CoolingRate", "HeatingRate", "RateOfChangeOf"))


def check_scope(root):
    if "Config" in root:
        config = root["Config"].attrs
        for option in ("ASSIGN_NEW_IDS", "NO_CHILD_IDS_IN_ICS"):
            need(option not in config,
                 f"Config/{option}: 会重设粒子身份，不能用本工具按原始身份衔接。")
        if "COOL_MOLECFRAC" in config:
            need(text_value(config["COOL_MOLECFRAC"]).strip() not in ("6", "6.0"),
                 "COOL_MOLECFRAC=6 非平衡分子网络超出本工具审查范围。")
    unsupported = ("Magnetic", "DivergenceOfMagnetic", "CosmicRay", "Photon",
                   "Radiation", "Chimes", "Dust", "Grain", "MolecularMassFraction")
    for p in range(6):
        if f"PartType{p}" not in root:
            continue
        for name in root[f"PartType{p}"]:
            need(not name.startswith(unsupported),
                 f"{name}: 超出本工具已审查的普通冷却/MFM 配置，不能自动转换。")
            if name in CORE or name in TIME_FIELDS or derived(name):
                continue
            need(not re.search(r"(?:Time|Position|Coordinates|Velocities)$", name),
                 f"{name}: 未定义该状态量的时间/参考系变换，拒绝猜测。")


def copy_root(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name in src:
        src.copy(name, dst)


def put(group, name, data, attrs=None):
    if name in group:
        old = dict(group[name].attrs)
        del group[name]
    else:
        old = {}
    ds = group.create_dataset(name, data=data)
    for k, v in (attrs if attrs is not None else old).items():
        ds.attrs[k] = v
    return ds


def temporary_path(target):
    fd, name = tempfile.mkstemp(prefix=".stage2_", suffix=".hdf5", dir=target.parent)
    os.close(fd)
    return Path(name)


def publish(temporary, target):
    # Same-filesystem hard link is an exclusive publication, never an overwrite.
    os.link(temporary, target)
    temporary.unlink()


def read_param(path):
    content = Path(path).read_text()
    values = {}
    for line in content.splitlines():
        words = line.strip().split()
        if not words or words[0].startswith(("#", "%")):
            continue
        need(len(words) >= 2 and words[0] not in values, f"参数重复/无值: {line}")
        values[words[0]] = words[1]
    for name, target in (("ComovingIntegrationOn", 0), ("ICFormat", 3), ("SnapFormat", 3),
                         ("NumFilesPerSnapshot", 1), ("InitGasTemp", 0)):
        need(name in values and float(values[name]) == target, f"要求 {name}={target}。")
    for name in ("BlackHoleAccretionFactor", "BlackHoleEddingtonFactor"):
        if name in values:
            need(float(values[name]) == 0, f"阶段二要求 {name}=0；本工具不是 AGN 阶段切换器。")
    for name in UNIT_KEYS:
        need(name in values and float(values[name]) > 0, f"参数文件缺少有效单位 {name}。")
    return content, values


def param_variant(content, updates):
    found = set()
    lines = []
    for line in content.splitlines():
        words = line.strip().split()
        if words and words[0] in updates:
            found.add(words[0])
            line = f"{words[0]:45s} {updates[words[0]]}"
        lines.append(line)
    need(found == set(updates), f"基础参数文件缺少键: {sorted(set(updates)-found)}")
    return "\n".join(lines) + "\n"


def check_units(root, params):
    for name in UNIT_KEYS:
        if name in root["Header"].attrs:
            equal(float(root["Header"].attrs[name]), float(params[name]), name, atol=0)


def check_executable(state, explicit=None):
    path = Path(explicit or text_value(state.attrs["ExecutablePath"])).expanduser().resolve()
    need(path.is_file(), f"找不到原可执行文件 {path}；可用 --gizmo-executable 指定同一文件的新路径。")
    need(digest(path) == text_value(state.attrs["ExecutableSHA256"]),
         "GIZMO 可执行文件改变。bootstrap 与正式读取必须使用相同编译配置/可执行文件。")
    return path


def paths_for(prefix):
    p = str(Path(prefix).expanduser().resolve())
    paths = {"state": Path(p + ".state.hdf5"),
             "bootstrap_ic": Path(p + "_bootstrap_000.hdf5"),
             "ready": Path(p + "_ready_000.hdf5"),
             "audit": Path(p + ".restore_audit.json")}
    for kind in ("bootstrap", "smoke", "production"):
        paths[kind + "_param"] = Path(p + f".{kind}.param")
        paths[kind + "_dir"] = Path(p + f"_{kind}_output")
    need(not any(re.search(r"\s", str(x)) for x in paths.values()), "路径请勿包含空白字符。")
    return paths


def capture(args):
    host_path, ic_path = [Path(x).expanduser().resolve() for x in (args.host_snapshot, args.combined_ic)]
    exe = Path(args.gizmo_executable).expanduser().resolve()
    need(exe.is_file(), f"找不到可执行文件: {exe}")
    content, params = read_param(args.param)
    unit_time = float(params[UNIT_KEYS[0]]) / float(params[UNIT_KEYS[1]]) / (1e9 * YEAR_S)
    need(np.isfinite(args.duration_gyr) and args.duration_gyr > 0, "duration-gyr 必须为正。")
    if args.satellite_star_age_gyr is not None:
        need(np.isfinite(args.satellite_star_age_gyr) and args.satellite_star_age_gyr > 0,
             "卫星恒星年龄必须为正。")
    paths = paths_for(args.output_prefix)
    for path in paths.values():
        need(not path.exists(), f"输出已存在，拒绝覆盖: {path}")
    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    maps, offsets, warnings = {}, {}, []
    dpos = dvel = None
    with h5py.File(host_path, "r") as host, h5py.File(ic_path, "r") as ic:
        nh, nc = inventory(host), inventory(ic)
        check_scope(host)
        check_scope(ic)
        check_units(host, params)
        check_units(ic, params)
        th, ts = float(host["Header"].attrs["Time"]), float(ic["Header"].attrs["Time"])
        need(th >= 0 and ts >= 0, "需要非负代码时间。")
        need(np.all(nc >= nh), "合并 IC 删除了宿主粒子。")
        need(nc[5] == nh[5], "此脚本不支持插入新 BH。")
        shift = ts - th
        next_id = 1
        for p in range(6):
            if not nc[p]:
                continue
            cg = ic[f"PartType{p}"]
            ckey = identity(cg)
            need(int(integer_column(cg, "ParticleIDs").max()) <= 2**args.id_bits - 1,
                 f"PartType{p} 原始 ID 超出 --id-bits={args.id_bits} 范围。")
            for name in ("Coordinates", "Velocities"):
                need(cg[name].shape == (nc[p], 3), f"{name} 需要 N×3。")
            host_index = np.full(nc[p], -1, dtype=np.int64)
            if nh[p]:
                hg = host[f"PartType{p}"]
                ci = subset_indices(identity(hg), ckey, f"PartType{p}")
                host_index[ci] = np.arange(nh[p])
                pdiff = cg["Coordinates"][:][ci] - hg["Coordinates"][:]
                vdiff = cg["Velocities"][:][ci] - hg["Velocities"][:]
                if dpos is None:
                    dpos, dvel = np.median(pdiff, axis=0), np.median(vdiff, axis=0)
                equal(pdiff, np.broadcast_to(dpos, pdiff.shape), "宿主位置平移", atol=2e-6)
                equal(vdiff, np.broadcast_to(dvel, vdiff.shape), "宿主速度平移", atol=2e-6)
                equal(masses(ic, p)[ci], masses(host, p), "宿主质量", atol=0)
                if p == 0:
                    equal(cg["InternalEnergy"][:][ci], hg["InternalEnergy"][:], "宿主内能")
                if p in (0, 4):
                    original_z, combined_z = metal_array(hg), metal_array(cg)
                    if not np.allclose(original_z, combined_z[ci], rtol=1e-8, atol=1e-12):
                        warnings.append(f"PartType{p} 宿主金属在合并 IC 中有差异；以宿主原快照为准恢复。")
            if p in (0, 4):
                metal_array(cg)
            if p == 4:
                birth = cg["StellarFormationTime"][:].astype(float)
                old = host_index >= 0
                birth[old] = host["PartType4/StellarFormationTime"][:][host_index[old]] + shift
                if args.satellite_star_age_gyr is not None:
                    birth[~old] = ts - args.satellite_star_age_gyr / unit_time
                elif args.satellite_time_frame == "host":
                    birth[~old] += shift
                need(np.all(np.isfinite(birth)) and np.all(birth <= ts + 1e-12),
                     "有恒星形成时间晚于插入时刻；检查 --satellite-time-frame。")
            maps[p] = host_index
            offsets[p] = next_id
            next_id += int(nc[p])
        need(next_id - 1 < 2**args.id_bits, "临时 ID 超出编译支持的整数范围。")
        need(dpos is not None, "未匹配到宿主。")
        if "OrbitReference" in ic:
            attrs = ic["OrbitReference"].attrs
            if "InsertionTime_Code" in attrs:
                equal(float(attrs["InsertionTime_Code"]), ts, "轨道插入时间", atol=1e-12)
            if "PresentTargetTime_Code" in attrs:
                orbit_duration = (float(attrs["PresentTargetTime_Code"]) - ts) * unit_time
                if not np.isclose(orbit_duration, args.duration_gyr, rtol=1e-5, atol=1e-7):
                    warnings.append("duration-gyr 与轨道元数据的目标时刻不同；未改写轨道元数据。")
        variants = {}
        for kind in ("bootstrap", "smoke", "production"):
            updates = {
                "InitCondFile": str(paths["bootstrap_ic" if kind == "bootstrap" else "ready"].with_suffix("")),
                "OutputDir": str(paths[kind + "_dir"]) + "/",
                "TimeBegin": f"{ts:.17g}", "TimeOfFirstSnapshot": f"{ts:.17g}",
                "SnapshotFileBase": "snapshot", "OutputListOn": "0",
                "TimeMax": f"{ts + args.duration_gyr / unit_time:.17g}",
            }
            if kind != "production":
                maxstep = min(float(params["MaxSizeTimestep"]), 1e-5)
                need(float(params["MinSizeTimestep"]) <= maxstep, "短测步长小于原 MinSizeTimestep。")
                updates.update(TimeMax=f"{ts + 1e-4:.17g}", TimeBetSnapshot="0.00005",
                               MaxSizeTimestep=f"{maxstep:.17g}")
            variants[kind] = param_variant(content, updates)
        temps, created = [], []
        try:
            state_tmp, boot_tmp = temporary_path(paths["state"]), temporary_path(paths["bootstrap_ic"])
            temps.extend((state_tmp, boot_tmp))
            with h5py.File(state_tmp, "w") as state:
                state.attrs.update(Version=VERSION, HostTime=th, InsertionTime=ts,
                                   TimeShift=shift, UnitTimeGyr=unit_time,
                                   SatelliteTimeFrame=args.satellite_time_frame,
                                   SatelliteStarAgeGyr=(args.satellite_star_age_gyr
                                                        if args.satellite_star_age_gyr is not None else -1.),
                                   ExecutablePath=str(exe), ExecutableSHA256=digest(exe),
                                   HostPath=str(host_path), CombinedICPath=str(ic_path),
                                   IDBits=args.id_bits,
                                   Warnings=json.dumps(warnings, ensure_ascii=False))
                state.attrs["PositionShift"] = dpos
                state.attrs["VelocityShift"] = dvel
                copy_root(host, state.create_group("HostSnapshot"))
                copy_root(ic, state.create_group("CombinedIC"))
                state.create_dataset("OriginalParameterText", data=content, dtype=h5py.string_dtype())
                for kind, path in paths.items():
                    state.attrs["Path_" + kind] = str(path)
                mg = state.create_group("Mapping")
                for p, index in maps.items():
                    g = mg.create_group(f"PartType{p}")
                    g.create_dataset("HostIndex", data=index)
                    g.create_dataset("BootstrapID", data=np.arange(offsets[p], offsets[p]+nc[p],
                                                                  dtype=f"u{args.id_bits//8}"))
            shutil.copyfile(ic_path, boot_tmp)
            with h5py.File(boot_tmp, "r+") as boot:
                boot["Header"].attrs["Flag_Metals"] = np.int32(11)
                boot.attrs["Stage2BootstrapOnly"] = 1
                boot.attrs["Stage2StateArchive"] = str(paths["state"])
                for p in maps:
                    g = boot[f"PartType{p}"]
                    put(g, "ParticleIDs", np.arange(offsets[p], offsets[p]+nc[p], dtype=f"u{args.id_bits//8}"))
                    for field in ID_FIELDS[1:]:
                        if field in g:
                            put(g, field, np.zeros_like(g[field][:]))
            publish(state_tmp, paths["state"])
            created.append(paths["state"])
            publish(boot_tmp, paths["bootstrap_ic"])
            created.append(paths["bootstrap_ic"])
            for kind, par in variants.items():
                with paths[kind + "_param"].open("x") as f:
                    f.write(par)
                created.append(paths[kind + "_param"])
                paths[kind + "_dir"].mkdir()
                created.append(paths[kind + "_dir"])
        except Exception:
            for path in reversed(created):
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink(missing_ok=True)
            raise
        finally:
            for temp in temps:
                temp.unlink(missing_ok=True)
    print(f"完整归档: {paths['state']}")
    print(f"宿主时间={th:.9g}, Stage-2 起点={ts:.9g}, 恒星形成时间平移={shift:.9g} code")
    print(f"宿主统一位置平移={dpos}; 速度平移={dvel}")
    print(f"宿主粒子数={nh.tolist()}; 合并后={nc.tolist()}")
    for warning in warnings:
        print("提示:", warning)
    print(f"下一步: mpirun -np 6 {exe} {paths['bootstrap_param']} 0")
    print("只把零 elapsed 的 bootstrap snapshot_000 交给 restore；勿用于科学分析。")


def canonical_boot_order(state, boot):
    ic = state["CombinedIC"]
    nc, nb = inventory(ic), inventory(boot)
    need(np.array_equal(nc, nb), "bootstrap 粒子数改变，不能恢复。")
    ts = float(state.attrs["InsertionTime"])
    need(abs(float(boot["Header"].attrs["Time"]) - ts) <= 1e-12,
         "bootstrap 已发生物理演化！必须使用与插入时间相同的 snapshot_000。")
    order, checks = {}, {}
    for p in range(6):
        if not nc[p]:
            continue
        g = boot[f"PartType{p}"]
        ids = integer_column(g, "ParticleIDs")
        need(len(np.unique(ids)) == len(ids), "bootstrap 临时 ID 重复；请用 capture 生成的专用 IC。")
        expected = state[f"Mapping/PartType{p}/BootstrapID"][:].astype(np.uint64)
        ix = subset_indices(expected, ids, "bootstrap 临时 ID")
        need(np.array_equal(np.sort(ids), np.sort(expected)), "临时 ID 集合不同。")
        order[p] = ix
        for field in ("Coordinates", "Velocities"):
            checks[f"PartType{p}/{field}"] = equal(ic[f"PartType{p}/{field}"][:], g[field][:][ix],
                                                    f"bootstrap {field}", rtol=1e-9, atol=2e-6)
        checks[f"PartType{p}/Masses"] = equal(masses(ic, p), masses(boot, p)[ix], "bootstrap 质量", atol=0)
        if p == 0:
            checks["PartType0/InternalEnergy"] = equal(ic["PartType0/InternalEnergy"][:],
                g["InternalEnergy"][:][ix], "bootstrap 内能", rtol=1e-7, atol=1e-10)
    return order, checks


def scalar_defaults(items):
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        need(sep and name and name not in out, "--satellite-default 格式为 FIELD=JSON，不可重复。")
        a = np.asarray(json.loads(value))
        need(a.dtype.kind in "fiu" and np.all(np.isfinite(a)), "默认值必须是有限数值或数值数组。")
        out[name] = a
    return out


def state_field(name):
    return name not in CORE and not derived(name)


def merge_state(state, boot, output, orders, defaults):
    host, ic = state["HostSnapshot"], state["CombinedIC"]
    nc = inventory(ic)
    ts, shift = float(state.attrs["InsertionTime"]), float(state.attrs["TimeShift"])
    report = {}
    for p in range(6):
        if not nc[p]:
            continue
        name = f"PartType{p}"
        cg, bg, dest = ic[name], boot[name], output[name]
        hg = host.get(name)
        index = state[f"Mapping/{name}/HostIndex"][:]
        old = index >= 0
        # Output follows combined IC ordering, regardless of MPI/domain reorder.
        for field in list(dest):
            put(dest, field, bg[field][:][orders[p]], dict(bg[field].attrs))
        field_report = {}
        for field in CORE:
            if field == "Masses":
                values = masses(ic, p)
            elif field in ID_FIELDS:
                values = integer_column(cg, field)
                if field == "ParticleIDs":
                    values = values.astype(cg[field].dtype)
                else:
                    values = values.astype(cg[field].dtype if field in cg else np.uint32)
            else:
                values = cg[field][:]
            put(dest, field, values, dict(cg[field].attrs) if field in cg else {})
            field_report[field] = "combined IC; original identity restored"
        source_names = set(cg) | (set(hg) if hg is not None else set())
        for field in sorted(source_names):
            if not state_field(field):
                if derived(field):
                    field_report[field] = "combined-system bootstrap (original host retained in archive)"
                continue
            if field in cg:
                values = cg[field][:]
                attrs = dict(cg[field].attrs)
                provenance = "combined IC"
            elif field in dest:
                values = dest[field][:]
                attrs = dict(dest[field].attrs)
                provenance = "bootstrap for newly added particles"
            elif hg is not None and field in hg:
                shape = (nc[p],) + hg[field].shape[1:]
                values = np.zeros(shape, dtype=hg[field].dtype)
                attrs = dict(hg[field].attrs)
                if np.any(~old):
                    need(field in defaults or field in LABELS,
                         f"{name}/{field}: 宿主有状态字段，但新卫星无对应值；"
                         "需通过 --satellite-default FIELD=JSON 明确提供，不能随意补零。")
                    values[~old] = defaults.get(field, 0)
                provenance = "explicit satellite default"
            else:
                raise ValueError(f"无法确定 {name}/{field} 来源")
            if hg is not None and field in hg:
                original = hg[field][:]
                need(values.shape[1:] == original.shape[1:], f"{name}/{field}: 数据形状不一致。")
                values = values.astype(np.result_type(values.dtype, original.dtype), copy=True)
                values[old] = original[index[old]]
                provenance = "relaxed host; " + provenance + " for satellites"
            if field in TIME_FIELDS:
                values = np.asarray(values, dtype=np.float64)
                values[old] += shift
                if field == "StellarFormationTime" and np.any(~old):
                    age = float(state.attrs["SatelliteStarAgeGyr"])
                    if age > 0:
                        values[~old] = ts - age / float(state.attrs["UnitTimeGyr"])
                    elif text_value(state.attrs["SatelliteTimeFrame"]) == "host":
                        values[~old] += shift
                need(np.all(np.isfinite(values)) and np.all(values <= ts + 1e-12),
                     f"{name}/{field}: 有未来形成时间。")
                provenance += "; clock rebased preserving ages"
            put(dest, field, values, attrs)
            field_report[field] = provenance
        report[name] = field_report
        if p in (0, 4):
            metal_array(dest)
        if p == 5:
            need("BH_Mass" in dest and np.all(dest["BH_Mass"][:] > 0), "恢复后 BH_Mass 非法。")
        # Exact in-file checks of persistent host state, apart from time rebase.
        if hg is not None:
            for field in hg:
                if not state_field(field):
                    continue
                expected = hg[field][:][index[old]]
                if field in TIME_FIELDS:
                    expected = expected.astype(float) + shift
                actual = dest[field][:][old]
                same = np.array_equal(actual, expected, equal_nan=(actual.dtype.kind in "fc"))
                need(same, f"{name}/{field}: 恢复后校验失败。")
            for field in ID_FIELDS:
                need(np.array_equal(integer_column(dest, field)[old], integer_column(hg, field)[index[old]]),
                     f"宿主 {field} 未完整恢复。")
        identity(dest)
    # Reintroduce orbit/reference metadata stripped by GIZMO outputs.
    for name in ic:
        if name == "Header" or name.startswith("PartType") or name in ("Config", "Parameters"):
            continue
        if name in output:
            del output[name]
        ic.copy(name, output)
    for k, v in ic.attrs.items():
        output.attrs[k] = v
    h = output["Header"].attrs
    h["Time"] = ts
    h["MassTable"] = np.zeros(6, dtype=np.float64)
    h["Flag_Metals"] = np.int32(11)
    h["Flag_StellarAge"] = np.int32(1)
    h["Flag_DoublePrecision"] = np.int32(1)
    for key in UNIT_KEYS:
        if key in ic["Header"].attrs:
            h[key] = ic["Header"].attrs[key]
    output.attrs["Stage2HandoffVersion"] = VERSION
    output.attrs["Stage2RequiredRestartFlag"] = 2
    output.attrs["Stage2InsertionTime_Code"] = ts
    output.attrs["Stage2OriginalHostTime_Code"] = float(state.attrs["HostTime"])
    output.attrs["Stage2InitializationDiagnosticsOnly"] = 1
    return report


def restore(args):
    defaults = scalar_defaults(args.satellite_default)
    with h5py.File(args.state, "r") as state, h5py.File(args.bootstrap_snapshot, "r") as boot:
        need(text_value(state.attrs["Version"]) == VERSION, "归档版本不兼容。")
        exe = check_executable(state, args.gizmo_executable)
        ready = Path(text_value(state.attrs["Path_ready"]))
        audit = Path(text_value(state.attrs["Path_audit"]))
        need(not ready.exists() and not audit.exists(), "恢复输出已存在；拒绝覆盖。")
        _, params = read_param(Path(text_value(state.attrs["Path_bootstrap_param"])))
        check_units(boot, params)
        check_scope(boot)
        orders, checks = canonical_boot_order(state, boot)
        # Compare saved compile options when both outputs expose them.
        host = state["HostSnapshot"]
        if "Config" in host and "Config" in boot:
            a, b = dict(host["Config"].attrs), dict(boot["Config"].attrs)
            need(set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a),
                 "宿主与 bootstrap 的 Config 属性不同；本工具要求阶段间保持相同物理配置。")
        temp = temporary_path(ready)
        audit_temp = None
        try:
            shutil.copyfile(args.bootstrap_snapshot, temp)
            with h5py.File(temp, "r+") as dst:
                fields = merge_state(state, boot, dst, orders, defaults)
                dst.attrs["Stage2StateArchive"] = str(Path(args.state).resolve())
                inventory(dst)
            report = {
                "version": VERSION, "host_time_code": float(state.attrs["HostTime"]),
                "insertion_time_code": float(state.attrs["InsertionTime"]),
                "birth_time_shift_code": float(state.attrs["TimeShift"]),
                "bootstrap_no_evolution_checks": checks, "field_sources": fields,
                "satellite_defaults": {k: v.tolist() for k, v in defaults.items()},
                "warnings": json.loads(text_value(state.attrs["Warnings"])),
                "ready": str(ready), "required_flag": 2,
                "note": "Full host archived; persistent state restored; combined-system derivatives recalculated by GIZMO."
            }
            audit_temp = temporary_path(audit)
            audit_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            publish(temp, ready)
            try:
                publish(audit_temp, audit)
            except Exception:
                ready.unlink()
                raise
        finally:
            temp.unlink(missing_ok=True)
            if audit_temp is not None:
                audit_temp.unlink(missing_ok=True)
        print(f"已生成: {ready}\n字段来源报告: {audit}")
        print("宿主物理状态与原始粒子身份已恢复；恒星年龄连续；轨道元数据保留。")
        print(f"先短测: mpirun -np 6 {exe} {text_value(state.attrs['Path_smoke_param'])} 2")
        print("此处未运行 GIZMO；输入中的派生诊断量不能当作科学输出。")


def verify(args):
    with h5py.File(args.state, "r") as state:
        check_executable(state, args.gizmo_executable)
        with h5py.File(text_value(state.attrs["Path_ready"]), "r") as ready, h5py.File(args.snapshot, "r") as snap:
            elapsed = float(snap["Header"].attrs["Time"]) - float(state.attrs["InsertionTime"])
            need(0 <= elapsed <= args.max_elapsed_code + 1e-12, "请使用 flag=2 短测的最早输出。")
            nr, ns = inventory(ready), inventory(snap)
            need(np.array_equal(nr, ns), "短测粒子数已改变；可能发生 SF/分裂/合并，需另行核验，不能判定为读取失败。")
            need(int(snap["Header"].attrs["Flag_Metals"]) == 11, "正式输出 Flag_Metals 不是 11。")
            passed = True
            print(f"elapsed={elapsed:.9g} code = {elapsed * float(state.attrs['UnitTimeGyr']):.9g} Gyr")
            for p in range(6):
                if not nr[p]:
                    continue
                rg, sg = ready[f"PartType{p}"], snap[f"PartType{p}"]
                ix = subset_indices(identity(rg), identity(sg), f"正式 PartType{p}")
                if p in (0, 4):
                    a, b = metal_array(rg), metal_array(sg)[ix]
                    for k, name in enumerate(ELEMENTS):
                        mismatch = np.count_nonzero(~np.isclose(a[:, k], b[:, k], rtol=1e-6, atol=1e-12))
                        lost = np.count_nonzero((a[:, k] > 0) & (b[:, k] == 0))
                        print(f"PartType{p} {name:2}: 不一致={mismatch}, 正值变零={lost}")
                        passed &= mismatch == 0
                if p == 4:
                    a, b = rg["StellarFormationTime"][:], sg["StellarFormationTime"][:][ix]
                    bad = np.count_nonzero(~np.isclose(a, b, rtol=1e-8, atol=1e-10))
                    print(f"恒星形成时间不一致={bad}")
                    passed &= bad == 0
                if p == 5:
                    equal(rg["BH_Mass"][:], sg["BH_Mass"][:][ix], "BH_Mass", rtol=1e-8, atol=0)
            if not passed:
                print("核验有差异：少量气体金属变化可能来自真实反馈/混合，不能一概判为读取错误。")
                return 2
            print("PASS：原始粒子身份、11 列丰度、恒星形成时间和 BH 质量通过短测核验。")
            print("该核验不等同于整个模拟的动力学/热力学验证。")
            print("正式运行参数:", text_value(state.attrs["Path_production_param"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture", help="完整归档宿主+合并IC，生成专用 bootstrap IC 和三套参数")
    p.add_argument("--host-snapshot", required=True)
    p.add_argument("--combined-ic", required=True)
    p.add_argument("--param", required=True, help="与宿主物理配置一致的参数文件；自动生成独立副本")
    p.add_argument("--gizmo-executable", required=True, help="记录二进制 SHA256，确保 bootstrap/正式读取同版")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--duration-gyr", type=float, default=1.0)
    p.add_argument("--satellite-time-frame", choices=("host", "stage2"), required=True,
                   help="v7 使用 host；已正确平移卫星出生时间的其他生成器使用 stage2")
    p.add_argument("--satellite-star-age-gyr", type=float, help="可选：明确覆盖所有新卫星恒星的初始年龄")
    p.add_argument("--id-bits", type=int, choices=(32, 64), default=32,
                   help="须与 GIZMO LONGIDS 编译配置一致；默认32位")
    p = sub.add_parser("restore", help="用严格零elapsed的 flag0 输出恢复阶段二正式输入")
    p.add_argument("--state", required=True)
    p.add_argument("--bootstrap-snapshot", required=True)
    p.add_argument("--gizmo-executable", help="仅当相同二进制移动路径时指定")
    p.add_argument("--satellite-default", action="append", default=[], metavar="FIELD=JSON")
    p = sub.add_parser("verify", help="按完整粒子身份检查 flag2 最早输出")
    p.add_argument("--state", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--gizmo-executable")
    p.add_argument("--max-elapsed-code", type=float, default=0.0001)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            capture(args)
        elif args.command == "restore":
            restore(args)
        else:
            return verify(args)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
