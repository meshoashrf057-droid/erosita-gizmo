#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Stage-2 LMC/SMC 轨道的 6D 相空间，而不是只比较距离和总速度。

输入由 add_lmc_smc_gaswake_orbit_v3.py 生成的 reference IC，
以及 GIZMO 输出目录。脚本使用卫星原生 PartType4 恒星核追踪中心，避免被
暗物质尾流拖偏；银河中心优先用 ID=90000000 的中心 BH，缺失时才用 MW
恒星的 shrinking-sphere 中心。除 K13 终点 6D 外，还检查固定孔径质量保留、
示踪粒子完整性和（若 snapshot 输出了 TimeStep）宿主气体短时间步比例。

v3.2 默认冻结 reference IC 中原始老恒星的精确复合身份，不纳入后续新星。
--stellar-tracers legacy 可重现旧版范围测量；切换口径后应重新诊断所有比较轮次。
固定孔径质量为同一原始老恒星示踪样本 + 原始 DM，不代表全部恒星质量或束缚质量。

输出：
  magellanic_6d_history.csv        每个快照的 LMC/SMC 6D 与参考残差
  magellanic_present_6d.csv        今天时刻逐分量 sim/obs/residual/sigma
  magellanic_validation_summary.json  当今目标时刻的逐分量判定
  magellanic_6d_dashboard.png      6D 历史、参考轨道和残差图板

示例：
python validation_dashboard_sat_6d_gaswake_v3.py \
  --snapshot-dir ./output \
  --reference-ic mw_mou_cgm_lmc_smc_tminus0p5.hdf5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


UNIT_TIME_GYR_DEFAULT = 0.9777922216807892
MW_BH_ID = 90_000_000
K13_RELATIVE_SPEED_KMS = 128.0
K13_RELATIVE_SPEED_SIGMA_KMS = 32.0
K13_LMC_SPEED_KMS = 321.0
K13_LMC_SPEED_SIGMA_KMS = 24.0
K13_SMC_SPEED_KMS = 217.0
K13_SMC_SPEED_SIGMA_KMS = 26.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attribute_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


IDENTITY_FIELDS = ("ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber")
OLD_STAR_BLOCKS = {"lmc": (120_000_000, 129_999_999), "smc": (210_000_000, 219_999_999)}


def measurement_signature(lmc_core_kpc: float, smc_core_kpc: float,
                          tracer_policy: str = "original-stars", identity_fields=None) -> dict[str, object]:
    """A change of tracer policy is a change of the shooting measurement."""
    return {
        "algorithm": ("fixed-original-star-identities-shrinking-core-v4" if tracer_policy == "original-stars"
                      else "legacy-id-range-shrinking-core-v3"),
        "tracer_policy": tracer_policy,
        "identity_fields": list(identity_fields or ("ParticleIDs",)),
        "host_frame": "BH-position+MW-inner10kpc-mass-weighted-star-velocity-v3",
        "lmc_core_radius_kpc": float(lmc_core_kpc),
        "smc_core_radius_kpc": float(smc_core_kpc),
    }


def identity_keys(group, fields=None, mask=None):
    """Read exact integer identities; never silently drop child/generation fields."""
    present = tuple(name for name in IDENTITY_FIELDS if name in group)
    fields = tuple(fields or present)
    if not fields or fields[0] != "ParticleIDs" or present != fields:
        raise RuntimeError(f"stellar identity fields differ: reference={fields}, snapshot={present}; cannot match safely")
    arrays = []
    for field in fields:
        value = np.asarray(group[field][:])
        if value.ndim != 1 or value.dtype.kind not in "ui" or np.any(value < 0):
            raise RuntimeError(f"{field} must be a one-dimensional non-negative integer identity")
        arrays.append(value if mask is None else value[mask])
    if any(len(a) != len(arrays[0]) for a in arrays):
        raise RuntimeError("stellar identity column lengths differ")
    keys = np.empty(len(arrays[0]), dtype=[(field, np.uint64) for field in fields])
    for field, values in zip(fields, arrays):
        keys[field] = values
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError("duplicate complete stellar identities; provide child/generation fields or repair the source")
    return keys


def fixed_tracer_mask(group, ref, sat):
    ids = np.asarray(group["ParticleIDs"][:], dtype=np.uint64)
    lo, hi = ref["old_star_id_ranges"][sat]
    candidates = (ids >= lo) & (ids <= hi)
    keys = identity_keys(group, ref["identity_fields"], candidates)
    mask = np.zeros(len(ids), dtype=bool)
    mask[candidates] = np.isin(keys, ref[f"{sat}_fixed_tracer_keys"])
    return mask


def selected_tracer_mask(group, ref, sat):
    if ref.get("tracer_policy", "legacy") == "original-stars":
        return fixed_tracer_mask(group, ref, sat)
    ids = np.asarray(group["ParticleIDs"][:], dtype=np.uint64)
    return (ids >= ref[f"{sat}_id_start"]) & (ids <= ref[f"{sat}_id_end"])


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def load_reference(path: Path, tracer_policy: str = "original-stars",
                   lmc_core_kpc: float = 8.0, smc_core_kpc: float = 3.0) -> dict[str, object]:
    if tracer_policy not in ("original-stars", "legacy"):
        raise ValueError("stellar tracer policy must be original-stars or legacy")
    with h5py.File(path, "r") as f:
        if "MagellanicOrbitReference" not in f:
            raise RuntimeError(
                f"{path} lacks MagellanicOrbitReference. --reference-ic must be the "
                "Stage-2 IC written by add_lmc_smc_gaswake_orbit_v3.py, "
                "not the Stage-1 isolated MW host IC."
            )
        g = f["MagellanicOrbitReference"]
        required = (
            "TimeFromPresent_Gyr", "LMC_Position_kpc", "LMC_Velocity_kms",
            "SMC_Position_kpc", "SMC_Velocity_kms",
            "LMC_Current_PositionSigma_kpc", "LMC_Current_VelocitySigma_kms",
            "SMC_Current_PositionSigma_kpc", "SMC_Current_VelocitySigma_kms",
        )
        missing = [key for key in required if key not in g]
        if missing:
            raise RuntimeError(f"incomplete MagellanicOrbitReference: {missing}")
        ref = {
            "time_from_present_gyr": np.asarray(g["TimeFromPresent_Gyr"][:], dtype=float),
            "lmc_pos": np.asarray(g["LMC_Position_kpc"][:], dtype=float),
            "lmc_vel": np.asarray(g["LMC_Velocity_kms"][:], dtype=float),
            "smc_pos": np.asarray(g["SMC_Position_kpc"][:], dtype=float),
            "smc_vel": np.asarray(g["SMC_Velocity_kms"][:], dtype=float),
            "lmc_pos_sigma": np.asarray(g["LMC_Current_PositionSigma_kpc"][:], dtype=float),
            "lmc_vel_sigma": np.asarray(g["LMC_Current_VelocitySigma_kms"][:], dtype=float),
            "smc_pos_sigma": np.asarray(g["SMC_Current_PositionSigma_kpc"][:], dtype=float),
            "smc_vel_sigma": np.asarray(g["SMC_Current_VelocitySigma_kms"][:], dtype=float),
            "lookback_gyr": float(g.attrs["Lookback_Gyr"]),
            "insertion_time_code": float(g.attrs["InsertionTime_Code"]),
            "target_time_code": float(g.attrs["PresentTargetTime_Code"]),
            "unit_time_gyr": float(g.attrs.get("UnitTime_Gyr", UNIT_TIME_GYR_DEFAULT)),
            # Preserve these ranges separately for an exact legacy comparison.
            "lmc_id_start": int(g.attrs.get("LMC_STAR_ID_START", g.attrs["LMC_ID_START"])),
            "lmc_id_end": int(g.attrs.get("LMC_STAR_ID_END", g.attrs["LMC_ID_END"])),
            "smc_id_start": int(g.attrs.get("SMC_STAR_ID_START", g.attrs["SMC_ID_START"])),
            "smc_id_end": int(g.attrs.get("SMC_STAR_ID_END", g.attrs["SMC_ID_END"])),
            "lmc_dm_id_start": int(g.attrs["LMC_DM_ID_START"]),
            "lmc_dm_id_end": int(g.attrs["LMC_DM_ID_END"]),
            "smc_dm_id_start": int(g.attrs["SMC_DM_ID_START"]),
            "smc_dm_id_end": int(g.attrs["SMC_DM_ID_END"]),
            "lmc_origin_id_start": int(g.attrs["LMC_ID_START"]),
            "lmc_origin_id_end": int(g.attrs["LMC_ID_END"]),
            "smc_origin_id_start": int(g.attrs["SMC_ID_START"]),
            "smc_origin_id_end": int(g.attrs["SMC_ID_END"]),
            "mw_bh_id": int(g.attrs.get("MW_BH_ID", MW_BH_ID)),
            "lmc_mass_msun": float(g.attrs["LMC_TotalMass_Msun"]),
            "smc_mass_msun": float(g.attrs["SMC_TotalMass_Msun"]),
            "frame": str(g.attrs.get("CoordinateFrame", "K13 Galactocentric XYZ")),
            "potential": str(g.attrs.get("Potential", "Mou et al. 2023 Eq.4")),
            "reference_curve_includes_applied_shooting_offsets": bool(
                g.attrs.get("ReferenceCurveIncludesAppliedShootingOffsets", 0)),
        }
        model = f.get("MagellanicModelReference")
        ref["model_key"] = attribute_text(model.attrs.get("SATELLITE_MODEL_KEY", "")) if model is not None else ""
        ref["trial_label"] = attribute_text(model.attrs.get("TRIAL_LABEL", "")) if model is not None else ""
        structure = attribute_text(model.attrs.get("SATELLITE_STRUCTURE_MODEL", "")) if model is not None else ""
        provenance = f.get("MergeProvenance")
        pipeline_version = attribute_text(provenance.attrs.get("VERSION", "")) if provenance is not None else ""
        ref["tracer_policy"] = tracer_policy
        ref["reference_core_radii_kpc"] = {"lmc": float(lmc_core_kpc), "smc": float(smc_core_kpc)}
        ref["legacy_star_id_ranges"] = {
            sat: [int(g.attrs.get(f"{sat.upper()}_ALL_STAR_ID_START", ref[f"{sat}_id_start"])),
                  int(g.attrs.get(f"{sat.upper()}_ALL_STAR_ID_END", ref[f"{sat}_id_end"]))]
            for sat in ("lmc", "smc")}
        if tracer_policy == "legacy":
            for sat, bounds in ref["legacy_star_id_ranges"].items():
                ref[f"{sat}_id_start"], ref[f"{sat}_id_end"] = bounds
        ranges, methods = {}, {}
        recognized = structure == "tg19-independent-relax" and pipeline_version == "tg19-independent-relax-v1"
        for sat, block in OLD_STAR_BLOCKS.items():
            lo_key, hi_key = f"{sat.upper()}_OLD_STAR_ID_START", f"{sat.upper()}_OLD_STAR_ID_END"
            if lo_key in g.attrs or hi_key in g.attrs:
                if lo_key not in g.attrs or hi_key not in g.attrs:
                    raise RuntimeError(f"incomplete explicit original-star range for {sat}")
                bounds = [int(g.attrs[lo_key]), int(g.attrs[hi_key])]
                if not block[0] <= bounds[0] <= bounds[1] <= block[1]:
                    raise RuntimeError(f"explicit {sat} original-star IDs are outside the documented reserved block")
                ranges[sat], methods[sat] = bounds, "explicit OLD_STAR metadata"
            elif recognized:
                ranges[sat], methods[sat] = list(block), "recognized tg19 independent-relax reserved old-star block"
            else:
                bounds = ref["legacy_star_id_ranges"][sat]
                if block[0] <= bounds[0] <= bounds[1] <= block[1]:
                    ranges[sat], methods[sat] = bounds, "STAR metadata entirely within documented original-star block"
        ref["old_star_id_ranges"] = ranges if len(ranges) == 2 else None
        ref["tracer_selection_method"] = methods
        if tracer_policy == "original-stars" and not ref["old_star_id_ranges"]:
            raise RuntimeError("cannot identify original-star IDs for this reference; supply OLD_STAR metadata or explicitly use --stellar-tracers legacy")
        star = f["PartType4"]
        sid = np.asarray(star["ParticleIDs"][:], dtype=np.uint64)
        spos = np.asarray(star["Coordinates"][:], dtype=float)
        smass = np.asarray(star["Masses"][:], dtype=float)
        ref["identity_fields"] = list(name for name in IDENTITY_FIELDS if name in star)
        if ref["old_star_id_ranges"]:
            try:
                for sat, (lo, hi) in ref["old_star_id_ranges"].items():
                    old = (sid >= lo) & (sid <= hi)
                    keys = identity_keys(star, ref["identity_fields"], old)
                    if not len(keys):
                        raise RuntimeError(f"reference has no original {sat} star identities")
                    ref[f"{sat}_fixed_tracer_keys"] = keys
                if tracer_policy == "original-stars":
                    for sat, bounds in ref["old_star_id_ranges"].items():
                        ref[f"{sat}_id_start"], ref[f"{sat}_id_end"] = bounds
            except RuntimeError as exc:
                if tracer_policy == "original-stars":
                    raise
                ref["original_star_identity_error"] = str(exc)
        dm = f["PartType1"]
        did = np.asarray(dm["ParticleIDs"][:], dtype=np.uint64)
        dpos = np.asarray(dm["Coordinates"][:], dtype=float)
        dmass = np.asarray(dm["Masses"][:], dtype=float)
        for sat, aperture in (("lmc", 30.0), ("smc", 15.0)):
            star_mask = selected_tracer_mask(star, ref, sat)
            dm_mask = (did >= ref[f"{sat}_dm_id_start"]) & (did <= ref[f"{sat}_dm_id_end"])
            if not np.any(star_mask) or not np.any(dm_mask):
                raise RuntimeError(f"reference IC lacks exact {sat.upper()} tracer/DM IDs")
            if tracer_policy == "original-stars":
                center = satellite_stellar_state(f, ref[f"{sat}_id_start"], ref[f"{sat}_id_end"],
                    ref["reference_core_radii_kpc"][sat], mask=star_mask)[0]
            else:
                center = np.average(spos[star_mask], axis=0, weights=smass[star_mask])
            star_inside = star_mask & (np.linalg.norm(spos-center, axis=1) <= aperture)
            dm_inside = dm_mask & (np.linalg.norm(dpos-center, axis=1) <= aperture)
            ref[f"{sat}_initial_nstar"] = int(np.count_nonzero(star_mask))
            ref[f"{sat}_mass_aperture_kpc"] = aperture
            ref[f"{sat}_initial_aperture_mass_code"] = float(
                np.sum(smass[star_inside]) + np.sum(dmass[dm_inside])
            )
    t = np.asarray(ref["time_from_present_gyr"])
    if np.any(np.diff(t) <= 0) or t[0] >= 0 or abs(t[-1]) > 1.0e-10:
        raise RuntimeError("reference time grid must increase from negative lookback to zero")
    return ref


def shrinking_core_state(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                         core_radius_kpc: float, min_particles: int = 40
                         ) -> tuple[np.ndarray, np.ndarray, int, float]:
    if len(pos) < min_particles:
        raise RuntimeError(f"only {len(pos)} tracer particles; need >= {min_particles}")
    center = np.average(pos, axis=0, weights=mass)
    mask = np.ones(len(pos), dtype=bool)
    radius = float("inf")
    for _ in range(30):
        dist = np.linalg.norm(pos - center, axis=1)
        active = dist[mask]
        if active.size <= 2 * min_particles:
            break
        radius = float(np.quantile(active, 0.80))
        new_mask = dist <= radius
        if new_mask.sum() < min_particles:
            break
        mask = new_mask
        center = np.average(pos[mask], axis=0, weights=mass[mask])
        if radius <= core_radius_kpc:
            break
    dist = np.linalg.norm(pos - center, axis=1)
    final = dist <= core_radius_kpc
    if final.sum() < min_particles:
        final = mask
    center = np.average(pos[final], axis=0, weights=mass[final])
    velocity = np.average(vel[final], axis=0, weights=mass[final])
    rms_radius = math.sqrt(float(np.average(
        np.sum((pos[final] - center) ** 2, axis=1), weights=mass[final]
    )))
    return center, velocity, int(final.sum()), rms_radius


def host_state(f: h5py.File, ref: dict[str, object]) -> tuple[np.ndarray, np.ndarray, str]:
    if "PartType5" in f and len(f["PartType5/Coordinates"]) > 0:
        g = f["PartType5"]
        ids = np.asarray(g["ParticleIDs"][:], dtype=np.uint64)
        match = np.flatnonzero(ids == int(ref["mw_bh_id"]))
        idx = int(match[0]) if match.size else 0
        center = np.asarray(g["Coordinates"][idx], dtype=float)
        velocity = np.asarray(g["Velocities"][idx], dtype=float)
        method = "MW BH"
        if "PartType4" in f:
            s = f["PartType4"]
            sids = np.asarray(s["ParticleIDs"][:], dtype=np.uint64)
            mw = ~(
                ((sids >= ref["lmc_origin_id_start"]) & (sids <= ref["lmc_origin_id_end"]))
                | ((sids >= ref["smc_origin_id_start"]) & (sids <= ref["smc_origin_id_end"]))
            )
            spos = np.asarray(s["Coordinates"][:], dtype=float)
            core = mw & (np.linalg.norm(spos - center, axis=1) <= 10.0)
            if core.sum() >= 100:
                smass = np.asarray(s["Masses"][:], dtype=float)
                svel = np.asarray(s["Velocities"][:], dtype=float)
                velocity = np.average(svel[core], axis=0, weights=smass[core])
                method = "MW BH position + inner-star velocity"
        return center, velocity, method
    if "PartType4" not in f:
        raise RuntimeError("cannot locate MW center: PartType5 and PartType4 absent")
    g = f["PartType4"]
    ids = np.asarray(g["ParticleIDs"][:], dtype=np.uint64)
    mw = ~(
        ((ids >= ref["lmc_origin_id_start"]) & (ids <= ref["lmc_origin_id_end"]))
        | ((ids >= ref["smc_origin_id_start"]) & (ids <= ref["smc_origin_id_end"]))
    )
    pos = np.asarray(g["Coordinates"][:], dtype=float)[mw]
    vel = np.asarray(g["Velocities"][:], dtype=float)[mw]
    mass = np.asarray(g["Masses"][:], dtype=float)[mw]
    center, velocity, _, _ = shrinking_core_state(pos, vel, mass, 5.0, 100)
    return center, velocity, "shrinking-sphere MW stars"


def satellite_stellar_state(f: h5py.File, id_start: int, id_end: int,
                            core_radius_kpc: float, *, mask=None) -> tuple[np.ndarray, np.ndarray, int, float, int]:
    if "PartType4" not in f:
        raise RuntimeError("PartType4 absent; cannot track satellite stellar COM")
    g = f["PartType4"]
    ids = np.asarray(g["ParticleIDs"][:], dtype=np.uint64)
    if mask is None:
        mask = (ids >= id_start) & (ids <= id_end)
    if not np.any(mask):
        raise RuntimeError(f"no PartType4 IDs in [{id_start}, {id_end}]")
    pos = np.asarray(g["Coordinates"][:], dtype=float)[mask]
    vel = np.asarray(g["Velocities"][:], dtype=float)[mask]
    mass = np.asarray(g["Masses"][:], dtype=float)[mask]
    # 正式分辨率通常远大于 40；也允许极低分辨率轨道筛选，但至少需 10 颗。
    min_particles = min(40, max(10, len(pos)//4))
    center, velocity, ncore, rms = shrinking_core_state(
        pos, vel, mass, core_radius_kpc, min_particles
    )
    return center, velocity, ncore, rms, int(np.count_nonzero(mask))


def satellite_aperture_mass(
    f: h5py.File, ref: dict[str, object], sat: str,
    center: np.ndarray,
) -> float:
    aperture = float(ref[f"{sat}_mass_aperture_kpc"])
    total = 0.0
    for ptype, start_key, end_key in (
        (1, f"{sat}_dm_id_start", f"{sat}_dm_id_end"),
        (4, f"{sat}_id_start", f"{sat}_id_end"),
    ):
        g = f[f"PartType{ptype}"]
        ids = np.asarray(g["ParticleIDs"][:], dtype=np.uint64)
        mask = (selected_tracer_mask(g, ref, sat) if ptype == 4
                else (ids >= ref[start_key]) & (ids <= ref[end_key]))
        pos = np.asarray(g["Coordinates"][:], dtype=float)
        mass = np.asarray(g["Masses"][:], dtype=float)
        inside = mask & (np.linalg.norm(pos-center, axis=1) <= aperture)
        total += float(np.sum(mass[inside]))
    return total


def gas_timestep_diagnostics(f: h5py.File, threshold: float) -> tuple[float, int, float, str]:
    if "PartType0" not in f:
        return float("nan"), 0, float("nan"), "PartType0 absent"
    g = f["PartType0"]
    for name in ("TimeStep", "Timestep", "TimestepSize"):
        if name in g:
            dt = np.ravel(np.asarray(g[name][:], dtype=float))
            finite = np.isfinite(dt) & (dt > 0.0)
            if not np.any(finite):
                return float("nan"), 0, float("nan"), name
            short = finite & (dt <= threshold)
            return float(np.min(dt[finite])), int(np.count_nonzero(short)), \
                float(np.count_nonzero(short)/np.count_nonzero(finite)), name
    return float("nan"), 0, float("nan"), "missing"


def derived_orbit(pos: np.ndarray, vel: np.ndarray) -> dict[str, float]:
    radius = float(np.linalg.norm(pos))
    speed = float(np.linalg.norm(vel))
    vr = float(np.dot(pos, vel) / radius) if radius > 0 else float("nan")
    vt = math.sqrt(max(speed * speed - vr * vr, 0.0))
    pole = np.cross(pos, vel)
    pole_norm = float(np.linalg.norm(pole))
    if pole_norm > 0:
        pole /= pole_norm
    else:
        pole[:] = np.nan
    return {
        "r_kpc": radius, "speed_kms": speed, "vr_kms": vr, "vt_kms": vt,
        "pole_x": float(pole[0]), "pole_y": float(pole[1]), "pole_z": float(pole[2]),
    }


def reference_at_elapsed(ref: dict[str, object], elapsed_gyr: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tfp = elapsed_gyr - float(ref["lookback_gyr"])
    grid = np.asarray(ref["time_from_present_gyr"])
    if tfp < grid[0] - 1.0e-9 or tfp > grid[-1] + 1.0e-9:
        nan = np.full(3, np.nan)
        return nan, nan.copy(), nan.copy(), nan.copy()
    tfp = float(np.clip(tfp, grid[0], grid[-1]))

    def interp(array):
        arr = np.asarray(array)
        return np.array([np.interp(tfp, grid, arr[:, j]) for j in range(3)])

    return interp(ref["lmc_pos"]), interp(ref["lmc_vel"]), \
        interp(ref["smc_pos"]), interp(ref["smc_vel"])


def inspect_snapshot(path: Path, ref: dict[str, object],
                     lmc_core_kpc: float, smc_core_kpc: float,
                     short_dt_threshold: float) -> dict[str, object]:
    with h5py.File(path, "r") as f:
        if ref.get("tracer_policy") == "original-stars":
            for sat, core in (("lmc", lmc_core_kpc), ("smc", smc_core_kpc)):
                if core != ref["reference_core_radii_kpc"][sat]:
                    raise RuntimeError("core radius differs from reference aperture baseline; pass the same core radii to load_reference")
        time_code = float(f["Header"].attrs["Time"])
        hp, hv, host_method = host_state(f, ref)
        lp_abs, lv_abs, nl, rl, nlt = satellite_stellar_state(
            f, ref["lmc_id_start"], ref["lmc_id_end"], lmc_core_kpc,
            mask=selected_tracer_mask(f["PartType4"], ref, "lmc")
        )
        sp_abs, sv_abs, ns, rs, nst = satellite_stellar_state(
            f, ref["smc_id_start"], ref["smc_id_end"], smc_core_kpc,
            mask=selected_tracer_mask(f["PartType4"], ref, "smc")
        )
        lmc_mass = satellite_aperture_mass(f, ref, "lmc", lp_abs)
        smc_mass = satellite_aperture_mass(f, ref, "smc", sp_abs)
        gas_dt_min, short_n, short_fraction, timestep_field = gas_timestep_diagnostics(
            f, short_dt_threshold
        )
    lp, lv = lp_abs - hp, lv_abs - hv
    sp, sv = sp_abs - hp, sv_abs - hv
    elapsed = (time_code - float(ref["insertion_time_code"])) * float(ref["unit_time_gyr"])
    lpr, lvr, spr, svr = reference_at_elapsed(ref, elapsed)
    ld, sd = derived_orbit(lp, lv), derived_orbit(sp, sv)
    rp, rv = sp - lp, sv - lv
    rd = derived_orbit(rp, rv)
    row: dict[str, object] = {
        "file": str(path), "time_code": time_code, "elapsed_gyr": elapsed,
        "stellar_tracer_policy": ref.get("tracer_policy", "legacy"),
        "time_from_present_gyr": elapsed - float(ref["lookback_gyr"]),
        "host_center_method": host_method,
        "host_x_kpc": hp[0], "host_y_kpc": hp[1], "host_z_kpc": hp[2],
        "host_vx_kms": hv[0], "host_vy_kms": hv[1], "host_vz_kms": hv[2],
        "lmc_core_nstar": nl, "lmc_core_rms_kpc": rl,
        "smc_core_nstar": ns, "smc_core_rms_kpc": rs,
        "lmc_total_original_nstar": nlt,
        "smc_total_original_nstar": nst,
        "lmc_tracer_fraction": nlt/float(ref["lmc_initial_nstar"]),
        "smc_tracer_fraction": nst/float(ref["smc_initial_nstar"]),
        "lmc_mass_r30_retention": lmc_mass/float(ref["lmc_initial_aperture_mass_code"]),
        "smc_mass_r15_retention": smc_mass/float(ref["smc_initial_aperture_mass_code"]),
        "gas_timestep_field": timestep_field,
        "gas_dt_min_code": gas_dt_min,
        "gas_short_dt_count": short_n,
        "gas_short_dt_fraction": short_fraction,
    }
    for name, arr in (
        ("lmc", np.concatenate((lp, lv))), ("smc", np.concatenate((sp, sv))),
        ("lmc_ref", np.concatenate((lpr, lvr))), ("smc_ref", np.concatenate((spr, svr))),
    ):
        for key, value in zip(("x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms"), arr):
            row[f"{name}_{key}"] = float(value)
    for prefix, values in (("lmc", ld), ("smc", sd), ("smc_lmc", rd)):
        for key, value in values.items():
            row[f"{prefix}_{key}"] = value
    row["lmc_model_dpos_kpc"] = float(np.linalg.norm(lp - lpr))
    row["lmc_model_dvel_kms"] = float(np.linalg.norm(lv - lvr))
    row["smc_model_dpos_kpc"] = float(np.linalg.norm(sp - spr))
    row["smc_model_dvel_kms"] = float(np.linalg.norm(sv - svr))
    return row


def interpolate_simulation_at_target(rows: list[dict[str, object]], target_code: float
                                     ) -> tuple[dict[str, np.ndarray], str, list[str]]:
    rows = sorted(rows, key=lambda r: float(r["time_code"]))
    times = np.array([float(r["time_code"]) for r in rows])
    warnings: list[str] = []
    components = ("x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms")
    result: dict[str, np.ndarray] = {}
    tolerance = 1.0e-8 * max(1.0, abs(target_code))
    exact = np.flatnonzero(np.abs(times-target_code) <= tolerance)
    if exact.size:
        idx = int(exact[np.argmin(np.abs(times[exact]-target_code))])
        method = "exact target snapshot"
        for sat in ("lmc", "smc"):
            result[sat] = np.array([float(rows[idx][f"{sat}_{c}"]) for c in components])
    elif times[0] - tolerance <= target_code <= times[-1] + tolerance \
            and len(np.unique(times)) >= 2:
        target_for_interp = float(np.clip(target_code, times[0], times[-1]))
        method = "linear interpolation between bracketing snapshots"
        for sat in ("lmc", "smc"):
            result[sat] = np.array([
                np.interp(target_for_interp, times,
                          [float(r[f"{sat}_{comp}"]) for r in rows])
                for comp in components
            ])
    else:
        idx = int(np.argmin(np.abs(times - target_code)))
        method = "nearest snapshot (target not bracketed)"
        warnings.append(
            f"present target Time={target_code:.8f} is outside snapshot range "
            f"[{times[0]:.8f}, {times[-1]:.8f}]"
        )
        for sat in ("lmc", "smc"):
            result[sat] = np.array([float(rows[idx][f"{sat}_{c}"]) for c in components])
    return result, method, warnings


def target_snapshot_sources(rows: list[dict[str, object]], target_code: float) -> list[dict[str, object]]:
    """Report the actual endpoint input files, with interpolation weights."""
    ordered = sorted(rows, key=lambda row: float(row["time_code"]))
    times = np.asarray([float(row["time_code"]) for row in ordered])
    tolerance = 1.0e-8 * max(1.0, abs(target_code))
    exact = np.flatnonzero(np.abs(times-target_code) <= tolerance)
    if exact.size:
        selected = [(int(exact[np.argmin(np.abs(times[exact]-target_code))]), 1.0)]
    elif len(times) >= 2 and times[0] < target_code < times[-1]:
        right = int(np.searchsorted(times, target_code, side="right"))
        left = right-1
        fraction = (target_code-times[left])/(times[right]-times[left])
        selected = [(left, float(1.0-fraction)), (right, float(fraction))]
    else:
        selected = [(int(np.argmin(np.abs(times-target_code))), 1.0)]
    return [{"file": str(Path(ordered[index]["file"]).resolve()),
             "time_code": float(times[index]), "weight": weight}
            for index, weight in selected]


def stellar_tracer_audit(path: Path, ref: dict[str, object], lmc_core_kpc: float,
                         smc_core_kpc: float) -> dict[str, object]:
    """Compare the frozen original-star sample with the actual legacy ranges."""
    ranges = ref.get("old_star_id_ranges")
    if not ranges or ref.get("original_star_identity_error"):
        return {"status": "not_available", "reason": "unrecognized pipeline; original-star ID ranges are not inferred"}
    result: dict[str, object] = {"status": "ok", "file": str(Path(path).resolve()), "satellites": {}}
    with h5py.File(path, "r") as f:
        result["time_code"] = float(f["Header"].attrs["Time"])
        ids = np.asarray(f["PartType4/ParticleIDs"][:], dtype=np.uint64)
        for sat, core in (("lmc", lmc_core_kpc), ("smc", smc_core_kpc)):
            old_lo, old_hi = ranges[sat]
            legacy_lo, legacy_hi = ref["legacy_star_id_ranges"][sat]
            legacy_mask = (ids >= legacy_lo) & (ids <= legacy_hi)
            try:
                old_mask = fixed_tracer_mask(f["PartType4"], ref, sat)
            except RuntimeError as exc:
                result["satellites"][sat] = {"status": "unavailable", "reason": str(exc)}
                result["status"] = "partial"
                continue
            record = {
                "legacy_id_range": [legacy_lo, legacy_hi], "original_star_id_range": [old_lo, old_hi],
                "legacy_count": int(legacy_mask.sum()), "original_star_count": int(old_mask.sum()),
                "legacy_non_original_star_count": int(np.count_nonzero(legacy_mask & ~old_mask)),
            }
            try:
                legacy = satellite_stellar_state(f, legacy_lo, legacy_hi, core)
                original = satellite_stellar_state(f, old_lo, old_hi, core, mask=old_mask)
                delta = np.concatenate((original[0]-legacy[0], original[1]-legacy[1]))
                sigma = np.concatenate((ref[f"{sat}_pos_sigma"], ref[f"{sat}_vel_sigma"]))
                record.update(status="ok", original_minus_legacy_6d=delta,
                    delta_over_observational_sigma=delta/sigma,
                    delta_position_norm_kpc=float(np.linalg.norm(delta[:3])),
                    delta_velocity_norm_kms=float(np.linalg.norm(delta[3:])),
                    legacy_core_nstar=legacy[2], original_core_nstar=original[2])
            except (RuntimeError, ValueError, KeyError) as exc:
                record.update(status="unavailable", reason=str(exc))
                result["status"] = "partial"
            result["satellites"][sat] = record
    return result


def build_stellar_tracer_audit(reference_path: Path, rows: list[dict[str, object]],
                              ref: dict[str, object], lmc_core_kpc: float,
                              smc_core_kpc: float) -> dict[str, object]:
    if not ref.get("old_star_id_ranges") or ref.get("original_star_identity_error"):
        return {"status": "not_available", "reason": "unrecognized pipeline; original-star ID ranges are not inferred"}
    sources = target_snapshot_sources(rows, float(ref["target_time_code"]))
    snapshots = []
    for source in sources:
        report = stellar_tracer_audit(Path(source["file"]), ref, lmc_core_kpc, smc_core_kpc)
        report["interpolation_weight"] = source["weight"]
        snapshots.append(report)
    endpoint = {}
    for sat in ("lmc", "smc"):
        if all(report.get("satellites", {}).get(sat, {}).get("status") == "ok" for report in snapshots):
            delta = sum(report["interpolation_weight"] * np.asarray(report["satellites"][sat]["original_minus_legacy_6d"])
                        for report in snapshots)
            sigma = np.concatenate((ref[f"{sat}_pos_sigma"], ref[f"{sat}_vel_sigma"]))
            endpoint[sat] = {"status": "ok", "original_minus_legacy_6d": delta,
                "delta_over_observational_sigma": delta/sigma,
                "delta_position_norm_kpc": float(np.linalg.norm(delta[:3])),
                "delta_velocity_norm_kms": float(np.linalg.norm(delta[3:]))}
        else:
            endpoint[sat] = {"status": "unavailable"}
    return {"status": "available", "baseline_measurement_changed": False,
            "csv_tracer_policy": ref.get("tracer_policy", "legacy"),
            "note": "Audit only: fixed reference original-star identities minus legacy evolving ID-range sample; this audit does not alter the selected CSV policy.",
            "reference_ic": stellar_tracer_audit(reference_path, ref, lmc_core_kpc, smc_core_kpc),
            "target_snapshot_audits": snapshots, "target_old_minus_legacy": endpoint}


def interpolate_scalar_at_target(rows: list[dict[str, object]], key: str,
                                 target_code: float) -> float:
    times = np.asarray([float(r["time_code"]) for r in rows])
    values = np.asarray([float(r[key]) for r in rows])
    finite = np.isfinite(times) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    order = np.argsort(times[finite])
    t, v = times[finite][order], values[finite][order]
    if len(t) == 1:
        return float(v[0])
    return float(np.interp(np.clip(target_code, t[0], t[-1]), t, v))


def evaluate_present(rows: list[dict[str, object]], ref: dict[str, object],
                     nsigma: float, require_scalar_velocity: bool,
                     min_lmc_retention: float, min_smc_retention: float,
                     max_short_dt_fraction: float, orbit_only: bool) -> dict[str, object]:
    sim, method, warnings = interpolate_simulation_at_target(
        rows, float(ref["target_time_code"])
    )
    lmc_obs = np.concatenate((np.asarray(ref["lmc_pos"])[-1], np.asarray(ref["lmc_vel"])[-1]))
    smc_obs = np.concatenate((np.asarray(ref["smc_pos"])[-1], np.asarray(ref["smc_vel"])[-1]))
    lmc_sigma = np.concatenate((ref["lmc_pos_sigma"], ref["lmc_vel_sigma"]))
    smc_sigma = np.concatenate((ref["smc_pos_sigma"], ref["smc_vel_sigma"]))
    lmc_res = sim["lmc"] - lmc_obs
    smc_res = sim["smc"] - smc_obs
    lmc_z = lmc_res / lmc_sigma
    smc_z = smc_res / smc_sigma

    lmc_derived = derived_orbit(sim["lmc"][:3], sim["lmc"][3:])
    smc_derived = derived_orbit(sim["smc"][:3], sim["smc"][3:])
    rel_derived = derived_orbit(sim["smc"][:3] - sim["lmc"][:3],
                                sim["smc"][3:] - sim["lmc"][3:])
    lmc_6d_pass = bool(np.all(np.abs(lmc_z) <= nsigma))
    smc_6d_pass = bool(np.all(np.abs(smc_z) <= nsigma))
    rel_speed_z = ((rel_derived["speed_kms"] - K13_RELATIVE_SPEED_KMS)
                   / K13_RELATIVE_SPEED_SIGMA_KMS)
    lmc_speed_z = ((lmc_derived["speed_kms"] - K13_LMC_SPEED_KMS)
                   / K13_LMC_SPEED_SIGMA_KMS)
    smc_speed_z = ((smc_derived["speed_kms"] - K13_SMC_SPEED_KMS)
                   / K13_SMC_SPEED_SIGMA_KMS)
    scalar_pass = bool(max(abs(rel_speed_z), abs(lmc_speed_z), abs(smc_speed_z)) <= nsigma)
    covered = "not bracketed" not in method
    lmc_retention = interpolate_scalar_at_target(
        rows, "lmc_mass_r30_retention", float(ref["target_time_code"])
    )
    smc_retention = interpolate_scalar_at_target(
        rows, "smc_mass_r15_retention", float(ref["target_time_code"])
    )
    lmc_tracer = interpolate_scalar_at_target(
        rows, "lmc_tracer_fraction", float(ref["target_time_code"])
    )
    smc_tracer = interpolate_scalar_at_target(
        rows, "smc_tracer_fraction", float(ref["target_time_code"])
    )
    short_fraction = interpolate_scalar_at_target(
        rows, "gas_short_dt_fraction", float(ref["target_time_code"])
    )
    structure_pass = bool(
        lmc_retention >= min_lmc_retention
        and smc_retention >= min_smc_retention
        and lmc_tracer >= 0.999 and smc_tracer >= 0.999
    )
    timestep_available = bool(np.isfinite(short_fraction))
    timestep_pass = bool(short_fraction <= max_short_dt_fraction) if timestep_available else None
    if not timestep_available:
        warnings.append(
            "snapshot lacks PartType0 TimeStep/Timestep/TimestepSize; "
            "compile OUTPUT_TIMESTEP before the production acceptance run"
        )
    science_quality_pass = bool(structure_pass and (timestep_pass is not False))
    return {
        "evaluation_method": method,
        "warnings": warnings,
        "target_time_code": ref["target_time_code"],
        "target_elapsed_gyr": ref["lookback_gyr"],
        "nsigma_threshold": nsigma,
        "diagonal_scaled_residual_l2": float(np.linalg.norm(np.concatenate((lmc_z, smc_z)))),
        "residual_score_note": "Comparison score under the selected tracer policy and diagonal tolerances; not a full-covariance confidence statistic.",
        "lmc_simulated_6d": sim["lmc"],
        "lmc_observed_6d": lmc_obs,
        "lmc_sigma_6d": lmc_sigma,
        "lmc_residual_6d": lmc_res,
        "lmc_normalized_residual_6d": lmc_z,
        "lmc_6d_pass": lmc_6d_pass,
        "smc_simulated_6d": sim["smc"],
        "smc_observed_6d": smc_obs,
        "smc_sigma_6d": smc_sigma,
        "smc_residual_6d": smc_res,
        "smc_normalized_residual_6d": smc_z,
        "smc_6d_pass": smc_6d_pass,
        "lmc_derived": lmc_derived,
        "smc_derived": smc_derived,
        "smc_lmc_derived": rel_derived,
        "lmc_speed_z": lmc_speed_z,
        "smc_speed_z": smc_speed_z,
        "smc_lmc_speed_z": rel_speed_z,
        "scalar_velocity_pass": scalar_pass,
        "target_time_bracketed": covered,
        # K13 明确指出 Galactocentric 速度分量误差高度相关；没有协方差矩阵时，
        # 由同一六个速度分量再构造的标量速度只作 REVIEW，默认不重复硬否决。
        "scalar_velocity_required": bool(require_scalar_velocity),
        "lmc_mass_r30_retention": lmc_retention,
        "smc_mass_r15_retention": smc_retention,
        "lmc_minimum_retention": min_lmc_retention,
        "smc_minimum_retention": min_smc_retention,
        "lmc_tracer_fraction": lmc_tracer,
        "smc_tracer_fraction": smc_tracer,
        "satellite_structure_pass": structure_pass,
        "gas_short_dt_fraction": short_fraction,
        "gas_short_dt_fraction_maximum": max_short_dt_fraction,
        "gas_timestep_available": timestep_available,
        "gas_timestep_pass": timestep_pass,
        "science_quality_required": not orbit_only,
        "science_quality_pass": science_quality_pass,
        "overall_pass": bool(
            covered and lmc_6d_pass and smc_6d_pass
            and (scalar_pass or not require_scalar_velocity)
            and (science_quality_pass or orbit_only)
        ),
    }


def model_tracking_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    out = {}
    for sat in ("lmc", "smc"):
        for kind, unit in (("dpos", "kpc"), ("dvel", "kms")):
            vals = np.array([float(r[f"{sat}_model_{kind}_{unit}"]) for r in rows])
            vals = vals[np.isfinite(vals)]
            out[f"{sat}_model_{kind}_rms_{unit}"] = (
                float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan")
            )
            out[f"{sat}_model_{kind}_max_{unit}"] = (
                float(np.max(vals)) if vals.size else float("nan")
            )
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_present_table(evaluation: dict[str, object], path: Path) -> None:
    components = ("x", "y", "z", "vx", "vy", "vz")
    units = ("kpc", "kpc", "kpc", "km/s", "km/s", "km/s")
    rows = []
    for sat in ("lmc", "smc"):
        simulated = np.asarray(evaluation[f"{sat}_simulated_6d"])
        observed = np.asarray(evaluation[f"{sat}_observed_6d"])
        residual = np.asarray(evaluation[f"{sat}_residual_6d"])
        zscore = np.asarray(evaluation[f"{sat}_normalized_residual_6d"])
        sigma = np.asarray(evaluation[f"{sat}_sigma_6d"])
        for j, (component, unit) in enumerate(zip(components, units)):
            rows.append({
                "satellite": sat.upper(), "component": component, "unit": unit,
                "simulated": simulated[j], "observed": observed[j],
                "residual": residual[j], "sigma": sigma[j], "zscore": zscore[j],
                "pass": bool(abs(zscore[j]) <= float(evaluation["nsigma_threshold"])),
            })
    write_csv(rows, path)


def write_present_provenance(path: Path, reference_path: Path, ref: dict[str, object],
                             rows: list[dict[str, object]], evaluation: dict[str, object],
                             lmc_core_kpc: float, smc_core_kpc: float) -> Path:
    sidecar = path.with_suffix(".provenance.json")
    payload = {
        "schema_version": 1,
        "reference_ic": str(reference_path.resolve()),
        "reference_ic_sha256": sha256_file(reference_path),
        "model_key": ref.get("model_key", ""),
        "trial_label": ref.get("trial_label", ""),
        "residual_csv": str(path.resolve()),
        "residual_csv_sha256": sha256_file(path),
        "target_time_code": float(ref["target_time_code"]),
        "unit_time_gyr": float(ref["unit_time_gyr"]),
        "target_covered": bool(evaluation["target_time_bracketed"]),
        "shooting_usable": bool(evaluation["target_time_bracketed"]),
        "evaluation_method": evaluation["evaluation_method"],
        "target_snapshot_sources": target_snapshot_sources(rows, float(ref["target_time_code"])),
        "measurement_signature": measurement_signature(lmc_core_kpc, smc_core_kpc,
            ref.get("tracer_policy", "legacy"), ref.get("identity_fields")),
        "stellar_tracer_policy": ref.get("tracer_policy", "legacy"),
        "tracer_selection_method": ref.get("tracer_selection_method"),
        "legacy_star_id_ranges": ref.get("legacy_star_id_ranges"),
        "reference_tracer_counts": {sat: ref[f"{sat}_initial_nstar"] for sat in ("lmc", "smc")},
        "stellar_tracer_id_ranges": {sat: [ref[f"{sat}_id_start"], ref[f"{sat}_id_end"]]
                                     for sat in ("lmc", "smc")},
        "coordinate_order": ["lmc_x", "lmc_y", "lmc_z", "lmc_vx", "lmc_vy", "lmc_vz",
                             "smc_x", "smc_y", "smc_z", "smc_vx", "smc_vy", "smc_vz"],
    }
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sidecar


def plot_dashboard(rows: list[dict[str, object]], ref: dict[str, object],
                   evaluation: dict[str, object], path: Path) -> plt.Figure:
    rows = sorted(rows, key=lambda r: float(r["elapsed_gyr"]))
    elapsed = np.array([float(r["elapsed_gyr"]) for r in rows])
    ref_elapsed = np.asarray(ref["time_from_present_gyr"]) + float(ref["lookback_gyr"])
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), dpi=160, sharex=False)
    time_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]]
    for time_ax in time_axes[1:]:
        time_ax.sharex(time_axes[0])

    def phase_panel(ax, sat: str, kind: str, ref_array: np.ndarray, ylabel: str, title: str):
        suffixes = ("x_kpc", "y_kpc", "z_kpc") if kind == "pos" else \
            ("vx_kms", "vy_kms", "vz_kms")
        for j, (lab, color, suffix) in enumerate(zip(labels, colors, suffixes)):
            values = [float(r[f"{sat}_{suffix}"]) for r in rows]
            ax.plot(elapsed, values, color=color, lw=1.8, label=f"{lab} simulation")
            ax.plot(ref_elapsed, ref_array[:, j], color=color, lw=1.1, ls="--",
                    alpha=0.8, label=f"{lab} reference")
        ax.axvline(ref["lookback_gyr"], color="k", ls=":", lw=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    phase_panel(axes[0, 0], "lmc", "pos", np.asarray(ref["lmc_pos"]),
                "Position [kpc]", "LMC position: full 3D")
    phase_panel(axes[0, 1], "lmc", "vel", np.asarray(ref["lmc_vel"]),
                "Velocity [km/s]", "LMC velocity: full 3D")
    phase_panel(axes[1, 0], "smc", "pos", np.asarray(ref["smc_pos"]),
                "Position [kpc]", "SMC position: full 3D")
    phase_panel(axes[1, 1], "smc", "vel", np.asarray(ref["smc_vel"]),
                "Velocity [km/s]", "SMC velocity: full 3D")

    ax = axes[2, 0]
    for key, label, color in (
        ("lmc_r_kpc", "MW-LMC", "tab:blue"),
        ("smc_r_kpc", "MW-SMC", "tab:orange"),
        ("smc_lmc_r_kpc", "LMC-SMC", "tab:purple"),
    ):
        ax.plot(elapsed, [float(r[key]) for r in rows], label=label, color=color, lw=2)
    ref_lmc_r = np.linalg.norm(ref["lmc_pos"], axis=1)
    ref_smc_r = np.linalg.norm(ref["smc_pos"], axis=1)
    ref_rel_r = np.linalg.norm(np.asarray(ref["smc_pos"]) - np.asarray(ref["lmc_pos"]), axis=1)
    for arr, color in ((ref_lmc_r, "tab:blue"), (ref_smc_r, "tab:orange"),
                       (ref_rel_r, "tab:purple")):
        ax.plot(ref_elapsed, arr, color=color, ls="--", lw=1.1)
    ax.axvline(ref["lookback_gyr"], color="k", ls=":", lw=1)
    ax.set(xlabel="Physical time since insertion [Gyr]", ylabel="Separation [kpc]",
           title="Orbital separations")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[2, 1]
    x = np.arange(6)
    width = 0.36
    ax.bar(x - width / 2, evaluation["lmc_normalized_residual_6d"], width,
           label="LMC", color="tab:blue")
    ax.bar(x + width / 2, evaluation["smc_normalized_residual_6d"], width,
           label="SMC", color="tab:orange")
    nsigma = float(evaluation["nsigma_threshold"])
    ax.axhline(nsigma, color="red", ls="--", lw=1)
    ax.axhline(-nsigma, color="red", ls="--", lw=1)
    ax.set_xticks(x, ("x", "y", "z", "vx", "vy", "vz"))
    ax.set(xlabel="6D component", ylabel="Residual / observational sigma",
           title=f"Present-epoch 6D residuals (pass: |z| <= {nsigma:g})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    axes[2, 1].set_xlim(-0.6, 5.6)
    time_min = min(0., float(np.min(elapsed)), float(np.min(ref_elapsed)))
    time_max = max(float(ref["lookback_gyr"]), float(np.max(elapsed)), float(np.max(ref_elapsed)))
    margin = max(time_max-time_min, 1e-6)*.025
    time_axes[0].set_xlim(time_min-margin, time_max+margin)
    for time_ax in time_axes:
        time_ax.set_xlabel("Physical time since insertion [Gyr]")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, 0.995))
    status = "PASS" if evaluation["overall_pass"] else "FAIL / RETUNE ORBIT"
    fig.suptitle(
        f"Magellanic 6D validation - {status}\n"
        f"frame: {ref['frame']} | reference: {ref['potential']}",
        y=1.035, fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate LMC/SMC 6D phase space")
    p.add_argument("--snapshot-dir", default="./output")
    p.add_argument("--pattern", default="snapshot_*.hdf5")
    p.add_argument("--reference-ic", required=True,
                   help="Stage-2 IC created by add_lmc_smc_gaswake_orbit_v3.py")
    p.add_argument("--stellar-tracers", choices=("original-stars", "legacy"), default="original-stars",
                   help="默认冻结 reference IC 的原生老恒星复合身份；legacy 重现旧版动态 ID 范围，仅供比较。更换策略须重新诊断所有相关 trial。")
    p.add_argument("--output-dir", default=None,
                   help="default: SNAPSHOT_DIR/magellanic_validation")
    p.add_argument("--lmc-core-radius-kpc", type=float, default=8.0)
    p.add_argument("--smc-core-radius-kpc", type=float, default=3.0)
    p.add_argument("--nsigma", type=float, default=3.0,
                   help="per-component present-day 6D threshold")
    p.add_argument(
        "--unit-time-gyr", type=float, default=None,
        help="可选：覆盖 reference IC 的代码时间单位 [Gyr]；通常不要设置。",
    )
    p.add_argument(
        "--require-scalar-velocity", action="store_true",
        help="同时把 |v_LMC|、|v_SMC|、|v_SMC-LMC| 的标量检查纳入硬判据。",
    )
    p.add_argument("--short-dt-threshold", type=float, default=1.0e-6,
                   help="GIZMO code units；短气体时间步阈值。")
    p.add_argument("--max-short-dt-fraction", type=float, default=1.0e-3,
                   help="今天时刻允许的短时间步气体比例。")
    p.add_argument("--min-lmc-mass-retention", type=float, default=0.50,
                   help="LMC 原生 DM+star 在 30 kpc 固定孔径内的最低质量保留率。")
    p.add_argument("--min-smc-mass-retention", type=float, default=0.30,
                   help="SMC 原生 DM+star 在 15 kpc 固定孔径内的最低质量保留率。")
    p.add_argument("--orbit-only", action="store_true",
                   help="只硬判 6D；结构/短时间步仍报告但不纳入 overall。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.nsigma <= 0.0:
        raise ValueError("--nsigma 必须大于 0。")
    if args.lmc_core_radius_kpc <= 0.0 or args.smc_core_radius_kpc <= 0.0:
        raise ValueError("卫星 core radius 必须大于 0。")
    if args.short_dt_threshold <= 0.0 or args.max_short_dt_fraction < 0.0:
        raise ValueError("时间步阈值必须为正，允许比例必须非负。")
    if not (0.0 <= args.min_lmc_mass_retention <= 1.5) or not \
            (0.0 <= args.min_smc_mass_retention <= 1.5):
        raise ValueError("质量保留率阈值必须在 [0,1.5]。")
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    reference_path = Path(args.reference_ic).expanduser().resolve()
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                  else snapshot_dir / "magellanic_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = load_reference(reference_path, args.stellar_tracers,
                         args.lmc_core_radius_kpc, args.smc_core_radius_kpc)
    print(f"恒星示踪策略: {ref['tracer_policy']}；身份字段: {ref['identity_fields']}")
    if not ref["reference_curve_includes_applied_shooting_offsets"]:
        print("说明：逐快照 d6 比较的是 nominal 静态参考轨道，未包含已施加的 shooting 初态修正。")
        print("      前期 d6 较大不等于赋值失败；shooting 收敛请比较目标时刻的 6D 残差。")
    if args.unit_time_gyr is not None:
        if not np.isfinite(args.unit_time_gyr) or args.unit_time_gyr <= 0.0:
            raise ValueError("--unit-time-gyr 必须为有限正数。")
        ref["unit_time_gyr"] = float(args.unit_time_gyr)
    files = [p for p in snapshot_dir.glob(args.pattern)
             if not any(tag in p.name for tag in (".hsml.", ".kdtree", ".ewah", ".tmp."))]
    if not files:
        raise FileNotFoundError(f"no snapshots matching {args.pattern} in {snapshot_dir}")

    timed_files = []
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                timed_files.append((float(f["Header"].attrs["Time"]), path))
        except Exception as exc:
            print(f"SKIP {path.name}: cannot read time ({exc})")
    timed_files.sort(key=lambda item: item[0])

    rows: list[dict[str, object]] = []
    failures = []
    for _, path in timed_files:
        try:
            row = inspect_snapshot(path, ref, args.lmc_core_radius_kpc,
                                   args.smc_core_radius_kpc, args.short_dt_threshold)
            rows.append(row)
            print(
                f"{path.name}: elapsed={row['elapsed_gyr']:.4f} Gyr, "
                f"LMC d6=({row['lmc_model_dpos_kpc']:.3f} kpc, "
                f"{row['lmc_model_dvel_kms']:.3f} km/s), "
                f"SMC d6=({row['smc_model_dpos_kpc']:.3f} kpc, "
                f"{row['smc_model_dvel_kms']:.3f} km/s)"
            )
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
            print(f"SKIP {path.name}: {exc}")
    if not rows:
        raise RuntimeError("no valid satellite snapshots were extracted")

    csv_path = output_dir / "magellanic_6d_history.csv"
    json_path = output_dir / "magellanic_validation_summary.json"
    png_path = output_dir / "magellanic_6d_dashboard.png"
    present_csv_path = output_dir / "magellanic_present_6d.csv"
    write_csv(rows, csv_path)
    evaluation = evaluate_present(
        rows, ref, args.nsigma, args.require_scalar_velocity,
        args.min_lmc_mass_retention, args.min_smc_mass_retention,
        args.max_short_dt_fraction, args.orbit_only,
    )
    write_present_table(evaluation, present_csv_path)
    provenance_path = write_present_provenance(present_csv_path, reference_path, ref, rows,
        evaluation, args.lmc_core_radius_kpc, args.smc_core_radius_kpc)
    tracer_audit = build_stellar_tracer_audit(reference_path, rows, ref,
        args.lmc_core_radius_kpc, args.smc_core_radius_kpc)
    summary = {
        "validator_revision": "v3.2-fixed-original-star-identities",
        "reference_ic": str(reference_path),
        "snapshot_count": len(rows),
        "failed_snapshot_count": len(failures),
        "failed_snapshots": failures,
        "coordinate_order": ["x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms"],
        "unit_time_gyr_used": ref["unit_time_gyr"],
        "stellar_tracer_id_ranges": {
            "lmc": [ref["lmc_id_start"], ref["lmc_id_end"]],
            "smc": [ref["smc_id_start"], ref["smc_id_end"]],
        },
        "satellite_origin_id_ranges_excluded_from_mw": {
            "lmc": [ref["lmc_origin_id_start"], ref["lmc_origin_id_end"]],
            "smc": [ref["smc_origin_id_start"], ref["smc_origin_id_end"]],
        },
        "tracking": model_tracking_summary(rows),
        "present_epoch": evaluation,
        "measurement_signature": measurement_signature(args.lmc_core_radius_kpc, args.smc_core_radius_kpc,
            ref["tracer_policy"], ref["identity_fields"]),
        "stellar_tracer_policy": ref["tracer_policy"],
        "tracer_selection_method": ref.get("tracer_selection_method"),
        "legacy_star_id_ranges": ref.get("legacy_star_id_ranges"),
        "stellar_tracer_audit": tracer_audit,
        "shooting_provenance": str(provenance_path),
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
    plot_dashboard(rows, ref, evaluation, png_path)

    print("\nMagellanic validation complete")
    print(f"  history : {csv_path}")
    print(f"  present : {present_csv_path}")
    print(f"  provenance: {provenance_path}")
    print(f"  summary : {json_path}")
    print(f"  dashboard: {png_path}")
    print(f"  LMC 6D pass: {evaluation['lmc_6d_pass']}")
    print(f"  SMC 6D pass: {evaluation['smc_6d_pass']}")
    print(f"  scalar velocity review: {evaluation['scalar_velocity_pass']} "
          f"(required={evaluation['scalar_velocity_required']})")
    print(f"  aperture mass retention LMC(R<30)/SMC(R<15): "
          f"{evaluation['lmc_mass_r30_retention']:.3f} / "
          f"{evaluation['smc_mass_r15_retention']:.3f}")
    if evaluation["gas_timestep_available"]:
        print(f"  gas short-dt fraction: {evaluation['gas_short_dt_fraction']:.4e} "
              f"(pass={evaluation['gas_timestep_pass']})")
    else:
        print("  gas short-dt fraction: REVIEW (snapshot 没有 TimeStep 字段)")
    print(f"  overall pass: {evaluation['overall_pass']}")
    print(f"  ||r/sigma||_12D = {evaluation['diagonal_scaled_residual_l2']:.6f} "
          "（仅在相同示踪定义和容差下比较；不是完整协方差置信度）")
    print("  LMC normalized residual [x,y,z,vx,vy,vz] = "
          f"{np.asarray(evaluation['lmc_normalized_residual_6d'])}")
    print("  SMC normalized residual [x,y,z,vx,vy,vz] = "
          f"{np.asarray(evaluation['smc_normalized_residual_6d'])}")
    if not evaluation["target_time_bracketed"]:
        print("  警告：未覆盖目标时刻；present CSV 仅供查看，provenance 标为 shooting_usable=false，不可用于 shooting。")
    if tracer_audit.get("status") == "available":
        print(f"  原始老恒星−legacy 审计（只报告；CSV 策略为 {ref['tracer_policy']}）：")
        for label, report in (("IC", tracer_audit["reference_ic"].get("satellites", {})),
                              ("target", tracer_audit["target_old_minus_legacy"])):
            for sat in ("lmc", "smc"):
                entry = report.get(sat, {})
                if entry.get("status") == "ok":
                    print(f"    {label} {sat.upper()}: old−legacy "
                          f"Δr={entry['delta_position_norm_kpc']:.4f} kpc, "
                          f"Δv={entry['delta_velocity_norm_kms']:.4f} km/s")
        extra = sum(entry.get("legacy_non_original_star_count", 0)
                    for entry in tracer_audit["reference_ic"].get("satellites", {}).values())
        if extra:
            print(f"  提示：IC 的 legacy 恒星 ID 范围包含 {extra} 颗非原始老恒星；终点数量和中心差异见 JSON 审计。")


if __name__ == "__main__":
    main()
