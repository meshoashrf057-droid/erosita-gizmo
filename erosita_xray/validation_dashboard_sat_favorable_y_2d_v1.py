#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 LMC/SMC 最大 -y 扰动反事实轨道的 2D 终点：r 与 |v|。

输入由 add_lmc_smc_gaswake_favorable_y_2d_v1.py 生成的 reference IC，
以及 GIZMO 输出目录。脚本使用卫星原生 PartType4 恒星核追踪中心，避免被
暗物质尾流拖偏；银河中心优先用 ID=90000000 的中心 BH，缺失时才用 MW
恒星的 shrinking-sphere 中心。终点硬判只使用每颗卫星的 Galactocentric
半径 r 与速度模长 |v|；真实 K13 六分量只作诊断。另将速度与设计 -y 方向
的夹角、是否从 +y 飞到 -y 作为“反事实设计完整性”检查。
这里的“2D”是参数空间 (r,|v|)，不是 x-y 投影图。

输出：
  magellanic_2d_history.csv        每个快照的 LMC/SMC 状态与设计轨道
  magellanic_present_2d.csv        今天的 r、|v| 残差；供下一轮 shooting
  magellanic_present_6d_diagnostic.csv  真实 K13 六分量偏差，仅诊断
  magellanic_2d_validation_summary.json  2D、方向、结构判定
  magellanic_2d_dashboard.png      设计轨道和 2D 残差图板

示例：
python validation_dashboard_sat_favorable_y_2d_v1.py \
  --snapshot-dir ./output \
  --reference-ic mw_mou_cgm_lmc_smc_tminus0p5.hdf5
"""

from __future__ import annotations

import argparse
import csv
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


def load_reference(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as f:
        if "MagellanicOrbitReference" not in f:
            raise RuntimeError(
                f"{path} lacks MagellanicOrbitReference. --reference-ic must be the "
                "counterfactual IC written by add_lmc_smc_gaswake_favorable_y_2d_v1.py, "
                "not the Stage-1 isolated MW host IC."
            )
        g = f["MagellanicOrbitReference"]
        mode = str(g.attrs.get("EndpointAcceptanceMode", ""))
        if mode != "radius_speed_2d":
            raise RuntimeError(
                f"{path} 的 EndpointAcceptanceMode={mode!r}；本脚本只接受 "
                "radius_speed_2d 反事实 IC，不能拿真实 6D IC 混检。"
            )
        required = (
            "TimeFromPresent_Gyr", "LMC_Position_kpc", "LMC_Velocity_kms",
            "SMC_Position_kpc", "SMC_Velocity_kms",
            "LMC_Current_PositionSigma_kpc", "LMC_Current_VelocitySigma_kms",
            "SMC_Current_PositionSigma_kpc", "SMC_Current_VelocitySigma_kms",
            "FavorableDirectionUnit",
            "LMC_ObservedPosition_kpc", "LMC_ObservedVelocity_kms",
            "SMC_ObservedPosition_kpc", "SMC_ObservedVelocity_kms",
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
            "favorable_direction": np.asarray(g["FavorableDirectionUnit"][:], dtype=float),
            "lmc_observed_position": np.asarray(g["LMC_ObservedPosition_kpc"][:], dtype=float),
            "lmc_observed_velocity": np.asarray(g["LMC_ObservedVelocity_kms"][:], dtype=float),
            "smc_observed_position": np.asarray(g["SMC_ObservedPosition_kpc"][:], dtype=float),
            "smc_observed_velocity": np.asarray(g["SMC_ObservedVelocity_kms"][:], dtype=float),
            "lmc_target_radius": float(g.attrs["LMC_TargetRadius_kpc"]),
            "lmc_target_speed": float(g.attrs["LMC_TargetSpeed_kms"]),
            "smc_target_radius": float(g.attrs["SMC_TargetRadius_kpc"]),
            "smc_target_speed": float(g.attrs["SMC_TargetSpeed_kms"]),
            "lmc_radius_sigma": float(g.attrs["LMC_RadiusSigma_kpc"]),
            "lmc_speed_sigma": float(g.attrs["LMC_SpeedSigma_kms"]),
            "smc_radius_sigma": float(g.attrs["SMC_RadiusSigma_kpc"]),
            "smc_speed_sigma": float(g.attrs["SMC_SpeedSigma_kms"]),
            "acceptance_mode": mode,
            "lookback_gyr": float(g.attrs["Lookback_Gyr"]),
            "insertion_time_code": float(g.attrs["InsertionTime_Code"]),
            "target_time_code": float(g.attrs["PresentTargetTime_Code"]),
            "unit_time_gyr": float(g.attrs.get("UnitTime_Gyr", UNIT_TIME_GYR_DEFAULT)),
            # 只使用原生恒星盘的精确 ID 范围；GALSF 产生的新星不参与质心。
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
        }
        direction_norm = float(np.linalg.norm(ref["favorable_direction"]))
        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            raise RuntimeError("FavorableDirectionUnit 非法。")
        ref["favorable_direction"] = ref["favorable_direction"] / direction_norm
        star = f["PartType4"]
        sid = np.asarray(star["ParticleIDs"][:], dtype=np.uint64)
        spos = np.asarray(star["Coordinates"][:], dtype=float)
        smass = np.asarray(star["Masses"][:], dtype=float)
        dm = f["PartType1"]
        did = np.asarray(dm["ParticleIDs"][:], dtype=np.uint64)
        dpos = np.asarray(dm["Coordinates"][:], dtype=float)
        dmass = np.asarray(dm["Masses"][:], dtype=float)
        for sat, aperture in (("lmc", 30.0), ("smc", 15.0)):
            star_mask = (sid >= ref[f"{sat}_id_start"]) & (sid <= ref[f"{sat}_id_end"])
            dm_mask = (did >= ref[f"{sat}_dm_id_start"]) & (did <= ref[f"{sat}_dm_id_end"])
            if not np.any(star_mask) or not np.any(dm_mask):
                raise RuntimeError(f"reference IC lacks exact {sat.upper()} tracer/DM IDs")
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
            radius = np.linalg.norm(spos - center, axis=1)
            core = mw & (radius < 5.0)
            if core.sum() < 32:
                core = mw & (radius < 10.0)
            if core.sum() >= 32:
                # 与 IC 生成器完全相同：BH 定位置、内盘老恒星速度中位数
                # 定宿主平动速度。避免 snapshot_000 人为出现数 km/s 偏差。
                svel = np.asarray(s["Velocities"][:], dtype=float)
                velocity = np.median(svel[core], axis=0)
                method = "MW BH position + robust inner-star velocity"
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
                            core_radius_kpc: float) -> tuple[np.ndarray, np.ndarray, int, float, int]:
    if "PartType4" not in f:
        raise RuntimeError("PartType4 absent; cannot track satellite stellar COM")
    g = f["PartType4"]
    ids = np.asarray(g["ParticleIDs"][:], dtype=np.uint64)
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
        mask = (ids >= ref[start_key]) & (ids <= ref[end_key])
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


def vector_angle_deg(vector: np.ndarray, direction: np.ndarray) -> float:
    """返回两个非零向量的夹角；退化向量返回 NaN。"""
    vector = np.asarray(vector, dtype=float)
    direction = np.asarray(direction, dtype=float)
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(direction))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    cosine = float(np.clip(np.dot(vector, direction) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


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
        time_code = float(f["Header"].attrs["Time"])
        hp, hv, host_method = host_state(f, ref)
        lp_abs, lv_abs, nl, rl, nlt = satellite_stellar_state(
            f, ref["lmc_id_start"], ref["lmc_id_end"], lmc_core_kpc
        )
        sp_abs, sv_abs, ns, rs, nst = satellite_stellar_state(
            f, ref["smc_id_start"], ref["smc_id_end"], smc_core_kpc
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
    row["lmc_favorable_direction_angle_deg"] = vector_angle_deg(
        lv, np.asarray(ref["favorable_direction"])
    )
    row["smc_favorable_direction_angle_deg"] = vector_angle_deg(
        sv, np.asarray(ref["favorable_direction"])
    )
    return row


def interpolate_simulation_at_target(rows: list[dict[str, object]], target_code: float
                                     ) -> tuple[dict[str, np.ndarray], str, list[str]]:
    rows = sorted(rows, key=lambda r: float(r["time_code"]))
    times = np.array([float(r["time_code"]) for r in rows])
    warnings: list[str] = []
    components = ("x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms")
    result: dict[str, np.ndarray] = {}
    tolerance = 1.0e-8 * max(1.0, abs(target_code))
    if times[0] - tolerance <= target_code <= times[-1] + tolerance \
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
                     nsigma: float, max_direction_angle_deg: float,
                     minimum_negative_y_fraction: float,
                     min_lmc_retention: float, min_smc_retention: float,
                     max_short_dt_fraction: float, orbit_only: bool) -> dict[str, object]:
    """在今天时刻硬判 r、|v|，并检查反事实 -y 穿越设计是否保持。"""
    sim, method, warnings = interpolate_simulation_at_target(
        rows, float(ref["target_time_code"])
    )
    covered = "not bracketed" not in method
    ordered_rows = sorted(rows, key=lambda row: float(row["time_code"]))
    first_time = float(ordered_rows[0]["time_code"])
    insertion_time = float(ref["insertion_time_code"])
    start_tolerance = 1.0e-8 * max(1.0, abs(insertion_time))
    insertion_sampled = bool(first_time <= insertion_time + start_tolerance)
    if not insertion_sampled:
        warnings.append(
            "最早快照晚于插入时刻，无法硬证实卫星由 +y 一侧出发；请保留 snapshot_000。"
        )

    evaluation: dict[str, object] = {
        "evaluation_method": method,
        "warnings": warnings,
        "acceptance_mode": "radius_speed_2d",
        "counterfactual_not_observed_6d": True,
        "target_time_code": ref["target_time_code"],
        "target_elapsed_gyr": ref["lookback_gyr"],
        "nsigma_threshold": nsigma,
        "target_time_bracketed": covered,
        "insertion_snapshot_sampled": insertion_sampled,
        "favorable_direction_unit": np.asarray(ref["favorable_direction"]),
        "maximum_direction_angle_deg": max_direction_angle_deg,
        "minimum_negative_y_velocity_fraction": minimum_negative_y_fraction,
    }

    # 真正的硬验收只有每颗卫星的 Galactocentric 半径和绝对速度模长。
    # 三维方向由反事实实验指定，不要求贴合 K13 的真实速度方向。
    hard_passes = []
    integrity_passes = []
    for sat in ("lmc", "smc"):
        state = np.asarray(sim[sat], dtype=float)
        simulated_2d = np.array([
            np.linalg.norm(state[:3]), np.linalg.norm(state[3:])
        ], dtype=float)
        target_2d = np.array([
            float(ref[f"{sat}_target_radius"]),
            float(ref[f"{sat}_target_speed"]),
        ])
        sigma_2d = np.array([
            float(ref[f"{sat}_radius_sigma"]),
            float(ref[f"{sat}_speed_sigma"]),
        ])
        residual_2d = simulated_2d - target_2d
        normalized_2d = residual_2d / sigma_2d
        two_d_pass = bool(np.all(np.abs(normalized_2d) <= nsigma))

        direction_angle = vector_angle_deg(
            state[3:], np.asarray(ref["favorable_direction"])
        )
        direction_pass = bool(
            np.isfinite(direction_angle) and direction_angle <= max_direction_angle_deg
        )
        first_y = float(ordered_rows[0][f"{sat}_y_kpc"])
        endpoint_y = float(state[1])
        vy_history = np.array(
            [float(row[f"{sat}_vy_kms"]) for row in ordered_rows], dtype=float
        )
        finite_vy = np.isfinite(vy_history)
        negative_y_fraction = (
            float(np.mean(vy_history[finite_vy] < 0.0))
            if np.any(finite_vy) else float("nan")
        )
        positive_y_start = bool(insertion_sampled and first_y > 0.0)
        negative_y_end = bool(endpoint_y < 0.0)
        crossing_pass = bool(
            positive_y_start and negative_y_end
            and np.isfinite(negative_y_fraction)
            and negative_y_fraction >= minimum_negative_y_fraction
        )

        # 真实 K13 六分量只提供诊断，绝不进入 overall_pass。
        observed_6d = np.concatenate((
            np.asarray(ref[f"{sat}_observed_position"]),
            np.asarray(ref[f"{sat}_observed_velocity"]),
        ))
        sigma_6d = np.concatenate((
            np.asarray(ref[f"{sat}_pos_sigma"]),
            np.asarray(ref[f"{sat}_vel_sigma"]),
        ))
        residual_6d = state - observed_6d
        normalized_6d = residual_6d / sigma_6d
        diagnostic_6d_pass = bool(np.all(np.abs(normalized_6d) <= nsigma))
        design_6d = np.concatenate((
            np.asarray(ref[f"{sat}_pos"])[-1],
            np.asarray(ref[f"{sat}_vel"])[-1],
        ))

        evaluation.update({
            f"{sat}_simulated_6d": state,
            f"{sat}_design_6d": design_6d,
            f"{sat}_observed_6d": observed_6d,
            f"{sat}_sigma_6d": sigma_6d,
            f"{sat}_residual_6d": residual_6d,
            f"{sat}_normalized_residual_6d": normalized_6d,
            f"{sat}_6d_diagnostic_pass": diagnostic_6d_pass,
            f"{sat}_simulated_2d": simulated_2d,
            f"{sat}_target_2d": target_2d,
            f"{sat}_sigma_2d": sigma_2d,
            f"{sat}_residual_2d": residual_2d,
            f"{sat}_normalized_residual_2d": normalized_2d,
            f"{sat}_2d_pass": two_d_pass,
            f"{sat}_favorable_direction_angle_deg": direction_angle,
            f"{sat}_direction_pass": direction_pass,
            f"{sat}_initial_y_kpc": first_y,
            f"{sat}_endpoint_y_kpc": endpoint_y,
            f"{sat}_positive_y_start": positive_y_start,
            f"{sat}_negative_y_end": negative_y_end,
            f"{sat}_negative_y_velocity_fraction": negative_y_fraction,
            f"{sat}_crossing_pass": crossing_pass,
            f"{sat}_derived": derived_orbit(state[:3], state[3:]),
        })
        hard_passes.append(two_d_pass)
        integrity_passes.extend((direction_pass, crossing_pass))

    rel_derived = derived_orbit(
        np.asarray(sim["smc"][:3]) - np.asarray(sim["lmc"][:3]),
        np.asarray(sim["smc"][3:]) - np.asarray(sim["lmc"][3:]),
    )
    evaluation["smc_lmc_derived"] = rel_derived
    evaluation["smc_lmc_speed_k13_z_diagnostic"] = (
        (rel_derived["speed_kms"] - K13_RELATIVE_SPEED_KMS)
        / K13_RELATIVE_SPEED_SIGMA_KMS
    )

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
    counterfactual_integrity_pass = bool(all(integrity_passes))
    evaluation.update({
        "two_dimensional_endpoint_pass": bool(all(hard_passes)),
        "counterfactual_integrity_pass": counterfactual_integrity_pass,
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
            covered and all(hard_passes) and counterfactual_integrity_pass
            and (science_quality_pass or orbit_only)
        ),
    })
    return evaluation


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


def write_present_2d_table(evaluation: dict[str, object], path: Path) -> None:
    """写出可直接回灌 IC shooting 的 r、|v| 标量残差。"""
    rows = []
    for sat in ("lmc", "smc"):
        simulated = np.asarray(evaluation[f"{sat}_simulated_2d"])
        target = np.asarray(evaluation[f"{sat}_target_2d"])
        residual = np.asarray(evaluation[f"{sat}_residual_2d"])
        zscore = np.asarray(evaluation[f"{sat}_normalized_residual_2d"])
        sigma = np.asarray(evaluation[f"{sat}_sigma_2d"])
        for j, (quantity, unit) in enumerate((("radius", "kpc"), ("speed", "km/s"))):
            rows.append({
                "satellite": sat.upper(), "quantity": quantity, "unit": unit,
                "simulated": simulated[j], "target": target[j],
                "residual": residual[j], "sigma": sigma[j], "zscore": zscore[j],
                "pass": bool(abs(zscore[j]) <= float(evaluation["nsigma_threshold"])),
            })
    write_csv(rows, path)


def write_present_6d_diagnostic(evaluation: dict[str, object], path: Path) -> None:
    """写出相对真实 K13 六分量的偏差；此表不参与 2D 反事实验收。"""
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
                "hard_acceptance": False,
                "pass": bool(abs(zscore[j]) <= float(evaluation["nsigma_threshold"])),
            })
    write_csv(rows, path)


def plot_dashboard(rows: list[dict[str, object]], ref: dict[str, object],
                   evaluation: dict[str, object], path: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r["elapsed_gyr"]))
    elapsed = np.array([float(r["elapsed_gyr"]) for r in rows])
    ref_elapsed = np.asarray(ref["time_from_present_gyr"]) + float(ref["lookback_gyr"])
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), dpi=160, sharex="col")

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
    x = np.arange(4)
    normalized_2d = np.concatenate((
        np.asarray(evaluation["lmc_normalized_residual_2d"]),
        np.asarray(evaluation["smc_normalized_residual_2d"]),
    ))
    ax.bar(x, normalized_2d,
           color=("tab:blue", "tab:blue", "tab:orange", "tab:orange"))
    nsigma = float(evaluation["nsigma_threshold"])
    ax.axhline(nsigma, color="red", ls="--", lw=1)
    ax.axhline(-nsigma, color="red", ls="--", lw=1)
    ax.set_xticks(x, ("LMC r", "LMC |v|", "SMC r", "SMC |v|"))
    ax.set(xlabel="2D hard-acceptance quantity",
           ylabel="Residual / scalar sigma",
           title=f"Present endpoint: r and |v| (pass: |z| <= {nsigma:g})")
    ax.grid(axis="y", alpha=0.25)
    integrity_text = (
        f"velocity angle to -y: LMC {evaluation['lmc_favorable_direction_angle_deg']:.1f}°, "
        f"SMC {evaluation['smc_favorable_direction_angle_deg']:.1f}°\n"
        f"fraction(vy<0): LMC {evaluation['lmc_negative_y_velocity_fraction']:.2f}, "
        f"SMC {evaluation['smc_negative_y_velocity_fraction']:.2f}"
    )
    ax.text(0.02, 0.98, integrity_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, 0.995))
    status = "PASS" if evaluation["overall_pass"] else "FAIL / RETUNE ORBIT"
    fig.suptitle(
        f"Magellanic favorable -y counterfactual: 2D endpoint validation - {status}\n"
        f"hard fit: r and |v| only | true K13 6D is diagnostic only | "
        f"potential: {ref['potential']}",
        y=1.035, fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate favorable -y LMC/SMC counterfactual in r and |v|"
    )
    p.add_argument("--snapshot-dir", default="./output")
    p.add_argument("--pattern", default="snapshot_*.hdf5")
    p.add_argument("--reference-ic", required=True,
                   help="由 add_lmc_smc_gaswake_favorable_y_2d_v1.py 生成的反事实 IC")
    p.add_argument("--output-dir", default=None,
                   help="default: SNAPSHOT_DIR/magellanic_validation")
    p.add_argument("--lmc-core-radius-kpc", type=float, default=8.0)
    p.add_argument("--smc-core-radius-kpc", type=float, default=3.0)
    p.add_argument("--nsigma", type=float, default=3.0,
                   help="今天时刻 r 与 |v| 各自允许的 sigma 阈值。")
    p.add_argument(
        "--unit-time-gyr", type=float, default=None,
        help="可选：覆盖 reference IC 的代码时间单位 [Gyr]；通常不要设置。",
    )
    p.add_argument("--max-direction-angle-deg", type=float, default=20.0,
                   help="今天速度相对设计 -y 方向允许的最大夹角 [deg]。")
    p.add_argument("--minimum-negative-y-fraction", type=float, default=0.80,
                   help="快照序列中 vy<0 所占的最低比例。")
    p.add_argument("--short-dt-threshold", type=float, default=1.0e-6,
                   help="GIZMO code units；短气体时间步阈值。")
    p.add_argument("--max-short-dt-fraction", type=float, default=1.0e-3,
                   help="今天时刻允许的短时间步气体比例。")
    p.add_argument("--min-lmc-mass-retention", type=float, default=0.50,
                   help="LMC 原生 DM+star 在 30 kpc 固定孔径内的最低质量保留率。")
    p.add_argument("--min-smc-mass-retention", type=float, default=0.30,
                   help="SMC 原生 DM+star 在 15 kpc 固定孔径内的最低质量保留率。")
    p.add_argument("--orbit-only", action="store_true",
                   help="只硬判 2D 终点与 -y 穿越；结构/短时间步只报告。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.nsigma <= 0.0:
        raise ValueError("--nsigma 必须大于 0。")
    if not (0.0 < args.max_direction_angle_deg <= 180.0):
        raise ValueError("--max-direction-angle-deg 必须在 (0,180]。")
    if not (0.0 <= args.minimum_negative_y_fraction <= 1.0):
        raise ValueError("--minimum-negative-y-fraction 必须在 [0,1]。")
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
    ref = load_reference(reference_path)
    if args.unit_time_gyr is not None:
        if not np.isfinite(args.unit_time_gyr) or args.unit_time_gyr <= 0.0:
            raise ValueError("--unit-time-gyr 必须为有限正数。")
        ref["unit_time_gyr"] = float(args.unit_time_gyr)
    files = list(snapshot_dir.glob(args.pattern))
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

    csv_path = output_dir / "magellanic_2d_history.csv"
    json_path = output_dir / "magellanic_2d_validation_summary.json"
    png_path = output_dir / "magellanic_2d_dashboard.png"
    present_2d_csv_path = output_dir / "magellanic_present_2d.csv"
    present_6d_csv_path = output_dir / "magellanic_present_6d_diagnostic.csv"
    write_csv(rows, csv_path)
    evaluation = evaluate_present(
        rows, ref, args.nsigma, args.max_direction_angle_deg,
        args.minimum_negative_y_fraction,
        args.min_lmc_mass_retention, args.min_smc_mass_retention,
        args.max_short_dt_fraction, args.orbit_only,
    )
    write_present_2d_table(evaluation, present_2d_csv_path)
    write_present_6d_diagnostic(evaluation, present_6d_csv_path)
    summary = {
        "experiment": "favorable -y counterfactual; fit radius and speed only",
        "warning": "This is not a fit to the observed K13 6D direction.",
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
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
    plot_dashboard(rows, ref, evaluation, png_path)

    print("\nMagellanic favorable -y 2D validation complete")
    print(f"  history : {csv_path}")
    print(f"  present 2D / shooting feedback: {present_2d_csv_path}")
    print(f"  true K13 6D diagnostic       : {present_6d_csv_path}")
    print(f"  summary : {json_path}")
    print(f"  dashboard: {png_path}")
    print(f"  LMC 2D [r,|v|] pass: {evaluation['lmc_2d_pass']}")
    print(f"  SMC 2D [r,|v|] pass: {evaluation['smc_2d_pass']}")
    print(f"  LMC -y angle / path pass: "
          f"{evaluation['lmc_favorable_direction_angle_deg']:.3f} deg / "
          f"{evaluation['lmc_crossing_pass']}")
    print(f"  SMC -y angle / path pass: "
          f"{evaluation['smc_favorable_direction_angle_deg']:.3f} deg / "
          f"{evaluation['smc_crossing_pass']}")
    print("  K13 6D diagnostic pass (not hard acceptance): "
          f"LMC={evaluation['lmc_6d_diagnostic_pass']}, "
          f"SMC={evaluation['smc_6d_diagnostic_pass']}")
    print(f"  aperture mass retention LMC(R<30)/SMC(R<15): "
          f"{evaluation['lmc_mass_r30_retention']:.3f} / "
          f"{evaluation['smc_mass_r15_retention']:.3f}")
    if evaluation["gas_timestep_available"]:
        print(f"  gas short-dt fraction: {evaluation['gas_short_dt_fraction']:.4e} "
              f"(pass={evaluation['gas_timestep_pass']})")
    else:
        print("  gas short-dt fraction: REVIEW (snapshot 没有 TimeStep 字段)")
    print(f"  overall pass: {evaluation['overall_pass']}")
    print("  LMC normalized 2D residual [r,|v|] = "
          f"{np.asarray(evaluation['lmc_normalized_residual_2d'])}")
    print("  SMC normalized 2D residual [r,|v|] = "
          f"{np.asarray(evaluation['smc_normalized_residual_2d'])}")


if __name__ == "__main__":
    main()
