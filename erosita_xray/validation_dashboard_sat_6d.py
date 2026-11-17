#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Stage-2 LMC/SMC 轨道的 6D 相空间，而不是只比较距离和总速度。

输入由 add_magellanic_clouds_0p5gyr.py 生成的 reference IC，以及 GIZMO
输出目录。脚本使用卫星 PartType4 恒星核追踪中心，避免被剥离气体/暗物质
尾流拖偏；银河中心优先用 ID=90000000 的中心 BH，缺失时才用 MW 恒星的
shrinking-sphere 中心。

输出：
  magellanic_6d_history.csv        每个快照的 LMC/SMC 6D 与参考残差
  magellanic_validation_summary.json  当今目标时刻的逐分量判定
  magellanic_6d_dashboard.png      6D 历史、参考轨道和残差图板

示例：
python validation_dashboard_sat_6d.py \
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
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
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
                f"{path} lacks MagellanicOrbitReference; use the paired add script"
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
            "lmc_id_start": int(g.attrs["LMC_ID_START"]),
            "lmc_id_end": int(g.attrs["LMC_ID_END"]),
            "smc_id_start": int(g.attrs["SMC_ID_START"]),
            "smc_id_end": int(g.attrs["SMC_ID_END"]),
            "lmc_mass_msun": float(g.attrs["LMC_TotalMass_Msun"]),
            "smc_mass_msun": float(g.attrs["SMC_TotalMass_Msun"]),
            "frame": str(g.attrs.get("CoordinateFrame", "K13 Galactocentric XYZ")),
            "potential": str(g.attrs.get("Potential", "Mou et al. 2023 Eq.4")),
        }
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
        match = np.flatnonzero(ids == MW_BH_ID)
        idx = int(match[0]) if match.size else 0
        center = np.asarray(g["Coordinates"][idx], dtype=float)
        velocity = np.asarray(g["Velocities"][idx], dtype=float)
        method = "MW BH"
        if "PartType4" in f:
            s = f["PartType4"]
            sids = np.asarray(s["ParticleIDs"][:], dtype=np.uint64)
            mw = ~(
                ((sids >= ref["lmc_id_start"]) & (sids <= ref["lmc_id_end"]))
                | ((sids >= ref["smc_id_start"]) & (sids <= ref["smc_id_end"]))
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
        ((ids >= ref["lmc_id_start"]) & (ids <= ref["lmc_id_end"]))
        | ((ids >= ref["smc_id_start"]) & (ids <= ref["smc_id_end"]))
    )
    pos = np.asarray(g["Coordinates"][:], dtype=float)[mw]
    vel = np.asarray(g["Velocities"][:], dtype=float)[mw]
    mass = np.asarray(g["Masses"][:], dtype=float)[mw]
    center, velocity, _, _ = shrinking_core_state(pos, vel, mass, 5.0, 100)
    return center, velocity, "shrinking-sphere MW stars"


def satellite_stellar_state(f: h5py.File, id_start: int, id_end: int,
                            core_radius_kpc: float) -> tuple[np.ndarray, np.ndarray, int, float]:
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
    return shrinking_core_state(pos, vel, mass, core_radius_kpc)


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
                     lmc_core_kpc: float, smc_core_kpc: float) -> dict[str, object]:
    with h5py.File(path, "r") as f:
        time_code = float(f["Header"].attrs["Time"])
        hp, hv, host_method = host_state(f, ref)
        lp_abs, lv_abs, nl, rl = satellite_stellar_state(
            f, ref["lmc_id_start"], ref["lmc_id_end"], lmc_core_kpc
        )
        sp_abs, sv_abs, ns, rs = satellite_stellar_state(
            f, ref["smc_id_start"], ref["smc_id_end"], smc_core_kpc
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
    if times[0] <= target_code <= times[-1] and len(np.unique(times)) >= 2:
        method = "linear interpolation between bracketing snapshots"
        for sat in ("lmc", "smc"):
            result[sat] = np.array([
                np.interp(target_code, times,
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


def evaluate_present(rows: list[dict[str, object]], ref: dict[str, object],
                     nsigma: float) -> dict[str, object]:
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
    return {
        "evaluation_method": method,
        "warnings": warnings,
        "target_time_code": ref["target_time_code"],
        "target_elapsed_gyr": ref["lookback_gyr"],
        "nsigma_threshold": nsigma,
        "lmc_simulated_6d": sim["lmc"],
        "lmc_observed_6d": lmc_obs,
        "lmc_residual_6d": lmc_res,
        "lmc_normalized_residual_6d": lmc_z,
        "lmc_6d_pass": lmc_6d_pass,
        "smc_simulated_6d": sim["smc"],
        "smc_observed_6d": smc_obs,
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
        "overall_pass": bool(covered and lmc_6d_pass and smc_6d_pass and scalar_pass),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate LMC/SMC 6D phase space")
    p.add_argument("--snapshot-dir", default="./output")
    p.add_argument("--pattern", default="snapshot_*.hdf5")
    p.add_argument("--reference-ic", required=True,
                   help="Stage-2 IC created by add_magellanic_clouds_0p5gyr.py")
    p.add_argument("--output-dir", default=None,
                   help="default: SNAPSHOT_DIR/magellanic_validation")
    p.add_argument("--lmc-core-radius-kpc", type=float, default=8.0)
    p.add_argument("--smc-core-radius-kpc", type=float, default=3.0)
    p.add_argument("--nsigma", type=float, default=3.0,
                   help="per-component present-day 6D threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    reference_path = Path(args.reference_ic).expanduser().resolve()
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                  else snapshot_dir / "magellanic_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = load_reference(reference_path)
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
                                   args.smc_core_radius_kpc)
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
    write_csv(rows, csv_path)
    evaluation = evaluate_present(rows, ref, args.nsigma)
    summary = {
        "reference_ic": str(reference_path),
        "snapshot_count": len(rows),
        "failed_snapshot_count": len(failures),
        "failed_snapshots": failures,
        "coordinate_order": ["x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms"],
        "tracking": model_tracking_summary(rows),
        "present_epoch": evaluation,
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
    plot_dashboard(rows, ref, evaluation, png_path)

    print("\nMagellanic validation complete")
    print(f"  history : {csv_path}")
    print(f"  summary : {json_path}")
    print(f"  dashboard: {png_path}")
    print(f"  LMC 6D pass: {evaluation['lmc_6d_pass']}")
    print(f"  SMC 6D pass: {evaluation['smc_6d_pass']}")
    print(f"  overall pass: {evaluation['overall_pass']}")
    print("  LMC normalized residual [x,y,z,vx,vy,vz] = "
          f"{np.asarray(evaluation['lmc_normalized_residual_6d'])}")
    print("  SMC normalized residual [x,y,z,vx,vy,vz] = "
          f"{np.asarray(evaluation['smc_normalized_residual_6d'])}")


if __name__ == "__main__":
    main()
