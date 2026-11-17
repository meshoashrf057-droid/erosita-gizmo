#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the *pre-jet* MW CGM of a Magellanic run with an isolated control.

For the Stage-2 IC made by add_lmc_smc_gaswake_orbit_v4.py and the Stage-1 IC
made by make_ic_live_mw_cgm_isolated_v3.py. This is an analysis-only script:
it never changes an IC, snapshot, or GIZMO parameter file.

Prerequisites: Python >=3.9, numpy, h5py, matplotlib; one-file GIZMO HDF5
snapshots with PartType0/Density and PartType0/InternalEnergy. Both branches
must start from the same relaxed host snapshot, use the same gas physics,
and have the same elapsed time since branching. The control must be run from
that host without adding satellites. A separately evolved Stage-1 relaxation
sequence is NOT an interchangeable control.

Example (replace snapshot patterns and launch snapshot with your real files):

  python diagnose_prejet_cgm_crosswind_v1.py \
    --reference-ic mw_relaxed_lmc_smc_gaswake_trial02.hdf5 \
    --satellite-snapshots 'output_sat/snapshot_*.hdf5' \
    --isolated-snapshots 'output_control/snapshot_*.hdf5' \
    --jet-start-snapshot output_sat/snapshot_100.hdf5 \
    --pre-window-myr 30 --outdir prejet_cgm_diagnosis

Alternatively replace --jet-start-snapshot with --jet-start-elapsed-gyr 1.0,
where elapsed time is measured since Stage-2 insertion, NOT since Stage-1.
With neither option the IC's nominal present-day target is used, which need
not equal the real jet launch time. The control run is assumed to reset its
code time to zero; change --isolated-start-code if it did not.

Output files in --outdir:
  prejet_region_history.csv: MW/LMC-origin hot-gas properties in three
    |z| shells on both sides of the stellar disc, inside cylindrical R<10 kpc.
  prejet_sector_history.csv: separate +/-X and +/-Y halves; optional +/-W
    if --wind-vector is provided. Unsigned ram stress alone is NOT a wind.
  prejet_sector_contrast.csv: plus/minus density and thermal-pressure contrasts.
  prejet_outer_history.csv: source-separated spherical 20-40, 40-60, and
    60-100 kpc hemispheres; check whether the wake remains outside the bubbles.
  prejet_core_history.csv: MW-CGM-origin hot/cold gas and converted stars
    inside spherical 5 and 20 kpc, for diagnosing central condensation.
  prejet_matched_comparison.csv: satellite minus control, matched by elapsed
    physical time. Includes induced velocity and its *illustrative* dynamic
    scale rho_sat |delta v_perp|^2, not an externally injected wind pressure.
  prejet_comparison.png: the latest matched snapshot pair, MW CGM only.
  prejet_summary.json: sample times, frame checks, and cautions.

Quantities: n is total particle number density, using mu=0.59; P/kB is in
K cm^-3; positions are BH-relative, velocities are relative to the mean
motion of inner original MW stars. The signed vx,vy and +/- sector contrasts
must be examined in addition to rho v_perp^2. T estimated from u assumes
fully ionized single-phase gas and is only used to exclude T<1e5 K.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys as _compat_sys
_compat_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gizmo_diagnostics_compat import gas_temperature, is_snapshot_cache, require_single_snapshot
from gizmo_units import resolve_units, require_noncosmological


K_B = 1.380649e-16
M_P = 1.67262192369e-24
MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
GAMMA = 5.0 / 3.0
UNIT_TIME_GYR = KPC_CM / 1.0e5 / (365.25 * 86400.0 * 1.0e9)
SHELLS = ((3.0, 8.0), (8.0, 15.0), (15.0, 25.0))
OUTER_SHELLS = ((20.0, 40.0), (40.0, 60.0), (60.0, 100.0))


def args_from_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-ic", required=True, type=Path,
                   help="The exact Stage-2 IC from which the satellite run branched")
    p.add_argument("--satellite-snapshots", required=True, nargs="+",
                   help="One or more snapshot paths, directories, or quoted globs")
    p.add_argument("--isolated-snapshots", required=True, nargs="+",
                   help="Same-host, same-physics control WITHOUT satellites")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--jet-start-snapshot", type=Path,
                        help="Pre-jet Stage-2 snapshot at the actual launch time")
    target.add_argument("--jet-start-elapsed-gyr", type=float,
                        help="Actual physical launch time since Stage-2 insertion")
    p.add_argument("--isolated-start-code", type=float, default=0.0)
    p.add_argument("--pre-window-myr", type=float, default=30.0,
                   help="Analyze snapshots in [launch-window, launch]; default 30")
    p.add_argument("--max-pair-offset-myr", type=float, default=12.0,
                   help="Maximum satellite/control elapsed-time difference")
    p.add_argument("--rmax-kpc", type=float, default=10.0,
                   help="Cylindrical radius of the two bubble corridors")
    p.add_argument("--wind-vector", type=str,
                   help="Optional physical crosswind direction vx,vy,vz; projected into XY")
    p.add_argument("--min-particles", type=int, default=16,
                   help="Flag individual regions with fewer gas particles")
    p.add_argument("--unit-mass-msun", type=float, default=None)
    p.add_argument("--unit-length-kpc", type=float, default=None)
    p.add_argument("--unit-velocity-kms", type=float, default=None)
    p.add_argument("--unit-time-gyr", type=float,
                   help="Overrides the Stage-2 IC UnitTime_Gyr attribute")
    p.add_argument("--outdir", type=Path, default=Path("prejet_cgm_diagnosis"))
    return p.parse_args()


def parse_vector(s):
    try:
        v = np.asarray([float(x) for x in s.split(",")], dtype=float)
    except ValueError as e:
        raise ValueError("--wind-vector must be three comma-separated numbers") from e
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v[:2]) == 0:
        raise ValueError("--wind-vector needs a nonzero XY projection")
    return v[:2] / np.linalg.norm(v[:2])


def load_reference(path):
    with h5py.File(path, "r") as f:
        if "ReservedParticleIDs" not in f or "MagellanicOrbitReference" not in f:
            raise ValueError("--reference-ic must be a Stage-2 IC with ReservedParticleIDs "
                             "and MagellanicOrbitReference")
        a = f["ReservedParticleIDs"].attrs
        needed = ("MW_CGM_ID_START", "MW_CGM_ID_END", "MW_DISK_ID_START",
                  "MW_DISK_ID_END", "MW_BULGE_ID_START", "MW_BULGE_ID_END")
        absent = [name for name in needed if name not in a]
        if absent:
            raise ValueError(f"Reference IC lacks exact MW origin ID ranges: {absent}")
        ids = {name: int(a[name]) for name in needed}
        for name in ("MW_ISM_ID_START", "MW_ISM_ID_END", "LMC_GAS_ID_START",
                     "LMC_GAS_ID_END", "LMC_STAR_ID_START", "LMC_STAR_ID_END"):
            if name in a:
                ids[name] = int(a[name])
        o = f["MagellanicOrbitReference"].attrs
        return {
            "ids": ids,
            "bh_id": int(o.get("MW_BH_ID", a.get("MW_BH_ID", 90_000_000))),
            "stage2_start_code": float(o["InsertionTime_Code"]),
            "target_time_code": float(o["PresentTargetTime_Code"]),
            "unit_time_gyr": float(o.get("UnitTime_Gyr", UNIT_TIME_GYR)),
            "lookback_gyr": float(o["Lookback_Gyr"]),
        }


def files_from_tokens(tokens):
    found = set()
    for token in tokens:
        path = Path(token)
        if path.is_dir():
            found.update(path.glob("snapshot_*.hdf5"))
        else:
            found.update(Path(s) for s in glob.glob(token))
    result = sorted((p.resolve() for p in found if p.is_file() and not is_snapshot_cache(p)), key=lambda x: str(x))
    if not result:
        raise FileNotFoundError(f"No snapshot HDF5 files found for: {tokens}")
    return result


def header_time(path):
    with h5py.File(path, "r") as f:
        return float(f["Header"].attrs["Time"])


def select_times(paths, start_code, unit_time, jet_gyr, window_myr):
    selected = []
    for path in paths:
        code = header_time(path)
        elapsed = (code - start_code) * unit_time
        if jet_gyr - window_myr / 1000 - 1e-8 <= elapsed <= jet_gyr + 1e-8:
            selected.append({"path": str(path), "code": code, "elapsed": elapsed})
    return sorted(selected, key=lambda r: r["elapsed"])


def mask_range(ids, ranges, prefix):
    lo, hi = ranges[f"{prefix}_ID_START"], ranges[f"{prefix}_ID_END"]
    return (ids >= lo) & (ids <= hi)


def read_mass(group, f):
    if "Masses" in group:
        return np.asarray(group["Masses"], dtype=float)
    mt = np.asarray(f["Header"].attrs.get("MassTable", np.zeros(6)))
    ptype = int(group.name.rsplit("PartType", 1)[-1])
    if mt[ptype] <= 0:
        raise ValueError(f"{group.name} needs Masses or a positive Header/MassTable")
    return np.full(len(group["ParticleIDs"]), mt[ptype], dtype=float)


def weighted_median(x, w):
    order = np.argsort(x)
    cumulative = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def stellar_core_center(pos, mass, radius_kpc=8.0):
    """Track the original LMC stellar core, rather than its stripped mean."""
    center = np.average(pos, axis=0, weights=mass)
    mask = np.ones(len(pos), dtype=bool)
    min_count = min(40, max(10, len(pos)//4))
    for _ in range(30):
        distance = np.linalg.norm(pos - center, axis=1)
        active = distance[mask]
        if len(active) <= 2*min_count:
            break
        cutoff = np.quantile(active, 0.80)
        new_mask = distance <= cutoff
        if new_mask.sum() < min_count:
            break
        mask = new_mask
        center = np.average(pos[mask], axis=0, weights=mass[mask])
        if cutoff <= radius_kpc:
            break
    final = np.linalg.norm(pos - center, axis=1) <= radius_kpc
    if final.sum() < min_count:
        final = mask
    return np.average(pos[final], axis=0, weights=mass[final])


def snapshot_data(info, ref, settings, warnings):
    ids = ref["ids"]
    path = info["path"]
    with h5py.File(path, "r") as f:
        require_single_snapshot(f)
        require_noncosmological(f)
        current_units, _ = resolve_units(f)
        if any(not np.isclose(current_units[k], value, rtol=1e-10, atol=0)
               for k, value in settings.get('metadata_cgs', current_units).items()):
            raise ValueError(f'{path}: unit metadata differs between satellite/control inputs')
        if "PartType0" not in f or "PartType4" not in f:
            raise ValueError(f"{path} lacks PartType0 or PartType4")
        gas, stars = f["PartType0"], f["PartType4"]
        needed = ("Coordinates", "Velocities", "ParticleIDs", "Density", "InternalEnergy")
        missing = [n for n in needed if n not in gas]
        if missing:
            raise ValueError(f"{path} lacks PartType0 fields {missing}; Density and "
                             "InternalEnergy must be in the saved snapshot")
        sid = np.asarray(stars["ParticleIDs"], dtype=np.uint64)
        spos = np.asarray(stars["Coordinates"], dtype=float) * settings["length"]
        svel = np.asarray(stars["Velocities"], dtype=float) * settings["velocity"]
        smass = read_mass(stars, f)
        old = mask_range(sid, ids, "MW_DISK") | mask_range(sid, ids, "MW_BULGE")
        if not np.any(old):
            raise ValueError(f"{path}: no original MW stars in the reference ID ranges")
        if "PartType5" in f and "ParticleIDs" in f["PartType5"]:
            bh = f["PartType5"]
            bi = np.flatnonzero(np.asarray(bh["ParticleIDs"], dtype=np.uint64) == ref["bh_id"])
            if len(bi) != 1:
                raise ValueError(f"{path}: expected one MW BH ID={ref['bh_id']}, found {len(bi)}")
            center = np.asarray(bh["Coordinates"][bi[0]], dtype=float) * settings["length"]
        else:
            raise ValueError(f"{path}: original MW BH ID missing; no reliable center")
        sep = np.linalg.norm(spos[old] - center, axis=1)
        inner = sep < 5.0
        if inner.sum() < 32:
            inner = sep < 10.0
        if inner.sum() < 32:
            raise ValueError(f"{path}: fewer than 32 original MW stars within 10 kpc")
        bulk = np.average(svel[old][inner], axis=0, weights=smass[old][inner])
        old_disk = mask_range(sid, ids, "MW_DISK")
        disk_pos = spos[old_disk] - center
        disk_vel = svel[old_disk] - bulk
        disk_m = smass[old_disk]
        if len(disk_m):
            j = np.sum(np.cross(disk_pos, disk_vel) * disk_m[:, None], axis=0)
            tilt = float(np.degrees(np.arctan2(np.linalg.norm(j[:2]), abs(j[2]))))
            if tilt > 5.0:
                warnings.append(f"{path}: original disk tilts {tilt:.1f} deg from global Z; "
                                "corridor still uses global Z, inspect jet direction")
        else:
            tilt = float("nan")
        lmc_distance = float("nan")
        if "LMC_STAR_ID_START" in ids:
            lmc_stars = mask_range(sid, ids, "LMC_STAR")
            if np.any(lmc_stars):
                lmc_distance = float(np.linalg.norm(
                    stellar_core_center(spos[lmc_stars], smass[lmc_stars]) - center
                ))
        gid = np.asarray(gas["ParticleIDs"], dtype=np.uint64)
        gpos = np.asarray(gas["Coordinates"], dtype=float) * settings["length"] - center
        gvel = np.asarray(gas["Velocities"], dtype=float) * settings["velocity"] - bulk
        gm = read_mass(gas, f)
        gd = np.asarray(gas["Density"], dtype=float)
        gu = np.asarray(gas["InternalEnergy"], dtype=float)
        if any(len(x) != len(gid) for x in (gpos, gvel, gm, gd, gu)):
            raise ValueError(f"{path}: gas fields differ in number of rows")
        if (np.any(~np.isfinite(gpos)) or np.any(~np.isfinite(gvel))
                or np.any(~np.isfinite(gm)) or np.any(~np.isfinite(gd))
                or np.any(gd <= 0) or np.any(gm <= 0)):
            raise ValueError(f"{path}: invalid or non-positive gas Density/Masses")
        if np.any(~np.isfinite(gu)) or np.any(gu < 0):
            raise ValueError(f"{path}: invalid gas InternalEnergy")
        unit_mass_g = settings["mass"] * MSUN_G
        unit_length_cm = settings["length"] * KPC_CM
        rho_g = gd * unit_mass_g / unit_length_cm**3
        u_cgs = gu * (settings["velocity"] * 1e5)**2
        temp, temperature_source = gas_temperature(gas, settings["velocity"]*1e5,
                                                   gamma=GAMMA, fallback_mu=settings["mu"])
        print(f"[temperature] {path}: {temperature_source}")
        gas_vol_code = gm / gd
        mw = mask_range(gid, ids, "MW_CGM") & (temp >= 1e5)
        if "LMC_GAS_ID_START" in ids and ids["LMC_GAS_ID_END"] >= ids["LMC_GAS_ID_START"]:
            lmc = mask_range(gid, ids, "LMC_GAS") & (temp >= 1e5)
        else:
            lmc = np.zeros(len(gid), dtype=bool)
        cgm_origin = mask_range(gid, ids, "MW_CGM")
        star_cgm = mask_range(sid, ids, "MW_CGM")
        rgas = np.linalg.norm(gpos, axis=1)
        rstar = np.linalg.norm(spos - center, axis=1)
        core_rows = []
        for radius in (5.0, 20.0):
            inside = cgm_origin & (rgas < radius)
            cold = inside & (temp < 1e5)
            hot = inside & (temp >= 1e5)
            core_rows.append({
                "radius_kpc": radius,
                "mw_cgm_origin_hot_gas_msun": float(np.sum(gm[hot]) * settings["mass"]),
                "mw_cgm_origin_cold_gas_msun": float(np.sum(gm[cold]) * settings["mass"]),
                "mw_cgm_origin_stars_msun": float(np.sum(smass[star_cgm & (rstar < radius)]) * settings["mass"]),
                "mw_cgm_origin_cold_gas_n": int(cold.sum()),
            })
    result = {
        "path": path, "time_code": info["code"], "elapsed_gyr": info["elapsed"],
        "center_kpc": center.tolist(), "bulk_kms": bulk.tolist(),
        "stellar_disk_tilt_deg": tilt, "lmc_distance_kpc": lmc_distance,
        "pos": gpos, "vel": gvel, "mass_code": gm, "volume_code": gas_vol_code,
        "rho_g_cm3": rho_g, "u_cgs": u_cgs, "temperature_K": temp,
        "temperature_source": temperature_source,
        "mask_mw": mw, "mask_lmc": lmc, "core_rows": core_rows,
    }
    if not mw.any():
        warnings.append(f"{path}: no hot MW CGM gas identified; check original IDs/T")
    return result


def weighted_mean(a, w):
    return float(np.average(a, weights=w)) if len(a) else float("nan")


def gas_stats(data, select, settings):
    n = int(select.sum())
    keys = ("n_gas", "mass_msun", "n_total_cm3", "n_median_cm3",
            "temperature_mass_median_K", "thermal_p_over_k_K_cm3", "vx_kms",
            "vy_kms", "vz_kms", "vxy_kms", "bulk_xy_p_over_k_K_cm3",
            "total_xy_stress_over_k_K_cm3", "turb_xy_stress_over_k_K_cm3")
    if n == 0:
        return {k: (0 if k in ("n_gas", "mass_msun") else float("nan")) for k in keys}
    mass = data["mass_code"][select]
    vol = data["volume_code"][select]
    rho = data["rho_g_cm3"][select]
    u = data["u_cgs"][select]
    vel = data["vel"][select]
    total_vol = np.sum(vol)
    rho_bar = (np.sum(mass) / total_vol) * (
        settings["mass"] * MSUN_G / (settings["length"] * KPC_CM)**3
    )
    vb = np.average(vel, axis=0, weights=mass)
    speed2 = np.sum(vel[:, :2]**2, axis=1)
    pressure = (GAMMA - 1) * np.average(rho * u, weights=vol) / K_B
    stress = np.average(rho * speed2, weights=vol) * (1e5**2) / K_B
    bulk = rho_bar * np.sum(vb[:2]**2) * (1e5**2) / K_B
    return {
        "n_gas": n,
        "mass_msun": float(np.sum(mass) * settings["mass"]),
        "n_total_cm3": float(rho_bar / (settings["mu"] * M_P)),
        "n_median_cm3": float(np.median(rho / (settings["mu"] * M_P))),
        "temperature_mass_median_K": weighted_median(data["temperature_K"][select], mass),
        "thermal_p_over_k_K_cm3": float(pressure),
        "vx_kms": float(vb[0]), "vy_kms": float(vb[1]), "vz_kms": float(vb[2]),
        "vxy_kms": float(np.linalg.norm(vb[:2])),
        "bulk_xy_p_over_k_K_cm3": float(bulk),
        "total_xy_stress_over_k_K_cm3": float(stress),
        "turb_xy_stress_over_k_K_cm3": float(max(0.0, stress - bulk)),
    }


def shell_mask(data, hemisphere, lo, hi, rmax):
    pos = data["pos"]
    inside_z = (hemisphere * pos[:, 2] >= lo) & (hemisphere * pos[:, 2] < hi)
    return inside_z & (np.sum(pos[:, :2]**2, axis=1) < rmax*rmax)


def region_and_sector_rows(run, data, args, settings, wind_axis, warnings):
    region, sector, outer = [], [], []
    axes = {"X": np.array([1.0, 0.0]), "Y": np.array([0.0, 1.0])}
    if wind_axis is not None:
        axes["W"] = wind_axis
    for hemi, hemisphere in (("north", 1), ("south", -1)):
        for lo, hi in SHELLS:
            shape = shell_mask(data, hemisphere, lo, hi, args.rmax_kpc)
            context = {"run": run, "file": data["path"], "elapsed_gyr": data["elapsed_gyr"],
                       "to_jet_myr": 1000 * (data["elapsed_gyr"] - settings["jet_gyr"]),
                       "hemisphere": hemi, "zmin_abs_kpc": lo, "zmax_abs_kpc": hi,
                       "cyl_rmax_kpc": args.rmax_kpc}
            for origin, origin_mask in (("MW_CGM", data["mask_mw"]),
                                        ("LMC_CGM", data["mask_lmc"])):
                chosen = shape & origin_mask
                stats = gas_stats(data, chosen, settings)
                if origin == "MW_CGM" and stats["n_gas"] < args.min_particles:
                    warnings.append(f"{Path(data['path']).name} {hemi} {lo}-{hi} kpc: "
                                    f"only {stats['n_gas']} hot MW-CGM particles")
                region.append({**context, "origin": origin, **stats})
                if origin != "MW_CGM":
                    continue
                for axis, unit in axes.items():
                    coordinate = data["pos"][:, :2] @ unit
                    for side, side_mask in (("plus", coordinate >= 0), ("minus", coordinate < 0)):
                        sector.append({**context, "origin": origin, "axis": axis, "side": side,
                                       **gas_stats(data, chosen & side_mask, settings)})
        for lo, hi in OUTER_SHELLS:
            position = data["pos"]
            radius = np.linalg.norm(position, axis=1)
            shape = (radius >= lo) & (radius < hi) & (hemisphere * position[:, 2] >= 0)
            for origin, origin_mask in (("MW_CGM", data["mask_mw"]),
                                        ("LMC_CGM", data["mask_lmc"])):
                outer.append({"run": run, "file": data["path"],
                              "elapsed_gyr": data["elapsed_gyr"],
                              "to_jet_myr": 1000*(data["elapsed_gyr"]-settings["jet_gyr"]),
                              "hemisphere": hemi, "spherical_rmin_kpc": lo,
                              "spherical_rmax_kpc": hi, "origin": origin,
                              **gas_stats(data, shape & origin_mask, settings)})
    core = [{"run": run, "file": data["path"], "elapsed_gyr": data["elapsed_gyr"],
             "to_jet_myr": 1000*(data["elapsed_gyr"]-settings["jet_gyr"]), **r}
            for r in data["core_rows"]]
    return region, sector, outer, core


def pair_snapshots(sat, iso, tolerance_myr, warnings):
    pairs = []
    for s in sat:
        i = min(iso, key=lambda x: abs(x["elapsed"] - s["elapsed"]))
        gap_myr = abs(i["elapsed"] - s["elapsed"]) * 1000
        if gap_myr <= tolerance_myr + 1e-8:
            pairs.append((s, i, gap_myr))
        else:
            warnings.append(f"No control within {tolerance_myr:g} Myr of {s['path']} "
                            f"(nearest offset {gap_myr:.2f} Myr); excluded")
    if not pairs:
        raise RuntimeError("No elapsed-time-matched pairs. Check isolated start time, "
                           "branch alignment and --max-pair-offset-myr")
    return pairs


def safe_ratio(a, b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b > 0 else float("nan")


def sector_contrasts(rows, min_particles, wind_axis):
    groups = {}
    for row in rows:
        key = (row["run"], row["file"], row["hemisphere"],
               row["zmin_abs_kpc"], row["axis"])
        groups.setdefault(key, {})[row["side"]] = row
    axis_vectors = {"X": np.array([1.0, 0.0]), "Y": np.array([0.0, 1.0])}
    if wind_axis is not None:
        axis_vectors["W"] = wind_axis
    result = []
    for sides in groups.values():
        if "plus" not in sides or "minus" not in sides:
            raise RuntimeError("Incomplete sector +/- pairing")
        plus, minus = sides["plus"], sides["minus"]
        valid = min(plus["n_gas"], minus["n_gas"]) >= min_particles
        axis_vector = axis_vectors[plus["axis"]]
        result.append({
            **{k: plus[k] for k in ("run", "file", "elapsed_gyr", "to_jet_myr",
                                     "hemisphere", "zmin_abs_kpc", "zmax_abs_kpc",
                                     "cyl_rmax_kpc", "axis")},
            "both_sides_enough_particles": int(valid),
            "plus_n": plus["n_gas"], "minus_n": minus["n_gas"],
            "plus_density_cm3": plus["n_total_cm3"],
            "minus_density_cm3": minus["n_total_cm3"],
            "plus_to_minus_density_ratio": safe_ratio(
                plus["n_total_cm3"], minus["n_total_cm3"]) if valid else float("nan"),
            "plus_to_minus_thermal_p_ratio": safe_ratio(
                plus["thermal_p_over_k_K_cm3"],
                minus["thermal_p_over_k_K_cm3"]) if valid else float("nan"),
            "plus_to_minus_thermal_p_difference_over_k_K_cm3": (
                plus["thermal_p_over_k_K_cm3"] - minus["thermal_p_over_k_K_cm3"]
                if valid else float("nan")),
            "plus_vaxis_kms": float(np.dot(
                [plus["vx_kms"], plus["vy_kms"]], axis_vector)) if valid else float("nan"),
            "minus_vaxis_kms": float(np.dot(
                [minus["vx_kms"], minus["vy_kms"]], axis_vector)) if valid else float("nan"),
        })
    return result


def matched_rows(pairs, by_path, settings, rmax, min_particles, warnings):
    rows = []
    fac = settings["mu"] * M_P * (1e5**2) / K_B
    for s, i, gap in pairs:
        sd, idata = by_path[s["path"]], by_path[i["path"]]
        for hemi, hemisphere in (("north", 1), ("south", -1)):
            for lo, hi in SHELLS:
                sshape = shell_mask(sd, hemisphere, lo, hi, rmax)
                ishape = shell_mask(idata, hemisphere, lo, hi, rmax)
                sw = gas_stats(sd, sshape & sd["mask_mw"], settings)
                iw = gas_stats(idata, ishape & idata["mask_mw"], settings)
                sl = gas_stats(sd, sshape & sd["mask_lmc"], settings)
                enough = sw["n_gas"] >= min_particles and iw["n_gas"] >= min_particles
                if not enough:
                    warnings.append(f"Matched {hemi} {lo}-{hi} at {s['elapsed']:.4f} Gyr: "
                                    "too few MW-CGM particles; derived contrasts set to NaN")
                dvx, dvy = (sw["vx_kms"]-iw["vx_kms"], sw["vy_kms"]-iw["vy_kms"]) if enough else (float("nan"),)*2
                dvxy = math.hypot(dvx, dvy)
                n_sat = sw["n_total_cm3"]
                rows.append({
                    "satellite_file": sd["path"], "isolated_file": idata["path"],
                    "satellite_elapsed_gyr": s["elapsed"], "isolated_elapsed_gyr": i["elapsed"],
                    "pair_offset_myr": gap,
                    "to_jet_myr": 1000*(s["elapsed"]-settings["jet_gyr"]),
                    "hemisphere": hemi, "zmin_abs_kpc": lo, "zmax_abs_kpc": hi,
                    "cyl_rmax_kpc": rmax, "enough_particles": int(enough),
                    "mw_sat_n": sw["n_gas"], "mw_control_n": iw["n_gas"],
                    "lmc_gas_n": sl["n_gas"],
                    "mw_sat_n_total_cm3": n_sat,
                    "mw_control_n_total_cm3": iw["n_total_cm3"],
                    "mw_density_ratio_sat_to_control": safe_ratio(n_sat, iw["n_total_cm3"]) if enough else float("nan"),
                    "mw_thermal_p_ratio_sat_to_control": safe_ratio(
                        sw["thermal_p_over_k_K_cm3"], iw["thermal_p_over_k_K_cm3"]) if enough else float("nan"),
                    "mw_sat_thermal_p_over_k_K_cm3": sw["thermal_p_over_k_K_cm3"],
                    "mw_control_thermal_p_over_k_K_cm3": iw["thermal_p_over_k_K_cm3"],
                    "delta_vx_kms": dvx, "delta_vy_kms": dvy,
                    "delta_vperp_kms": dvxy,
                    "mw_sat_bulk_xy_p_over_k_K_cm3": sw["bulk_xy_p_over_k_K_cm3"],
                    "mw_control_bulk_xy_p_over_k_K_cm3": iw["bulk_xy_p_over_k_K_cm3"],
                    "induced_rho_dv2_over_k_K_cm3": n_sat * fac * dvxy**2 if enough else float("nan"),
                    "lmc_hot_gas_mass_msun": sl["mass_msun"],
                    "lmc_to_mw_hot_gas_mass_ratio": safe_ratio(sl["mass_msun"], sw["mass_msun"]),
                    "lmc_distance_kpc": sd["lmc_distance_kpc"],
                    "sat_stellar_disk_tilt_deg": sd["stellar_disk_tilt_deg"],
                })
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_latest(rows, output, min_particles):
    last = max(r["satellite_elapsed_gyr"] for r in rows)
    subset = [r for r in rows if r["satellite_elapsed_gyr"] == last]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = (
        ("mw_density_ratio_sat_to_control", "MW density: sat / control", 1),
        ("delta_vperp_kms", "Induced signed-mean XY velocity [km/s]", 0),
        ("mw_thermal_p_ratio_sat_to_control", "MW thermal pressure: sat / control", 1),
        ("induced_rho_dv2_over_k_K_cm3", "rho_sat |delta v_XY|² / kB [K cm⁻³]", 0),
    )
    for ax, (key, label, baseline) in zip(axes.flat, metrics):
        for hemi, color, marker in (("north", "tab:blue", "o"), ("south", "tab:orange", "s")):
            samples = sorted((r for r in subset if r["hemisphere"] == hemi),
                             key=lambda x: x["zmin_abs_kpc"])
            z = [(r["zmin_abs_kpc"]+r["zmax_abs_kpc"])/2 for r in samples]
            y = [float(r[key]) if r["enough_particles"] else float("nan") for r in samples]
            ax.plot(z, y, marker=marker, color=color, label=hemi)
        ax.axhline(baseline, color="gray", lw=0.8, linestyle=":")
        ax.set(ylabel=label, xlabel="|z| shell center [kpc]")
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.suptitle(f"Pre-jet MW-origin hot CGM, latest paired elapsed time {last:.4f} Gyr\n"
                 f"Minimum for valid contrast: {min_particles} particles in each branch")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    args = args_from_cli()
    sat_paths_all = files_from_tokens(args.satellite_snapshots)
    iso_paths_all = files_from_tokens(args.isolated_snapshots)
    metadata_cgs, _ = resolve_units(sat_paths_all[0])
    overrides = {
        'UnitLength_in_cm': None if args.unit_length_kpc is None else args.unit_length_kpc*KPC_CM,
        'UnitMass_in_g': None if args.unit_mass_msun is None else args.unit_mass_msun*MSUN_G,
        'UnitVelocity_in_cm_per_s': None if args.unit_velocity_kms is None else args.unit_velocity_kms*1e5,
    }
    cgs, unit_sources = resolve_units(sat_paths_all[0], overrides=overrides)
    args.unit_length_kpc = cgs['UnitLength_in_cm']/KPC_CM
    args.unit_mass_msun = cgs['UnitMass_in_g']/MSUN_G
    args.unit_velocity_kms = cgs['UnitVelocity_in_cm_per_s']/1e5
    print('[units]', cgs, unit_sources)
    if (args.pre_window_myr <= 0 or args.max_pair_offset_myr < 0 or args.rmax_kpc <= 0
            or args.min_particles < 1 or any(x <= 0 or not np.isfinite(x)
                                           for x in (args.unit_mass_msun, args.unit_length_kpc,
                                                     args.unit_velocity_kms))):
        raise ValueError("Window, Rmax, particle count and code units must be valid positive values")
    ref = load_reference(args.reference_ic)
    time_unit = args.unit_time_gyr if args.unit_time_gyr is not None else ref["unit_time_gyr"]
    if time_unit <= 0 or not np.isfinite(time_unit):
        raise ValueError("--unit-time-gyr must be positive")
    implied_time = UNIT_TIME_GYR * args.unit_length_kpc / args.unit_velocity_kms
    if abs(implied_time/time_unit-1) > .01:
        raise ValueError(f"Length/velocity units imply {implied_time:g} Gyr/code time, "
                         f"but --unit-time-gyr/reference says {time_unit:g}; correct the units")
    if args.jet_start_snapshot:
        jet_gyr = (header_time(args.jet_start_snapshot) - ref["stage2_start_code"]) * time_unit
        source = str(args.jet_start_snapshot.resolve())
    elif args.jet_start_elapsed_gyr is not None:
        jet_gyr = args.jet_start_elapsed_gyr
        source = "user-provided Stage-2 elapsed time"
    else:
        jet_gyr = (ref["target_time_code"]-ref["stage2_start_code"]) * time_unit
        source = "nominal present-day target in Stage-2 IC (verify actual jet start)"
    if not np.isfinite(jet_gyr) or jet_gyr < 0:
        raise ValueError("Jet start must be a finite nonnegative Stage-2 elapsed time")
    sat = select_times(sat_paths_all, ref["stage2_start_code"],
                       time_unit, jet_gyr, args.pre_window_myr)
    iso = select_times(iso_paths_all, args.isolated_start_code,
                       time_unit, jet_gyr, args.pre_window_myr)
    if not sat or not iso:
        raise RuntimeError(f"No pre-jet snapshots in chosen window: sat={len(sat)}, "
                           f"control={len(iso)}. Check launch time and start-code offsets")
    warnings = []
    pairs = pair_snapshots(sat, iso, args.max_pair_offset_myr, warnings)
    paths = {r["path"]: r for a, b, _ in pairs for r in (a, b)}
    settings = {"mass": args.unit_mass_msun, "length": args.unit_length_kpc,
                "velocity": args.unit_velocity_kms, "mu": .59, "jet_gyr": jet_gyr,
                "metadata_cgs": metadata_cgs}
    wind_axis = parse_vector(args.wind_vector) if args.wind_vector else None
    by_path = {}
    for path, info in paths.items():
        by_path[path] = snapshot_data(info, ref, settings, warnings)
    all_regions, all_sectors, all_outer, all_cores = [], [], [], []
    sat_paths = {r["path"] for r in sat}
    for path, data in by_path.items():
        run = "satellite" if path in sat_paths else "isolated"
        region, sector, outer, core = region_and_sector_rows(run, data, args, settings,
                                                             wind_axis, warnings)
        all_regions.extend(region)
        all_sectors.extend(sector)
        all_outer.extend(outer)
        all_cores.extend(core)
    matched = matched_rows(pairs, by_path, settings, args.rmax_kpc,
                           args.min_particles, warnings)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "prejet_region_history.csv", all_regions)
    write_csv(args.outdir / "prejet_sector_history.csv", all_sectors)
    write_csv(args.outdir / "prejet_sector_contrast.csv",
              sector_contrasts(all_sectors, args.min_particles, wind_axis))
    write_csv(args.outdir / "prejet_outer_history.csv", all_outer)
    write_csv(args.outdir / "prejet_core_history.csv", all_cores)
    write_csv(args.outdir / "prejet_matched_comparison.csv", matched)
    plot_latest(matched, args.outdir / "prejet_comparison.png", args.min_particles)
    if min(abs(r["elapsed"]-jet_gyr) for r in sat) > .012:
        warnings.append("Last pre-jet satellite snapshot is >12 Myr before launch; "
                        "near-launch environment was not directly measured")
    summary = {
        "jet_start_elapsed_gyr": jet_gyr, "jet_start_source": source,
        "time_unit_gyr_per_code": time_unit,
        "satellite_start_code": ref["stage2_start_code"],
        "isolated_start_code": args.isolated_start_code,
        "satellite_selected_n": len(sat), "isolated_selected_n": len(iso),
        "matched_pairs_n": len(pairs),
        "pairs": [{"satellite": s["path"], "isolated": i["path"],
                   "satellite_elapsed_gyr": s["elapsed"], "offset_myr": gap}
                  for s, i, gap in pairs],
        "region_shells_kpc": SHELLS, "outer_spherical_shells_kpc": OUTER_SHELLS,
        "cylindrical_rmax_kpc": args.rmax_kpc,
        "wind_axis_XY": wind_axis.tolist() if wind_axis is not None else None,
        "warning": list(dict.fromkeys(warnings)),
        "interpretation": [
            "Requires same relaxed host and physics for causal satellite/control differences",
            "Wind axis cannot be inferred from LMC location alone; inspect signed velocity and X/Y sectors",
            "rho_sat |delta v|^2 is an illustrative perturbation scale, not a measured injected wind",
            "Hot T uses fully ionized mu=0.59; effective-EOS cold ISM is excluded by original IDs",
            "No hard pass/fail threshold is implied by Mou wind normalization",
        ],
    }
    (args.outdir / "prejet_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Analyzed {len(pairs)} matched pre-jet pairs; latest elapsed "
          f"{max(x['satellite_elapsed_gyr'] for x in matched):.4f} Gyr")
    print(f"CSV, figure and summary: {args.outdir.resolve()}")
    for warning in summary["warning"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
