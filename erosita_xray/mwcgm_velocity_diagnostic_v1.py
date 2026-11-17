#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MW-CGM velocity diagnostic for GIZMO LMC/SMC runs.

The script deliberately avoids yt's unfiltered ``("gas", velocity_*)``
streamlines.  It

1. removes LMC/SMC gas by the ID ranges stored by the Stage-2 builder;
2. by default intersects the Lagrangian MW-CGM ID range with a configurable
   hot-phase cut, so cooled gas that retains its original CGM ID does not
   dominate the present-day hot-CGM map;
3. recentres every snapshot on the MW black hole (or host stellar core) and
   subtracts the instantaneous MW-core velocity;
4. by default writes a temporary MW-CGM-only GIZMO snapshot and lets yt perform
   the SPH slab projection using the original smoothing lengths;
5. writes quantitative velocity, Mach-number, ram-pressure and displacement
   summaries directly from the selected particles.

This makes the sign of the MW-CGM response distinguishable from the reflex
motion introduced when the full MW+LMC+SMC IC is put in its barycentric frame.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
import sys as _compat_sys
_compat_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gizmo_units import resolve_units, physical_units, require_noncosmological, MSUN_G, UNIT_ALIASES
from gizmo_diagnostics_compat import gas_temperature, is_snapshot_cache, require_single_snapshot
from gizmo_compat import load_gizmo


DEFAULT_UNIT_TIME_GYR = 0.9777922216807892
DEFAULT_MASS_UNIT_MSUN = 1.0e10
DEFAULT_VELOCITY_UNIT_CMS = 1.0e5
K_B_CGS = 1.380649e-16
M_P_CGS = 1.67262192369e-24
KMS_MYR_TO_KPC = 1.0227121650537077e-3


@dataclass
class SnapshotData:
    path: str
    time_code: float
    elapsed_gyr: float
    center_kpc: np.ndarray
    bulk_kms: np.ndarray
    gas_position_kpc: np.ndarray
    gas_velocity_raw_kms: np.ndarray
    gas_velocity_mw_kms: np.ndarray
    gas_mass_msun: np.ndarray
    gas_temperature_k: np.ndarray
    gas_ids: np.ndarray
    selection: np.ndarray
    selection_label: str
    lmc_position_kpc: np.ndarray | None
    lmc_velocity_kms: np.ndarray | None
    smc_position_kpc: np.ndarray | None
    smc_velocity_kms: np.ndarray | None
    units_cgs: dict | None = None
    temperature_source: str = ""


def _attrs(handle: h5py.File, group_name: str) -> dict[str, object]:
    if group_name not in handle:
        return {}
    return {str(k): v for k, v in handle[group_name].attrs.items()}


def collect_metadata(handle: h5py.File) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in (
        "ReservedParticleIDs",
        "MagellanicOrbitReference",
        "MagellanicModelReference",
        "MWModelReference",
        "ModelNotes",
    ):
        out.update(_attrs(handle, name))
    return out


def _scalar_int(value: object) -> int | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    try:
        return int(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None


def find_id_ranges(metadata: dict[str, object], token: str) -> list[tuple[int, int, str]]:
    """Find START/END attribute pairs whose common prefix contains token."""
    normalized = {str(k).upper(): v for k, v in metadata.items()}
    ranges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for key, value in normalized.items():
        if token.upper() not in key or not re.search(r"(?:_ID)?_START$", key):
            continue
        end_key = re.sub(r"START$", "END", key)
        if end_key not in normalized:
            continue
        start = _scalar_int(value)
        end = _scalar_int(normalized[end_key])
        if start is None or end is None or end < start or (start, end) in seen:
            continue
        ranges.append((start, end, key.rsplit("_", 1)[0]))
        seen.add((start, end))
    return sorted(ranges)


def mask_in_ranges(ids: np.ndarray, ranges: list[tuple[int, int, str]]) -> np.ndarray:
    mask = np.zeros(len(ids), dtype=bool)
    for start, end, _ in ranges:
        mask |= (ids >= np.uint64(start)) & (ids <= np.uint64(end))
    return mask


def masses_msun(group: h5py.Group, header: h5py.Group, ptype: int,
                 mass_unit_msun: float) -> np.ndarray:
    if "Masses" in group:
        return np.asarray(group["Masses"], dtype=np.float64) * mass_unit_msun
    table = np.asarray(header.attrs.get("MassTable", np.zeros(6)), dtype=float)
    if ptype >= len(table) or table[ptype] <= 0.0:
        raise RuntimeError(f"PartType{ptype} has neither Masses nor a positive MassTable entry")
    return np.full(len(group["Coordinates"]), table[ptype] * mass_unit_msun)


def temperature_from_internal_energy(internal_energy: np.ndarray, mu: float,
                                     gamma: float, velocity_unit_cms: float) -> np.ndarray:
    return (
        (gamma - 1.0) * mu * M_P_CGS / K_B_CGS
        * np.asarray(internal_energy, dtype=np.float64) * velocity_unit_cms**2
    )


def host_star_mask(ids: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    satellite_ranges = find_id_ranges(metadata, "LMC") + find_id_ranges(metadata, "SMC")
    return ~mask_in_ranges(ids, satellite_ranges)


def find_mw_center_and_bulk(handle: h5py.File, metadata: dict[str, object],
                            core_radius_kpc: float) -> tuple[np.ndarray, np.ndarray]:
    if "PartType5" in handle and len(handle["PartType5/Coordinates"]) > 0:
        bh_pos = np.asarray(handle["PartType5/Coordinates"], dtype=float)
        bh_ids = np.asarray(handle["PartType5/ParticleIDs"], dtype=np.uint64)
        wanted = _scalar_int(metadata.get("MW_BH_ID", 90_000_000))
        match = np.flatnonzero(bh_ids == np.uint64(wanted)) if wanted is not None else np.array([])
        center = bh_pos[int(match[0]) if match.size else 0]
    else:
        center = None

    if "PartType4" not in handle:
        raise RuntimeError("PartType4 is required to determine the MW-core velocity")
    stars = handle["PartType4"]
    spos = np.asarray(stars["Coordinates"], dtype=float)
    svel = np.asarray(stars["Velocities"], dtype=float)
    sids = np.asarray(stars["ParticleIDs"], dtype=np.uint64)
    host = host_star_mask(sids, metadata)
    if not np.any(host):
        raise RuntimeError("No host MW stars remain after excluding satellite ID ranges")
    if center is None:
        center = np.median(spos[host], axis=0)

    radius = np.linalg.norm(spos - center[None, :], axis=1)
    inner = host & (radius < core_radius_kpc)
    if np.count_nonzero(inner) < 32:
        inner = host & (radius < 2.0 * core_radius_kpc)
    if np.count_nonzero(inner) < 32:
        raise RuntimeError("Fewer than 32 host stars are available in the MW core")
    bulk = np.median(svel[inner], axis=0)
    return np.asarray(center, dtype=float), np.asarray(bulk, dtype=float)


def satellite_phase_space(handle: h5py.File, metadata: dict[str, object], token: str,
                          center: np.ndarray, bulk: np.ndarray,
                          mass_unit_msun: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "PartType4" not in handle:
        return None, None
    group = handle["PartType4"]
    ids = np.asarray(group["ParticleIDs"], dtype=np.uint64)
    ranges = [item for item in find_id_ranges(metadata, token) if "STAR" in item[2]]
    if not ranges:
        ranges = find_id_ranges(metadata, token)
    mask = mask_in_ranges(ids, ranges)
    if not np.any(mask):
        return None, None
    mass = masses_msun(group, handle["Header"], 4, mass_unit_msun)[mask]
    pos = np.average(np.asarray(group["Coordinates"], dtype=float)[mask], axis=0, weights=mass)
    vel = np.average(np.asarray(group["Velocities"], dtype=float)[mask], axis=0, weights=mass)
    return pos - center, vel - bulk


def select_mw_cgm(ids: np.ndarray, temperature: np.ndarray, radius: np.ndarray,
                  metadata: dict[str, object], mode: str, hot_threshold: float,
                  minimum_radius_kpc: float) -> tuple[np.ndarray, str]:
    satellite_ranges = find_id_ranges(metadata, "LMC") + find_id_ranges(metadata, "SMC")
    host = ~mask_in_ranges(ids, satellite_ranges)
    cgm_ranges = [item for item in find_id_ranges(metadata, "CGM")
                  if "LMC" not in item[2] and "SMC" not in item[2]]

    if mode in ("auto", "ids", "ids-hot") and cgm_ranges:
        selected = host & mask_in_ranges(ids, cgm_ranges)
        phase_label = ""
        if mode in ("auto", "ids-hot"):
            selected &= np.isfinite(temperature) & (temperature >= hot_threshold)
            if minimum_radius_kpc > 0.0:
                selected &= radius >= minimum_radius_kpc
            phase_label = f" and T >= {hot_threshold:.3g} K"
            if minimum_radius_kpc > 0.0:
                phase_label += f", r >= {minimum_radius_kpc:g} kpc"
        if np.any(selected):
            labels = ", ".join(f"{start}-{end}" for start, end, _ in cgm_ranges)
            return selected, (
                f"MW CGM by reserved particle IDs ({labels}){phase_label}"
            )
    if mode in ("ids", "ids-hot"):
        raise RuntimeError(
            f"--selection {mode} was requested but no usable MW-CGM ID range "
            "was found. "
            "Normal GIZMO snapshots omit /ReservedParticleIDs; add "
            "--reference-ic <the Stage-2 IC used for this run>, or provide both "
            "--mw-cgm-id-start and --mw-cgm-id-end."
        )
    if mode == "all-host":
        return host, "all MW PartType0 gas (satellite gas excluded)"

    selected = host & np.isfinite(temperature) & (temperature >= hot_threshold)
    if minimum_radius_kpc > 0.0:
        selected &= radius >= minimum_radius_kpc
    return selected, (
        f"MW hot CGM fallback: T >= {hot_threshold:.3g} K"
        + (f", r >= {minimum_radius_kpc:g} kpc" if minimum_radius_kpc > 0.0 else "")
    )


def load_snapshot(path: str, args: argparse.Namespace,
                  reference_metadata: dict[str, object]) -> SnapshotData:
    with h5py.File(path, "r") as handle:
        # Normal GIZMO snapshots usually do not preserve custom root groups such
        # as ReservedParticleIDs or MagellanicOrbitReference.  Start with the
        # metadata from the original Stage-2 IC, then let any snapshot-resident
        # values override it.
        metadata = dict(reference_metadata)
        metadata.update(collect_metadata(handle))
        if args.mw_cgm_id_start is not None:
            metadata["MW_CGM_ID_START"] = int(args.mw_cgm_id_start)
            metadata["MW_CGM_ID_END"] = int(args.mw_cgm_id_end)
        require_single_snapshot(handle)
        require_noncosmological(handle)
        overrides = dict(UnitMass_in_g=None if args.mass_unit_msun is None else args.mass_unit_msun*MSUN_G,
                         UnitVelocity_in_cm_per_s=args.velocity_unit_cms)
        cgs, unit_sources = resolve_units(handle, overrides=overrides)
        units = physical_units(cgs)
        center_code, bulk_code = find_mw_center_and_bulk(handle, metadata, args.core_radius/units['length_kpc'])
        center, bulk = center_code*units['length_kpc'], bulk_code*units['velocity_kms']
        gas = handle["PartType0"]
        pos = np.asarray(gas["Coordinates"], dtype=float)*units['length_kpc'] - center[None, :]
        vel_raw = np.asarray(gas["Velocities"], dtype=float)*units['velocity_kms']
        vel_mw = vel_raw - bulk[None, :]
        print('[units]', cgs, unit_sources)
        ids = np.asarray(gas["ParticleIDs"], dtype=np.uint64)
        mass = masses_msun(gas, handle["Header"], 0, units["mass_msun"])
        temperature, temperature_source = gas_temperature(gas, cgs['UnitVelocity_in_cm_per_s'],
                                                           gamma=args.gamma, fallback_mu=args.mu)
        print('[temperature]', temperature_source)

        radius = np.linalg.norm(pos, axis=1)
        selected, label = select_mw_cgm(
            ids, temperature, radius, metadata, args.selection,
            args.hot_threshold, args.minimum_radius,
        )
        lmc_pos, lmc_vel = satellite_phase_space(
            handle, metadata, "LMC", center_code, bulk_code, units["mass_msun"]
        )
        smc_pos, smc_vel = satellite_phase_space(
            handle, metadata, "SMC", center_code, bulk_code, units["mass_msun"]
        )

        lmc_pos = None if lmc_pos is None else lmc_pos*units['length_kpc']
        smc_pos = None if smc_pos is None else smc_pos*units['length_kpc']
        lmc_vel = None if lmc_vel is None else lmc_vel*units['velocity_kms']
        smc_vel = None if smc_vel is None else smc_vel*units['velocity_kms']
        time_code = float(handle["Header"].attrs.get("Time", 0.0))
        unit_time = units['time_gyr'] if args.unit_time_gyr is None else args.unit_time_gyr
        if 'UnitTime_Gyr' in metadata and not np.isclose(float(metadata['UnitTime_Gyr']), units['time_gyr'], rtol=1e-6):
            raise ValueError(f"{path}: UnitTime_Gyr conflicts with length/velocity units")
        insertion_time = float(np.asarray(metadata.get("InsertionTime_Code", 0.0)))
        elapsed_gyr = (time_code - insertion_time) * unit_time

    if not np.any(selected):
        raise RuntimeError(f"{path}: the MW-CGM selection is empty")
    return SnapshotData(
        path=path,
        time_code=time_code,
        elapsed_gyr=elapsed_gyr,
        center_kpc=center,
        bulk_kms=bulk,
        gas_position_kpc=pos,
        gas_velocity_raw_kms=vel_raw,
        gas_velocity_mw_kms=vel_mw,
        gas_mass_msun=mass,
        gas_temperature_k=temperature,
        gas_ids=ids,
        selection=selected,
        selection_label=label,
        lmc_position_kpc=lmc_pos,
        lmc_velocity_kms=lmc_vel,
        smc_position_kpc=smc_pos,
        smc_velocity_kms=smc_vel,
        units_cgs=cgs,
        temperature_source=temperature_source,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[good], weights=weights[good])) if np.any(good) else np.nan


def weighted_percentile(values: np.ndarray, weights: np.ndarray,
                        percentiles: tuple[float, ...]) -> np.ndarray:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return np.full(len(percentiles), np.nan)
    value = np.asarray(values[good], dtype=float)
    weight = np.asarray(weights[good], dtype=float)
    order = np.argsort(value)
    value = value[order]
    weight = weight[order]
    cumulative = np.cumsum(weight) - 0.5*weight
    cumulative /= np.sum(weight)
    return np.interp(np.asarray(percentiles)/100.0, cumulative, value)


def deposit_yz(data: SnapshotData, args: argparse.Namespace) -> dict[str, np.ndarray]:
    pos = data.gas_position_kpc
    vel = data.gas_velocity_mw_kms
    slab = data.selection & (np.abs(pos[:, 0]) <= args.slab_half_width)
    slab &= (np.abs(pos[:, 1]) <= args.fov) & (np.abs(pos[:, 2]) <= args.fov)
    if np.count_nonzero(slab) < 32:
        raise RuntimeError(f"{data.path}: fewer than 32 selected particles fall in the y-z slab")

    edges = np.linspace(-args.fov, args.fov, args.bins + 1)
    y = pos[slab, 1]
    z = pos[slab, 2]
    mass = data.gas_mass_msun[slab]
    mass_grid = np.histogram2d(y, z, bins=(edges, edges), weights=mass)[0].T
    count_grid = np.histogram2d(y, z, bins=(edges, edges))[0].T
    mom_y = np.histogram2d(y, z, bins=(edges, edges), weights=mass*vel[slab, 1])[0].T
    mom_z = np.histogram2d(y, z, bins=(edges, edges), weights=mass*vel[slab, 2])[0].T

    sigma = args.smoothing_bins
    smooth_mass = gaussian_filter(mass_grid, sigma=sigma, mode="constant")
    smooth_count = gaussian_filter(count_grid, sigma=sigma, mode="constant")
    smooth_my = gaussian_filter(mom_y, sigma=sigma, mode="constant")
    smooth_mz = gaussian_filter(mom_z, sigma=sigma, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        vy = smooth_my / smooth_mass
        vz = smooth_mz / smooth_mass
    valid = (smooth_count >= args.minimum_smoothed_count) & (smooth_mass > 0.0)
    vy[~valid] = np.nan
    vz[~valid] = np.nan
    pixel_area = (edges[1] - edges[0])**2
    surface_density = smooth_mass / pixel_area
    surface_density[~valid] = np.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "centers": centers,
        "surface_density": surface_density,
        "vy": vy,
        "vz": vz,
        "valid": valid,
        "slab_mask": slab,
        "renderer_label": "2D mass histogram + Gaussian smoothing",
    }


def write_filtered_gizmo_for_yt(data: SnapshotData, output_path: str,
                                args: argparse.Namespace) -> None:
    """Write the selected MW CGM only, with velocities already in the MW frame."""
    selected = np.asarray(data.selection, dtype=bool)
    with h5py.File(data.path, "r") as source:
        source_gas = source["PartType0"]
        n = len(source_gas['Coordinates'])
        if selected.shape != (n,):
            raise ValueError('MW-CGM mask does not match full gas particle order')
        required = ('Coordinates', 'ParticleIDs', 'Density')
        missing = [name for name in required if name not in source_gas]
        kernels = [name for name in ('KernelMaxRadius', 'SmoothingLength') if name in source_gas]
        if missing or not kernels:
            raise ValueError(f'{data.path}: missing {missing} or native kernel radius; '
                             'reconstruct on the full sample explicitly before filtering')
        if len(kernels) == 2:
            a, b = (np.asarray(source_gas[name]) for name in kernels)
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(f'{data.path}: conflicting native kernel radius aliases')
        cgs = data.units_cgs or resolve_units(source)[0]
        units = physical_units(cgs)
        with h5py.File(output_path, "x") as out:
            header = out.create_group('Header')
            for key, value in source['Header'].attrs.items():
                header.attrs[key] = value
            counts = np.zeros(6, dtype=np.uint32)
            counts[0] = np.count_nonzero(selected)
            header.attrs['NumPart_ThisFile'] = counts
            header.attrs['NumPart_Total'] = counts
            header.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)
            header.attrs['MassTable'] = np.zeros(6, dtype=np.float64)
            header.attrs['NumFilesPerSnapshot'] = 1
            for key, aliases in UNIT_ALIASES.items():
                for alias in aliases:
                    if alias in header.attrs:
                        del header.attrs[alias]
                header.attrs[key] = cgs[key]
            gas = out.create_group('PartType0')
            # Every particle array uses the same full-sample mask. Keep native
            # kernel radii, abundances and composite identity fields unchanged.
            for name, dataset in source_gas.items():
                if dataset.ndim and dataset.shape[0] == n:
                    target = gas.create_dataset(name, data=np.asarray(dataset)[selected])
                    for key, value in dataset.attrs.items():
                        target.attrs[key] = value
                elif dataset.ndim:
                    raise ValueError(f'{data.path}: misaligned field {name}, shape={dataset.shape}')
            replacements = dict(Velocities=data.gas_velocity_mw_kms[selected]/units['velocity_kms'],
                                Masses=data.gas_mass_msun[selected]/units['mass_msun'],
                                Temperature=data.gas_temperature_k[selected])
            for name, values in replacements.items():
                if name in gas:
                    del gas[name]
                gas.create_dataset(name, data=values)
            header.attrs['CompatibilityMask'] = 'same full-sample row mask; native kernel radii retained'


def orient_yt_maps_to_yz(
    surface_density: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    data: SnapshotData,
    slab_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Choose the FRB orientation that matches particle y-z coordinates.

    yt's plotting window and NumPy-facing FRB conventions have historically
    been easy to confuse.  A direct particle mass histogram supplies an
    independent orientation reference: rows must be z and columns must be y
    for Matplotlib ``imshow(origin="lower")``.
    """
    edges = np.linspace(-args.fov, args.fov, args.bins + 1)
    pos = data.gas_position_kpc[slab_mask]
    mass = data.gas_mass_msun[slab_mask]
    reference = np.histogram2d(
        pos[:, 1], pos[:, 2], bins=(edges, edges), weights=mass
    )[0].T
    reference = gaussian_filter(
        reference, sigma=max(1.0, args.smoothing_bins), mode="constant"
    )

    def correlation(candidate: np.ndarray) -> float:
        a = np.nan_to_num(np.asarray(candidate, dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        b = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.maximum(a, 0.0)
        b = np.maximum(b, 0.0)
        apos = a[a > 0.0]
        bpos = b[b > 0.0]
        if apos.size < 16 or bpos.size < 16:
            return np.nan
        a = np.log1p(a/max(float(np.median(apos)), np.finfo(float).tiny))
        b = np.log1p(b/max(float(np.median(bpos)), np.finfo(float).tiny))
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return np.nan
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    native_correlation = correlation(surface_density)
    transposed_correlation = correlation(surface_density.T)
    use_transpose = (
        np.isfinite(transposed_correlation)
        and (
            not np.isfinite(native_correlation)
            or transposed_correlation > native_correlation + 0.01
        )
    )
    if use_transpose:
        surface_density = surface_density.T
        vy = vy.T
        vz = vz.T
    orientation = "transposed" if use_transpose else "native"
    label = (
        f"FRB axis check: {orientation}; "
        f"corr(native)={native_correlation:.3f}, "
        f"corr(transposed)={transposed_correlation:.3f}"
    )
    return surface_density, vy, vz, label


def deposit_yz_yt(data: SnapshotData, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Use yt's GIZMO/SPH frontend for a density-weighted y-z slab projection."""
    try:
        import yt
    except ImportError as exc:
        raise ImportError(
            "--renderer yt requires yt in this Python environment. "
            "Use the same environment as the existing yt plotting script, "
            "or temporarily select --renderer histogram."
        ) from exc

    pos = data.gas_position_kpc
    slab_mask = data.selection & (np.abs(pos[:, 0]) <= args.slab_half_width)
    slab_mask &= (np.abs(pos[:, 1]) <= args.fov) & (np.abs(pos[:, 2]) <= args.fov)
    if np.count_nonzero(slab_mask) < 32:
        raise RuntimeError(f"{data.path}: fewer than 32 selected particles fall in the y-z slab")

    temp_parent = args.yt_temp_dir if args.yt_temp_dir else args.output_dir
    with tempfile.TemporaryDirectory(prefix="mwcgm_yt_", dir=temp_parent) as temp_dir:
        filtered_path = os.path.join(temp_dir, "mwcgm_only.hdf5")
        write_filtered_gizmo_for_yt(data, filtered_path, args)
        yt.funcs.mylog.setLevel(40)
        ds = load_gizmo(filtered_path, cache_dir=Path(temp_dir) / "yt_cache")
        center = ds.arr(data.center_kpc, "kpc")
        left = center + ds.arr(
            [-args.slab_half_width, -args.fov, -args.fov], "kpc"
        )
        right = center + ds.arr(
            [args.slab_half_width, args.fov, args.fov], "kpc"
        )
        source = ds.box(left, right)
        width = ((2.0*args.fov, "kpc"), (2.0*args.fov, "kpc"))
        common = dict(
            center=center,
            width=width,
            data_source=source,
            buff_size=(args.bins, args.bins),
        )
        density_plot = yt.ProjectionPlot(
            ds, "x", ("gas", "density"), weight_field=None, **common
        )
        velocity_plot = yt.ProjectionPlot(
            ds, "x", [("gas", "velocity_y"), ("gas", "velocity_z")],
            weight_field=("gas", "density"), **common
        )
        # Read the native FRB arrays first.  Their orientation is checked below
        # against a direct particle y-z histogram instead of blindly applying
        # a transpose.
        surface_density = np.asarray(
            density_plot.frb[("gas", "density")].to("Msun/kpc**2")
        )
        vy = np.asarray(
            velocity_plot.frb[("gas", "velocity_y")].to("km/s")
        )
        vz = np.asarray(
            velocity_plot.frb[("gas", "velocity_z")].to("km/s")
        )
        del density_plot, velocity_plot, ds

    surface_density, vy, vz, orientation_label = orient_yt_maps_to_yz(
        surface_density, vy, vz, data, slab_mask, args
    )
    finite_density = np.isfinite(surface_density) & (surface_density > 0.0)
    if not np.any(finite_density):
        raise RuntimeError(f"{data.path}: yt returned no finite MW-CGM surface density")
    density_max = float(np.nanmax(surface_density[finite_density]))
    valid = finite_density & (surface_density >= args.yt_density_fraction*density_max)
    valid &= np.isfinite(vy) & np.isfinite(vz)
    surface_density[~valid] = np.nan
    vy[~valid] = np.nan
    vz[~valid] = np.nan
    edges = np.linspace(-args.fov, args.fov, args.bins + 1)
    return {
        "centers": 0.5*(edges[:-1] + edges[1:]),
        "surface_density": surface_density,
        "vy": vy,
        "vz": vz,
        "valid": valid,
        "slab_mask": slab_mask,
        "renderer_label": (
            "yt SPH slab projection (horizontal y, vertical z); "
            + orientation_label
        ),
    }


def add_satellite_marker(ax, label: str, position: np.ndarray | None,
                         velocity: np.ndarray | None, color: str) -> None:
    if position is None or velocity is None:
        return
    y, z = position[1], position[2]
    ax.scatter([y], [z], marker="*", s=95, color=color, edgecolor="black",
               linewidth=0.5, zorder=7, label=label)
    scale = 0.05  # 200 km/s -> 10 kpc reference arrow
    ax.arrow(y, z, scale*velocity[1], scale*velocity[2], color=color,
             width=0.25, head_width=2.0, length_includes_head=True, zorder=6)


def plot_snapshot(data: SnapshotData, maps: dict[str, np.ndarray], args: argparse.Namespace,
                  output_png: str) -> None:
    centers = maps["centers"]
    extent = (-args.fov, args.fov, -args.fov, args.fov)
    density = maps["surface_density"]
    vy = maps["vy"]
    vz = maps["vz"]
    finite_density = density[np.isfinite(density) & (density > 0.0)]
    if finite_density.size == 0:
        raise RuntimeError(f"{data.path}: no valid surface-density pixels after masking")
    dmin, dmax = np.nanpercentile(finite_density, [5.0, 99.5])
    dmin = max(float(dmin), float(dmax)*1.0e-5)
    finite_vy = np.abs(vy[np.isfinite(vy)])
    vmax = max(args.minimum_velocity_scale,
               float(np.nanpercentile(finite_vy, 97.0)) if finite_vy.size else 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    image0 = axes[0].imshow(density, origin="lower", extent=extent, cmap="viridis",
                            norm=LogNorm(vmin=dmin, vmax=dmax), aspect="equal")
    fig.colorbar(image0, ax=axes[0], label=r"MW-CGM slab surface density [$M_\odot\,\mathrm{kpc}^{-2}$]")
    image1 = axes[1].imshow(vy, origin="lower", extent=extent, cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), aspect="equal")
    fig.colorbar(image1, ax=axes[1], label=r"$\langle v_y-v_{y,\mathrm{MW}}\rangle_M$ [km s$^{-1}$]")

    step = max(1, args.quiver_step)
    yy, zz = np.meshgrid(centers, centers)
    qmask = maps["valid"][::step, ::step]
    qy = np.where(qmask, vy[::step, ::step], np.nan)
    qz = np.where(qmask, vz[::step, ::step], np.nan)
    if args.quiver_mode == "vy":
        qz = np.zeros_like(qy)
    elif args.quiver_mode == "none":
        qy[:] = np.nan
        qz[:] = np.nan
    speed = np.hypot(qy, qz)
    keep = np.isfinite(speed) & (speed >= args.minimum_arrow_speed)
    qy = np.where(keep, qy, np.nan)
    qz = np.where(keep, qz, np.nan)
    for ax in axes:
        ax.quiver(yy[::step, ::step], zz[::step, ::step], qy, qz,
                  color="white" if ax is axes[0] else "black", alpha=0.82,
                  angles="xy", scale_units="xy", scale=args.quiver_scale,
                  width=0.0025, pivot="middle")
        ax.axhline(0.0, color="0.65", lw=0.5, alpha=0.5)
        ax.axvline(0.0, color="0.65", lw=0.5, alpha=0.5)
        ax.scatter([0.0], [0.0], marker="+", s=65, color="white" if ax is axes[0] else "black")
        add_satellite_marker(ax, "LMC", data.lmc_position_kpc, data.lmc_velocity_kms, "gold")
        add_satellite_marker(ax, "SMC", data.smc_position_kpc, data.smc_velocity_kms, "lime")
        ax.set_xlim(-args.fov, args.fov)
        ax.set_ylim(-args.fov, args.fov)
        ax.set_xlabel("y relative to MW [kpc]")
        ax.set_ylabel("z relative to MW [kpc]")
    vector_label = {
        "full": r"$(v_y,v_z)$ arrows",
        "vy": r"$v_y$-only arrows",
        "none": "no arrows",
    }[args.quiver_mode]
    axes[0].set_title(f"Selected MW gas: density + {vector_label}")
    axes[1].set_title(r"Signed MW-frame $v_y$ (blue: $-y$, red: $+y$)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{Path(data.path).name} | elapsed = {data.elapsed_gyr:.4f} Gyr "
        f"(Header Time={data.time_code:.6g})\n{data.selection_label}\n"
        f"renderer: {maps.get('renderer_label', 'unknown')}\n"
        f"subtracted MW-core velocity = ({data.bulk_kms[0]:.2f}, "
        f"{data.bulk_kms[1]:.2f}, {data.bulk_kms[2]:.2f}) km/s",
        fontsize=11,
    )
    fig.savefig(output_png, dpi=args.dpi)
    plt.close(fig)


def region_rows(data: SnapshotData, maps: dict[str, np.ndarray],
                args: argparse.Namespace) -> list[dict[str, object]]:
    pos = data.gas_position_kpc
    vel_raw = data.gas_velocity_raw_kms
    vel = data.gas_velocity_mw_kms
    mass = data.gas_mass_msun
    radius = np.linalg.norm(pos, axis=1)
    cylindrical_radius = np.hypot(pos[:, 0], pos[:, 1])
    regions = (
        ("selected_all", data.selection),
        ("r_lt_20", data.selection & (radius < 20.0)),
        ("r_20_50", data.selection & (radius >= 20.0) & (radius < 50.0)),
        ("r_50_100", data.selection & (radius >= 50.0) & (radius < 100.0)),
        (
            "bubble_zone",
            data.selection
            & (cylindrical_radius < args.bubble_radius)
            & (np.abs(pos[:, 2]) >= args.bubble_z_min)
            & (np.abs(pos[:, 2]) < args.bubble_z_max),
        ),
        ("yz_slab", maps["slab_mask"]),
    )
    rows = []
    for label, mask in regions:
        if not np.any(mask):
            continue
        w = mass[mask]
        vy_region = vel[mask, 1]
        mean_vy = weighted_mean(vy_region, w)
        vy_p16, vy_p50, vy_p84 = weighted_percentile(
            vy_region, w, (16.0, 50.0, 84.0)
        )
        rms_vy = np.sqrt(weighted_mean(vy_region**2, w))
        coherent_fraction = abs(mean_vy)/rms_vy if rms_vy > 0.0 else np.nan
        temperature = data.gas_temperature_k[mask]
        sound_speed = np.sqrt(
            args.gamma*K_B_CGS*np.maximum(temperature, 0.0)/(args.mu*M_P_CGS)
        )/1.0e5
        mean_sound_speed = weighted_mean(sound_speed, w)
        coherent_mach_y = (
            abs(mean_vy)/mean_sound_speed if mean_sound_speed > 0.0 else np.nan
        )
        ram_to_thermal = args.gamma*coherent_mach_y**2
        reference_wind_fraction = abs(mean_vy)/args.reference_wind_kms
        y_position = pos[mask, 1]
        mass_y_negative = float(np.sum(w[y_position < 0.0]))
        mass_y_positive = float(np.sum(w[y_position >= 0.0]))
        row: dict[str, object] = {
            "snapshot": os.path.basename(data.path),
            "time_code": data.time_code,
            "elapsed_gyr": data.elapsed_gyr,
            "region": label,
            "particle_count": int(np.count_nonzero(mask)),
            "mass_msun": float(np.sum(w)),
            "raw_mean_vy_kms": weighted_mean(vel_raw[mask, 1], w),
            "mwframe_mean_vx_kms": weighted_mean(vel[mask, 0], w),
            "mwframe_mean_vy_kms": mean_vy,
            "mwframe_mean_vz_kms": weighted_mean(vel[mask, 2], w),
            "mwframe_vy_p16_kms": float(vy_p16),
            "mwframe_vy_p50_kms": float(vy_p50),
            "mwframe_vy_p84_kms": float(vy_p84),
            "mwframe_vy_rms_kms": float(rms_vy),
            "coherent_fraction_absmean_over_rms_vy": float(coherent_fraction),
            "mass_fraction_vy_negative": float(np.sum(w[vy_region < 0.0]) / np.sum(w)),
            "mean_sound_speed_kms": float(mean_sound_speed),
            "coherent_mach_y": float(coherent_mach_y),
            "coherent_pram_over_pthermal": float(ram_to_thermal),
            "coherent_speed_fraction_of_reference_wind": float(reference_wind_fraction),
            "pram_fraction_vs_reference_same_density": float(reference_wind_fraction**2),
            "signed_drift_over_bubble_age_kpc": float(
                mean_vy*args.bubble_age_myr*KMS_MYR_TO_KPC
            ),
            "reference_wind_drift_over_bubble_age_kpc": float(
                args.reference_wind_kms*args.bubble_age_myr*KMS_MYR_TO_KPC
            ),
            "mass_ynegative_to_ypositive": (
                mass_y_negative/mass_y_positive if mass_y_positive > 0.0 else np.nan
            ),
        }
        rows.append(row)
    return rows


def write_json(data: SnapshotData, output_path: str) -> None:
    payload = {
        "snapshot": os.path.abspath(data.path),
        "time_code": data.time_code,
        "elapsed_gyr": data.elapsed_gyr,
        "mw_center_kpc": data.center_kpc.tolist(),
        "mw_core_bulk_velocity_km_s": data.bulk_kms.tolist(),
        "selection": data.selection_label,
        "selected_particle_count": int(np.count_nonzero(data.selection)),
        "lmc_position_relative_mw_kpc": None if data.lmc_position_kpc is None else data.lmc_position_kpc.tolist(),
        "lmc_velocity_relative_mw_km_s": None if data.lmc_velocity_kms is None else data.lmc_velocity_kms.tolist(),
        "smc_position_relative_mw_kpc": None if data.smc_position_kpc is None else data.smc_position_kpc.tolist(),
        "smc_velocity_relative_mw_km_s": None if data.smc_velocity_kms is None else data.smc_velocity_kms.tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def expand_inputs(items: list[str]) -> list[str]:
    paths: list[str] = []
    for item in items:
        matches = sorted(glob.glob(item))
        paths.extend(matches if matches else [item])
    unique = []
    seen = set()
    for path in paths:
        if is_snapshot_cache(path):
            continue
        absolute = os.path.abspath(path)
        if absolute not in seen:
            unique.append(path)
            seen.add(absolute)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MW-CGM velocities in the instantaneous MW rest frame."
    )
    parser.add_argument("snapshots", nargs="+", help="Snapshot paths or quoted glob patterns")
    parser.add_argument("--output-dir", default="mwcgm_velocity_diagnostic")
    parser.add_argument(
        "--reference-ic", default=None,
        help=(
            "Original Stage-2 IC used to start this run. Its custom metadata "
            "provides MW-CGM and satellite ID ranges that normal GIZMO snapshots omit."
        ),
    )
    parser.add_argument(
        "--mw-cgm-id-start", type=int, default=None,
        help="Inclusive MW-CGM ParticleID lower bound; use together with --mw-cgm-id-end.",
    )
    parser.add_argument(
        "--mw-cgm-id-end", type=int, default=None,
        help="Inclusive MW-CGM ParticleID upper bound; use together with --mw-cgm-id-start.",
    )
    parser.add_argument(
        "--selection",
        choices=("auto", "ids-hot", "ids", "hot", "all-host"),
        default="auto",
        help=(
            "auto/ids-hot: MW-CGM IDs intersected with the hot phase; "
            "ids: all Lagrangian MW-CGM particles including cooled gas."
        ),
    )
    parser.add_argument(
        "--hot-threshold", type=float, default=3.0e5,
        help="K; applied by auto, ids-hot, and hot selections.",
    )
    parser.add_argument("--minimum-radius", type=float, default=0.0, help="kpc; optional extra cut for the hot fallback")
    parser.add_argument("--core-radius", type=float, default=5.0, help="kpc; host stars used for MW bulk velocity")
    parser.add_argument("--fov", type=float, default=100.0, help="half-width in y and z [kpc]")
    parser.add_argument("--slab-half-width", type=float, default=5.0, help="|x| limit about the MW [kpc]")
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument(
        "--renderer", choices=("yt", "histogram"), default="yt",
        help="yt=SPH projection from a temporary MW-CGM-only GIZMO snapshot (default).",
    )
    parser.add_argument(
        "--yt-density-fraction", type=float, default=1.0e-5,
        help="Mask yt pixels below this fraction of the peak MW-CGM surface density.",
    )
    parser.add_argument(
        "--yt-temp-dir", default=None,
        help="Optional parent directory for the automatically removed filtered yt snapshot.",
    )
    parser.add_argument("--smoothing-bins", type=float, default=1.4)
    parser.add_argument("--minimum-smoothed-count", type=float, default=0.08)
    parser.add_argument("--quiver-step", type=int, default=9)
    parser.add_argument(
        "--quiver-mode", choices=("vy", "full", "none"), default="vy",
        help=(
            "vy draws horizontal transverse-flow arrows (default); full draws "
            "the complete y-z vector and can be dominated by vertical motion."
        ),
    )
    parser.add_argument("--quiver-scale", type=float, default=18.0,
                        help="Matplotlib quiver scale; smaller means longer arrows")
    parser.add_argument("--minimum-arrow-speed", type=float, default=2.0, help="km/s")
    parser.add_argument("--minimum-velocity-scale", type=float, default=20.0, help="km/s colorbar half-range floor")
    parser.add_argument("--mass-unit-msun", type=float, default=None, help="Explicit mass unit override; default reads snapshot metadata")
    parser.add_argument("--velocity-unit-cms", type=float, default=None, help="Explicit velocity unit override; default reads snapshot metadata")
    parser.add_argument("--unit-time-gyr", type=float, default=None, help="Explicit time unit override; default length/velocity")
    parser.add_argument("--mu", type=float, default=0.59)
    parser.add_argument("--gamma", type=float, default=5.0/3.0)
    parser.add_argument(
        "--bubble-age-myr", type=float, default=19.0,
        help="Age used to convert coherent transverse velocity into drift distance.",
    )
    parser.add_argument(
        "--reference-wind-kms", type=float, default=160.0,
        help="Comparison transverse wind speed (Mou et al. fiducial: 160 km/s).",
    )
    parser.add_argument(
        "--bubble-radius", type=float, default=20.0,
        help="Cylindrical radius of the bubble-zone diagnostic [kpc].",
    )
    parser.add_argument(
        "--bubble-z-min", type=float, default=2.0,
        help="Minimum |z| of the bubble-zone diagnostic [kpc]; excludes the disk.",
    )
    parser.add_argument(
        "--bubble-z-max", type=float, default=20.0,
        help="Maximum |z| of the bubble-zone diagnostic [kpc].",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.fov <= 0.0 or args.slab_half_width <= 0.0 or args.bins < 16:
        parser.error("--fov and --slab-half-width must be positive; --bins must be >=16")
    if not (0.0 < args.yt_density_fraction < 1.0):
        parser.error("--yt-density-fraction must be between 0 and 1")
    if args.bubble_age_myr <= 0.0:
        parser.error("--bubble-age-myr must be positive")
    if args.reference_wind_kms <= 0.0:
        parser.error("--reference-wind-kms must be positive")
    if (args.bubble_radius <= 0.0 or args.bubble_z_min < 0.0
            or args.bubble_z_max <= args.bubble_z_min):
        parser.error(
            "the bubble zone requires Rmax > 0 and 0 <= |z|min < |z|max"
        )
    if args.yt_temp_dir is not None and not os.path.isdir(args.yt_temp_dir):
        parser.error("--yt-temp-dir must be an existing directory")
    if (args.mw_cgm_id_start is None) != (args.mw_cgm_id_end is None):
        parser.error("--mw-cgm-id-start and --mw-cgm-id-end must be supplied together")
    if (args.mw_cgm_id_start is not None
            and (args.mw_cgm_id_start < 0 or args.mw_cgm_id_end < args.mw_cgm_id_start)):
        parser.error("the manual MW-CGM ID range must satisfy 0 <= START <= END")
    return args


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.snapshots)
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing snapshots: " + ", ".join(missing))
    if args.reference_ic is not None and not os.path.isfile(args.reference_ic):
        raise FileNotFoundError(f"Missing --reference-ic: {args.reference_ic}")
    os.makedirs(args.output_dir, exist_ok=True)

    reference_metadata: dict[str, object] = {}
    if args.reference_ic is not None:
        with h5py.File(args.reference_ic, "r") as reference_handle:
            reference_metadata = collect_metadata(reference_handle)
        print(f"[metadata] loaded from Stage-2 IC: {args.reference_ic}")
        cgm_ranges = find_id_ranges(reference_metadata, "CGM")
        if cgm_ranges:
            print("[metadata] CGM ID ranges: " + ", ".join(
                f"{start}-{end} ({label})" for start, end, label in cgm_ranges
            ))
        else:
            print("[metadata] warning: reference IC contains no recognizable CGM ID range")

    all_rows: list[dict[str, object]] = []
    for path in paths:
        print(f"[MW-CGM] {path}")
        data = load_snapshot(path, args, reference_metadata)
        maps = deposit_yz_yt(data, args) if args.renderer == "yt" else deposit_yz(data, args)
        stem = Path(path).stem
        plot_snapshot(data, maps, args, os.path.join(args.output_dir, f"{stem}_mwcgm_velocity.png"))
        write_json(data, os.path.join(args.output_dir, f"{stem}_mwcgm_frame.json"))
        rows = region_rows(data, maps, args)
        all_rows.extend(rows)
        print(f"  selection: {data.selection_label}")
        print(f"  renderer: {maps['renderer_label']}")
        print(f"  MW core velocity subtracted: {data.bulk_kms} km/s")
        print(
            "  region                              <vy>     Mach_y  "
            "P_ram/P_th  drift  wind fraction"
        )
        for row in rows:
            print(
                f"  {row['region']:<32s} "
                f"{row['mwframe_mean_vy_kms']:>8.2f} "
                f"{row['coherent_mach_y']:>10.3f} "
                f"{row['coherent_pram_over_pthermal']:>11.4f} "
                f"{row['signed_drift_over_bubble_age_kpc']:>6.3f} kpc "
                f"{row['coherent_speed_fraction_of_reference_wind']:>8.3f}"
            )

    if all_rows:
        csv_path = os.path.join(args.output_dir, "mwcgm_velocity_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
