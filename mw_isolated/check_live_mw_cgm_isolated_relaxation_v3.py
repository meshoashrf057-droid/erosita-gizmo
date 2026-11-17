#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1：live MW＋冷 ISM＋热 CGM 联合弛豫检测脚本
================================================

配套初始条件生成器
------------------
make_ic_live_mw_cgm_isolated_v3.py

检测目的
--------
这一阶段包含 live DM、恒星盘、核球、冷 ISM 盘、热 CGM 和关闭反馈的
中心 BH；不含 LMC/SMC 或 AGN jet。检测集中于：

1. live DM 累积质量剖面是否相对 IC 发生过度变化；
2. 指数恒星盘的尺度长度、厚度和中心位置是否稳定；
3. 盘的 m=1/m=2 非轴对称振幅是否增长；
4. 盘比角动量、倾角、平均径向速度和反向旋转比例是否异常；
5. Hernquist 核球半质量半径是否明显膨胀或收缩；
6. 原始 CGM 在 50--200 kpc 的平均径向速度、热相比例、温度和质量保持；
7. 初始冷 ISM 的尺度长度、厚度、径向流、剩余气体和星形成转换；
8. 中心 5 kpc 的原始 CGM 是否出现灾难性凝聚；
9. TimeStep 输出中 dt<=1e-6 code units 的气体数量、来源和后半段趋势；
10. 上述量在模拟后半段是否仍有持续漂移。

本项目允许 cooling 造成约 30--50 km/s 的温和 CGM 内流，因此这一项
不会按“必须接近 0”判定。真正的 FAIL 条件是持续超过约 100 km/s、
R<200 kpc 气体大量损失、热相比例显著下降或中心质量增长数倍。
SFR 只输出而不作 PASS/FAIL 判据。

数据识别
--------
初始老恒星盘和核球都在 PartType4，热 CGM 和冷 ISM 都在 PartType0。
GIZMO 不保证把自定义
``StellarComponent`` 字段复制进普通 snapshot，所以本脚本始终从
``--reference-ic`` 的 ``ReservedParticleIDs`` 读取 ID 范围，再在每个
snapshot 中按 ParticleID 识别老恒星盘、核球、原始 CGM 和初始 ISM。
气体转化为 PartType4 后仍会按原始 ID 计入来源账本；额外生成的新 ID
则列入 other-new-stars。这与后续 LMC/SMC 工作流兼容。

中心定义
--------
默认优先使用 PartType5 中心 BH 的坐标作为银河中心；系统速度使用中心
5 kpc 内老恒星的质量加权平均值，避免单个 BH 速度噪声。若没有 BH，
则对老恒星执行 shrinking-sphere 定心。

建议命令
--------
python check_live_mw_cgm_isolated_relaxation_v3.py \
    --reference-ic mw_live_cgm_ism_isolated_v3.hdf5 \
    --snapshots output/snapshot_*.hdf5 \
    --outdir live_mw_cgm_ism_isolated_relaxation_v3

如果 shell 没有展开通配符，也可以写成：

python check_live_mw_cgm_isolated_relaxation_v3.py \
    --reference-ic mw_live_cgm_ism_isolated_v3.hdf5 \
    --snapshots 'output/snapshot_*.hdf5'

输出
----
live_mw_cgm_ism_isolated_summary.csv
live_mw_cgm_isolated_dm_profiles.csv
live_mw_cgm_ism_short_timestep.csv
live_mw_cgm_ism_isolated_relaxation.png（安装 matplotlib 时）

判定原则
--------
阈值用于筛选是否值得进入 LMC/SMC 阶段，不把 live 粒子涨落或正常
cooling inflow 当成失败。
最终结论分为 PASS / REVIEW / FAIL；必须同时查看后半段漂移率，不能只看
最后一个 snapshot。只有 1--2 个快照时，趋势量不足，脚本会给 REVIEW。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
from pathlib import Path as _CompatPath
import sys as _compat_sys
_compat_sys.path.insert(0, str(_CompatPath(__file__).resolve().parents[1]))
from gizmo_diagnostics_compat import gas_temperature, is_snapshot_cache, require_single_snapshot
from gizmo_units import resolve_units, physical_units, require_noncosmological

try:
    import h5py
except ImportError as exc:
    raise ImportError(
        "本脚本需要 h5py；请安装：python -m pip install h5py"
    ) from exc

import matplotlib
matplotlib.use("Agg")

# =============================================================================
# 0. GIZMO 单位与默认检测设置
# =============================================================================

UNIT_LENGTH_CM = 3.0856775814913673e21   # 1 kpc
UNIT_VELOCITY_CMS = 1.0e5                # 1 km/s
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
UNIT_TIME_GYR = UNIT_LENGTH_CM / UNIT_VELOCITY_CMS / SECONDS_PER_GYR
K_B_CGS = 1.380649e-16
M_P_CGS = 1.67262192369e-24
GAMMA = 5.0 / 3.0
MU_HOT = 0.59
MU_NEUTRAL = 1.22

DEFAULT_DM_RADII_KPC = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
DEFAULT_DISK_FIT_RMIN = 2.0
DEFAULT_DISK_FIT_RMAX = 12.0
DEFAULT_MODE_RMAX = 15.0
DEFAULT_SOLAR_RMIN = 7.7
DEFAULT_SOLAR_RMAX = 8.7
DEFAULT_SHORT_DT_THRESHOLD_CODE = 1.0e-6


@dataclass
class ParticleBlock:
    pos: np.ndarray
    vel: np.ndarray
    ids: np.ndarray
    mass: np.ndarray


@dataclass
class IdRanges:
    cgm_start: int
    cgm_end: int
    ism_start: int
    ism_end: int
    disk_start: int
    disk_end: int
    bulge_start: int
    bulge_end: int


# =============================================================================
# 1. 文件、粒子和参考信息读取
# =============================================================================

def natural_key(path):
    """让 snapshot_2 排在 snapshot_10 前面。"""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path)]


def expand_snapshot_paths(tokens):
    paths = []
    for token in tokens:
        matches = glob.glob(token)
        if matches:
            paths.extend(matches)
        elif os.path.isfile(token):
            paths.append(token)
    paths = sorted({os.path.abspath(p) for p in paths if not is_snapshot_cache(p)}, key=natural_key)
    if not paths:
        raise FileNotFoundError("没有找到任何 snapshot；请检查 --snapshots。")
    return paths


def snapshot_units(handle):
    require_single_snapshot(handle)
    require_noncosmological(handle)
    # Legacy model ICs predate unit metadata. These values are independently
    # confirmed in this project's mw_isolated.param; resolver reports the fallback.
    fallback = (dict(UnitLength_in_cm=3.085678e21, UnitMass_in_g=1.989e43,
                     UnitVelocity_in_cm_per_s=1e5)
                if 'MWModelReference' in handle and 'ReservedParticleIDs' in handle else None)
    cgs, sources = resolve_units(handle, fallback=fallback)
    return cgs, physical_units(cgs)


def read_mass_array(handle, ptype, n):
    group = handle[f"PartType{ptype}"]
    mass_unit = snapshot_units(handle)[1]["mass_msun"]
    if "Masses" in group:
        return np.asarray(group["Masses"], dtype=np.float64) * mass_unit
    mass_table = np.asarray(handle["Header"].attrs["MassTable"], dtype=np.float64)
    if mass_table[ptype] <= 0.0:
        raise RuntimeError(
            f"PartType{ptype} 没有 Masses，Header/MassTable 也没有常数质量。"
        )
    return np.full(n, mass_table[ptype] * mass_unit, dtype=np.float64)


def read_particle_block(handle, ptype):
    name = f"PartType{ptype}"
    if name not in handle:
        return None
    group = handle[name]
    units = snapshot_units(handle)[1]
    pos = np.asarray(group["Coordinates"], dtype=np.float64) * units["length_kpc"]
    vel = np.asarray(group["Velocities"], dtype=np.float64) * units["velocity_kms"]
    ids = np.asarray(group["ParticleIDs"], dtype=np.uint64)
    mass = read_mass_array(handle, ptype, len(pos))
    if pos.shape != (len(ids), 3) or vel.shape != pos.shape:
        raise RuntimeError(f"{name} 的 Coordinates/Velocities/IDs 形状不一致。")
    return ParticleBlock(pos=pos, vel=vel, ids=ids, mass=mass)


def read_id_ranges(reference_path):
    with h5py.File(reference_path, "r") as handle:
        if "ReservedParticleIDs" not in handle:
            raise RuntimeError(
                "reference IC 缺少 ReservedParticleIDs，无法可靠区分盘和核球。"
            )
        group = handle["ReservedParticleIDs"]
        for name in ("MW_ISM_ID_START", "MW_ISM_ID_END"):
            if name not in group.attrs:
                raise RuntimeError(
                    "reference IC 缺少冷 ISM 的保留 ID；请使用加入冷盘后的 "
                    "make_ic_live_mw_cgm_isolated_v3.py 重新生成 IC。"
                )
        return IdRanges(
            cgm_start=int(group.attrs["MW_CGM_ID_START"]),
            cgm_end=int(group.attrs["MW_CGM_ID_END"]),
            ism_start=int(group.attrs["MW_ISM_ID_START"]),
            ism_end=int(group.attrs["MW_ISM_ID_END"]),
            disk_start=int(group.attrs["MW_DISK_ID_START"]),
            disk_end=int(group.attrs["MW_DISK_ID_END"]),
            bulge_start=int(group.attrs["MW_BULGE_ID_START"]),
            bulge_end=int(group.attrs["MW_BULGE_ID_END"]),
        )


def id_mask(ids, start, end):
    return (ids >= np.uint64(start)) & (ids <= np.uint64(end))


def read_optional_1d(group, names, expected_length):
    """按候选名称读取可选的一维字段；不存在时返回 NaN 数组和 None。"""
    for name in names:
        if name in group:
            value = np.asarray(group[name], dtype=np.float64)
            if value.shape != (expected_length,):
                raise RuntimeError(f"{group.name}/{name} 形状应为 N。")
            return value, name
    return np.full(expected_length, np.nan, dtype=np.float64), None


def mass_weighted_mean(values, mass):
    total = np.sum(mass)
    if total <= 0.0:
        raise ValueError("质量和必须为正。")
    return np.sum(values * mass[:, None], axis=0) / total


# =============================================================================
# 2. 定心与坐标变换
# =============================================================================

def shrinking_sphere_center(pos, mass, min_particles=512):
    """简单稳健的 shrinking-sphere 定心，只在无 BH 时启用。"""
    center = mass_weighted_mean(pos, mass)
    radius = np.max(np.linalg.norm(pos - center[None, :], axis=1))
    selected = np.ones(len(pos), dtype=bool)
    while np.count_nonzero(selected) > min_particles and radius > 0.05:
        distance = np.linalg.norm(pos - center[None, :], axis=1)
        selected = distance <= radius
        if np.count_nonzero(selected) < min_particles:
            selected = np.argsort(distance)[:min_particles]
            center = mass_weighted_mean(pos[selected], mass[selected])
            break
        center = mass_weighted_mean(pos[selected], mass[selected])
        radius *= 0.8
    return center


def determine_center_and_velocity(handle, stars, ids):
    if "PartType5" in handle and len(handle["PartType5/Coordinates"]) > 0:
        center = np.asarray(handle["PartType5/Coordinates"][0], dtype=np.float64) * snapshot_units(handle)[1]["length_kpc"]
        bh_vel = np.asarray(handle["PartType5/Velocities"][0], dtype=np.float64) * snapshot_units(handle)[1]["velocity_kms"]
        method = "BH_position"
    else:
        old_mask = (
            id_mask(stars.ids, ids.disk_start, ids.disk_end)
            | id_mask(stars.ids, ids.bulge_start, ids.bulge_end)
        )
        center = shrinking_sphere_center(
            stars.pos[old_mask], stars.mass[old_mask]
        )
        bh_vel = np.zeros(3, dtype=np.float64)
        method = "shrinking_sphere"

    relative = stars.pos - center[None, :]
    inner = np.linalg.norm(relative, axis=1) < 5.0
    old = (
        id_mask(stars.ids, ids.disk_start, ids.disk_end)
        | id_mask(stars.ids, ids.bulge_start, ids.bulge_end)
    )
    inner &= old
    if np.count_nonzero(inner) >= 32:
        bulk_vel = mass_weighted_mean(stars.vel[inner], stars.mass[inner])
    else:
        bulk_vel = bh_vel
    return center, bulk_vel, method


def cylindrical_quantities(pos, vel):
    x, y, z = pos.T
    vx, vy, vz = vel.T
    radius = np.hypot(x, y)
    safe_r = np.maximum(radius, 1.0e-30)
    vr = (x * vx + y * vy) / safe_r
    vphi = (x * vy - y * vx) / safe_r
    phi = np.arctan2(y, x)
    return radius, z, phi, vr, vphi, vz


def gas_temperature_from_internal_energy(internal_energy, mu=MU_HOT):
    """
    由 GIZMO specific internal energy 估算热 CGM 温度。

    CGM 默认使用完全电离气体 mu=0.59；ISM 调用端改用中性气体
    mu=1.22。effective-EOS 气体的该温度仅用于粗略冷热相诊断。
    """
    u = np.asarray(internal_energy, dtype=np.float64)
    return (
        (GAMMA - 1.0)
        * u
        * UNIT_VELOCITY_CMS**2
        * mu
        * M_P_CGS
        / K_B_CGS
    )


def spherical_radial_velocity(pos, vel):
    radius = np.linalg.norm(pos, axis=1)
    safe = np.maximum(radius, 1.0e-30)
    vr = np.sum(pos * vel, axis=1) / safe
    return radius, vr


# =============================================================================
# 3. 单个快照的结构诊断
# =============================================================================

def enclosed_mass_profile(r, mass, radii):
    order = np.argsort(r)
    r_sorted = r[order]
    cumulative = np.cumsum(mass[order])
    index = np.searchsorted(r_sorted, radii, side="right") - 1
    result = np.zeros(len(radii), dtype=np.float64)
    valid = index >= 0
    result[valid] = cumulative[index[valid]]
    return result


def fourier_amplitude(phi, mass, mode):
    if len(phi) == 0 or np.sum(mass) <= 0.0:
        return np.nan
    value = np.sum(mass * np.exp(1j * mode * phi)) / np.sum(mass)
    return float(np.abs(value))


def fit_exponential_scale_length(radius, mass, rmin, rmax, nbins=20):
    bins = np.linspace(rmin, rmax, nbins + 1)
    m_bin, _ = np.histogram(radius, bins=bins, weights=mass)
    n_bin, _ = np.histogram(radius, bins=bins)
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    sigma = m_bin / area
    centers = 0.5 * (bins[1:] + bins[:-1])
    good = (n_bin >= 20) & (sigma > 0.0)
    if np.count_nonzero(good) < 5:
        return np.nan
    weight = np.sqrt(n_bin[good])
    slope, _ = np.polyfit(
        centers[good], np.log(sigma[good]), 1, w=weight
    )
    return float(-1.0 / slope) if slope < 0.0 else np.nan


def weighted_quantile(values, mass, quantile):
    order = np.argsort(values)
    v = values[order]
    w = mass[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0.0:
        return np.nan
    return float(np.interp(quantile * cumulative[-1], cumulative, v))


def disk_angular_momentum(pos, vel, mass):
    angular = np.cross(pos, vel) * mass[:, None]
    total = np.sum(angular, axis=0)
    mass_sum = np.sum(mass)
    specific = total / mass_sum
    norm = np.linalg.norm(total)
    unit = total / norm if norm > 0.0 else np.array([np.nan, np.nan, np.nan])
    return specific, unit


def inner_halo_axis_ratio(pos, mass, rmax=50.0):
    """用 reduced inertia tensor 粗略监测内晕形状；不是精细 halo-shape 拟合。"""
    r = np.linalg.norm(pos, axis=1)
    use = (r > 0.0) & (r < rmax)
    if np.count_nonzero(use) < 100:
        return np.nan
    x = pos[use]
    w = mass[use] / np.maximum(r[use] ** 2, 1.0e-20)
    tensor = np.einsum("n,ni,nj->ij", w, x, x) / np.sum(w)
    eig = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    if eig[0] <= 0.0 or eig[2] < 0.0:
        return np.nan
    return float(np.sqrt(eig[2] / eig[0]))


def analyze_file(
    path,
    ids,
    dm_radii,
    disk_fit_rmin,
    disk_fit_rmax,
    mode_rmax,
    solar_rmin,
    solar_rmax,
    short_dt_threshold_code,
):
    with h5py.File(path, "r") as handle:
        cgs, units = snapshot_units(handle)
        unit_time_gyr = units["time_gyr"]
        gas = read_particle_block(handle, 0)
        dm = read_particle_block(handle, 1)
        stars = read_particle_block(handle, 4)
        if gas is None or dm is None or stars is None:
            raise RuntimeError(
                f"{path} 必须同时包含 PartType0、PartType1 和 PartType4。"
            )
        gas_group = handle["PartType0"]
        gas_temp_hot, temp_source = gas_temperature(gas_group, cgs["UnitVelocity_in_cm_per_s"], fallback_mu=MU_HOT)
        gas_temp_neutral, _ = gas_temperature(gas_group, cgs["UnitVelocity_in_cm_per_s"], fallback_mu=MU_NEUTRAL)
        print(f"[temperature] {path}: {temp_source}")
        gas_timestep, timestep_field = read_optional_1d(
            gas_group, ("TimeStep", "Timestep", "TimestepSize"), len(gas.ids)
        )
        gas_sfr, sfr_field = read_optional_1d(
            gas_group, ("StarFormationRate", "SFR"), len(gas.ids)
        )
        gas_density, density_field = read_optional_1d(
            gas_group, ("Density",), len(gas.ids)
        )

        center, bulk_vel, center_method = determine_center_and_velocity(
            handle, stars, ids
        )
        box_size = float(handle["Header"].attrs.get("BoxSize", np.nan)) * units["length_kpc"]
        time_code = float(handle["Header"].attrs.get("Time", 0.0))
        time_gyr = time_code * unit_time_gyr

        dm_pos = dm.pos - center[None, :]
        gas_pos = gas.pos - center[None, :]
        gas_vel = gas.vel - bulk_vel[None, :]
        star_pos = stars.pos - center[None, :]
        star_vel = stars.vel - bulk_vel[None, :]

        disk_mask = id_mask(stars.ids, ids.disk_start, ids.disk_end)
        bulge_mask = id_mask(stars.ids, ids.bulge_start, ids.bulge_end)
        cgm_mask = id_mask(gas.ids, ids.cgm_start, ids.cgm_end)
        ism_mask = id_mask(gas.ids, ids.ism_start, ids.ism_end)
        cgm_star_mask = id_mask(stars.ids, ids.cgm_start, ids.cgm_end)
        ism_star_mask = id_mask(stars.ids, ids.ism_start, ids.ism_end)
        old_star_mask = disk_mask | bulge_mask
        new_star_mask = ~old_star_mask
        other_new_star_mask = new_star_mask & ~cgm_star_mask & ~ism_star_mask
        if np.count_nonzero(disk_mask) == 0 or np.count_nonzero(bulge_mask) == 0:
            raise RuntimeError(f"{path} 中没有找到参考 ID 范围内的盘或核球粒子。")
        if np.count_nonzero(cgm_mask) + np.count_nonzero(cgm_star_mask) == 0:
            raise RuntimeError(f"{path} 中没有找到参考 ID 范围内的 CGM 来源粒子。")
        if np.count_nonzero(ism_mask) + np.count_nonzero(ism_star_mask) == 0:
            raise RuntimeError(f"{path} 中没有找到参考 ID 范围内的 ISM 来源粒子。")

        dpos = star_pos[disk_mask]
        dvel = star_vel[disk_mask]
        dmass = stars.mass[disk_mask]
        bpos = star_pos[bulge_mask]
        bmass = stars.mass[bulge_mask]
        gpos = gas_pos[cgm_mask]
        gvel = gas_vel[cgm_mask]
        gmass = gas.mass[cgm_mask]
        gtemp = gas_temp_hot[cgm_mask]
        ipos = gas_pos[ism_mask]
        ivel = gas_vel[ism_mask]
        imass = gas.mass[ism_mask]
        itemp = gas_temp_neutral[ism_mask]

        cgm_spos = star_pos[cgm_star_mask]
        cgm_smass = stars.mass[cgm_star_mask]
        ism_smass = stars.mass[ism_star_mask]

        radius, z, phi, vr, vphi, vz = cylindrical_quantities(dpos, dvel)
        spherical_dm_r = np.linalg.norm(dm_pos, axis=1)
        spherical_bulge_r = np.linalg.norm(bpos, axis=1)
        spherical_cgm_r, cgm_vr = spherical_radial_velocity(gpos, gvel)
        spherical_cgm_star_r = np.linalg.norm(cgm_spos, axis=1)
        ism_radius, ism_z, ism_phi, ism_vr, ism_vphi, ism_vz = (
            cylindrical_quantities(ipos, ivel)
        )
        spherical_ism_r = np.linalg.norm(ipos, axis=1)

        cgm_r5 = spherical_cgm_r < 5.0
        cgm_r20 = spherical_cgm_r < 20.0
        cgm_r200 = spherical_cgm_r < 200.0
        cgm_shell_5_20= (
            (spherical_cgm_r >= 5.0) & (spherical_cgm_r < 20.0)
        )
        cgm_shell_20_200= (
            (spherical_cgm_r >= 20.0) & (spherical_cgm_r < 200.0)
        )
        cgm_shell_50_200 = (
            (spherical_cgm_r >= 50.0) & (spherical_cgm_r < 200.0)
        )
        hot_r200 = cgm_r200 & (gtemp >= 1.0e5)
        cold_r20 = cgm_r20 & (gtemp < 1.0e5)
        cgm_star_r5 = spherical_cgm_star_r < 5.0
        cgm_star_r20 = spherical_cgm_star_r < 20.0
        cgm_star_r200 = spherical_cgm_star_r < 200.0

        ism_vertical_sample = (
            (ism_radius >= disk_fit_rmin)
            & (ism_radius <= disk_fit_rmax)
        )
        ism_z_median = (
            weighted_quantile(
                np.abs(ism_z[ism_vertical_sample]),
                imass[ism_vertical_sample],
                0.5,
            )
            if np.any(ism_vertical_sample) else np.nan
        )
        ism_zd_est = ism_z_median / np.arctanh(0.5)
        ism_inner20 = spherical_ism_r < 20.0
        ism_cold = itemp < 1.0e5

        finite_dt = np.isfinite(gas_timestep) & (gas_timestep > 0.0)
        short_dt = finite_dt & (gas_timestep <= short_dt_threshold_code)
        cgm_short_dt = short_dt & cgm_mask
        ism_short_dt = short_dt & ism_mask

        density_cgs_factor = units["density_cgs"]
        gas_nh = (
            gas_density
            * density_cgs_factor
            * 0.76
            / M_P_CGS
        )

        mode_mask = (radius < mode_rmax) & (np.abs(z) < 3.0)
        solar = (radius >= solar_rmin) & (radius < solar_rmax)
        disk_j, disk_lhat = disk_angular_momentum(dpos, dvel, dmass)
        disk_com = mass_weighted_mean(dpos, dmass)

        # 对 sech^2(z/zd)，median(|z|)=atanh(0.5)*zd。
        vertical_sample = (
            (radius >= disk_fit_rmin)
            & (radius <= disk_fit_rmax)
        )
        z_median = weighted_quantile(
            np.abs(z[vertical_sample]), dmass[vertical_sample], 0.5
        )
        zd_est = z_median / np.arctanh(0.5)

        result = {
            "file": os.path.abspath(path),
            "time_code": time_code,
            "time_gyr": time_gyr,
            "center_method": center_method,
            "center_x_kpc": center[0],
            "center_y_kpc": center[1],
            "center_z_kpc": center[2],
            "center_offset_box_kpc": (
                float(np.linalg.norm(center - box_size / 2.0))
                if np.isfinite(box_size)
                else np.nan
            ),
            "bulk_speed_km_s": float(np.linalg.norm(bulk_vel)),
            "n_dm": len(dm.ids),
            "n_cgm": int(np.count_nonzero(cgm_mask)),
            "n_ism": int(np.count_nonzero(ism_mask)),
            "n_cgm_origin_stars": int(np.count_nonzero(cgm_star_mask)),
            "n_ism_origin_stars": int(np.count_nonzero(ism_star_mask)),
            "n_other_new_stars": int(np.count_nonzero(other_new_star_mask)),
            "n_disk": int(np.count_nonzero(disk_mask)),
            "n_bulge": int(np.count_nonzero(bulge_mask)),
            "disk_mass_msun": float(np.sum(dmass)),
            "bulge_mass_msun": float(np.sum(bmass)),
            "disk_com_offset_kpc": float(np.linalg.norm(disk_com)),
            "disk_rd_fit_kpc": fit_exponential_scale_length(
                radius, dmass, disk_fit_rmin, disk_fit_rmax
            ),
            "disk_zd_est_kpc": zd_est,
            "disk_z_rms_kpc": float(np.sqrt(np.average(z**2, weights=dmass))),
            "disk_a1_r15": fourier_amplitude(
                phi[mode_mask], dmass[mode_mask], 1
            ),
            "disk_a2_r15": fourier_amplitude(
                phi[mode_mask], dmass[mode_mask], 2
            ),
            "disk_jx_kpc_km_s": disk_j[0],
            "disk_jy_kpc_km_s": disk_j[1],
            "disk_jz_kpc_km_s": disk_j[2],
            "disk_lhat_x": disk_lhat[0],
            "disk_lhat_y": disk_lhat[1],
            "disk_lhat_z": disk_lhat[2],
            "disk_mean_vr_km_s": float(np.average(vr, weights=dmass)),
            "disk_mean_vz_km_s": float(np.average(vz, weights=dmass)),
            "disk_mean_vphi_km_s": float(np.average(vphi, weights=dmass)),
            "disk_counterrotating_fraction": float(
                np.sum(dmass[vphi < 0.0]) / np.sum(dmass)
            ),
            "disk_mass_fraction_r15_z3": float(
                np.sum(dmass[(radius < 15.0) & (np.abs(z) < 3.0)])
                / np.sum(dmass)
            ),
            "disk_mass_fraction_r30_z3": float(
                np.sum(dmass[(radius < 30.0) & (np.abs(z) < 3.0)])
                / np.sum(dmass)
            ),
            "solar_n": int(np.count_nonzero(solar)),
            "solar_vphi_median_km_s": (
                float(np.median(vphi[solar])) if np.any(solar) else np.nan
            ),
            "solar_sigma_r_km_s": (
                float(np.std(vr[solar])) if np.any(solar) else np.nan
            ),
            "solar_sigma_z_km_s": (
                float(np.std(vz[solar])) if np.any(solar) else np.nan
            ),
            "bulge_rhalf_kpc": weighted_quantile(
                spherical_bulge_r, bmass, 0.5
            ),
            "new_star_mass_msun": float(np.sum(stars.mass[new_star_mask])),
            "cgm_origin_star_mass_msun": float(np.sum(cgm_smass)),
            "ism_origin_star_mass_msun": float(np.sum(ism_smass)),
            "other_new_star_mass_msun": float(
                np.sum(stars.mass[other_new_star_mask])
            ),
            "sfr_total_msun_yr": (
                float(np.nansum(gas_sfr)) if sfr_field is not None else np.nan
            ),
            "sfr_cgm_origin_msun_yr": (
                float(np.nansum(gas_sfr[cgm_mask]))
                if sfr_field is not None else np.nan
            ),
            "sfr_ism_origin_msun_yr": (
                float(np.nansum(gas_sfr[ism_mask]))
                if sfr_field is not None else np.nan
            ),
            "ism_gas_mass_msun": float(np.sum(imass)),
            "ism_origin_baryon_mass_msun": float(
                np.sum(imass) + np.sum(ism_smass)
            ),
            "ism_gas_mass_r20_msun": float(np.sum(imass[ism_inner20])),
            "ism_cold_gas_fraction": (
                float(np.sum(imass[ism_cold]) / np.sum(imass))
                if np.sum(imass) > 0.0 else np.nan
            ),
            "ism_rd_fit_kpc": fit_exponential_scale_length(
                ism_radius, imass, disk_fit_rmin, disk_fit_rmax
            ),
            "ism_zd_est_kpc": ism_zd_est,
            "ism_z_rms_kpc": (
                float(np.sqrt(np.average(ism_z**2, weights=imass)))
                if np.sum(imass) > 0.0 else np.nan
            ),
            "ism_a1_r15": fourier_amplitude(
                ism_phi[(ism_radius < mode_rmax) & (np.abs(ism_z) < 3.0)],
                imass[(ism_radius < mode_rmax) & (np.abs(ism_z) < 3.0)],
                1,
            ),
            "ism_a2_r15": fourier_amplitude(
                ism_phi[(ism_radius < mode_rmax) & (np.abs(ism_z) < 3.0)],
                imass[(ism_radius < mode_rmax) & (np.abs(ism_z) < 3.0)],
                2,
            ),
            "ism_mean_vr_km_s": (
                float(np.average(ism_vr, weights=imass))
                if np.sum(imass) > 0.0 else np.nan
            ),
            "ism_mean_vz_km_s": (
                float(np.average(ism_vz, weights=imass))
                if np.sum(imass) > 0.0 else np.nan
            ),
            "ism_mean_vphi_km_s": (
                float(np.average(ism_vphi, weights=imass))
                if np.sum(imass) > 0.0 else np.nan
            ),
            "ism_temperature_median_K": (
                weighted_quantile(itemp, imass, 0.5)
                if np.sum(imass) > 0.0 else np.nan
            ),
            "dm_c_over_a_r50": inner_halo_axis_ratio(
                dm_pos, dm.mass, rmax=50.0
            ),
            "cgm_gas_mass_r5_msun": float(np.sum(gmass[cgm_r5])),
            "cgm_gas_mass_r20_msun": float(np.sum(gmass[cgm_r20])),
            "cgm_gas_mass_r200_msun": float(np.sum(gmass[cgm_r200])),
            "cgm_mass_r5_msun": float(
                np.sum(gmass[cgm_r5]) + np.sum(cgm_smass[cgm_star_r5])
            ),
            "cgm_mass_r20_msun": float(
                np.sum(gmass[cgm_r20]) + np.sum(cgm_smass[cgm_star_r20])
            ),
            "cgm_mass_r200_msun": float(
                np.sum(gmass[cgm_r200])
                + np.sum(cgm_smass[cgm_star_r200])
            ),
            "cgm_cold_mass_r20_msun": float(np.sum(gmass[cold_r20])),
            "cgm_condensed_mass_r5_msun": float(
                np.sum(gmass[cgm_r5 & (gtemp < 1.0e5)])
                + np.sum(cgm_smass[cgm_star_r5])
            ),
            "cgm_hot_mass_fraction_r200": (
                float(
                    np.sum(gmass[hot_r200])
                    / (
                        np.sum(gmass[cgm_r200])
                        + np.sum(cgm_smass[cgm_star_r200])
                    )
                )
                if (
                    np.sum(gmass[cgm_r200])
                    + np.sum(cgm_smass[cgm_star_r200])
                ) > 0.0 else np.nan
            ),
            "cgm_temperature_median_r20_200_K": (
                weighted_quantile(
                    gtemp[cgm_shell_20_200],
                    gmass[cgm_shell_20_200],
                    0.5,
                )
                if np.any(cgm_shell_20_200) else np.nan
            ),
            "cgm_mean_vr_r5_20_km_s": (
                float(np.average(
                    cgm_vr[cgm_shell_5_20],
                    weights=gmass[cgm_shell_5_20],
                ))
                if np.any(cgm_shell_5_20) else np.nan
            ),
            "cgm_mean_vr_r20_200_km_s": (
                float(np.average(
                    cgm_vr[cgm_shell_20_200],
                    weights=gmass[cgm_shell_20_200],
                ))
                if np.any(cgm_shell_20_200) else np.nan
            ),
            "cgm_mean_vr_r50_200_km_s": (
                float(np.average(
                    cgm_vr[cgm_shell_50_200],
                    weights=gmass[cgm_shell_50_200],
                ))
                if np.any(cgm_shell_50_200) else np.nan
            ),
            "timestep_field": timestep_field or "",
            "sfr_field": sfr_field or "",
            "density_field": density_field or "",
            "short_dt_threshold_code": short_dt_threshold_code,
            "gas_dt_min_code": (
                float(np.min(gas_timestep[finite_dt]))
                if np.any(finite_dt) else np.nan
            ),
            "gas_dt_min_yr": (
                float(np.min(gas_timestep[finite_dt]) * unit_time_gyr * 1.0e9)
                if np.any(finite_dt) else np.nan
            ),
            "gas_short_dt_count": int(np.count_nonzero(short_dt)),
            "gas_short_dt_fraction": (
                float(np.count_nonzero(short_dt) / np.count_nonzero(finite_dt))
                if np.any(finite_dt) else np.nan
            ),
            "cgm_short_dt_count": int(np.count_nonzero(cgm_short_dt)),
            "cgm_short_dt_fraction": (
                float(np.count_nonzero(cgm_short_dt) / np.count_nonzero(cgm_mask))
                if np.any(cgm_mask) and timestep_field is not None else np.nan
            ),
            "ism_short_dt_count": int(np.count_nonzero(ism_short_dt)),
            "ism_short_dt_fraction": (
                float(np.count_nonzero(ism_short_dt) / np.count_nonzero(ism_mask))
                if np.any(ism_mask) and timestep_field is not None else np.nan
            ),
            "short_dt_cgm_origin_fraction": (
                float(np.count_nonzero(cgm_short_dt) / np.count_nonzero(short_dt))
                if np.any(short_dt) else 0.0
            ),
            "short_dt_ism_origin_fraction": (
                float(np.count_nonzero(ism_short_dt) / np.count_nonzero(short_dt))
                if np.any(short_dt) else 0.0
            ),
            "gas_nh_max_cm3": (
                float(np.nanmax(gas_nh))
                if density_field is not None else np.nan
            ),
            "cgm_nh_max_cm3": (
                float(np.nanmax(gas_nh[cgm_mask]))
                if density_field is not None and np.any(cgm_mask) else np.nan
            ),
            "ism_nh_max_cm3": (
                float(np.nanmax(gas_nh[ism_mask]))
                if density_field is not None and np.any(ism_mask) else np.nan
            ),
        }

        dm_menc = enclosed_mass_profile(
            spherical_dm_r, dm.mass, np.asarray(dm_radii)
        )
        for radius_value, mass_value in zip(dm_radii, dm_menc):
            result[f"dm_menc_{radius_value:g}kpc_msun"] = float(mass_value)

        # 返回额外数组只用于可选的深入诊断，目前 summary 不直接保存粒子数据。
        return result


# =============================================================================
# 4. 相对 IC 的变化、漂移率和 PASS/REVIEW/FAIL
# =============================================================================

def safe_fractional_change(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return np.nan
    return float(value / reference - 1.0)


def angle_degrees(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0.0:
        return np.nan
    cosine = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def add_reference_comparisons(row, reference, dm_radii):
    dm_fractional = []
    for radius in dm_radii:
        key = f"dm_menc_{radius:g}kpc_msun"
        frac = safe_fractional_change(row[key], reference[key])
        row[f"dm_menc_frac_{radius:g}kpc"] = frac
        if np.isfinite(frac):
            dm_fractional.append(frac)
    row["dm_menc_rms_fraction"] = (
        float(np.sqrt(np.mean(np.asarray(dm_fractional) ** 2)))
        if dm_fractional else np.nan
    )
    row["dm_menc_maxabs_fraction"] = (
        float(np.max(np.abs(dm_fractional))) if dm_fractional else np.nan
    )

    for name in (
        "disk_rd_fit_kpc",
        "disk_zd_est_kpc",
        "disk_z_rms_kpc",
        "disk_jz_kpc_km_s",
        "disk_mass_fraction_r15_z3",
        "disk_mass_fraction_r30_z3",
        "bulge_rhalf_kpc",
        "dm_c_over_a_r50",
        "ism_rd_fit_kpc",
        "ism_zd_est_kpc",
        "ism_z_rms_kpc",
    ):
        row[f"{name}_fractional_change"] = safe_fractional_change(
            row[name], reference[name]
        )

    row["disk_tilt_deg"] = angle_degrees(
        (row["disk_lhat_x"], row["disk_lhat_y"], row["disk_lhat_z"]),
        (
            reference["disk_lhat_x"],
            reference["disk_lhat_y"],
            reference["disk_lhat_z"],
        ),
    )

    row["cgm_mass_r200_fractional_change"] = safe_fractional_change(
        row["cgm_mass_r200_msun"], reference["cgm_mass_r200_msun"]
    )
    row["cgm_temperature_r20_200_fractional_change"] = safe_fractional_change(
        row["cgm_temperature_median_r20_200_K"],
        reference["cgm_temperature_median_r20_200_K"],
    )
    row["cgm_mass_r5_ratio_to_initial"] = (
        float(row["cgm_mass_r5_msun"] / reference["cgm_mass_r5_msun"])
        if reference["cgm_mass_r5_msun"] > 0.0 else np.nan
    )
    row["cgm_condensed_mass_r5_ratio_to_initial"] = (
        float(
            row["cgm_condensed_mass_r5_msun"]
            / reference["cgm_mass_r5_msun"]
        )
        if reference["cgm_mass_r5_msun"] > 0.0 else np.nan
    )
    row["ism_gas_mass_fraction_remaining"] = (
        float(row["ism_gas_mass_msun"] / reference["ism_gas_mass_msun"])
        if reference["ism_gas_mass_msun"] > 0.0 else np.nan
    )
    row["ism_origin_mass_fraction_remaining"] = (
        float(
            row["ism_origin_baryon_mass_msun"]
            / reference["ism_origin_baryon_mass_msun"]
        )
        if reference["ism_origin_baryon_mass_msun"] > 0.0 else np.nan
    )


def linear_slope(time, values, tail_fraction):
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]
    if len(time) < 3:
        return np.nan
    order = np.argsort(time)
    time = time[order]
    values = values[order]
    start = time[-1] - tail_fraction * (time[-1] - time[0])
    use = time >= start
    if np.count_nonzero(use) < 3:
        use = np.zeros(len(time), dtype=bool)
        use[-min(3, len(time)):] = True
    if np.count_nonzero(use) < 3 or np.ptp(time[use]) <= 0.0:
        return np.nan
    return float(np.polyfit(time[use], values[use], 1)[0])


def classify_abs(value, pass_limit, fail_limit):
    if not np.isfinite(value):
        return "REVIEW"
    value = abs(value)
    if value <= pass_limit:
        return "PASS"
    if value <= fail_limit:
        return "REVIEW"
    return "FAIL"


def classify_upper(value, pass_limit, fail_limit):
    if not np.isfinite(value):
        return "REVIEW"
    if value <= pass_limit:
        return "PASS"
    if value <= fail_limit:
        return "REVIEW"
    return "FAIL"


def classify_lower(value, pass_limit, fail_limit):
    """数值越大越好：>=pass 为 PASS，>=fail 为 REVIEW，其余 FAIL。"""
    if not np.isfinite(value):
        return "REVIEW"
    if value >= pass_limit:
        return "PASS"
    if value >= fail_limit:
        return "REVIEW"
    return "FAIL"


def overall_status(statuses):
    if "FAIL" in statuses:
        return "FAIL"
    if "REVIEW" in statuses:
        return "REVIEW"
    return "PASS"


def evaluate_final(rows, tail_fraction):
    final = rows[-1]
    time = [r["time_gyr"] for r in rows]

    trends = {
        "disk_rd_drift_kpc_per_gyr": linear_slope(
            time, [r["disk_rd_fit_kpc"] for r in rows], tail_fraction
        ),
        "disk_zd_drift_kpc_per_gyr": linear_slope(
            time, [r["disk_zd_est_kpc"] for r in rows], tail_fraction
        ),
        "disk_a2_drift_per_gyr": linear_slope(
            time, [r["disk_a2_r15"] for r in rows], tail_fraction
        ),
        "disk_jz_frac_drift_per_gyr": linear_slope(
            time,
            [r["disk_jz_kpc_km_s_fractional_change"] for r in rows],
            tail_fraction,
        ),
        "center_offset_drift_kpc_per_gyr": linear_slope(
            time, [r["center_offset_box_kpc"] for r in rows], tail_fraction
        ),
        "cgm_vr_drift_km_s_per_gyr": linear_slope(
            time,
            [r["cgm_mean_vr_r20_200_km_s"] for r in rows],
            tail_fraction,
        ),
        "cgm_mass_r200_frac_drift_per_gyr": linear_slope(
            time,
            [r["cgm_mass_r200_fractional_change"] for r in rows],
            tail_fraction,
        ),
        "ism_rd_drift_kpc_per_gyr": linear_slope(
            time, [r["ism_rd_fit_kpc"] for r in rows], tail_fraction
        ),
        "ism_zd_drift_kpc_per_gyr": linear_slope(
            time, [r["ism_zd_est_kpc"] for r in rows], tail_fraction
        ),
        "ism_mass_frac_drift_per_gyr": linear_slope(
            time,
            [r["ism_gas_mass_fraction_remaining"] for r in rows],
            tail_fraction,
        ),
        "gas_short_dt_fraction_drift_per_gyr": linear_slope(
            time, [r["gas_short_dt_fraction"] for r in rows], tail_fraction
        ),
        "cgm_condensed_r5_ratio_drift_per_gyr": linear_slope(
            time,
            [r["cgm_condensed_mass_r5_ratio_to_initial"] for r in rows],
            tail_fraction,
        ),
    }

    # 阈值有意比纯解析势测试宽松，容许 live 粒子重排，但不接受持续漂移。
    checks = {
        "Relaxation_duration": classify_lower(
            final["time_gyr"], 0.28, 0.20
        ),
        "DM_Menc_RMS": classify_upper(
            final["dm_menc_rms_fraction"], 0.15, 0.25
        ),
        "DM_Menc_Max": classify_upper(
            final["dm_menc_maxabs_fraction"], 0.30, 0.50
        ),
        "Disk_A1": classify_upper(final["disk_a1_r15"], 0.04, 0.10),
        "Disk_A2": classify_upper(final["disk_a2_r15"], 0.05, 0.10),
        "Disk_Rd_change": classify_abs(
            final["disk_rd_fit_kpc_fractional_change"], 0.10, 0.20
        ),
        "Disk_zd_change": classify_abs(
            final["disk_zd_est_kpc_fractional_change"], 0.25, 0.50
        ),
        "Disk_Jz_change": classify_abs(
            final["disk_jz_kpc_km_s_fractional_change"], 0.05, 0.12
        ),
        "Disk_tilt": classify_upper(final["disk_tilt_deg"], 3.0, 8.0),
        "Disk_center": classify_upper(
            final["disk_com_offset_kpc"], 0.20, 0.50
        ),
        "Disk_mean_vr": classify_abs(
            final["disk_mean_vr_km_s"], 10.0, 20.0
        ),
        "Counterrotation": classify_upper(
            final["disk_counterrotating_fraction"], 0.01, 0.03
        ),
        "Bulge_rhalf_change": classify_abs(
            final["bulge_rhalf_kpc_fractional_change"], 0.15, 0.30
        ),
        "ISM_A2": classify_upper(final["ism_a2_r15"], 0.10, 0.20),
        "ISM_Rd_change": classify_abs(
            final["ism_rd_fit_kpc_fractional_change"], 0.15, 0.30
        ),
        "ISM_zd_change": classify_abs(
            final["ism_zd_est_kpc_fractional_change"], 0.35, 0.75
        ),
        "ISM_mean_vr": classify_abs(
            final["ism_mean_vr_km_s"], 20.0, 50.0
        ),
        "ISM_gas_remaining": classify_lower(
            final["ism_gas_mass_fraction_remaining"], 0.80, 0.50
        ),
        "ISM_origin_mass_accounting": classify_lower(
            final["ism_origin_mass_fraction_remaining"], 0.95, 0.85
        ),
        "Rd_tail_drift": classify_abs(
            trends["disk_rd_drift_kpc_per_gyr"], 0.10, 0.25
        ),
        "zd_tail_drift": classify_abs(
            trends["disk_zd_drift_kpc_per_gyr"], 0.05, 0.12
        ),
        "Jz_tail_drift": classify_abs(
            trends["disk_jz_frac_drift_per_gyr"], 0.05, 0.12
        ),
        "ISM_Rd_tail_drift": classify_abs(
            trends["ism_rd_drift_kpc_per_gyr"], 0.20, 0.50
        ),
        "ISM_zd_tail_drift": classify_abs(
            trends["ism_zd_drift_kpc_per_gyr"], 0.10, 0.30
        ),
        # 允许 cooling 产生约 30--50 km/s 内流；只有更强且持续的整体
        # 收缩才阻止该宿主进入 LMC/SMC 阶段。
        "CGM_mean_vr_20_200": classify_abs(
            final["cgm_mean_vr_r20_200_km_s"], 60.0, 100.0
        ),
        "CGM_vr_tail_drift": classify_abs(
            trends["cgm_vr_drift_km_s_per_gyr"], 20.0, 50.0
        ),
        "CGM_mass_r200_change": classify_abs(
            final["cgm_mass_r200_fractional_change"], 0.15, 0.30
        ),
        "CGM_hot_fraction_r200": classify_lower(
            final["cgm_hot_mass_fraction_r200"], 0.90, 0.70
        ),
        "CGM_temperature_change": classify_abs(
            final["cgm_temperature_r20_200_fractional_change"], 0.30, 0.60
        ),
        "CGM_inner_mass_growth": classify_upper(
            final["cgm_mass_r5_ratio_to_initial"], 2.5, 5.0
        ),
        "CGM_condensed_core": classify_upper(
            final["cgm_condensed_mass_r5_ratio_to_initial"], 0.50, 1.00
        ),
        "Gas_short_dt_fraction": classify_upper(
            final["gas_short_dt_fraction"], 1.0e-3, 5.0e-3
        ),
        "CGM_short_dt_fraction": classify_upper(
            final["cgm_short_dt_fraction"], 5.0e-4, 2.0e-3
        ),
        "Gas_short_dt_tail_growth": classify_upper(
            trends["gas_short_dt_fraction_drift_per_gyr"], 2.0e-3, 1.0e-2
        ),
    }
    return final, trends, checks, overall_status(list(checks.values()))


# =============================================================================
# 5. CSV 和图形输出
# =============================================================================

def write_summary_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dm_profile_csv(path, rows, reference, dm_radii):
    fields = (
        "file", "time_gyr", "radius_kpc", "mass_msun",
        "reference_mass_msun", "fractional_change"
    )
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for radius in dm_radii:
                key = f"dm_menc_{radius:g}kpc_msun"
                writer.writerow(
                    {
                        "file": row["file"],
                        "time_gyr": row["time_gyr"],
                        "radius_kpc": radius,
                        "mass_msun": row[key],
                        "reference_mass_msun": reference[key],
                        "fractional_change": row[
                            f"dm_menc_frac_{radius:g}kpc"
                        ],
                    }
                )


def write_short_timestep_csv(path, rows):
    fields = (
        "file",
        "time_gyr",
        "short_dt_threshold_code",
        "gas_dt_min_code",
        "gas_dt_min_yr",
        "gas_short_dt_count",
        "gas_short_dt_fraction",
        "cgm_short_dt_count",
        "cgm_short_dt_fraction",
        "ism_short_dt_count",
        "ism_short_dt_fraction",
        "short_dt_cgm_origin_fraction",
        "short_dt_ism_origin_fraction",
        "gas_nh_max_cm3",
        "cgm_nh_max_cm3",
        "ism_nh_max_cm3",
    )
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def make_plot(path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib：跳过 PNG，不影响 CSV 和数值判定。")
        return False

    time = np.array([r["time_gyr"] for r in rows])
    panels = (
        ("dm_menc_rms_fraction", "DM Menc RMS"),
        ("disk_a2_r15", "Disk A2 (R<15 kpc)"),
        ("disk_rd_fit_kpc", "Disk Rd [kpc]"),
        ("disk_zd_est_kpc", "Disk zd [kpc]"),
        ("disk_jz_kpc_km_s_fractional_change", "Delta jz / jz0"),
        ("disk_tilt_deg", "Disk tilt [deg]"),
        ("disk_mean_vr_km_s", "Mean vR [km/s]"),
        ("disk_com_offset_kpc", "Disk-center offset [kpc]"),
        ("bulge_rhalf_kpc_fractional_change", "Delta rhalf,bulge / r0"),
        ("cgm_mean_vr_r20_200_km_s", "CGM <vr> 20-200 [km/s]"),
        ("cgm_mass_r200_fractional_change", "Delta M_CGM(<200) / M0"),
        ("cgm_hot_mass_fraction_r200", "Hot CGM fraction (<200)"),
        ("ism_rd_fit_kpc", "ISM Rd [kpc]"),
        ("ism_zd_est_kpc", "ISM zd [kpc]"),
        ("ism_gas_mass_fraction_remaining", "ISM gas / initial"),
        ("ism_mean_vr_km_s", "ISM <vR> [km/s]"),
        ("gas_short_dt_fraction", "Gas fraction dt<=threshold"),
        (
            "cgm_condensed_mass_r5_ratio_to_initial",
            "Condensed CGM(<5) / M_CGM,0(<5)",
        ),
    )
    fig, axes = plt.subplots(6, 3, figsize=(13, 18), sharex=True)
    for ax, (key, label) in zip(axes.flat, panels):
        value = np.array([r[key] for r in rows], dtype=float)
        ax.plot(time, value, marker="o", ms=3, lw=1.2)
        ax.axhline(0.0, color="0.7", lw=0.7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time [Gyr]")
    fig.suptitle("Isolated live MW + cold-ISM + hot-CGM relaxation diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


# =============================================================================
# 6. 命令行和主程序
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="检测孤立 live MW + cold-ISM + hot-CGM 联合弛豫。"
    )
    parser.add_argument("--reference-ic", required=True)
    parser.add_argument("--snapshots", nargs="+", required=True)
    parser.add_argument(
        "--outdir", default="live_mw_cgm_ism_isolated_relaxation_v3"
    )
    parser.add_argument(
        "--dm-radii",
        nargs="+",
        type=float,
        default=list(DEFAULT_DM_RADII_KPC),
    )
    parser.add_argument("--disk-fit-rmin", type=float, default=DEFAULT_DISK_FIT_RMIN)
    parser.add_argument("--disk-fit-rmax", type=float, default=DEFAULT_DISK_FIT_RMAX)
    parser.add_argument("--mode-rmax", type=float, default=DEFAULT_MODE_RMAX)
    parser.add_argument("--solar-rmin", type=float, default=DEFAULT_SOLAR_RMIN)
    parser.add_argument("--solar-rmax", type=float, default=DEFAULT_SOLAR_RMAX)
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.5,
        help="用模拟末尾这一时间比例拟合漂移率，默认 0.5。",
    )
    parser.add_argument(
        "--short-dt-threshold",
        type=float,
        default=DEFAULT_SHORT_DT_THRESHOLD_CODE,
        help="短时间步阈值 [code units]，默认 1e-6。",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    reference_path = os.path.abspath(args.reference_ic)
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(reference_path)
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction 必须在 (0,1] 内。")
    if not np.isfinite(args.short_dt_threshold) or args.short_dt_threshold <= 0.0:
        raise ValueError("--short-dt-threshold 必须为有限正数。")
    if args.disk_fit_rmin >= args.disk_fit_rmax:
        raise ValueError("disk-fit-rmin 必须小于 disk-fit-rmax。")
    if args.solar_rmin >= args.solar_rmax:
        raise ValueError("solar-rmin 必须小于 solar-rmax。")

    dm_radii = np.asarray(sorted(set(args.dm_radii)), dtype=np.float64)
    if np.any(dm_radii <= 0.0):
        raise ValueError("所有 DM 检测半径都必须为正。")
    snapshot_paths = expand_snapshot_paths(args.snapshots)
    ids = read_id_ranges(reference_path)

    print("读取 reference IC...", flush=True)
    reference = analyze_file(
        reference_path,
        ids,
        dm_radii,
        args.disk_fit_rmin,
        args.disk_fit_rmax,
        args.mode_rmax,
        args.solar_rmin,
        args.solar_rmax,
        args.short_dt_threshold,
    )

    rows = []
    for index, path in enumerate(snapshot_paths, start=1):
        print(f"[{index}/{len(snapshot_paths)}] {path}", flush=True)
        row = analyze_file(
            path,
            ids,
            dm_radii,
            args.disk_fit_rmin,
            args.disk_fit_rmax,
            args.mode_rmax,
            args.solar_rmin,
            args.solar_rmax,
            args.short_dt_threshold,
        )
        add_reference_comparisons(row, reference, dm_radii)
        rows.append(row)
    rows.sort(key=lambda item: (item["time_gyr"], natural_key(item["file"])))

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "live_mw_cgm_ism_isolated_summary.csv")
    dm_profile_path = os.path.join(outdir, "live_mw_cgm_isolated_dm_profiles.csv")
    short_dt_path = os.path.join(outdir, "live_mw_cgm_ism_short_timestep.csv")
    plot_path = os.path.join(outdir, "live_mw_cgm_ism_isolated_relaxation.png")
    write_summary_csv(summary_path, rows)
    write_dm_profile_csv(dm_profile_path, rows, reference, dm_radii)
    write_short_timestep_csv(short_dt_path, rows)
    made_plot = False if args.no_plots else make_plot(plot_path, rows)

    final, trends, checks, status = evaluate_final(rows, args.tail_fraction)
    if len(rows) < 3 and status == "PASS":
        status = "REVIEW"

    print("=" * 78)
    print("孤立 live MW + cold-ISM + hot-CGM 联合弛豫检测结果")
    print("=" * 78)
    print(f"snapshot 数                  = {len(rows)}")
    print(f"最后时刻                    = {final['time_gyr']:.6f} Gyr")
    print(
        "DM Menc RMS / max          = "
        f"{final['dm_menc_rms_fraction']:.4f} / "
        f"{final['dm_menc_maxabs_fraction']:.4f}"
    )
    print(
        "Disk A1 / A2 (R<15 kpc)   = "
        f"{final['disk_a1_r15']:.4f} / {final['disk_a2_r15']:.4f}"
    )
    print(
        "Disk Rd / change           = "
        f"{final['disk_rd_fit_kpc']:.4f} kpc / "
        f"{final['disk_rd_fit_kpc_fractional_change']:+.4f}"
    )
    print(
        "Disk zd / change           = "
        f"{final['disk_zd_est_kpc']:.4f} kpc / "
        f"{final['disk_zd_est_kpc_fractional_change']:+.4f}"
    )
    print(
        "Disk Delta jz / tilt       = "
        f"{final['disk_jz_kpc_km_s_fractional_change']:+.4f} / "
        f"{final['disk_tilt_deg']:.4f} deg"
    )
    print(
        "Disk <vR>, <vz>, <vphi>    = "
        f"{final['disk_mean_vr_km_s']:.3f}, "
        f"{final['disk_mean_vz_km_s']:.3f}, "
        f"{final['disk_mean_vphi_km_s']:.3f} km/s"
    )
    print(
        "Tail dRd/dt, dzd/dt        = "
        f"{trends['disk_rd_drift_kpc_per_gyr']:.4g}, "
        f"{trends['disk_zd_drift_kpc_per_gyr']:.4g} kpc/Gyr"
    )
    print(
        "CGM <vr> (20-200/50-200)   = "
        f"{final['cgm_mean_vr_r20_200_km_s']:.3f} / "
        f"{final['cgm_mean_vr_r50_200_km_s']:.3f} km/s"
    )
    print(
        "CGM d<vr>/dt (tail)         = "
        f"{trends['cgm_vr_drift_km_s_per_gyr']:.3f} km/s/Gyr"
    )
    print(
        "CGM M(<200) change / hot f = "
        f"{final['cgm_mass_r200_fractional_change']:+.4f} / "
        f"{final['cgm_hot_mass_fraction_r200']:.4f}"
    )
    print(
        "CGM Tmed(20-200) / change   = "
        f"{final['cgm_temperature_median_r20_200_K']:.4e} K / "
        f"{final['cgm_temperature_r20_200_fractional_change']:+.4f}"
    )
    print(
        "CGM M(<5)/initial ratio     = "
        f"{final['cgm_mass_r5_ratio_to_initial']:.4f}"
    )
    print(
        "CGM condensed(<5)/M0(<5)   = "
        f"{final['cgm_condensed_mass_r5_ratio_to_initial']:.4f}"
    )
    print(
        "ISM Rd / zd / gas remaining = "
        f"{final['ism_rd_fit_kpc']:.4f} / "
        f"{final['ism_zd_est_kpc']:.4f} kpc / "
        f"{final['ism_gas_mass_fraction_remaining']:.4f}"
    )
    print(
        "ISM <vR>, <vz>, <vphi>      = "
        f"{final['ism_mean_vr_km_s']:.3f}, "
        f"{final['ism_mean_vz_km_s']:.3f}, "
        f"{final['ism_mean_vphi_km_s']:.3f} km/s"
    )
    print(
        "SFR total / ISM / CGM       = "
        f"{final['sfr_total_msun_yr']:.4g} / "
        f"{final['sfr_ism_origin_msun_yr']:.4g} / "
        f"{final['sfr_cgm_origin_msun_yr']:.4g} Msun/yr"
    )
    print(
        "Gas dt_min / short N,f      = "
        f"{final['gas_dt_min_code']:.4e} code / "
        f"{final['gas_short_dt_count']:,}, "
        f"{final['gas_short_dt_fraction']:.4e}"
    )
    print(
        "Short dt CGM / ISM N        = "
        f"{final['cgm_short_dt_count']:,} / "
        f"{final['ism_short_dt_count']:,}"
    )
    print(
        "Tail d(short fraction)/dt   = "
        f"{trends['gas_short_dt_fraction_drift_per_gyr']:.4e} /Gyr"
    )
    print("-" * 78)
    for name, value in checks.items():
        print(f"{name:24s}: {value}")
    print("-" * 78)
    print(f"OVERALL                     : {status}")
    if len(rows) < 3:
        print("注意：少于 3 个快照，无法可靠计算后半段漂移率，因此至少 REVIEW。")
    if not final["timestep_field"]:
        print(
            "注意：snapshot 缺少 TimeStep；请编译 OUTPUT_TIMESTEP，"
            "短时间步相关判据保持 REVIEW。"
        )
    if not final["sfr_field"]:
        print("注意：snapshot 缺少 StarFormationRate；SFR 仅缺测，不影响判定。")
    print(f"Summary CSV                 : {summary_path}")
    print(f"DM profile CSV              : {dm_profile_path}")
    print(f"Short-timestep CSV          : {short_dt_path}")
    if made_plot:
        print(f"Figure                      : {plot_path}")


if __name__ == "__main__":
    main()
