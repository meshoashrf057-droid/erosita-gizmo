#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GIZMO -> yt/APEC -> BH-centered X-ray cube (v2-lg-photometry)
-> Sun=(-8.2,0,0) kpc -> HEALPix LOS -> Mollweide全天球

Metallicity：
    Z/Zsun = Metallicity_00 / 0.02（不能累加 00--10，避免总量/元素重复计数）

强度：
    I_X = 1/(4*pi) * integral epsilon_X ds
    F_X = sum(I_X * pixel_solid_angle)                 [erg/s/cm^2]
    L_X = sum(pixel_solid_angle * integral epsilon_X*s^2 ds) [erg/s]
    L_X 与 I_X 使用同一 cube、LOS 范围和盘面筛选；s 是到太阳的距离。
    太阳位于延展发射源内部，不能用一个统一距离把 F_X 转为本征 L_X。

坐标：
    BH/GC = (0,0,0) kpc
    Sun   = (-8.2,0,0) kpc
    l=0   -> +x
    l=90  -> +y
    b=90  -> +z

说明：
    默认采集 BH 中心 ±30 kpc 的发射率，排除 |z| < 5 kpc；可从命令行修改。
    射线离开该cube后 emissivity=0。
    色条显示 lg(I_X/单位)，.npy 和所有积分保留线性物理量，色标截断不影响总量。
    每帧输出原始强度、每像素光度、photometry.json，并更新 allsky_xray_photometry.csv。
    统计全天及南北半球；结果为所选全部气体的本征 APEC 发射，未包含吸收、
    仪器响应或背景扣除，不能直接称为“气泡净光度”。

运行：
    python allsky_xray_yt_v2_log_luminosity.py --snapshot-dir ./allsky
    python allsky_xray_yt_v2_log_luminosity.py --snapshot-dir ./allsky --reuse-grid --reuse-allsky
    python allsky_xray_yt_v2_log_luminosity.py --snapshot-dir ./allsky --disk-mask-half-height 0
"""

import argparse
import csv
import json
import os
import glob
import gc
import re
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import healpy as hp
import yt
from scipy.ndimage import map_coordinates


# =============================================================================
# 参数
# =============================================================================

SNAPSHOT_DIR = "./allsky"
OUTPUT_DIR = os.path.join(SNAPSHOT_DIR, "figures_allsky_xray")
GRID_DIR = os.path.join(SNAPSHOT_DIR, "allsky_xray_grids")

SNAPSHOT_GLOB = "snapshot_*.hdf5"
# SNAPSHOT_GLOB = "snapshot_043.hdf5"

E_MIN_KEV = 0.6
E_MAX_KEV = 1.0
XRAY_FIELD = ("gas", "xray_emissivity_0.6_1.0_keV")

# 中央区域采集范围：±GRID_RADIUS_KPC；这是立方体半宽，不是球半径。
GRID_RADIUS_KPC = 30.0
GRID_DIMS = (256, 256, 256)

# 首次运行重建网格；之后可用 --reuse-grid 复用已校验的缓存。
OVERWRITE_GRID = True

SUN_OFFSET_KPC = np.array([-8.2, 0.0, 0.0], dtype=np.float64)
DISK_MASK_HALF_HEIGHT_KPC = 5.0
APEC_DATA_DIR = None
CACHE_SCHEMA_VERSION = 2
START_SNAPSHOT = None
STOP_SNAPSHOT = None
SNAPSHOT_STRIDE = 1

# 太阳LOS；cube外自动epsilon=0
LOS_MIN_KPC = 0.05         #LOS 积分离太阳的起始距离。 当前从 50 pc 开始，避免恰好在观察者位置采样。
LOS_MAX_KPC = 60.0         #每条 LOS 的最大长度；cube 外发射率为零。这不是以 GC 为中心的球半径。
N_LOS = 1024                #每条 HEALPix 射线的径向采样点数。 与cube cell 大致相当，设置合理。

NSIDE = 128                #HEALPix 全天球角分辨率参数。
PIXEL_CHUNK = 4096         #一次处理多少个 HEALPix 像素。 只影响内存和速度，不改变物理结果。太大增加内存，太小增加循环开销。

# 可用 --reuse-allsky 复用已校验的强度和光度数组。
OVERWRITE_ALLSKY = True

CMAP = "inferno"           #最终全天球图使用的 matplotlib colormap。 纯绘图参数，不改变物理量。
DPI = 220                  #PNG 输出分辨率。 纯绘图参数。

COLOR_PERCENTILE_MIN = 1.0 #自动色标下限采用所有正有限像素的第 1 百分位。 只影响显示，不改变 .npy 数据。
COLOR_PERCENTILE_MAX = 99.5 #自动色标上限采用第 99.5 百分位。 防止极少数超亮像素压缩整个图的动态范围。

# 正式比较多个snapshot时可改True，所有图用同一色标
USE_FIXED_COLORBAR = False  #是否强制多个 snapshot 使用统一物理色标。 做时间演化/孤立盘与 LMC 比较时建议最终改为 True。
FIXED_VMIN = 1.0e-12       #固定色标时的最小物理强度。 输入的是线性
FIXED_VMAX = 5.0e-8        #固定色标时最大物理强度。

KPC_TO_CM = 3.0856775814913673e21
yt.funcs.mylog.setLevel(30)


# =============================================================================
# 金属丰度：与切片脚本一致
# =============================================================================
# GIZMO标准定义：Metallicity_00 = 总金属质量分数
# 本项目采用 Z_sun = 0.02，与GIZMO/FIRE参考脚本保持一致
GIZMO_ZSUN = 0.02

def _apec_metallicity(field, data):
    z_mass_fraction = data["PartType0", "Metallicity_00"]
    z_over_zsun = z_mass_fraction / GIZMO_ZSUN

    # 显式告诉yt：这里的数值已经是 Z/Z_sun
    return z_over_zsun * data.ds.quan(1.0, "Zsun")

def add_apec_metallicity_field(ds):
    ds.add_field(
        ("gas", "apec_metallicity"),
        function=_apec_metallicity,
        units="Zsun",
        sampling_type="cell",
        force_override=True,
    )


# =============================================================================
# BH中心、时间和文件名
# =============================================================================

def get_bh_center_and_time(filename):
    with h5py.File(filename, "r") as f:
        time = float(f["Header"].attrs.get("Time", np.nan))

        if "PartType5" not in f:
            raise RuntimeError(f"{filename} 缺少PartType5。")
        if "Coordinates" not in f["PartType5"]:
            raise RuntimeError(f"{filename} 缺少PartType5/Coordinates。")
        if len(f["PartType5"]["Coordinates"]) == 0:
            raise RuntimeError(f"{filename} 中没有BH粒子。")

        bh = np.asarray(
            f["PartType5"]["Coordinates"][0],
            dtype=np.float64
        )

    return bh, time


def stem(snapshot_file):
    return os.path.splitext(os.path.basename(snapshot_file))[0]


def band_tag():
    return f"xray{E_MIN_KEV:g}_{E_MAX_KEV:g}".replace(".", "p")


def grid_filename(snapshot_file):
    return os.path.join(
        GRID_DIR,
        f"{stem(snapshot_file)}_{band_tag()}_metal00_Zsun02_"
        f"R{GRID_RADIUS_KPC:.0f}_grid{GRID_DIMS[0]}.hdf5"
    )


def npy_filename(snapshot_file):
    return os.path.join(
        OUTPUT_DIR,
        f"{stem(snapshot_file)}_{band_tag()}_metal00_Zsun02_"
        f"R{GRID_RADIUS_KPC:.0f}_nside{NSIDE}_"
        f"los{LOS_MAX_KPC:.0f}.npy"
    )


def png_filename(snapshot_file):
    return os.path.join(
        OUTPUT_DIR,
        f"{stem(snapshot_file)}_{band_tag()}_metal00_Zsun02_"
        f"R{GRID_RADIUS_KPC:.0f}_allsky.png"
    )


# =============================================================================
# yt：GIZMO/SPH -> APEC -> BH 中心规则 X-ray 网格
# =============================================================================

def build_xray_grid(snapshot_file, grid_file):
    bh, time_code = get_bh_center_and_time(snapshot_file)
    ds = yt.load(snapshot_file)
    ds.force_periodicity(False)
    add_apec_metallicity_field(ds)
    bh = ds.arr(bh, "code_length").to_value("kpc")
    time = float(ds.current_time.to_value("Gyr"))
    print(f"  BH中心 = {bh} kpc")
    print(f"  Time = {time:.6f} Gyr (Header/Time={time_code:.6f} code_time)")

    # 与你的切片脚本相同
    yt.add_xray_emissivity_field(
        ds,
        E_MIN_KEV,
        E_MAX_KEV,
        table_type="apec",
        metallicity=("gas", "apec_metallicity"),
        data_dir=APEC_DATA_DIR,
    )

    center = ds.arr(bh, "kpc")
    radius = ds.quan(GRID_RADIUS_KPC, "kpc")

    cell_size = 2.0 * GRID_RADIUS_KPC / GRID_DIMS[0]

    print(
        f"  yt X-ray网格 = {GRID_DIMS}, "
        f"范围 ±{GRID_RADIUS_KPC:.1f} kpc"
    )
    print(f"  规则网格分辨率 ≈ {cell_size:.4f} kpc/cell")

    grid = ds.arbitrary_grid(
        center - radius,
        center + radius,
        GRID_DIMS
    )

    cube = np.asarray(
        grid[XRAY_FIELD].in_units("erg/s/cm**3").d,
        dtype=np.float32
    )

    cube = np.nan_to_num(
        cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    cube[cube < 0.0] = 0.0

    with h5py.File(grid_file, "w") as f:
        f.create_dataset(
            "xray_emissivity",
            data=cube,
            compression="gzip",
            compression_opts=1
        )

        f.attrs["units"] = "erg/s/cm**3"
        f.attrs["time"] = time
        f.attrs["time_unit"] = "Gyr"
        f.attrs["time_code"] = time_code
        f.attrs["grid_radius_kpc"] = GRID_RADIUS_KPC
        f.attrs["grid_n"] = GRID_DIMS[0]
        f.attrs["metallicity"] = "Metallicity_00 / 0.02 in Zsun"
        f.attrs["cache_signature"] = json.dumps(grid_signature(snapshot_file), sort_keys=True)

        f.attrs["bh_x_kpc"] = bh[0]
        f.attrs["bh_y_kpc"] = bh[1]
        f.attrs["bh_z_kpc"] = bh[2]

    print(f"  已保存规则网格：{grid_file}")

    del cube, grid, ds
    gc.collect()


def load_xray_grid(grid_file):
    with h5py.File(grid_file, "r") as f:
        cube = np.asarray(
            f["xray_emissivity"][...],
            dtype=np.float32
        )

        time = float(f.attrs["time"])
        radius_kpc = float(f.attrs["grid_radius_kpc"])

    cube = np.nan_to_num(
        cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    cube[cube < 0.0] = 0.0

    return cube, time, radius_kpc


# =============================================================================
# 三线性采样
# =============================================================================

def sample_cube(cube, points_kpc, radius_kpc):
    """
    cube范围：
        x,y,z = [-R,+R]

    cube外：
        emissivity = 0

    arbitrary_grid保存cell-center值，因此index减0.5。
    """

    nx, ny, nz = cube.shape

    dx = 2.0 * radius_kpc / nx
    dy = 2.0 * radius_kpc / ny
    dz = 2.0 * radius_kpc / nz

    ix = (points_kpc[:, 0] + radius_kpc) / dx - 0.5
    iy = (points_kpc[:, 1] + radius_kpc) / dy - 0.5
    iz = (points_kpc[:, 2] + radius_kpc) / dz - 0.5

    values = map_coordinates(
        cube,
        np.vstack((ix, iy, iz)),
        order=1,
        mode="nearest",
        prefilter=False
    )

    values = np.asarray(values, dtype=np.float64)
    # 常数延拓到最外层 cell 的物理边界；只有 cube 外部才置零，避免丢掉半格边缘。
    inside = np.all((points_kpc >= -radius_kpc) & (points_kpc <= radius_kpc), axis=1)
    values[~inside] = 0.0

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    values[values < 0.0] = 0.0

    return values


# =============================================================================
# HEALPix全天球LOS
# =============================================================================

def make_allsky(cube, radius_kpc):
    """
    从Sun=(-8.2,0,0) kpc出发，对全部HEALPix方向积分。

    注意：
        不再要求 LOS_MAX < cube半宽。
        射线离开 ±radius_kpc cube 后，map_coordinates自动返回0。
        屏蔽 |Z| < DISK_MASK_HALF_HEIGHT_KPC 的区域（银盘附近）。
        返回线性强度 [erg/s/cm^2/sr] 和每像素对应体积的本征光度 [erg/s]。
    """

    npix = hp.nside2npix(NSIDE)
    pix = np.arange(npix, dtype=np.int64)

    vx, vy, vz = hp.pix2vec(
        NSIDE,
        pix,
        nest=False
    )

    directions = np.column_stack(
        (vx, vy, vz)
    ).astype(np.float64)

    los_kpc = np.linspace(
        LOS_MIN_KPC,
        LOS_MAX_KPC,
        N_LOS,
        dtype=np.float64
    )

    los_cm = los_kpc * KPC_TO_CM
    intensity = np.zeros(npix, dtype=np.float64)
    luminosity_pixels = np.zeros(npix, dtype=np.float64)
    pixel_solid_angle = 4.0 * np.pi / npix

    los_step = (
        LOS_MAX_KPC - LOS_MIN_KPC
    ) / (N_LOS - 1)

    print(
        f"  NSIDE={NSIDE}, Npix={npix}, "
        f"LOS={LOS_MIN_KPC:.2f}-{LOS_MAX_KPC:.2f} kpc"
    )
    print(
        f"  N_LOS={N_LOS}, "
        f"LOS采样间距 ≈ {los_step:.4f} kpc"
    )

    for i0 in range(0, npix, PIXEL_CHUNK):
        i1 = min(i0 + PIXEL_CHUNK, npix)
        direction = directions[i0:i1]

        points = (
            SUN_OFFSET_KPC[None, None, :]
            + direction[:, None, :]
            * los_kpc[None, :, None]
        )

        eps = sample_cube(
            cube,
            points.reshape(-1, 3),
            radius_kpc
        ).reshape(
            i1 - i0,
            N_LOS
        )

        # ---------- 对强度和光度使用完全相同的三维盘面筛选 ----------
        z_coords = points[:, :, 2]          # shape (n_pix, N_LOS)
        mask_z = np.abs(z_coords) >= DISK_MASK_HALF_HEIGHT_KPC
        eps_masked = eps * mask_z          # 盘面区域置零

        # ---------- 梯形积分（只使用 eps_masked） ----------
        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(eps_masked, x=los_cm, axis=1)
        else:
            integral = np.trapz(eps_masked, x=los_cm, axis=1)

        intensity[i0:i1] = integral / (4.0 * np.pi)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        luminosity_pixels[i0:i1] = pixel_solid_angle * integrate(
            eps_masked * los_cm[None, :]**2, x=los_cm, axis=1
        )

        print(
            f"    pixels "
            f"{i0:6d}-{i1 - 1:6d} / {npix - 1}"
        )

        del points, eps, eps_masked, integral

    intensity = np.nan_to_num(
        intensity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    intensity[intensity < 0.0] = 0.0

    if np.any(~np.isfinite(luminosity_pixels)) or np.any(luminosity_pixels < 0.0):
        raise RuntimeError("光度积分出现非有限值或负值，请检查发射率和单位。")
    return intensity, luminosity_pixels


# =============================================================================
# 色标
# =============================================================================

def grid_signature(snapshot_file):
    """校验物理输入，防止复用不同能段、金属定义或旧快照的网格。"""
    stat = os.stat(snapshot_file)
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "snapshot": os.path.realpath(snapshot_file),
        "snapshot_size": stat.st_size,
        "snapshot_mtime_ns": stat.st_mtime_ns,
        "energy_keV": [E_MIN_KEV, E_MAX_KEV],
        "grid_radius_kpc": GRID_RADIUS_KPC,
        "grid_dims": list(GRID_DIMS),
        "metallicity": "Metallicity_00 / 0.02 in Zsun",
        "apec_data_dir": os.path.abspath(APEC_DATA_DIR) if APEC_DATA_DIR else None,
    }


def allsky_signature(snapshot_file):
    return {
        "grid": grid_signature(snapshot_file),
        "sun_offset_kpc": SUN_OFFSET_KPC.tolist(),
        "los_range_kpc": [LOS_MIN_KPC, LOS_MAX_KPC],
        "los_samples": N_LOS,
        "nside": NSIDE,
        "disk_mask_half_height_kpc": DISK_MASK_HALF_HEIGHT_KPC,
        "sampling": "trilinear_inside_physical_cube; trapezoidal_LOS",
    }


def grid_cache_matches(snapshot_file, grid_file):
    if not os.path.exists(grid_file):
        return False
    try:
        with h5py.File(grid_file, "r") as handle:
            stored = json.loads(handle.attrs.get("cache_signature", "null"))
            return (
                stored == grid_signature(snapshot_file)
                and tuple(handle["xray_emissivity"].shape) == tuple(GRID_DIMS)
                and handle.attrs.get("time_unit") == "Gyr"
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cube_luminosity_check(cube, radius_kpc):
    """独立的体素求和参考值；同样按 LOS 距离和 |z| 筛选 cell 中心。"""
    nx, ny, nz = cube.shape
    dx, dy, dz = 2.0 * radius_kpc / np.array([nx, ny, nz], dtype=float)
    x = -radius_kpc + (np.arange(nx) + 0.5) * dx
    y = -radius_kpc + (np.arange(ny) + 0.5) * dy
    z = -radius_kpc + (np.arange(nz) + 0.5) * dz
    yz_distance2 = (y[:, None] - SUN_OFFSET_KPC[1])**2 + (z[None, :] - SUN_OFFSET_KPC[2])**2
    z_keep = np.abs(z[None, :]) >= DISK_MASK_HALF_HEIGHT_KPC
    selected_emissivity_sum = 0.0
    for index, x_value in enumerate(x):
        distance2 = (x_value - SUN_OFFSET_KPC[0])**2 + yz_distance2
        selected = z_keep & (distance2 >= LOS_MIN_KPC**2) & (distance2 <= LOS_MAX_KPC**2)
        selected_emissivity_sum += np.sum(cube[index][selected], dtype=np.float64)
    cell_volume_cm3 = dx * dy * dz * KPC_TO_CM**3
    return {
        "whole_cube_luminosity_erg_s": float(np.sum(cube, dtype=np.float64) * cell_volume_cm3),
        "selected_voxel_luminosity_erg_s": float(selected_emissivity_sum * cell_volume_cm3),
        "note": "Cell-centre volume quadrature; compare with LOS luminosity for resolution convergence, not an exact equality at finite resolution.",
    }


def log10_or_none(value):
    return float(np.log10(value)) if value > 0.0 else None


def make_photometry(hpxmap, luminosity_pixels, time_gyr, snapshot_file, voxel_check=None):
    """所有总量来自未取对数、未截色标的线性数据。"""
    intensity = np.asarray(hpxmap, dtype=np.float64).reshape(-1)
    luminosity = np.asarray(luminosity_pixels, dtype=np.float64).reshape(-1)
    expected = hp.nside2npix(NSIDE)
    if intensity.size != expected or luminosity.size != expected:
        raise ValueError("强度/光度像素数与 NSIDE 不一致。")
    if any(np.any(~np.isfinite(a)) or np.any(a < 0.0) for a in (intensity, luminosity)):
        raise ValueError("光度统计要求有限、非负的线性强度和光度数组。")
    solid_angle = 4.0 * np.pi / expected
    _, _, sky_z = hp.pix2vec(NSIDE, np.arange(expected), nest=False)
    # 赤道中心的有限面积像素各分一半，保证 N + S = allsky。
    north_weight = np.where(np.abs(sky_z) < 1e-12, 0.5, (sky_z > 0.0).astype(float))
    regions = {}
    for name, weight in (
        ("allsky", np.ones(expected)), ("north", north_weight), ("south", 1.0 - north_weight)
    ):
        area = float(np.sum(weight) * solid_angle)
        flux = float(np.sum(intensity * weight, dtype=np.float64) * solid_angle)
        emitted_power = float(np.sum(luminosity * weight, dtype=np.float64))
        regions[name] = {
            "solid_angle_sr": area,
            "mean_intensity_erg_s_cm2_sr": flux / area,
            "flux_erg_s_cm2": flux,
            "luminosity_erg_s": emitted_power,
            "lg_flux_erg_s_cm2": log10_or_none(flux),
            "lg_luminosity_erg_s": log10_or_none(emitted_power),
        }
    check = dict(voxel_check or {})
    voxel_luminosity = check.get("selected_voxel_luminosity_erg_s", 0.0)
    if voxel_luminosity > 0.0:
        check["los_to_voxel_luminosity_ratio"] = regions["allsky"]["luminosity_erg_s"] / voxel_luminosity
    return {
        "cache_signature": allsky_signature(snapshot_file),
        "snapshot": os.path.abspath(snapshot_file),
        "time_gyr": float(time_gyr),
        "definitions": {
            "intensity": "I_X = integral(epsilon_X ds)/(4*pi), erg/s/cm^2/sr",
            "flux": "F_X = sum(I_X dOmega), erg/s/cm^2",
            "luminosity": "L_X = sum[dOmega integral(epsilon_X s^2 ds)], erg/s; s is distance from observer",
            "selection": "same cube, LOS interval and physical |z| cut as the all-sky map",
            "emission_model": "intrinsic, optically thin APEC; no absorption or instrumental response",
            "background_subtracted": False,
            "hemispheres": "observer Galactic latitude; equatorial pixels split equally",
        },
        "regions": regions,
        "voxel_crosscheck": check,
    }


def save_photometry(summary, summary_file, map_file):
    with open(summary_file, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    csv_file = os.path.join(OUTPUT_DIR, "allsky_xray_photometry.csv")
    map_path = os.path.abspath(map_file)
    rows = []
    if os.path.exists(csv_file):
        with open(csv_file, newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("map_file") != map_path]
    for name, values in summary["regions"].items():
        row = {
            "snapshot": summary["snapshot"], "time_gyr": summary["time_gyr"],
            "energy_min_keV": E_MIN_KEV, "energy_max_keV": E_MAX_KEV,
            "grid_half_width_kpc": GRID_RADIUS_KPC,
            "disk_mask_half_height_kpc": DISK_MASK_HALF_HEIGHT_KPC,
            "los_min_kpc": LOS_MIN_KPC, "los_max_kpc": LOS_MAX_KPC,
            "nside": NSIDE, "region": name, **values, "map_file": map_path,
        }
        rows.append(row)
    fieldnames = list(row)
    temporary = csv_file + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_file)
    print("  本征 APEC 统计（所选全部气体；未吸收、未扣背景）：")
    for name, values in summary["regions"].items():
        print(f"    {name:7s}: L_X={values['luminosity_erg_s']:.6e} erg/s; "
              f"F_X={values['flux_erg_s_cm2']:.6e} erg/s/cm^2; "
              f"lg L_X={values['lg_luminosity_erg_s']}")
    ratio = summary["voxel_crosscheck"].get("los_to_voxel_luminosity_ratio")
    if ratio is not None:
        print(f"  分辨率交叉检查 L_LOS/L_voxel = {ratio:.6f}")
    print(f"  光度统计：{summary_file}")
    print(f"  时间序列表：{csv_file}")

def get_color_limits(hpxmap):
    good = hpxmap[
        np.isfinite(hpxmap)
        & (hpxmap > 0.0)
    ]

    if good.size == 0:
        raise RuntimeError("全天球X-ray强度全部为0。")

    if USE_FIXED_COLORBAR:
        vmin = float(FIXED_VMIN)
        vmax = float(FIXED_VMAX)
    else:
        vmin = float(
            np.percentile(
                good,
                COLOR_PERCENTILE_MIN
            )
        )

        vmax = float(
            np.percentile(
                good,
                COLOR_PERCENTILE_MAX
            )
        )

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(good.min())

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(good.max())

    if vmax <= vmin:
        vmax = vmin * 10.0

    return vmin, vmax


def scientific_label(value):
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent

    return (
        rf"${mantissa:.1f}"
        rf"\times10^{{{exponent}}}$"
    )


# =============================================================================
# 全天球坐标文字
# =============================================================================

def sky_text(lon, lat, text, fontsize=10):
    t = hp.projtext(
        lon,
        lat,
        text,
        lonlat=True,
        color="black",
        fontsize=fontsize,
        ha="center",
        va="center"
    )

    try:
        t.set_path_effects([
            pe.withStroke(
                linewidth=2.5,
                foreground="white"
            )
        ])
    except Exception:
        pass


# =============================================================================
# Mollweide全天球图
# =============================================================================

def plot_allsky(hpxmap, time, output_png, photometry=None):
    vmin, vmax = get_color_limits(hpxmap)

    log_vmin = np.log10(vmin)
    log_vmax = np.log10(vmax)

    safe = np.asarray(
        hpxmap,
        dtype=np.float64
    ).copy()

    bad = (
        ~np.isfinite(safe)
        | (safe <= 0.0)
    )

    safe[bad] = vmin
    safe = np.clip(safe, vmin, vmax)

    log_map = np.log10(safe)
    log_map[bad] = hp.UNSEEN

    print(
        f"  色标范围 = "
        f"{vmin:.4e} -- {vmax:.4e} "
        "erg s^-1 cm^-2 sr^-1"
    )

    plt.close("all")

    hp.mollview(
        log_map,
        nest=False,
        flip="astro",
        coord=['G'],
        rot=(0.0, 0.0, 0.0),
        cmap=CMAP,
        min=log_vmin,
        max=log_vmax,
        cbar=False,
        title=(
            f"Intrinsic APEC X-ray intensity {E_MIN_KEV:g}--{E_MAX_KEV:g} keV"
            f"   |   Time = {time:.3f} Gyr"
        ),
        xsize=1800
    )

    fig = plt.gcf()
    fig.set_size_inches(12.0, 7.0)

    # 观测图风格经纬网
    hp.graticule(
        dpar=30,
        dmer=60,
        color="white",
        alpha=0.50,
        linewidth=0.8
    )

    # Galactic longitude：左120/60，中0，右300/240
    for lon, label in [
        (120.0, "120°"),
        (60.0, "60°"),
        (0.0, "0°"),
        (300.0, "300°"),
        (240.0, "240°")
    ]:
        sky_text(
            lon,
            2.0,
            label,
            fontsize=11
        )

    # Galactic latitude
    for lat in [60, 30, 0, -30, -60]:
        sky_text(
            170.0,
            float(lat),
            f"{lat}°",
            fontsize=10
        )

    sky_text(
        0.0,
        -5.0,
        "GC",
        fontsize=9
    )

    fig.text(
        0.50,
        0.165,
        r"Galactic longitude $l$ [deg]",
        ha="center",
        fontsize=11
    )

    fig.text(
        0.035,
        0.52,
        r"Galactic latitude $b$ [deg]",
        rotation=90,
        va="center",
        fontsize=11
    )

    # inferno水平colorbar
    norm = Normalize(
        vmin=log_vmin,
        vmax=log_vmax
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=CMAP
    )

    sm.set_array([])

    cax = fig.add_axes([
        0.30,
        0.085,
        0.40,
        0.032
    ])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal"
    )

    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(log_vmin, log_vmax)
    ticks = ticks[(ticks >= log_vmin) & (ticks <= log_vmax)]
    if ticks.size < 2:
        ticks = np.linspace(log_vmin, log_vmax, 5)
    cbar.set_ticks(ticks)
    cbar.formatter = FormatStrFormatter("%g")
    cbar.update_ticks()

    cbar.set_label(
        r"$\lg\!\left[I_X/(\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}})\right]$",
        fontsize=11
    )

    cbar.ax.tick_params(
        labelsize=9
    )

    if photometry is not None:
        total = photometry["regions"]["allsky"]
        north = photometry["regions"]["north"]
        south = photometry["regions"]["south"]
        if south["luminosity_erg_s"] > 0.0:
            ratio_label = f"{north['luminosity_erg_s']/south['luminosity_erg_s']:.3f}"
        else:
            ratio_label = r"\infty" if north["luminosity_erg_s"] > 0.0 else r"\mathrm{undefined}"
        fig.text(
            0.5, -0.015,
            rf"$L_X={total['luminosity_erg_s']:.3e}\ \mathrm{{erg\,s^{{-1}}}}$"
            rf";  $F_X={total['flux_erg_s_cm2']:.3e}\ \mathrm{{erg\,s^{{-1}}\,cm^{{-2}}}}$"
            rf";  $L_N/L_S={ratio_label}$",
            ha="center", fontsize=10,
        )
        fig.text(
            0.5, -0.05,
            rf"Selected gas: cube $\pm{GRID_RADIUS_KPC:g}$ kpc; "
            rf"$|z|\geq {DISK_MASK_HALF_HEIGHT_KPC:g}$ kpc; "
            rf"LOS {LOS_MIN_KPC:g}--{LOS_MAX_KPC:g} kpc; unabsorbed, no background subtraction",
            ha="center", fontsize=9,
        )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"  已保存全天球图：{output_png}")


# =============================================================================
# 单个snapshot
# =============================================================================

def process_snapshot(snapshot_file):
    print("\n" + "=" * 78)
    print(f"正在处理：{snapshot_file}")

    grid_file = grid_filename(snapshot_file)
    npy_file = npy_filename(snapshot_file)
    png_file = png_filename(snapshot_file)
    luminosity_file = os.path.splitext(npy_file)[0] + "_luminosity_erg_s.npy"
    summary_file = os.path.splitext(npy_file)[0] + "_photometry.json"

    cached = None
    if not OVERWRITE_ALLSKY and all(os.path.exists(p) for p in (npy_file, luminosity_file, summary_file)):
        try:
            with open(summary_file, encoding="utf-8") as stream:
                candidate = json.load(stream)
            if isinstance(candidate, dict) and candidate.get("cache_signature") == allsky_signature(snapshot_file):
                cached = candidate
        except (OSError, ValueError, TypeError):
            pass

    # 复用必须同时包含强度、光度和匹配的物理元数据。
    if (
        cached is not None
    ):
        hpxmap = np.asarray(
            np.load(npy_file),
            dtype=np.float64
        ).reshape(-1)

        expected = hp.nside2npix(NSIDE)

        if hpxmap.size != expected:
            raise RuntimeError(
                f"已有.npy像素数={hpxmap.size}，"
                f"NSIDE={NSIDE}要求{expected}。"
            )

        luminosity_pixels = np.asarray(np.load(luminosity_file), dtype=np.float64).reshape(-1)
        time = cached["time_gyr"]
        photometry = make_photometry(
            hpxmap, luminosity_pixels, time, snapshot_file, cached.get("voxel_crosscheck")
        )
        save_photometry(photometry, summary_file, npy_file)

        print(
            f"  复用同物理参数的线性全天图与光度数组："
            f"{npy_file}"
        )

        plot_allsky(
            hpxmap,
            time,
            png_file,
            photometry=photometry,
        )

        return

    # 建立/复用 X-ray cube；元数据不匹配时自动重建。
    if (
        OVERWRITE_GRID
        or not grid_cache_matches(snapshot_file, grid_file)
    ):
        build_xray_grid(
            snapshot_file,
            grid_file
        )
    else:
        print(
            f"  使用已核对物理参数的规则网格："
            f"{grid_file}"
        )

    cube, time, radius_kpc = load_xray_grid(
        grid_file
    )

    hpxmap, luminosity_pixels = make_allsky(
        cube,
        radius_kpc
    )

    np.save(
        npy_file,
        hpxmap
    )
    np.save(luminosity_file, luminosity_pixels)
    photometry = make_photometry(
        hpxmap, luminosity_pixels, time, snapshot_file,
        cube_luminosity_check(cube, radius_kpc),
    )
    save_photometry(photometry, summary_file, npy_file)

    print(
        f"  已保存线性全天球数组："
        f"{npy_file}"
    )

    plot_allsky(
        hpxmap,
        time,
        png_file,
        photometry=photometry,
    )

    del cube, hpxmap, luminosity_pixels
    gc.collect()


# =============================================================================
# 主程序
# =============================================================================

def configure_from_arguments(argv=None):
    global SNAPSHOT_DIR, OUTPUT_DIR, GRID_DIR, GRID_RADIUS_KPC, GRID_DIMS
    global NSIDE, N_LOS, LOS_MIN_KPC, LOS_MAX_KPC, PIXEL_CHUNK
    global SUN_OFFSET_KPC, DISK_MASK_HALF_HEIGHT_KPC, APEC_DATA_DIR
    global E_MIN_KEV, E_MAX_KEV, XRAY_FIELD, OVERWRITE_GRID, OVERWRITE_ALLSKY
    global USE_FIXED_COLORBAR, FIXED_VMIN, FIXED_VMAX
    global START_SNAPSHOT, STOP_SNAPSHOT, SNAPSHOT_STRIDE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-dir", default=SNAPSHOT_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--grid-dir", default=None)
    parser.add_argument("--grid-radius-kpc", type=float, default=GRID_RADIUS_KPC)
    parser.add_argument("--grid-n", type=int, default=GRID_DIMS[0])
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--n-los", type=int, default=N_LOS)
    parser.add_argument("--los-min-kpc", type=float, default=LOS_MIN_KPC)
    parser.add_argument("--los-max-kpc", type=float, default=LOS_MAX_KPC)
    parser.add_argument("--pixel-chunk", type=int, default=PIXEL_CHUNK)
    parser.add_argument("--sun-offset-kpc", type=float, nargs=3, default=SUN_OFFSET_KPC)
    parser.add_argument("--disk-mask-half-height", type=float, default=DISK_MASK_HALF_HEIGHT_KPC)
    parser.add_argument("--emin-kev", type=float, default=E_MIN_KEV)
    parser.add_argument("--emax-kev", type=float, default=E_MAX_KEV)
    parser.add_argument("--apec-data-dir", default=APEC_DATA_DIR)
    parser.add_argument("--reuse-grid", dest="overwrite_grid", action="store_false", default=OVERWRITE_GRID,
                        help="物理参数及源文件校验一致时复用网格；旧的无签名缓存会重建")
    parser.add_argument("--reuse-allsky", dest="overwrite_allsky", action="store_false", default=OVERWRITE_ALLSKY,
                        help="复用已校验的线性强度/光度数组，只重绘 lg 色条和统计")
    parser.add_argument("--intensity-limits", type=float, nargs=2, default=None,
                        help="固定色标上下限，输入线性 erg/s/cm^2/sr，积分总量不受影响")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)
    numeric = [args.grid_radius_kpc, args.los_min_kpc, args.los_max_kpc,
               args.disk_mask_half_height, args.emin_kev, args.emax_kev, *args.sun_offset_kpc]
    if not np.all(np.isfinite(numeric)):
        parser.error("几何和能段参数必须为有限数。")
    if args.grid_radius_kpc <= 0 or args.grid_n < 2:
        parser.error("grid-radius-kpc 必须 >0，grid-n 必须 >=2。")
    if not hp.isnsideok(args.nside) or args.n_los < 2 or args.pixel_chunk < 1:
        parser.error("NSIDE 必须有效，n-los >=2，pixel-chunk >=1。")
    if not (0 <= args.los_min_kpc < args.los_max_kpc):
        parser.error("要求 0 <= los-min-kpc < los-max-kpc。")
    if not (0 <= args.disk_mask_half_height < args.grid_radius_kpc):
        parser.error("要求 0 <= disk-mask-half-height < grid-radius-kpc。")
    if not (0 < args.emin_kev < args.emax_kev):
        parser.error("要求 0 < emin-kev < emax-kev。")
    if args.stride < 1 or (args.start is not None and args.stop is not None and args.start > args.stop):
        parser.error("stride 必须 >=1，start 不能大于 stop。")
    if args.intensity_limits is not None:
        low, high = args.intensity_limits
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
            parser.error("intensity-limits 必须为递增的有限正数。")
        USE_FIXED_COLORBAR = True
        FIXED_VMIN, FIXED_VMAX = low, high
    SNAPSHOT_DIR = args.snapshot_dir
    OUTPUT_DIR = args.output_dir or os.path.join(SNAPSHOT_DIR, "figures_allsky_xray")
    GRID_DIR = args.grid_dir or os.path.join(SNAPSHOT_DIR, "allsky_xray_grids")
    GRID_RADIUS_KPC = args.grid_radius_kpc
    GRID_DIMS = (args.grid_n,) * 3
    NSIDE, N_LOS, PIXEL_CHUNK = args.nside, args.n_los, args.pixel_chunk
    LOS_MIN_KPC, LOS_MAX_KPC = args.los_min_kpc, args.los_max_kpc
    SUN_OFFSET_KPC = np.asarray(args.sun_offset_kpc, dtype=np.float64)
    DISK_MASK_HALF_HEIGHT_KPC = args.disk_mask_half_height
    E_MIN_KEV, E_MAX_KEV = args.emin_kev, args.emax_kev
    XRAY_FIELD = ("gas", f"xray_emissivity_{E_MIN_KEV}_{E_MAX_KEV}_keV")
    APEC_DATA_DIR = args.apec_data_dir
    OVERWRITE_GRID, OVERWRITE_ALLSKY = args.overwrite_grid, args.overwrite_allsky
    START_SNAPSHOT, STOP_SNAPSHOT, SNAPSHOT_STRIDE = args.start, args.stop, args.stride
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRID_DIR, exist_ok=True)


def snapshot_index(path):
    match = re.search(r"snapshot_(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"无法从文件名解析快照编号：{path}")
    return int(match.group(1))


def main():
    configure_from_arguments()
    snapshots = sorted(
        glob.glob(
            os.path.join(
                SNAPSHOT_DIR,
                SNAPSHOT_GLOB
            )
        ), key=snapshot_index,
    )
    snapshots = [path for path in snapshots
                 if (START_SNAPSHOT is None or snapshot_index(path) >= START_SNAPSHOT)
                 and (STOP_SNAPSHOT is None or snapshot_index(path) <= STOP_SNAPSHOT)][::SNAPSHOT_STRIDE]

    if not snapshots:
        raise FileNotFoundError(
            f"没有找到："
            f"{os.path.join(SNAPSHOT_DIR, SNAPSHOT_GLOB)}"
        )

    cell_size = 2.0 * GRID_RADIUS_KPC / GRID_DIMS[0]
    los_step = (LOS_MAX_KPC - LOS_MIN_KPC) / (N_LOS - 1)

    print("脚本版本：allsky-v2-log-luminosity / 2026-10-05")
    print(f"发现 {len(snapshots)} 个snapshot。")
    print(f"yt={yt.__version__}, healpy={hp.__version__}")
    print("Metallicity = Metallicity_00 / 0.02 = Z/Zsun")
    print(f"APEC = {E_MIN_KEV:g}--{E_MAX_KEV:g} keV；未吸收、本征能量辐射")
    print(
        f"Grid = {GRID_DIMS}, "
        f"范围 ±{GRID_RADIUS_KPC:.1f} kpc, "
        f"dx≈{cell_size:.4f} kpc"
    )
    print(f"Sun = {SUN_OFFSET_KPC} kpc")
    print(f"保留 |z| >= {DISK_MASK_HALF_HEIGHT_KPC:g} kpc 的发射")
    print(
        f"LOS = {LOS_MIN_KPC:.2f}-{LOS_MAX_KPC:.2f} kpc, "
        f"N_LOS={N_LOS}, ds≈{los_step:.4f} kpc"
    )
    print(f"NSIDE={NSIDE}, CMAP={CMAP}")
    print(f"OVERWRITE_GRID={OVERWRITE_GRID}")
    print(f"OVERWRITE_ALLSKY={OVERWRITE_ALLSKY}")

    for snapshot_file in snapshots:
        process_snapshot(snapshot_file)

    print("\n全天球图及通量/光度统计处理完成。")


if __name__ == "__main__":
    main()
