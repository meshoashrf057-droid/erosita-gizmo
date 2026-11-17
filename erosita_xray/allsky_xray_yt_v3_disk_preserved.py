#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GIZMO -> yt/APEC -> BH-centered X-ray cube (v3-disk-preserved)
-> Sun=(-8.29,0,0) kpc -> HEALPix LOS -> Mollweide全天球

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
    Sun   = (-8.29,0,0) kpc
    l=0   -> +x
    l=90  -> +y
    b=90  -> +z

说明：
    默认采集 BH 中心 ±30 kpc 的发射率，保留全部银盘发射。
    默认色标上限 = 仅积分 |z|>=5 kpc 发射的参考全天图的最大强度。
    主图仍显示全气体；此高度仅定义色标参考，不删除主图中的气体。
    5 kpc 是距 z=0 盘面的高度，不是球半径或银纬；盘面沿模拟 xy 平面。
    高于色标上限的银盘像素饱和；色标不能分离同一视线上的盘/晕发射。
    射线离开该cube后 emissivity=0。
    色条显示 lg(I_X/单位)，.npy 和所有积分保留线性物理量，色标截断不影响总量。
    每帧输出原始强度、每像素光度、photometry.json，并更新 allsky_xray_photometry.csv。
    统计全天及南北半球；结果为所选全部气体的本征 APEC 发射，未包含吸收、
    仪器响应或背景扣除，不能直接称为“气泡净光度”。

运行：
    python allsky_xray_yt_v3_disk_preserved.py --snapshot-dir ./allsky
    python allsky_xray_yt_v3_disk_preserved.py --snapshot-dir ./allsky --reuse-grid --reuse-allsky
    python allsky_xray_yt_v3_disk_preserved.py --snapshot-dir ./allsky --intensity-limits 1e-12 5e-8

    --disk-mask-half-height 默认 0；非零时会真正删除盘面发射，改变光度。
    --color-reference-height-kpc 默认 5；只影响参考图/自动色标。
    --color-scale offplane-percentile 可使用参考图第 99.5 百分位代替严格最大值。
    yt 负责温度、APEC 和三维场；SciPy 做 LOS 三线性采样，healpy 负责全天球。
    依赖：numpy scipy h5py matplotlib yt>=4 healpy；需 APEC v3 数据表。
"""

import argparse
import csv
import json
import os
import glob
import gc
import re
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gizmo_compat import load_gizmo, discover_snapshots
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
GRID_RADIUS_KPC = 40.0
GRID_DIMS = (256, 256, 256)

# 首次运行重建网格；之后可用 --reuse-grid 复用已校验的缓存。
OVERWRITE_GRID = True

SUN_OFFSET_KPC = np.array([-8.29, 0.0, 0.0], dtype=np.float64)
DISK_MASK_HALF_HEIGHT_KPC = 0.0
COLOR_REFERENCE_HEIGHT_KPC = 5.0
COLOR_SCALE = "offplane-max"
BH_ID = None
APEC_DATA_DIR = None
CACHE_SCHEMA_VERSION = 3
START_SNAPSHOT = None
STOP_SNAPSHOT = None
SNAPSHOT_STRIDE = 1

# 太阳LOS；cube外自动epsilon=0
LOS_MIN_KPC = 0.0          #I 的 ds 积分在观察者处没有 1/s^2 奇点；默认不人为挖掉本地 50 pc。
LOS_MAX_KPC = 60.0         #每条 LOS 的最大长度；cube 外发射率为零。这不是以 GC 为中心的球半径。
N_LOS = 1024                #每条射线有效 cube 路径的中点积分区间数；并非物理分辨率。

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

        group = f["PartType5"]
        wanted_id = BH_ID
        if wanted_id is None:
            wanted_id = f["Header"].attrs.get("MW_BH_ID", f.attrs.get("MW_BH_ID"))
        if wanted_id is not None:
            if "ParticleIDs" not in group:
                raise RuntimeError("指定了 MW BH ID，但缺少 PartType5/ParticleIDs。")
            indices = np.flatnonzero(group["ParticleIDs"][...] == int(wanted_id))
            if len(indices) != 1:
                raise RuntimeError(f"BH ID={wanted_id} 未唯一匹配；请检查 --bh-id。")
            index = int(indices[0])
        elif len(group["Coordinates"]) == 1:
            index = 0
        else:
            raise RuntimeError("存在多颗 BH；请用 --bh-id 明确指定 MW 中心。")
        bh = np.asarray(group["Coordinates"][index], dtype=np.float64)

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
    ds = load_gizmo(snapshot_file, cache_dir=Path(GRID_DIR) / "yt_cache")
    # GIZMO 某些版本对原生 Temperature 未附单位，必须在字段初始化前注册。
    ds.field_units = dict(ds.field_units or {})
    ds.field_units[("PartType0", "Temperature")] = "K"
    if ds.cosmological_simulation:
        raise RuntimeError("本脚本面向非宇宙学 MW 模型；宇宙学光锥需要另行处理。")
    ds.force_periodicity(False)
    add_apec_metallicity_field(ds)
    for field, unit in ((('gas', 'temperature'), 'K'),
                        (('gas', 'H_nuclei_density'), 'cm**-3'),
                        (('gas', 'El_number_density'), 'cm**-3')):
        try:
            ds.quan(1.0, ds.field_info[field].units).to(unit)
        except Exception as exc:
            raise RuntimeError(f"APEC 所需字段 {field} 缺失或单位错误：{exc}") from exc
    bh = ds.arr(bh, "code_length").to_value("kpc")
    time = float(ds.current_time.to_value("Gyr"))
    print(f"  BH中心 = {bh} kpc")
    print(f"  Time = {time:.6f} Gyr (Header/Time={time_code:.6f} code_time)")
    print(f"  yt code units: length={ds.length_unit.to('kpc')}; "
          f"mass={ds.mass_unit.to('Msun')}; velocity={ds.velocity_unit.to('km/s')}")

    # 与你的切片脚本相同
    yt.add_xray_emissivity_field(
        ds,
        E_MIN_KEV,
        E_MAX_KEV,
        table_type="apec",
        redshift=0.0,
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

    if np.any(~np.isfinite(cube)) or np.any(cube < 0.0):
        raise RuntimeError("APEC cube 出现负值或非有限值；停止，避免静默清零影响光度。")

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
        f.attrs["yt_version"] = yt.__version__

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

def ray_cube_intervals(directions, radius_kpc):
    """物理 cube 与 LOS 的精确交集；避免在 cube 外采样或跨边界半格误差。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-radius_kpc - SUN_OFFSET_KPC) / directions
        t2 = (radius_kpc - SUN_OFFSET_KPC) / directions
    parallel = np.abs(directions) < 1e-15
    inside_axis = np.abs(SUN_OFFSET_KPC) <= radius_kpc
    lower = np.where(parallel, np.where(inside_axis, -np.inf, np.inf), np.minimum(t1, t2))
    upper = np.where(parallel, np.where(inside_axis, np.inf, -np.inf), np.maximum(t1, t2))
    entry = np.maximum(LOS_MIN_KPC, lower.max(axis=1))
    leave = np.minimum(LOS_MAX_KPC, upper.min(axis=1))
    valid = leave > entry
    return np.where(valid, entry, 0.0), np.where(valid, leave, 0.0)


def outside_slab_weights(left, right, z_direction, half_height):
    """每个径向区间落在 |z|>=h 的精确 ds 和 s^2 ds 权重（kpc 单位）。

    发射率采用区间中点值；几何截断单独积分，避免把一整个采样区间硬切掉。
    """
    width = right - left
    moment = width * (left**2 + left * right + right**2) / 3.0
    if half_height <= 0.0:
        return width, moment
    dz = z_direction[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half_height - SUN_OFFSET_KPC[2]) / dz
        t2 = (half_height - SUN_OFFSET_KPC[2]) / dz
    parallel = np.abs(dz) < 1e-15
    in_slab = abs(SUN_OFFSET_KPC[2]) < half_height
    low = np.where(parallel, -np.inf if in_slab else np.inf, np.minimum(t1, t2))
    high = np.where(parallel, np.inf if in_slab else -np.inf, np.maximum(t1, t2))
    a = np.minimum(np.maximum(low, left), right)
    b = np.maximum(np.minimum(high, right), a)
    removed_width = b - a
    removed_moment = removed_width * (a**2 + a * b + b**2) / 3.0
    return np.maximum(width - removed_width, 0.0), np.maximum(moment - removed_moment, 0.0)


def make_allsky(cube, radius_kpc):
    """保留主图和盘外参考图；所有输出都是线性物理量。

    yt cube 的发射率经三线性插值后中点积分。对 I 积分 ds，
    对 L 积分 s^2 ds；绝不对 lg(epsilon) 积分或用统一距离换算 L。
    """
    npix = hp.nside2npix(NSIDE)
    directions = np.column_stack(hp.pix2vec(NSIDE, np.arange(npix), nest=False))
    intensity = np.zeros(npix, dtype=np.float64)
    luminosity_pixels = np.zeros(npix, dtype=np.float64)
    offplane_intensity = np.zeros(npix, dtype=np.float64)
    offplane_luminosity = np.zeros(npix, dtype=np.float64)
    pixel_solid_angle = 4.0 * np.pi / npix
    fractions = np.linspace(0.0, 1.0, N_LOS + 1)
    reference_height = max(COLOR_REFERENCE_HEIGHT_KPC, DISK_MASK_HALF_HEIGHT_KPC)
    print(f"  NSIDE={NSIDE}, Npix={npix}; 每条有效 LOS {N_LOS} 个积分区间")
    for i0 in range(0, npix, PIXEL_CHUNK):
        i1 = min(i0 + PIXEL_CHUNK, npix)
        direction = directions[i0:i1]
        start, end = ray_cube_intervals(direction, radius_kpc)
        edges = start[:, None] + (end - start)[:, None] * fractions
        left, right = edges[:, :-1], edges[:, 1:]
        midpoint = (left + right) * 0.5
        points = SUN_OFFSET_KPC + direction[:, None, :] * midpoint[:, :, None]
        eps = sample_cube(cube, points.reshape(-1, 3), radius_kpc).reshape(i1 - i0, N_LOS)
        ds_weight, luminosity_weight = outside_slab_weights(
            left, right, direction[:, 2], DISK_MASK_HALF_HEIGHT_KPC)
        intensity[i0:i1] = np.sum(eps * ds_weight, axis=1) * KPC_TO_CM / (4.0 * np.pi)
        luminosity_pixels[i0:i1] = np.sum(eps * luminosity_weight, axis=1) * pixel_solid_angle * KPC_TO_CM**3
        ds_ref, lum_ref = outside_slab_weights(left, right, direction[:, 2], reference_height)
        offplane_intensity[i0:i1] = np.sum(eps * ds_ref, axis=1) * KPC_TO_CM / (4.0 * np.pi)
        offplane_luminosity[i0:i1] = np.sum(eps * lum_ref, axis=1) * pixel_solid_angle * KPC_TO_CM**3
        if i0 == 0 or i1 == npix or (i0 // PIXEL_CHUNK) % 8 == 0:
            print(f"    pixels {i0:6d}--{i1 - 1:6d} / {npix - 1}")
        del points, eps, edges, left, right, midpoint, ds_weight, luminosity_weight, ds_ref, lum_ref
    for values in (intensity, luminosity_pixels, offplane_intensity, offplane_luminosity):
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise RuntimeError("LOS 积分出现非有限值或负值，请检查发射率和单位。")
    return intensity, luminosity_pixels, offplane_intensity, offplane_luminosity


# =============================================================================
# 色标
# =============================================================================

def apec_table_signature():
    from yt.config import ytcfg
    from yt.fields.xray_emission_fields import data_version
    version = data_version["apec"]
    directory = APEC_DATA_DIR
    if directory is None:
        candidate = ytcfg.get("yt", "supp_data_dir")
        directory = candidate if os.path.exists(candidate) else "."
    filename = os.path.abspath(os.path.join(directory, f"apec_emissivity_v{version}.h5"))
    info = {"path": filename, "version": version}
    if os.path.exists(filename):
        stat = os.stat(filename)
        info.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return info


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
        "apec_table": apec_table_signature(),
        "yt_version": yt.__version__,
        "bh_id": BH_ID,
        "temperature": "native yt temperature; stored Temperature registered in K",
    }


def allsky_signature(snapshot_file):
    return {
        "grid": grid_signature(snapshot_file),
        "sun_offset_kpc": SUN_OFFSET_KPC.tolist(),
        "los_range_kpc": [LOS_MIN_KPC, LOS_MAX_KPC],
        "los_samples": N_LOS,
        "nside": NSIDE,
        "disk_mask_half_height_kpc": DISK_MASK_HALF_HEIGHT_KPC,
        "color_reference_height_kpc": COLOR_REFERENCE_HEIGHT_KPC,
        "sampling": "trilinear cube; ray-box-clipped midpoint epsilon; exact ds and s^2ds slab weights",
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


def make_photometry(hpxmap, luminosity_pixels, time_gyr, snapshot_file, voxel_check=None,
                    offplane_map=None, offplane_luminosity=None):
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
    summary = {
        "cache_signature": allsky_signature(snapshot_file),
        "snapshot": os.path.abspath(snapshot_file),
        "time_gyr": float(time_gyr),
        "definitions": {
            "intensity": "I_X = integral(epsilon_X ds)/(4*pi), erg/s/cm^2/sr",
            "flux": "F_X = sum(I_X dOmega), erg/s/cm^2",
            "luminosity": "L_X = sum[dOmega integral(epsilon_X s^2 ds)], erg/s; s is distance from observer",
            "selection": "same cube, LOS interval and physical |z| cut as the all-sky map",
            "emission_model": "intrinsic, optically thin APEC; no absorption or instrumental response",
            "abundances": "solar-pattern APEC metals scaled by GIZMO Metallicity_00 / 0.02; not element-by-element spectra",
            "axes": "fixed simulation xy disk plane; l=0:+x, l=90:+y, b=90:+z; no automatic disk tilt rotation",
            "background_subtracted": False,
            "hemispheres": "observer Galactic latitude; equatorial pixels split equally",
        },
        "regions": regions,
        "voxel_crosscheck": check,
    }
    if offplane_map is not None and offplane_luminosity is not None:
        reference = make_photometry(offplane_map, offplane_luminosity, time_gyr, snapshot_file)
        summary["offplane_reference"] = {
            "height_kpc": max(COLOR_REFERENCE_HEIGHT_KPC, DISK_MASK_HALF_HEIGHT_KPC),
            "regions": reference["regions"],
            "note": "Contribution from spatial |z|>=height; not a sky latitude mask and not a bubble-only selection.",
        }
    return summary


def save_photometry(summary, summary_file, map_file):
    with open(summary_file, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    csv_file = os.path.join(OUTPUT_DIR, "allsky_xray_photometry.csv")
    map_path = os.path.abspath(map_file)
    rows = []
    if os.path.exists(csv_file):
        with open(csv_file, newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("map_file") != map_path]
    region_rows = list(summary["regions"].items())
    if "offplane_reference" in summary:
        region_rows += [("offplane_" + name, values)
                        for name, values in summary["offplane_reference"]["regions"].items()]
    for name, values in region_rows:
        row = {
            "snapshot": summary["snapshot"], "time_gyr": summary["time_gyr"],
            "energy_min_keV": E_MIN_KEV, "energy_max_keV": E_MAX_KEV,
            "grid_half_width_kpc": GRID_RADIUS_KPC,
            "disk_mask_half_height_kpc": DISK_MASK_HALF_HEIGHT_KPC,
            "los_min_kpc": LOS_MIN_KPC, "los_max_kpc": LOS_MAX_KPC,
            "nside": NSIDE, "region": name, **values, "map_file": map_path,
            "sun_x_kpc": SUN_OFFSET_KPC[0], "sun_y_kpc": SUN_OFFSET_KPC[1],
            "sun_z_kpc": SUN_OFFSET_KPC[2],
            "color_reference_height_kpc": COLOR_REFERENCE_HEIGHT_KPC,
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
        if abs(ratio - 1.0) > 0.10:
            warnings.warn("L_LOS 与体素积分相差超过 10%；先提高角/径向采样并检查网格收敛，再比较绝对光度。")
    if "offplane_reference" in summary:
        values = summary["offplane_reference"]["regions"]["allsky"]
        print(f"  盘外参考：L_X={values['luminosity_erg_s']:.6e} erg/s; "
              f"F_X={values['flux_erg_s_cm2']:.6e} erg/s/cm^2")
    print(f"  光度统计：{summary_file}")
    print(f"  时间序列表：{csv_file}")

def get_color_limits(hpxmap, offplane_map):
    """用强度参考图选色标；不改主图数据或任何积分。"""
    good = hpxmap[np.isfinite(hpxmap) & (hpxmap > 0.0)]
    if good.size == 0:
        raise RuntimeError("全天球 X-ray 强度全部为 0，无法使用对数色标。")
    reference = np.asarray(offplane_map if COLOR_SCALE.startswith("offplane") else hpxmap)
    positive = reference[np.isfinite(reference) & (reference > 0.0)]
    if USE_FIXED_COLORBAR:
        vmin, vmax = float(FIXED_VMIN), float(FIXED_VMAX)
        source = "fixed physical intensity limits"
    else:
        if positive.size == 0:
            raise RuntimeError("盘外参考图没有正发射；请降低参考高度，或指定 --intensity-limits。")
        vmin = float(np.percentile(positive, COLOR_PERCENTILE_MIN))
        vmax = float(np.max(positive) if COLOR_SCALE == "offplane-max"
                     else np.percentile(positive, COLOR_PERCENTILE_MAX))
        if vmin >= vmax:
            vmin = vmax / 10.0
        source = COLOR_SCALE
    return vmin, vmax, {
        "mode": source,
        "reference_height_kpc": COLOR_REFERENCE_HEIGHT_KPC,
        "effective_reference_height_kpc": max(COLOR_REFERENCE_HEIGHT_KPC, DISK_MASK_HALF_HEIGHT_KPC),
        "vmin_erg_s_cm2_sr": vmin, "vmax_erg_s_cm2_sr": vmax,
        "lg_vmin": float(np.log10(vmin)), "lg_vmax": float(np.log10(vmax)),
        "reference_max_erg_s_cm2_sr": float(np.max(positive)) if positive.size else 0.0,
        "fraction_allsky_pixels_above_vmax": float(np.mean(hpxmap > vmax)),
        "fraction_allsky_pixels_below_vmin": float(np.mean(hpxmap < vmin)),
        "data_clipped": False,
        "note": "Only the displayed logarithmic colors saturate; saved maps and luminosities remain linear and unclipped.",
    }


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

def plot_allsky(hpxmap, time, output_png, photometry=None, offplane_map=None):
    vmin, vmax, display = get_color_limits(hpxmap, offplane_map)

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
    print(f"  色标模式={display['mode']}; 主图饱和像素比例="
          f"{display['fraction_allsky_pixels_above_vmax']:.2%}")

    plt.close("all")

    hp.mollview(
        log_map,
        nest=False,
        flip="astro",
        coord=['G'],
        rot=(0.0, 0.0, 0.0),
        notext=True,
        cmap=CMAP,
        min=log_vmin,
        max=log_vmax,
        cbar=False,
        title=(
            f"Intrinsic APEC X-ray intensity {E_MIN_KEV:g}--{E_MAX_KEV:g} keV"
            f"   |   Time = {time:.3f} Gyr"
            f"\nSun = ({SUN_OFFSET_KPC[0]:g}, {SUN_OFFSET_KPC[1]:g}, {SUN_OFFSET_KPC[2]:g}) kpc"
            + ("; disk retained" if DISK_MASK_HALF_HEIGHT_KPC == 0 else "; physical disk cut applied")
        ),
        xsize=1800
    )

    fig = plt.gcf()
    fig.set_size_inches(12.0, 8.0)
    plt.gca().set_position([0.075, 0.30, 0.90, 0.60])

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
        0.245,
        r"Galactic longitude $l$ [deg]",
        ha="center",
        fontsize=11
    )

    fig.text(
        0.025,
        0.60,
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
        0.27,
        0.19,
        0.50,
        0.025
    ])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        extend="max" if display["fraction_allsky_pixels_above_vmax"] > 0 else "neither",
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
        photometry["display"] = display
        total = photometry["regions"]["allsky"]
        north = photometry["regions"]["north"]
        south = photometry["regions"]["south"]
        if south["luminosity_erg_s"] > 0.0:
            ratio_label = f"{north['luminosity_erg_s']/south['luminosity_erg_s']:.3f}"
        else:
            ratio_label = r"\infty" if north["luminosity_erg_s"] > 0.0 else r"\mathrm{undefined}"
        fig.text(
            0.5, 0.09,
            rf"$L_X={total['luminosity_erg_s']:.3e}\ \mathrm{{erg\,s^{{-1}}}}$"
            rf";  $F_X={total['flux_erg_s_cm2']:.3e}\ \mathrm{{erg\,s^{{-1}}\,cm^{{-2}}}}$"
            rf";  $L_N/L_S={ratio_label}$",
            ha="center", fontsize=10,
        )
        fig.text(
            0.5, 0.052,
            rf"Selected gas: cube $\pm{GRID_RADIUS_KPC:g}$ kpc; "
            rf"$|z|\geq {DISK_MASK_HALF_HEIGHT_KPC:g}$ kpc; "
            rf"LOS {LOS_MIN_KPC:g}--{LOS_MAX_KPC:g} kpc; unabsorbed, no background subtraction",
            ha="center", fontsize=9,
        )
        fig.text(0.5, 0.022,
                 f"Color scale: {display['mode']}; off-plane reference |z| >= "
                 f"{display['effective_reference_height_kpc']:g} kpc; "
                 f"saturated sky pixels {display['fraction_allsky_pixels_above_vmax']:.1%}",
                 ha="center", fontsize=9)

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
    root = os.path.splitext(npy_file)[0]
    luminosity_file = root + "_luminosity_erg_s.npy"
    reference_file = root + "_offplane_reference.npy"
    reference_luminosity_file = root + "_offplane_luminosity_erg_s.npy"
    summary_file = root + "_photometry.json"
    array_files = (npy_file, luminosity_file, reference_file, reference_luminosity_file)
    cached = None
    if not OVERWRITE_ALLSKY and all(os.path.exists(p) for p in (*array_files, summary_file)):
        try:
            with open(summary_file, encoding="utf-8") as stream:
                candidate = json.load(stream)
            if isinstance(candidate, dict) and candidate.get("cache_signature") == allsky_signature(snapshot_file):
                cached = candidate
        except (OSError, ValueError, TypeError):
            pass
    if cached is not None:
        arrays = [np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
                  for path in array_files]
        if any(arr.size != hp.nside2npix(NSIDE) for arr in arrays):
            raise RuntimeError("缓存像素数与 NSIDE 不符；请去掉 --reuse-allsky 重建。")
        time = cached["time_gyr"]
        voxel_check = cached.get("voxel_crosscheck")
        print("  复用已校验的主图、光度和盘外参考线性数组。")
    else:
        if OVERWRITE_GRID or not grid_cache_matches(snapshot_file, grid_file):
            build_xray_grid(snapshot_file, grid_file)
        else:
            print(f"  使用已核对物理参数的规则网格：{grid_file}")
        cube, time, radius_kpc = load_xray_grid(grid_file)
        arrays = make_allsky(cube, radius_kpc)
        voxel_check = cube_luminosity_check(cube, radius_kpc)
        for filename, array in zip(array_files, arrays):
            np.save(filename, array)
        del cube
    hpxmap, luminosity_pixels, offplane_map, offplane_luminosity = arrays
    photometry = make_photometry(
        hpxmap, luminosity_pixels, time, snapshot_file, voxel_check,
        offplane_map=offplane_map, offplane_luminosity=offplane_luminosity)
    # 先确定并记录色标；任何显示操作都不会改动上面的线性数组。
    _, _, photometry["display"] = get_color_limits(hpxmap, offplane_map)
    photometry["output_files"] = {
        "intensity": os.path.abspath(npy_file), "luminosity": os.path.abspath(luminosity_file),
        "offplane_intensity": os.path.abspath(reference_file),
        "offplane_luminosity": os.path.abspath(reference_luminosity_file),
    }
    save_photometry(photometry, summary_file, npy_file)
    plot_allsky(hpxmap, time, png_file, photometry=photometry, offplane_map=offplane_map)
    del arrays, hpxmap, luminosity_pixels, offplane_map, offplane_luminosity
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
    global COLOR_REFERENCE_HEIGHT_KPC, COLOR_SCALE, BH_ID
    global COLOR_PERCENTILE_MIN, COLOR_PERCENTILE_MAX
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
    parser.add_argument("--disk-mask-half-height", type=float, default=DISK_MASK_HALF_HEIGHT_KPC,
                        help="实际排除 |z|<此值的发射；默认 0 保留银盘，会影响光度")
    parser.add_argument("--color-reference-height-kpc", type=float, default=COLOR_REFERENCE_HEIGHT_KPC,
                        help="盘外参考图的 |z| 下限，默认 5 kpc；不筛除主图发射")
    parser.add_argument("--color-scale", choices=("offplane-max", "offplane-percentile", "allsky-percentile"),
                        default=COLOR_SCALE, help="默认严格采用盘外参考图的最大强度为色标上限")
    parser.add_argument("--color-percentile-min", type=float, default=COLOR_PERCENTILE_MIN)
    parser.add_argument("--color-percentile-max", type=float, default=COLOR_PERCENTILE_MAX)
    parser.add_argument("--bh-id", type=int, default=None, help="多颗 BH 时必须指定 MW 的 BH ID")
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
               args.disk_mask_half_height, args.color_reference_height_kpc,
               args.color_percentile_min, args.color_percentile_max,
               args.emin_kev, args.emax_kev, *args.sun_offset_kpc]
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
    if not (0 <= args.color_reference_height_kpc < args.grid_radius_kpc):
        parser.error("要求 0 <= color-reference-height-kpc < grid-radius-kpc。")
    if not (0 <= args.color_percentile_min < args.color_percentile_max <= 100):
        parser.error("要求 0 <= color-percentile-min < color-percentile-max <= 100。")
    if np.any(np.abs(args.sun_offset_kpc) >= args.grid_radius_kpc):
        parser.error("太阳必须位于采集 cube 内；请增大 grid-radius-kpc。")
    if not (0 < args.emin_kev < args.emax_kev):
        parser.error("要求 0 < emin-kev < emax-kev。")
    if args.stride < 1 or (args.start is not None and args.stop is not None and args.start > args.stop):
        parser.error("stride 必须 >=1，start 不能大于 stop。")
    USE_FIXED_COLORBAR = args.intensity_limits is not None
    if args.intensity_limits is not None:
        low, high = args.intensity_limits
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
            parser.error("intensity-limits 必须为递增的有限正数。")
        USE_FIXED_COLORBAR = True
        FIXED_VMIN, FIXED_VMAX = low, high
    SNAPSHOT_DIR = args.snapshot_dir
    OUTPUT_DIR = args.output_dir or os.path.join(SNAPSHOT_DIR, "figures_allsky_xray_v3")
    GRID_DIR = args.grid_dir or os.path.join(SNAPSHOT_DIR, "allsky_xray_grids_v3")
    GRID_RADIUS_KPC = args.grid_radius_kpc
    GRID_DIMS = (args.grid_n,) * 3
    NSIDE, N_LOS, PIXEL_CHUNK = args.nside, args.n_los, args.pixel_chunk
    LOS_MIN_KPC, LOS_MAX_KPC = args.los_min_kpc, args.los_max_kpc
    SUN_OFFSET_KPC = np.asarray(args.sun_offset_kpc, dtype=np.float64)
    DISK_MASK_HALF_HEIGHT_KPC = args.disk_mask_half_height
    COLOR_REFERENCE_HEIGHT_KPC = args.color_reference_height_kpc
    COLOR_SCALE = args.color_scale
    COLOR_PERCENTILE_MIN, COLOR_PERCENTILE_MAX = args.color_percentile_min, args.color_percentile_max
    BH_ID = args.bh_id
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
    snapshots = [path for path in snapshots if re.fullmatch(r"snapshot_\d+(?:\.0)?\.hdf5", os.path.basename(path))]
    snapshots = [path for path in snapshots
                 if (START_SNAPSHOT is None or snapshot_index(path) >= START_SNAPSHOT)
                 and (STOP_SNAPSHOT is None or snapshot_index(path) <= STOP_SNAPSHOT)][::SNAPSHOT_STRIDE]

    if not snapshots:
        raise FileNotFoundError(
            f"没有找到："
            f"{os.path.join(SNAPSHOT_DIR, SNAPSHOT_GLOB)}"
        )

    cell_size = 2.0 * GRID_RADIUS_KPC / GRID_DIMS[0]
    los_step = (LOS_MAX_KPC - LOS_MIN_KPC) / N_LOS

    print("脚本版本：allsky-v3-disk-preserved / 2026-10-08")
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
    print(f"色标模式={COLOR_SCALE if not USE_FIXED_COLORBAR else 'fixed'}；"
          f"盘外参考高度={COLOR_REFERENCE_HEIGHT_KPC:g} kpc（不删除主图气体）")
    print(
        f"LOS = {LOS_MIN_KPC:.2f}-{LOS_MAX_KPC:.2f} kpc, "
        f"N_LOS={N_LOS}, 每区间 ds<={los_step:.4f} kpc"
    )
    print(f"NSIDE={NSIDE}, CMAP={CMAP}")
    angular_deg = hp.nside2resol(NSIDE, arcmin=True) / 60.0
    farthest_corner = np.linalg.norm(GRID_RADIUS_KPC + np.abs(SUN_OFFSET_KPC))
    print(f"HEALPix 像素面积等效角尺度≈{angular_deg:.4f} deg；"
          f"太阳到 cube 最远角点={farthest_corner:.3f} kpc")
    if LOS_MAX_KPC < farthest_corner:
        warnings.warn("LOS 上限未覆盖全部 cube 角点；图和光度只代表 cube 与太阳距离范围的交集。")
    if los_step > cell_size:
        warnings.warn("最大 LOS 积分步长大于网格宽度；建议提高 --n-los。")
    if not (SUN_OFFSET_KPC[0] < 0 and np.allclose(SUN_OFFSET_KPC[1:], 0.0)):
        warnings.warn("自定义太阳位置偏离负 x 轴；当前 l=0 仍指向 +x，GC 标签不再严格指向银心。")
    print(f"OVERWRITE_GRID={OVERWRITE_GRID}")
    print(f"OVERWRITE_ALLSKY={OVERWRITE_ALLSKY}")

    for snapshot_file in snapshots:
        process_snapshot(snapshot_file)

    print("\n全天球图及通量/光度统计处理完成。")


if __name__ == "__main__":
    main()
