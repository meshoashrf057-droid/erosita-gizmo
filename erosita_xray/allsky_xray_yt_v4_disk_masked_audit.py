#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GIZMO -> yt/APEC -> BH-centered X-ray cube (v4-disk-masked-audit)
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
    默认采集 BH 中心 ±40 kpc 的发射率，仅积分 |z|>=5 kpc 的发射。
    主图、通量、光度使用同一高度筛选；默认保留的气体不等于气泡净辐射。
    默认色标上限为筛选后全天图最大值，所有颜色使用 lg 强度。
    同时记录未筛选、排除和保留的 LOS 光度，以及原始粒子 sum(epsilon*m/rho)。
    L_LOS/L_voxel 检查射线积分；L_voxel/L_particles 检查场重建/网格影响。
    这两个比值都不包含观测背景扣除，也不会强制光度归一到观测值。
    默认用 yt scatter 按每粒子光度作核叠加，不作 Shepard 归一化。
    屏蔽分两步：先将盘内粒子的发射置零，再仅积分盘外路径；
    防止盘内亮气体的平滑核污染盘外图。有限核跨边界仍需收敛检查。
    5 kpc 是距 z=0 盘面的高度，不是球半径或银纬；盘面沿模拟 xy 平面。
    高于色标上限的银盘像素饱和；色标不能分离同一视线上的盘/晕发射。
    射线离开该cube后 emissivity=0。
    色条显示 lg(I_X/单位)，.npy 和所有积分保留线性物理量，色标截断不影响总量。
    每帧输出原始强度、每像素光度、photometry.json，并更新 allsky_xray_photometry.csv。
    统计全天及南北半球；结果为所选全部气体的本征 APEC 发射，未包含吸收、
    仪器响应或背景扣除，不能直接称为“气泡净光度”。

运行：
    python allsky_xray_yt_v4_disk_masked_audit.py --snapshot-dir ./allsky
    python allsky_xray_yt_v4_disk_masked_audit.py --snapshot-dir ./allsky --reuse-grid --reuse-allsky
    python allsky_xray_yt_v4_disk_masked_audit.py --snapshot-dir ./allsky --intensity-limits 1e-12 5e-8

    --disk-mask-half-height 默认 5 kpc；设为 0 才会保留全部盘面。
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
SELECTED_XRAY_FIELD = ("gas", "selected_xray_emissivity")
EMISSIVITY_RECONSTRUCTION = "luminosity"

# 中央区域采集范围：±GRID_RADIUS_KPC；这是立方体半宽，不是球半径。
GRID_RADIUS_KPC = 30.0
GRID_DIMS = (256, 256, 256)

# 首次运行重建网格；之后可用 --reuse-grid 复用已校验的缓存。
OVERWRITE_GRID = True

SUN_OFFSET_KPC = np.array([-8.29, 0.0, 0.0], dtype=np.float64)
DISK_MASK_HALF_HEIGHT_KPC = 5.0
COLOR_REFERENCE_HEIGHT_KPC = 5.0
COLOR_SCALE = "offplane-max"
BH_ID = None
APEC_DATA_DIR = None
CACHE_SCHEMA_VERSION = 4
START_SNAPSHOT = None
STOP_SNAPSHOT = None
SNAPSHOT_STRIDE = 1

# 太阳LOS；cube外自动epsilon=0
LOS_MIN_KPC = 0.0          #I 的 ds 积分在观察者处没有 1/s^2 奇点；默认不人为挖掉本地 50 pc。
LOS_MAX_KPC = 60.0         #每条 LOS 的最大长度；cube 外发射率为零。这不是以 GC 为中心的球半径。
N_LOS = 1024                #每条射线有效 cube 路径的中点积分区间数；并非物理分辨率。

NSIDE = 128                #HEALPix 全天球角分辨率参数。
PIXEL_CHUNK = 1024         #同时采样完整/屏蔽两张 cube，降低每批峰值内存。

# 可用 --reuse-allsky 复用已校验的强度和光度数组。
OVERWRITE_ALLSKY = True

CMAP = "inferno"           #最终全天球图使用的 matplotlib colormap。 纯绘图参数，不改变物理量。
DPI = 220                  #PNG 输出分辨率。 纯绘图参数。

COLOR_PERCENTILE_MIN = 1.0 #自动色标下限采用所有正有限像素的第 1 百分位。 只影响显示，不改变 .npy 数据。
COLOR_PERCENTILE_MAX = 99.5 #自动色标上限采用第 99.5 百分位。 防止极少数超亮像素压缩整个图的动态范围。

# 正式比较多个snapshot时可改True，所有图用同一色标
USE_FIXED_COLORBAR = True  #是否强制多个 snapshot 使用统一物理色标。 做时间演化/孤立盘与 LMC 比较时建议最终改为 True。
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

def prepare_xray_dataset(snapshot_file):
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
    # yt scatter 的未归一化公式为 sum_i (m_i/rho_i)*epsilon_i*W_i。
    # W_i 对体积归一化，因此完整、充分解析的核积分为 sum_i L_i。
    # 归一化插值另除以 sum_i (m_i/rho_i)*W_i，并不保证这个光度总量。
    ds.sph_smoothing_style = "scatter"
    ds.use_sph_normalization = EMISSIVITY_RECONSTRUCTION == "normalized"

    def _selected_emissivity(field, data):
        z = data["PartType0", "particle_position_z"].to_value("kpc") - bh[2]
        keep = np.abs(z) >= DISK_MASK_HALF_HEIGHT_KPC
        return data[XRAY_FIELD] * keep

    ds.add_field(SELECTED_XRAY_FIELD, function=_selected_emissivity,
                 sampling_type="local", units="erg/s/cm**3", force_override=True)
    print(f"  发射率重建 = {EMISSIVITY_RECONSTRUCTION}; 盘内粒子发射在核重建前置零")

    return ds, bh, time, time_code


def particle_audit_signature(snapshot_file):
    return {
        "grid": grid_signature(snapshot_file),
        "sun_offset_kpc": SUN_OFFSET_KPC.tolist(),
        "los_range_kpc": [LOS_MIN_KPC, LOS_MAX_KPC],
        "disk_mask_half_height_kpc": DISK_MASK_HALF_HEIGHT_KPC,
        "method": "particle-centre selection; sum epsilon_i*m_i/rho_i; v1",
    }


def particle_luminosity_audit(ds, bh_kpc):
    """独立于规则网格/HEALPix 的气体粒子光度求和。

    这里直接在原始粒子上求 APEC 发射率，按每粒子体积 m/rho 积分。
    选择条件与主图具有相同的几何定义，但边界按粒子中心判断，
    而主图筛选的是核重建后的连续场，两者在有限分辨率下不必严格相等。
    """
    fields = [("PartType0", "Coordinates"), ("PartType0", "Masses"),
              ("PartType0", "Density"), XRAY_FIELD, ("gas", "temperature"),
              ("gas", "H_nuclei_density"), ("gas", "apec_metallicity")]
    groups = {
        key: {"particle_count": 0, "luminosity_erg_s": 0.0}
        for key in ("unmasked", "excluded_disk", "retained", "north", "south")
    }
    temp_bins = [(0., 1e5, "T_lt_1e5"), (1e5, 1e6, "T_1e5_1e6"),
                 (1e6, 1e7, "T_1e6_1e7"), (1e7, np.inf, "T_ge_1e7")]
    density_bins = [(0., 1e-4, "nH_lt_1e-4"), (1e-4, 1e-3, "nH_1e-4_1e-3"),
                    (1e-3, 1e-2, "nH_1e-3_1e-2"), (1e-2, .1, "nH_1e-2_1e-1"),
                    (.1, np.inf, "nH_ge_1e-1")]
    temperature_luminosity = {name: 0.0 for _, _, name in temp_bins}
    density_luminosity = {name: 0.0 for _, _, name in density_bins}
    weighted = {"temperature_K": 0.0, "nH_cm3": 0.0, "metallicity_Zsun": 0.0}
    max_particle_luminosity = 0.0
    print("  独立核查：原始粒子 L_X = sum(epsilon_X * m/rho) ...")
    for chunk in ds.all_data().chunks(fields, "io"):
        xyz = chunk["PartType0", "Coordinates"].to_value("kpc") - bh_kpc
        distance2 = np.sum((xyz - SUN_OFFSET_KPC)**2, axis=1)
        base = (np.all(np.abs(xyz) <= GRID_RADIUS_KPC, axis=1)
                & (distance2 >= LOS_MIN_KPC**2) & (distance2 <= LOS_MAX_KPC**2))
        if not np.any(base):
            continue
        eps = chunk[XRAY_FIELD].to_value("erg/s/cm**3")
        mass = chunk["PartType0", "Masses"].to_value("g")
        rho = chunk["PartType0", "Density"].to_value("g/cm**3")
        temperature = chunk["gas", "temperature"].to_value("K")
        nh = chunk["gas", "H_nuclei_density"].to_value("cm**-3")
        metal = chunk["gas", "apec_metallicity"].to_value("Zsun")
        values = (eps, mass, rho, temperature, nh, metal)
        if any(a.shape != base.shape for a in values):
            raise RuntimeError("原始气体粒子字段长度不一致，无法独立核查光度。")
        if (any(np.any(~np.isfinite(a[base])) or np.any(a[base] < 0) for a in values)
                or np.any(rho[base] <= 0)):
            raise RuntimeError("所选原始气体字段有负值/非有限值/零密度；请先检查快照。")
        lum = np.zeros_like(eps, dtype=np.float64)
        lum[base] = eps[base] * (mass[base] / rho[base])
        keep = base & (np.abs(xyz[:, 2]) >= DISK_MASK_HALF_HEIGHT_KPC)
        masks = {"unmasked": base, "retained": keep, "excluded_disk": base & ~keep}
        for key, mask in masks.items():
            groups[key]["particle_count"] += int(np.count_nonzero(mask))
            groups[key]["luminosity_erg_s"] += float(np.sum(lum[mask], dtype=np.float64))
        # 与天空南北半球一致；z_sun 默认为 0。
        dz = xyz[:, 2] - SUN_OFFSET_KPC[2]
        for key, sign in (("north", 1.), ("south", -1.)):
            weight = np.where(dz == 0, 0.5, (sign * dz > 0).astype(float))
            groups[key]["particle_count"] += float(np.sum(weight[keep]))
            groups[key]["luminosity_erg_s"] += float(np.sum(lum[keep] * weight[keep]))
        for lo, hi, name in temp_bins:
            temperature_luminosity[name] += float(np.sum(lum[keep & (temperature >= lo) & (temperature < hi)]))
        for lo, hi, name in density_bins:
            density_luminosity[name] += float(np.sum(lum[keep & (nh >= lo) & (nh < hi)]))
        for name, field_values in (("temperature_K", temperature), ("nH_cm3", nh), ("metallicity_Zsun", metal)):
            weighted[name] += float(np.sum(lum[keep] * field_values[keep]))
        if np.any(keep):
            max_particle_luminosity = max(max_particle_luminosity, float(np.max(lum[keep])))
    selected = groups["retained"]["luminosity_erg_s"]
    total = groups["unmasked"]["luminosity_erg_s"]
    return {
        "definition": "sum_i epsilon_X(T_i,nH_i,ne_i,Z_i) * m_i/rho_i; erg/s",
        "selection": "particle centres in the same BH-centred cube and observer LOS range; retained also has |z|>=cut",
        "groups": groups,
        "retained_fraction": selected / total if total > 0 else None,
        "retained_temperature_bins_luminosity_erg_s": temperature_luminosity,
        "retained_nH_bins_luminosity_erg_s": density_luminosity,
        "retained_luminosity_weighted_means": {k: v / selected if selected > 0 else None for k, v in weighted.items()},
        "retained_brightest_particle_fraction": max_particle_luminosity / selected if selected > 0 else None,
        "yt_sph_normalization": bool(getattr(ds, "use_sph_normalization", True)),
        "yt_sph_smoothing_style": str(getattr(ds, "sph_smoothing_style", "scatter")),
        "code_units": {"length_kpc": float(ds.length_unit.to_value("kpc")),
                       "mass_msun": float(ds.mass_unit.to_value("Msun")),
                       "velocity_kms": float(ds.velocity_unit.to_value("km/s"))},
        "note": "Independent quadrature check, not an automatic calibration. SPH kernel support across boundaries, grid resolution, and normalized interpolation can cause differences.",
    }


def load_particle_audit(snapshot_file, grid_file):
    signature = particle_audit_signature(snapshot_file)
    with h5py.File(grid_file, "r") as handle:
        if json.loads(handle.attrs.get("particle_audit_signature", "null")) == signature:
            return json.loads(handle.attrs["particle_audit"])
    # A different geometric selection need not rebuild the unmasked cube.
    ds, bh, _, _ = prepare_xray_dataset(snapshot_file)
    audit = particle_luminosity_audit(ds, bh)
    del ds
    with h5py.File(grid_file, "r+") as handle:
        handle.attrs["particle_audit_signature"] = json.dumps(signature, sort_keys=True)
        handle.attrs["particle_audit"] = json.dumps(audit, allow_nan=False)
    return audit


def luminosity_audit_ratios(summary):
    los = summary["regions"]["allsky"]["luminosity_erg_s"]
    voxel = summary["voxel_crosscheck"]["selected_voxel_luminosity_erg_s"]
    particles = summary["particle_audit"]["groups"]["retained"]["luminosity_erg_s"]
    return {
        "L_los_erg_s": los, "L_voxel_erg_s": voxel, "L_particles_erg_s": particles,
        "los_to_voxel": los / voxel if voxel > 0 else None,
        "voxel_to_particles": voxel / particles if particles > 0 else None,
        "los_to_particles": los / particles if particles > 0 else None,
        "note": "The map reports L_los. Agreement of LOS and voxel sums alone cannot validate particle-to-grid reconstruction. No result is rescaled to match observations.",
    }


def build_xray_grid(snapshot_file, grid_file):
    ds, bh, time, time_code = prepare_xray_dataset(snapshot_file)
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
    selected_cube = np.asarray(grid[SELECTED_XRAY_FIELD].to_value("erg/s/cm**3"), dtype=np.float32)

    if np.any(~np.isfinite(cube)) or np.any(cube < 0.0):
        raise RuntimeError("APEC cube 出现负值或非有限值；停止，避免静默清零影响光度。")
    if np.any(~np.isfinite(selected_cube)) or np.any(selected_cube < 0.0):
        raise RuntimeError("屏蔽后的 APEC cube 有非法值。")

    particle_audit = particle_luminosity_audit(ds, bh)

    with h5py.File(grid_file, "w") as f:
        f.create_dataset(
            "xray_emissivity",
            data=cube,
            compression="gzip",
            compression_opts=1
        )
        f.create_dataset("selected_xray_emissivity", data=selected_cube,
                         compression="gzip", compression_opts=1)

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
        f.attrs["particle_audit_signature"] = json.dumps(particle_audit_signature(snapshot_file), sort_keys=True)
        f.attrs["particle_audit"] = json.dumps(particle_audit, allow_nan=False)

    print(f"  已保存规则网格：{grid_file}")

    del cube, selected_cube, grid, ds
    gc.collect()


def load_xray_grid(grid_file):
    with h5py.File(grid_file, "r") as f:
        cube = np.asarray(
            f["xray_emissivity"][...],
            dtype=np.float32
        )
        selected_cube = np.asarray(f["selected_xray_emissivity"][...], dtype=np.float32)

        time = float(f.attrs["time"])
        radius_kpc = float(f.attrs["grid_radius_kpc"])

    cube = np.nan_to_num(
        cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    cube[cube < 0.0] = 0.0

    if np.any(~np.isfinite(selected_cube)) or np.any(selected_cube < 0):
        raise RuntimeError("缓存的屏蔽发射率场有非法值；请重建网格。")
    return cube, selected_cube, time, radius_kpc


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


def make_allsky(cube, radius_kpc, unmasked_cube=None):
    """保留主图和盘外参考图；所有输出都是线性物理量。

    yt cube 的发射率经三线性插值后中点积分。对 I 积分 ds，
    对 L 积分 s^2 ds；绝不对 lg(epsilon) 积分或用统一距离换算 L。
    """
    npix = hp.nside2npix(NSIDE)
    if unmasked_cube is None:
        unmasked_cube = cube
    directions = np.column_stack(hp.pix2vec(NSIDE, np.arange(npix), nest=False))
    intensity = np.zeros(npix, dtype=np.float64)
    luminosity_pixels = np.zeros(npix, dtype=np.float64)
    offplane_intensity = np.zeros(npix, dtype=np.float64)
    offplane_luminosity = np.zeros(npix, dtype=np.float64)
    pixel_solid_angle = 4.0 * np.pi / npix
    unmasked_luminosity = 0.0
    unmasked_flux = 0.0
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
        eps_full = sample_cube(unmasked_cube, points.reshape(-1, 3), radius_kpc).reshape(i1 - i0, N_LOS)
        width = right - left
        moment = width * (left**2 + left * right + right**2) / 3.0
        unmasked_flux += float(np.sum(eps_full * width)) * KPC_TO_CM * pixel_solid_angle / (4.0 * np.pi)
        unmasked_luminosity += float(np.sum(eps_full * moment)) * pixel_solid_angle * KPC_TO_CM**3
        del width, moment, eps_full
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
    retained_luminosity = float(np.sum(luminosity_pixels))
    retained_flux = float(np.sum(intensity)) * pixel_solid_angle
    height_budget = {
        "unmasked_luminosity_erg_s": unmasked_luminosity,
        "retained_luminosity_erg_s": retained_luminosity,
        "excluded_luminosity_erg_s": max(0.0, unmasked_luminosity - retained_luminosity),
        "retained_luminosity_fraction": retained_luminosity / unmasked_luminosity if unmasked_luminosity > 0 else None,
        "unmasked_flux_erg_s_cm2": unmasked_flux,
        "retained_flux_erg_s_cm2": retained_flux,
        "excluded_flux_erg_s_cm2": max(0.0, unmasked_flux - retained_flux),
        "note": "Same cube and LOS range. Retained emission excludes disk-centred source particles before kernel reconstruction AND excludes disk LOS paths. Not a bubble-only luminosity.",
    }
    return (intensity, luminosity_pixels, offplane_intensity, offplane_luminosity), height_budget


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
        "emissivity_reconstruction": EMISSIVITY_RECONSTRUCTION,
        "source_particle_disk_mask_half_height_kpc": DISK_MASK_HALF_HEIGHT_KPC,
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
                and "selected_xray_emissivity" in handle
                and handle.attrs.get("time_unit") == "Gyr"
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cube_luminosity_check(cube, radius_kpc):
    """体素求和；高度边界按体素体积比例裁剪，LOS 距离按 cell 中心。"""
    nx, ny, nz = cube.shape
    dx, dy, dz = 2.0 * radius_kpc / np.array([nx, ny, nz], dtype=float)
    x = -radius_kpc + (np.arange(nx) + 0.5) * dx
    y = -radius_kpc + (np.arange(ny) + 0.5) * dy
    z = -radius_kpc + (np.arange(nz) + 0.5) * dz
    yz_distance2 = (y[:, None] - SUN_OFFSET_KPC[1])**2 + (z[None, :] - SUN_OFFSET_KPC[2])**2
    z_left, z_right = z - 0.5 * dz, z + 0.5 * dz
    slab_width = np.maximum(0.0, np.minimum(z_right, DISK_MASK_HALF_HEIGHT_KPC)
                            - np.maximum(z_left, -DISK_MASK_HALF_HEIGHT_KPC))
    z_keep = (1.0 - slab_width / dz)[None, :]
    selected_emissivity_sum = 0.0
    unmasked_sum = 0.0
    for index, x_value in enumerate(x):
        distance2 = (x_value - SUN_OFFSET_KPC[0])**2 + yz_distance2
        distance_keep = (distance2 >= LOS_MIN_KPC**2) & (distance2 <= LOS_MAX_KPC**2)
        unmasked_sum += np.sum(cube[index] * distance_keep, dtype=np.float64)
        selected_emissivity_sum += np.sum(cube[index] * distance_keep * z_keep, dtype=np.float64)
    cell_volume_cm3 = dx * dy * dz * KPC_TO_CM**3
    return {
        "whole_cube_luminosity_erg_s": float(np.sum(cube, dtype=np.float64) * cell_volume_cm3),
        "selected_voxel_luminosity_erg_s": float(selected_emissivity_sum * cell_volume_cm3),
        "unmasked_within_los_voxel_luminosity_erg_s": float(unmasked_sum * cell_volume_cm3),
        "note": "Piecewise-constant voxel quadrature with fractional |z| boundary cells and cell-centre LOS radial cuts. Compare with trilinear LOS integration for convergence.",
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
            "source_mask": "particle centres with |z|<cut have zero emissivity before yt kernel reconstruction; LOS paths with |z|<cut are also excluded",
            "emissivity_reconstruction": EMISSIVITY_RECONSTRUCTION,
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
        audit = summary.get("luminosity_audit", {})
        budget = summary.get("height_cut_budget", {})
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
            "allsky_unmasked_L_erg_s": budget.get("unmasked_luminosity_erg_s"),
            "allsky_excluded_L_erg_s": budget.get("excluded_luminosity_erg_s"),
            "allsky_retained_fraction": budget.get("retained_luminosity_fraction"),
            "allsky_particle_L_erg_s": audit.get("L_particles_erg_s"),
            "allsky_L_los_over_voxel": audit.get("los_to_voxel"),
            "allsky_L_voxel_over_particles": audit.get("voxel_to_particles"),
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
    if "height_cut_budget" in summary:
        budget = summary["height_cut_budget"]
        print("  高度筛选光度收支（同一 cube/LOS）：")
        print(f"    未屏蔽 = {budget['unmasked_luminosity_erg_s']:.6e} erg/s")
        print(f"    排除   = {budget['excluded_luminosity_erg_s']:.6e} erg/s")
        print(f"    保留   = {budget['retained_luminosity_erg_s']:.6e} erg/s")
        fraction = budget['retained_luminosity_fraction']
        if fraction is not None:
            print(f"    保留比例 = {fraction:.3%}")
    if "luminosity_audit" in summary:
        audit = summary["luminosity_audit"]
        print(f"  独立粒子求和 L_particles = {audit['L_particles_erg_s']:.6e} erg/s")
        print(f"  网格/粒子 L_voxel/L_particles = {audit['voxel_to_particles']}")
        print(f"  LOS/粒子 L_LOS/L_particles = {audit['los_to_particles']}")
        ratio = audit['voxel_to_particles']
        if ratio is not None and (ratio > 1.2 or ratio < 0.8):
            warnings.warn("网格与原始粒子光度相差超过 20%；这不是观测标定误差。检查 SPH 归一化插值、网格分辨率和边界核支持，不要自动缩放到观测光度。")
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
            rf"$L_{{X,\mathrm{{selected}}}}={total['luminosity_erg_s']:.3e}\ \mathrm{{erg\,s^{{-1}}}}$"
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
            if (isinstance(candidate, dict)
                    and candidate.get("cache_signature") == allsky_signature(snapshot_file)
                    and "particle_audit" in candidate and "height_cut_budget" in candidate):
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
        particle_audit = cached["particle_audit"]
        height_budget = cached["height_cut_budget"]
        print("  复用已校验的主图、光度和盘外参考线性数组。")
    else:
        if OVERWRITE_GRID or not grid_cache_matches(snapshot_file, grid_file):
            build_xray_grid(snapshot_file, grid_file)
        else:
            print(f"  使用已核对物理参数的规则网格：{grid_file}")
        cube, selected_cube, time, radius_kpc = load_xray_grid(grid_file)
        arrays, height_budget = make_allsky(selected_cube, radius_kpc, unmasked_cube=cube)
        particle_audit = load_particle_audit(snapshot_file, grid_file)
        voxel_check = cube_luminosity_check(selected_cube, radius_kpc)
        voxel_check["unmasked_cube_luminosity_erg_s"] = float(np.sum(cube, dtype=np.float64)
            * (2 * radius_kpc / cube.shape[0] * KPC_TO_CM)**3)
        for filename, array in zip(array_files, arrays):
            np.save(filename, array)
        del cube, selected_cube
    hpxmap, luminosity_pixels, offplane_map, offplane_luminosity = arrays
    photometry = make_photometry(
        hpxmap, luminosity_pixels, time, snapshot_file, voxel_check,
        offplane_map=offplane_map, offplane_luminosity=offplane_luminosity)
    photometry["height_cut_budget"] = height_budget
    photometry["particle_audit"] = particle_audit
    photometry["luminosity_audit"] = luminosity_audit_ratios(photometry)
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
    global EMISSIVITY_RECONSTRUCTION
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
                        help="实际排除 |z|<此值的发射；默认 5 kpc；设为 0 保留银盘，会影响光度")
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
    parser.add_argument("--emissivity-reconstruction", choices=("luminosity", "normalized"),
                        default=EMISSIVITY_RECONSTRUCTION,
                        help="luminosity=yt 按粒子光度作未归一化核叠加（默认）；normalized=采用旧版归一化插值供对照")
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
    OUTPUT_DIR = args.output_dir or os.path.join(SNAPSHOT_DIR, "figures_allsky_xray_v4_masked")
    GRID_DIR = args.grid_dir or os.path.join(SNAPSHOT_DIR, "allsky_xray_grids_v4")
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
    EMISSIVITY_RECONSTRUCTION = args.emissivity_reconstruction
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

    print("脚本版本：allsky-v4-disk-masked-audit / 2026-10-08")
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
          f"色标参考高度={COLOR_REFERENCE_HEIGHT_KPC:g} kpc；"
          f"实际气体/路径屏蔽高度={DISK_MASK_HALF_HEIGHT_KPC:g} kpc")
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
    print("注意：屏蔽盘面后仍包括背景 CGM；所选全天总光度不是背景扣除后的气泡净光度。")


if __name__ == "__main__":
    main()
