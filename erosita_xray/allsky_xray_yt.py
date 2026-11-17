#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIZMO HDF5 三维数据 -> 黑洞中心规则网格 -> 太阳视角 HEALPix 软 X-ray 全天球图

脚本设计依据你提供的三个参考脚本：
1. firelib.py：先将 GIZMO/SPH 场插值到 yt 规则网格，便于后续重复采样。
2. pressure_at_surface.py：使用 HEALPix 方向、yt 多点采样以及 Mollweide 全天球绘图。
3. erosita_bubble_xray_v6.py：使用中心黑洞定位银河系中心，并通过 yt/APEC
   构造 0.6--1.0 keV 软 X-ray 发射率场。

坐标约定（全部采用“相对中心黑洞”的笛卡尔坐标）：
    银河系中心 / BH = (0, 0, 0) kpc
    太阳观测点       = (-8.2, 0, 0) kpc
    l =   0 deg      = +x，太阳看向银河系中心
    l =  90 deg      = +y
    b = +90 deg      = +z

全天球图绘制的是光学薄近似下的软 X-ray 面亮度：
    I_X = (1 / 4pi) * integral epsilon_X ds
单位：erg s^-1 cm^-2 sr^-1

本版本不加入银河系前景吸收。
"""

import os
import glob
import gc
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import healpy as hp
import yt


# =============================================================================
# 0. 用户参数
# =============================================================================

# 快照所在目录。根据你本次运行日志，默认使用 ./allsky
SNAPSHOT_DIR = "./allsky"

# 输出全天球图片与 HEALPix 数组
OUTPUT_DIR = os.path.join(SNAPSHOT_DIR, "figures_allsky_xray")

# 保存 GIZMO/SPH -> 规则网格的缓存文件
GRID_DIR = os.path.join(SNAPSHOT_DIR, "allsky_xray_grids")

# 批量输入方式：
#   所有快照："snapshot_*.hdf5"
#   单个快照："snapshot_043.hdf5"
SNAPSHOT_GLOB = "snapshot_*.hdf5"

# 软 X-ray 能段。yt 的能量输入单位是 keV。
E_MIN_KEV = 0.6
E_MAX_KEV = 1.0
TABLE_TYPE = "apec"

# 如果 yt 找不到 apec_emissivity_v3.h5，
# 将这里改成该文件所在目录，例如：
# APEC_DATA_DIR = "/home/xingdi/yt_data"
APEC_DATA_DIR = None

# -----------------------------------------------------------------------------
# 金属丰度设置
# -----------------------------------------------------------------------------
# 标准 GIZMO/yt 情况：
#   Metallicity_00 = 总金属质量分数 Z
#   Metallicity_01 = He
#   Metallicity_02... = C、N、O ...
# 因此标准 GIZMO 数据不要将 00--10 全部求和。
#
# 推荐保持："gizmo_total"
# 只有确认自己的 HDF5 是“11 个互不重复的金属质量分数”时，
# 才使用："sum_11_custom"
METALLICITY_MODE = "gizmo_total"
Z_SOLAR = 0.02
N_METAL_ELEMENTS = 11

# -----------------------------------------------------------------------------
# BH 中心规则网格
# -----------------------------------------------------------------------------
# 规则网格范围为：[-90,+90]^3 kpc（相对于黑洞）
GRID_RADIUS_KPC = 90.0
GRID_DIMS = (256, 256, 256)

# False：若已经存在规则网格缓存，就直接重复使用。
# 你本次 snapshot_043 已经成功生成缓存，所以建议保持 False，
# 重新运行时不会再次进行约 1 分钟以上的 SPH 插值。
OVERWRITE_GRID = False

# -----------------------------------------------------------------------------
# 太阳位置与视线积分
# -----------------------------------------------------------------------------
# 注意：这是“相对于中心黑洞”的太阳位置，而不是模拟盒子原点。
SUN_OFFSET_KPC = np.array([-8.2, 0.0, 0.0], dtype=np.float64)

LOS_MIN_KPC = 0.20
LOS_MAX_KPC = 60.0
N_LOS = 256

# -----------------------------------------------------------------------------
# HEALPix 分辨率
# -----------------------------------------------------------------------------
NSIDE = 128

# 每次处理多少个 HEALPix 像素。
# 4096 像素 * 256 LOS = 1,048,576 个三维采样点。
# 如果内存不足可改为 2048 或 1024。
PIXEL_CHUNK = 4096

# -----------------------------------------------------------------------------
# 绘图参数
# -----------------------------------------------------------------------------
CMAP = "inferno"

# False：每张图根据百分位数自动设定颜色范围。
# True ：所有图统一使用 FIXED_VMIN / FIXED_VMAX。
USE_FIXED_LIMITS = False
FIXED_VMIN = 1.0e-10
FIXED_VMAX = 1.0e-6

PERCENTILE_VMIN = 1.0
PERCENTILE_VMAX = 99.5
DPI = 220

# 是否额外保存 HEALPix 原始数组 / FITS
SAVE_NPY = True
SAVE_FITS = False

# yt 日志级别
# 30 = WARNING，可以减少大量输出；需要调试时可改为 20。
yt.funcs.mylog.setLevel(30)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

KPC_TO_CM = 3.0856775814913673e21


# =============================================================================
# 1. 从 GIZMO HDF5 获取中心黑洞和时间
# =============================================================================

def get_bh_center_and_time(filename):
    """
    读取 PartType5 中第一个黑洞粒子的坐标，并读取 Header/Time。

    本项目严格使用中心黑洞作为银河系中心：
    如果快照中没有 PartType5，则直接报错，不再用恒星质心替代。
    """
    with h5py.File(filename, "r") as f:
        time = float(f["Header"].attrs.get("Time", np.nan))

        if "PartType5" not in f:
            raise RuntimeError(f"{filename} 中没有 PartType5，无法使用黑洞定心。")

        if "Coordinates" not in f["PartType5"]:
            raise RuntimeError(f"{filename} 中没有 PartType5/Coordinates。")

        if len(f["PartType5"]["Coordinates"]) < 1:
            raise RuntimeError(f"{filename} 的 PartType5 中没有黑洞粒子。")

        bh = np.asarray(f["PartType5"]["Coordinates"][0], dtype=np.float64)

    return bh, time


# =============================================================================
# 2. 构造适用于 yt APEC 的太阳金属丰度场
# =============================================================================

def add_total_metallicity_solar_field(ds):
    """
    增加 ('gas', 'total_metallicity_solar') 派生场，单位为 Z/Z_sun。

    默认标准 GIZMO 模式：
        Z/Z_sun = Metallicity_00 / 0.02

    这里另外将负值/非有限值处理为 0，避免 APEC 表插值时因为 log10
    收到非法金属丰度而产生不必要的 RuntimeWarning。
    """
    available = set(ds.field_list) | set(ds.derived_field_list)

    if METALLICITY_MODE == "gizmo_total":
        required = ("PartType0", "Metallicity_00")
        if required not in available:
            raise RuntimeError(f"缺少 GIZMO 金属丰度场：{required}")

        def _total_metallicity_solar(field, data):
            z = data[("PartType0", "Metallicity_00")] / Z_SOLAR
            # 保留 yt/unyt 单位对象，只对数值做清理
            return data.ds.arr(
                np.nan_to_num(np.asarray(z), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0),
                "dimensionless",
            )

    elif METALLICITY_MODE == "sum_11_custom":
        names = [
            ("PartType0", f"Metallicity_{i:02d}")
            for i in range(N_METAL_ELEMENTS)
        ]
        missing = [name for name in names if name not in available]

        if missing:
            raise RuntimeError(
                "缺少自定义金属丰度场："
                + ", ".join(str(x) for x in missing[:4])
                + (" ..." if len(missing) > 4 else "")
            )

        def _total_metallicity_solar(field, data):
            total = data[("PartType0", "Metallicity_00")].copy()
            for i in range(1, N_METAL_ELEMENTS):
                total += data[("PartType0", f"Metallicity_{i:02d}")]

            z = total / Z_SOLAR
            return data.ds.arr(
                np.nan_to_num(np.asarray(z), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0),
                "dimensionless",
            )

    else:
        raise ValueError(
            f"未知 METALLICITY_MODE={METALLICITY_MODE!r}；"
            "只能使用 'gizmo_total' 或 'sum_11_custom'。"
        )

    ds.add_field(
        ("gas", "total_metallicity_solar"),
        function=_total_metallicity_solar,
        units="dimensionless",
        sampling_type="local",
        force_override=True,
    )


# =============================================================================
# 3. 使用 yt/APEC 构造 0.6--1.0 keV X-ray 发射率场
# =============================================================================

def add_soft_xray_field(ds):
    """
    调用 yt.add_xray_emissivity_field 构造 APEC 软 X-ray 发射率场。

    返回真正的能量发射率字段键，例如：
        ('gas', 'xray_emissivity_0.6_1.0_keV')
    """
    kwargs = {
        "table_type": TABLE_TYPE,
        "metallicity": ("gas", "total_metallicity_solar"),
    }

    if APEC_DATA_DIR is not None:
        kwargs["data_dir"] = APEC_DATA_DIR

    fields = yt.add_xray_emissivity_field(
        ds,
        E_MIN_KEV,
        E_MAX_KEV,
        **kwargs,
    )

    # 不把完整字段名称写死，避免不同 yt 版本对浮点数字符串格式处理略有差异。
    for field in fields:
        if "xray_emissivity_" in field[1] and "photon" not in field[1]:
            return field

    raise RuntimeError(f"无法从 yt 返回字段中识别 X-ray 发射率：{fields}")


# =============================================================================
# 4. GIZMO/SPH -> BH 中心规则 X-ray 网格
# =============================================================================

def grid_cache_name(snapshot_file):
    """生成规则网格缓存文件名。"""
    stem = os.path.splitext(os.path.basename(snapshot_file))[0]
    n = GRID_DIMS[0]
    return os.path.join(
        GRID_DIR,
        f"{stem}_xray06_10_bhcenter_grid{n}.hdf5",
    )


def build_xray_grid(snapshot_file, grid_file):
    """
    读取原始 GIZMO 快照，用 yt 在 BH 周围建立规则网格，
    并保存 0.6--1.0 keV X-ray 发射率。

    这一步是整个程序中最耗时的一步，因此写成 HDF5 缓存。
    """
    bh, time = get_bh_center_and_time(snapshot_file)

    print(f"  黑洞中心 = {bh} kpc | Time = {time:.6f}")

    ds = yt.load(snapshot_file)
    ds.force_periodicity(False)

    add_total_metallicity_solar_field(ds)
    xray_field = add_soft_xray_field(ds)

    print(f"  yt X-ray 场：{xray_field}")

    center = ds.arr(bh, "kpc")
    radius = ds.quan(GRID_RADIUS_KPC, "kpc")

    left = center - radius
    right = center + radius

    print(
        f"  正在插值到 BH 中心规则网格 {GRID_DIMS}，"
        f"范围 +/-{GRID_RADIUS_KPC:.1f} kpc ..."
    )

    grid = ds.arbitrary_grid(left, right, GRID_DIMS)

    emiss = grid[xray_field].in_units("erg/s/cm**3")
    emiss_np = np.asarray(emiss.d, dtype=np.float32)

    # 对非有限值做最后一道清理，防止 NaN/Inf 进入后面的 LOS 积分。
    bad = ~np.isfinite(emiss_np)
    if np.any(bad):
        print(f"  [警告] X-ray 网格中发现 {bad.sum()} 个 NaN/Inf，已置为 0。")
        emiss_np[bad] = 0.0

    # 物理发射率不应为负；如果因为数值插值出现很小的负数，也置为 0。
    neg = emiss_np < 0.0
    if np.any(neg):
        print(f"  [警告] X-ray 网格中发现 {neg.sum()} 个负值，已置为 0。")
        emiss_np[neg] = 0.0

    with h5py.File(grid_file, "w") as f:
        f.create_dataset(
            "xray_emissivity",
            data=emiss_np,
            compression="gzip",
            compression_opts=1,
        )

        f.attrs["xray_units"] = "erg/s/cm**3"
        f.attrs["grid_radius_kpc"] = GRID_RADIUS_KPC
        f.attrs["bh_center_x_kpc"] = bh[0]
        f.attrs["bh_center_y_kpc"] = bh[1]
        f.attrs["bh_center_z_kpc"] = bh[2]
        f.attrs["time"] = time
        f.attrs["e_min_keV"] = E_MIN_KEV
        f.attrs["e_max_keV"] = E_MAX_KEV
        f.attrs["z_solar"] = Z_SOLAR
        f.attrs["metallicity_mode"] = METALLICITY_MODE

    print(f"  已保存规则网格缓存：{grid_file}")

    del emiss_np, emiss, grid, ds
    gc.collect()


# =============================================================================
# 5. 将缓存的 X-ray 立方体重新加载为 yt UniformGrid
# =============================================================================

def load_xray_grid(grid_file):
    """
    将缓存的 X-ray 三维数组加载为 yt UniformGrid。

    这里直接把 BH 放在规则网格坐标原点：
        x,y,z in [-GRID_RADIUS,+GRID_RADIUS] kpc
    因此后面的太阳坐标可以直接写成 (-8.2,0,0) kpc。
    """
    with h5py.File(grid_file, "r") as f:
        emiss = np.asarray(f["xray_emissivity"][...], dtype=np.float32)
        units = str(f.attrs["xray_units"])
        radius_kpc = float(f.attrs["grid_radius_kpc"])
        time = float(f.attrs.get("time", np.nan))

    bbox_cm = np.array(
        [
            [-radius_kpc, radius_kpc],
            [-radius_kpc, radius_kpc],
            [-radius_kpc, radius_kpc],
        ],
        dtype=np.float64,
    ) * KPC_TO_CM

    data = {
        ("gas", "xray_emissivity"): (emiss, units),
    }

    ds = yt.load_uniform_grid(
        data,
        emiss.shape,
        length_unit="cm",
        bbox=bbox_cm,
        geometry="cartesian",
        nprocs=1,
        periodicity=(False, False, False),
        unit_system="cgs",
    )

    ds.force_periodicity(False)

    # 立即检查字段是否真正注册成功，避免到了 LOS 阶段才发现字段不对。
    target = ("gas", "xray_emissivity")
    available = set(ds.field_list) | set(ds.derived_field_list)
    if target not in available:
        raise RuntimeError(
            f"规则网格加载后找不到字段 {target}。\n"
            f"当前字段示例：{list(ds.field_list)[:10]}"
        )

    return ds, time, radius_kpc


# =============================================================================
# 6. LOS 几何检查
# =============================================================================

def validate_geometry(radius_kpc):
    """检查太阳到 LOS 终点是否始终位于规则网格内部。"""
    needed_halfwidth = np.max(np.abs(SUN_OFFSET_KPC)) + LOS_MAX_KPC

    if needed_halfwidth > radius_kpc:
        raise ValueError(
            "LOS 会离开规则网格："
            f"至少需要半宽 {needed_halfwidth:.2f} kpc，"
            f"但当前网格半宽只有 {radius_kpc:.2f} kpc。"
        )


# =============================================================================
# 7. yt 多点采样兼容函数 —— 本次报错的核心修复
# =============================================================================

def sample_one_field_at_points(ds, field, points, expected_size):
    """
    使用 yt.find_field_values_at_points() 对一个场进行大量三维点采样。

    为什么要单独写这个函数：
    ---------------------------------
    不同 yt 版本 / 不同数据集 frontend 对“单字段”返回值存在形式差异：

    情况 A：返回 [YTArray(Npoints)]
    情况 B：直接返回 YTArray(Npoints)
    情况 C：返回 YTArray(1, Npoints)

    旧代码固定写成：
        ds.find_field_values_at_points([field], points)[0]

    如果恰好遇到情况 B，末尾 [0] 就会把整个 Npoints 数组错误地缩成
    一个标量，这正是本次：
        cannot reshape array of size 1 into shape (4096,256)
    的原因。

    本函数将三种返回形式统一整理为一个长度 expected_size 的一维 YTArray。
    """
    result = ds.find_field_values_at_points([field], points)

    # -------------------------------------------------------------------------
    # 第一步：如果 yt 返回的是 list / tuple，则取第一个字段。
    # -------------------------------------------------------------------------
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise RuntimeError(
                f"yt 返回了 {len(result)} 个字段，但本函数只请求了 1 个字段。"
            )
        sampled = result[0]
    else:
        # 单字段情况下，某些 yt/stream 组合可能直接返回 YTArray。
        sampled = result

    # -------------------------------------------------------------------------
    # 第二步：检查形状。
    # 先不调用 [0]，而是根据真实数组大小判断是否存在多余的字段维度。
    # -------------------------------------------------------------------------
    raw_shape = np.shape(sampled)
    raw_size = np.size(sampled)

    # 常见情况：(1, Npoints)，去掉最前面的“字段维度”。
    if raw_size == expected_size and len(raw_shape) >= 2 and raw_shape[0] == 1:
        sampled = sampled[0]
        raw_shape = np.shape(sampled)
        raw_size = np.size(sampled)

    # -------------------------------------------------------------------------
    # 第三步：必须严格等于请求的采样点数。
    # 如果不一致，给出清晰诊断信息。
    # -------------------------------------------------------------------------
    if raw_size != expected_size:
        raise RuntimeError(
            "yt 多点采样返回的数据量不正确：\n"
            f"  请求字段       = {field}\n"
            f"  采样点数量     = {expected_size}\n"
            f"  返回对象类型   = {type(result)}\n"
            f"  提取后对象类型 = {type(sampled)}\n"
            f"  返回 shape      = {np.shape(result)}\n"
            f"  提取后 shape    = {raw_shape}\n"
            f"  提取后 size     = {raw_size}\n"
            "请把以上诊断信息完整发给我。"
        )

    return sampled


# =============================================================================
# 8. HEALPix 全天球 + 从太阳出发的 X-ray LOS 积分
# =============================================================================

def make_allsky_intensity(ds_grid, radius_kpc):
    """
    从太阳位置沿所有 HEALPix 方向对 X-ray 发射率进行 LOS 积分。

    每个 HEALPix 像素：
        1. hp.pix2vec 给出单位方向矢量 n
        2. r(s) = r_sun + s*n
        3. 用 yt 在 r(s) 上读取三维 X-ray 发射率
        4. 对 epsilon_X 沿 s 积分
        5. 除以 4*pi 得到每立体角面亮度
    """
    validate_geometry(radius_kpc)

    npix = hp.nside2npix(NSIDE)
    pix = np.arange(npix, dtype=np.int64)

    # HEALPix 笛卡尔方向：
    # lon=0,b=0 -> (+x,0,0)
    # 由于太阳位于 x=-8.2 kpc，因此 +x 正好指向中心黑洞。
    vx, vy, vz = hp.pix2vec(NSIDE, pix, nest=False)
    dirs = np.column_stack((vx, vy, vz)).astype(np.float64)

    # LOS 采样距离
    los_kpc = np.linspace(
        LOS_MIN_KPC,
        LOS_MAX_KPC,
        N_LOS,
        dtype=np.float64,
    )
    los_cm = los_kpc * KPC_TO_CM

    intensity = np.empty(npix, dtype=np.float64)

    field = ("gas", "xray_emissivity")

    print(
        f"  HEALPix NSIDE={NSIDE}, Npix={npix}, "
        f"LOS={LOS_MIN_KPC:.2f}-{LOS_MAX_KPC:.2f} kpc, "
        f"N_LOS={N_LOS}"
    )

    # 分块处理，避免一次建立全部 196608*256 个 LOS 点造成内存压力。
    for i0 in range(0, npix, PIXEL_CHUNK):
        i1 = min(i0 + PIXEL_CHUNK, npix)

        d = dirs[i0:i1]
        n_pix_chunk = i1 - i0

        # ---------------------------------------------------------------------
        # 构造当前这一批像素的全部 LOS 三维坐标。
        # shape = (n_pix_chunk, N_LOS, 3)
        # 坐标均为“相对于 BH”的 kpc。
        # ---------------------------------------------------------------------
        points_kpc = (
            SUN_OFFSET_KPC[None, None, :]
            + d[:, None, :] * los_kpc[None, :, None]
        )

        flat_kpc = points_kpc.reshape(-1, 3)
        expected_size = n_pix_chunk * N_LOS

        # 将坐标交给 yt。
        # 直接保留物理单位 kpc 即可，yt 会根据 UniformGrid 的单位系统转换。
        points_yt = ds_grid.arr(flat_kpc, "kpc")

        # ---------------------------------------------------------------------
        # 核心修复：不要再无条件在 find_field_values_at_points() 后写 [0]。
        # ---------------------------------------------------------------------
        sampled = sample_one_field_at_points(
            ds_grid,
            field,
            points_yt,
            expected_size,
        )

        # 转换到 X-ray 体发射率单位，并严格整理为
        # (当前像素数, N_LOS)。
        if hasattr(sampled, "in_units"):
            sampled_cgs = sampled.in_units("erg/s/cm**3")
            eps_flat = np.asarray(sampled_cgs.d, dtype=np.float64).reshape(-1)
        else:
            # 理论上 yt 应返回带单位数组；此分支用于增加兼容性。
            eps_flat = np.asarray(sampled, dtype=np.float64).reshape(-1)

        if eps_flat.size != expected_size:
            raise RuntimeError(
                f"单位转换后采样点数异常：得到 {eps_flat.size}，"
                f"预期 {expected_size}。"
            )

        eps = eps_flat.reshape(n_pix_chunk, N_LOS)

        # NaN/Inf/负发射率不参与积分。
        eps[~np.isfinite(eps)] = 0.0
        eps[eps < 0.0] = 0.0

        # ---------------------------------------------------------------------
        # 光学薄各向同性辐射：
        # I = integral(epsilon ds) / (4*pi)
        # ---------------------------------------------------------------------
        # NumPy 2.x 推荐 trapezoid；旧版本则回退到 trapz。
        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(eps, x=los_cm, axis=1)
        else:
            integral = np.trapz(eps, x=los_cm, axis=1)

        intensity[i0:i1] = integral / (4.0 * np.pi)

        del points_kpc, flat_kpc, points_yt, sampled, eps_flat, eps, integral

        print(
            f"    已完成像素 {i0:6d}-{i1 - 1:6d} / {npix - 1}"
        )

    return intensity


# =============================================================================
# 9. 自动确定对数色标范围
# =============================================================================

def robust_log_limits(hpxmap):
    """从正的有限值中确定 Mollweide 图的对数颜色范围。"""
    good = hpxmap[np.isfinite(hpxmap) & (hpxmap > 0.0)]

    if good.size == 0:
        raise RuntimeError("全天球 X-ray 强度全部为 0 或非有限值。")

    if USE_FIXED_LIMITS:
        return FIXED_VMIN, FIXED_VMAX

    vmin = np.percentile(good, PERCENTILE_VMIN)
    vmax = np.percentile(good, PERCENTILE_VMAX)

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = good.min()

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = good.max()

    return vmin, vmax


# =============================================================================
# 10. Mollweide 全天球绘图
# =============================================================================

def plot_mollweide(hpxmap, time, output_png):
    """将 HEALPix X-ray 面亮度绘制成 Mollweide 全天球图。"""
    vmin, vmax = robust_log_limits(hpxmap)

    plotmap = hpxmap.copy()

    # 对数色标无法显示 <=0 的数值，因此标记为 HEALPix UNSEEN。
    plotmap[~np.isfinite(plotmap) | (plotmap <= 0.0)] = hp.UNSEEN

    fig = plt.figure(figsize=(11.5, 6.6))

    hp.mollview(
        plotmap,
        fig=fig.number,
        nest=False,
        coord=None,
        flip="astro",
        rot=(0.0, 0.0, 0.0),
        cmap=CMAP,
        norm="log",
        min=vmin,
        max=vmax,
        cbar=True,
        unit=(
            r"$I_{0.6-1.0\,\mathrm{keV}}$ "
            r"$[\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}}]$"
        ),
        title=(
            rf"eROSITA-like soft X-ray 0.6--1.0 keV | "
            rf"Sun=(-8.2,0,0) kpc | Time={time:.3f} Gyr"
        ),
        xsize=1800,
    )

    # 经纬网
    hp.graticule(dpar=30, dmer=30, alpha=0.35)

    # 标记银河系中心方向。
    # 因为太阳位于 -x，所以 l=0 正好是 +x，即中心黑洞方向。
    hp.projtext(
        0.0,
        0.0,
        "GC  l=0°",
        lonlat=True,
        color="white",
        fontsize=9,
        ha="center",
    )

    hp.projtext(
        90.0,
        0.0,
        "l=90°",
        lonlat=True,
        color="white",
        fontsize=8,
    )

    hp.projtext(
        -90.0,
        0.0,
        "l=270°",
        lonlat=True,
        color="white",
        fontsize=8,
    )

    plt.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  已保存全天球图：{output_png}")
    print(
        "  绘图颜色范围："
        f"{vmin:.4e} -- {vmax:.4e} "
        "erg s^-1 cm^-2 sr^-1"
    )


# =============================================================================
# 11. 单个快照处理
# =============================================================================

def process_snapshot(snapshot_file):
    """完成一个 snapshot 的规则网格、LOS 积分以及全天球绘图。"""
    stem = os.path.splitext(os.path.basename(snapshot_file))[0]
    grid_file = grid_cache_name(snapshot_file)

    print("\n" + "=" * 78)
    print(f"正在处理：{snapshot_file}")

    # -------------------------------------------------------------------------
    # 第一步：建立或重复使用 BH-centered X-ray 规则网格。
    # -------------------------------------------------------------------------
    if OVERWRITE_GRID or not os.path.exists(grid_file):
        build_xray_grid(snapshot_file, grid_file)
    else:
        print(f"  直接重复使用已有规则网格缓存：{grid_file}")

    # -------------------------------------------------------------------------
    # 第二步：载入规则网格并做太阳视角 LOS 积分。
    # -------------------------------------------------------------------------
    ds_grid, time, radius_kpc = load_xray_grid(grid_file)

    print(
        f"  yt 规则网格加载完成：domain_dimensions={ds_grid.domain_dimensions}, "
        f"yt={yt.__version__}"
    )

    hpxmap = make_allsky_intensity(ds_grid, radius_kpc)

    # -------------------------------------------------------------------------
    # 第三步：保存 HEALPix 数组。
    # -------------------------------------------------------------------------
    if SAVE_NPY:
        npy_file = os.path.join(
            OUTPUT_DIR,
            f"{stem}_xray06_10_allsky.npy",
        )
        np.save(npy_file, hpxmap)
        print(f"  已保存 HEALPix 数组：{npy_file}")

    # -------------------------------------------------------------------------
    # 第四步：可选 FITS 输出。
    # -------------------------------------------------------------------------
    if SAVE_FITS:
        fits_file = os.path.join(
            OUTPUT_DIR,
            f"{stem}_xray06_10_allsky.fits",
        )

        hp.write_map(
            fits_file,
            hpxmap,
            overwrite=True,
            dtype=np.float64,
            column_names=["I_XRAY"],
            column_units=["erg s^-1 cm^-2 sr^-1"],
        )

        print(f"  已保存 HEALPix FITS：{fits_file}")

    # -------------------------------------------------------------------------
    # 第五步：绘制 Mollweide 全天球图。
    # -------------------------------------------------------------------------
    png_file = os.path.join(
        OUTPUT_DIR,
        f"{stem}_xray06_10_allsky.png",
    )

    plot_mollweide(hpxmap, time, png_file)

    del hpxmap, ds_grid
    gc.collect()


# =============================================================================
# 12. 主程序
# =============================================================================

def main():
    snapshots = sorted(
        glob.glob(os.path.join(SNAPSHOT_DIR, SNAPSHOT_GLOB))
    )

    if not snapshots:
        raise FileNotFoundError(
            "没有找到匹配的 GIZMO HDF5 快照："
            f"{os.path.join(SNAPSHOT_DIR, SNAPSHOT_GLOB)}"
        )

    print(f"发现 {len(snapshots)} 个快照。")
    print(f"全天球输出目录：{OUTPUT_DIR}")
    print(f"规则网格缓存目录：{GRID_DIR}")
    print(f"当前 yt 版本：{yt.__version__}")

    for snapshot_file in snapshots:
        try:
            process_snapshot(snapshot_file)
        except Exception as exc:
            print(f"[错误] {snapshot_file}: {exc}")
            raise

    print("\n全部指定快照的软 X-ray 全天球图处理完成。")


if __name__ == "__main__":
    main()
