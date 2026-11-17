#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mou (2023) live MW + hot-CGM 的无 ISM 诊断 IC
================================================

用途
----
该脚本用于本项目的单变量诊断实验：

    Mou-like MW（全粒子化、默认不含 live 冷暖 ISM）
        + MFM 热 CGM
        -> 检查 CGM 冷却内流与 AGN 喷流能否形成气泡

默认 ``--include-ism 0``。如需复现 v4 的 McMillan (2017) 全局亚网格
ISM，可显式使用 ``--include-ism 1``。无 ISM 模式不会简单删除盘质量：
Mou 的完整 M_disc=1e11 Msun 全部由无碰撞 MN 恒星盘承载，因此总盘
引力目标不变。

设计原则
--------
1. 尽量不修改 GIZMO 源码：MW 的引力全部由粒子自身产生。
2. Mou 的 M_disc=1e11 Msun 被解释为总盘引力质量目标。默认无 ISM 时，
   该质量全部赋给 live 老恒星盘；启用 ISM 时，再从老恒星盘中扣除
   实际生成的气体盘质量，避免双重计数。
3. 无 ISM 是为了隔离“气体盘阻挡喷流”与“CGM 冷却内流”两种效应，
   不代表银河系物理上没有冷暖气体。
4. DM、恒星盘、核球的空间密度都直接对应 Mou et al. (2023) Eq.(4)
   中的解析势，而不是继续使用以前的“指数盘近似 MN 势”方案。
5. 热 CGM 在外部继续使用 Mou 的等温 HSE：
       T = 2e6 K
       rho0 = 8.5 m_H cm^-3
       f_rot = 0.0（Mou 主模型的非旋转初始晕）
   为避免 live+cooling 条件下中央 8.5 m_H cm^-3 热气体快速过冷，默认在
   与 R=0.5 kpc 中面等势面相接的位置构造严格静力的常密度熵核：外部
   Mou 密度完全不变，核内通过提高温度而非恒星反馈支撑压力。
6. 初始速度尽量采用 Jeans / epicyclic / hydrostatic 平衡近似，然后必须在 GIZMO 中
   做 isolated relaxation，再选择通过检验的 snapshot 接入 LMC/SMC。
7. 初始老恒星盘与核球统一存入 PartType4，并写入形成时间；新生恒星
   仍由 GIZMO 写入 PartType4，常规可视化只需读取一种恒星类型。
8. 预留 LMC/SMC ParticleID 区间，避免后续插入冲突。

必须明确的物理事实
------------------
Mou et al. (2023) 原始大尺度模拟使用“固定解析银河势”，并没有 live DM
或 live stellar particles。因此这里的“全粒子化”是对 Mou 解析质量模型的
等价 live realization，而不是 Mou 原文已有的 N-body IC。

Mou 的 logarithmic halo 总质量发散，因此任何有限粒子实现都必须截断。
默认将 DM 截断到 300 kpc。在连续、球对称极限下，截断外部球壳不改变
截断半径以内的径向引力；但有限 N 噪声、截断边界和 Jeans 速度近似
仍会使 live halo 演化，必须用实际 GIZMO 弛豫结果验证。

Miyamoto-Nagai 盘也具有无限尾部。本脚本默认采样到：
    R <= 300 kpc, |z| <= 100 kpc
并保留解析密度的真实归一化，不把被截断的尾部质量重新塞回内部。
默认截断后保留约 98% 的 MN 盘质量。这个质量分数不能直接转换为
内区引力误差；后者必须从 live realization 的实际力场测量。

默认 GIZMO 粒子类型
------------------
PartType0 : 默认仅热 CGM（MFM）；可选 McMillan 全局亚网格 ISM
PartType1 : MW DM halo（collisionless）
PartType4 : MW 老恒星盘 + 老核球；运行中新生恒星也进入该类型
PartType5 : 中心 BH（默认 4e6 Msun，仅为后续 AGN 预留；弛豫期反馈关闭）

建议运行阶段
------------
弛豫期：
    - HYDRO_MESHLESS_FINITE_MASS
    - COOLING + METALS
    - 为严格单变量比较，首次无 ISM 测试保持与 v4 相同的运行时编译和参数
    - BH accretion/feedback = 0

生成：
    python make_ic_mou_hot_cgm_noism_v5.py

输出：
    mw_mou_core_noism_ic.hdf5
"""

import argparse
import atexit
import os
import tempfile
import numpy as np

try:
    import h5py
except ImportError:  # permits numerical --dry-run checks on minimal systems
    h5py = None


# =============================================================================
# 0. 常数与 GIZMO 单位
# =============================================================================

SEED_DEFAULT = 42

UNIT_LENGTH_CM = 3.0856775814913673e21   # 1 kpc
UNIT_MASS_G = 1.989e43                   # 1e10 Msun
UNIT_VELOCITY_CMS = 1.0e5                # 1 km/s
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
UNIT_TIME_GYR = UNIT_LENGTH_CM / UNIT_VELOCITY_CMS / SECONDS_PER_GYR

MSUN_G = 1.989e33
K_B_CGS = 1.380649e-16
M_P_CGS = 1.67262192369e-24

# kpc (km/s)^2 / (1e10 Msun)
G_CODE = 43007.1

GAMMA = 5.0 / 3.0
MU_HOT = 0.61
MU_ISM = 1.27


# =============================================================================
# 1. Mou et al. (2023) 解析银河模型
# =============================================================================

# logarithmic halo
MOU_V_HALO = 131.5       # km/s
MOU_D_HALO = 12.0        # kpc

# Miyamoto-Nagai total disk target (old stars + global ISM)
MOU_M_DISC = 10.0        # 1e10 Msun = 1e11 Msun
MOU_A_DISC = 6.5         # kpc
MOU_B_DISC = 0.26        # kpc

# Hernquist bulge
MOU_M_BULGE = 3.4        # 1e10 Msun = 3.4e10 Msun
MOU_D_BULGE = 0.7        # kpc

# 热 CGM
MOU_T_CGM = 2.0e6        # K
# Mou (2023) 主模型的初始晕不旋转；论文另外考察 f=0.3/0.5/0.7，
# 且对每个旋转算例重新调整 rho0 以匹配 X-ray surface brightness。
MOU_F_ROT = 0.00
MOU_RHO0_MH_CM3 = 8.5
MOU_CGM_Z_ZSUN = 0.5
ISM_Z_ZSUN = 1.0
Z_SUN_MASS_FRACTION = 0.02

# McMillan (2017) axisymmetric gas disks. Sigma0 is in Msun/pc^2;
# the published normalizations already adopt the model's gas-mass convention.
MCM_HI_SIGMA0 = 53.1
MCM_HI_RD = 7.0
MCM_HI_RM = 4.0
MCM_HI_ZD = 0.085

MCM_H2_SIGMA0 = 2180.0
MCM_H2_RD = 1.5
MCM_H2_RM = 12.0
MCM_H2_ZD = 0.045

ISM_T_INITIAL = 1.0e4
OLD_STAR_AGE_GYR = 10.0
CGM_CORE_RADIUS_DEFAULT = 0.5

# 这些是本 live-particle realization 的数值选择，不是 Mou (2023) 参数。
Q_TARGET_DEFAULT = 1.5
HDF5_WRITE_CHUNK_DEFAULT = 262_144


# =============================================================================
# 2. 默认粒子分辨率与截断
# =============================================================================

# DM halo 截断到 300 kpc。
RMAX_DM_DEFAULT = 300.0

# MN 恒星盘的数值采样窗口。该盘解析上无限延伸。
RMAX_DISC_DEFAULT = 300.0
ZMAX_DISC_DEFAULT = 100.0

# Hernquist 核球长尾截断到 300 kpc，遗漏质量 < 0.5%。
RMAX_BULGE_DEFAULT = 300.0

# CGM 延伸到 200 kpc，满足后续 LMC/SMC 核心实验需要。
RMAX_CGM_DEFAULT = 200.0

# McMillan HI+H2 disk is truncated at 30 kpc.  Its observed vertical scale
# heights are recorded as provenance, but the live gas is initialized from
# hydrostatic equilibrium at 1e4 K in the Mou total potential.
RMAX_ISM_DEFAULT = 30.0
ZMAX_ISM_DEFAULT = 5.0

# BoxSize 保持与后续 LMC/SMC 流程兼容。
BOXSIZE_DEFAULT = 1000.0

# 目标质量分辨率：
# DM ~ 1e7 Msun，stellar/bulge ~ 1e6 Msun，CGM ~ 1e5 Msun。
TARGET_DM_MASS_MSUN = 1.0e7
TARGET_STAR_MASS_MSUN = 1.0e6
TARGET_CGM_MASS_MSUN = 1.0e5
TARGET_ISM_MASS_MSUN = 1.0e5

M_BH_MSUN_DEFAULT = 4.0e6


# =============================================================================
# 3. ParticleID 规划
# =============================================================================

# 各 MW 组分使用互不重叠的大区间。
MW_CGM_ID_START = np.uint64(1)
MW_DM_ID_START = np.uint64(10_000_000)
MW_DISC_ID_START = np.uint64(20_000_000)
MW_BULGE_ID_START = np.uint64(30_000_000)
MW_BH_ID = np.uint64(90_000_000)

# 后续 LMC/SMC 的完整 ID 区间。
LMC_ID_START = np.uint64(100_000_000)
LMC_ID_END = np.uint64(199_999_999)

SMC_ID_START = np.uint64(200_000_000)
SMC_ID_END = np.uint64(299_999_999)

FUTURE_ID_START = np.uint64(300_000_000)


# =============================================================================
# 4. 金属丰度
# =============================================================================

SOLAR_METAL_MASS = np.array(
    [
        0.00326,   # C
        0.00132,   # N
        0.00865,   # O
        0.00222,   # Ne
        0.000931,  # Mg
        0.00108,   # Si
        0.000644,  # S
        0.000101,  # Ca
        0.00173,   # Fe
    ],
    dtype=np.float64,
)
HYDROGEN_MASS_FRACTION_PRIMORDIAL = 0.76
HE_MASS_FRACTION_SOLAR = 0.28


def gizmo_metallicity_vector(z_zsun=MOU_CGM_Z_ZSUN):
    """返回标准 GIZMO/FIRE 11 维金属质量分数向量。"""
    z_zsun = float(z_zsun)
    if not np.isfinite(z_zsun) or z_zsun < 0.0:
        raise ValueError("z_zsun 必须是有限非负数。")
    z_total = z_zsun * Z_SUN_MASS_FRACTION
    out = np.zeros(11, dtype=np.float64)
    out[0] = z_total
    primordial_he = 1.0 - HYDROGEN_MASS_FRACTION_PRIMORDIAL
    out[1] = primordial_he + (
        HE_MASS_FRACTION_SOLAR - primordial_he
    ) * z_zsun
    # 与 GIZMO public init.c 中的逐元素 solar abundance 缩放一致。
    # 九种显式金属之和与 Z_total 的微小差额来自 abundance table 本身。
    out[2:] = z_zsun * SOLAR_METAL_MASS
    if out[0] + out[1] >= 1.0:
        raise ValueError("z_zsun 使 Z_total + Y_He >= 1，超出质量分数定义域。")
    return out


def fill_gizmo_metallicity(n, z_zsun=MOU_CGM_Z_ZSUN):
    """兼容旧调用：构造 N x 11 金属质量分数数组。"""
    row = gizmo_metallicity_vector(z_zsun)
    return np.broadcast_to(row, (int(n), row.size)).copy()


# =============================================================================
# 5. Mou 势、加速度与圆周速度
# =============================================================================

def mou_potential_cyl(R, z):
    """Mou Eq.(4) 轴对称解析势，单位 (km/s)^2。"""
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    r = np.sqrt(R * R + z * z)

    phi_halo = MOU_V_HALO**2 * np.log(r * r + MOU_D_HALO**2)

    xi = np.sqrt(z * z + MOU_B_DISC**2)
    phi_disc = -G_CODE * MOU_M_DISC / np.sqrt(
        R * R + (MOU_A_DISC + xi) ** 2
    )

    phi_bulge = -G_CODE * MOU_M_BULGE / (r + MOU_D_BULGE)

    return phi_halo + phi_disc + phi_bulge


def mou_accel_cyl(R, z):
    """
    返回 Mou 势的 (a_R, a_z)，单位 (km/s)^2/kpc。
    对 R>0、z>0，向中心/盘面的加速度为负。
    """
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    r2 = R * R + z * z
    r = np.sqrt(r2)
    rs = np.maximum(r, 1.0e-12)

    # logarithmic halo
    dphi_dR_h = (
        2.0 * MOU_V_HALO**2 * R
        / (r2 + MOU_D_HALO**2)
    )
    dphi_dz_h = (
        2.0 * MOU_V_HALO**2 * z
        / (r2 + MOU_D_HALO**2)
    )

    # Miyamoto-Nagai disk
    xi = np.sqrt(z * z + MOU_B_DISC**2)
    A = MOU_A_DISC + xi
    D3 = np.maximum(
        (R * R + A * A) ** 1.5,
        1.0e-300,
    )

    dphi_dR_d = G_CODE * MOU_M_DISC * R / D3
    dphi_dz_d = (
        G_CODE * MOU_M_DISC * A * z
        / np.maximum(xi * D3, 1.0e-300)
    )

    # Hernquist bulge
    dphi_dr_b = (
        G_CODE * MOU_M_BULGE
        / np.maximum((r + MOU_D_BULGE) ** 2, 1.0e-300)
    )
    dphi_dR_b = dphi_dr_b * R / rs
    dphi_dz_b = dphi_dr_b * z / rs

    return (
        -(dphi_dR_h + dphi_dR_d + dphi_dR_b),
        -(dphi_dz_h + dphi_dz_d + dphi_dz_b),
    )


def v_circ_mou(R):
    """Mou 势在 z=0 的圆周速度，单位 km/s。"""
    R = np.asarray(R, dtype=np.float64)
    R_safe = np.maximum(R, 1.0e-10)
    R2 = R_safe * R_safe

    vc2_h = (
        2.0 * MOU_V_HALO**2 * R2
        / (R2 + MOU_D_HALO**2)
    )

    B = MOU_A_DISC + MOU_B_DISC
    vc2_d = (
        G_CODE * MOU_M_DISC * R2
        / np.maximum((R2 + B * B) ** 1.5, 1.0e-300)
    )

    vc2_b = (
        G_CODE * MOU_M_BULGE * R_safe
        / np.maximum((R_safe + MOU_D_BULGE) ** 2, 1.0e-300)
    )

    return np.sqrt(
        np.maximum(vc2_h + vc2_d + vc2_b, 0.0)
    )


def epicyclic_kappa_mou(R):
    """由 Mou rotation curve 数值计算 epicyclic frequency。"""
    R = np.asarray(R, dtype=np.float64)
    Rs = np.maximum(R, 1.0e-3)

    eps = 1.0e-3
    Rp = Rs * (1.0 + eps)
    Rm = np.maximum(Rs * (1.0 - eps), 1.0e-4)

    vc2 = v_circ_mou(Rs) ** 2
    vc2p = v_circ_mou(Rp) ** 2
    vc2m = v_circ_mou(Rm) ** 2

    dvc2_dR = (
        (vc2p - vc2m)
        / np.maximum(Rp - Rm, 1.0e-12)
    )

    kappa2 = (
        dvc2_dR / Rs
        + 2.0 * vc2 / (Rs * Rs)
    )

    return np.sqrt(np.maximum(kappa2, 0.0))


# =============================================================================
# 6. Mou logarithmic halo 的粒子化
# =============================================================================

def mou_halo_density_shape(r):
    """
    Mou logarithmic halo 对应的密度形状（省略常数）：

        rho ~ (r^2 + 3 d^2) / (r^2 + d^2)^2
    """
    r = np.asarray(r, dtype=np.float64)
    d2 = MOU_D_HALO**2
    return (
        (r * r + 3.0 * d2)
        / np.maximum((r * r + d2) ** 2, 1.0e-300)
    )


def mou_halo_mass_msun(r):
    """
    Mou logarithmic halo 在半径 r 内对应的解析质量：

        M(<r) = 2 v_h^2 r^3 / [G (r^2+d_h^2)]

    这里 G 使用 Msun 单位。
    """
    r = np.asarray(r, dtype=np.float64)
    G_msun = 4.30071e-6
    return (
        2.0 * MOU_V_HALO**2 * r**3
        / (
            G_msun
            * np.maximum(
                r * r + MOU_D_HALO**2,
                1.0e-300,
            )
        )
    )


def sample_mou_halo_positions(rng, n, rmax):
    """按 Mou logarithmic halo 的精确累计质量数值反演采样。"""
    rr = np.concatenate(
        (
            np.array([0.0]),
            np.geomspace(1.0e-4, float(rmax), 200000),
        )
    )
    mm = mou_halo_mass_msun(rr)
    cdf = mm / mm[-1]

    r = np.interp(
        rng.random(int(n)),
        cdf,
        rr,
    )

    mu = 2.0 * rng.random(int(n)) - 1.0
    phi = 2.0 * np.pi * rng.random(int(n))
    sin_theta = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))

    pos = np.empty((int(n), 3), dtype=np.float64)
    pos[:, 0] = r * sin_theta * np.cos(phi)
    pos[:, 1] = r * sin_theta * np.sin(phi)
    pos[:, 2] = r * mu

    return pos, r


# =============================================================================
# 7. Hernquist bulge
# =============================================================================

def sample_truncated_hernquist(rng, n, a, rmax):
    """按 Hernquist 累计质量精确采样，并截断长尾。"""
    fmax = (rmax / (rmax + a)) ** 2
    F = rng.random(int(n)) * fmax
    s = np.sqrt(F)

    r = a * s / np.maximum(1.0 - s, 1.0e-14)

    mu = 2.0 * rng.random(int(n)) - 1.0
    phi = 2.0 * np.pi * rng.random(int(n))
    sin_theta = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))

    pos = np.empty((int(n), 3), dtype=np.float64)
    pos[:, 0] = r * sin_theta * np.cos(phi)
    pos[:, 1] = r * sin_theta * np.sin(phi)
    pos[:, 2] = r * mu

    return pos, r, fmax


def hernquist_density_shape(r):
    """Hernquist 密度形状，省略常数。"""
    r = np.asarray(r, dtype=np.float64)
    return 1.0 / np.maximum(
        r * (r + MOU_D_BULGE) ** 3,
        1.0e-300,
    )


# =============================================================================
# 8. Miyamoto-Nagai stellar disk 的精确密度与数值采样
# =============================================================================

def mn_density(R, z, M=MOU_M_DISC, a=MOU_A_DISC, b=MOU_B_DISC):
    """
    与 Miyamoto-Nagai 势严格对应的三维质量密度。

    返回单位：code mass / kpc^3。
    """
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    zz = np.sqrt(z * z + b * b)

    numerator = (
        a * R * R
        + (a + 3.0 * zz) * (a + zz) ** 2
    )

    denominator = (
        np.maximum(
            R * R + (a + zz) ** 2,
            1.0e-300,
        ) ** 2.5
        * np.maximum(zz, 1.0e-300) ** 3
    )

    return (
        b * b * M
        / (4.0 * np.pi)
        * numerator
        / denominator
    )


def _trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def build_mn_sampling_table(rmax, zmax, M, nr=1201, nz=1001):
    """
    构造 MN density 的数值 CDF。

    注意：
    这不是把恒星放在网格点上。网格只用于积分/CDF，
    输出粒子仍然是连续随机位置。
    """
    sR = np.linspace(0.0, 1.0, int(nr))
    sz = np.linspace(0.0, 1.0, int(nz))

    # 近中心加密，同时保留很长的解析尾部。
    R_grid = float(rmax) * sR**1.7
    z_grid = float(zmax) * sz**2.3

    RR = R_grid[:, None]
    zz = z_grid[None, :]

    M = float(M)
    if not np.isfinite(M) or M <= 0.0:
        raise ValueError("MN live stellar mass normalization must be positive.")

    rho = mn_density(RR, zz, M=M)

    # Sigma(R)，只积分 z>=0 后乘 2。
    sigma = 2.0 * _trapz(rho, z_grid, axis=1)

    pR = 2.0 * np.pi * R_grid * sigma

    cdf_R = np.zeros_like(R_grid)
    dR = np.diff(R_grid)
    cdf_R[1:] = np.cumsum(
        0.5 * (pR[1:] + pR[:-1]) * dR
    )

    retained_mass_code = float(cdf_R[-1])
    retained_fraction = retained_mass_code / M

    if retained_mass_code <= 0.0:
        raise RuntimeError("MN 盘积分质量为零。")

    cdf_R /= retained_mass_code

    # 每个 R 下的 |z| 条件 CDF。
    cdf_z = np.zeros_like(rho)
    dz = np.diff(z_grid)

    cdf_z[:, 1:] = np.cumsum(
        0.5 * (rho[:, 1:] + rho[:, :-1])
        * dz[None, :],
        axis=1,
    )

    norm = cdf_z[:, -1]
    good = norm > 0.0
    cdf_z[good] /= norm[good, None]

    return {
        "R": R_grid,
        "z": z_grid,
        "rho": rho,
        "Sigma": sigma,
        "cdf_R": cdf_R,
        "cdf_z": cdf_z,
        "retained_mass_code": retained_mass_code,
        "retained_fraction": retained_fraction,
        "analytic_mass_code": M,
    }


def sample_mn_positions(rng, n, table):
    """从 MN density 的连续 CDF 中采样位置。"""
    R_grid = table["R"]
    z_grid = table["z"]
    cdf_R = table["cdf_R"]
    cdf_z = table["cdf_z"]

    n = int(n)

    R = np.interp(
        rng.random(n),
        cdf_R,
        R_grid,
    )

    # 用最近的 R 表格行采样条件 z 分布。
    idx = np.searchsorted(R_grid, R)
    idx = np.clip(idx, 1, len(R_grid) - 1)

    left = idx - 1
    choose_right = (
        np.abs(R_grid[idx] - R)
        < np.abs(R - R_grid[left])
    )
    row = np.where(choose_right, idx, left)

    zabs = np.empty(n, dtype=np.float64)
    uz = rng.random(n)

    for j in np.unique(row):
        sel = row == j
        zabs[sel] = np.interp(
            uz[sel],
            cdf_z[j],
            z_grid,
        )

    sign = np.where(
        rng.random(n) < 0.5,
        -1.0,
        1.0,
    )
    z = sign * zabs

    phi = 2.0 * np.pi * rng.random(n)

    pos = np.empty((n, 3), dtype=np.float64)
    pos[:, 0] = R * np.cos(phi)
    pos[:, 1] = R * np.sin(phi)
    pos[:, 2] = z

    return pos, R, z


# =============================================================================
# 8b. McMillan (2017) global ISM：观测表面密度 + 静力垂直结构
# =============================================================================

def thermal_scale_kms2(T, mu):
    """Return k_B T/(mu m_p) in (km/s)^2."""
    return K_B_CGS * float(T) / (float(mu) * M_P_CGS) / 1.0e10


def specific_internal_energy(T, mu):
    """Ideal-gas specific internal energy in GIZMO velocity-squared units."""
    return thermal_scale_kms2(T, mu) / (GAMMA - 1.0)


def _mcmillan_sigma_component(R, sigma0, rd, rm):
    """McMillan gas surface density in Msun/pc^2."""
    R = np.asarray(R, dtype=np.float64)
    out = np.zeros_like(R)
    good = R > 0.0
    out[good] = float(sigma0) * np.exp(
        -float(rm) / R[good] - R[good] / float(rd)
    )
    return out


def mcmillan_ism_surface_density_msun_pc2(R):
    """McMillan (2017) HI+H2 surface density."""
    return (
        _mcmillan_sigma_component(
            R, MCM_HI_SIGMA0, MCM_HI_RD, MCM_HI_RM
        )
        + _mcmillan_sigma_component(
            R, MCM_H2_SIGMA0, MCM_H2_RD, MCM_H2_RM
        )
    )


def build_ism_sampling_table(rmax, zmax, nr=1601, nz=801):
    """
    Build a continuous McMillan-ISM sampler.

    Sigma(R) is fixed by McMillan (2017). At every R, the vertical density
    is the 1e4-K isothermal HSE solution in the total Mou target potential:

        rho(R,z) propto exp[-(Phi(R,z)-Phi(R,0))/c_s^2].

    The table is only an integration/CDF device; particle positions remain
    continuous random variables.
    """
    sR = np.linspace(0.0, 1.0, int(nr))
    sz = np.linspace(0.0, 1.0, int(nz))
    R_grid = float(rmax) * sR**1.35
    z_grid = float(zmax) * sz**2.2

    sigma_msun_pc2 = mcmillan_ism_surface_density_msun_pc2(R_grid)
    # Msun/pc^2 -> (1e10 Msun)/kpc^2.
    sigma_code = sigma_msun_pc2 * 1.0e6 / 1.0e10

    RR = R_grid[:, None]
    zz = z_grid[None, :]
    phi_mid = mou_potential_cyl(R_grid, np.zeros_like(R_grid))[:, None]
    dphi = mou_potential_cyl(RR, zz) - phi_mid
    cs2 = thermal_scale_kms2(ISM_T_INITIAL, MU_ISM)
    vertical_shape = np.exp(np.clip(-dphi / cs2, -745.0, 0.0))

    vertical_norm = 2.0 * _trapz(vertical_shape, z_grid, axis=1)
    if np.any(~np.isfinite(vertical_norm)) or np.any(vertical_norm <= 0.0):
        raise RuntimeError("ISM vertical HSE normalization is invalid.")
    rho = sigma_code[:, None] * vertical_shape / vertical_norm[:, None]

    radial_weight = 2.0 * np.pi * R_grid * sigma_code
    cdf_R = np.zeros_like(R_grid)
    dR = np.diff(R_grid)
    cdf_R[1:] = np.cumsum(
        0.5 * (radial_weight[1:] + radial_weight[:-1]) * dR
    )
    retained_mass_code = float(cdf_R[-1])
    if retained_mass_code <= 0.0:
        raise RuntimeError("Integrated McMillan ISM mass is zero.")
    cdf_R /= retained_mass_code

    cdf_z = np.zeros_like(vertical_shape)
    dz = np.diff(z_grid)
    cdf_z[:, 1:] = np.cumsum(
        0.5 * (vertical_shape[:, 1:] + vertical_shape[:, :-1])
        * dz[None, :],
        axis=1,
    )
    cdf_z /= cdf_z[:, -1][:, None]

    rho_mid = np.maximum(rho[:, 0], 1.0e-300)
    R_safe = np.maximum(R_grid, 1.0e-3)
    dlnrho_dlnR = np.gradient(
        np.log(rho_mid), np.log(R_safe), edge_order=2
    )
    # The exact central hole has negligible mass. Bounding this derivative
    # prevents its empty first cells from controlling interpolation.
    dlnrho_dlnR = np.clip(dlnrho_dlnR, -12.0, 12.0)

    return {
        "R": R_grid,
        "z": z_grid,
        "Sigma_msun_pc2": sigma_msun_pc2,
        "Sigma_code": sigma_code,
        "rho": rho,
        "cdf_R": cdf_R,
        "cdf_z": cdf_z,
        "retained_mass_code": retained_mass_code,
        "dlnrho_dlnR": dlnrho_dlnR,
        "cs2": cs2,
    }


def sample_ism_positions(rng, n, table):
    """Sample continuous positions from the McMillan+HSE ISM table."""
    n = int(n)
    R_grid = table["R"]
    z_grid = table["z"]
    R = np.interp(rng.random(n), table["cdf_R"], R_grid)

    idx = np.searchsorted(R_grid, R)
    idx = np.clip(idx, 1, len(R_grid) - 1)
    left = idx - 1
    row = np.where(
        np.abs(R_grid[idx] - R) < np.abs(R - R_grid[left]), idx, left
    )

    zabs = np.empty(n, dtype=np.float64)
    uz = rng.random(n)
    for j in np.unique(row):
        sel = row == j
        zabs[sel] = np.interp(uz[sel], table["cdf_z"][j], z_grid)
    z = np.where(rng.random(n) < 0.5, -zabs, zabs)

    phi = 2.0 * np.pi * rng.random(n)
    pos = np.empty((n, 3), dtype=np.float64)
    pos[:, 0] = R * np.cos(phi)
    pos[:, 1] = R * np.sin(phi)
    pos[:, 2] = z
    return pos, R, z


def assign_ism_velocities(pos, table):
    """Circular rotation including the initial isothermal pressure gradient."""
    x = pos[:, 0]
    y = pos[:, 1]
    R = np.sqrt(x * x + y * y)
    Rs = np.maximum(R, 1.0e-8)
    vc = v_circ_mou(Rs)
    dlnrho = np.interp(Rs, table["R"], table["dlnrho_dlnR"])
    vphi2_raw = vc**2 + table["cs2"] * dlnrho
    vphi2 = np.clip(vphi2_raw, (0.25 * vc) ** 2, (1.25 * vc) ** 2)
    vphi = np.sqrt(np.maximum(vphi2, 0.0))

    vel = np.zeros_like(pos)
    vel[:, 0] = -vphi * y / Rs
    vel[:, 1] = +vphi * x / Rs

    Rt = np.maximum(table["R"], 1.0e-3)
    vct = v_circ_mou(Rt)
    rawt = vct**2 + table["cs2"] * table["dlnrho_dlnR"]
    clipped = (rawt < (0.25 * vct) ** 2) | (rawt > (1.25 * vct) ** 2)
    radial_weight = 2.0 * np.pi * table["R"] * table["Sigma_code"]
    norm = float(_trapz(radial_weight, table["R"]))
    clipped_fraction = float(
        _trapz(np.where(clipped, radial_weight, 0.0), table["R"]) / norm
    )
    return vel, clipped_fraction


# =============================================================================
# 9. Jeans 速度：halo / bulge
# =============================================================================

def spherical_average_inward_acceleration(r_grid, nmu=129):
    """
    对 Mou 总解析势的径向 inward acceleration 做球面平均。

    用于 spherical halo/bulge 的 isotropic Jeans 初始化。
    """
    r_grid = np.asarray(r_grid, dtype=np.float64)
    mu = np.linspace(-1.0, 1.0, int(nmu))
    sin_theta = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))

    out = np.empty_like(r_grid)

    chunk = 256

    for i0 in range(0, len(r_grid), chunk):
        i1 = min(i0 + chunk, len(r_grid))

        rr = r_grid[i0:i1, None]
        R = rr * sin_theta[None, :]
        z = rr * mu[None, :]

        aR, az = mou_accel_cyl(R, z)

        rs = np.maximum(rr, 1.0e-12)

        # 径向加速度分量；取 inward magnitude。
        ar = (
            aR * R / rs
            + az * z / rs
        )
        g_in = -ar

        out[i0:i1] = (
            0.5 * _trapz(g_in, mu, axis=1)
        )

    return np.maximum(out, 0.0)


def build_isotropic_jeans_table(
    rmax,
    rho_shape_func,
    boundary_sigma_kms,
    ngrid=30000,
):
    """
    解 spherical isotropic Jeans：

        d(rho sigma_r^2)/dr = -rho g

    使用外边界：
        sigma_r(rmax) = boundary_sigma_kms

    对一个人为硬截断、外部无压的 isolated realization，应使用
    boundary_sigma_kms=0。非零边界压只适用于明确存在外部承压介质时。
    """
    rr = np.geomspace(1.0e-3, float(rmax), int(ngrid))

    rho = np.maximum(
        rho_shape_func(rr),
        1.0e-300,
    )

    g = spherical_average_inward_acceleration(rr)

    integrand = rho * g

    trap = (
        0.5
        * (integrand[:-1] + integrand[1:])
        * np.diff(rr)
    )

    integ = np.zeros_like(rr)
    integ[:-1] = np.cumsum(trap[::-1])[::-1]

    boundary_term = (
        rho[-1] * float(boundary_sigma_kms) ** 2
    )

    sigma2 = (
        boundary_term + integ
    ) / rho

    return rr, np.sqrt(np.maximum(sigma2, 0.0))


def sample_isotropic_velocities(rng, sigma):
    """从各粒子的 isotropic Gaussian Jeans dispersion 采样速度。"""
    v = rng.normal(size=(len(sigma), 3))
    v *= sigma[:, None]
    return v


# =============================================================================
# 10. MN disk 速度初始化
# =============================================================================

def build_disk_kinematic_table(mn_table, q_target=Q_TARGET_DEFAULT):
    """
    为 live MN disk 构造近似平衡的速度表。

    这里没有 Mou 原文可直接复制的 stellar DF，因为 Mou 使用固定解析盘。
    因此 Q_target 是本粒子化实现的数值稳定选择，不是 Mou 的参数。

    径向方向：
        Toomre Q + epicyclic relation + radial Jeans asymmetric drift

    垂直方向：
        在完整 (R,z) 表上解 vertical Jeans：
        rho sigma_z^2(R,z) = int_z^inf rho dPhi/dz' dz'
    """
    R_grid = mn_table["R"]
    z_grid = mn_table["z"]
    sigma_surface = mn_table["Sigma"]
    rho_grid = mn_table["rho"]

    R_safe = np.maximum(R_grid, 1.0e-3)

    vc = v_circ_mou(R_safe)
    kappa = epicyclic_kappa_mou(R_safe)

    sigma_R_raw = (
        float(q_target)
        * 3.36
        * G_CODE
        * sigma_surface
        / np.maximum(kappa, 1.0e-10)
    )

    # 极低密度远尾部的 Q 公式数值意义变差，设置温和上下限。
    sigma_R = np.clip(sigma_R_raw, 3.0, 140.0)

    Omega = vc / R_safe

    sigma_phi = (
        sigma_R
        * kappa
        / np.maximum(2.0 * Omega, 1.0e-10)
    )
    sigma_phi = np.clip(sigma_phi, 2.0, 140.0)

    # vertical Jeans on the full 2D grid. The old generator evaluated only
    # the midplane moment and assigned it at every |z|, producing a vertical
    # tensor-virial mismatch and a strong breathing/thickening transient.
    RR = R_grid[:, None]
    zz = z_grid[None, :]

    _, az = mou_accel_cyl(RR, zz)
    dphi_dz = -az

    integrand = rho_grid * dphi_dz

    dz = np.diff(z_grid)
    vertical_trap = (
        0.5 * (integrand[:, :-1] + integrand[:, 1:]) * dz[None, :]
    )
    vertical_integral = np.zeros_like(rho_grid)
    vertical_integral[:, :-1] = np.cumsum(
        vertical_trap[:, ::-1], axis=1
    )[:, ::-1]
    sigma_z2_grid = vertical_integral / np.maximum(rho_grid, 1.0e-300)
    sigma_z_raw_grid = np.sqrt(np.maximum(sigma_z2_grid, 0.0))
    sigma_z_grid = np.clip(sigma_z_raw_grid, 2.0, 120.0)

    # radial Jeans asymmetric drift：
    # v_c^2 - vphi_bar^2 =
    # sigma_R^2 [sigma_phi^2/sigma_R^2 - 1
    #            - d ln(nu sigma_R^2)/d ln R]
    nu_mid = np.maximum(rho_grid[:, 0], 1.0e-300)
    A = np.maximum(nu_mid * sigma_R**2, 1.0e-300)

    lnR = np.log(R_safe)
    lnA = np.log(A)

    dlnA_dlnR = np.gradient(
        lnA,
        lnR,
        edge_order=2,
    )

    ratio = (
        sigma_phi
        / np.maximum(sigma_R, 1.0e-12)
    ) ** 2

    drift_term = sigma_R**2 * (
        ratio - 1.0 - dlnA_dlnR
    )

    vphi2_raw = vc**2 - drift_term

    # 中心和极远尾部的局域 Jeans 近似会变差，避免出现非物理解。
    vphi2 = np.clip(
        vphi2_raw,
        (0.25 * vc) ** 2,
        vc**2,
    )

    vphi_mean = np.sqrt(np.maximum(vphi2, 0.0))

    # 用盘的径向质量权重记录经验性裁剪影响了多少初始质量。
    # 这些数字是诊断量，不是稳定性通过标准。
    radial_mass_weight = 2.0 * np.pi * R_grid * sigma_surface
    radial_mass_norm = float(_trapz(radial_mass_weight, R_grid))

    def clipped_radial_mass_fraction(mask):
        if radial_mass_norm <= 0.0:
            return np.nan
        return float(
            _trapz(
                np.where(mask, radial_mass_weight, 0.0),
                R_grid,
            )
            / radial_mass_norm
        )

    return {
        "R": R_grid,
        "vc": vc,
        "sigma_R": sigma_R,
        "sigma_phi": sigma_phi,
        "z": z_grid,
        "sigma_z_grid": sigma_z_grid,
        "vphi_mean": vphi_mean,
        "sigma_R_clipped_mass_fraction": clipped_radial_mass_fraction(
            (sigma_R_raw < 3.0) | (sigma_R_raw > 140.0)
        ),
        "sigma_z_clipped_mass_fraction": float(
            _trapz(
                2.0 * np.pi * R_grid
                * 2.0
                * _trapz(
                    np.where(
                        (sigma_z_raw_grid < 2.0) | (sigma_z_raw_grid > 120.0),
                        rho_grid,
                        0.0,
                    ),
                    z_grid,
                    axis=1,
                ),
                R_grid,
            )
            / radial_mass_norm
        ),
        "vphi_clipped_mass_fraction": clipped_radial_mass_fraction(
            (vphi2_raw < (0.25 * vc) ** 2) | (vphi2_raw > vc**2)
        ),
    }


def _interp_rectilinear_2d(x, y, xg, yg, values):
    """Bilinear interpolation on monotonic, non-uniform rectilinear grids."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ix = np.clip(np.searchsorted(xg, x), 1, len(xg) - 1)
    iy = np.clip(np.searchsorted(yg, y), 1, len(yg) - 1)
    ix0 = ix - 1
    iy0 = iy - 1
    tx = (x - xg[ix0]) / np.maximum(xg[ix] - xg[ix0], 1.0e-300)
    ty = (y - yg[iy0]) / np.maximum(yg[iy] - yg[iy0], 1.0e-300)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)
    return (
        (1.0 - tx) * (1.0 - ty) * values[ix0, iy0]
        + tx * (1.0 - ty) * values[ix, iy0]
        + (1.0 - tx) * ty * values[ix0, iy]
        + tx * ty * values[ix, iy]
    )


def assign_disk_velocities(rng, pos, kin):
    """将 kinematic table 插值到每个 stellar-disk 粒子。"""
    x = pos[:, 0]
    y = pos[:, 1]

    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)

    Rg = kin["R"]

    sR = np.interp(R, Rg, kin["sigma_R"])
    sp = np.interp(R, Rg, kin["sigma_phi"])
    sz = _interp_rectilinear_2d(
        R,
        np.abs(pos[:, 2]),
        Rg,
        kin["z"],
        kin["sigma_z_grid"],
    )
    vp_mean = np.interp(R, Rg, kin["vphi_mean"])

    vR = rng.normal(0.0, sR)
    vphi = rng.normal(vp_mean, sp)
    vz = rng.normal(0.0, sz)

    cp = np.cos(phi)
    spn = np.sin(phi)

    vel = np.empty_like(pos)
    vel[:, 0] = vR * cp - vphi * spn
    vel[:, 1] = vR * spn + vphi * cp
    vel[:, 2] = vz

    return vel


# =============================================================================
# 11. Mou isothermal HSE CGM（公式保留可选旋转项）
# =============================================================================

def cgm_thermal_scale_kms2(T=MOU_T_CGM):
    """k_B T / (mu m_p)，单位 (km/s)^2。"""
    return thermal_scale_kms2(T, MU_HOT)


def internal_energy_from_temperature(T=MOU_T_CGM):
    """理想单原子气体 specific internal energy，GIZMO code unit。"""
    return specific_internal_energy(T, MU_HOT)


def cgm_effective_potential(R, z, f_rot=MOU_F_ROT):
    """Mou effective potential Phi(R,z)-f_rot^2 Phi(R,0)."""
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return mou_potential_cyl(R, z) - float(f_rot) ** 2 * mou_potential_cyl(
        R, np.zeros_like(R)
    )


def cgm_rho_relative(
    R,
    z,
    T=MOU_T_CGM,
    f_rot=MOU_F_ROT,
    core_radius=CGM_CORE_RADIUS_DEFAULT,
):
    """
    Mou rotating isothermal HSE：

        Phi_eff = Phi(R,z) - f^2 Phi(R,0)

    外部包络沿用 Mou 以 Phi(0) 定义的归一化；若 core_radius>0，
    实际中心密度由等势熵核截断，不再等于 rho0。
    """
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    phi_eff = cgm_effective_potential(R, z, f_rot=f_rot)
    phi_eff0 = float(cgm_effective_potential(0.0, 0.0, f_rot=f_rot))
    scale = cgm_thermal_scale_kms2(T)

    # Outside the core, retain Mou exactly. Inside the chosen equipotential,
    # cap rho at its boundary value; cgm_internal_energy then raises P/rho
    # inward by Phi_core-Phi, giving an exact constant-density HSE core.
    if float(core_radius) > 0.0:
        phi_core = float(
            cgm_effective_potential(float(core_radius), 0.0, f_rot=f_rot)
        )
        phi_for_rho = np.maximum(phi_eff, phi_core)
    else:
        phi_for_rho = phi_eff

    arg = -(phi_for_rho - phi_eff0) / scale

    return np.exp(np.clip(arg, -745.0, 80.0))


def cgm_rho_mass_cgs(
    R,
    z,
    rho0_mh_cm3=MOU_RHO0_MH_CM3,
    T=MOU_T_CGM,
    f_rot=MOU_F_ROT,
    core_radius=CGM_CORE_RADIUS_DEFAULT,
):
    """Entropy-cored Mou CGM mass density [g/cm^3]."""
    return (
        float(rho0_mh_cm3)
        * M_P_CGS
        * cgm_rho_relative(
            R, z, T=T, f_rot=f_rot, core_radius=core_radius
        )
    )


def cgm_internal_energy(pos, T, f_rot, core_radius):
    """Specific internal energy for the isothermal envelope + HSE entropy core."""
    pos = np.asarray(pos, dtype=np.float64)
    R = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi_eff = cgm_effective_potential(R, pos[:, 2], f_rot=f_rot)
    cs2 = cgm_thermal_scale_kms2(T)
    pressure_over_rho = np.full(len(pos), cs2, dtype=np.float64)
    if float(core_radius) > 0.0:
        phi_core = float(
            cgm_effective_potential(float(core_radius), 0.0, f_rot=f_rot)
        )
        inside = phi_eff < phi_core
        pressure_over_rho[inside] += phi_core - phi_eff[inside]
    return pressure_over_rho / (GAMMA - 1.0)


def integrate_cgm_mass_msun(
    rmax,
    rho0_mh_cm3,
    T,
    f_rot,
    core_radius,
    nr=8193,
    nmu=513,
):
    """球坐标积分 Mou HSE CGM 总质量。"""
    r_grid = np.linspace(
        0.0,
        float(rmax),
        int(nr),
        dtype=np.float64,
    )
    mu_grid = np.linspace(
        -1.0,
        1.0,
        int(nmu),
        dtype=np.float64,
    )

    sin_theta = np.sqrt(
        np.maximum(1.0 - mu_grid**2, 0.0)
    )

    int_mu = np.empty_like(r_grid)

    chunk = 256

    for i0 in range(0, len(r_grid), chunk):
        i1 = min(i0 + chunk, len(r_grid))

        rr = r_grid[i0:i1, None]
        RR = rr * sin_theta[None, :]
        zz = rr * mu_grid[None, :]

        rho = cgm_rho_mass_cgs(
            RR,
            zz,
            rho0_mh_cm3=rho0_mh_cm3,
            T=T,
            f_rot=f_rot,
            core_radius=core_radius,
        )

        int_mu[i0:i1] = _trapz(
            rho,
            mu_grid,
            axis=1,
        )

    integrand = r_grid**2 * int_mu

    mass_g = (
        2.0
        * np.pi
        * _trapz(integrand, r_grid)
        * UNIT_LENGTH_CM**3
    )

    return mass_g / MSUN_G


def sample_cgm_positions(
    rng,
    n,
    rmax,
    T,
    f_rot,
    core_radius,
    n_r_env=4096,
    n_mu_env=513,
    safety=1.35,
):
    """
    连续球壳 rejection sampling。

    球壳表只作为 sampling envelope；
    实际粒子位置是连续随机变量，并非网格粒子。
    """
    print("构造 Mou CGM 连续采样包络...", flush=True)

    s = np.linspace(0.0, 1.0, int(n_r_env) + 1)
    r_edges = float(rmax) * s**1.35

    rin = r_edges[:-1]
    rout = r_edges[1:]
    rmid = 0.5 * (rin + rout)

    mu_env = np.linspace(-1.0, 1.0, int(n_mu_env))
    sin_env = np.sqrt(np.maximum(1.0 - mu_env**2, 0.0))

    def sphere_max(rv):
        out = np.empty_like(rv)
        chunk = 256

        for i0 in range(0, len(rv), chunk):
            i1 = min(i0 + chunk, len(rv))

            rr = rv[i0:i1, None]
            RR = rr * sin_env[None, :]
            zz = rr * mu_env[None, :]

            out[i0:i1] = np.max(
                cgm_rho_relative(
                    RR,
                    zz,
                    T=T,
                    f_rot=f_rot,
                    core_radius=core_radius,
                ),
                axis=1,
            )

        return out

    env = np.maximum.reduce(
        [
            sphere_max(rin),
            sphere_max(rmid),
            sphere_max(rout),
        ]
    )
    env *= float(safety)

    shell_weight = env * (rout**3 - rin**3)
    shell_weight = np.maximum(shell_weight, 0.0)

    cdf = np.cumsum(shell_weight)
    cdf /= cdf[-1]

    pos = np.empty((int(n), 3), dtype=np.float64)

    count = 0
    proposed = 0
    accepted_total = 0
    next_report = 0.10

    while count < int(n):
        need = int(n) - count
        ntry = min(
            300000,
            max(int(need * 2.0), 20000),
        )

        shell = np.searchsorted(
            cdf,
            rng.random(ntry),
            side="right",
        )
        shell = np.minimum(shell, len(rin) - 1)

        ri = rin[shell]
        ro = rout[shell]

        r = (
            ri**3
            + rng.random(ntry) * (ro**3 - ri**3)
        ) ** (1.0 / 3.0)

        mu = 2.0 * rng.random(ntry) - 1.0
        st = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
        phi = 2.0 * np.pi * rng.random(ntry)

        R = r * st
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z = r * mu

        rho = cgm_rho_relative(
            R,
            z,
            T=T,
            f_rot=f_rot,
            core_radius=core_radius,
        )

        p = rho / np.maximum(env[shell], 1.0e-300)

        # rejection sampling 要求 envelope 在所有候选点都不小于目标密度。
        # 旧逻辑把 p>1 静默裁剪到 1，会引入不可控的采样偏差。
        if np.any(p > 1.0):
            pmax = float(np.max(p))
            nbad = int(np.count_nonzero(p > 1.0))
            raise RuntimeError(
                "CGM rejection envelope 失效："
                f"本批有 {nbad:,} 个候选点满足 p>1，max(p)={pmax:.8g}。"
                "请提高 --cgm-envelope-safety 或增加包络分辨率；"
                "本次不会输出带偏差的 IC。"
            )

        good = np.nonzero(rng.random(ntry) < p)[0]

        proposed += ntry
        accepted_total += good.size

        take = min(good.size, need)

        if take > 0:
            good = good[:take]
            pos[count:count + take, 0] = x[good]
            pos[count:count + take, 1] = y[good]
            pos[count:count + take, 2] = z[good]
            count += take

        frac = count / int(n)

        if frac >= next_report or count == int(n):
            rate = (
                100.0 * accepted_total
                / max(proposed, 1)
            )
            print(
                f"  CGM {count:,}/{int(n):,} "
                f"({100*frac:.1f}%), "
                f"proposal 接受率约 {rate:.1f}%",
                flush=True,
            )
            next_report += 0.10

    return pos


def assign_cgm_velocities(pos, f_rot=MOU_F_ROT):
    """Mou v_phi = f_rot * v_K(R,0)，无额外 bulk turbulence。"""
    x = pos[:, 0]
    y = pos[:, 1]

    R = np.sqrt(x * x + y * y)
    Rs = np.maximum(R, 1.0e-12)

    vphi = float(f_rot) * v_circ_mou(Rs)

    vel = np.zeros_like(pos)
    vel[:, 0] = -vphi * y / Rs
    vel[:, 1] = +vphi * x / Rs
    vel[:, 2] = 0.0

    return vel


# =============================================================================
# 12. 参数
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="生成 Mou (2023) 势模型驱动的 live MW + CGM IC"
    )

    p.add_argument(
        "--outfile",
        default="mw_mou_core_noism_ic.hdf5",
    )

    p.add_argument(
        "--include-ism",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "0: PartType0 仅含热 CGM，并由无碰撞 MN 盘承载完整 Mou "
            "M_disc；1: 恢复 v4 的 McMillan HI+H2 亚网格 ISM"
        ),
    )

    p.add_argument(
        "--boxsize",
        type=float,
        default=BOXSIZE_DEFAULT,
    )

    p.add_argument(
        "--rmax-dm",
        type=float,
        default=RMAX_DM_DEFAULT,
    )

    p.add_argument(
        "--rmax-disc",
        type=float,
        default=RMAX_DISC_DEFAULT,
    )

    p.add_argument(
        "--zmax-disc",
        type=float,
        default=ZMAX_DISC_DEFAULT,
    )

    p.add_argument(
        "--rmax-bulge",
        type=float,
        default=RMAX_BULGE_DEFAULT,
    )

    p.add_argument(
        "--rmax-cgm",
        type=float,
        default=RMAX_CGM_DEFAULT,
    )

    p.add_argument(
        "--rmax-ism",
        type=float,
        default=RMAX_ISM_DEFAULT,
    )

    p.add_argument(
        "--zmax-ism",
        type=float,
        default=ZMAX_ISM_DEFAULT,
    )

    p.add_argument(
        "--dm-target-mass",
        type=float,
        default=TARGET_DM_MASS_MSUN,
    )

    p.add_argument(
        "--star-target-mass",
        type=float,
        default=TARGET_STAR_MASS_MSUN,
    )

    p.add_argument(
        "--cgm-target-mass",
        type=float,
        default=TARGET_CGM_MASS_MSUN,
    )

    p.add_argument(
        "--ism-target-mass",
        type=float,
        default=TARGET_ISM_MASS_MSUN,
    )

    p.add_argument(
        "--cgm-core-radius",
        type=float,
        default=CGM_CORE_RADIUS_DEFAULT,
        help=(
            "equatorial radius of the constant-density hydrostatic CGM entropy "
            "core; 0 disables the core"
        ),
    )

    p.add_argument(
        "--q-target",
        type=float,
        default=Q_TARGET_DEFAULT,
        help="live MN disk 的数值稳定 Toomre-Q；不是 Mou 原文参数",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )

    p.add_argument(
        "--mn-nr",
        type=int,
        default=1201,
    )

    p.add_argument(
        "--mn-nz",
        type=int,
        default=1001,
    )

    p.add_argument(
        "--ism-nr",
        type=int,
        default=1601,
    )

    p.add_argument(
        "--ism-nz",
        type=int,
        default=801,
    )

    p.add_argument(
        "--cgm-envelope-r",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--cgm-envelope-mu",
        type=int,
        default=513,
    )

    p.add_argument(
        "--cgm-envelope-safety",
        type=float,
        default=1.35,
        help="CGM rejection envelope 的安全系数，必须 > 1",
    )

    p.add_argument(
        "--write-chunk-size",
        type=int,
        default=HDF5_WRITE_CHUNK_DEFAULT,
        help="HDF5 顺序写入与校验的粒子块大小",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="允许原子替换已经存在的输出文件",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="run all sampling/physics checks but do not write HDF5",
    )

    p.add_argument(
        "--include-bh",
        type=int,
        choices=[0, 1],
        default=1,
    )

    p.add_argument(
        "--bh-mass-msun",
        type=float,
        default=M_BH_MSUN_DEFAULT,
    )

    return p.parse_args()


def validate_args(args):
    """在分配大数组之前拒绝不自洽或危险的参数。"""
    positive_float_names = [
        "boxsize",
        "rmax_dm",
        "rmax_disc",
        "zmax_disc",
        "rmax_bulge",
        "rmax_cgm",
        "rmax_ism",
        "zmax_ism",
        "dm_target_mass",
        "star_target_mass",
        "cgm_target_mass",
        "ism_target_mass",
        "q_target",
    ]
    for name in positive_float_names:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是有限正数。")

    if bool(args.include_bh):
        if not np.isfinite(args.bh_mass_msun) or args.bh_mass_msun <= 0.0:
            raise ValueError("--bh-mass-msun 在包含 BH 时必须是有限正数。")

    if not np.isfinite(args.cgm_core_radius) or args.cgm_core_radius < 0.0:
        raise ValueError("--cgm-core-radius must be finite and non-negative.")
    if args.cgm_core_radius >= args.rmax_cgm:
        raise ValueError("--cgm-core-radius must be smaller than --rmax-cgm.")

    # 所有组分都平移到 box center；必须把球形 bulge 截断也计入。
    max_extent = max(
        args.rmax_dm,
        args.rmax_disc,
        args.zmax_disc,
        args.rmax_bulge,
        args.rmax_cgm,
        args.rmax_ism,
        args.zmax_ism,
    )
    if args.boxsize <= 2.0 * max_extent:
        raise ValueError(
            "BoxSize 太小，无法容纳当前所有组分的截断范围；"
            f"需要 BoxSize > {2.0 * max_extent:.8g} kpc。"
        )

    if args.mn_nr < 33 or args.mn_nz < 33:
        raise ValueError("--mn-nr 与 --mn-nz 至少应为 33。")
    if args.ism_nr < 33 or args.ism_nz < 33:
        raise ValueError("--ism-nr and --ism-nz must both be at least 33.")
    if args.cgm_envelope_r < 32:
        raise ValueError("--cgm-envelope-r 至少应为 32。")
    if args.cgm_envelope_mu < 33 or args.cgm_envelope_mu % 2 == 0:
        raise ValueError(
            "--cgm-envelope-mu 必须是 >=33 的奇数，以确保包络包含赤道 mu=0。"
        )
    if (
        not np.isfinite(args.cgm_envelope_safety)
        or args.cgm_envelope_safety <= 1.0
    ):
        raise ValueError("--cgm-envelope-safety 必须是大于 1 的有限数。")
    if args.write_chunk_size <= 0:
        raise ValueError("--write-chunk-size 必须是正整数。")

    outfile = os.path.abspath(os.path.expanduser(args.outfile))
    parent = os.path.dirname(outfile) or os.curdir
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"输出目录不存在：{parent}")
    if os.path.isdir(outfile):
        raise IsADirectoryError(f"输出路径是目录：{outfile}")
    if os.path.exists(outfile) and not args.overwrite:
        raise FileExistsError(
            f"输出文件已存在：{outfile}；如需原子替换，请显式添加 --overwrite。"
        )


# =============================================================================
# 13. HDF5 写入
# =============================================================================

def iter_particle_slices(n, chunk_size):
    """按粒子数产生顺序切片，避免构造完整的写入临时副本。"""
    n = int(n)
    chunk_size = int(chunk_size)
    for i0 in range(0, n, chunk_size):
        yield slice(i0, min(i0 + chunk_size, n))


def _validate_ptype_inputs(
    pos,
    vel,
    ids,
    mass_code,
    internal_energy,
    metallicity,
    bh_mass,
    stellar_formation_time,
):
    """只验证形状、长度与 ID 可表示性，不额外复制大数组。"""
    pos = np.asarray(pos)
    vel = np.asarray(vel)
    ids = np.asarray(ids)
    n = len(pos)

    if n <= 0:
        raise ValueError("write_ptype 不应为零粒子类型创建空 group。")
    if pos.shape != (n, 3):
        raise ValueError(f"Coordinates 形状错误：{pos.shape}，期望 {(n, 3)}。")
    if vel.shape != (n, 3):
        raise ValueError(f"Velocities 形状错误：{vel.shape}，期望 {(n, 3)}。")
    if ids.shape != (n,):
        raise ValueError(f"ParticleIDs 形状错误：{ids.shape}，期望 {(n,)}。")

    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("ParticleIDs 必须是整数数组。")
    id_min = int(np.min(ids))
    id_max = int(np.max(ids))
    if id_min <= 0 or id_max > int(np.iinfo(np.uint32).max):
        raise OverflowError(
            "当前 IC 使用 uint32 ParticleIDs；ID 必须位于 [1, 2^32-1]。"
        )

    if np.ndim(mass_code) != 0 and np.asarray(mass_code).shape != (n,):
        raise ValueError("Masses 必须是标量或长度为 N 的一维数组。")
    if internal_energy is not None:
        if np.ndim(internal_energy) != 0 and np.asarray(internal_energy).shape != (n,):
            raise ValueError("InternalEnergy 必须是标量或长度为 N 的一维数组。")
    if metallicity is not None:
        z = np.asarray(metallicity)
        if z.shape not in {(11,), (n, 11)}:
            raise ValueError(
                "Metallicity 必须是标准 11 维常量向量或 N x 11 数组；"
                f"实际为 {z.shape}。"
            )
    if bh_mass is not None:
        if np.ndim(bh_mass) != 0 and np.asarray(bh_mass).shape != (n,):
            raise ValueError("BH_Mass 必须是标量或长度为 N 的一维数组。")
    if stellar_formation_time is not None:
        if (
            np.ndim(stellar_formation_time) != 0
            and np.asarray(stellar_formation_time).shape != (n,)
        ):
            raise ValueError(
                "StellarFormationTime must be scalar or a length-N array."
            )

    return pos, vel, ids, n


def write_ptype(
    f,
    ptype,
    pos,
    vel,
    ids,
    mass_code,
    box_center,
    internal_energy=None,
    metallicity=None,
    bh_mass=None,
    stellar_formation_time=None,
    write_chunk_size=HDF5_WRITE_CHUNK_DEFAULT,
):
    pos, vel, ids, n = _validate_ptype_inputs(
        pos,
        vel,
        ids,
        mass_code,
        internal_energy,
        metallicity,
        bh_mass,
        stellar_formation_time,
    )
    box_center = np.asarray(box_center, dtype=np.float64)
    if box_center.shape != (3,):
        raise ValueError("box_center 必须是长度为 3 的向量。")

    g = f.create_group(f"PartType{ptype}")

    dset = g.create_dataset(
        "Coordinates",
        shape=(n, 3),
        dtype=np.float64,
    )
    for sl in iter_particle_slices(n, write_chunk_size):
        dset[sl] = np.asarray(pos[sl], dtype=np.float64) + box_center[None, :]

    dset = g.create_dataset(
        "Velocities",
        shape=(n, 3),
        dtype=np.float64,
    )
    for sl in iter_particle_slices(n, write_chunk_size):
        dset[sl] = np.asarray(vel[sl], dtype=np.float64)

    # 当前所有 ID < 3e8，32-bit 足够，不需要 LONGIDS
    dset = g.create_dataset(
        "ParticleIDs",
        shape=(n,),
        dtype=np.uint32,
    )
    for sl in iter_particle_slices(n, write_chunk_size):
        dset[sl] = np.asarray(ids[sl], dtype=np.uint32)

    dset = g.create_dataset(
        "Masses",
        shape=(n,),
        dtype=np.float64,
    )
    mass_is_scalar = np.ndim(mass_code) == 0
    for sl in iter_particle_slices(n, write_chunk_size):
        if mass_is_scalar:
            dset[sl] = float(mass_code)
        else:
            dset[sl] = np.asarray(mass_code[sl], dtype=np.float64)

    if internal_energy is not None:
        dset = g.create_dataset(
            "InternalEnergy",
            shape=(n,),
            dtype=np.float64,
        )
        u_is_scalar = np.ndim(internal_energy) == 0
        for sl in iter_particle_slices(n, write_chunk_size):
            if u_is_scalar:
                dset[sl] = float(internal_energy)
            else:
                dset[sl] = np.asarray(internal_energy[sl], dtype=np.float64)

    if metallicity is not None:
        dset = g.create_dataset(
            "Metallicity",
            shape=(n, 11),
            dtype=np.float64,
        )
        z = np.asarray(metallicity, dtype=np.float64)
        for sl in iter_particle_slices(n, write_chunk_size):
            if z.ndim == 1:
                block_shape = (sl.stop - sl.start, 11)
                dset[sl] = np.broadcast_to(z, block_shape)
            else:
                dset[sl] = z[sl]

    if bh_mass is not None:
        dset = g.create_dataset(
            "BH_Mass",
            shape=(n,),
            dtype=np.float64,
        )
        bh_is_scalar = np.ndim(bh_mass) == 0
        for sl in iter_particle_slices(n, write_chunk_size):
            if bh_is_scalar:
                dset[sl] = float(bh_mass)
            else:
                dset[sl] = np.asarray(bh_mass[sl], dtype=np.float64)

    if stellar_formation_time is not None:
        dset = g.create_dataset(
            "StellarFormationTime",
            shape=(n,),
            dtype=np.float64,
        )
        tform_is_scalar = np.ndim(stellar_formation_time) == 0
        for sl in iter_particle_slices(n, write_chunk_size):
            if tform_is_scalar:
                dset[sl] = float(stellar_formation_time)
            else:
                dset[sl] = np.asarray(
                    stellar_formation_time[sl], dtype=np.float64
                )


def _check_finite_dataset(dataset, name, chunk_size, positive=False):
    for sl in iter_particle_slices(dataset.shape[0], chunk_size):
        block = np.asarray(dataset[sl])
        if not np.all(np.isfinite(block)):
            raise RuntimeError(f"{name} 存在 NaN/Inf。")
        if positive and np.any(block <= 0.0):
            raise RuntimeError(f"{name} 存在非正值。")


def validate_ic_file(path, expected_counts, boxsize, chunk_size):
    """分块验证 GIZMO IC 的 Header、形状、数值范围与 ID 单调唯一性。"""
    expected = np.zeros(6, dtype=np.uint32)
    for ptype, count in expected_counts.items():
        expected[int(ptype)] = np.uint32(count)

    with h5py.File(path, "r") as f:
        if "Header" not in f:
            raise RuntimeError("IC 缺少 Header group。")
        header = f["Header"]
        for attr in [
            "NumPart_ThisFile",
            "NumPart_Total",
            "NumPart_Total_HighWord",
            "MassTable",
            "NumFilesPerSnapshot",
            "Flag_Entropy_ICs",
            "Flag_Metals",
        ]:
            if attr not in header.attrs:
                raise RuntimeError(f"Header 缺少属性 {attr}。")

        if not np.array_equal(header.attrs["NumPart_ThisFile"], expected):
            raise RuntimeError("Header/NumPart_ThisFile 与实际计划不一致。")
        if not np.array_equal(header.attrs["NumPart_Total"], expected):
            raise RuntimeError("Header/NumPart_Total 与实际计划不一致。")
        if np.any(header.attrs["NumPart_Total_HighWord"] != 0):
            raise RuntimeError("单文件 IC 的 NumPart_Total_HighWord 应为零。")
        if np.any(header.attrs["MassTable"] != 0.0):
            raise RuntimeError("当前逐粒子写 Masses，Header/MassTable 必须全为零。")
        if int(header.attrs["NumFilesPerSnapshot"]) != 1:
            raise RuntimeError("当前生成器只写单文件 IC。")
        if int(header.attrs["Flag_Entropy_ICs"]) != 0:
            raise RuntimeError("InternalEnergy 数据要求 Flag_Entropy_ICs=0。")
        if int(header.attrs["Flag_Metals"]) != 11:
            raise RuntimeError("标准逐元素 Metallicity 要求 Flag_Metals=11。")

        previous_global_id = None
        for ptype in sorted(expected_counts):
            n = int(expected_counts[ptype])
            group_name = f"PartType{ptype}"
            if group_name not in f:
                raise RuntimeError(f"IC 缺少 {group_name}。")
            g = f[group_name]

            required_shapes = {
                "Coordinates": (n, 3),
                "Velocities": (n, 3),
                "ParticleIDs": (n,),
                "Masses": (n,),
            }
            if ptype == 0:
                required_shapes["InternalEnergy"] = (n,)
                required_shapes["Metallicity"] = (n, 11)
            if ptype == 4:
                required_shapes["Metallicity"] = (n, 11)
                required_shapes["StellarFormationTime"] = (n,)
            if ptype == 5:
                required_shapes["BH_Mass"] = (n,)

            for name, shape in required_shapes.items():
                if name not in g:
                    raise RuntimeError(f"{group_name} 缺少 {name}。")
                if g[name].shape != shape:
                    raise RuntimeError(
                        f"{group_name}/{name} 形状为 {g[name].shape}，期望 {shape}。"
                    )

            _check_finite_dataset(
                g["Coordinates"], f"{group_name}/Coordinates", chunk_size
            )
            _check_finite_dataset(
                g["Velocities"], f"{group_name}/Velocities", chunk_size
            )
            _check_finite_dataset(
                g["Masses"], f"{group_name}/Masses", chunk_size, positive=True
            )

            for sl in iter_particle_slices(n, chunk_size):
                coords = np.asarray(g["Coordinates"][sl])
                if np.any(coords < 0.0) or np.any(coords > float(boxsize)):
                    raise RuntimeError(f"{group_name}/Coordinates 超出 [0, BoxSize]。")

                ids = np.asarray(g["ParticleIDs"][sl], dtype=np.uint64)
                if np.any(ids == 0):
                    raise RuntimeError(f"{group_name}/ParticleIDs 含 0。")
                if ids.size > 1 and np.any(ids[1:] <= ids[:-1]):
                    raise RuntimeError(f"{group_name}/ParticleIDs 非严格递增。")
                if previous_global_id is not None and int(ids[0]) <= previous_global_id:
                    raise RuntimeError("不同 PartType 的 ParticleIDs 重叠或次序错误。")
                previous_global_id = int(ids[-1])

            if ptype == 0:
                _check_finite_dataset(
                    g["InternalEnergy"],
                    f"{group_name}/InternalEnergy",
                    chunk_size,
                    positive=True,
                )

            if ptype in (0, 4):
                _check_finite_dataset(
                    g["Metallicity"],
                    f"{group_name}/Metallicity",
                    chunk_size,
                )
                for sl in iter_particle_slices(n, chunk_size):
                    z = np.asarray(g["Metallicity"][sl])
                    if not np.all(np.isfinite(z)):
                        raise RuntimeError(f"{group_name}/Metallicity has NaN/Inf.")
                    if np.any(z < 0.0) or np.any(z > 1.0):
                        raise RuntimeError(
                            f"{group_name}/Metallicity is outside [0,1]."
                        )
                    if np.any(z[:, 0] + z[:, 1] >= 1.0):
                        raise RuntimeError(
                            f"{group_name}: Z_total + Y_He must be below 1."
                        )

            if ptype == 4:
                _check_finite_dataset(
                    g["StellarFormationTime"],
                    f"{group_name}/StellarFormationTime",
                    chunk_size,
                )

            if ptype == 5:
                _check_finite_dataset(
                    g["BH_Mass"],
                    f"{group_name}/BH_Mass",
                    chunk_size,
                    positive=True,
                )


def particle_id_end(start, count):
    if int(count) <= 0:
        raise ValueError("粒子数必须为正才能规划 ParticleID。")
    return int(start) + int(count) - 1


def validate_particle_id_plan(n_gas, n_dm, n_disc, n_bulge):
    """验证固定 ID 区间互不重叠，并且可由 uint32 精确表示。"""
    ranges = [
        ("MW_GAS", int(MW_CGM_ID_START), int(n_gas), int(MW_DM_ID_START)),
        ("MW_DM", int(MW_DM_ID_START), int(n_dm), int(MW_DISC_ID_START)),
        ("MW_DISC", int(MW_DISC_ID_START), int(n_disc), int(MW_BULGE_ID_START)),
        ("MW_BULGE", int(MW_BULGE_ID_START), int(n_bulge), int(MW_BH_ID)),
    ]
    ends = {}
    for name, start, count, next_start in ranges:
        end = particle_id_end(start, count)
        ends[name] = end
        if end >= next_start:
            raise RuntimeError(
                f"{name} 的 ID 区间 [{start:,}, {end:,}] 进入下一预留区间 "
                f"（起点 {next_start:,}）。请增大固定区间或降低粒子数；"
                "不能继续写出含重复/冲突 ID 的 IC。"
            )

    if not (
        int(MW_BH_ID) < int(LMC_ID_START)
        <= int(LMC_ID_END)
        < int(SMC_ID_START)
        <= int(SMC_ID_END)
        < int(FUTURE_ID_START)
    ):
        raise RuntimeError("LMC/SMC/FUTURE 固定 ParticleID 规划自身不合法。")

    max_reserved_id = max(
        list(ends.values())
        + [
            int(MW_BH_ID),
            int(LMC_ID_END),
            int(SMC_ID_END),
            int(FUTURE_ID_START),
        ]
    )
    if max_reserved_id > int(np.iinfo(np.uint32).max):
        raise OverflowError("ParticleID 规划超过 uint32；需要启用 LONGIDS 并改写 IC。")

    return ends


def remove_file_if_exists(path):
    """仅用于清理本进程创建的未完成临时输出。"""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# =============================================================================
# 14. 主程序
# =============================================================================

def main():
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)
    include_ism = bool(args.include_ism)

    print("============================================================")
    print("Mou (2023) live MW + hot-CGM diagnostic IC")
    print("============================================================")
    print(
        "ISM mode: "
        + (
            "McMillan (2017) HI+H2 sub-grid disk enabled"
            if include_ism
            else "disabled; PartType0 contains CGM only"
        )
    )
    print("Mou M_disc is retained as the total disk gravity target")
    print("All old stars are PartType4; new stars will remain PartType4")
    print(
        "For the first controlled comparison, keep the same GIZMO build and "
        "runtime parameters as the v4 run."
    )
    print("============================================================")

    # -------------------------------------------------------------------------
    # DM halo
    # -------------------------------------------------------------------------
    halo_mass_msun = float(
        mou_halo_mass_msun(args.rmax_dm)
    )

    n_dm = max(
        1,
        int(round(
            halo_mass_msun / args.dm_target_mass
        )),
    )

    dm_particle_mass_msun = halo_mass_msun / n_dm

    print(
        f"DM: Rmax={args.rmax_dm:.1f} kpc, "
        f"M={halo_mass_msun:.6e} Msun, "
        f"N={n_dm:,}, "
        f"mp={dm_particle_mass_msun:.6e} Msun"
    )

    dm_pos, dm_r = sample_mou_halo_positions(
        rng,
        n_dm,
        args.rmax_dm,
    )

    print("构造 DM isotropic Jeans 速度...", flush=True)

    rj_dm, sj_dm = build_isotropic_jeans_table(
        args.rmax_dm,
        mou_halo_density_shape,
        boundary_sigma_kms=0.0,
    )

    dm_sigma = np.interp(dm_r, rj_dm, sj_dm)
    dm_vel = sample_isotropic_velocities(
        rng,
        dm_sigma,
    )

    # -------------------------------------------------------------------------
    # Optional McMillan global ISM.  When disabled, use empty arrays and put
    # the complete Mou disk mass into the collisionless MN stellar disk.  This
    # removes hydrodynamic disk obstruction without changing the analytic disk
    # gravity target.
    # -------------------------------------------------------------------------
    if include_ism:
        print("构造 McMillan HI+H2 global-ISM sampling table...", flush=True)
        ism_table = build_ism_sampling_table(
            args.rmax_ism,
            args.zmax_ism,
            nr=args.ism_nr,
            nz=args.ism_nz,
        )
        ism_mass_code = float(ism_table["retained_mass_code"])
        ism_mass_msun = ism_mass_code * 1.0e10
        n_ism = max(1, int(round(ism_mass_msun / args.ism_target_mass)))
        ism_particle_mass_msun = ism_mass_msun / n_ism
        print(
            f"ISM: Rmax={args.rmax_ism:.1f} kpc, "
            f"M={ism_mass_msun:.6e} Msun, N={n_ism:,}, "
            f"mp={ism_particle_mass_msun:.6e} Msun"
        )
        ism_pos, _ism_R, _ism_z = sample_ism_positions(
            rng,
            n_ism,
            ism_table,
        )
        ism_vel, ism_vphi_clip_fraction = assign_ism_velocities(
            ism_pos,
            ism_table,
        )
        ism_u = specific_internal_energy(ISM_T_INITIAL, MU_ISM)
        ism_Z = gizmo_metallicity_vector(ISM_Z_ZSUN)
    else:
        print(
            "跳过 live ISM：完整 Mou 盘质量由 collisionless MN 恒星盘承载。",
            flush=True,
        )
        ism_table = None
        ism_mass_code = 0.0
        ism_mass_msun = 0.0
        n_ism = 0
        ism_particle_mass_msun = 0.0
        ism_pos = np.empty((0, 3), dtype=np.float64)
        ism_vel = np.empty((0, 3), dtype=np.float64)
        ism_vphi_clip_fraction = 0.0
        ism_u = 0.0
        ism_Z = gizmo_metallicity_vector(ISM_Z_ZSUN)

    stellar_disc_analytic_mass_code = MOU_M_DISC - ism_mass_code
    if stellar_disc_analytic_mass_code <= 0.0:
        raise RuntimeError(
            "The retained McMillan gas disk exceeds the Mou total disk mass."
        )

    # -------------------------------------------------------------------------
    # MN stellar disk
    # -------------------------------------------------------------------------
    print("构造 Miyamoto-Nagai stellar disk 采样表...", flush=True)

    mn_table = build_mn_sampling_table(
        args.rmax_disc,
        args.zmax_disc,
        M=stellar_disc_analytic_mass_code,
        nr=args.mn_nr,
        nz=args.mn_nz,
    )

    disc_mass_code = mn_table["retained_mass_code"]
    disc_mass_msun = disc_mass_code * 1.0e10

    n_disc = max(
        1,
        int(round(
            disc_mass_msun / args.star_target_mass
        )),
    )

    disc_particle_mass_msun = disc_mass_msun / n_disc

    print(
        f"MN disk retained fraction = "
        f"{mn_table['retained_fraction']:.6f}"
    )
    print(
        f"MN disk live mass = {disc_mass_msun:.6e} Msun, "
        f"N={n_disc:,}, "
        f"mp={disc_particle_mass_msun:.6e} Msun"
    )

    disc_pos, disc_R, disc_z = sample_mn_positions(
        rng,
        n_disc,
        mn_table,
    )

    print("构造 MN disk Jeans/epicyclic 速度...", flush=True)

    disc_kin = build_disk_kinematic_table(
        mn_table,
        q_target=args.q_target,
    )

    print(
        "Disk kinematic empirical-clip mass fractions: "
        f"sigma_R={disc_kin['sigma_R_clipped_mass_fraction']:.3%}, "
        f"sigma_z={disc_kin['sigma_z_clipped_mass_fraction']:.3%}, "
        f"vphi_mean={disc_kin['vphi_clipped_mass_fraction']:.3%}. "
        "这些是近似 DF 的诊断量，不代表已达到平衡。"
    )

    disc_vel = assign_disk_velocities(
        rng,
        disc_pos,
        disc_kin,
    )

    # -------------------------------------------------------------------------
    # Hernquist bulge
    # -------------------------------------------------------------------------
    bulge_fraction = (
        args.rmax_bulge
        / (args.rmax_bulge + MOU_D_BULGE)
    ) ** 2

    bulge_mass_msun = (
        MOU_M_BULGE
        * 1.0e10
        * bulge_fraction
    )

    n_bulge = max(
        1,
        int(round(
            bulge_mass_msun / args.star_target_mass
        )),
    )

    bulge_particle_mass_msun = (
        bulge_mass_msun / n_bulge
    )

    print(
        f"Bulge retained fraction = {bulge_fraction:.6f}"
    )
    print(
        f"Bulge live mass = {bulge_mass_msun:.6e} Msun, "
        f"N={n_bulge:,}, "
        f"mp={bulge_particle_mass_msun:.6e} Msun"
    )

    bulge_pos, bulge_r, _ = sample_truncated_hernquist(
        rng,
        n_bulge,
        MOU_D_BULGE,
        args.rmax_bulge,
    )

    print("构造 bulge isotropic Jeans 速度...", flush=True)

    rj_b, sj_b = build_isotropic_jeans_table(
        args.rmax_bulge,
        hernquist_density_shape,
        boundary_sigma_kms=0.0,
    )

    bulge_sigma = np.interp(
        bulge_r,
        rj_b,
        sj_b,
    )

    bulge_vel = sample_isotropic_velocities(
        rng,
        bulge_sigma,
    )

    # -------------------------------------------------------------------------
    # 合并恒星盘与核球到 PartType4
    # -------------------------------------------------------------------------
    # 本项目关注 CGM/LMC/AGN/eROSITA 气泡，不需要区分 disk 与 bulge 的恒星类型。
    # 因此统一写入 GIZMO 的 stellar particle 类型 PartType4。
    star_pos = np.concatenate(
        (disc_pos, bulge_pos),
        axis=0,
    )
    star_vel = np.concatenate(
        (disc_vel, bulge_vel),
        axis=0,
    )

    star_masses_code = np.concatenate(
        (
            np.full(
                n_disc,
                disc_particle_mass_msun / 1.0e10,
                dtype=np.float64,
            ),
            np.full(
                n_bulge,
                bulge_particle_mass_msun / 1.0e10,
                dtype=np.float64,
            ),
        )
    )
    n_star = len(star_pos)
    star_Z = gizmo_metallicity_vector(ISM_Z_ZSUN)
    # Non-cosmological StellarFormationTime is a formation epoch in code time,
    # not an age.  A negative epoch represents stars formed before TimeBegin=0.
    old_star_formation_time_code = -OLD_STAR_AGE_GYR / UNIT_TIME_GYR

    # -------------------------------------------------------------------------
    # CGM
    # -------------------------------------------------------------------------
    print("积分 Mou isothermal-HSE CGM 总质量...", flush=True)

    cgm_mass_msun = integrate_cgm_mass_msun(
        args.rmax_cgm,
        MOU_RHO0_MH_CM3,
        MOU_T_CGM,
        MOU_F_ROT,
        args.cgm_core_radius,
    )

    n_cgm = max(
        1,
        int(round(
            cgm_mass_msun / args.cgm_target_mass
        )),
    )

    cgm_particle_mass_msun = (
        cgm_mass_msun / n_cgm
    )

    print(
        f"CGM: Rmax={args.rmax_cgm:.1f} kpc, "
        f"entropy-core={args.cgm_core_radius:.3f} kpc, "
        f"M={cgm_mass_msun:.6e} Msun, "
        f"N={n_cgm:,}, "
        f"mp={cgm_particle_mass_msun:.6e} Msun"
    )

    cgm_pos = sample_cgm_positions(
        rng,
        n_cgm,
        args.rmax_cgm,
        MOU_T_CGM,
        MOU_F_ROT,
        args.cgm_core_radius,
        n_r_env=args.cgm_envelope_r,
        n_mu_env=args.cgm_envelope_mu,
        safety=args.cgm_envelope_safety,
    )

    cgm_vel = assign_cgm_velocities(
        cgm_pos,
        f_rot=MOU_F_ROT,
    )

    cgm_u = cgm_internal_energy(
        cgm_pos,
        MOU_T_CGM,
        MOU_F_ROT,
        args.cgm_core_radius,
    )

    # 写入器按块广播常量向量，避免为大规模 CGM 预先分配 N x 11 数组。
    cgm_Z = gizmo_metallicity_vector(
        MOU_CGM_Z_ZSUN,
    )

    # -------------------------------------------------------------------------
    # Merge PartType0.  In the default no-ISM mode these arrays contain CGM
    # only.  If ISM is enabled, it comes first to preserve the v4 ID layout.
    # -------------------------------------------------------------------------
    n_gas = n_ism + n_cgm
    validate_particle_id_plan(n_gas, n_dm, n_disc, n_bulge)
    gas_pos = np.concatenate((ism_pos, cgm_pos), axis=0)
    gas_vel = np.concatenate((ism_vel, cgm_vel), axis=0)
    gas_masses_code = np.concatenate(
        (
            np.full(n_ism, ism_particle_mass_msun / 1.0e10),
            np.full(n_cgm, cgm_particle_mass_msun / 1.0e10),
        )
    )
    gas_u = np.concatenate((np.full(n_ism, ism_u), cgm_u))
    gas_Z = np.empty((n_gas, 11), dtype=np.float64)
    gas_Z[:n_ism] = ism_Z
    gas_Z[n_ism:] = cgm_Z

    # -------------------------------------------------------------------------
    # BH
    # -------------------------------------------------------------------------
    include_bh = bool(args.include_bh)

    if include_bh:
        bh_pos = np.zeros((1, 3), dtype=np.float64)
        bh_vel = np.zeros((1, 3), dtype=np.float64)
        bh_mass_code = args.bh_mass_msun / 1.0e10
    else:
        bh_pos = np.zeros((0, 3), dtype=np.float64)
        bh_vel = np.zeros((0, 3), dtype=np.float64)
        bh_mass_code = 0.0

    # -------------------------------------------------------------------------
    # ParticleIDs
    # -------------------------------------------------------------------------
    gas_ids = (
        MW_CGM_ID_START
        + np.arange(n_gas, dtype=np.uint64)
    )
    ism_ids = gas_ids[:n_ism]
    cgm_ids = gas_ids[n_ism:]
    dm_ids = (
        MW_DM_ID_START
        + np.arange(n_dm, dtype=np.uint64)
    )
    disc_ids = (
        MW_DISC_ID_START
        + np.arange(n_disc, dtype=np.uint64)
    )
    bulge_ids = (
        MW_BULGE_ID_START
        + np.arange(n_bulge, dtype=np.uint64)
    )
    star_ids = np.concatenate(
        (disc_ids, bulge_ids),
    ).astype(np.uint64, copy=False)

    # -------------------------------------------------------------------------
    # 基本物理 sanity checks
    # -------------------------------------------------------------------------
    vc82 = float(v_circ_mou(8.2))

    n29 = (
        MOU_RHO0_MH_CM3
        * float(
            cgm_rho_relative(
                29.0,
                0.0,
                T=MOU_T_CGM,
                f_rot=MOU_F_ROT,
                core_radius=args.cgm_core_radius,
            )
        )
    )
    n0_cored = MOU_RHO0_MH_CM3 * float(
        cgm_rho_relative(
            0.0,
            0.0,
            T=MOU_T_CGM,
            f_rot=MOU_F_ROT,
            core_radius=args.cgm_core_radius,
        )
    )
    u0_cored = float(
        cgm_internal_energy(
            np.zeros((1, 3)),
            MOU_T_CGM,
            MOU_F_ROT,
            args.cgm_core_radius,
        )[0]
    )
    t0_cored = MOU_T_CGM * u0_cored / internal_energy_from_temperature(
        MOU_T_CGM
    )

    print("\n========== 基准检查 ==========")
    print(f"v_c,Mou(8.2 kpc) = {vc82:.3f} km/s")
    print(
        f"n_CGM-equivalent(R=29,z=0) = "
        f"{n29:.4e} cm^-3"
    )
    print(
        f"entropy-core center: n={n0_cored:.4e} cm^-3, "
        f"T={t0_cored:.4e} K"
    )
    print(
        f"disk mass budget: old-star analytic={stellar_disc_analytic_mass_code*1e10:.6e}, "
        f"ISM={ism_mass_msun:.6e}, target={MOU_M_DISC*1e10:.6e} Msun"
    )

    if not (215.0 <= vc82 <= 235.0):
        raise RuntimeError("Mou rotation curve 公式/单位异常。")

    if args.dry_run:
        print("--dry-run completed: HDF5 output intentionally skipped.")
        return

    if h5py is None:
        raise RuntimeError(
            "h5py is required to write the IC. Install h5py in the Python "
            "environment or use --dry-run for numerical validation only."
        )


    # -------------------------------------------------------------------------
    # 写 HDF5
    # -------------------------------------------------------------------------
    final_out = os.path.abspath(os.path.expanduser(args.outfile))
    box_center = np.array(
        [args.boxsize / 2.0] * 3,
        dtype=np.float64,
    )

    fd, out = tempfile.mkstemp(
        prefix=f".{os.path.basename(final_out)}.partial-",
        suffix=".hdf5",
        dir=os.path.dirname(final_out) or os.curdir,
    )
    os.close(fd)
    atexit.register(remove_file_if_exists, out)

    print(f"\n写入 {final_out} ...", flush=True)

    with h5py.File(out, "w") as f:
        header = f.create_group("Header")

        npart64 = np.array(
            [
                n_gas,
                n_dm,
                0,
                0,
                n_star,
                1 if include_bh else 0,
            ],
            dtype=np.uint64,
        )
        if np.any(npart64 > np.iinfo(np.uint32).max):
            raise OverflowError(
                "单文件 IC 的某一 PartType 超过 uint32 粒子计数上限；"
                "需要实现多文件 IC，不能把计数静默截断。"
            )
        npart = npart64.astype(np.uint32)

        header.attrs["NumPart_ThisFile"] = npart
        header.attrs["NumPart_Total"] = npart
        header.attrs["NumPart_Total_HighWord"] = np.zeros(
            6,
            dtype=np.uint32,
        )
        header.attrs["MassTable"] = np.zeros(
            6,
            dtype=np.float64,
        )

        header.attrs["Time"] = np.float64(0.0)
        header.attrs["Redshift"] = np.float64(0.0)
        header.attrs["BoxSize"] = np.float64(args.boxsize)
        header.attrs["NumFilesPerSnapshot"] = np.int32(1)

        header.attrs["Omega0"] = np.float64(0.0)
        header.attrs["OmegaLambda"] = np.float64(0.0)
        header.attrs["HubbleParam"] = np.float64(1.0)

        header.attrs["Flag_Sfr"] = np.int32(1)
        header.attrs["Flag_Cooling"] = np.int32(1)
        header.attrs["Flag_StellarAge"] = np.int32(1)
        # GIZMO public io.c 将该字段写为 NUM_METAL_SPECIES，而不是布尔值。
        header.attrs["Flag_Metals"] = np.int32(11)
        header.attrs["Flag_Feedback"] = np.int32(0)
        header.attrs["Flag_DoublePrecision"] = np.int32(1)
        header.attrs["Flag_Entropy_ICs"] = np.int32(0)
        header.attrs["Flag_IC_Info"] = np.int32(0)

        write_ptype(
            f,
            0,
            gas_pos,
            gas_vel,
            gas_ids,
            gas_masses_code,
            box_center,
            internal_energy=gas_u,
            metallicity=gas_Z,
            write_chunk_size=args.write_chunk_size,
        )

        write_ptype(
            f,
            1,
            dm_pos,
            dm_vel,
            dm_ids,
            dm_particle_mass_msun / 1.0e10,
            box_center,
            write_chunk_size=args.write_chunk_size,
        )

        write_ptype(
            f,
            4,
            star_pos,
            star_vel,
            star_ids,
            star_masses_code,
            box_center,
            metallicity=star_Z,
            stellar_formation_time=old_star_formation_time_code,
            write_chunk_size=args.write_chunk_size,
        )

        if include_bh:
            write_ptype(
                f,
                5,
                bh_pos,
                bh_vel,
                np.array([MW_BH_ID], dtype=np.uint32),
                bh_mass_code,
                box_center,
                bh_mass=bh_mass_code,
                write_chunk_size=args.write_chunk_size,
            )

        # ---------------------------------------------------------------------
        # 保存完整基准参数，供检验脚本自动读取。
        # ---------------------------------------------------------------------
        ref = f.create_group("MouReference")

        ref.attrs["V_HALO_kms"] = MOU_V_HALO
        ref.attrs["D_HALO_kpc"] = MOU_D_HALO
        ref.attrs["DM_RMAX_kpc"] = float(args.rmax_dm)
        ref.attrs["DM_MASS_Msun"] = halo_mass_msun

        ref.attrs["M_DISC_ANALYTIC_Msun"] = (
            MOU_M_DISC * 1.0e10
        )
        ref.attrs["M_OLD_STELLAR_DISC_ANALYTIC_Msun"] = (
            stellar_disc_analytic_mass_code * 1.0e10
        )
        ref.attrs["A_DISC_kpc"] = MOU_A_DISC
        ref.attrs["B_DISC_kpc"] = MOU_B_DISC
        ref.attrs["DISC_RMAX_kpc"] = float(args.rmax_disc)
        ref.attrs["DISC_ZMAX_kpc"] = float(args.zmax_disc)
        ref.attrs["DISC_RETAINED_FRACTION"] = float(
            mn_table["retained_fraction"]
        )
        ref.attrs["DISC_LIVE_MASS_Msun"] = disc_mass_msun
        ref.attrs["DISC_Q_INIT"] = float(args.q_target)
        ref.attrs["DISC_SIGMA_R_CLIPPED_MASS_FRACTION"] = float(
            disc_kin["sigma_R_clipped_mass_fraction"]
        )
        ref.attrs["DISC_SIGMA_Z_CLIPPED_MASS_FRACTION"] = float(
            disc_kin["sigma_z_clipped_mass_fraction"]
        )
        ref.attrs["DISC_VPHI_CLIPPED_MASS_FRACTION"] = float(
            disc_kin["vphi_clipped_mass_fraction"]
        )
        ref.attrs["STAR_TOTAL_PARTICLES"] = int(n_star)
        ref.attrs["STAR_TOTAL_MASS_Msun"] = float(disc_mass_msun + bulge_mass_msun)
        ref.attrs["OLD_STAR_AGE_GYR"] = OLD_STAR_AGE_GYR
        ref.attrs["OLD_STAR_FORMATION_TIME_CODE"] = old_star_formation_time_code

        ref.attrs["ISM_PRESENT"] = int(include_ism)
        ref.attrs["ISM_MODEL"] = (
            "McMillan2017_HI_plus_H2_surface_density"
            if include_ism
            else "none_diagnostic_no_live_ISM"
        )
        ref.attrs["ISM_RMAX_kpc"] = (
            float(args.rmax_ism) if include_ism else 0.0
        )
        ref.attrs["ISM_ZMAX_SAMPLING_kpc"] = (
            float(args.zmax_ism) if include_ism else 0.0
        )
        ref.attrs["ISM_MASS_Msun"] = ism_mass_msun
        ref.attrs["ISM_T_INITIAL_K"] = (
            ISM_T_INITIAL if include_ism else 0.0
        )
        ref.attrs["ISM_MU_INITIAL"] = MU_ISM if include_ism else 0.0
        ref.attrs["ISM_Z_Zsun"] = ISM_Z_ZSUN if include_ism else 0.0
        ref.attrs["ISM_VPHI_CLIPPED_MASS_FRACTION"] = ism_vphi_clip_fraction
        ref.attrs["MCMILLAN_ISM_PARAMETERS_ARE_REFERENCE_ONLY"] = int(
            not include_ism
        )
        ref.attrs["MCM_HI_SIGMA0_Msun_pc2"] = MCM_HI_SIGMA0
        ref.attrs["MCM_HI_RD_kpc"] = MCM_HI_RD
        ref.attrs["MCM_HI_RM_kpc"] = MCM_HI_RM
        ref.attrs["MCM_HI_ZD_REFERENCE_kpc"] = MCM_HI_ZD
        ref.attrs["MCM_H2_SIGMA0_Msun_pc2"] = MCM_H2_SIGMA0
        ref.attrs["MCM_H2_RD_kpc"] = MCM_H2_RD
        ref.attrs["MCM_H2_RM_kpc"] = MCM_H2_RM
        ref.attrs["MCM_H2_ZD_REFERENCE_kpc"] = MCM_H2_ZD

        ref.attrs["M_BULGE_ANALYTIC_Msun"] = (
            MOU_M_BULGE * 1.0e10
        )
        ref.attrs["D_BULGE_kpc"] = MOU_D_BULGE
        ref.attrs["BULGE_RMAX_kpc"] = float(args.rmax_bulge)
        ref.attrs["BULGE_RETAINED_FRACTION"] = bulge_fraction
        ref.attrs["BULGE_LIVE_MASS_Msun"] = bulge_mass_msun

        ref.attrs["T_CGM_K"] = MOU_T_CGM
        ref.attrs["MU_CGM"] = MU_HOT
        ref.attrs["F_ROT"] = MOU_F_ROT
        ref.attrs["RHO0_mH_cm3"] = MOU_RHO0_MH_CM3
        ref.attrs["RHO0_DEFINITION"] = (
            "Mou envelope normalization extrapolated to Phi(0); the actual live "
            "center is capped by the hydrostatic entropy core"
        )
        ref.attrs["CGM_ENTROPY_CORE_EQUATORIAL_RADIUS_kpc"] = float(
            args.cgm_core_radius
        )
        ref.attrs["CGM_ACTUAL_CENTER_NH_cm3"] = n0_cored
        ref.attrs["CGM_ACTUAL_CENTER_T_K"] = t0_cored
        ref.attrs["CGM_Z_Zsun"] = MOU_CGM_Z_ZSUN
        ref.attrs["CGM_RMAX_kpc"] = float(args.rmax_cgm)
        ref.attrs["CGM_MASS_Msun"] = cgm_mass_msun

        ref.attrs["BH_IS_PROJECT_ADDITION"] = 1 if include_bh else 0
        ref.attrs["BH_MASS_Msun"] = (
            args.bh_mass_msun if include_bh else 0.0
        )

        # ---------------------------------------------------------------------
        # 记录 ID 规划。
        # ---------------------------------------------------------------------
        rid = f.create_group("ReservedParticleIDs")

        rid.attrs["MW_GAS_ID_START"] = gas_ids[0]
        rid.attrs["MW_GAS_ID_END"] = gas_ids[-1]
        rid.attrs["MW_ISM_PRESENT"] = int(include_ism)
        rid.attrs["MW_ISM_COUNT"] = int(n_ism)
        # Zero is a sentinel and is never assigned to a particle in this IC.
        rid.attrs["MW_ISM_ID_START"] = (
            ism_ids[0] if include_ism else np.uint64(0)
        )
        rid.attrs["MW_ISM_ID_END"] = (
            ism_ids[-1] if include_ism else np.uint64(0)
        )
        rid.attrs["MW_CGM_ID_START"] = cgm_ids[0]
        rid.attrs["MW_CGM_ID_END"] = cgm_ids[-1]

        rid.attrs["MW_DM_ID_START"] = MW_DM_ID_START
        rid.attrs["MW_DM_ID_END"] = dm_ids[-1]

        rid.attrs["MW_DISC_ID_START"] = MW_DISC_ID_START
        rid.attrs["MW_DISC_ID_END"] = disc_ids[-1]

        rid.attrs["MW_BULGE_ID_START"] = MW_BULGE_ID_START
        rid.attrs["MW_BULGE_ID_END"] = bulge_ids[-1]

        rid.attrs["MW_BH_ID"] = MW_BH_ID

        rid.attrs["LMC_ID_START"] = LMC_ID_START
        rid.attrs["LMC_ID_END"] = LMC_ID_END

        rid.attrs["SMC_ID_START"] = SMC_ID_START
        rid.attrs["SMC_ID_END"] = SMC_ID_END

        rid.attrs["FUTURE_ID_START"] = FUTURE_ID_START

        notes = f.create_group("ModelNotes")
        notes.attrs["MODEL_KIND"] = (
            "Mou2023 live total-potential target; "
            + (
                "McMillan2017 sub-grid ISM enabled; "
                if include_ism
                else "no live cold/warm ISM diagnostic; "
            )
            + "old stellar disk and bulge stored together as PartType4"
        )
        notes.attrs["MW_COLD_WARM_ISM"] = int(include_ism)
        notes.attrs["MW_STAR_FORMATION_REQUIRED"] = int(include_ism)
        notes.attrs["MW_EFFECTIVE_EQS_REQUIRED"] = int(include_ism)
        notes.attrs["NO_ISM_SINGLE_VARIABLE_CONTROL"] = int(not include_ism)
        notes.attrs["NO_ISM_RUNTIME_COMPARISON_NOTE"] = (
            "For the first differential test, keep the same GIZMO compile-time "
            "and parameter-file physics as the v4 ISM run; change only the IC."
        )
        notes.attrs["MW_STELLAR_FEEDBACK_REQUIRED"] = 0
        notes.attrs["COOL_LOW_TEMPERATURES_REQUIRED"] = 0
        notes.attrs["GIZMO_SOURCE_EXTERNAL_MW_POTENTIAL_REQUIRED"] = 0
        notes.attrs["ISOLATED_RELAXATION_REQUIRED"] = 1
        notes.attrs["COLLISIONLESS_VELOCITY_MODEL"] = (
            "approximate Jeans moments with local Gaussian sampling; not an exact DF"
        )
        notes.attrs["DM_OUTER_BOUNDARY_MODEL"] = (
            "sharp truncation with zero-pressure Jeans boundary; no artificial "
            "external confining pressure"
        )
        notes.attrs["CGM_SELF_GRAVITY_NOTE"] = (
            "GIZMO live gas self-gravity is present unless disabled; "
            "Mou original fixed-potential gas did not define a live CGM gravity term."
        )
        notes.attrs["PUBLIC_GIZMO_RESTARTFLAG0_IO_NOTE"] = (
            "public read_ic.c skips ordinary IC blocks after InternalEnergy; "
            "verify from the first output that Metallicity, StellarFormationTime, "
            "and BH_Mass were consumed by the exact compiled commit"
        )
        notes.attrs["REQUIRED_InitGasTemp_K"] = 0.0
        notes.attrs["TARGET_InitMetallicity_Zsun"] = MOU_CGM_Z_ZSUN
        notes.attrs["InitMetallicity_APPLIES_IF"] = (
            "public GIZMO build defines INIT_STELLAR_METALS_AGES_DEFINED; "
            "e.g. COOL_METAL_LINES_BY_SPECIES or GALSF+METALS"
        )
        notes.attrs["REQUIRED_SeedBlackHoleMass_code"] = (
            bh_mass_code if include_bh else 0.0
        )

    # -------------------------------------------------------------------------
    # 分块最终检查；确认完整后才原子替换目标文件。
    # -------------------------------------------------------------------------
    expected_counts = {
        0: n_gas,
        1: n_dm,
        4: n_star,
    }
    if include_bh:
        expected_counts[5] = 1

    validate_ic_file(
        out,
        expected_counts=expected_counts,
        boxsize=args.boxsize,
        chunk_size=args.write_chunk_size,
    )
    # 避免在长时间生成期间由其他进程新建的目标被意外覆盖。
    if os.path.exists(final_out) and not args.overwrite:
        raise FileExistsError(
            f"输出文件在生成期间已出现：{final_out}；"
            "本次保留现有目标，并清理本进程的临时 IC。"
        )
    os.replace(out, final_out)
    atexit.unregister(remove_file_if_exists)
    out = final_out

    print("\n========== 完成 ==========")
    print(f"输出：{out}")
    print(
        "下一步不要直接接 LMC/SMC；先做 isolated run，"
        "用质量/力场/剖面/质心/能量与角动量的收敛判据选择 relaxed host。"
    )
    print(
        f"LMC ID: {int(LMC_ID_START):,} - {int(LMC_ID_END):,}"
    )
    print(
        f"SMC ID: {int(SMC_ID_START):,} - {int(SMC_ID_END):,}"
    )
    print(
        "GIZMO public 版普通 IC 启动需配套：InitGasTemp=0；"
        f"若编译定义 INIT_STELLAR_METALS_AGES_DEFINED，"
        f"则 InitMetallicity={MOU_CGM_Z_ZSUN:g}；"
        + (
            f"SeedBlackHoleMass={bh_mass_code:.8g}。"
            if include_bh
            else "本次未包含 BH。"
        )
    )


if __name__ == "__main__":
    main()
