#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1：GIZMO 用 live Milky-Way 恒星盘弛豫 IC
================================================

本脚本生成一个不含气体的孤立银河初始条件：

    live tapered-NFW 暗物质晕
    + 指数–sech^2 恒星盘
    + Hernquist 恒星核球
    + 可选中心黑洞

用途是先检验“暗晕＋恒星盘＋核球”本身能否在 GIZMO 中稳定演化。
本阶段故意不加入 MW CGM、ISM、LMC/SMC 或 AGN jet；只有这个宿主通过
弛豫检查后，才应从通过检验的 snapshot 构造 Stage 2（CGM＋LMC/SMC），
再从 Stage 2 的对照结果构造 Stage 3（AGN jet）。

-------------------------------------------------------------------------------
一、参数与方法的文献出处（也是文件内参数 provenance）
-------------------------------------------------------------------------------

[R24] Roy et al. 2024, MNRAS, 527, 265–280,
      “Seeding the CGM: how satellites populate the cold phase of
       Milky Way haloes”, arXiv:2310.04404.
      https://arxiv.org/abs/2310.04404
      相关内容：GIZMO-MFM；球对称各向同性 NFW 暗晕、Hernquist 核球、
      指数旋转恒星/气体盘；恒星盘 M=5e10 Msun、Rd=3 kpc；盘初始
      Toomre Q 约为 1；加入卫星前先做孤立弛豫。

[S19] Su et al. 2019, MNRAS, 487, 4393–4408,
      “The failure of stellar feedback, magnetic fields, conduction, and
       morphological quenching in maintaining red galaxies”,
      arXiv:1809.09120.
      https://arxiv.org/abs/1809.09120
      相关内容：R24 所用 m12 IC 的母模型；NFW 暗晕、Hernquist 核球、
      指数盘的分量形式；m12 表中恒星盘 M=5e10 Msun、Rd=3 kpc，
      核球 M=1.5e10 Msun、a=1 kpc。

[M17] McMillan 2017, MNRAS, 465, 76–94,
      “The mass distribution and gravitational potential of the Milky Way”,
      arXiv:1608.00971.
      https://arxiv.org/abs/1608.00971
      相关内容：R0=8.20+-0.09 kpc、v_c(R0)=232.8+-3.0 km/s、
      M_vir=(1.30+-0.30)e12 Msun、总恒星质量=(5.43+-0.57)e10 Msun。

[C25] Carr et al. 2025, ApJ, 983, 151,
      “The All-Sky Impact of the LMC on the Milky Way Circumgalactic Medium”,
      arXiv:2408.10358.
      https://arxiv.org/abs/2408.10358
      相关内容：后续 LMC/CGM 阶段采用较轻的 MW（约 1.1e12 Msun）和
      live halo；本脚本只借它约束宿主质量量级，不包含 LMC 或 CGM。

[H90] Hernquist 1990, ApJ, 356, 359.
      https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H
      相关内容：核球密度、累积质量与势的解析形式。

[SW99] Springel & White 1999, MNRAS, 307, 162–178.
       https://ui.adsabs.harvard.edu/abs/1999MNRAS.307..162S
       相关内容：多分量盘星系的近似平衡构造；R24/S19 也引用这一类方法。

-------------------------------------------------------------------------------
二、本脚本采用的“文献综合基线”（不是任一论文的逐项原样复制）
-------------------------------------------------------------------------------

暗晕：M200=1.2e12 Msun，c=11；由 H0=70 km/s/Mpc 定义 R200。
      M200 位于 M17 的范围内，也接近 C25 的 LMC 模拟宿主。
      R200 外接连续对数斜率的指数 taper，默认在 350 kpc 截止；这是
      live 粒子 realization 的数值边界，不是额外的观测约束。

恒星盘：Mdisk=5e10 Msun，Rd=3 kpc（R24/S19）；zd=0.3 kpc（薄盘尺度
        的简化选择）；Q=1.5 沿用本项目此前较稳健的弛豫基线，而不是
        直接使用 R24 的 Q~1。

核球：Hernquist，Mbulge=1.5e10 Msun，a=1 kpc（R24/S19）。

黑洞：4e6 Msun，沿用本项目后续 AGN 实验的 Sgr A* 质量基线；
      Stage 1 必须关闭其吸积与反馈。

用薄指数盘的中面圆周速度公式估计，上述组合在 R=8.2 kpc 给出
v_c 约 233 km/s，目标是与 M17 对齐。实际 live 粒子力场必须再用
GIZMO EVALPOTENTIAL 输出验证。

-------------------------------------------------------------------------------
三、必须知道的数值限制
-------------------------------------------------------------------------------

1. 本脚本用 Jeans 二阶矩＋局部截断高斯生成暗晕/核球速度；盘速度采用
   Toomre-Q、epicyclic approximation、asymmetric drift 和近似垂直平衡。
   它不是精确分布函数（DF），因此必须弛豫，不能把 t=0 当生产模型。
2. 盘的中面引力使用 razor-thin exponential-disc 公式；垂直方向使用
   sech^2 自引力片＋球对称分量的局部近似。初始盘厚度仍可能小幅调整。
3. 暗晕 Jeans 方程中把盘的累积质量作球对称近似，仅用于赋初速度；
   GIZMO 演化时所有分量的引力都是 live 粒子引力，不存在固定外势。
4. 使用 quiet-start 反演粒子对降低质心、净动量和初始 m=1/m=2 采样噪声。
5. 为兼容本项目既有分析约定，恒星盘和核球都写入 PartType4；二者用
   ParticleID 范围以及额外的 StellarComponent（1=盘、2=核球）区分。

默认粒子类型
------------
PartType0 : 空（本阶段无气体）
PartType1 : MW live DM halo
PartType4 : 老恒星盘＋老核球
PartType5 : 中心 BH（默认包含，但弛豫时反馈关闭）

生成示例
--------
python make_ic_live_mw_disk_stage1_v1.py

快速低分辨率测试
----------------
python make_ic_live_mw_disk_stage1_v1.py \
    --dm-particle-mass 1e9 --star-particle-mass 1e8 \
    --output mw_live_disk_stage1_quick.hdf5

正式默认分辨率
--------------
DM 粒子目标质量       = 5e6 Msun
恒星盘/核球目标质量   = 5e5 Msun

这里把上一版的 1e7/1e6 Msun 下调了一倍：后者仍可用作低分辨率筛选，
但对于约 0.5--1 Gyr 的 live 盘弛豫，5e6/5e5 Msun 能降低重 DM 粒子造成的
人工盘加热。后续加入 CGM 时，建议 m_gas=1e5 Msun；Stage 1 本身没有气体。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.integrate import cumulative_trapezoid
    from scipy.special import iv, kv
except ImportError as exc:
    raise ImportError(
        "本脚本需要 scipy（integrate.cumulative_trapezoid 和 special.iv/kv）。"
    ) from exc


# =============================================================================
# 0. 单位、默认参数与 ParticleID 规划
# =============================================================================

G_KPC_KMS2_MSUN = 4.30091e-6

UNIT_LENGTH_CM = 3.0856775814913673e21   # 1 kpc
UNIT_MASS_G = 1.989e43                   # 1e10 Msun
UNIT_VELOCITY_CMS = 1.0e5                # 1 km/s
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
UNIT_TIME_GYR = UNIT_LENGTH_CM / UNIT_VELOCITY_CMS / SECONDS_PER_GYR

SEED_DEFAULT = 42
BOXSIZE_DEFAULT = 1000.0

# 文献综合基线，出处见文件头和输出文件 MWModelReference。
M200_DEFAULT = 1.20e12
CONCENTRATION_DEFAULT = 11.0
H0_DEFAULT = 70.0                       # km/s/Mpc
HALO_TAPER_DECAY_FRAC_DEFAULT = 0.20
RMAX_HALO_DEFAULT = 350.0

M_DISK_DEFAULT = 5.0e10
R_DISK_DEFAULT = 3.0
Z_DISK_DEFAULT = 0.30
RMAX_DISK_DEFAULT = 30.0
ZMAX_DISK_DEFAULT = 3.0
Q_DEFAULT = 1.50

M_BULGE_DEFAULT = 1.5e10
A_BULGE_DEFAULT = 1.0
RMAX_BULGE_DEFAULT = 300.0

M_BH_DEFAULT = 4.0e6
R_SOLAR_DEFAULT = 8.20
OLD_STAR_AGE_GYR_DEFAULT = 10.0

TARGET_DM_MASS_DEFAULT = 5.0e6
TARGET_STAR_MASS_DEFAULT = 5.0e5

# 保持与此前 Stage-2 插入流程兼容的 ID 大区间。
MW_DM_ID_START = np.uint64(10_000_000)
MW_DISK_ID_START = np.uint64(20_000_000)
MW_BULGE_ID_START = np.uint64(30_000_000)
MW_BH_ID = np.uint64(90_000_000)
LMC_ID_START = np.uint64(100_000_000)
LMC_ID_END = np.uint64(199_999_999)
SMC_ID_START = np.uint64(200_000_000)
SMC_ID_END = np.uint64(299_999_999)
FUTURE_ID_START = np.uint64(300_000_000)

Z_SUN_MASS_FRACTION = 0.02
SOLAR_METAL_MASS = np.array(
    [0.00326, 0.00132, 0.00865, 0.00222, 0.000931,
     0.00108, 0.000644, 0.000101, 0.00173],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Model:
    """所有物理和数值参数的不可变集合。质量单位 Msun，长度 kpc。"""

    m200: float
    concentration: float
    h0: float
    halo_taper_decay_fraction: float
    rmax_halo: float

    disk_mass: float
    disk_rd: float
    disk_zd: float
    rmax_disk: float
    zmax_disk: float
    q_target: float

    bulge_mass: float
    bulge_a: float
    rmax_bulge: float

    bh_mass: float
    include_bh: bool
    r_solar: float
    old_star_age_gyr: float

    box_size: float
    dm_particle_mass: float
    star_particle_mass: float

    @property
    def h0_kms_kpc(self) -> float:
        return self.h0 / 1000.0

    @property
    def r200(self) -> float:
        # M200=(4*pi/3)*200*rho_crit*R200^3，rho_crit=3H0^2/(8*pi*G)
        return (
            G_KPC_KMS2_MSUN * self.m200
            / (100.0 * self.h0_kms_kpc**2)
        ) ** (1.0 / 3.0)

    @property
    def halo_rs(self) -> float:
        return self.r200 / self.concentration

    @property
    def halo_taper_decay(self) -> float:
        return self.halo_taper_decay_fraction * self.r200


# =============================================================================
# 1. 基础解析函数
# =============================================================================

def nfw_f(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x) - x / (1.0 + x)


def hernquist_mass_enclosed(r, mass, a):
    r = np.asarray(r, dtype=np.float64)
    return mass * r**2 / (r + a) ** 2


def exponential_disk_mass_inside_cylinder(r, mass, rd):
    """指数盘在柱半径 R 内的质量；在球对称 Jeans 近似中作 monopole。"""
    x = np.asarray(r, dtype=np.float64) / rd
    return mass * (1.0 - (1.0 + x) * np.exp(-x))


def thin_exponential_disk_vc2(r, mass, rd):
    """razor-thin exponential disc 的中面 v_c^2（Freeman 1970 公式）。"""
    r = np.asarray(r, dtype=np.float64)
    y = np.maximum(r / (2.0 * rd), 1.0e-8)
    sigma0 = mass / (2.0 * np.pi * rd**2)
    bessel = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
    vc2 = 4.0 * np.pi * G_KPC_KMS2_MSUN * sigma0 * rd * y**2 * bessel
    return np.maximum(vc2, 0.0)


def stellar_metallicity_vector(z_zsun=1.0):
    """生成 GIZMO/FIRE 常用的 11 维金属质量分数向量。"""
    out = np.zeros(11, dtype=np.float64)
    out[0] = z_zsun * Z_SUN_MASS_FRACTION
    out[1] = 0.24 + (0.28 - 0.24) * z_zsun
    out[2:] = z_zsun * SOLAR_METAL_MASS
    return out


def even_count_from_target(total_mass, target_particle_mass):
    """取最接近目标分辨率的偶数粒子数，便于 quiet-start 成对采样。"""
    n = max(2, int(round(total_mass / target_particle_mass)))
    if n % 2:
        n += 1
    return n


# =============================================================================
# 2. tapered NFW halo：密度、累积质量与抽样
# =============================================================================

def halo_density_shape(r, model: Model):
    """
    R200 内为 NFW；R200 外使用连续对数斜率的指数 taper。

    外侧形式：rho=rho(R200)*(r/R200)^epsilon*exp[-(r-R200)/r_decay]
    epsilon 的选择使 d ln rho / d ln r 在 R200 连续。
    """
    r = np.asarray(r, dtype=np.float64)
    safe_r = np.maximum(r, model.halo_rs * 1.0e-12)
    x = safe_r / model.halo_rs
    rho_nfw = 1.0 / (x * (1.0 + x) ** 2)

    c = model.concentration
    slope_at_r200 = -1.0 - 2.0 * c / (1.0 + c)
    epsilon = slope_at_r200 + model.r200 / model.halo_taper_decay
    rho_at_r200 = 1.0 / (c * (1.0 + c) ** 2)
    outer = (
        rho_at_r200
        * (safe_r / model.r200) ** epsilon
        * np.exp(-(safe_r - model.r200) / model.halo_taper_decay)
    )
    return np.where(safe_r <= model.r200, rho_nfw, outer)


def build_halo_table(model: Model, n_grid=50000):
    """建立 halo 的数值 CDF；归一化严格满足 M(<R200)=M200。"""
    rmin = model.halo_rs * 1.0e-7
    r = np.concatenate(
        ([0.0], np.geomspace(rmin, model.rmax_halo, int(n_grid)))
    )
    rho_shape = np.zeros_like(r)
    rho_shape[1:] = halo_density_shape(r[1:], model)
    integrand = 4.0 * np.pi * r**2 * rho_shape
    raw_mass = cumulative_trapezoid(integrand, r, initial=0.0)
    raw_m200 = np.interp(model.r200, r, raw_mass)
    normalization = model.m200 / raw_m200
    mass = raw_mass * normalization
    rho = rho_shape * normalization
    return {
        "r": r,
        "rho": rho,
        "mass": mass,
        "total_mass": float(mass[-1]),
        "normalization": float(normalization),
    }


def halo_mass_enclosed(r, halo_table):
    r = np.asarray(r, dtype=np.float64)
    return np.interp(
        np.clip(r, 0.0, halo_table["r"][-1]),
        halo_table["r"],
        halo_table["mass"],
    )


def sample_halo_position_pairs(rng, n, halo_table):
    """按 halo CDF 抽样，并加入相反位置的 quiet-start 粒子。"""
    if n % 2:
        raise ValueError("quiet-start halo 粒子数必须是偶数。")
    nh = n // 2
    target_mass = rng.random(nh) * halo_table["total_mass"]
    r = np.interp(target_mass, halo_table["mass"], halo_table["r"])
    mu = rng.uniform(-1.0, 1.0, nh)
    phi = rng.uniform(0.0, 2.0 * np.pi, nh)
    sint = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    pos_half = np.column_stack(
        (r * sint * np.cos(phi), r * sint * np.sin(phi), r * mu)
    )
    return np.vstack((pos_half, -pos_half)), np.concatenate((r, r))


# =============================================================================
# 3. 球对称分量的 Jeans 初速度
# =============================================================================

def spherical_total_enclosed_mass(r, model: Model, halo_table):
    """仅为 Jeans 初速度使用的球平均总累积质量。"""
    total = halo_mass_enclosed(r, halo_table)
    total += hernquist_mass_enclosed(r, model.bulge_mass, model.bulge_a)
    total += exponential_disk_mass_inside_cylinder(r, model.disk_mass, model.disk_rd)
    if model.include_bh:
        total += model.bh_mass
    return total


def build_isotropic_jeans_table(r, rho, model: Model, halo_table):
    """
    解 sigma_r^2(r)=rho^-1 * integral_r^rmax rho(s)*g(s) ds。
    外边界采用零压边界；因此外侧少量粒子会在弛豫中调整。
    """
    r = np.asarray(r, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if np.any(r <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("Jeans 网格要求 r>0 且 rho>0。")

    menc = spherical_total_enclosed_mass(r, model, halo_table)
    gravity = G_KPC_KMS2_MSUN * menc / r**2
    source = rho * gravity
    integral = -cumulative_trapezoid(
        source[::-1], r[::-1], initial=0.0
    )[::-1]
    sigma2 = np.maximum(integral / rho, 0.0)

    # 计算近似逃逸速度，用于去掉局部高斯的非物理解绑定尾部。
    potential_drop = -cumulative_trapezoid(
        gravity[::-1], r[::-1], initial=0.0
    )[::-1]
    outer_mass = float(spherical_total_enclosed_mass(r[-1], model, halo_table))
    phi_to_infinity = potential_drop + G_KPC_KMS2_MSUN * outer_mass / r[-1]
    vesc = np.sqrt(np.maximum(2.0 * phi_to_infinity, 0.0))

    return {
        "r": r,
        "sigma": np.sqrt(sigma2),
        "vesc": vesc,
    }


def sample_truncated_gaussian_velocities(rng, sigma, vmax):
    """各向同性局部高斯，并拒绝 v>vmax 的尾部。"""
    sigma = np.asarray(sigma, dtype=np.float64)
    vmax = np.asarray(vmax, dtype=np.float64)
    vel = rng.normal(size=(len(sigma), 3)) * sigma[:, None]
    bad = np.sum(vel**2, axis=1) >= vmax**2
    attempts = 0
    while np.any(bad) and attempts < 100:
        vel[bad] = (
            rng.normal(size=(int(np.count_nonzero(bad)), 3))
            * sigma[bad, None]
        )
        bad = np.sum(vel**2, axis=1) >= vmax**2
        attempts += 1
    if np.any(bad):
        speed = np.linalg.norm(vel[bad], axis=1)
        vel[bad] *= (0.999 * vmax[bad] / np.maximum(speed, 1.0e-30))[:, None]
    return vel


def assign_spherical_velocity_pairs(rng, radii_full, jeans_table):
    """先生成一半速度，再给镜像粒子赋相反速度，保证净动量严格为零。"""
    n = len(radii_full)
    if n % 2:
        raise ValueError("quiet-start 速度赋值要求偶数粒子。")
    nh = n // 2
    r_half = radii_full[:nh]
    sigma = np.interp(r_half, jeans_table["r"], jeans_table["sigma"])
    vesc = np.interp(r_half, jeans_table["r"], jeans_table["vesc"])
    vel_half = sample_truncated_gaussian_velocities(
        rng, sigma, 0.995 * vesc
    )
    return np.vstack((vel_half, -vel_half))


# =============================================================================
# 4. 指数–sech^2 恒星盘的位置与近似 Jeans/epicyclic 速度
# =============================================================================

def disk_retained_fraction(model: Model):
    radial = 1.0 - (
        1.0 + model.rmax_disk / model.disk_rd
    ) * np.exp(-model.rmax_disk / model.disk_rd)
    vertical = np.tanh(model.zmax_disk / model.disk_zd)
    return float(radial * vertical)


def sample_disk_position_pairs(rng, n, model: Model):
    if n % 2:
        raise ValueError("quiet-start disk 粒子数必须是偶数。")
    nh = n // 2

    r_grid = np.linspace(0.0, model.rmax_disk, 20001)
    x = r_grid / model.disk_rd
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    cdf /= cdf[-1]
    radius = np.interp(rng.random(nh), cdf, r_grid)
    phi = rng.uniform(0.0, 2.0 * np.pi, nh)

    tanh_max = np.tanh(model.zmax_disk / model.disk_zd)
    z = model.disk_zd * np.arctanh(
        rng.uniform(-tanh_max, tanh_max, nh)
    )
    pos_half = np.column_stack(
        (radius * np.cos(phi), radius * np.sin(phi), z)
    )
    return np.vstack((pos_half, -pos_half)), radius, phi


def total_midplane_vc2(r, model: Model, halo_table):
    r = np.asarray(r, dtype=np.float64)
    safe_r = np.maximum(r, 1.0e-7)
    halo = G_KPC_KMS2_MSUN * halo_mass_enclosed(safe_r, halo_table) / safe_r
    bulge = (
        G_KPC_KMS2_MSUN * model.bulge_mass * safe_r
        / (safe_r + model.bulge_a) ** 2
    )
    disk = thin_exponential_disk_vc2(safe_r, model.disk_mass, model.disk_rd)
    bh = (
        G_KPC_KMS2_MSUN * model.bh_mass / safe_r
        if model.include_bh
        else 0.0
    )
    return np.maximum(halo + bulge + disk + bh, 0.0)


def spherical_external_vertical_sigma2(r, model: Model, halo_table):
    """
    对 sech^2 盘在每个 R 数值积分球对称外势的垂直 Jeans 支撑。

    使用
        sigma_z,ext^2(R,0) = integral_0^zmax sech^2(z/zd) g_z(R,z) dz

    相比简单写成 nu^2(R)*zd^2，这一积分在 R 接近 0、zd 不再远小于 R
    时不会人为产生过大的中心垂直速度弥散。
    """
    r = np.asarray(r, dtype=np.float64)
    zmin = model.disk_zd * 1.0e-6
    z = np.concatenate(
        ([0.0], np.geomspace(zmin, model.zmax_disk, 384))
    )
    vertical_weight = 1.0 / np.cosh(z / model.disk_zd) ** 2
    out = np.empty_like(r)

    # 分块避免一次性构造很大的二维数组。
    chunk = 512
    for i0 in range(0, len(r), chunk):
        rr = r[i0:i0 + chunk, None]
        zz = z[None, :]
        spherical_radius = np.sqrt(rr**2 + zz**2)
        safe_s = np.maximum(spherical_radius, 1.0e-10)
        enclosed = (
            halo_mass_enclosed(safe_s, halo_table)
            + hernquist_mass_enclosed(
                safe_s, model.bulge_mass, model.bulge_a
            )
            + (model.bh_mass if model.include_bh else 0.0)
        )
        gz = G_KPC_KMS2_MSUN * enclosed * zz / safe_s**3
        out[i0:i0 + chunk] = np.trapezoid(
            gz * vertical_weight[None, :], z, axis=1
        )
    return np.maximum(out, 0.0)


def build_disk_kinematic_table(model: Model, halo_table, n_grid=12000):
    """构造 Q、epicyclic、asymmetric-drift 与垂直速度弥散表。"""
    r = np.geomspace(1.0e-3, model.rmax_disk, int(n_grid))
    vc2 = total_midplane_vc2(r, model, halo_table)
    vc = np.sqrt(vc2)
    omega2 = vc2 / r**2
    d_omega2_dr = np.gradient(omega2, r, edge_order=2)
    kappa2 = np.maximum(r * d_omega2_dr + 4.0 * omega2, 1.0e-12)
    kappa = np.sqrt(kappa2)

    sigma_surface = (
        model.disk_mass / (2.0 * np.pi * model.disk_rd**2)
        * np.exp(-r / model.disk_rd)
    )
    sigma_r = (
        model.q_target * 3.36 * G_KPC_KMS2_MSUN * sigma_surface / kappa
    )
    # 很外侧 Σ 极小，给数值上过冷的尾部一个温和下限。
    sigma_r = np.maximum(sigma_r, 5.0)

    sigma_phi = sigma_r * kappa / np.maximum(2.0 * np.sqrt(omega2), 1.0e-20)
    sigma_phi = np.minimum(sigma_phi, sigma_r)

    # sech^2 自引力片：sigma_z,self^2≈pi G Sigma zd；球对称分量则
    # 直接积分垂直 Jeans 方程，避免中心 R<<zd 时的局部谐振近似失效。
    sigma_z_external2 = spherical_external_vertical_sigma2(
        r, model, halo_table
    )
    sigma_z2 = (
        np.pi * G_KPC_KMS2_MSUN * sigma_surface * model.disk_zd
        + sigma_z_external2
    )
    sigma_z = np.sqrt(np.maximum(sigma_z2, 1.0))

    log_nu_sigma_r2 = -r / model.disk_rd + 2.0 * np.log(sigma_r)
    dln_nu_sigma_r2_dlnr = np.gradient(
        log_nu_sigma_r2, np.log(r), edge_order=2
    )
    mean_vphi2 = (
        vc2
        + sigma_r**2 * (1.0 + dln_nu_sigma_r2_dlnr)
        - sigma_phi**2
    )
    clipped = mean_vphi2 < 0.0
    mean_vphi = np.sqrt(np.maximum(mean_vphi2, 0.0))

    return {
        "r": r,
        "vc": vc,
        "sigma_r": sigma_r,
        "sigma_phi": sigma_phi,
        "sigma_z": sigma_z,
        "mean_vphi": mean_vphi,
        "mean_vphi_clipped_fraction_grid": float(np.mean(clipped)),
    }


def assign_disk_velocity_pairs(rng, radius_half, phi_half, kin_table):
    sr = np.interp(radius_half, kin_table["r"], kin_table["sigma_r"])
    sp = np.interp(radius_half, kin_table["r"], kin_table["sigma_phi"])
    sz = np.interp(radius_half, kin_table["r"], kin_table["sigma_z"])
    vbar = np.interp(radius_half, kin_table["r"], kin_table["mean_vphi"])

    vr = rng.normal(0.0, sr)
    vphi = rng.normal(vbar, sp)
    vz = rng.normal(0.0, sz)

    cosphi = np.cos(phi_half)
    sinphi = np.sin(phi_half)
    vx = vr * cosphi - vphi * sinphi
    vy = vr * sinphi + vphi * cosphi
    vel_half = np.column_stack((vx, vy, vz))
    return np.vstack((vel_half, -vel_half))


# =============================================================================
# 5. Hernquist 核球位置和速度
# =============================================================================

def bulge_retained_fraction(model: Model):
    x = model.rmax_bulge / (model.rmax_bulge + model.bulge_a)
    return float(x**2)


def sample_bulge_position_pairs(rng, n, model: Model):
    if n % 2:
        raise ValueError("quiet-start bulge 粒子数必须是偶数。")
    nh = n // 2
    fmax = bulge_retained_fraction(model)
    y = np.sqrt(rng.random(nh) * fmax)
    r = model.bulge_a * y / np.maximum(1.0 - y, 1.0e-14)
    mu = rng.uniform(-1.0, 1.0, nh)
    phi = rng.uniform(0.0, 2.0 * np.pi, nh)
    sint = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    pos_half = np.column_stack(
        (r * sint * np.cos(phi), r * sint * np.sin(phi), r * mu)
    )
    return np.vstack((pos_half, -pos_half)), np.concatenate((r, r))


def bulge_density_shape(r, model: Model):
    r = np.asarray(r, dtype=np.float64)
    return 1.0 / (
        np.maximum(r, model.bulge_a * 1.0e-12)
        * (r + model.bulge_a) ** 3
    )


# =============================================================================
# 6. HDF5 写入与验证
# =============================================================================

def require_h5py():
    if h5py is None:
        raise RuntimeError(
            "写 HDF5 需要 h5py；请在运行环境安装：python -m pip install h5py"
        )


def write_dataset(group, name, data):
    group.create_dataset(name, data=np.asarray(data))


def write_particle_group(
    handle,
    ptype,
    positions,
    velocities,
    ids,
    masses_msun,
    box_center,
    metallicity=None,
    formation_time=None,
    stellar_component=None,
    bh_mass_msun=None,
):
    group = handle.create_group(f"PartType{ptype}")
    write_dataset(group, "Coordinates", positions + box_center[None, :])
    write_dataset(group, "Velocities", velocities)
    write_dataset(group, "ParticleIDs", np.asarray(ids, dtype=np.uint32))
    write_dataset(group, "Masses", np.asarray(masses_msun, dtype=np.float64) / 1.0e10)

    if metallicity is not None:
        metal = np.asarray(metallicity, dtype=np.float64)
        if metal.ndim == 1:
            metal = np.broadcast_to(metal, (len(positions), len(metal)))
        write_dataset(group, "Metallicity", metal)
    if formation_time is not None:
        write_dataset(
            group,
            "StellarFormationTime",
            np.full(len(positions), formation_time, dtype=np.float64),
        )
    if stellar_component is not None:
        write_dataset(
            group,
            "StellarComponent",
            np.asarray(stellar_component, dtype=np.uint8),
        )
    if bh_mass_msun is not None:
        write_dataset(
            group,
            "BH_Mass",
            np.asarray(bh_mass_msun, dtype=np.float64) / 1.0e10,
        )


def write_reference_groups(handle, model: Model, halo_table, diagnostics, counts):
    ref = handle.create_group("MWModelReference")
    ref.attrs["MODEL_NAME"] = "Literature_hybrid_live_MW_stage1_v1"
    ref.attrs["MODEL_IS_EXACT_COPY_OF_ONE_PAPER"] = 0
    ref.attrs["MODEL_SCOPE"] = (
        "collisionless host relaxation only; no gas, LMC, SMC, or AGN jet"
    )
    ref.attrs["REFERENCE_ROY_2024"] = (
        "Roy et al. 2024, MNRAS 527, 265; arXiv:2310.04404"
    )
    ref.attrs["REFERENCE_SU_2019"] = (
        "Su et al. 2019, MNRAS 487, 4393; arXiv:1809.09120"
    )
    ref.attrs["REFERENCE_MCMILLAN_2017"] = (
        "McMillan 2017, MNRAS 465, 76; arXiv:1608.00971"
    )
    ref.attrs["REFERENCE_CARR_2025"] = (
        "Carr et al. 2025, ApJ 983, 151; arXiv:2408.10358"
    )
    ref.attrs["REFERENCE_HERNQUIST_1990"] = "Hernquist 1990, ApJ 356, 359"
    ref.attrs["REFERENCE_SPRINGEL_WHITE_1999"] = (
        "Springel & White 1999, MNRAS 307, 162"
    )

    ref.attrs["H0_km_s_Mpc"] = model.h0
    ref.attrs["M200_Msun"] = model.m200
    ref.attrs["R200_kpc"] = model.r200
    ref.attrs["HALO_CONCENTRATION"] = model.concentration
    ref.attrs["HALO_RS_kpc"] = model.halo_rs
    ref.attrs["HALO_TAPER_START_kpc"] = model.r200
    ref.attrs["HALO_TAPER_DECAY_kpc"] = model.halo_taper_decay
    ref.attrs["HALO_RMAX_kpc"] = model.rmax_halo
    ref.attrs["HALO_LIVE_MASS_Msun"] = halo_table["total_mass"]
    ref.attrs["HALO_PARAMETER_PROVENANCE"] = (
        "M200 chosen within McMillan2017 range and near Carr2025 host; "
        "NFW live-halo architecture follows Su2019/Roy2024; c=11 is project baseline"
    )

    ref.attrs["DISK_ANALYTIC_MASS_Msun"] = model.disk_mass
    ref.attrs["DISK_LIVE_MASS_Msun"] = diagnostics["disk_live_mass"]
    ref.attrs["DISK_RD_kpc"] = model.disk_rd
    ref.attrs["DISK_ZD_kpc"] = model.disk_zd
    ref.attrs["DISK_RMAX_kpc"] = model.rmax_disk
    ref.attrs["DISK_ZMAX_kpc"] = model.zmax_disk
    ref.attrs["DISK_RETAINED_FRACTION"] = diagnostics["disk_retained_fraction"]
    ref.attrs["DISK_Q_TARGET"] = model.q_target
    ref.attrs["DISK_PARAMETER_PROVENANCE"] = (
        "Mdisk and Rd from Su2019/Roy2024 m12; zd is a McMillan-like thin-disc "
        "simplification; Q=1.5 is the project stability baseline"
    )

    ref.attrs["BULGE_ANALYTIC_MASS_Msun"] = model.bulge_mass
    ref.attrs["BULGE_LIVE_MASS_Msun"] = diagnostics["bulge_live_mass"]
    ref.attrs["BULGE_A_kpc"] = model.bulge_a
    ref.attrs["BULGE_RMAX_kpc"] = model.rmax_bulge
    ref.attrs["BULGE_RETAINED_FRACTION"] = diagnostics["bulge_retained_fraction"]
    ref.attrs["BULGE_PARAMETER_PROVENANCE"] = (
        "Hernquist form and m12 mass/scale from Su2019/Roy2024"
    )

    ref.attrs["BH_INCLUDED"] = int(model.include_bh)
    ref.attrs["BH_MASS_Msun"] = model.bh_mass if model.include_bh else 0.0
    ref.attrs["R_SOLAR_kpc"] = model.r_solar
    ref.attrs["VCIRC_RSOLAR_ANALYTIC_km_s"] = diagnostics["vc_solar"]
    ref.attrs["VCIRC_TARGET_REFERENCE"] = (
        "McMillan2017: 232.8 +/- 3.0 km/s at R0=8.20 +/- 0.09 kpc"
    )
    ref.attrs["OLD_STAR_AGE_GYR"] = model.old_star_age_gyr
    ref.attrs["COLLISIONLESS_VELOCITY_MODEL"] = (
        "approximate Jeans moments + local truncated Gaussian; disc uses "
        "Toomre-Q + epicyclic + asymmetric drift + approximate vertical balance"
    )
    ref.attrs["ISOLATED_RELAXATION_REQUIRED"] = 1

    ref.attrs["N_DM"] = counts[1]
    ref.attrs["N_OLD_STARS"] = counts[4]
    ref.attrs["N_BH"] = counts[5]

    rid = handle.create_group("ReservedParticleIDs")
    rid.attrs["MW_DM_ID_START"] = MW_DM_ID_START
    rid.attrs["MW_DM_ID_END"] = MW_DM_ID_START + counts[1] - 1
    rid.attrs["MW_DISK_ID_START"] = MW_DISK_ID_START
    rid.attrs["MW_DISK_ID_END"] = (
        MW_DISK_ID_START + diagnostics["n_disk"] - 1
    )
    rid.attrs["MW_BULGE_ID_START"] = MW_BULGE_ID_START
    rid.attrs["MW_BULGE_ID_END"] = (
        MW_BULGE_ID_START + diagnostics["n_bulge"] - 1
    )
    rid.attrs["MW_BH_ID"] = MW_BH_ID
    rid.attrs["LMC_ID_START"] = LMC_ID_START
    rid.attrs["LMC_ID_END"] = LMC_ID_END
    rid.attrs["SMC_ID_START"] = SMC_ID_START
    rid.attrs["SMC_ID_END"] = SMC_ID_END
    rid.attrs["FUTURE_ID_START"] = FUTURE_ID_START

    notes = handle.create_group("ModelNotes")
    notes.attrs["PARTTYPE1"] = "live tapered-NFW MW dark matter halo"
    notes.attrs["PARTTYPE4"] = (
        "old stellar disk + old stellar bulge; distinguish with IDs or "
        "StellarComponent (1=disk, 2=bulge)"
    )
    notes.attrs["PARTTYPE5"] = "central BH; accretion and feedback off in Stage 1"
    notes.attrs["NO_PARTTYPE0_REASON"] = (
        "Stage 1 isolates collisionless host stability before rebuilding the CGM"
    )
    notes.attrs["GIZMO_SOURCE_EXTERNAL_POTENTIAL_REQUIRED"] = 0
    notes.attrs["AGN_FEEDBACK_MUST_BE_OFF"] = 1
    notes.attrs["LMC_SMC_PRESENT"] = 0
    notes.attrs["CGM_PRESENT"] = 0


def validate_written_ic(path, expected_counts, box_size):
    require_h5py()
    with h5py.File(path, "r") as handle:
        header_counts = np.asarray(handle["Header"].attrs["NumPart_Total"])
        if not np.array_equal(header_counts, expected_counts):
            raise RuntimeError(
                f"Header 粒子数 {header_counts} 与期望 {expected_counts} 不一致。"
            )

        all_ids = []
        for ptype, n_expected in enumerate(expected_counts):
            name = f"PartType{ptype}"
            if n_expected == 0:
                if name in handle:
                    raise RuntimeError(f"{name} 应为空但 group 存在。")
                continue
            if name not in handle:
                raise RuntimeError(f"缺少 {name}。")
            group = handle[name]
            for dset in ("Coordinates", "Velocities", "ParticleIDs", "Masses"):
                if dset not in group:
                    raise RuntimeError(f"{name} 缺少 {dset}。")
            if len(group["Coordinates"]) != n_expected:
                raise RuntimeError(f"{name} 粒子数不正确。")
            pos = np.asarray(group["Coordinates"])
            vel = np.asarray(group["Velocities"])
            mass = np.asarray(group["Masses"])
            ids = np.asarray(group["ParticleIDs"])
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
                raise RuntimeError(f"{name} 坐标或速度存在 NaN/Inf。")
            if np.any(pos < 0.0) or np.any(pos >= box_size):
                raise RuntimeError(f"{name} 有粒子越出 BoxSize。")
            if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
                raise RuntimeError(f"{name} 粒子质量非法。")
            all_ids.append(ids.astype(np.uint64))

        ids = np.concatenate(all_ids)
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError("ParticleIDs 存在重复。")


# =============================================================================
# 7. 参数解析、主流程和诊断输出
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 Stage-1 live MW（DM+恒星盘+核球+可选BH）GIZMO IC。"
    )
    parser.add_argument("--output", default="mw_live_disk_stage1_v1.hdf5")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--box-size", type=float, default=BOXSIZE_DEFAULT)

    parser.add_argument("--m200", type=float, default=M200_DEFAULT)
    parser.add_argument("--concentration", type=float, default=CONCENTRATION_DEFAULT)
    parser.add_argument("--h0", type=float, default=H0_DEFAULT)
    parser.add_argument(
        "--halo-taper-decay-fraction",
        type=float,
        default=HALO_TAPER_DECAY_FRAC_DEFAULT,
    )
    parser.add_argument("--rmax-halo", type=float, default=RMAX_HALO_DEFAULT)

    parser.add_argument("--disk-mass", type=float, default=M_DISK_DEFAULT)
    parser.add_argument("--disk-rd", type=float, default=R_DISK_DEFAULT)
    parser.add_argument("--disk-zd", type=float, default=Z_DISK_DEFAULT)
    parser.add_argument("--rmax-disk", type=float, default=RMAX_DISK_DEFAULT)
    parser.add_argument("--zmax-disk", type=float, default=ZMAX_DISK_DEFAULT)
    parser.add_argument("--q-target", type=float, default=Q_DEFAULT)

    parser.add_argument("--bulge-mass", type=float, default=M_BULGE_DEFAULT)
    parser.add_argument("--bulge-a", type=float, default=A_BULGE_DEFAULT)
    parser.add_argument("--rmax-bulge", type=float, default=RMAX_BULGE_DEFAULT)

    parser.add_argument("--include-bh", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bh-mass", type=float, default=M_BH_DEFAULT)
    parser.add_argument("--r-solar", type=float, default=R_SOLAR_DEFAULT)
    parser.add_argument(
        "--old-star-age-gyr", type=float, default=OLD_STAR_AGE_GYR_DEFAULT
    )

    parser.add_argument(
        "--dm-particle-mass", type=float, default=TARGET_DM_MASS_DEFAULT
    )
    parser.add_argument(
        "--star-particle-mass", type=float, default=TARGET_STAR_MASS_DEFAULT
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算参数、粒子数和旋转曲线，不抽样也不写 HDF5。",
    )
    return parser.parse_args()


def build_model(args):
    model = Model(
        m200=args.m200,
        concentration=args.concentration,
        h0=args.h0,
        halo_taper_decay_fraction=args.halo_taper_decay_fraction,
        rmax_halo=args.rmax_halo,
        disk_mass=args.disk_mass,
        disk_rd=args.disk_rd,
        disk_zd=args.disk_zd,
        rmax_disk=args.rmax_disk,
        zmax_disk=args.zmax_disk,
        q_target=args.q_target,
        bulge_mass=args.bulge_mass,
        bulge_a=args.bulge_a,
        rmax_bulge=args.rmax_bulge,
        bh_mass=args.bh_mass,
        include_bh=bool(args.include_bh),
        r_solar=args.r_solar,
        old_star_age_gyr=args.old_star_age_gyr,
        box_size=args.box_size,
        dm_particle_mass=args.dm_particle_mass,
        star_particle_mass=args.star_particle_mass,
    )
    values = [
        model.m200, model.concentration, model.h0,
        model.halo_taper_decay_fraction, model.rmax_halo,
        model.disk_mass, model.disk_rd, model.disk_zd,
        model.rmax_disk, model.zmax_disk, model.q_target,
        model.bulge_mass, model.bulge_a, model.rmax_bulge,
        model.bh_mass, model.r_solar, model.old_star_age_gyr,
        model.box_size, model.dm_particle_mass, model.star_particle_mass,
    ]
    if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
        raise ValueError("全部质量、长度、浓度、Q、H0 和分辨率参数都必须为有限正数。")
    if model.rmax_halo <= model.r200:
        raise ValueError(
            f"rmax_halo={model.rmax_halo:.3f} kpc 必须大于 "
            f"R200={model.r200:.3f} kpc，才能容纳外侧 taper。"
        )
    maximum_extent = max(model.rmax_halo, model.rmax_bulge, model.rmax_disk)
    if model.box_size <= 2.0 * maximum_extent:
        raise ValueError("BoxSize 必须大于所有分量最大直径。")
    return model


def print_summary(model, halo_table, n_dm, n_disk, n_bulge, vc_solar):
    print("=" * 78)
    print("Stage-1 live MW collisionless relaxation IC")
    print("=" * 78)
    print(f"R200                     = {model.r200:.6f} kpc")
    print(f"NFW rs                   = {model.halo_rs:.6f} kpc")
    print(f"M200                      = {model.m200:.6e} Msun")
    print(f"live halo M(<Rmax)        = {halo_table['total_mass']:.6e} Msun")
    print(f"halo Rmax                 = {model.rmax_halo:.3f} kpc")
    print(f"disk analytic mass        = {model.disk_mass:.6e} Msun")
    print(f"disk Rd / zd              = {model.disk_rd:.3f} / {model.disk_zd:.3f} kpc")
    print(f"bulge mass / a            = {model.bulge_mass:.6e} Msun / {model.bulge_a:.3f} kpc")
    print(f"Q target                  = {model.q_target:.3f}")
    print(f"analytic vc({model.r_solar:.2f} kpc) = {vc_solar:.3f} km/s")
    print(f"N_DM / N_disk / N_bulge   = {n_dm:,} / {n_disk:,} / {n_bulge:,}")
    print(f"BH included               = {model.include_bh}")
    print("PartType4                 = disk + bulge (IDs and StellarComponent separate)")
    print("CGM / LMC / SMC / jet     = absent in Stage 1")


def main():
    args = parse_args()
    model = build_model(args)
    rng = np.random.default_rng(args.seed)

    print("构造 tapered-NFW 累积质量表...", flush=True)
    halo_table = build_halo_table(model)

    disk_fraction = disk_retained_fraction(model)
    disk_live_mass = model.disk_mass * disk_fraction
    bulge_fraction = bulge_retained_fraction(model)
    bulge_live_mass = model.bulge_mass * bulge_fraction

    n_dm = even_count_from_target(
        halo_table["total_mass"], model.dm_particle_mass
    )
    n_disk = even_count_from_target(disk_live_mass, model.star_particle_mass)
    n_bulge = even_count_from_target(bulge_live_mass, model.star_particle_mass)

    vc_solar = float(
        np.sqrt(total_midplane_vc2(model.r_solar, model, halo_table))
    )
    print_summary(model, halo_table, n_dm, n_disk, n_bulge, vc_solar)
    if args.dry_run:
        print("dry-run：未抽样，未写 HDF5。")
        return

    require_h5py()

    # -------------------------------------------------------------------------
    # live halo
    # -------------------------------------------------------------------------
    print("抽样 live tapered-NFW halo...", flush=True)
    dm_pos, dm_r = sample_halo_position_pairs(rng, n_dm, halo_table)
    halo_jeans_mask = halo_table["r"] > 0.0
    halo_jeans = build_isotropic_jeans_table(
        halo_table["r"][halo_jeans_mask],
        halo_table["rho"][halo_jeans_mask],
        model,
        halo_table,
    )
    dm_vel = assign_spherical_velocity_pairs(rng, dm_r, halo_jeans)
    dm_particle_mass = halo_table["total_mass"] / n_dm

    # -------------------------------------------------------------------------
    # stellar disk
    # -------------------------------------------------------------------------
    print("抽样 exponential-sech^2 stellar disk...", flush=True)
    disk_pos, disk_r_half, disk_phi_half = sample_disk_position_pairs(
        rng, n_disk, model
    )
    disk_kin = build_disk_kinematic_table(model, halo_table)
    disk_vel = assign_disk_velocity_pairs(
        rng, disk_r_half, disk_phi_half, disk_kin
    )
    disk_particle_mass = disk_live_mass / n_disk

    # -------------------------------------------------------------------------
    # stellar bulge
    # -------------------------------------------------------------------------
    print("抽样 Hernquist stellar bulge...", flush=True)
    bulge_pos, bulge_r = sample_bulge_position_pairs(rng, n_bulge, model)
    bulge_grid = np.geomspace(
        model.bulge_a * 1.0e-7, model.rmax_bulge, 30000
    )
    bulge_jeans = build_isotropic_jeans_table(
        bulge_grid,
        bulge_density_shape(bulge_grid, model),
        model,
        halo_table,
    )
    bulge_vel = assign_spherical_velocity_pairs(rng, bulge_r, bulge_jeans)
    bulge_particle_mass = bulge_live_mass / n_bulge

    # 按既有项目约定将老恒星统一放入 PartType4。
    star_pos = np.vstack((disk_pos, bulge_pos))
    star_vel = np.vstack((disk_vel, bulge_vel))
    star_mass = np.concatenate(
        (
            np.full(n_disk, disk_particle_mass, dtype=np.float64),
            np.full(n_bulge, bulge_particle_mass, dtype=np.float64),
        )
    )
    star_ids = np.concatenate(
        (
            np.arange(
                MW_DISK_ID_START,
                MW_DISK_ID_START + n_disk,
                dtype=np.uint64,
            ),
            np.arange(
                MW_BULGE_ID_START,
                MW_BULGE_ID_START + n_bulge,
                dtype=np.uint64,
            ),
        )
    )
    stellar_component = np.concatenate(
        (
            np.ones(n_disk, dtype=np.uint8),
            np.full(n_bulge, 2, dtype=np.uint8),
        )
    )
    old_star_tform = -model.old_star_age_gyr / UNIT_TIME_GYR

    dm_ids = np.arange(
        MW_DM_ID_START, MW_DM_ID_START + n_dm, dtype=np.uint64
    )
    dm_mass = np.full(n_dm, dm_particle_mass, dtype=np.float64)

    counts = np.zeros(6, dtype=np.uint32)
    counts[1] = n_dm
    counts[4] = n_disk + n_bulge
    counts[5] = 1 if model.include_bh else 0

    diagnostics = {
        "disk_retained_fraction": disk_fraction,
        "disk_live_mass": disk_live_mass,
        "bulge_retained_fraction": bulge_fraction,
        "bulge_live_mass": bulge_live_mass,
        "n_disk": n_disk,
        "n_bulge": n_bulge,
        "vc_solar": vc_solar,
    }

    output = os.path.abspath(args.output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    box_center = np.full(3, model.box_size / 2.0, dtype=np.float64)

    print(f"写出 {output} ...", flush=True)
    with h5py.File(output, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["NumPart_ThisFile"] = counts
        header.attrs["NumPart_Total"] = counts
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        header.attrs["Time"] = 0.0
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = model.box_size
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Omega0"] = 0.0
        header.attrs["OmegaLambda"] = 0.0
        header.attrs["HubbleParam"] = 1.0
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_StellarAge"] = 1
        header.attrs["Flag_Metals"] = 11
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1

        write_particle_group(
            handle,
            1,
            dm_pos,
            dm_vel,
            dm_ids,
            dm_mass,
            box_center,
        )
        write_particle_group(
            handle,
            4,
            star_pos,
            star_vel,
            star_ids,
            star_mass,
            box_center,
            metallicity=stellar_metallicity_vector(1.0),
            formation_time=old_star_tform,
            stellar_component=stellar_component,
        )
        if model.include_bh:
            write_particle_group(
                handle,
                5,
                np.zeros((1, 3), dtype=np.float64),
                np.zeros((1, 3), dtype=np.float64),
                np.array([MW_BH_ID], dtype=np.uint64),
                np.array([model.bh_mass], dtype=np.float64),
                box_center,
                bh_mass_msun=np.array([model.bh_mass], dtype=np.float64),
            )

        write_reference_groups(
            handle, model, halo_table, diagnostics, counts
        )

    validate_written_ic(output, counts, model.box_size)

    # 输出便于人工检查的统计量。
    dm_speed = np.linalg.norm(dm_vel, axis=1)
    disk_speed = np.linalg.norm(disk_vel, axis=1)
    bulge_speed = np.linalg.norm(bulge_vel, axis=1)
    print("-" * 78)
    print("HDF5 自检通过。")
    print(f"文件：{output}")
    print(f"DM particle mass          = {dm_particle_mass:.6e} Msun")
    print(f"disk particle mass        = {disk_particle_mass:.6e} Msun")
    print(f"bulge particle mass       = {bulge_particle_mass:.6e} Msun")
    print(
        "median speed DM/disk/bulge = "
        f"{np.median(dm_speed):.3f} / {np.median(disk_speed):.3f} / "
        f"{np.median(bulge_speed):.3f} km/s"
    )
    print(
        "quiet-start COM position / velocity residual = "
        f"{np.linalg.norm(np.mean(np.vstack((dm_pos, star_pos)), axis=0)):.3e} kpc / "
        f"{np.linalg.norm(np.mean(np.vstack((dm_vel, star_vel)), axis=0)):.3e} km/s"
    )
    print("下一步：只做 collisionless isolated relaxation；不要加入 CGM/LMC/AGN。")


if __name__ == "__main__":
    main()
