#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1：GIZMO 用 live Milky-Way＋冷 ISM＋热 CGM 联合弛豫 IC
================================================================

本脚本生成后续 LMC/SMC 与 AGN-jet 实验共同使用的孤立宿主：

    PartType0：连续粒子采样的热 CGM＋指数–sech^2 冷气体盘/ISM
    PartType1：live tapered-NFW 暗物质晕
    PartType4：指数–sech^2 老恒星盘＋Hernquist 老恒星核球
    PartType5：中心黑洞（默认保留；联合弛豫时吸积和反馈均关闭）

这一版不再把纯碰撞无关盘与 CGM 分成两个生产弛豫阶段。DM、恒星盘、
核球、冷气体盘和 CGM 从 t=0 一起演化，并从通过检查的孤立 snapshot 同时派生
“无卫星控制组”和“LMC/SMC 扰动组”。

-------------------------------------------------------------------------------
一、参数和方法的文献出处
-------------------------------------------------------------------------------

[C25] Carr et al. 2025, ApJ, 983, 151,
      “The All-Sky Impact of the LMC on the Milky Way Circumgalactic Medium”,
      arXiv:2408.10358, https://arxiv.org/abs/2408.10358
      直接采用/匹配：MW CGM 的 beta-profile 形式、beta=0.5、
      M_CGM(<R200)=2.8e10 Msun，以及 n(50 kpc) 约 2e-4 cm^-3 的外晕尺度；
      也参考其 live NFW halo＋指数恒星盘＋孤立稳定性测试的整体框架。
      注意：C25 使用 Enzo 且不含辐射冷却，本脚本是 GIZMO-MFM realization。

[R24] Roy et al. 2024, MNRAS, 527, 265–280,
      “Seeding the CGM: how satellites populate the cold phase of
       Milky Way haloes”, arXiv:2310.04404.
      https://arxiv.org/abs/2310.04404
      参考：GIZMO-MFM、NFW halo、Hernquist bulge、指数盘和先进行孤立
      宿主弛豫再加入卫星的工作流。

[S19] Su et al. 2019, MNRAS, 487, 4393–4408,
      arXiv:1809.09120, https://arxiv.org/abs/1809.09120
      直接采用/匹配：R24/C25 所继承的 m12 宿主分量；恒星盘
      M=5e10 Msun、Rd=3 kpc，冷气体盘 M=5e9 Msun、Rd=6 kpc，
      两者 zd=0.3 kpc；核球 M=1.5e10 Msun、a=1 kpc。

[SH03] Springel & Hernquist 2003, MNRAS, 339, 289,
       arXiv:astro-ph/0206393.
       用途：GALSF_EFFECTIVE_EQS 的未解析多相 ISM 压力模型。IC 中的
       冷气体盘按目标尺度高所需的中面压力初始化；运行时仍必须开启
       GIZMO 的 GALSF＋GALSF_EFFECTIVE_EQS，不能把 IC 压力当成替代品。

[M17] McMillan 2017, MNRAS, 465, 76–94,
      arXiv:1608.00971, https://arxiv.org/abs/1608.00971
      约束：R0=8.20+-0.09 kpc、v_c(R0)=232.8+-3.0 km/s、
      Mvir=(1.30+-0.30)e12 Msun、总恒星质量=(5.43+-0.57)e10 Msun。

[M98] Makino, Sasaki & Suto 1998, ApJ, 497, 555.
      参考：球对称 beta-profile CGM 的解析形式。

[H90] Hernquist 1990, ApJ, 356, 359.
      参考：核球密度、累积质量和势。

[SW99] Springel & White 1999, MNRAS, 307, 162–178.
       参考：多分量盘星系的近似平衡构造。

-------------------------------------------------------------------------------
二、本脚本采用的文献综合基线
-------------------------------------------------------------------------------

暗晕：M200=1.2e12 Msun、c=11，位于 M17 允许范围并接近 C25 宿主；
      R200 外使用连续斜率指数 taper，粒子延伸到 350 kpc。

恒星：Mdisk=5e10 Msun、Rd=3 kpc、zd=0.3 kpc、Q=1.5；
      Hernquist 核球 Mbulge=1.5e10 Msun、a=1 kpc。

冷盘：Mgas=5e9 Msun、Rd=6 kpc、zd=0.3 kpc，采用 Su2019/Roy2024 m12
      参数。位置按指数–sech^2 分布；内部能量由目标垂直密度在
      “球对称分量＋恒星盘薄片＋气体盘自引力”中的中面压力积分得到，
      转速同时包含径向压力梯度修正。这里的温度是未解析多相气体的
      有效压力温度，不应解释成单相冷中性介质温度。

CGM：rho(r) 正比于 [1+(r/rc)^2]^(-3 beta/2)，beta=0.5；
     严格归一化到 M_CGM(<R200)=2.8e10 Msun。气体连续采样到 350 kpc，
     其中 200 kpc 内为科学分析区，200--350 kpc 是非周期外边界缓冲层。

     C25 原始参数为 rc=0.35 kpc、n0=0.35 cm^-3，但其模拟没有 cooling。
     本脚本默认 rc=2.0 kpc，并保持相同的 R200 内总质量与约相同的
     n(50 kpc)，从而降低 cooling 开启后中心灾难性冷却和短时间步风险。
     这是明确标注的“cooling-enabled 数值修正”，不是伪称论文原参数。

温度：在“halo＋盘球平均 monopole＋核球＋CGM 自引力”的总球平均引力中
      反向积分 dP/dr=-rho*g；默认外边界 T(350 kpc)=1e6 K。这样得到
      径向变化的 HSE 温度，而不是把旧 Mou 等温气体直接拼进新势。

金属：MW所有气体与恒星统一默认 Z=0.3 Zsun；由 InitMetallicity 全局初始化，
      LMC/SMC 阶段和 jet 阶段应使用同一 cooling＋metal 配置。

-------------------------------------------------------------------------------
三、数值限制和判定口径
-------------------------------------------------------------------------------

1. collisionless 速度来自 Jeans 二阶矩和近似盘动力学，不是精确 DF，
   因而仍需约 0.3--0.5 Gyr 孤立联合弛豫。
2. CGM 的 HSE 使用恒星盘＋气体盘的球平均引力；GIZMO 中的真实盘势是轴对称 live 势，
   内晕会有温和调整。允许 cooling 产生约 30--50 km/s 的平均径向内流，
   不以“vr 必须接近 0”作为通过条件。
3. 真正需要否决的是灾难性冷却：中心冷相质量失控、CGM 大量流失、
   温度整体塌缩，或大量气体粒子的时间步长期压到 1e-6 code units。
4. 采用 quiet-start 反演粒子对降低初始质心、净动量和低阶采样噪声。
5. 恒星盘和核球都写入 PartType4，用 ParticleID 与 StellarComponent
   （1=盘、2=核球）区分；CGM 粒子 ID 预留在 1 开始的独立区间。

生成正式 IC
-----------
python make_ic_live_mw_cgm_isolated_v3.py

低分辨率筛选
------------
python make_ic_live_mw_cgm_isolated_v3.py \
    --dm-particle-mass 1e7 --star-particle-mass 1e6 \
    --cgm-particle-mass 1e5 --ism-particle-mass 1e5 \
    --cgm-outer-temperature 3.0e6 \
    --metallicity 0.3\
    --output mw_live_cgm_ism_isolated_v3_lowres.hdf5

建议弛豫物理
------------
COOLING、METALS、COOL_METAL_LINES_BY_SPECIES、GALSF 和
GALSF_EFFECTIVE_EQS 从 t=0 同时开启；第一轮关闭低温冷却、subgrid winds、
显式 FIRE feedback、BH accretion/feedback 和 AGN jet。0.3 Gyr 对应本文件
单位下 TimeMax=0.306814。

GIZMO 官方文档的 stiff-EOS 起始参数可直接写为：

    MaxSfrTimescale       4.0
    TempSupernova         3.0e8
    TempClouds            1000.0
    FactorSN              0.1
    FactorEVP             3000.0
    FactorForSofterEQS    1.0

不要靠提高 MinSizeTimestep 掩盖中心塌缩；建议保持约 1e-12，并用
OUTPUT_TIMESTEP 让检测脚本统计 dt<=1e-6 的粒子。effective-EOS 模式下
不使用 SfEffPerFreeFall；若所用分支要求 CritPhysDensity，则采用 0 让
SH03 模型内部计算阈值，并以启动日志实际接受的参数为准。

正式默认分辨率
--------------
DM=5e6 Msun，恒星=5e5 Msun，CGM/ISM=1e5 Msun。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gizmo_ic_compat import checked_ids, finalize_new_ic, validate_generated_ic
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.integrate import cumulative_trapezoid, trapezoid
    from scipy.special import iv, kv
except ImportError as exc:
    raise ImportError(
        "本脚本需要 scipy（integrate.cumulative_trapezoid 和 special.iv/kv）。"
    ) from exc


# =============================================================================
# 0. 单位、默认参数与 ParticleID 规划
# =============================================================================

G_KPC_KMS2_MSUN = 4.30091e-6
K_B_CGS = 1.380649e-16
M_P_CGS = 1.67262192369e-24
MSUN_G = 1.98847e33
GAMMA = 5.0 / 3.0
MU_HOT = 0.59
MU_NEUTRAL = 1.22
HYDROGEN_MASS_FRACTION = 0.76

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
TARGET_CGM_MASS_DEFAULT = 1.0e5

# Su et al. (2019) / Roy et al. (2024) 的 m12 冷气体盘。
ISM_DISK_MASS_DEFAULT = 5.0e9
ISM_DISK_RD_DEFAULT = 6.0
ISM_DISK_ZD_DEFAULT = 0.30
ISM_DISK_RMAX_DEFAULT = 30.0
ISM_DISK_ZMAX_DEFAULT = 3.0
ISM_DISK_METALLICITY_ZSUN_DEFAULT = 1.0

# Carr et al. (2025) 的外晕 beta-profile 质量和斜率基线。
# 默认把原论文 rc=0.35 kpc 扩大到 2 kpc，以便在开启 cooling 的 GIZMO
# 联合弛豫中抑制最中心灾难性冷却；R200 内总质量保持 2.8e10 Msun。
CGM_MASS_R200_DEFAULT = 2.8e10
CGM_BETA_DEFAULT = 0.50
CGM_CORE_RADIUS_DEFAULT = 2.0
CGM_RMAX_DEFAULT = 350.0
CGM_SCIENCE_RADIUS_DEFAULT = 200.0
CGM_OUTER_TEMPERATURE_DEFAULT = 1.0e6
CGM_METALLICITY_ZSUN_DEFAULT = 0.30
CGM_ROTATION_FRACTION_DEFAULT = 0.0

# 保持与此前 LMC/SMC 插入流程兼容的 ID 大区间。
MW_CGM_ID_START = np.uint64(1)
MW_ISM_ID_START = np.uint64(5_000_000)
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
    cgm_particle_mass: float
    ism_particle_mass: float

    ism_disk_mass: float
    ism_disk_rd: float
    ism_disk_zd: float
    ism_disk_rmax: float
    ism_disk_zmax: float
    ism_disk_metallicity_zsun: float

    cgm_mass_r200: float
    cgm_beta: float
    cgm_core_radius: float
    cgm_rmax: float
    cgm_science_radius: float
    cgm_outer_temperature: float
    cgm_metallicity_zsun: float
    cgm_rotation_fraction: float

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


def thermal_scale_kms2(temperature, mu=MU_HOT):
    """返回 k_B*T/(mu*m_p)，单位为 (km/s)^2。"""
    return K_B_CGS * np.asarray(temperature) / (mu * M_P_CGS) / 1.0e10


def temperature_from_thermal_scale(scale_kms2, mu=MU_HOT):
    """把 P/rho 或 kT/(mu mp) 从 (km/s)^2 转回 K。"""
    return (
        np.asarray(scale_kms2)
        * 1.0e10
        * mu
        * M_P_CGS
        / K_B_CGS
    )


def gas_metallicity_vector(z_zsun=CGM_METALLICITY_ZSUN_DEFAULT):
    """CGM 使用的 GIZMO/FIRE 11 维金属质量分数向量。"""
    return stellar_metallicity_vector(z_zsun)


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
# 3. Carr-inspired beta-profile CGM 与球平均 HSE 温度
# =============================================================================

def cgm_density_shape(r, model: Model):
    """
    标准球对称 beta profile：

        rho(r) ∝ [1 + (r/rc)^2]^(-3 beta / 2).

    Carr et al. (2025) 采用 beta=0.5。默认 rc=2 kpc 是本项目为了开启
    cooling 所做的显式中心修正；外晕质量和约 50 kpc 密度仍匹配其基线。
    """
    r = np.asarray(r, dtype=np.float64)
    return (
        1.0 + (r / model.cgm_core_radius) ** 2
    ) ** (-1.5 * model.cgm_beta)


def build_cgm_table(model: Model, halo_table, n_grid=50000):
    """
    建立 CGM 质量 CDF，并在总球平均引力中反向积分 HSE 压力。

    密度归一化严格满足 M_CGM(<R200)=model.cgm_mass_r200。压力边界取
    T(rmax)=model.cgm_outer_temperature。引力包含 live DM、盘 monopole、
    核球和 CGM 自引力；忽略只影响极小半径的 4e6 Msun BH 项。
    """
    rmin = min(model.cgm_core_radius, 1.0) * 1.0e-6
    r = np.concatenate(
        ([0.0], np.geomspace(rmin, model.cgm_rmax, int(n_grid)))
    )
    shape = cgm_density_shape(r, model)
    raw_mass = cumulative_trapezoid(
        4.0 * np.pi * r**2 * shape, r, initial=0.0
    )
    raw_mass_r200 = float(np.interp(model.r200, r, raw_mass))
    if raw_mass_r200 <= 0.0:
        raise RuntimeError("CGM beta-profile 归一化失败。")

    rho0_msun_kpc3 = model.cgm_mass_r200 / raw_mass_r200
    rho = rho0_msun_kpc3 * shape
    mass = rho0_msun_kpc3 * raw_mass

    enclosed = (
        halo_mass_enclosed(r, halo_table)
        + hernquist_mass_enclosed(r, model.bulge_mass, model.bulge_a)
        + exponential_disk_mass_inside_cylinder(
            r, model.disk_mass, model.disk_rd
        )
        + exponential_disk_mass_inside_cylinder(
            r, model.ism_disk_mass, model.ism_disk_rd
        )
        + mass
    )
    gravity = np.zeros_like(r)
    gravity[1:] = G_KPC_KMS2_MSUN * enclosed[1:] / r[1:] ** 2

    # P 的单位可写成 rho*(km/s)^2，因此 P/rho 可直接作为热速度平方。
    source = rho * gravity
    pressure_integral = -cumulative_trapezoid(
        source[::-1], r[::-1], initial=0.0
    )[::-1]
    outer_scale = float(thermal_scale_kms2(model.cgm_outer_temperature))
    pressure = pressure_integral + rho[-1] * outer_scale
    pressure_over_density = pressure / np.maximum(rho, 1.0e-300)
    temperature = temperature_from_thermal_scale(pressure_over_density)
    internal_energy = pressure_over_density / (GAMMA - 1.0)

    # n 是总粒子数密度，与 Carr et al. 表中 n0/n(50 kpc) 的定义一致。
    number_density = (
        rho * MSUN_G / UNIT_LENGTH_CM**3 / (MU_HOT * M_P_CGS)
    )

    return {
        "r": r,
        "rho_msun_kpc3": rho,
        "mass": mass,
        "total_mass": float(mass[-1]),
        "rho0_msun_kpc3": float(rho0_msun_kpc3),
        "number_density_cm3": number_density,
        "temperature_K": temperature,
        "internal_energy": internal_energy,
        "pressure_over_density_kms2": pressure_over_density,
    }


def cgm_mass_enclosed(r, cgm_table):
    r = np.asarray(r, dtype=np.float64)
    return np.interp(
        np.clip(r, 0.0, cgm_table["r"][-1]),
        cgm_table["r"],
        cgm_table["mass"],
    )


def sample_cgm_position_pairs(rng, n, cgm_table):
    """按球对称质量 CDF 连续抽样 CGM，并加入反演 quiet-start 粒子对。"""
    if n % 2:
        raise ValueError("quiet-start CGM 粒子数必须是偶数。")
    nh = n // 2
    target_mass = rng.random(nh) * cgm_table["total_mass"]
    radius = np.interp(target_mass, cgm_table["mass"], cgm_table["r"])
    mu = rng.uniform(-1.0, 1.0, nh)
    phi = rng.uniform(0.0, 2.0 * np.pi, nh)
    sint = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    pos_half = np.column_stack(
        (
            radius * sint * np.cos(phi),
            radius * sint * np.sin(phi),
            radius * mu,
        )
    )
    return np.vstack((pos_half, -pos_half)), np.concatenate((radius, radius))


def assign_cgm_velocity_pairs(pos, model: Model, halo_table, cgm_table):
    """
    默认 CGM 不旋转。若 f_rot>0，采用 v_phi=f_rot*v_c(R)；反演粒子
    自动获得相反笛卡尔速度，因此系统净动量保持为零。
    """
    n = len(pos)
    if n % 2:
        raise ValueError("quiet-start CGM 速度赋值要求偶数粒子。")
    nh = n // 2
    half = pos[:nh]
    radius = np.hypot(half[:, 0], half[:, 1])
    safe_radius = np.maximum(radius, 1.0e-12)
    vphi = model.cgm_rotation_fraction * np.sqrt(
        total_midplane_vc2(safe_radius, model, halo_table, cgm_table)
    )
    vel_half = np.zeros_like(half)
    vel_half[:, 0] = -vphi * half[:, 1] / safe_radius
    vel_half[:, 1] = +vphi * half[:, 0] / safe_radius
    return np.vstack((vel_half, -vel_half))


# =============================================================================
# 4. 球对称分量的 Jeans 初速度
# =============================================================================

def spherical_total_enclosed_mass(r, model: Model, halo_table, cgm_table=None):
    """仅为 Jeans 初速度使用的球平均总累积质量。"""
    total = halo_mass_enclosed(r, halo_table)
    total += hernquist_mass_enclosed(r, model.bulge_mass, model.bulge_a)
    total += exponential_disk_mass_inside_cylinder(r, model.disk_mass, model.disk_rd)
    total += exponential_disk_mass_inside_cylinder(
        r, model.ism_disk_mass, model.ism_disk_rd
    )
    if cgm_table is not None:
        total += cgm_mass_enclosed(r, cgm_table)
    if model.include_bh:
        total += model.bh_mass
    return total


def build_isotropic_jeans_table(
    r, rho, model: Model, halo_table, cgm_table=None
):
    """
    解 sigma_r^2(r)=rho^-1 * integral_r^rmax rho(s)*g(s) ds。
    外边界采用零压边界；因此外侧少量粒子会在弛豫中调整。
    """
    r = np.asarray(r, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if np.any(r <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("Jeans 网格要求 r>0 且 rho>0。")

    menc = spherical_total_enclosed_mass(r, model, halo_table, cgm_table)
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
    outer_mass = float(
        spherical_total_enclosed_mass(r[-1], model, halo_table, cgm_table)
    )
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

def exponential_sech2_retained_fraction(rd, zd, rmax, zmax):
    radial = 1.0 - (1.0 + rmax / rd) * np.exp(-rmax / rd)
    vertical = np.tanh(zmax / zd)
    return float(radial * vertical)


def disk_retained_fraction(model: Model):
    return exponential_sech2_retained_fraction(
        model.disk_rd, model.disk_zd, model.rmax_disk, model.zmax_disk
    )


def ism_disk_retained_fraction(model: Model):
    return exponential_sech2_retained_fraction(
        model.ism_disk_rd,
        model.ism_disk_zd,
        model.ism_disk_rmax,
        model.ism_disk_zmax,
    )


def sample_exponential_sech2_position_pairs(rng, n, rd, zd, rmax, zmax):
    if n % 2:
        raise ValueError("quiet-start disk 粒子数必须是偶数。")
    nh = n // 2

    r_grid = np.linspace(0.0, rmax, 20001)
    x = r_grid / rd
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    cdf /= cdf[-1]
    radius = np.interp(rng.random(nh), cdf, r_grid)
    phi = rng.uniform(0.0, 2.0 * np.pi, nh)

    tanh_max = np.tanh(zmax / zd)
    z = zd * np.arctanh(
        rng.uniform(-tanh_max, tanh_max, nh)
    )
    pos_half = np.column_stack(
        (radius * np.cos(phi), radius * np.sin(phi), z)
    )
    return np.vstack((pos_half, -pos_half)), radius, phi


def sample_disk_position_pairs(rng, n, model: Model):
    return sample_exponential_sech2_position_pairs(
        rng,
        n,
        model.disk_rd,
        model.disk_zd,
        model.rmax_disk,
        model.zmax_disk,
    )


def sample_ism_disk_position_pairs(rng, n, model: Model):
    return sample_exponential_sech2_position_pairs(
        rng,
        n,
        model.ism_disk_rd,
        model.ism_disk_zd,
        model.ism_disk_rmax,
        model.ism_disk_zmax,
    )


def total_midplane_vc2(r, model: Model, halo_table, cgm_table=None):
    r = np.asarray(r, dtype=np.float64)
    safe_r = np.maximum(r, 1.0e-7)
    halo = G_KPC_KMS2_MSUN * halo_mass_enclosed(safe_r, halo_table) / safe_r
    bulge = (
        G_KPC_KMS2_MSUN * model.bulge_mass * safe_r
        / (safe_r + model.bulge_a) ** 2
    )
    disk = thin_exponential_disk_vc2(safe_r, model.disk_mass, model.disk_rd)
    gas_disk = thin_exponential_disk_vc2(
        safe_r, model.ism_disk_mass, model.ism_disk_rd
    )
    cgm = (
        G_KPC_KMS2_MSUN * cgm_mass_enclosed(safe_r, cgm_table) / safe_r
        if cgm_table is not None
        else 0.0
    )
    bh = (
        G_KPC_KMS2_MSUN * model.bh_mass / safe_r
        if model.include_bh
        else 0.0
    )
    return np.maximum(halo + bulge + disk + gas_disk + cgm + bh, 0.0)


def spherical_external_vertical_sigma2(
    r,
    model: Model,
    halo_table,
    cgm_table=None,
    tracer_zd=None,
    tracer_zmax=None,
):
    """
    对 sech^2 盘在每个 R 数值积分球对称外势的垂直 Jeans 支撑。

    使用
        sigma_z,ext^2(R,0) = integral_0^zmax sech^2(z/zd) g_z(R,z) dz

    相比简单写成 nu^2(R)*zd^2，这一积分在 R 接近 0、zd 不再远小于 R
    时不会人为产生过大的中心垂直速度弥散。
    """
    r = np.asarray(r, dtype=np.float64)
    zd = model.disk_zd if tracer_zd is None else float(tracer_zd)
    zmax = model.zmax_disk if tracer_zmax is None else float(tracer_zmax)
    zmin = zd * 1.0e-6
    z = np.concatenate(
        ([0.0], np.geomspace(zmin, zmax, 384))
    )
    vertical_weight = 1.0 / np.cosh(z / zd) ** 2
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
            + (
                cgm_mass_enclosed(safe_s, cgm_table)
                if cgm_table is not None
                else 0.0
            )
            + (model.bh_mass if model.include_bh else 0.0)
        )
        gz = G_KPC_KMS2_MSUN * enclosed * zz / safe_s**3
        out[i0:i0 + chunk] = trapezoid(
            gz * vertical_weight[None, :], z, axis=1
        )
    return np.maximum(out, 0.0)


def sheet_vertical_pressure_scale2(tracer_zd, tracer_zmax, sheets):
    """
    返回 P(0)/rho(0)=integral sech^2(z/h_tracer) g_sheet(z) dz。

    ``sheets`` 中每项为 (Sigma, h)，Sigma 可为按 R 排列的数组。无限
    sech^2 薄片的垂直加速度为 2*pi*G*Sigma*tanh(z/h)。
    """
    zd = float(tracer_zd)
    zmin = zd * 1.0e-6
    z = np.concatenate(([0.0], np.geomspace(zmin, tracer_zmax, 384)))
    weight = 1.0 / np.cosh(z / zd) ** 2
    sigma0 = np.asarray(sheets[0][0], dtype=np.float64)
    gravity = np.zeros((len(sigma0), len(z)), dtype=np.float64)
    for surface_density, sheet_zd in sheets:
        sigma = np.asarray(surface_density, dtype=np.float64)
        gravity += (
            2.0
            * np.pi
            * G_KPC_KMS2_MSUN
            * sigma[:, None]
            * np.tanh(z[None, :] / float(sheet_zd))
        )
    return trapezoid(gravity * weight[None, :], z, axis=1)


def build_disk_kinematic_table(
    model: Model, halo_table, cgm_table=None, n_grid=12000
):
    """构造 Q、epicyclic、asymmetric-drift 与垂直速度弥散表。"""
    r = np.geomspace(1.0e-3, model.rmax_disk, int(n_grid))
    vc2 = total_midplane_vc2(r, model, halo_table, cgm_table)
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

    gas_surface = (
        model.ism_disk_mass
        / (2.0 * np.pi * model.ism_disk_rd**2)
        * np.exp(-r / model.ism_disk_rd)
    )

    # 同时加入恒星盘自引力、冷气体盘薄片引力和球对称外势。
    sigma_z_external2 = spherical_external_vertical_sigma2(
        r, model, halo_table, cgm_table
    )
    sigma_z_sheet2 = sheet_vertical_pressure_scale2(
        model.disk_zd,
        model.zmax_disk,
        (
            (sigma_surface, model.disk_zd),
            (gas_surface, model.ism_disk_zd),
        ),
    )
    sigma_z2 = sigma_z_sheet2 + sigma_z_external2
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


def build_ism_disk_kinematic_table(
    model: Model, halo_table, cgm_table=None, n_grid=12000
):
    """
    构造冷气体盘的中面有效压力、径向压力修正转速和近似 Toomre Q。

    该压力使指定的指数–sech^2 密度在中面满足积分垂直 Jeans/HSE 条件；
    GIZMO 的 GALSF_EFFECTIVE_EQS 会在运行时接管超过阈值的多相气体。
    """
    r = np.geomspace(1.0e-3, model.ism_disk_rmax, int(n_grid))
    vc2 = total_midplane_vc2(r, model, halo_table, cgm_table)
    omega2 = vc2 / r**2
    kappa2 = np.maximum(
        r * np.gradient(omega2, r, edge_order=2) + 4.0 * omega2,
        1.0e-12,
    )
    kappa = np.sqrt(kappa2)

    stellar_surface = (
        model.disk_mass / (2.0 * np.pi * model.disk_rd**2)
        * np.exp(-r / model.disk_rd)
    )
    gas_surface = (
        model.ism_disk_mass
        / (2.0 * np.pi * model.ism_disk_rd**2)
        * np.exp(-r / model.ism_disk_rd)
    )
    rho_midplane = gas_surface / (2.0 * model.ism_disk_zd)

    pressure_scale2 = spherical_external_vertical_sigma2(
        r,
        model,
        halo_table,
        cgm_table,
        tracer_zd=model.ism_disk_zd,
        tracer_zmax=model.ism_disk_zmax,
    )
    pressure_scale2 += sheet_vertical_pressure_scale2(
        model.ism_disk_zd,
        model.ism_disk_zmax,
        (
            (stellar_surface, model.disk_zd),
            (gas_surface, model.ism_disk_zd),
        ),
    )

    # 未进入多相阈值的外盘仍至少从约 10^4 K 的压力起步。
    warm_floor2 = float(thermal_scale_kms2(1.0e4, mu=MU_NEUTRAL))
    pressure_scale2 = np.maximum(pressure_scale2, warm_floor2)
    pressure_midplane = rho_midplane * pressure_scale2
    radial_pressure_gradient = np.gradient(
        pressure_midplane, r, edge_order=2
    )
    mean_vphi2 = vc2 + r * radial_pressure_gradient / np.maximum(
        rho_midplane, 1.0e-300
    )
    clipped = mean_vphi2 < 0.0
    mean_vphi = np.sqrt(np.maximum(mean_vphi2, 0.0))
    sound_speed_eff = np.sqrt(pressure_scale2)
    toomre_q = (
        sound_speed_eff
        * kappa
        / np.maximum(np.pi * G_KPC_KMS2_MSUN * gas_surface, 1.0e-30)
    )

    return {
        "r": r,
        "vc": np.sqrt(vc2),
        "mean_vphi": mean_vphi,
        "pressure_scale2": pressure_scale2,
        "internal_energy": pressure_scale2 / (GAMMA - 1.0),
        "effective_temperature_K": temperature_from_thermal_scale(
            pressure_scale2, mu=MU_NEUTRAL
        ),
        "toomre_q": toomre_q,
        "mean_vphi_clipped_fraction_grid": float(np.mean(clipped)),
    }


def assign_ism_disk_velocity_pairs(radius_half, phi_half, kin_table):
    vphi = np.interp(radius_half, kin_table["r"], kin_table["mean_vphi"])
    cosphi = np.cos(phi_half)
    sinphi = np.sin(phi_half)
    vel_half = np.column_stack((-vphi * sinphi, vphi * cosphi, np.zeros_like(vphi)))
    return np.vstack((vel_half, -vel_half))


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
    internal_energy=None,
    metallicity=None,
    gas_component=None,
    formation_time=None,
    stellar_component=None,
    bh_mass_msun=None,
):
    group = handle.create_group(f"PartType{ptype}")
    write_dataset(group, "Coordinates", positions + box_center[None, :])
    write_dataset(group, "Velocities", velocities)
    write_dataset(group, "ParticleIDs", checked_ids(ids, 32))
    write_dataset(group, "Masses", np.asarray(masses_msun, dtype=np.float64) / 1.0e10)

    if internal_energy is not None:
        energy = np.asarray(internal_energy, dtype=np.float64)
        if energy.ndim == 0:
            energy = np.full(len(positions), float(energy), dtype=np.float64)
        if energy.shape != (len(positions),):
            raise ValueError("InternalEnergy 必须是标量或长度为 N 的数组。")
        write_dataset(group, "InternalEnergy", energy)
    if metallicity is not None:
        metal = np.asarray(metallicity, dtype=np.float64)
        if metal.ndim == 1:
            metal = np.broadcast_to(metal, (len(positions), len(metal)))
        write_dataset(group, "Metallicity", metal)
    if gas_component is not None:
        write_dataset(
            group,
            "GasComponent",
            np.asarray(gas_component, dtype=np.uint8),
        )
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


def write_reference_groups(
    handle, model: Model, halo_table, cgm_table, diagnostics, counts
):
    ref = handle.create_group("MWModelReference")
    ref.attrs["MODEL_NAME"] = "Literature_hybrid_live_MW_CGM_ISM_isolated_v3"
    ref.attrs["MODEL_IS_EXACT_COPY_OF_ONE_PAPER"] = 0
    ref.attrs["MODEL_SCOPE"] = (
        "joint isolated relaxation of live DM, stars, bulge, cold ISM disk, "
        "and hot CGM; no LMC, SMC, or AGN jet"
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
    ref.attrs["REFERENCE_MAKINO_1998"] = (
        "Makino, Sasaki & Suto 1998, ApJ 497, 555"
    )
    ref.attrs["REFERENCE_HERNQUIST_1990"] = "Hernquist 1990, ApJ 356, 359"
    ref.attrs["REFERENCE_SPRINGEL_WHITE_1999"] = (
        "Springel & White 1999, MNRAS 307, 162"
    )
    ref.attrs["REFERENCE_SPRINGEL_HERNQUIST_2003"] = (
        "Springel & Hernquist 2003, MNRAS 339, 289; astro-ph/0206393"
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
        "Mdisk, Rd and zd from Su2019/Roy2024 m12; "
        "Q=1.5 is the project stability baseline"
    )

    ref.attrs["ISM_PROFILE"] = "exponential radial x sech^2 vertical disk"
    ref.attrs["ISM_ANALYTIC_MASS_Msun"] = model.ism_disk_mass
    ref.attrs["ISM_LIVE_MASS_Msun"] = diagnostics["ism_live_mass"]
    ref.attrs["ISM_RD_kpc"] = model.ism_disk_rd
    ref.attrs["ISM_ZD_kpc"] = model.ism_disk_zd
    ref.attrs["ISM_RMAX_kpc"] = model.ism_disk_rmax
    ref.attrs["ISM_ZMAX_kpc"] = model.ism_disk_zmax
    ref.attrs["ISM_RETAINED_FRACTION"] = diagnostics["ism_retained_fraction"]
    ref.attrs["ISM_Z_Zsun"] = model.ism_disk_metallicity_zsun
    ref.attrs["ISM_PARTICLE_MASS_Msun"] = diagnostics["ism_particle_mass"]
    ref.attrs["ISM_PRESSURE_MODEL"] = (
        "midplane vertical-HSE pressure for target sech^2 scale height; "
        "spherical halo+bulge+CGM+BH plus stellar/gas sheet gravity; "
        "runtime requires GALSF_EFFECTIVE_EQS"
    )
    ref.attrs["ISM_PARAMETER_PROVENANCE"] = (
        "Su2019/Roy2024 m12: Mgas=5e9 Msun, Rd=6 kpc, zd=0.3 kpc; "
        "Rmax=30 and zmax=3 kpc are explicit numerical truncations"
    )
    ref.attrs["ISM_Q_MIN_R1_15"] = diagnostics["ism_q_min_r1_15"]
    ref.attrs["ISM_EFFECTIVE_T_MEDIAN_R1_15_K"] = diagnostics[
        "ism_effective_temperature_median_r1_15"
    ]

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

    ref.attrs["CGM_PROFILE"] = "spherical beta profile"
    ref.attrs["CGM_BETA"] = model.cgm_beta
    ref.attrs["CGM_CORE_RADIUS_kpc"] = model.cgm_core_radius
    ref.attrs["CGM_RMAX_kpc"] = model.cgm_rmax
    ref.attrs["CGM_SCIENCE_RADIUS_kpc"] = model.cgm_science_radius
    ref.attrs["CGM_MASS_R200_TARGET_Msun"] = model.cgm_mass_r200
    ref.attrs["CGM_MASS_R200_ACTUAL_Msun"] = float(
        np.interp(model.r200, cgm_table["r"], cgm_table["mass"])
    )
    ref.attrs["CGM_MASS_RMAX_Msun"] = cgm_table["total_mass"]
    ref.attrs["CGM_N0_cm3"] = cgm_table["number_density_cm3"][0]
    ref.attrs["CGM_N50_cm3"] = float(
        np.interp(50.0, cgm_table["r"], cgm_table["number_density_cm3"])
    )
    ref.attrs["CGM_T0_K"] = cgm_table["temperature_K"][0]
    ref.attrs["CGM_T50_K"] = float(
        np.interp(50.0, cgm_table["r"], cgm_table["temperature_K"])
    )
    ref.attrs["CGM_TOUTER_K"] = model.cgm_outer_temperature
    ref.attrs["CGM_Z_Zsun"] = model.cgm_metallicity_zsun
    ref.attrs["CGM_ROTATION_FRACTION"] = model.cgm_rotation_fraction
    ref.attrs["CGM_PARAMETER_PROVENANCE"] = (
        "Carr2025 beta-profile architecture; beta and M(<R200) are configurable; "
        f"rc changed from paper value 0.35 kpc to {model.cgm_core_radius:g} kpc "
        "in this realization for a cooling-enabled run; HSE temperature was "
        "recomputed in full spherical-average live-host gravity"
    )
    ref.attrs["CGM_COOLING_POLICY"] = (
        "cooling+metals on from isolated relaxation; 30-50 km/s inflow allowed"
    )

    ref.attrs["N_GAS_TOTAL"] = counts[0]
    ref.attrs["N_CGM"] = diagnostics["n_cgm"]
    ref.attrs["N_ISM"] = diagnostics["n_ism"]
    ref.attrs["N_DM"] = counts[1]
    ref.attrs["N_OLD_STARS"] = counts[4]
    ref.attrs["N_BH"] = counts[5]

    rid = handle.create_group("ReservedParticleIDs")
    rid.attrs["MW_CGM_ID_START"] = MW_CGM_ID_START
    rid.attrs["MW_CGM_ID_END"] = (
        MW_CGM_ID_START + diagnostics["n_cgm"] - 1
    )
    rid.attrs["MW_ISM_ID_START"] = MW_ISM_ID_START
    rid.attrs["MW_ISM_ID_END"] = (
        MW_ISM_ID_START + diagnostics["n_ism"] - 1
    )
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
    notes.attrs["PARTTYPE0"] = (
        "hot MW CGM + cold ISM disk; distinguish with reserved ParticleIDs"
    )
    notes.attrs["PARTTYPE1"] = "live tapered-NFW MW dark matter halo"
    notes.attrs["PARTTYPE4"] = (
        "old stellar disk + old stellar bulge; distinguish with IDs or "
        "StellarComponent (1=disk, 2=bulge)"
    )
    notes.attrs["PARTTYPE5"] = "central BH; accretion and feedback off in Stage 1"
    notes.attrs["GIZMO_SOURCE_EXTERNAL_POTENTIAL_REQUIRED"] = 0
    notes.attrs["AGN_FEEDBACK_MUST_BE_OFF"] = 1
    notes.attrs["LMC_SMC_PRESENT"] = 0
    notes.attrs["CGM_PRESENT"] = 1
    notes.attrs["ISM_PRESENT"] = 1
    notes.attrs["REQUIRED_RUNTIME_PHYSICS"] = (
        "COOLING + METALS + COOL_METAL_LINES_BY_SPECIES + GALSF + "
        "GALSF_EFFECTIVE_EQS"
    )
    notes.attrs["FIRST_RELAXATION_PHYSICS_OFF"] = (
        "COOL_LOW_TEMPERATURES, GALSF_SUBGRID_WINDS, explicit FIRE feedback, "
        "BH accretion/feedback, AGN jet"
    )
    notes.attrs["RECOMMENDED_RELAXATION_TIMEMAX_CODE"] = 0.3 / UNIT_TIME_GYR
    notes.attrs["COOLING_EXPECTATION"] = (
        "moderate cooling inflow is accepted; only catastrophic core collapse "
        "or sustained minimum-timestep failure rejects the host"
    )


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
            if ptype == 0:
                for dset in ("InternalEnergy", "Metallicity"):
                    if dset not in group:
                        raise RuntimeError(f"{name} 缺少气体字段 {dset}。")
                energy = np.asarray(group["InternalEnergy"])
                metal = np.asarray(group["Metallicity"])
                if energy.shape != (n_expected,) or np.any(energy <= 0.0):
                    raise RuntimeError("PartType0/InternalEnergy 非法。")
                if metal.shape != (n_expected, 11):
                    raise RuntimeError("PartType0/Metallicity 必须为 N x 11。")
                if "GasComponent" not in group:
                    raise RuntimeError("PartType0 缺少 GasComponent。")
                component = np.asarray(group["GasComponent"])
                if component.shape != (n_expected,) or not np.all(
                    np.isin(component, (1, 2))
                ):
                    raise RuntimeError("PartType0/GasComponent 必须只含 1/2。")
            all_ids.append(ids.astype(np.uint64))

        ids = np.concatenate(all_ids)
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError("ParticleIDs 存在重复。")


# =============================================================================
# 7. 参数解析、主流程和诊断输出
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "生成 live MW（冷ISM+热CGM+DM+恒星盘+核球+可选BH）"
            "联合弛豫 GIZMO IC。"
        )
    )
    parser.add_argument(
        "--output", default="mw_live_cgm_ism_isolated_v3.hdf5"
    )
    parser.add_argument("--metallicity", type=float, default=0.3,
                        help="MW统一初始丰度 Z/Zsun；对应 INIT_STELLAR_METALS_AGES_DEFINED + InitMetallicity，默认0.3")
    parser.add_argument("--legacy-component-metallicity", action="store_true",
                        help="仅历史复现：启用旧CGM/ISM分项选项与恒星1Zsun；当前主线不使用")
    parser.add_argument("--target-format", choices=("c","cpp"), default="c", help="Output field naming; flag0 may reinitialize metals/ages/sink state")
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
        "--cgm-particle-mass", type=float, default=TARGET_CGM_MASS_DEFAULT
    )
    parser.add_argument(
        "--ism-particle-mass",
        type=float,
        default=TARGET_CGM_MASS_DEFAULT,
        help="冷气体盘目标粒子质量 [Msun]；默认与 CGM 相同。",
    )
    parser.add_argument(
        "--ism-disk-mass", type=float, default=ISM_DISK_MASS_DEFAULT
    )
    parser.add_argument(
        "--ism-disk-rd", type=float, default=ISM_DISK_RD_DEFAULT
    )
    parser.add_argument(
        "--ism-disk-zd", type=float, default=ISM_DISK_ZD_DEFAULT
    )
    parser.add_argument(
        "--ism-disk-rmax", type=float, default=ISM_DISK_RMAX_DEFAULT
    )
    parser.add_argument(
        "--ism-disk-zmax", type=float, default=ISM_DISK_ZMAX_DEFAULT
    )
    parser.add_argument(
        "--ism-disk-metallicity-zsun",
        type=float,
        default=ISM_DISK_METALLICITY_ZSUN_DEFAULT,
    )
    parser.add_argument(
        "--cgm-mass-r200", type=float, default=CGM_MASS_R200_DEFAULT,
        help="R200 内目标热 CGM 质量 [Msun]；Carr2025 基线为 2.8e10。",
    )
    parser.add_argument("--cgm-beta", type=float, default=CGM_BETA_DEFAULT)
    parser.add_argument(
        "--cgm-core-radius", type=float, default=CGM_CORE_RADIUS_DEFAULT,
        help=(
            "beta-profile 核半径 [kpc]；默认 2 kpc 是 cooling-enabled "
            "修正，Carr2025 原值为 0.35 kpc。"
        ),
    )
    parser.add_argument("--cgm-rmax", type=float, default=CGM_RMAX_DEFAULT)
    parser.add_argument(
        "--cgm-science-radius",
        type=float,
        default=CGM_SCIENCE_RADIUS_DEFAULT,
    )
    parser.add_argument(
        "--cgm-outer-temperature",
        type=float,
        default=CGM_OUTER_TEMPERATURE_DEFAULT,
        help="CGM 外边界 HSE 温度 [K]。",
    )
    parser.add_argument(
        "--cgm-metallicity-zsun",
        type=float,
        default=CGM_METALLICITY_ZSUN_DEFAULT,
    )
    parser.add_argument(
        "--cgm-rotation-fraction",
        type=float,
        default=CGM_ROTATION_FRACTION_DEFAULT,
        help="v_phi/v_c；默认 0 为无旋转基线。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算参数、粒子数和旋转曲线，不抽样也不写 HDF5。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的 --output；默认拒绝意外覆盖。",
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
        cgm_particle_mass=args.cgm_particle_mass,
        ism_particle_mass=args.ism_particle_mass,
        ism_disk_mass=args.ism_disk_mass,
        ism_disk_rd=args.ism_disk_rd,
        ism_disk_zd=args.ism_disk_zd,
        ism_disk_rmax=args.ism_disk_rmax,
        ism_disk_zmax=args.ism_disk_zmax,
        ism_disk_metallicity_zsun=args.ism_disk_metallicity_zsun,
        cgm_mass_r200=args.cgm_mass_r200,
        cgm_beta=args.cgm_beta,
        cgm_core_radius=args.cgm_core_radius,
        cgm_rmax=args.cgm_rmax,
        cgm_science_radius=args.cgm_science_radius,
        cgm_outer_temperature=args.cgm_outer_temperature,
        cgm_metallicity_zsun=args.cgm_metallicity_zsun,
        cgm_rotation_fraction=args.cgm_rotation_fraction,
    )
    values = [
        model.m200, model.concentration, model.h0,
        model.halo_taper_decay_fraction, model.rmax_halo,
        model.disk_mass, model.disk_rd, model.disk_zd,
        model.rmax_disk, model.zmax_disk, model.q_target,
        model.bulge_mass, model.bulge_a, model.rmax_bulge,
        model.bh_mass, model.r_solar, model.old_star_age_gyr,
        model.box_size, model.dm_particle_mass, model.star_particle_mass,
        model.cgm_particle_mass, model.ism_particle_mass,
        model.ism_disk_mass, model.ism_disk_rd, model.ism_disk_zd,
        model.ism_disk_rmax, model.ism_disk_zmax,
        model.cgm_mass_r200, model.cgm_beta,
        model.cgm_core_radius, model.cgm_rmax, model.cgm_science_radius,
        model.cgm_outer_temperature,
    ]
    if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
        raise ValueError("全部质量、长度、浓度、Q、H0 和分辨率参数都必须为有限正数。")
    if model.rmax_halo <= model.r200:
        raise ValueError(
            f"rmax_halo={model.rmax_halo:.3f} kpc 必须大于 "
            f"R200={model.r200:.3f} kpc，才能容纳外侧 taper。"
        )
    if model.cgm_rmax < model.r200:
        raise ValueError("cgm_rmax 必须不小于 R200，才能执行质量归一化。")
    if model.cgm_science_radius > model.cgm_rmax:
        raise ValueError("cgm_science_radius 不能大于 cgm_rmax。")
    if not 0.0 <= model.cgm_metallicity_zsun <= 5.0:
        raise ValueError("cgm_metallicity_zsun 必须位于 [0,5]。")
    if not 0.0 <= model.ism_disk_metallicity_zsun <= 5.0:
        raise ValueError("ism_disk_metallicity_zsun 必须位于 [0,5]。")
    if not 0.0 <= model.cgm_rotation_fraction < 1.0:
        raise ValueError("cgm_rotation_fraction 必须位于 [0,1)。")
    maximum_extent = max(
        model.rmax_halo,
        model.rmax_bulge,
        model.rmax_disk,
        model.ism_disk_rmax,
        model.cgm_rmax,
    )
    if model.box_size <= 2.0 * maximum_extent:
        raise ValueError("BoxSize 必须大于所有分量最大直径。")
    return model


def print_summary(
    model,
    halo_table,
    cgm_table,
    n_cgm,
    n_ism,
    n_dm,
    n_disk,
    n_bulge,
    vc_solar,
    ism_q_min,
):
    print("=" * 78)
    print("Stage-1 live MW + cold-ISM + hot-CGM isolated relaxation IC")
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
    print(
        "ISM mass / Rd / zd         = "
        f"{model.ism_disk_mass:.6e} Msun / {model.ism_disk_rd:.3f} / "
        f"{model.ism_disk_zd:.3f} kpc"
    )
    print(
        f"ISM Z / min Q(1-15 kpc)   = "
        f"{model.ism_disk_metallicity_zsun:.3f} Zsun / {ism_q_min:.3f}"
    )
    print(
        f"CGM beta / rc             = {model.cgm_beta:.3f} / "
        f"{model.cgm_core_radius:.3f} kpc"
    )
    print(f"CGM M(<R200)              = {model.cgm_mass_r200:.6e} Msun")
    print(f"CGM M(<Rmax)              = {cgm_table['total_mass']:.6e} Msun")
    print(
        "CGM n(0) / n(50 kpc)      = "
        f"{cgm_table['number_density_cm3'][0]:.4e} / "
        f"{np.interp(50.0, cgm_table['r'], cgm_table['number_density_cm3']):.4e} cm^-3"
    )
    print(
        "CGM T(0) / T(50 kpc)      = "
        f"{cgm_table['temperature_K'][0]:.4e} / "
        f"{np.interp(50.0, cgm_table['r'], cgm_table['temperature_K']):.4e} K"
    )
    print(f"CGM Z / frot              = {model.cgm_metallicity_zsun:.3f} Zsun / {model.cgm_rotation_fraction:.3f}")
    print(f"analytic vc({model.r_solar:.2f} kpc) = {vc_solar:.3f} km/s")
    print(
        f"N_CGM / N_ISM / N_DM / N_disk / N_bulge = "
        f"{n_cgm:,} / {n_ism:,} / {n_dm:,} / {n_disk:,} / {n_bulge:,}"
    )
    print(f"BH included               = {model.include_bh}")
    print("PartType4                 = disk + bulge (IDs and StellarComponent separate)")
    print("PartType0                 = hot CGM + cold ISM (reserved IDs separate)")
    print("LMC / SMC / jet           = absent in isolated relaxation")


def main():
    args = parse_args()
    if not np.isfinite(args.metallicity) or not 0 <= args.metallicity <= 5:
        raise ValueError("--metallicity must be finite and in [0,5] Zsun")
    if not args.legacy_component_metallicity:
        args.cgm_metallicity_zsun = args.metallicity
        args.ism_disk_metallicity_zsun = args.metallicity
        print(f"MW统一初始丰度={args.metallicity:g} Zsun；运行时使用 INIT_STELLAR_METALS_AGES_DEFINED 和 InitMetallicity={args.metallicity:g}")
    else:
        print("历史分组丰度复现模式；GIZMO flag0仍可能按全局InitMetallicity覆盖")
    model = build_model(args)
    rng = np.random.default_rng(args.seed)

    print("构造 tapered-NFW 累积质量表...", flush=True)
    halo_table = build_halo_table(model)
    print("构造 beta-profile CGM 质量与 HSE 温度表...", flush=True)
    cgm_table = build_cgm_table(model, halo_table)

    disk_fraction = disk_retained_fraction(model)
    disk_live_mass = model.disk_mass * disk_fraction
    ism_fraction = ism_disk_retained_fraction(model)
    ism_live_mass = model.ism_disk_mass * ism_fraction
    bulge_fraction = bulge_retained_fraction(model)
    bulge_live_mass = model.bulge_mass * bulge_fraction

    n_dm = even_count_from_target(
        halo_table["total_mass"], model.dm_particle_mass
    )
    n_disk = even_count_from_target(disk_live_mass, model.star_particle_mass)
    n_bulge = even_count_from_target(bulge_live_mass, model.star_particle_mass)
    n_cgm = even_count_from_target(
        cgm_table["total_mass"], model.cgm_particle_mass
    )
    n_ism = even_count_from_target(ism_live_mass, model.ism_particle_mass)

    if MW_CGM_ID_START + n_cgm > MW_ISM_ID_START:
        raise ValueError(
            "CGM 粒子数超出预留 ID 区间；请增大 cgm-particle-mass。"
        )
    if MW_ISM_ID_START + n_ism > MW_DM_ID_START:
        raise ValueError(
            "ISM 粒子数超出预留 ID 区间；请增大 ism-particle-mass。"
        )

    print("构造 cold-ISM 中面压力和旋转表...", flush=True)
    ism_kin = build_ism_disk_kinematic_table(model, halo_table, cgm_table)
    ism_q_mask = (ism_kin["r"] >= 1.0) & (ism_kin["r"] <= 15.0)
    ism_q_min = float(np.min(ism_kin["toomre_q"][ism_q_mask]))
    ism_teff_median = float(
        np.median(ism_kin["effective_temperature_K"][ism_q_mask])
    )

    vc_solar = float(
        np.sqrt(
            total_midplane_vc2(
                model.r_solar, model, halo_table, cgm_table
            )
        )
    )
    print_summary(
        model,
        halo_table,
        cgm_table,
        n_cgm,
        n_ism,
        n_dm,
        n_disk,
        n_bulge,
        vc_solar,
        ism_q_min,
    )
    if args.dry_run:
        print("dry-run：未抽样，未写 HDF5。")
        return

    require_h5py()

    # -------------------------------------------------------------------------
    # hot CGM：连续球对称位置、HSE 温度和可选缓慢旋转
    # -------------------------------------------------------------------------
    print("连续抽样 beta-profile hot CGM...", flush=True)
    cgm_pos, cgm_r = sample_cgm_position_pairs(rng, n_cgm, cgm_table)
    cgm_vel = assign_cgm_velocity_pairs(
        cgm_pos, model, halo_table, cgm_table
    )
    cgm_u = np.interp(cgm_r, cgm_table["r"], cgm_table["internal_energy"])
    cgm_particle_mass = cgm_table["total_mass"] / n_cgm
    cgm_mass = np.full(n_cgm, cgm_particle_mass, dtype=np.float64)
    cgm_ids = np.arange(
        MW_CGM_ID_START, MW_CGM_ID_START + n_cgm, dtype=np.uint64
    )

    # -------------------------------------------------------------------------
    # cold ISM disk：Su/Roy m12 几何；垂直 HSE 压力和径向压力修正转速
    # -------------------------------------------------------------------------
    print("抽样 exponential-sech^2 cold ISM disk...", flush=True)
    ism_pos, ism_r_half, ism_phi_half = sample_ism_disk_position_pairs(
        rng, n_ism, model
    )
    ism_vel = assign_ism_disk_velocity_pairs(
        ism_r_half, ism_phi_half, ism_kin
    )
    ism_r_full = np.concatenate((ism_r_half, ism_r_half))
    ism_u = np.interp(
        ism_r_full, ism_kin["r"], ism_kin["internal_energy"]
    )
    ism_particle_mass = ism_live_mass / n_ism
    ism_mass = np.full(n_ism, ism_particle_mass, dtype=np.float64)
    ism_ids = np.arange(
        MW_ISM_ID_START, MW_ISM_ID_START + n_ism, dtype=np.uint64
    )

    gas_pos = np.vstack((cgm_pos, ism_pos))
    gas_vel = np.vstack((cgm_vel, ism_vel))
    gas_u = np.concatenate((cgm_u, ism_u))
    gas_mass = np.concatenate((cgm_mass, ism_mass))
    gas_ids = np.concatenate((cgm_ids, ism_ids))
    gas_metallicity = np.vstack(
        (
            np.broadcast_to(
                gas_metallicity_vector(model.cgm_metallicity_zsun),
                (n_cgm, 11),
            ),
            np.broadcast_to(
                gas_metallicity_vector(model.ism_disk_metallicity_zsun),
                (n_ism, 11),
            ),
        )
    )

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
        cgm_table,
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
    disk_kin = build_disk_kinematic_table(model, halo_table, cgm_table)
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
        cgm_table,
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
    counts[0] = n_cgm + n_ism
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
        "n_cgm": n_cgm,
        "ism_retained_fraction": ism_fraction,
        "ism_live_mass": ism_live_mass,
        "ism_particle_mass": ism_particle_mass,
        "ism_q_min_r1_15": ism_q_min,
        "ism_effective_temperature_median_r1_15": ism_teff_median,
        "n_ism": n_ism,
        "vc_solar": vc_solar,
    }

    output = os.path.abspath(args.output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(
            f"输出文件已经存在：{output}；确认覆盖时请显式添加 --overwrite。"
        )
    box_center = np.full(3, model.box_size / 2.0, dtype=np.float64)

    print(f"写出 {output} ...", flush=True)
    with h5py.File(output, "w" if args.overwrite else "x") as handle:
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
        header.attrs["Flag_Cooling"] = 1
        header.attrs["Flag_StellarAge"] = 1
        header.attrs["Flag_Metals"] = 11
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1

        write_particle_group(
            handle,
            0,
            gas_pos,
            gas_vel,
            gas_ids,
            gas_mass,
            box_center,
            internal_energy=gas_u,
            metallicity=gas_metallicity,
            gas_component=np.concatenate(
                (
                    np.ones(n_cgm, dtype=np.uint8),
                    np.full(n_ism, 2, dtype=np.uint8),
                )
            ),
        )
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
            metallicity=stellar_metallicity_vector(1.0 if args.legacy_component_metallicity else args.metallicity),
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
            handle, model, halo_table, cgm_table, diagnostics, counts
        )

        handle["ModelNotes"].attrs["INITIAL_METALLICITY_POLICY"] = "legacy_components" if args.legacy_component_metallicity else "uniform_galaxy_flag0"
        handle["ModelNotes"].attrs["INIT_METALLICITY_ZSUN"] = args.metallicity
        handle["ModelNotes"].attrs["STELLAR_Z_Zsun"] = 1.0 if args.legacy_component_metallicity else args.metallicity
        handle["ModelNotes"].attrs["METALLICITY_INITIALIZER"] = "INIT_STELLAR_METALS_AGES_DEFINED + InitMetallicity"
        finalize_new_ic(handle, args.target_format, {
            "UnitLength_in_cm":UNIT_LENGTH_CM, "UnitMass_in_g":UNIT_MASS_G,
            "UnitVelocity_in_cm_per_s":UNIT_VELOCITY_CMS})
    validate_generated_ic(output)
    validate_written_ic(output, counts, model.box_size)

    # 输出便于人工检查的统计量。
    dm_speed = np.linalg.norm(dm_vel, axis=1)
    disk_speed = np.linalg.norm(disk_vel, axis=1)
    bulge_speed = np.linalg.norm(bulge_vel, axis=1)
    print("-" * 78)
    print("HDF5 自检通过。")
    print(f"文件：{output}")
    print(f"CGM particle mass         = {cgm_particle_mass:.6e} Msun")
    print(f"ISM particle mass         = {ism_particle_mass:.6e} Msun")
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
        f"{np.linalg.norm(np.mean(np.vstack((gas_pos, dm_pos, star_pos)), axis=0)):.3e} kpc / "
        f"{np.linalg.norm(np.mean(np.vstack((gas_vel, dm_vel, star_vel)), axis=0)):.3e} km/s"
    )
    print(
        "下一步：开启 cooling+metals+GALSF+GALSF_EFFECTIVE_EQS 做 "
        "0.3 Gyr 孤立 MW+ISM+CGM 联合弛豫；暂不加入 LMC/SMC，"
        "不开启低温冷却、subgrid winds 或 AGN jet。"
    )


if __name__ == "__main__":
    main()
