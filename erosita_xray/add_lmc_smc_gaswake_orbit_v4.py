#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 live LMC/SMC 加入已弛豫的 MW+冷 ISM+热 CGM 快照（GIZMO HDF5 IC）
=======================================================================

本脚本面向“1 Gyr 内 LMC/SMC 引力/流体扰动及其对后续 Galactic-centre
AGN jet 的影响”。轨道仍使用 v4 的 K13 终点反推与 shooting；气体默认改为
经检查能复现旧 v1 定向冲压流型的双卫星紧致暖气体模型。默认生成：

    PartType0：弛豫后的 MW 热 CGM + 冷 ISM + LMC/SMC 紧致暖气体
    PartType1：弛豫后的 MW live DM + LMC live DM + SMC live DM
    PartType4：弛豫后的 MW 老恒星 + LMC/SMC 指数恒星盘
    PartType5：原 MW 中心黑洞（只保留；卫星阶段继续关闭 AGN feedback）

默认 ``--satellite-gas-model v1-compact`` 继承 v1 的关键流体参数：LMC/SMC
气体质量分别为 1.5e9/1.0e9 Msun，Hernquist 标长为 6/3 kpc，温度 5e4 K，
内部一维速度弥散 25 km/s。与 v1 唯一有意的结构差别是把无限 Hernquist 尾
截到 5a（30/15 kpc），避免极少量远端粒子和瞬时全晕重叠。该模式用于研究
卫星携气体造成的定向冲压，不宣称精确复现冷 HI Magellanic Stream。

旧 v4 的 C25 热 beta 晕仍可用 ``--satellite-gas-model c25-hot-corona`` 显式
选择；它只给 LMC 加气体，并保留 ``--lmc-cgm-mode`` 等参数。由于其 50 kpc
硬截断边界和 1e6 K 外边界会使压力释放与定向冲压竞争，不再作为默认值。

-------------------------------------------------------------------------------
一、为什么不是“K13 或某篇模拟论文二选一”
-------------------------------------------------------------------------------

[K13] Kallivayalil et al. 2013, ApJ, 764, 161,
      arXiv:1301.0832, https://arxiv.org/abs/1301.0832
      用途：今天（t=0）的 Galactocentric 速度终点约束。
      默认值（K13 Table 5, new solar motion, 3-epoch）：
        LMC v=(-57,-226,+221) km/s
        SMC v=(+19,-153,+153) km/s
      今天的位置采用文献常用的同一 Galactocentric Cartesian 约定：
        LMC r=(-0.8,-41.5,-26.9) kpc
        SMC r=(15.3,-36.9,-43.3) kpc
      K13 是观测约束，不提供可直接采样的 DM/恒星/气体 IC。

[C25] Carr et al. 2025, ApJ, 983, 151,
      arXiv:2408.10358, https://arxiv.org/abs/2408.10358
      用途：最贴近本项目的 LMC--MW CGM 相互作用基线。
      采用的 LMC 参数：
        M200,DM=1.8e11 Msun, R200=117 kpc, Hernquist a=21.4 kpc；
        Mstar=2.5e9 Msun, Rd=1.7 kpc；
        M_CGM(<R200)=2.4e9 Msun, beta=0.5。
      C25 不含 SMC、ISM、辐射冷却、恒星形成或反馈；其原始 LMC CGM
      rc=0.2 kpc。本脚本在 cooling-on 默认采用 rc=1.0 kpc；完整模式仍令
      M_CGM(<117 kpc)=2.4e9 Msun，截断模式保持这条 beta-profile 的密度归一化，
      作为与宿主 rc=2 kpc 相同思想的中心数值修正。
      如要做 C25 密度参数对照，可指定 --lmc-cgm-core-radius 0.2。

[B12] Besla et al. 2012, MNRAS, 421, 2109,
      arXiv:1201.1299, https://arxiv.org/abs/1201.1299
      用途：LMC/SMC 多分量结构参数。
      SMC：总 infall mass=2.1e10 Msun，Hernquist a=7.3 kpc，R200=57.1 kpc，
      Mstar=2.6e8 Msun，Rd,star=1.1 kpc；论文另有 7.9e8 Msun 冷气体盘。
      v1-compact 不照搬该冷盘，但加入 1.0e9 Msun 暖气体；为保持 B12 的
      2.1e10 Msun 总质量，SMC DM 相应取 1.974e10 Msun。C25 热晕/无气体
      对照仍保留旧 v4 的 1.995e10 Msun DM。B12 的 LMC 参数也是 C25 所继承
      的结构来源。

[TG19] Tepper-Garcia et al. 2019, MNRAS, 488, 918,
       arXiv:1901.05636, https://arxiv.org/abs/1901.05636
       对照而非默认：其 LMC/SMC 各含 2e9 Msun 气体，气盘尺度长度分别为
       9.5/4.7 kpc，是为 Magellanic Stream/Leading Arm 构造的极端富气模型。
       不把这些气盘用于本项目默认 IC。

[H90] Hernquist 1990, ApJ, 356, 359.
      用途：卫星 DM 密度、累积质量与势。

-------------------------------------------------------------------------------
二、轨道的处理
-------------------------------------------------------------------------------

1. 从 K13 的今天位置/速度出发，在“输入的实际弛豫快照”的球平均质量剖面
   中反向积分默认 1.0 Gyr；同时包含 LMC--SMC 相互引力与 MW recoil。
2. 可选 Chandrasekhar 动力学摩擦（默认开启，lnLambda=3）；密度直接由
   输入快照的 live MW DM 粒子估计。
3. 反推只产生 trial00：真实前向计算包含非球对称盘、live-halo wake、质量
   剥离和 MW reflex motion，因此必须跑到“今天”后用 LMC/SMC 相对 MW 中心
   的 6D 残差再做 shooting 修正。绝不能把 B12/TG19 在另一套 MW 势中的
   初始坐标原样复制进来。
4. v4 可直接读取配套 validator 产生的 ``magellanic_present_6d.csv``。
   它先把观测端改成 ``target-alpha*residual``，在同一近似势中重新反推，
   再把两次反推初值之差作为 12D shooting offset；写入 IC 的观测目标与
   参考轨道仍保持原始 K13 值。首次迭代推荐 alpha=0.5。

本项目默认从今天反推 1.0 Gyr，使完整多分量卫星从更远处加入，降低瞬时插入
对 MW 内 CGM 的直接冲击。默认采用 v1-compact；C25 热晕及其
``truncated/full/none`` 选择只作为结构系统误差对照。

-------------------------------------------------------------------------------
三、推荐工作流
-------------------------------------------------------------------------------

生成低分辨率轨道试算 IC：

python add_lmc_smc_gaswake_orbit_v4_v1compact_fixed.py \
    --host-snapshot output/snapshot_010.hdf5 \
    --host-reference-ic mw_live_cgm_isolated_prod.hdf5 \
    --output mw_relaxed_lmc_smc_trial00.hdf5 \
    --lookback-gyr 1.0 \
    --satellite-gas-model v1-compact

trial01 前向验证后生成阻尼修正的 trial02（卫星内部结构不变）：

python add_lmc_smc_gaswake_orbit_v4_v1compact_fixed.py \
    --host-snapshot output_isolated/snapshot_006.hdf5 \
    --host-reference-ic mw_live_cgm_ism_isolated_v3_lowres.hdf5 \
    --output mw_relaxed_lmc_smc_gaswake_trial02.hdf5 \
    --lookback-gyr 1.0 --satellite-gas-model v1-compact \
    --shooting-base-ic mw_relaxed_lmc_smc_gaswake_trial01.hdf5 \
    --shooting-residual-csv output/magellanic_validation/magellanic_present_6d.csv \
    --shooting-alpha 0.5 --trial-label trial02

说明：
* 正式默认分辨率为 m_gas=宿主气体质量（你的 production 为 1e5 Msun）、
  m_DM,sat=min(m_DM,host,1e6 Msun)、m_star,sat=min(m_star,host,1e5 Msun)。
  这样约有 2e4 个 SMC DM 粒子，不会把 Carr 的昂贵粒子数硬套到 GIZMO，
  也避免仅继承 5e6 Msun 宿主 DM 后令 SMC 只有约 4000 个 DM 粒子。
  lowres 筛选可显式传入 1e7/1e6/5e5 Msun，见 --help。
* 输出时间重置为 0；保持 Stage-1 的 cooling+metals+GALSF+
  GALSF_EFFECTIVE_EQS，关闭显式 stellar feedback、subgrid winds、BH accretion
  和 AGN jet，前向演化 1.0 Gyr。到达 K13 终点并完成轨道校准后，再分支开启 jet。
  比较 v1 时必须用 Stage-2 已演化时间，而不能直接比较 Header/Time：v1 继承
  Stage-1 时钟，v4 默认从 0 计时。
* 输出旁边会生成同名 .orbit.json，记录目标、反推初值和全部文献/项目选择。
* IC 内同时写入 MagellanicOrbitReference，可直接交给配套的
  validation_dashboard_sat_6d_gaswake_v3.py；接口与 v2 保持向后兼容。
* 本脚本给卫星内部速度使用 Jeans/epicyclic 近似。生产前建议先做一次
  0.1--0.2 Gyr 的卫星自稳定性小测试，或至少检查 LMC/SMC 的 M(<r)、Rd、
  厚度和束缚质量；轨道 trial00 本身仍需一次完整前向校准。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise ImportError("需要 h5py：python -m pip install h5py") from exc

try:
    from scipy.integrate import cumulative_trapezoid, solve_ivp
    from scipy.special import erf, iv, kv
except ImportError as exc:
    raise ImportError("需要 scipy：python -m pip install scipy") from exc


# =============================================================================
# 0. 单位、文献默认值和 ID 规划
# =============================================================================

SCRIPT_REVISION = "v4-v1compact-fixed-2026-09-24"

G_KPC_KMS2_MSUN = 4.30091e-6
KMS_TO_KPC_GYR = 1.0227121650537077
UNIT_TIME_GYR = 1.0 / KMS_TO_KPC_GYR
G_KPC3_MSUN_GYR2 = G_KPC_KMS2_MSUN * KMS_TO_KPC_GYR**2

K_B_CGS = 1.380649e-16
M_P_CGS = 1.67262192369e-24
GAMMA = 5.0 / 3.0
MU_HOT = 0.59
UNIT_MASS_MSUN = 1.0e10

Z_SUN_MASS_FRACTION = 0.02
SOLAR_METAL_MASS = np.array(
    [0.00326, 0.00132, 0.00865, 0.00222, 0.000931,
     0.00108, 0.000644, 0.000101, 0.00173],
    dtype=np.float64,
)

# K13 Table 5 的“new solar / 3-epoch”速度；位置为 B12/C25 使用的
# 同一 Galactocentric Cartesian 约定。坐标原点在 Galactic centre，
# +X 方向由太阳指向 Galactic centre（因此太阳位于 X=-R0），
# +Y 沿银河旋转方向，+Z 指向北银极。
K13_LMC_POSITION_KPC = np.array([-0.8, -41.5, -26.9], dtype=np.float64)
K13_LMC_VELOCITY_KMS = np.array([-57.0, -226.0, 221.0], dtype=np.float64)
K13_SMC_POSITION_KPC = np.array([15.3, -36.9, -43.3], dtype=np.float64)
K13_SMC_VELOCITY_KMS = np.array([19.0, -153.0, 153.0], dtype=np.float64)

# 与 Stage-1 生成器一致的保留区间；为不同分量再划分子区间。
LMC_GAS_ID_START = np.uint64(100_000_000)
LMC_DM_ID_START = np.uint64(110_000_000)
LMC_STAR_ID_START = np.uint64(120_000_000)
SMC_DM_ID_START = np.uint64(200_000_000)
SMC_STAR_ID_START = np.uint64(210_000_000)
SMC_GAS_ID_START = np.uint64(220_000_000)
ID_BLOCK_SIZE = 10_000_000
LMC_ID_START = np.uint64(100_000_000)
LMC_ID_END = np.uint64(199_999_999)
SMC_ID_START = np.uint64(200_000_000)
SMC_ID_END = np.uint64(299_999_999)
MW_BH_ID = np.uint64(90_000_000)

C25_LMC_CGM_MASS_R200 = 2.4e9
C25_LMC_CGM_R200_KPC = 117.0
DEFAULT_LMC_CGM_TRUNCATION_KPC = 50.0
V1_LMC_GAS_MASS_MSUN = 1.5e9
V1_SMC_GAS_MASS_MSUN = 1.0e9
V1_LMC_GAS_HERNQUIST_A_KPC = 6.0
V1_SMC_GAS_HERNQUIST_A_KPC = 3.0
V1_COMPACT_TRUNCATION_IN_A = 5.0
V1_WARM_GAS_TEMPERATURE_K = 5.0e4
V1_WARM_GAS_SIGMA_KMS = 25.0

# K13 的速度分量 1-sigma；位置误差由其天球中心/距离误差传播得到，
# 数值与旧版成套脚本一致，用于 dashboard 的逐分量判定。
K13_LMC_POSITION_SIGMA_KPC = np.array([0.39645665, 1.89267257, 1.28905242])
K13_SMC_POSITION_SIGMA_KPC = np.array([1.07999820, 1.75861938, 2.03931350])
K13_LMC_VELOCITY_SIGMA_KMS = np.array([13.0, 15.0, 19.0])
K13_SMC_VELOCITY_SIGMA_KMS = np.array([18.0, 21.0, 17.0])


@dataclass(frozen=True)
class Satellite:
    """卫星结构。质量单位 Msun，长度 kpc。"""

    name: str
    dm_mass_rmax: float
    dm_a: float
    dm_rmax: float
    star_mass: float
    star_rd: float
    star_zd: float
    star_rmax: float
    star_zmax: float
    gas_profile: str = "none"
    cgm_mass_rmax: float = 0.0
    cgm_beta: float = 0.5
    cgm_rc: float = 1.0
    cgm_rmax: float = 0.0
    cgm_outer_temperature: float = 1.0e6
    cgm_metallicity_zsun: float = 0.2
    gas_velocity_dispersion: float = 0.0
    gas_mean_molecular_weight: float = MU_HOT

    @property
    def dm_analytic_mass(self) -> float:
        """令截断半径内的 Hernquist 质量严格等于 dm_mass_rmax。"""
        retained = (self.dm_rmax / (self.dm_rmax + self.dm_a)) ** 2
        return self.dm_mass_rmax / retained

    @property
    def live_mass(self) -> float:
        return self.dm_mass_rmax + self.star_mass + self.cgm_mass_rmax


def beta_profile_mass_fraction(radius: float, reference_radius: float,
                               core_radius: float, beta: float) -> float:
    """同一 beta-profile 归一化下的 M(<radius)/M(<reference_radius)。"""
    if not (0.0 < radius <= reference_radius):
        raise ValueError("beta-profile 截断半径必须在 (0, reference_radius] 内。")
    grid = np.concatenate((
        [0.0],
        np.geomspace(max(core_radius * 1.0e-7, 1.0e-9), reference_radius, 30000),
    ))
    shape = (1.0 + (grid / core_radius) ** 2) ** (-1.5 * beta)
    raw = cumulative_trapezoid(4.0 * np.pi * grid**2 * shape, grid, initial=0.0)
    return float(np.interp(radius, grid, raw) / raw[-1])


def build_default_satellites(
    lmc_cgm_core_radius: float,
    lmc_cgm_mode: str,
    lmc_cgm_rmax: float,
    satellite_gas_model: str,
) -> tuple[Satellite, Satellite]:
    """C25/B12 碰撞less 结构 + 可切换的 v1 紧致气体或 C25 热晕。"""
    if satellite_gas_model == "v1-compact":
        lmc_gas = {
            "gas_profile": "v1_compact_hernquist",
            "cgm_mass_rmax": V1_LMC_GAS_MASS_MSUN,
            "cgm_rc": V1_LMC_GAS_HERNQUIST_A_KPC,
            "cgm_rmax": V1_COMPACT_TRUNCATION_IN_A * V1_LMC_GAS_HERNQUIST_A_KPC,
            "cgm_outer_temperature": V1_WARM_GAS_TEMPERATURE_K,
            "cgm_metallicity_zsun": 0.3,
            "gas_velocity_dispersion": V1_WARM_GAS_SIGMA_KMS,
            "gas_mean_molecular_weight": 0.61,
        }
        smc_gas = {
            "gas_profile": "v1_compact_hernquist",
            "cgm_mass_rmax": V1_SMC_GAS_MASS_MSUN,
            "cgm_rc": V1_SMC_GAS_HERNQUIST_A_KPC,
            "cgm_rmax": V1_COMPACT_TRUNCATION_IN_A * V1_SMC_GAS_HERNQUIST_A_KPC,
            "cgm_outer_temperature": V1_WARM_GAS_TEMPERATURE_K,
            "cgm_metallicity_zsun": 0.1,
            "gas_velocity_dispersion": V1_WARM_GAS_SIGMA_KMS,
            "gas_mean_molecular_weight": 0.61,
        }
    elif satellite_gas_model == "none":
        lmc_gas = {}
        smc_gas = {}
    elif satellite_gas_model != "c25-hot-corona":
        raise ValueError(f"未知 --satellite-gas-model={satellite_gas_model!r}。")
    elif lmc_cgm_mode == "none":
        lmc_cgm_mass = 0.0
        lmc_cgm_radius = 0.0
    elif lmc_cgm_mode == "full":
        lmc_cgm_mass = C25_LMC_CGM_MASS_R200
        lmc_cgm_radius = C25_LMC_CGM_R200_KPC
    elif lmc_cgm_mode == "truncated":
        if not (0.0 < lmc_cgm_rmax <= C25_LMC_CGM_R200_KPC):
            raise ValueError(
                "--lmc-cgm-rmax 在 truncated 模式下必须位于 (0,117] kpc。"
            )
        lmc_cgm_radius = float(lmc_cgm_rmax)
        lmc_cgm_mass = C25_LMC_CGM_MASS_R200 * beta_profile_mass_fraction(
            lmc_cgm_radius, C25_LMC_CGM_R200_KPC,
            lmc_cgm_core_radius, 0.5,
        )
    else:
        raise ValueError(f"未知 --lmc-cgm-mode={lmc_cgm_mode!r}。")

    if satellite_gas_model == "c25-hot-corona":
        lmc_gas = {
            "gas_profile": "c25_beta_hydrostatic" if lmc_cgm_mass > 0.0 else "none",
            "cgm_mass_rmax": lmc_cgm_mass,
            "cgm_beta": 0.5,
            "cgm_rc": lmc_cgm_core_radius,
            "cgm_rmax": lmc_cgm_radius,
            "cgm_outer_temperature": 1.0e6,
            "cgm_metallicity_zsun": 0.2,
            "gas_velocity_dispersion": 0.0,
            "gas_mean_molecular_weight": MU_HOT,
        }
        smc_gas = {}

    lmc = Satellite(
        name="LMC",
        dm_mass_rmax=1.8e11,
        dm_a=21.4,
        dm_rmax=117.0,
        star_mass=2.5e9,
        star_rd=1.7,
        star_zd=0.34,
        star_rmax=6.8,
        star_zmax=1.7,
        **lmc_gas,
    )
    smc = Satellite(
        name="SMC",
        # v1-compact 时保持 B12 Mtotal=2.1e10；其他模式沿用旧 v4 的 DM 质量。
        dm_mass_rmax=(
            2.1e10 - 2.6e8 - V1_SMC_GAS_MASS_MSUN
            if satellite_gas_model == "v1-compact" else 1.995e10
        ),
        dm_a=7.3,
        dm_rmax=57.1,
        star_mass=2.6e8,
        star_rd=1.1,
        star_zd=0.22,
        star_rmax=4.4,
        star_zmax=1.1,
        **smc_gas,
    )
    return lmc, smc


# =============================================================================
# 1. 通用数值函数
# =============================================================================

def even_count_from_target(total_mass: float, target_mass: float) -> int:
    n = max(2, int(round(total_mass / target_mass)))
    return n if n % 2 == 0 else n + 1


def masses_msun(group, mass_table_value: float) -> np.ndarray:
    if "Masses" in group:
        return np.asarray(group["Masses"], dtype=np.float64) * UNIT_MASS_MSUN
    if mass_table_value <= 0.0:
        raise RuntimeError(f"{group.name} 没有 Masses，且 Header/MassTable 也为 0。")
    return np.full(len(group["Coordinates"]), mass_table_value * UNIT_MASS_MSUN)


def robust_median_particle_mass(group, mass_table_value: float) -> float:
    mass = masses_msun(group, mass_table_value)
    if len(mass) == 0 or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise RuntimeError(f"{group.name} 的粒子质量非法。")
    return float(np.median(mass))


def stellar_metallicity_vector(z_zsun: float, ncol: int = 11) -> np.ndarray:
    """生成与宿主 Metallicity 列数兼容的金属质量分数。"""
    if ncol == 1:
        return np.array([z_zsun * Z_SUN_MASS_FRACTION], dtype=np.float64)
    if ncol != 11:
        raise ValueError(f"目前只支持 1 或 11 列 Metallicity，收到 {ncol}。")
    out = np.zeros(11, dtype=np.float64)
    out[0] = z_zsun * Z_SUN_MASS_FRACTION
    out[1] = 0.24 + (0.28 - 0.24) * z_zsun
    out[2:] = z_zsun * SOLAR_METAL_MASS
    return out


def internal_energy_from_temperature(
    temperature: np.ndarray | float,
    mean_molecular_weight: float = MU_HOT,
) -> np.ndarray:
    """理想单原子气体比内能，输出 GIZMO 内部单位 (km/s)^2。"""
    thermal_scale_kms2 = (
        K_B_CGS * np.asarray(temperature, dtype=np.float64)
        / (mean_molecular_weight * M_P_CGS) / 1.0e10
    )
    return thermal_scale_kms2 / (GAMMA - 1.0)


def hernquist_mass_enclosed(r, analytic_mass: float, a: float):
    r = np.asarray(r, dtype=np.float64)
    return analytic_mass * r**2 / (r + a) ** 2


def exponential_disk_mass_enclosed(r, mass: float, rd: float):
    x = np.asarray(r, dtype=np.float64) / rd
    return mass * (1.0 - (1.0 + x) * np.exp(-x))


def thin_exponential_disk_vc2(r, mass: float, rd: float):
    r = np.asarray(r, dtype=np.float64)
    y = np.maximum(r / (2.0 * rd), 1.0e-8)
    sigma0 = mass / (2.0 * np.pi * rd**2)
    bessel = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
    return np.maximum(
        4.0 * np.pi * G_KPC_KMS2_MSUN * sigma0 * rd * y**2 * bessel,
        0.0,
    )


def rotation_from_z_to_vector(normal: np.ndarray) -> np.ndarray:
    """Rodrigues 旋转：把局部盘面的 +z 轴转到给定单位向量。"""
    target = np.asarray(normal, dtype=np.float64)
    target /= np.linalg.norm(target)
    source = np.array([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1.0e-14:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    k = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + sine * k + (1.0 - cosine) * (k @ k)


# =============================================================================
# 2. 从实际弛豫快照建立球平均宿主势和 DM 密度
# =============================================================================

def find_host_center_and_velocity(handle) -> tuple[np.ndarray, np.ndarray]:
    """优先用 BH 定中心；中心速度用内 5 kpc 老恒星的稳健中位数。"""
    if "PartType5" in handle and len(handle["PartType5/Coordinates"]) > 0:
        bh_group = handle["PartType5"]
        bh_ids = np.asarray(bh_group["ParticleIDs"], dtype=np.uint64)
        match = np.flatnonzero(bh_ids == MW_BH_ID)
        bh_index = int(match[0]) if match.size else 0
        center = np.asarray(bh_group["Coordinates"])[bh_index].astype(np.float64)
    else:
        stars = np.asarray(handle["PartType4/Coordinates"], dtype=np.float64)
        center = np.median(stars, axis=0)

    if "PartType4" not in handle:
        raise RuntimeError("宿主快照必须含 PartType4，才能稳健定义银河中心速度。")
    spos = np.asarray(handle["PartType4/Coordinates"], dtype=np.float64)
    svel = np.asarray(handle["PartType4/Velocities"], dtype=np.float64)
    radius = np.linalg.norm(spos - center[None, :], axis=1)
    inner = radius < 5.0
    if np.count_nonzero(inner) < 32:
        inner = radius < 10.0
    if np.count_nonzero(inner) < 32:
        raise RuntimeError("中心 10 kpc 内恒星太少，无法定义宿主整体速度。")
    velocity = np.median(svel[inner], axis=0)
    return center, velocity


def ensure_host_ids_are_available(handle) -> None:
    """在昂贵采样前确认宿主未占用 LMC/SMC 的保留 ID 区间。"""
    collisions = []
    for ptype in range(6):
        name = f"PartType{ptype}"
        if name not in handle or "ParticleIDs" not in handle[name]:
            continue
        ids = np.asarray(handle[name]["ParticleIDs"], dtype=np.uint64)
        bad = (
            ((ids >= LMC_ID_START) & (ids <= LMC_ID_END))
            | ((ids >= SMC_ID_START) & (ids <= SMC_ID_END))
        )
        if np.any(bad):
            examples = ids[bad][:5].tolist()
            collisions.append(f"{name}: {np.count_nonzero(bad)} 个，例如 {examples}")
    if collisions:
        raise RuntimeError(
            "宿主已经占用 LMC/SMC 保留 ID 区间；不能安全添加卫星。\n  "
            + "\n  ".join(collisions)
        )


class HostProfile:
    """由输入快照直接建立的球平均 M(<r) 与 rho_DM(r)。"""

    def __init__(self, radius_sorted, mass_cumulative, rho_r, rho_dm):
        self.radius = np.asarray(radius_sorted, dtype=np.float64)
        self.mass = np.asarray(mass_cumulative, dtype=np.float64)
        self.rho_r = np.asarray(rho_r, dtype=np.float64)
        self.rho_dm = np.asarray(rho_dm, dtype=np.float64)

    def menc(self, r):
        value = np.asarray(r, dtype=np.float64)
        return np.interp(value, self.radius, self.mass, left=0.0, right=self.mass[-1])

    def dm_density(self, r):
        value = np.maximum(np.asarray(r, dtype=np.float64), self.rho_r[0])
        log_rho = np.interp(
            np.log(value), np.log(self.rho_r), np.log(self.rho_dm),
            left=np.log(self.rho_dm[0]), right=np.log(self.rho_dm[-1]),
        )
        return np.exp(log_rho)


def build_host_profile(handle, center: np.ndarray) -> HostProfile:
    header = handle["Header"].attrs
    mass_table = np.asarray(header.get("MassTable", np.zeros(6)), dtype=np.float64)

    all_radius = []
    all_mass = []
    for ptype in range(6):
        name = f"PartType{ptype}"
        if name not in handle:
            continue
        group = handle[name]
        pos = np.asarray(group["Coordinates"], dtype=np.float64) - center[None, :]
        mass = masses_msun(group, float(mass_table[ptype]))
        all_radius.append(np.linalg.norm(pos, axis=1))
        all_mass.append(mass)
    radius = np.concatenate(all_radius)
    mass = np.concatenate(all_mass)
    order = np.argsort(radius)
    radius_sorted = radius[order]
    mass_cumulative = np.cumsum(mass[order], dtype=np.float64)

    dm = handle["PartType1"]
    dm_pos = np.asarray(dm["Coordinates"], dtype=np.float64) - center[None, :]
    dm_radius = np.linalg.norm(dm_pos, axis=1)
    dm_mass = masses_msun(dm, float(mass_table[1]))
    positive = dm_radius[dm_radius > 0.0]
    rmin = max(0.1, float(np.percentile(positive, 0.05)))
    rmax = max(400.0, float(np.max(positive)) * 1.05)
    edges = np.geomspace(rmin, rmax, 121)
    shell_mass, _ = np.histogram(dm_radius, bins=edges, weights=dm_mass)
    shell_volume = 4.0 * np.pi / 3.0 * (edges[1:]**3 - edges[:-1]**3)
    rho = shell_mass / shell_volume
    centers = np.sqrt(edges[:-1] * edges[1:])

    # 空 bin 用相邻有效 bin 的 log 插值补齐；摩擦仅在外晕使用。
    valid = rho > 0.0
    if np.count_nonzero(valid) < 5:
        raise RuntimeError("PartType1 太少，无法估计动力学摩擦所需的 DM 密度。")
    rho_filled = np.exp(np.interp(
        np.log(centers), np.log(centers[valid]), np.log(rho[valid])
    ))
    return HostProfile(radius_sorted, mass_cumulative, centers, rho_filled)


# =============================================================================
# 3. K13 终点的反向轨道积分
# =============================================================================

def satellite_enclosed_mass(separation, sat: Satellite):
    s = np.asarray(separation, dtype=np.float64)
    value = hernquist_mass_enclosed(s, sat.dm_analytic_mass, sat.dm_a)
    value += exponential_disk_mass_enclosed(s, sat.star_mass, sat.star_rd)
    if sat.cgm_mass_rmax > 0.0:
        if sat.gas_profile == "v1_compact_hernquist":
            retained = (sat.cgm_rmax / (sat.cgm_rmax + sat.cgm_rc)) ** 2
            gas_analytic_mass = sat.cgm_mass_rmax / retained
            enclosed_gas = hernquist_mass_enclosed(
                np.minimum(s, sat.cgm_rmax), gas_analytic_mass, sat.cgm_rc
            )
            value += enclosed_gas
        else:
            # C25 beta 晕仅占总质量的小部分；轨道反推沿用旧 v4 的线性近似。
            value += sat.cgm_mass_rmax * np.minimum(s / sat.cgm_rmax, 1.0)
    return np.minimum(value, sat.live_mass)


def host_acceleration(position: np.ndarray, profile: HostProfile) -> np.ndarray:
    radius = float(np.linalg.norm(position))
    if radius < 1.0e-8:
        return np.zeros(3)
    return -G_KPC3_MSUN_GYR2 * profile.menc(radius) * position / radius**3


def dynamical_friction_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    satellite_mass: float,
    profile: HostProfile,
    coulomb_log: float,
) -> np.ndarray:
    """Chandrasekhar 近似；velocity 的单位为 kpc/Gyr。"""
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    if radius < 1.0e-6 or speed < 1.0e-8:
        return np.zeros(3)
    rho = float(profile.dm_density(radius))
    vc = math.sqrt(max(G_KPC3_MSUN_GYR2 * profile.menc(radius) / radius, 1.0e-30))
    # sigma≈vc/sqrt(2)，故 X=v/(sqrt(2)sigma)≈v/vc。
    x = speed / vc
    factor = float(erf(x) - 2.0 * x * np.exp(-x*x) / np.sqrt(np.pi))
    amplitude = (
        4.0 * np.pi * G_KPC3_MSUN_GYR2**2 * satellite_mass
        * rho * coulomb_log * max(factor, 0.0) / speed**3
    )
    return -amplitude * velocity


def integrate_magellanic_orbit(
    profile: HostProfile,
    lmc: Satellite,
    smc: Satellite,
    lookback_gyr: float,
    use_dynamical_friction: bool,
    include_host_recoil: bool,
    coulomb_log: float,
    lmc_target_position: np.ndarray,
    lmc_target_velocity: np.ndarray,
    smc_target_position: np.ndarray,
    smc_target_velocity: np.ndarray,
    orbit_samples: int,
):
    """从 K13 今天终点反向积分，返回插入时 COM 状态和数值闭合残差。"""
    state_today = np.concatenate((
        lmc_target_position,
        lmc_target_velocity * KMS_TO_KPC_GYR,
        smc_target_position,
        smc_target_velocity * KMS_TO_KPC_GYR,
    ))

    def rhs(_time, state):
        rl = state[0:3]
        vl = state[3:6]
        rs = state[6:9]
        vs = state[9:12]
        delta = rs - rl
        separation = float(np.linalg.norm(delta))
        if separation < 1.0e-6:
            raise RuntimeError("反向积分中 LMC/SMC COM 发生数值重合。")

        al = host_acceleration(rl, profile)
        a_s_on_l = (
            G_KPC3_MSUN_GYR2 * satellite_enclosed_mass(separation, smc)
            * delta / separation**3
        )
        al += a_s_on_l

        a_s = host_acceleration(rs, profile)
        a_l_on_s = (
            -G_KPC3_MSUN_GYR2 * satellite_enclosed_mass(separation, lmc)
            * delta / separation**3
        )
        a_s += a_l_on_s

        if include_host_recoil:
            # 状态量是相对 MW 中心的坐标。live MW 会被两颗卫星加速，
            # 因此在这个非惯性参考系中必须减去 MW 中心的平动加速度。
            r_l = max(float(np.linalg.norm(rl)), 1.0e-6)
            r_s = max(float(np.linalg.norm(rs)), 1.0e-6)
            a_mw = G_KPC3_MSUN_GYR2 * (
                satellite_enclosed_mass(r_l, lmc) * rl/r_l**3
                + satellite_enclosed_mass(r_s, smc) * rs/r_s**3
            )
            al -= a_mw
            a_s -= a_mw

        if use_dynamical_friction:
            al += dynamical_friction_acceleration(
                rl, vl, lmc.live_mass, profile, coulomb_log
            )
            a_s += dynamical_friction_acceleration(
                rs, vs, smc.live_mass, profile, coulomb_log
            )
        return np.concatenate((vl, al, vs, a_s))

    time_backward = np.linspace(0.0, -lookback_gyr, orbit_samples)
    backward = solve_ivp(
        rhs,
        (0.0, -lookback_gyr),
        state_today,
        t_eval=time_backward,
        method="DOP853",
        rtol=2.0e-10,
        atol=1.0e-11,
        max_step=min(0.002, lookback_gyr / 100.0),
    )
    if not backward.success:
        raise RuntimeError(f"反向轨道积分失败：{backward.message}")
    state_initial = backward.y[:, -1]

    # 保存为时间从过去到今天递增的参考轨道；速度转换回 km/s，供配套
    # validation dashboard 逐快照比较。该曲线仍只是球平均宿主势中的初猜。
    reference_time = backward.t[::-1].copy()
    reference_states = backward.y.T[::-1].copy()
    reference_states[:, 3:6] /= KMS_TO_KPC_GYR
    reference_states[:, 9:12] /= KMS_TO_KPC_GYR

    # 同一近似方程下再前向积分，检查纯数值闭合；这不是 live GIZMO 的校准。
    forward = solve_ivp(
        rhs,
        (-lookback_gyr, 0.0),
        state_initial,
        method="DOP853",
        rtol=2.0e-10,
        atol=1.0e-11,
        max_step=min(0.002, lookback_gyr / 100.0),
    )
    if not forward.success:
        raise RuntimeError(f"闭合测试的前向轨道积分失败：{forward.message}")
    residual = forward.y[:, -1] - state_today
    residual[[3, 4, 5, 9, 10, 11]] /= KMS_TO_KPC_GYR
    return state_initial, residual, reference_time, reference_states


# =============================================================================
# 4. live 卫星粒子的采样
# =============================================================================

def sample_hernquist_position_pairs(rng, n: int, sat: Satellite):
    if n % 2:
        raise ValueError("quiet-start Hernquist 粒子数必须为偶数。")
    half = n // 2
    retained = (sat.dm_rmax / (sat.dm_rmax + sat.dm_a)) ** 2
    y = np.sqrt(rng.random(half) * retained)
    radius = sat.dm_a * y / np.maximum(1.0 - y, 1.0e-14)
    mu = rng.uniform(-1.0, 1.0, half)
    phi = rng.uniform(0.0, 2.0 * np.pi, half)
    sint = np.sqrt(1.0 - mu**2)
    pos_half = np.column_stack((
        radius * sint * np.cos(phi),
        radius * sint * np.sin(phi),
        radius * mu,
    ))
    return np.vstack((pos_half, -pos_half)), np.concatenate((radius, radius))


def build_satellite_cgm_table(sat: Satellite, n_grid: int = 30000):
    if sat.cgm_mass_rmax <= 0.0:
        return None

    if sat.gas_profile == "v1_compact_hernquist":
        # v1 的气体用 Hernquist 球；这里仅把无限尾安全截在 5a，并重新归一化，
        # 保持命令行所声明的 live gas mass 全部位于截断半径内。
        a = sat.cgm_rc
        r = np.concatenate((
            [0.0], np.geomspace(max(a * 1.0e-6, 1.0e-8), sat.cgm_rmax, n_grid)
        ))
        retained = (sat.cgm_rmax / (sat.cgm_rmax + a)) ** 2
        analytic_mass = sat.cgm_mass_rmax / retained
        mass = hernquist_mass_enclosed(r, analytic_mass, a)
        safe_r = np.maximum(r, r[1])
        rho = analytic_mass * a / (2.0*np.pi*safe_r*(safe_r+a)**3)
        rho[0] = rho[1]
        temperature = np.full_like(r, sat.cgm_outer_temperature)
        return {
            "r": r,
            "mass": mass,
            "rho": rho,
            "temperature": temperature,
            "profile": sat.gas_profile,
        }

    if sat.gas_profile != "c25_beta_hydrostatic":
        raise ValueError(
            f"{sat.name} 有气体质量但 gas_profile={sat.gas_profile!r} 不受支持。"
        )

    r = np.concatenate((
        [0.0], np.geomspace(max(sat.cgm_rc * 1.0e-6, 1.0e-8), sat.cgm_rmax, n_grid)
    ))
    shape = (1.0 + (r / sat.cgm_rc) ** 2) ** (-1.5 * sat.cgm_beta)
    raw = cumulative_trapezoid(4.0 * np.pi * r**2 * shape, r, initial=0.0)
    rho = shape * sat.cgm_mass_rmax / raw[-1]
    mass = raw * sat.cgm_mass_rmax / raw[-1]

    enclosed = (
        hernquist_mass_enclosed(r, sat.dm_analytic_mass, sat.dm_a)
        + exponential_disk_mass_enclosed(r, sat.star_mass, sat.star_rd)
        + mass
    )
    gravity = np.zeros_like(r)
    gravity[1:] = G_KPC_KMS2_MSUN * enclosed[1:] / r[1:]**2
    source = rho * gravity
    pressure_integral = -cumulative_trapezoid(
        source[::-1], r[::-1], initial=0.0
    )[::-1]
    outer_scale = (
        K_B_CGS * sat.cgm_outer_temperature / (MU_HOT * M_P_CGS) / 1.0e10
    )
    pressure = pressure_integral + rho[-1] * outer_scale
    temperature = (
        pressure / np.maximum(rho, 1.0e-300) * 1.0e10
        * MU_HOT * M_P_CGS / K_B_CGS
    )
    return {
        "r": r,
        "mass": mass,
        "rho": rho,
        "temperature": temperature,
        "profile": sat.gas_profile,
    }


def cgm_mass_enclosed(r, table):
    if table is None:
        return np.zeros_like(np.asarray(r, dtype=np.float64))
    return np.interp(np.asarray(r), table["r"], table["mass"], left=0.0, right=table["mass"][-1])


def sample_cgm_position_pairs(rng, n: int, table):
    if n % 2:
        raise ValueError("quiet-start CGM 粒子数必须为偶数。")
    half = n // 2
    target = rng.random(half) * table["mass"][-1]
    radius = np.interp(target, table["mass"], table["r"])
    mu = rng.uniform(-1.0, 1.0, half)
    phi = rng.uniform(0.0, 2.0 * np.pi, half)
    sint = np.sqrt(1.0 - mu**2)
    pos_half = np.column_stack((
        radius * sint * np.cos(phi),
        radius * sint * np.sin(phi),
        radius * mu,
    ))
    return np.vstack((pos_half, -pos_half)), np.concatenate((radius, radius))


def sample_gas_internal_velocities(
    rng,
    relative_positions: np.ndarray,
    one_dimensional_sigma: float,
) -> np.ndarray:
    """v1 式各向同性弥散，但严格去掉 COM 漂移和整体径向膨胀/收缩。"""
    if one_dimensional_sigma <= 0.0:
        return np.zeros_like(relative_positions)
    velocity = rng.normal(size=relative_positions.shape)
    radius = np.linalg.norm(relative_positions, axis=1)
    direction = relative_positions / np.maximum(radius[:, None], 1.0e-30)
    for _ in range(4):
        velocity -= np.mean(velocity, axis=0, keepdims=True)
        mean_radial = float(np.mean(np.sum(velocity * direction, axis=1)))
        velocity -= mean_radial * direction
    rms_1d = float(np.sqrt(np.mean(velocity**2)))
    if not np.isfinite(rms_1d) or rms_1d <= 0.0:
        raise RuntimeError("卫星气体内部速度采样失败。")
    velocity *= one_dimensional_sigma / rms_1d
    velocity -= np.mean(velocity, axis=0, keepdims=True)
    mean_radial = float(np.mean(np.sum(velocity * direction, axis=1)))
    velocity -= mean_radial * direction
    return velocity


def build_dm_jeans_table(sat: Satellite, cgm_table, n_grid: int = 16000):
    r = np.geomspace(max(1.0e-5, sat.dm_a * 1.0e-6), sat.dm_rmax, n_grid)
    rho_shape = 1.0 / (r * (r + sat.dm_a) ** 3)
    enclosed = (
        hernquist_mass_enclosed(r, sat.dm_analytic_mass, sat.dm_a)
        + exponential_disk_mass_enclosed(r, sat.star_mass, sat.star_rd)
        + cgm_mass_enclosed(r, cgm_table)
    )
    gravity = G_KPC_KMS2_MSUN * enclosed / r**2
    integral = -cumulative_trapezoid(
        (rho_shape * gravity)[::-1], r[::-1], initial=0.0
    )[::-1]
    sigma = np.sqrt(np.maximum(integral / rho_shape, 0.0))
    approximate_potential_depth = G_KPC_KMS2_MSUN * (
        sat.dm_analytic_mass / (r + sat.dm_a)
        + (sat.star_mass + sat.cgm_mass_rmax) / (r + sat.star_rd)
    )
    vesc = np.sqrt(np.maximum(2.0 * approximate_potential_depth, 0.0))
    return {"r": r, "sigma": sigma, "vesc": vesc}


def sample_dm_velocity_pairs(rng, radii: np.ndarray, table):
    half = len(radii) // 2
    sigma = np.interp(radii[:half], table["r"], table["sigma"])
    vmax = 0.995 * np.interp(radii[:half], table["r"], table["vesc"])
    vel = rng.normal(size=(half, 3)) * sigma[:, None]
    bad = np.sum(vel**2, axis=1) >= vmax**2
    for _ in range(100):
        if not np.any(bad):
            break
        vel[bad] = rng.normal(size=(np.count_nonzero(bad), 3)) * sigma[bad, None]
        bad = np.sum(vel**2, axis=1) >= vmax**2
    if np.any(bad):
        speed = np.linalg.norm(vel[bad], axis=1)
        vel[bad] *= (0.99 * vmax[bad] / np.maximum(speed, 1.0e-20))[:, None]
    return np.vstack((vel, -vel))


def sample_disk_position_pairs(rng, n: int, sat: Satellite):
    if n % 2:
        raise ValueError("quiet-start 盘粒子数必须为偶数。")
    half = n // 2
    rgrid = np.linspace(0.0, sat.star_rmax, 20001)
    x = rgrid / sat.star_rd
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    cdf /= cdf[-1]
    radius = np.interp(rng.random(half), cdf, rgrid)
    phi = rng.uniform(0.0, 2.0 * np.pi, half)
    zlim = np.tanh(sat.star_zmax / sat.star_zd)
    z = sat.star_zd * np.arctanh(rng.uniform(-zlim, zlim, half))
    pos_half = np.column_stack((radius*np.cos(phi), radius*np.sin(phi), z))
    return np.vstack((pos_half, -pos_half)), radius, phi


def build_disk_kinematic_table(sat: Satellite, cgm_table, n_grid: int = 8000):
    """Q=1.5 的近似 epicyclic 盘；足以作为待短时稳定性检查的 live IC。"""
    r = np.geomspace(1.0e-3, sat.star_rmax, n_grid)
    dm_vc2 = (
        G_KPC_KMS2_MSUN
        * hernquist_mass_enclosed(r, sat.dm_analytic_mass, sat.dm_a) / r
    )
    cgm_vc2 = G_KPC_KMS2_MSUN * cgm_mass_enclosed(r, cgm_table) / r
    vc2 = dm_vc2 + cgm_vc2 + thin_exponential_disk_vc2(r, sat.star_mass, sat.star_rd)
    omega2 = vc2 / r**2
    kappa2 = np.maximum(r*np.gradient(omega2, r, edge_order=2) + 4.0*omega2, 1.0e-12)
    kappa = np.sqrt(kappa2)
    surface = sat.star_mass / (2.0*np.pi*sat.star_rd**2) * np.exp(-r/sat.star_rd)
    sigma_r = np.maximum(1.5 * 3.36 * G_KPC_KMS2_MSUN * surface / kappa, 4.0)
    sigma_phi = np.minimum(sigma_r * kappa / (2.0*np.sqrt(omega2)), sigma_r)

    spherical = (
        hernquist_mass_enclosed(r, sat.dm_analytic_mass, sat.dm_a)
        + cgm_mass_enclosed(r, cgm_table)
    )
    sigma_z2 = (
        np.pi * G_KPC_KMS2_MSUN * surface * sat.star_zd
        + G_KPC_KMS2_MSUN * spherical * sat.star_zd**2 / r**3
    )
    sigma_z = np.sqrt(np.maximum(sigma_z2, 4.0))
    derivative = np.gradient(
        -r/sat.star_rd + 2.0*np.log(sigma_r), np.log(r), edge_order=2
    )
    mean_vphi2 = vc2 + sigma_r**2 * (1.0 + derivative) - sigma_phi**2
    return {
        "r": r,
        "sigma_r": sigma_r,
        "sigma_phi": sigma_phi,
        "sigma_z": sigma_z,
        "mean_vphi": np.sqrt(np.maximum(mean_vphi2, 0.0)),
    }


def sample_disk_velocity_pairs(rng, radius, phi, table):
    sr = np.interp(radius, table["r"], table["sigma_r"])
    sp = np.interp(radius, table["r"], table["sigma_phi"])
    sz = np.interp(radius, table["r"], table["sigma_z"])
    vbar = np.interp(radius, table["r"], table["mean_vphi"])
    vr = rng.normal(0.0, sr)
    vphi = rng.normal(vbar, sp)
    vz = rng.normal(0.0, sz)
    c = np.cos(phi)
    s = np.sin(phi)
    vel_half = np.column_stack((vr*c-vphi*s, vr*s+vphi*c, vz))
    return np.vstack((vel_half, -vel_half))


def make_satellite_particles(
    rng,
    sat: Satellite,
    com_position: np.ndarray,
    com_velocity_kms: np.ndarray,
    target_dm_mass: float,
    target_star_mass: float,
    target_gas_mass: float,
):
    """生成单个卫星的 DM、恒星盘和所选气体结构。"""
    cgm_table = build_satellite_cgm_table(sat)
    n_dm = even_count_from_target(sat.dm_mass_rmax, target_dm_mass)
    n_star = even_count_from_target(sat.star_mass, target_star_mass)

    dm_pos, dm_r = sample_hernquist_position_pairs(rng, n_dm, sat)
    dm_jeans = build_dm_jeans_table(sat, cgm_table)
    dm_vel = sample_dm_velocity_pairs(rng, dm_r, dm_jeans)

    star_pos, star_radius, star_phi = sample_disk_position_pairs(rng, n_star, sat)
    disk_kin = build_disk_kinematic_table(sat, cgm_table)
    star_vel = sample_disk_velocity_pairs(rng, star_radius, star_phi, disk_kin)

    # 以插入时轨道角动量方向作为盘法向；这是面向 CGM 扰动的明确简化，
    # 不宣称精确重建 LMC/SMC 的观测盘面姿态。
    normal = np.cross(com_position, com_velocity_kms)
    if np.linalg.norm(normal) < 1.0e-12:
        normal = np.array([0.0, 0.0, 1.0])
    rotation = rotation_from_z_to_vector(normal)
    star_pos = star_pos @ rotation.T
    star_vel = star_vel @ rotation.T

    out = {
        "dm_pos": dm_pos + com_position[None, :],
        "dm_vel": dm_vel + com_velocity_kms[None, :],
        "dm_mass": np.full(n_dm, sat.dm_mass_rmax/n_dm),
        "star_pos": star_pos + com_position[None, :],
        "star_vel": star_vel + com_velocity_kms[None, :],
        "star_mass": np.full(n_star, sat.star_mass/n_star),
        "disk_normal": normal / np.linalg.norm(normal),
    }

    if cgm_table is not None:
        n_gas = even_count_from_target(sat.cgm_mass_rmax, target_gas_mass)
        gas_pos, gas_radius = sample_cgm_position_pairs(rng, n_gas, cgm_table)
        gas_temp = np.interp(gas_radius, cgm_table["r"], cgm_table["temperature"])
        gas_internal_vel = sample_gas_internal_velocities(
            rng, gas_pos, sat.gas_velocity_dispersion
        )
        gas_direction = gas_pos / np.maximum(gas_radius[:, None], 1.0e-30)
        gas_mean_radial_velocity = float(np.mean(np.sum(
            gas_internal_vel * gas_direction, axis=1
        )))
        gas_bulk_velocity_residual = np.mean(gas_internal_vel, axis=0)
        out.update({
            "gas_pos": gas_pos + com_position[None, :],
            "gas_vel": gas_internal_vel + com_velocity_kms[None, :],
            "gas_mass": np.full(n_gas, sat.cgm_mass_rmax/n_gas),
            "gas_u": internal_energy_from_temperature(
                gas_temp, sat.gas_mean_molecular_weight
            ),
            "gas_temperature": gas_temp,
            "gas_profile": sat.gas_profile,
            "gas_metallicity_zsun": sat.cgm_metallicity_zsun,
            "gas_mean_radial_velocity": gas_mean_radial_velocity,
            "gas_bulk_velocity_residual": gas_bulk_velocity_residual,
        })
    return out


# =============================================================================
# 5. 读取宿主、合并并写出新的 GIZMO IC
# =============================================================================

def read_host_particles(handle, center, velocity):
    """只保留重新作为 IC 所需的字段；snapshot 派生诊断字段会由 GIZMO 重算。"""
    mass_table = np.asarray(handle["Header"].attrs.get("MassTable", np.zeros(6)))
    particles = {}
    for ptype in range(6):
        name = f"PartType{ptype}"
        if name not in handle:
            continue
        group = handle[name]
        item = {
            "Coordinates": np.asarray(group["Coordinates"], dtype=np.float64) - center[None, :],
            "Velocities": np.asarray(group["Velocities"], dtype=np.float64) - velocity[None, :],
            "ParticleIDs": np.asarray(group["ParticleIDs"], dtype=np.uint64),
            "Masses": masses_msun(group, float(mass_table[ptype])),
        }
        for optional in (
            "InternalEnergy", "Metallicity", "StellarFormationTime",
            "StellarComponent", "BH_Mass", "ParticleChildIDsNumber",
            "ParticleIDGenerationNumber",
        ):
            if optional in group:
                item[optional] = np.asarray(group[optional])
        if "BH_Mass" in item:
            # 快照中的 BH_Mass 与 Masses 一样采用 1e10 Msun 的代码质量单位。
            item["BH_Mass"] = np.asarray(item["BH_Mass"], dtype=np.float64) * UNIT_MASS_MSUN
        particles[ptype] = item
    return particles


def make_ids(start: np.uint64, n: int, label: str):
    if n >= ID_BLOCK_SIZE:
        raise RuntimeError(f"{label} 粒子数 {n} 超过预留的 {ID_BLOCK_SIZE} ID 子区间。")
    return np.arange(start, start + np.uint64(n), dtype=np.uint64)


def append_zero_bookkeeping_fields(item: dict, n_new: int) -> None:
    """为新加粒子补齐宿主快照中已有的 ID-generation 记账字段。"""
    for name in ("ParticleChildIDsNumber", "ParticleIDGenerationNumber"):
        if name not in item:
            continue
        base = np.asarray(item[name])
        extra = np.zeros((n_new,) + base.shape[1:], dtype=base.dtype)
        item[name] = np.concatenate((base, extra), axis=0)


def append_satellites_to_host(host, lmc_data, smc_data, gas_metal_cols: int):
    # PartType0：MW CGM/ISM + 按所选模型加入的 LMC/SMC 气体。
    gas = host[0]
    n_lmc_gas = len(lmc_data.get("gas_pos", []))
    n_smc_gas = len(smc_data.get("gas_pos", []))
    n_satellite_gas = n_lmc_gas + n_smc_gas
    if n_satellite_gas:
        if "InternalEnergy" not in gas or "Metallicity" not in gas:
            raise RuntimeError("宿主 PartType0 必须含 InternalEnergy 和 Metallicity。")
        gas_batches = []
        if n_lmc_gas:
            gas_batches.append((
                "LMC", lmc_data, n_lmc_gas, LMC_GAS_ID_START,
            ))
        if n_smc_gas:
            gas_batches.append((
                "SMC", smc_data, n_smc_gas, SMC_GAS_ID_START,
            ))
        gas["Coordinates"] = np.vstack(
            [gas["Coordinates"]] + [item[1]["gas_pos"] for item in gas_batches]
        )
        gas["Velocities"] = np.vstack(
            [gas["Velocities"]] + [item[1]["gas_vel"] for item in gas_batches]
        )
        gas["ParticleIDs"] = np.concatenate(
            [gas["ParticleIDs"]] + [
                make_ids(item[3], item[2], f"{item[0]} gas") for item in gas_batches
            ]
        )
        gas["Masses"] = np.concatenate(
            [gas["Masses"]] + [item[1]["gas_mass"] for item in gas_batches]
        )
        gas["InternalEnergy"] = np.concatenate(
            [np.ravel(gas["InternalEnergy"])]
            + [item[1]["gas_u"] for item in gas_batches]
        )
        metal_batches = [gas["Metallicity"]]
        for _, data, count, _ in gas_batches:
            metal_batches.append(np.broadcast_to(
                stellar_metallicity_vector(
                    float(data["gas_metallicity_zsun"]), gas_metal_cols
                ),
                (count, gas_metal_cols),
            ))
        gas["Metallicity"] = np.vstack(metal_batches)
        append_zero_bookkeeping_fields(gas, n_satellite_gas)

    # PartType1：MW + LMC + SMC live DM。
    dm = host[1]
    n_lmc_dm = len(lmc_data["dm_pos"])
    n_smc_dm = len(smc_data["dm_pos"])
    dm["Coordinates"] = np.vstack((dm["Coordinates"], lmc_data["dm_pos"], smc_data["dm_pos"]))
    dm["Velocities"] = np.vstack((dm["Velocities"], lmc_data["dm_vel"], smc_data["dm_vel"]))
    dm["ParticleIDs"] = np.concatenate((
        dm["ParticleIDs"],
        make_ids(LMC_DM_ID_START, n_lmc_dm, "LMC DM"),
        make_ids(SMC_DM_ID_START, n_smc_dm, "SMC DM"),
    ))
    dm["Masses"] = np.concatenate((dm["Masses"], lmc_data["dm_mass"], smc_data["dm_mass"]))
    append_zero_bookkeeping_fields(dm, n_lmc_dm + n_smc_dm)

    # PartType4：MW + LMC + SMC 老恒星；StellarComponent 11/21 标记卫星。
    stars = host[4]
    n_lmc_star = len(lmc_data["star_pos"])
    n_smc_star = len(smc_data["star_pos"])
    stars["Coordinates"] = np.vstack((
        stars["Coordinates"], lmc_data["star_pos"], smc_data["star_pos"]
    ))
    stars["Velocities"] = np.vstack((
        stars["Velocities"], lmc_data["star_vel"], smc_data["star_vel"]
    ))
    stars["ParticleIDs"] = np.concatenate((
        stars["ParticleIDs"],
        make_ids(LMC_STAR_ID_START, n_lmc_star, "LMC stars"),
        make_ids(SMC_STAR_ID_START, n_smc_star, "SMC stars"),
    ))
    stars["Masses"] = np.concatenate((
        stars["Masses"], lmc_data["star_mass"], smc_data["star_mass"]
    ))
    append_zero_bookkeeping_fields(stars, n_lmc_star + n_smc_star)

    if "Metallicity" in stars:
        ncol = 1 if stars["Metallicity"].ndim == 1 else stars["Metallicity"].shape[1]
        lmc_z = np.broadcast_to(stellar_metallicity_vector(0.5, ncol), (n_lmc_star, ncol))
        smc_z = np.broadcast_to(stellar_metallicity_vector(0.2, ncol), (n_smc_star, ncol))
        base = stars["Metallicity"]
        if base.ndim == 1:
            base = base[:, None]
        stars["Metallicity"] = np.vstack((base, lmc_z, smc_z))
    if "StellarFormationTime" in stars:
        old_tform = float(np.median(np.ravel(stars["StellarFormationTime"])))
        stars["StellarFormationTime"] = np.concatenate((
            np.ravel(stars["StellarFormationTime"]),
            # 继承宿主老恒星的内部时间单位，而不是误把 -10 当成 -10 Gyr。
            np.full(n_lmc_star+n_smc_star, old_tform),
        ))
    component = stars.get(
        "StellarComponent", np.zeros(len(stars["Coordinates"])-n_lmc_star-n_smc_star, dtype=np.uint8)
    )
    stars["StellarComponent"] = np.concatenate((
        np.ravel(component).astype(np.uint8),
        np.full(n_lmc_star, 11, dtype=np.uint8),
        np.full(n_smc_star, 21, dtype=np.uint8),
    ))


def barycentric_recenter(particles, box_size: float):
    """保持相对相空间不变，同时令整个 MW+LMC+SMC 系统 COM/P=0。"""
    total_mass = 0.0
    weighted_position = np.zeros(3)
    weighted_velocity = np.zeros(3)
    for item in particles.values():
        mass = item["Masses"]
        total_mass += float(np.sum(mass))
        weighted_position += np.sum(item["Coordinates"] * mass[:, None], axis=0)
        weighted_velocity += np.sum(item["Velocities"] * mass[:, None], axis=0)
    com_position = weighted_position / total_mass
    com_velocity = weighted_velocity / total_mass
    box_center = np.full(3, 0.5*box_size)
    for item in particles.values():
        item["Coordinates"] = item["Coordinates"] - com_position[None, :] + box_center[None, :]
        item["Velocities"] = item["Velocities"] - com_velocity[None, :]
    return com_position, com_velocity


def write_dataset(group, name: str, data):
    array = np.asarray(data)
    if name == "ParticleIDs":
        array = array.astype(np.uint32)
    elif array.dtype.kind == "f":
        array = array.astype(np.float64)
    group.create_dataset(name, data=array)


def write_output(
    header_source,
    metadata_source,
    output_path: str,
    particles,
    box_size: float,
    orbit_metadata: dict,
    lmc: Satellite,
    smc: Satellite,
    reference_time: np.ndarray,
    reference_states: np.ndarray,
):
    counts = np.zeros(6, dtype=np.uint32)
    for ptype, item in particles.items():
        counts[ptype] = len(item["Coordinates"])

    with h5py.File(output_path, "w") as out:
        header = out.create_group("Header")
        for key, value in header_source["Header"].attrs.items():
            header.attrs[key] = value
        header.attrs["NumPart_ThisFile"] = counts
        header.attrs["NumPart_Total"] = counts
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = float(orbit_metadata["INSERTION_TIME_CODE"])
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = box_size
        header.attrs["Flag_Cooling"] = 1
        header.attrs["Flag_Metals"] = 1
        header.attrs["Flag_Sfr"] = 1
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1

        for ptype in sorted(particles):
            group = out.create_group(f"PartType{ptype}")
            item = particles[ptype]
            for name in (
                "Coordinates", "Velocities", "ParticleIDs", "Masses",
                "InternalEnergy", "Metallicity", "StellarFormationTime",
                "StellarComponent", "BH_Mass", "ParticleChildIDsNumber",
                "ParticleIDGenerationNumber",
            ):
                if name not in item:
                    continue
                data = item[name]
                if name in ("Masses", "BH_Mass"):
                    data = np.asarray(data, dtype=np.float64) / UNIT_MASS_MSUN
                write_dataset(group, name, data)

        # 保留宿主生成器的说明组；snapshot 若没有，可由 --host-reference-ic 复制。
        for name in ("MWModelReference", "ModelNotes", "ReservedParticleIDs"):
            if name in metadata_source and name not in out:
                metadata_source.copy(name, out)
        if "ModelNotes" in out:
            out["ModelNotes"].attrs["LMC_SMC_PRESENT"] = 1
            out["ModelNotes"].attrs["LMC_HOT_CGM_PRESENT"] = int(
                lmc.gas_profile == "c25_beta_hydrostatic"
            )
            out["ModelNotes"].attrs["LMC_SMC_COMPACT_WARM_GAS_PRESENT"] = int(
                lmc.gas_profile == "v1_compact_hernquist"
                or smc.gas_profile == "v1_compact_hernquist"
            )
            out["ModelNotes"].attrs["LMC_SMC_COLD_ISM_PRESENT"] = 0
            out["ModelNotes"].attrs["AGN_FEEDBACK_MUST_BE_OFF"] = 1
            out["ModelNotes"].attrs["GALSF_EFFECTIVE_EQS_MUST_BE_ON"] = 1
            out["ModelNotes"].attrs["EXPLICIT_STELLAR_FEEDBACK_MUST_BE_OFF"] = 1
            out["ModelNotes"].attrs["STAGE"] = (
                "LMC/SMC infall and K13 endpoint calibration before AGN jet"
            )

        if "ReservedParticleIDs" in out:
            rid = out["ReservedParticleIDs"]
            exact_ranges = {
                "LMC_GAS_ID_START": LMC_GAS_ID_START,
                "LMC_GAS_ID_END": int(LMC_GAS_ID_START) + int(orbit_metadata["N_LMC_GAS"]) - 1,
                "LMC_DM_ID_START": LMC_DM_ID_START,
                "LMC_DM_ID_END": LMC_DM_ID_START + int(orbit_metadata["N_LMC_DM"]) - 1,
                "LMC_STAR_ID_START": LMC_STAR_ID_START,
                "LMC_STAR_ID_END": LMC_STAR_ID_START + int(orbit_metadata["N_LMC_STARS"]) - 1,
                "SMC_DM_ID_START": SMC_DM_ID_START,
                "SMC_DM_ID_END": SMC_DM_ID_START + int(orbit_metadata["N_SMC_DM"]) - 1,
                "SMC_STAR_ID_START": SMC_STAR_ID_START,
                "SMC_STAR_ID_END": SMC_STAR_ID_START + int(orbit_metadata["N_SMC_STARS"]) - 1,
                "SMC_GAS_ID_START": SMC_GAS_ID_START,
                "SMC_GAS_ID_END": int(SMC_GAS_ID_START) + int(orbit_metadata["N_SMC_GAS"]) - 1,
            }
            for key, value in exact_ranges.items():
                rid.attrs[key] = np.uint64(value)

        ref = out.create_group("MagellanicModelReference")
        ref.attrs["MODEL_NAME"] = (
            "C25_B12_dynamics_K13_endpoint_orbit_v4_"
            + str(orbit_metadata["SATELLITE_GAS_MODEL"])
        )
        ref.attrs["MODEL_IS_EXACT_COPY_OF_ONE_PAPER"] = 0
        ref.attrs["ORBIT_STATUS"] = str(orbit_metadata["TRIAL_LABEL"])
        ref.attrs["REFERENCE_K13"] = "Kallivayalil et al. 2013, ApJ 764, 161; arXiv:1301.0832"
        ref.attrs["REFERENCE_C25"] = "Carr et al. 2025, ApJ 983, 151; arXiv:2408.10358"
        ref.attrs["REFERENCE_B12"] = "Besla et al. 2012, MNRAS 421, 2109; arXiv:1201.1299"
        ref.attrs["REFERENCE_TG19_COMPARISON_ONLY"] = "Tepper-Garcia et al. 2019, MNRAS 488, 918; arXiv:1901.05636"
        ref.attrs["REFERENCE_H90"] = "Hernquist 1990, ApJ 356, 359"
        ref.attrs["DEFAULT_SCIENCE_SCOPE"] = "MW CGM response + later AGN jet; not Magellanic Stream"
        ref.attrs["COLD_LMC_SMC_ISM_INCLUDED"] = 0
        ref.attrs["SATELLITE_GAS_MODEL"] = str(orbit_metadata["SATELLITE_GAS_MODEL"])
        ref.attrs["LMC_CGM_MODE"] = str(orbit_metadata["LMC_CGM_MODE"])
        if lmc.gas_profile == "c25_beta_hydrostatic":
            ref.attrs["LMC_CGM_COOLING_CORRECTION"] = (
                f"rc={lmc.cgm_rc:g} kpc instead of C25 rc=0.2 kpc; "
                f"mode={orbit_metadata['LMC_CGM_MODE']}; "
                f"retained M_CGM={lmc.cgm_mass_rmax:.8e} Msun within {lmc.cgm_rmax:g} kpc"
            )
        else:
            ref.attrs["LMC_CGM_COOLING_CORRECTION"] = "not_applicable"
        for prefix, sat in (("LMC", lmc), ("SMC", smc)):
            ref.attrs[f"{prefix}_DM_MASS_RMAX_Msun"] = sat.dm_mass_rmax
            ref.attrs[f"{prefix}_DM_HERNQUIST_A_kpc"] = sat.dm_a
            ref.attrs[f"{prefix}_DM_RMAX_kpc"] = sat.dm_rmax
            ref.attrs[f"{prefix}_STAR_MASS_Msun"] = sat.star_mass
            ref.attrs[f"{prefix}_STAR_RD_kpc"] = sat.star_rd
            ref.attrs[f"{prefix}_STAR_ZD_kpc"] = sat.star_zd
            ref.attrs[f"{prefix}_CGM_MASS_RMAX_Msun"] = sat.cgm_mass_rmax
            ref.attrs[f"{prefix}_GAS_PROFILE"] = sat.gas_profile
            ref.attrs[f"{prefix}_GAS_MASS_Msun"] = sat.cgm_mass_rmax
            ref.attrs[f"{prefix}_GAS_SCALE_kpc"] = sat.cgm_rc
            ref.attrs[f"{prefix}_GAS_RMAX_kpc"] = sat.cgm_rmax
            ref.attrs[f"{prefix}_GAS_TEMPERATURE_K"] = sat.cgm_outer_temperature
            ref.attrs[f"{prefix}_GAS_SIGMA_km_s"] = sat.gas_velocity_dispersion
            ref.attrs[f"{prefix}_GAS_MEAN_MOLECULAR_WEIGHT"] = sat.gas_mean_molecular_weight
        for key, value in orbit_metadata.items():
            if isinstance(value, (str, int, float, np.number)):
                ref.attrs[key] = value
            else:
                ref.attrs[key] = np.asarray(value)

        # 与 validation_dashboard_sat_6d.py 保持完全一致的接口。这里保存的是
        # 球平均实际宿主势中的“名义参考轨道”；带 shooting offset 的粒子初值
        # 单独记录，验证脚本仍以 K13 今天终点作为最终判据。
        oref = out.create_group("MagellanicOrbitReference")
        oref.attrs["Reference"] = (
            "Kallivayalil et al. 2013 endpoint; Carr2025/Besla2012 structure"
        )
        oref.attrs["Potential"] = (
            "spherical relaxed-MW profile + LMC-SMC gravity + MW recoil"
        )
        oref.attrs["CoordinateFrame"] = (
            "right-handed K13 Galactocentric XYZ; X Sun->GC, "
            "Y Galactic rotation, Z north"
        )
        oref.attrs["Lookback_Gyr"] = float(orbit_metadata["LOOKBACK_GYR"])
        oref.attrs["InsertionTime_Code"] = float(orbit_metadata["INSERTION_TIME_CODE"])
        oref.attrs["PresentTargetTime_Code"] = float(orbit_metadata["PRESENT_TARGET_TIME_CODE"])
        oref.attrs["UnitTime_Gyr"] = UNIT_TIME_GYR
        oref.attrs["LMC_ID_START"] = LMC_ID_START
        oref.attrs["LMC_ID_END"] = LMC_ID_END
        oref.attrs["SMC_ID_START"] = SMC_ID_START
        oref.attrs["SMC_ID_END"] = SMC_ID_END
        oref.attrs["MW_BH_ID"] = MW_BH_ID
        oref.attrs["LMC_GAS_ID_START"] = LMC_GAS_ID_START
        n_lmc_gas = int(orbit_metadata["N_LMC_GAS"])
        oref.attrs["LMC_GAS_ID_END"] = (
            int(LMC_GAS_ID_START) + n_lmc_gas - 1
        )
        oref.attrs["LMC_DM_ID_START"] = LMC_DM_ID_START
        oref.attrs["LMC_DM_ID_END"] = (
            LMC_DM_ID_START + int(orbit_metadata["N_LMC_DM"]) - 1
        )
        oref.attrs["LMC_STAR_ID_START"] = LMC_STAR_ID_START
        oref.attrs["LMC_STAR_ID_END"] = (
            LMC_STAR_ID_START + int(orbit_metadata["N_LMC_STARS"]) - 1
        )
        oref.attrs["SMC_DM_ID_START"] = SMC_DM_ID_START
        oref.attrs["SMC_DM_ID_END"] = (
            SMC_DM_ID_START + int(orbit_metadata["N_SMC_DM"]) - 1
        )
        oref.attrs["SMC_STAR_ID_START"] = SMC_STAR_ID_START
        oref.attrs["SMC_STAR_ID_END"] = (
            SMC_STAR_ID_START + int(orbit_metadata["N_SMC_STARS"]) - 1
        )
        oref.attrs["SMC_GAS_ID_START"] = SMC_GAS_ID_START
        n_smc_gas = int(orbit_metadata["N_SMC_GAS"])
        oref.attrs["SMC_GAS_ID_END"] = (
            int(SMC_GAS_ID_START) + n_smc_gas - 1
        )
        oref.attrs["LMC_TotalMass_Msun"] = lmc.live_mass
        oref.attrs["SMC_TotalMass_Msun"] = smc.live_mass
        oref.attrs["LMC_CGM_Mass_Msun"] = lmc.cgm_mass_rmax
        oref.attrs["LMC_CGM_Rmax_kpc"] = lmc.cgm_rmax
        oref.attrs["LMC_CGM_CoreRadius_kpc"] = lmc.cgm_rc
        oref.attrs["LMC_GasProfile"] = lmc.gas_profile
        oref.attrs["SMC_GasProfile"] = smc.gas_profile
        oref.attrs["LMC_GasMass_Msun"] = lmc.cgm_mass_rmax
        oref.attrs["SMC_GasMass_Msun"] = smc.cgm_mass_rmax
        oref.attrs["LMC_GasRmax_kpc"] = lmc.cgm_rmax
        oref.attrs["SMC_GasRmax_kpc"] = smc.cgm_rmax
        oref.attrs["ForwardClosurePositionVelocityMax"] = float(
            np.max(np.abs(orbit_metadata["STATIC_ORBIT_CLOSURE_RESIDUAL"]))
        )
        oref.create_dataset("TimeFromPresent_Gyr", data=np.asarray(reference_time))
        oref.create_dataset("LMC_Position_kpc", data=reference_states[:, 0:3])
        oref.create_dataset("LMC_Velocity_kms", data=reference_states[:, 3:6])
        oref.create_dataset("SMC_Position_kpc", data=reference_states[:, 6:9])
        oref.create_dataset("SMC_Velocity_kms", data=reference_states[:, 9:12])
        oref.create_dataset("LMC_Current_PositionSigma_kpc", data=K13_LMC_POSITION_SIGMA_KPC)
        oref.create_dataset("SMC_Current_PositionSigma_kpc", data=K13_SMC_POSITION_SIGMA_KPC)
        oref.create_dataset("LMC_Current_VelocitySigma_kms", data=K13_LMC_VELOCITY_SIGMA_KMS)
        oref.create_dataset("SMC_Current_VelocitySigma_kms", data=K13_SMC_VELOCITY_SIGMA_KMS)
        oref.create_dataset(
            "LMC_AppliedInitialPosition_kpc",
            data=np.asarray(orbit_metadata["LMC_INITIAL_POSITION_RELATIVE_MW_kpc"]),
        )
        oref.create_dataset(
            "LMC_AppliedInitialVelocity_kms",
            data=np.asarray(orbit_metadata["LMC_INITIAL_VELOCITY_RELATIVE_MW_km_s"]),
        )
        oref.create_dataset(
            "SMC_AppliedInitialPosition_kpc",
            data=np.asarray(orbit_metadata["SMC_INITIAL_POSITION_RELATIVE_MW_kpc"]),
        )
        oref.create_dataset(
            "SMC_AppliedInitialVelocity_kms",
            data=np.asarray(orbit_metadata["SMC_INITIAL_VELOCITY_RELATIVE_MW_km_s"]),
        )
        for name, key in (
            ("LMC_ShootingPositionOffset_kpc", "LMC_SHOOTING_POSITION_OFFSET_kpc"),
            ("LMC_ShootingVelocityOffset_kms", "LMC_SHOOTING_VELOCITY_OFFSET_km_s"),
            ("SMC_ShootingPositionOffset_kpc", "SMC_SHOOTING_POSITION_OFFSET_kpc"),
            ("SMC_ShootingVelocityOffset_kms", "SMC_SHOOTING_VELOCITY_OFFSET_km_s"),
        ):
            oref.create_dataset(name, data=np.asarray(orbit_metadata[key]))


def validate_output(path: str, box_size: float):
    with h5py.File(path, "r") as handle:
        counts = np.asarray(handle["Header"].attrs["NumPart_Total"], dtype=np.int64)
        all_ids = []
        for ptype, expected in enumerate(counts):
            name = f"PartType{ptype}"
            if expected == 0:
                continue
            if name not in handle:
                raise RuntimeError(f"输出缺少 {name}。")
            group = handle[name]
            for field in ("Coordinates", "Velocities", "ParticleIDs", "Masses"):
                if field not in group:
                    raise RuntimeError(f"{name} 缺少 {field}。")
            if len(group["Coordinates"]) != expected:
                raise RuntimeError(f"{name} 粒子数与 Header 不一致。")
            for field_name, dataset in group.items():
                if dataset.ndim > 0 and dataset.shape[0] != expected:
                    raise RuntimeError(
                        f"{name}/{field_name} 首维 {dataset.shape[0]} 与粒子数 {expected} 不一致。"
                    )
            pos = np.asarray(group["Coordinates"])
            vel = np.asarray(group["Velocities"])
            mass = np.asarray(group["Masses"])
            if np.any(~np.isfinite(pos)) or np.any(~np.isfinite(vel)):
                raise RuntimeError(f"{name} 存在 NaN/Inf 坐标或速度。")
            if np.any(pos < 0.0) or np.any(pos >= box_size):
                print(f"[警告] {name} 有粒子在 [0, {box_size}] 范围外；"
                      f"如果未启用 BOX_PERIODIC，物理无影响。")
            if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
                raise RuntimeError(f"{name} 质量非法。")
            all_ids.append(np.asarray(group["ParticleIDs"], dtype=np.uint64))
        ids = np.concatenate(all_ids)
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError("输出 ParticleIDs 存在重复。")
        if "PartType0/InternalEnergy" not in handle or "PartType0/Metallicity" not in handle:
            raise RuntimeError("PartType0 缺少 cooling/metals 所需字段。")

        oref = handle["MagellanicOrbitReference"].attrs
        for label, ptype, start_key, end_key in (
            ("LMC gas", 0, "LMC_GAS_ID_START", "LMC_GAS_ID_END"),
            ("SMC gas", 0, "SMC_GAS_ID_START", "SMC_GAS_ID_END"),
            ("LMC stars", 4, "LMC_STAR_ID_START", "LMC_STAR_ID_END"),
            ("SMC stars", 4, "SMC_STAR_ID_START", "SMC_STAR_ID_END"),
            ("LMC DM", 1, "LMC_DM_ID_START", "LMC_DM_ID_END"),
            ("SMC DM", 1, "SMC_DM_ID_START", "SMC_DM_ID_END"),
        ):
            pids = np.asarray(handle[f"PartType{ptype}/ParticleIDs"], dtype=np.uint64)
            start, end = int(oref[start_key]), int(oref[end_key])
            actual = int(np.count_nonzero((pids >= start) & (pids <= end)))
            expected = end - start + 1
            if actual != expected:
                raise RuntimeError(f"{label} ID 计数 {actual} != 期望 {expected}。")

        model_ref = handle["MagellanicModelReference"].attrs
        for prefix in ("LMC", "SMC"):
            if int(model_ref[f"N_{prefix}_GAS"]) == 0:
                continue
            mean_radial = float(
                model_ref[f"{prefix}_GAS_INITIAL_MEAN_RADIAL_VELOCITY_km_s"]
            )
            bulk_residual = np.asarray(
                model_ref[f"{prefix}_GAS_INITIAL_BULK_RESIDUAL_km_s"],
                dtype=np.float64,
            )
            if abs(mean_radial) > 1.0e-8:
                raise RuntimeError(
                    f"{prefix} 气体初始平均径向速度 {mean_radial:.3e} km/s 非零。"
                )
            if np.linalg.norm(bulk_residual) > 1.0e-8:
                raise RuntimeError(
                    f"{prefix} 气体未严格继承卫星 COM 速度：{bulk_residual} km/s。"
                )
        return counts


# =============================================================================
# 6. 命令行和主流程
# =============================================================================

def vector_argument(text: str) -> np.ndarray:
    values = [float(piece) for piece in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("三维向量必须写成 x,y,z。")
    return np.array(values, dtype=np.float64)


def load_present_residuals(path: str):
    """读取 validator 的逐分量终点残差，顺序固定为 x,y,z,vx,vy,vz。"""
    required = {
        (satellite, component)
        for satellite in ("LMC", "SMC")
        for component in ("x", "y", "z", "vx", "vy", "vz")
    }
    values = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_columns = {"satellite", "component", "residual"}
        if reader.fieldnames is None or not expected_columns.issubset(reader.fieldnames):
            raise RuntimeError(
                f"{path} 不是配套 validator 输出的 magellanic_present_6d.csv。"
            )
        for row in reader:
            key = (str(row["satellite"]).upper(), str(row["component"]).lower())
            if key not in required:
                continue
            if key in values:
                raise RuntimeError(f"{path} 中 {key[0]} {key[1]} 出现重复行。")
            value = float(row["residual"])
            if not np.isfinite(value):
                raise RuntimeError(f"{path} 中 {key[0]} {key[1]} 残差不是有限数。")
            values[key] = value
    missing = sorted(required.difference(values))
    if missing:
        raise RuntimeError(f"{path} 缺少 6D 残差行：{missing}")

    def packed(satellite):
        position = np.array(
            [values[(satellite, key)] for key in ("x", "y", "z")], dtype=float
        )
        velocity = np.array(
            [values[(satellite, key)] for key in ("vx", "vy", "vz")], dtype=float
        )
        return position, velocity

    return (*packed("LMC"), *packed("SMC"))


def load_base_shooting_offsets(path: str, expected_gas_model: str):
    """读取同一气体模型上一轮 IC 的累计 12D offset，供增量 shooting。"""
    keys = (
        "LMC_SHOOTING_POSITION_OFFSET_kpc",
        "LMC_SHOOTING_VELOCITY_OFFSET_km_s",
        "SMC_SHOOTING_POSITION_OFFSET_kpc",
        "SMC_SHOOTING_VELOCITY_OFFSET_km_s",
    )
    with h5py.File(path, "r") as handle:
        if "MagellanicModelReference" not in handle:
            raise RuntimeError(f"{path} 缺少 MagellanicModelReference。")
        attrs = handle["MagellanicModelReference"].attrs
        stored_gas_model = attrs.get("SATELLITE_GAS_MODEL")
        if stored_gas_model is None:
            # 没有该属性的是修改前的旧 v4，默认对应 C25 hot corona。
            stored_gas_model = "c25-hot-corona"
        if isinstance(stored_gas_model, bytes):
            stored_gas_model = stored_gas_model.decode("utf-8")
        stored_gas_model = str(stored_gas_model)
        if stored_gas_model != expected_gas_model:
            raise RuntimeError(
                f"{path} 的卫星气体模型是 {stored_gas_model!r}，当前选择是 "
                f"{expected_gas_model!r}。改变气体/总质量后不能沿用旧 shooting "
                "offset；请从新模型 trial00 重新校准。"
            )
        missing = [key for key in keys if key not in attrs]
        if missing:
            raise RuntimeError(f"{path} 缺少累计 shooting offset：{missing}")
        values = tuple(np.asarray(attrs[key], dtype=np.float64) for key in keys)
    if any(value.shape != (3,) or np.any(~np.isfinite(value)) for value in values):
        raise RuntimeError(f"{path} 的 shooting offset 必须是有限三维向量。")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 C25/B12 结构的 live LMC/SMC 加入弛豫 MW，并以 K13 为今天终点。"
    )
    parser.add_argument(
        "--host-snapshot", "--input", dest="host_snapshot", required=True,
        help="通过检查的孤立 MW+CGM snapshot；--input 是兼容旧脚本的别名。",
    )
    parser.add_argument(
        "--host-reference-ic", default=None,
        help="可选：若 snapshot 未保留 MWModelReference，则从此 Stage-1 IC 复制说明组。",
    )
    parser.add_argument("--output", default="mw_relaxed_lmc_smc_trial00.hdf5")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆盖已经存在的输出 IC 和同名 .orbit.json；默认拒绝。",
    )
    parser.add_argument("--seed", type=int, default=20250908)
    parser.add_argument(
        "--lookback-gyr", type=float, default=1.0,
        help="从观测今天反推的物理时间 [Gyr]；gas-wake 主方案默认 1.0。",
    )
    parser.add_argument(
        "--stage2-start-code", type=float, default=0.0,
        help="新 Stage-2 IC 的 GIZMO 起始代码时间；默认重新从 0 开始。",
    )
    parser.add_argument(
        "--orbit-samples", type=int, default=501,
        help="写入 MagellanicOrbitReference 的参考轨道采样数。",
    )
    parser.add_argument("--dynamical-friction", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--host-recoil", type=int, choices=(0, 1), default=1,
        help="在 MW-centered 反推方程中加入 LMC/SMC 对 MW 中心的 recoil。",
    )
    parser.add_argument("--coulomb-log", type=float, default=3.0)
    parser.add_argument(
        "--satellite-dm-particle-mass", type=float, default=0.0,
        help="Msun；0=自动取 min(宿主 DM 中位质量,1e6)。lowres 可设 1e7。",
    )
    parser.add_argument(
        "--satellite-star-particle-mass", type=float, default=0.0,
        help="Msun；0=自动取 min(宿主恒星中位质量,1e5)。lowres 可设 1e6。",
    )
    parser.add_argument(
        "--satellite-gas-particle-mass", type=float, default=0.0,
        help="Msun；0=继承宿主 CGM 中位质量。lowres 可设 5e5。",
    )
    parser.add_argument(
        "--satellite-gas-model",
        choices=("v1-compact", "c25-hot-corona", "none"),
        default="v1-compact",
        help=(
            "默认 v1-compact：LMC+SMC 紧致 5e4 K 暖气体，产生定向冲压；"
            "c25-hot-corona：恢复旧 v4 的仅 LMC 热 beta 晕；none：纯引力。"
        ),
    )
    parser.add_argument(
        "--lmc-cgm-core-radius", type=float, default=1.0,
        help="仅 c25-hot-corona 使用：kpc；默认 1.0，C25 原参数对照用 0.2。",
    )
    parser.add_argument(
        "--lmc-cgm-mode", choices=("truncated", "none", "full"),
        default="truncated",
        help=(
            "仅 c25-hot-corona 使用；truncated=默认，none=去掉热晕，full=Carr2025 的 "
            "M(<117 kpc)=2.4e9 Msun 原始 corona。"
        ),
    )
    parser.add_argument(
        "--lmc-cgm-rmax", type=float, default=DEFAULT_LMC_CGM_TRUNCATION_KPC,
        help=(
            "truncated 模式的截断半径 [kpc]；质量按 Carr beta-profile 同密度归一化缩放，"
            "默认 50 kpc 对应约 6.7e8 Msun。"
        ),
    )
    parser.add_argument(
        "--minimum-lmc-cgm-edge-kpc", type=float, default=50.0,
        help="LMC 气体近边缘距 MW 中心的安全下限 [kpc]。",
    )
    parser.add_argument(
        "--allow-strong-cgm-overlap", action="store_true",
        help="显式允许卫星气体初始近边缘深入安全下限；仅用于受控对照。",
    )
    parser.add_argument(
        "--lmc-target-position", type=vector_argument, default=K13_LMC_POSITION_KPC,
        help="今天的目标位置；默认 -0.8,-41.5,-26.9 kpc。",
    )
    parser.add_argument(
        "--lmc-target-velocity", type=vector_argument, default=K13_LMC_VELOCITY_KMS,
        help="今天的目标速度；默认 -57,-226,221 km/s。",
    )
    parser.add_argument(
        "--smc-target-position", type=vector_argument, default=K13_SMC_POSITION_KPC,
        help="今天的目标位置；默认 15.3,-36.9,-43.3 kpc。",
    )
    parser.add_argument(
        "--smc-target-velocity", type=vector_argument, default=K13_SMC_VELOCITY_KMS,
        help="今天的目标速度；默认 19,-153,153 km/s。",
    )
    parser.add_argument(
        "--lmc-dr-kpc", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"), help="加到反推 LMC 初始位置上的 shooting 修正。",
    )
    parser.add_argument(
        "--lmc-dv-kms", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        metavar=("DVX", "DVY", "DVZ"), help="加到反推 LMC 初始速度上的 shooting 修正。",
    )
    parser.add_argument(
        "--smc-dr-kpc", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"), help="加到反推 SMC 初始位置上的 shooting 修正。",
    )
    parser.add_argument(
        "--smc-dv-kms", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        metavar=("DVX", "DVY", "DVZ"), help="加到反推 SMC 初始速度上的 shooting 修正。",
    )
    parser.add_argument(
        "--shooting-residual-csv", default=None,
        help=(
            "上一轮 validator 输出的 magellanic_present_6d.csv；读取 sim-observed "
            "残差并在同一反推模型中换算为初始 12D offset。"
        ),
    )
    parser.add_argument(
        "--shooting-base-ic", default=None,
        help=(
            "可选：上一轮 trial IC。读取其中已经累计的 12D shooting offset；"
            "trial03 及以后必须使用，避免丢掉前一轮修正；气体模型必须相同。"
        ),
    )
    parser.add_argument(
        "--shooting-alpha", type=float, default=0.5,
        help="残差修正的阻尼系数；首次迭代推荐 0.5，要求 0<alpha<=1。",
    )
    parser.add_argument(
        "--trial-label", default=None,
        help="写入元数据的试算标签；自动 shooting 时默认 trial02，否则 trial00。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"脚本版本：{SCRIPT_REVISION}")
    if args.lookback_gyr <= 0.0:
        raise ValueError("--lookback-gyr 必须大于 0。")
    if args.orbit_samples < 2:
        raise ValueError("--orbit-samples 必须至少为 2。")
    if not np.isfinite(args.stage2_start_code):
        raise ValueError("--stage2-start-code 必须是有限数。")
    if args.lmc_cgm_core_radius <= 0.0:
        raise ValueError("--lmc-cgm-core-radius 必须大于 0。")
    if args.minimum_lmc_cgm_edge_kpc < 0.0:
        raise ValueError("--minimum-lmc-cgm-edge-kpc 不能为负。")
    if args.coulomb_log < 0.0:
        raise ValueError("--coulomb-log 不能为负。")
    if not np.isfinite(args.shooting_alpha) or not (0.0 < args.shooting_alpha <= 1.0):
        raise ValueError("--shooting-alpha 必须满足 0 < alpha <= 1。")
    for name in (
        "satellite_dm_particle_mass", "satellite_star_particle_mass",
        "satellite_gas_particle_mass",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为有限非负数（0=自动）。")

    orbit_path = os.path.splitext(args.output)[0] + ".orbit.json"
    existing = [path for path in (args.output, orbit_path) if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在；请换 --output，或确认后加入 --overwrite："
            + ", ".join(existing)
        )

    lmc_target_position = np.asarray(args.lmc_target_position, dtype=np.float64)
    lmc_target_velocity = np.asarray(args.lmc_target_velocity, dtype=np.float64)
    smc_target_position = np.asarray(args.smc_target_position, dtype=np.float64)
    smc_target_velocity = np.asarray(args.smc_target_velocity, dtype=np.float64)

    lmc, smc = build_default_satellites(
        args.lmc_cgm_core_radius,
        args.lmc_cgm_mode,
        args.lmc_cgm_rmax,
        args.satellite_gas_model,
    )
    rng = np.random.default_rng(args.seed)

    print("读取弛豫宿主并建立实际球平均质量剖面...")
    with h5py.File(args.host_snapshot, "r") as host_file:
        for required in ("PartType0", "PartType1", "PartType4"):
            if required not in host_file:
                raise RuntimeError(f"宿主快照缺少 {required}。")
        ensure_host_ids_are_available(host_file)
        center, host_velocity = find_host_center_and_velocity(host_file)
        profile = build_host_profile(host_file, center)
        mass_table = np.asarray(host_file["Header"].attrs.get("MassTable", np.zeros(6)))
        host_gas_mass = robust_median_particle_mass(host_file["PartType0"], mass_table[0])
        host_dm_mass = robust_median_particle_mass(host_file["PartType1"], mass_table[1])
        host_star_mass = robust_median_particle_mass(host_file["PartType4"], mass_table[4])
        target_gas_mass = (
            args.satellite_gas_particle_mass if args.satellite_gas_particle_mass > 0.0
            else host_gas_mass
        )
        target_dm_mass = (
            args.satellite_dm_particle_mass if args.satellite_dm_particle_mass > 0.0
            else min(host_dm_mass, 1.0e6)
        )
        target_star_mass = (
            args.satellite_star_particle_mass if args.satellite_star_particle_mass > 0.0
            else min(host_star_mass, 1.0e5)
        )
        box_size = float(host_file["Header"].attrs["BoxSize"])

        print("从 K13 今天终点反向积分 LMC/SMC COM...")
        state_initial, closure, reference_time, reference_states = integrate_magellanic_orbit(
            profile, lmc, smc, args.lookback_gyr,
            bool(args.dynamical_friction), bool(args.host_recoil), args.coulomb_log,
            lmc_target_position, lmc_target_velocity,
            smc_target_position, smc_target_velocity,
            args.orbit_samples,
        )
        auto_lmc_dr = np.zeros(3, dtype=np.float64)
        auto_lmc_dv = np.zeros(3, dtype=np.float64)
        auto_smc_dr = np.zeros(3, dtype=np.float64)
        auto_smc_dv = np.zeros(3, dtype=np.float64)
        base_lmc_dr = np.zeros(3, dtype=np.float64)
        base_lmc_dv = np.zeros(3, dtype=np.float64)
        base_smc_dr = np.zeros(3, dtype=np.float64)
        base_smc_dv = np.zeros(3, dtype=np.float64)
        lmc_residual_position = np.zeros(3, dtype=np.float64)
        lmc_residual_velocity = np.zeros(3, dtype=np.float64)
        smc_residual_position = np.zeros(3, dtype=np.float64)
        smc_residual_velocity = np.zeros(3, dtype=np.float64)
        pseudo_lmc_position = lmc_target_position.copy()
        pseudo_lmc_velocity = lmc_target_velocity.copy()
        pseudo_smc_position = smc_target_position.copy()
        pseudo_smc_velocity = smc_target_velocity.copy()

        if args.shooting_base_ic:
            (
                base_lmc_dr, base_lmc_dv,
                base_smc_dr, base_smc_dv,
            ) = load_base_shooting_offsets(
                args.shooting_base_ic, args.satellite_gas_model
            )

        if args.shooting_residual_csv:
            (
                lmc_residual_position, lmc_residual_velocity,
                smc_residual_position, smc_residual_velocity,
            ) = load_present_residuals(args.shooting_residual_csv)
            alpha = float(args.shooting_alpha)
            pseudo_lmc_position = lmc_target_position - alpha*lmc_residual_position
            pseudo_lmc_velocity = lmc_target_velocity - alpha*lmc_residual_velocity
            pseudo_smc_position = smc_target_position - alpha*smc_residual_position
            pseudo_smc_velocity = smc_target_velocity - alpha*smc_residual_velocity
            corrected_state, _, _, _ = integrate_magellanic_orbit(
                profile, lmc, smc, args.lookback_gyr,
                bool(args.dynamical_friction), bool(args.host_recoil), args.coulomb_log,
                pseudo_lmc_position, pseudo_lmc_velocity,
                pseudo_smc_position, pseudo_smc_velocity,
                args.orbit_samples,
            )
            auto_delta = corrected_state - state_initial
            auto_lmc_dr = auto_delta[0:3]
            auto_lmc_dv = auto_delta[3:6] / KMS_TO_KPC_GYR
            auto_smc_dr = auto_delta[6:9]
            auto_smc_dv = auto_delta[9:12] / KMS_TO_KPC_GYR

        lmc_pos = state_initial[0:3]
        lmc_vel = state_initial[3:6] / KMS_TO_KPC_GYR
        smc_pos = state_initial[6:9]
        smc_vel = state_initial[9:12] / KMS_TO_KPC_GYR
        manual_lmc_dr = np.asarray(args.lmc_dr_kpc, dtype=np.float64)
        manual_lmc_dv = np.asarray(args.lmc_dv_kms, dtype=np.float64)
        manual_smc_dr = np.asarray(args.smc_dr_kpc, dtype=np.float64)
        manual_smc_dv = np.asarray(args.smc_dv_kms, dtype=np.float64)
        lmc_dr = base_lmc_dr + auto_lmc_dr + manual_lmc_dr
        lmc_dv = base_lmc_dv + auto_lmc_dv + manual_lmc_dv
        smc_dr = base_smc_dr + auto_smc_dr + manual_smc_dr
        smc_dv = base_smc_dv + auto_smc_dv + manual_smc_dv
        lmc_pos = lmc_pos + lmc_dr
        lmc_vel = lmc_vel + lmc_dv
        smc_pos = smc_pos + smc_dr
        smc_vel = smc_vel + smc_dv

        lmc_start_radius = float(np.linalg.norm(lmc_pos))
        smc_start_radius = float(np.linalg.norm(smc_pos))
        lmc_gas_inner_edge = (
            lmc_start_radius - lmc.cgm_rmax if lmc.cgm_mass_rmax > 0.0 else -1.0
        )
        smc_gas_inner_edge = (
            smc_start_radius - smc.cgm_rmax if smc.cgm_mass_rmax > 0.0 else -1.0
        )
        overlap_labels = []
        if (
            lmc.cgm_mass_rmax > 0.0
            and lmc_gas_inner_edge < args.minimum_lmc_cgm_edge_kpc
        ):
            overlap_labels.append(f"LMC({lmc_gas_inner_edge:.3f} kpc)")
        if (
            smc.cgm_mass_rmax > 0.0
            and smc_gas_inner_edge < args.minimum_lmc_cgm_edge_kpc
        ):
            overlap_labels.append(f"SMC({smc_gas_inner_edge:.3f} kpc)")
        strong_initial_overlap = int(bool(overlap_labels))
        if overlap_labels and not args.allow_strong_cgm_overlap:
            raise RuntimeError(
                "卫星气体的初始近边缘过度深入 MW："
                + ", ".join(overlap_labels)
                + f"；安全下限为 {args.minimum_lmc_cgm_edge_kpc:.3f} kpc。请延长 "
                "lookback、改用 --satellite-gas-model none，或在 C25 模式下缩小 "
                "--lmc-cgm-rmax，"
                "或仅在明确接受该非平衡重叠时加入 --allow-strong-cgm-overlap。"
            )

        print("采样 live LMC/SMC 粒子...")
        lmc_data = make_satellite_particles(
            rng, lmc, lmc_pos, lmc_vel,
            target_dm_mass, target_star_mass, target_gas_mass,
        )
        smc_data = make_satellite_particles(
            rng, smc, smc_pos, smc_vel,
            target_dm_mass, target_star_mass, target_gas_mass,
        )
        host_particles = read_host_particles(host_file, center, host_velocity)
        host_metal = host_particles[0].get("Metallicity")
        if host_metal is None:
            raise RuntimeError("宿主 PartType0 没有 Metallicity。")
        if host_metal.ndim == 1:
            host_particles[0]["Metallicity"] = host_metal[:, None]
            gas_metal_cols = 1
        else:
            gas_metal_cols = host_metal.shape[1]
        append_satellites_to_host(host_particles, lmc_data, smc_data, gas_metal_cols)
        system_shift, system_velocity_shift = barycentric_recenter(host_particles, box_size)

        orbit_metadata = {
            "SCRIPT_REVISION": SCRIPT_REVISION,
            "LOOKBACK_GYR": args.lookback_gyr,
            "INSERTION_TIME_CODE": args.stage2_start_code,
            "PRESENT_TARGET_TIME_CODE": (
                args.stage2_start_code + args.lookback_gyr / UNIT_TIME_GYR
            ),
            "DYNAMICAL_FRICTION_INCLUDED": int(args.dynamical_friction),
            "HOST_RECOIL_INCLUDED": int(args.host_recoil),
            "COULOMB_LOG": args.coulomb_log,
            "HOST_CENTER_IN_INPUT_kpc": center,
            "HOST_BULK_VELOCITY_IN_INPUT_km_s": host_velocity,
            "SYSTEM_COM_SHIFT_BEFORE_BOX_CENTER_kpc": system_shift,
            "SYSTEM_VELOCITY_SHIFT_km_s": system_velocity_shift,
            "LMC_TARGET_POSITION_kpc": lmc_target_position,
            "LMC_TARGET_VELOCITY_km_s": lmc_target_velocity,
            "SMC_TARGET_POSITION_kpc": smc_target_position,
            "SMC_TARGET_VELOCITY_km_s": smc_target_velocity,
            "TRIAL_LABEL": (
                args.trial_label
                if args.trial_label is not None
                else ("trial02" if args.shooting_residual_csv else "trial00")
            ),
            "SHOOTING_MODE": (
                "damped_endpoint_residual_rewind"
                if args.shooting_residual_csv else "manual_or_none"
            ),
            "SHOOTING_ALPHA": (
                float(args.shooting_alpha) if args.shooting_residual_csv else 0.0
            ),
            "SHOOTING_RESIDUAL_SOURCE": (
                os.path.abspath(args.shooting_residual_csv)
                if args.shooting_residual_csv else ""
            ),
            "SHOOTING_BASE_IC": (
                os.path.abspath(args.shooting_base_ic)
                if args.shooting_base_ic else ""
            ),
            "LMC_MEASURED_POSITION_RESIDUAL_kpc": lmc_residual_position,
            "LMC_MEASURED_VELOCITY_RESIDUAL_km_s": lmc_residual_velocity,
            "SMC_MEASURED_POSITION_RESIDUAL_kpc": smc_residual_position,
            "SMC_MEASURED_VELOCITY_RESIDUAL_km_s": smc_residual_velocity,
            "LMC_PSEUDO_TARGET_POSITION_kpc": pseudo_lmc_position,
            "LMC_PSEUDO_TARGET_VELOCITY_km_s": pseudo_lmc_velocity,
            "SMC_PSEUDO_TARGET_POSITION_kpc": pseudo_smc_position,
            "SMC_PSEUDO_TARGET_VELOCITY_km_s": pseudo_smc_velocity,
            "LMC_AUTO_SHOOTING_POSITION_OFFSET_kpc": auto_lmc_dr,
            "LMC_AUTO_SHOOTING_VELOCITY_OFFSET_km_s": auto_lmc_dv,
            "SMC_AUTO_SHOOTING_POSITION_OFFSET_kpc": auto_smc_dr,
            "SMC_AUTO_SHOOTING_VELOCITY_OFFSET_km_s": auto_smc_dv,
            "LMC_BASE_SHOOTING_POSITION_OFFSET_kpc": base_lmc_dr,
            "LMC_BASE_SHOOTING_VELOCITY_OFFSET_km_s": base_lmc_dv,
            "SMC_BASE_SHOOTING_POSITION_OFFSET_kpc": base_smc_dr,
            "SMC_BASE_SHOOTING_VELOCITY_OFFSET_km_s": base_smc_dv,
            "LMC_MANUAL_SHOOTING_POSITION_OFFSET_kpc": manual_lmc_dr,
            "LMC_MANUAL_SHOOTING_VELOCITY_OFFSET_km_s": manual_lmc_dv,
            "SMC_MANUAL_SHOOTING_POSITION_OFFSET_kpc": manual_smc_dr,
            "SMC_MANUAL_SHOOTING_VELOCITY_OFFSET_km_s": manual_smc_dv,
            "LMC_INITIAL_POSITION_RELATIVE_MW_kpc": lmc_pos,
            "LMC_INITIAL_VELOCITY_RELATIVE_MW_km_s": lmc_vel,
            "SMC_INITIAL_POSITION_RELATIVE_MW_kpc": smc_pos,
            "SMC_INITIAL_VELOCITY_RELATIVE_MW_km_s": smc_vel,
            "LMC_SHOOTING_POSITION_OFFSET_kpc": lmc_dr,
            "LMC_SHOOTING_VELOCITY_OFFSET_km_s": lmc_dv,
            "SMC_SHOOTING_POSITION_OFFSET_kpc": smc_dr,
            "SMC_SHOOTING_VELOCITY_OFFSET_km_s": smc_dv,
            "STATIC_ORBIT_CLOSURE_RESIDUAL": closure,
            "LMC_DISK_NORMAL_PROJECT_CHOICE": lmc_data["disk_normal"],
            "SMC_DISK_NORMAL_PROJECT_CHOICE": smc_data["disk_normal"],
            "TARGET_GAS_PARTICLE_MASS_Msun": target_gas_mass,
            "TARGET_DM_PARTICLE_MASS_Msun": target_dm_mass,
            "TARGET_STAR_PARTICLE_MASS_Msun": target_star_mass,
            "HOST_GAS_PARTICLE_MASS_MEDIAN_Msun": host_gas_mass,
            "HOST_DM_PARTICLE_MASS_MEDIAN_Msun": host_dm_mass,
            "HOST_STAR_PARTICLE_MASS_MEDIAN_Msun": host_star_mass,
            "SATELLITE_GAS_MODEL": args.satellite_gas_model,
            "LMC_GAS_PROFILE": lmc.gas_profile,
            "SMC_GAS_PROFILE": smc.gas_profile,
            "N_LMC_GAS": len(lmc_data.get("gas_pos", [])),
            "N_SMC_GAS": len(smc_data.get("gas_pos", [])),
            "N_LMC_DM": len(lmc_data["dm_pos"]),
            "N_LMC_STARS": len(lmc_data["star_pos"]),
            "N_SMC_DM": len(smc_data["dm_pos"]),
            "N_SMC_STARS": len(smc_data["star_pos"]),
            "LMC_GAS_INITIAL_MEAN_RADIAL_VELOCITY_km_s": float(
                lmc_data.get("gas_mean_radial_velocity", 0.0)
            ),
            "SMC_GAS_INITIAL_MEAN_RADIAL_VELOCITY_km_s": float(
                smc_data.get("gas_mean_radial_velocity", 0.0)
            ),
            "LMC_GAS_INITIAL_BULK_RESIDUAL_km_s": np.asarray(
                lmc_data.get("gas_bulk_velocity_residual", np.zeros(3))
            ),
            "SMC_GAS_INITIAL_BULK_RESIDUAL_km_s": np.asarray(
                smc_data.get("gas_bulk_velocity_residual", np.zeros(3))
            ),
            "LMC_CGM_MODE": (
                args.lmc_cgm_mode
                if args.satellite_gas_model == "c25-hot-corona" else "not_applicable"
            ),
            "LMC_CGM_FIDUCIAL_MASS_R200_Msun": C25_LMC_CGM_MASS_R200,
            "LMC_CGM_FIDUCIAL_R200_kpc": C25_LMC_CGM_R200_KPC,
            "LMC_CGM_RETAINED_MASS_Msun": lmc.cgm_mass_rmax,
            "LMC_CGM_RETAINED_RMAX_kpc": lmc.cgm_rmax,
            "LMC_GAS_MASS_Msun": lmc.cgm_mass_rmax,
            "SMC_GAS_MASS_Msun": smc.cgm_mass_rmax,
            "LMC_GAS_RMAX_kpc": lmc.cgm_rmax,
            "SMC_GAS_RMAX_kpc": smc.cgm_rmax,
            "MINIMUM_LMC_CGM_EDGE_kpc": args.minimum_lmc_cgm_edge_kpc,
            "MINIMUM_SATELLITE_GAS_EDGE_kpc": args.minimum_lmc_cgm_edge_kpc,
            "LMC_CGM_NEAR_EDGE_FROM_MW_kpc": lmc_gas_inner_edge,
            "LMC_GAS_NEAR_EDGE_FROM_MW_kpc": lmc_gas_inner_edge,
            "SMC_GAS_NEAR_EDGE_FROM_MW_kpc": smc_gas_inner_edge,
            "STRONG_INITIAL_CGM_OVERLAP_WARNING": strong_initial_overlap,
            "STRONG_INITIAL_GAS_OVERLAP_WARNING": strong_initial_overlap,
        }

        # 若 snapshot 丢失自定义说明组，临时以 reference IC 作为这些组的来源。
        reference_file = None
        metadata_source = host_file
        if "MWModelReference" not in host_file and args.host_reference_ic:
            reference_file = h5py.File(args.host_reference_ic, "r")
            metadata_source = reference_file
        try:
            print(f"写出 {args.output} ...")
            write_output(
                host_file, metadata_source, args.output, host_particles, box_size,
                orbit_metadata, lmc, smc, reference_time, reference_states,
            )
        finally:
            if reference_file is not None:
                reference_file.close()

    counts = validate_output(args.output, box_size)
    json_payload = {
        "model": (
            "C25+B12 collisionless structure; K13 present-day endpoint; gas="
            + str(orbit_metadata["SATELLITE_GAS_MODEL"])
            + "; "
            + str(orbit_metadata["TRIAL_LABEL"])
        ),
        "warning": (
            "The backward orbit is a first guess. Calibrate by a forward live-GIZMO "
            "run and compare satellite COM states relative to the MW center."
        ),
        "parameters": {
            key: (np.asarray(value).tolist() if not isinstance(value, (str, int, float)) else value)
            for key, value in orbit_metadata.items()
        },
        "particle_counts_by_type": counts.tolist(),
    }
    with open(orbit_path, "w", encoding="utf-8") as stream:
        json.dump(json_payload, stream, ensure_ascii=False, indent=2)

    print("="*78)
    print(f"生成完成：live MW + LMC/SMC {orbit_metadata['TRIAL_LABEL']}")
    print("="*78)
    print(f"输出 IC                    = {os.path.abspath(args.output)}")
    print(f"轨道元数据                 = {os.path.abspath(orbit_path)}")
    print(f"PartType counts            = {counts.tolist()}")
    print(
        f"Stage-2 code time start/target = {args.stage2_start_code:.6f} / "
        f"{orbit_metadata['PRESENT_TARGET_TIME_CODE']:.6f}"
    )
    print(
        "比较不同 IC 时请用 elapsed_Gyr=(Header/Time-INSERTION_TIME_CODE)"
        f"*{UNIT_TIME_GYR:.9f}，不要直接比较 Header/Time。"
    )
    print(f"卫星 m_gas/m_DM/m_star     = {target_gas_mass:.3e} / {target_dm_mass:.3e} / {target_star_mass:.3e} Msun")
    print(f"LMC 初始 r / |v|           = {np.linalg.norm(lmc_pos):.3f} kpc / {np.linalg.norm(lmc_vel):.3f} km/s")
    print(f"SMC 初始 r / |v|           = {np.linalg.norm(smc_pos):.3f} kpc / {np.linalg.norm(smc_vel):.3f} km/s")
    print(f"初始 LMC-SMC separation    = {np.linalg.norm(lmc_pos-smc_pos):.3f} kpc")
    print(f"卫星气体模型               = {args.satellite_gas_model}")
    for sat, data, edge in (
        (lmc, lmc_data, lmc_gas_inner_edge),
        (smc, smc_data, smc_gas_inner_edge),
    ):
        if sat.cgm_mass_rmax > 0.0:
            print(
                f"{sat.name} gas profile/M/Rmax      = {sat.gas_profile} / "
                f"{sat.cgm_mass_rmax:.6e} Msun / {sat.cgm_rmax:.3f} kpc"
            )
            print(
                f"{sat.name} gas near-edge/<vr>/dV   = {edge:.3f} kpc / "
                f"{float(data['gas_mean_radial_velocity']):.3e} km/s / "
                f"{np.linalg.norm(data['gas_bulk_velocity_residual']):.3e} km/s"
            )
        else:
            print(f"{sat.name} gas                    = absent")
    if strong_initial_overlap:
        print("WARNING: 卫星气体在插入时已深入 MW 内晕，请增大起始距离或缩小气体半径。")
    print(f"静态近似轨道闭合 max residual = {np.max(np.abs(closure)):.3e}（位置 kpc / 速度 km/s 混合）")
    if args.shooting_residual_csv:
        print(f"shooting alpha / source     = {args.shooting_alpha:.3f} / {os.path.abspath(args.shooting_residual_csv)}")
        print(f"LMC total dr / dv           = {lmc_dr} kpc / {lmc_dv} km/s")
        print(f"SMC total dr / dv           = {smc_dr} kpc / {smc_dv} km/s")
        print("K13 真目标与 MagellanicOrbitReference 保持不变。")
    print("下一步：保持 cooling+metals+GALSF+GALSF_EFFECTIVE_EQS；")
    print("        关闭显式 stellar feedback、subgrid winds、BH accretion 和 AGN jet，")
    print(f"        先前向演化约 {args.lookback_gyr:g} Gyr 并按 K13 做一次轨道 shooting 校准。")


if __name__ == "__main__":
    main()