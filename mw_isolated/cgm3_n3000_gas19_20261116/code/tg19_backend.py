#!/usr/bin/env python3
"""TG19 sampler/orbit functions extracted from the user-supplied script.
The analytic model/sampling formulae and K13 target values are retained.
Only extension: satellite_enclosed_mass accepts measured relaxed profiles.
The old IC writer and old EQS/feedback policy are NOT used by this pipeline.
This is an idealized TG19-inspired model, not a reproduction of 6 Gyr binary evolution.
Source: add_lmc_smc_gaswake_orbit_v3.py
Source SHA256: db8af7a36519779d727f72f004e6ef394161102cf12b0afac38582610d657bac
"""
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

SCRIPT_REVISION = "v6-tg19-oprea-no-binary-2026-09-24"

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
DEFAULT_V1_SAFE_GAS_TRUNCATION_IN_A = 5.0
V1_WARM_GAS_TEMPERATURE_K = 5.0e4
V1_WARM_GAS_SIGMA_KMS = 25.0

# 旧 v1 的“强冲压控制”质量与紧致度。a=8/3 kpc 对应的 NFW 等效浓度
# 约为 c=33/46，显著高于 B12 的 c=9/15，因此必须标成上限控制而非 fiducial。
V1_STRONG_LMC_DM_MASS_RMAX_MSUN = 1.5e11
V1_STRONG_SMC_DM_MASS_RMAX_MSUN = 1.5e10
V1_STRONG_LMC_DM_A_KPC = 8.0
V1_STRONG_SMC_DM_A_KPC = 3.0
V1_STRONG_LMC_STAR_MASS_MSUN = 2.7e9
V1_STRONG_SMC_STAR_MASS_MSUN = 3.0e8

# B12 的结构基线。
B12_LMC_TOTAL_MASS_MSUN = 1.8e11
B12_SMC_TOTAL_MASS_MSUN = 2.1e10
B12_LMC_R200_KPC = 117.1
B12_SMC_R200_KPC = 57.1
B12_LMC_DM_A_KPC = 21.4
B12_SMC_DM_A_KPC = 7.3
B12_LMC_STAR_MASS_MSUN = 2.5e9
B12_SMC_STAR_MASS_MSUN = 2.6e8

# Tepper-Garcia et al. (2019), Table 1. 论文给出的 Mvir 是
# 1.75e11/1.5e10 Msun；截断 live 模型在 rtr 内实际保留约
# 1.3e11/1.3e10 Msun，因此粒子质量预算采用后者。
TG19_LMC_DM_MASS_RTR_MSUN = 1.3e11
TG19_SMC_DM_MASS_RTR_MSUN = 1.3e10
TG19_LMC_DM_RS_KPC = 13.0
TG19_SMC_DM_RS_KPC = 3.8
TG19_LMC_DM_RTR_KPC = 75.0
TG19_SMC_DM_RTR_KPC = 28.0
TG19_LMC_STAR_MASS_MSUN = 2.5e9
TG19_SMC_STAR_MASS_MSUN = 0.8e9
TG19_LMC_STAR_RD_KPC = 2.4
TG19_SMC_STAR_RD_KPC = 1.2
TG19_LMC_STAR_ZD_KPC = 0.48
TG19_SMC_STAR_ZD_KPC = 0.25
TG19_LMC_STAR_RTR_KPC = 9.6
TG19_SMC_STAR_RTR_KPC = 4.8
TG19_LMC_GAS_MASS_MSUN = 2.0e9
TG19_SMC_GAS_MASS_MSUN = 2.0e9
TG19_LMC_GAS_RD_KPC = 9.5
TG19_SMC_GAS_RD_KPC = 4.7
TG19_LMC_GAS_RTR_KPC = 38.0
TG19_SMC_GAS_RTR_KPC = 19.0
TG19_GAS_TEMPERATURE_K = 1.0e4

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
    star_vertical_profile: str = "sech2"
    dm_profile: str = "truncated_hernquist"
    gas_profile: str = "none"
    cgm_mass_rmax: float = 0.0
    cgm_beta: float = 0.5
    cgm_rc: float = 1.0
    cgm_rmax: float = 0.0
    cgm_outer_temperature: float = 1.0e6
    cgm_metallicity_zsun: float = 0.2
    gas_velocity_dispersion: float = 0.0
    gas_mean_molecular_weight: float = MU_HOT
    gas_rd: float = 0.0
    gas_zmax: float = 0.0
    gas_vertical_model: str = "none"

    @property
    def dm_analytic_mass(self) -> float:
        """令截断半径内的 Hernquist 质量严格等于 dm_mass_rmax。"""
        if self.dm_profile != "truncated_hernquist":
            raise ValueError(
                f"{self.name}: dm_analytic_mass 仅适用于 truncated_hernquist，"
                f"当前是 {self.dm_profile!r}。"
            )
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
    satellite_structure_model: str,
    satellite_gas_model: str,
    v1_gas_truncation_in_a: float,
) -> tuple[Satellite, Satellite]:
    """构造 TG19/Oprea 默认模型，或保留的 v1/B12/C25 对照模型。"""
    is_tg19 = satellite_structure_model == "tg19-oprea-no-binary"
    if is_tg19:
        if satellite_gas_model == "tg19-extended-disks":
            lmc_gas = {
                "gas_profile": "tg19_truncated_exponential_disk",
                "cgm_mass_rmax": TG19_LMC_GAS_MASS_MSUN,
                "cgm_rmax": TG19_LMC_GAS_RTR_KPC,
                "cgm_outer_temperature": TG19_GAS_TEMPERATURE_K,
                "cgm_metallicity_zsun": 0.2,
                "gas_mean_molecular_weight": 1.23,
                "gas_rd": TG19_LMC_GAS_RD_KPC,
                "gas_zmax": 0.0,
                "gas_vertical_model": "local_isothermal_hse",
            }
            smc_gas = {
                "gas_profile": "tg19_truncated_exponential_disk",
                "cgm_mass_rmax": TG19_SMC_GAS_MASS_MSUN,
                "cgm_rmax": TG19_SMC_GAS_RTR_KPC,
                "cgm_outer_temperature": TG19_GAS_TEMPERATURE_K,
                "cgm_metallicity_zsun": 0.1,
                "gas_mean_molecular_weight": 1.23,
                "gas_rd": TG19_SMC_GAS_RD_KPC,
                "gas_zmax": 0.0,
                "gas_vertical_model": "local_isothermal_hse",
            }
        elif satellite_gas_model == "none":
            lmc_gas, smc_gas = {}, {}
        else:
            raise ValueError(
                "tg19-oprea-no-binary 只允许 --satellite-gas-model "
                "tg19-extended-disks 或 none；不能把 v1/C25 气体拼接成 TG19。"
            )

        lmc_collisionless = {
            "dm_profile": "truncated_nfw",
            "dm_mass_rmax": TG19_LMC_DM_MASS_RTR_MSUN,
            "dm_a": TG19_LMC_DM_RS_KPC,
            "dm_rmax": TG19_LMC_DM_RTR_KPC,
            "star_mass": TG19_LMC_STAR_MASS_MSUN,
            "star_rd": TG19_LMC_STAR_RD_KPC,
            "star_zd": TG19_LMC_STAR_ZD_KPC,
            "star_rmax": TG19_LMC_STAR_RTR_KPC,
            "star_zmax": 5.0 * TG19_LMC_STAR_ZD_KPC,
            "star_vertical_profile": "double_exponential",
        }
        smc_collisionless = {
            "dm_profile": "truncated_nfw",
            "dm_mass_rmax": TG19_SMC_DM_MASS_RTR_MSUN,
            "dm_a": TG19_SMC_DM_RS_KPC,
            "dm_rmax": TG19_SMC_DM_RTR_KPC,
            "star_mass": TG19_SMC_STAR_MASS_MSUN,
            "star_rd": TG19_SMC_STAR_RD_KPC,
            "star_zd": TG19_SMC_STAR_ZD_KPC,
            "star_rmax": TG19_SMC_STAR_RTR_KPC,
            "star_zmax": 5.0 * TG19_SMC_STAR_ZD_KPC,
            "star_vertical_profile": "double_exponential",
        }
        return (
            Satellite(name="LMC", **lmc_collisionless, **lmc_gas),
            Satellite(name="SMC", **smc_collisionless, **smc_gas),
        )

    if satellite_gas_model == "tg19-extended-disks":
        raise ValueError(
            "tg19-extended-disks 只能与 tg19-oprea-no-binary 结构配套。"
        )
    if satellite_gas_model == "v1-safe-warm":
        lmc_gas = {
            "gas_profile": "v1_safe_warm_hernquist",
            "cgm_mass_rmax": V1_LMC_GAS_MASS_MSUN,
            "cgm_rc": V1_LMC_GAS_HERNQUIST_A_KPC,
            "cgm_rmax": v1_gas_truncation_in_a * V1_LMC_GAS_HERNQUIST_A_KPC,
            "cgm_outer_temperature": V1_WARM_GAS_TEMPERATURE_K,
            "cgm_metallicity_zsun": 0.3,
            "gas_velocity_dispersion": V1_WARM_GAS_SIGMA_KMS,
            "gas_mean_molecular_weight": 0.61,
        }
        smc_gas = {
            "gas_profile": "v1_safe_warm_hernquist",
            "cgm_mass_rmax": V1_SMC_GAS_MASS_MSUN,
            "cgm_rc": V1_SMC_GAS_HERNQUIST_A_KPC,
            "cgm_rmax": v1_gas_truncation_in_a * V1_SMC_GAS_HERNQUIST_A_KPC,
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

    if satellite_structure_model == "v1-strong-control":
        # 旧 v1 的质量与紧致暗晕；恒星改用观测启发的盘和近似平衡速度，
        # 避免旧脚本的球状恒星、无限尾与固定 15 km/s 随机速度。
        lmc_collisionless = {
            "dm_profile": "truncated_hernquist",
            "dm_mass_rmax": V1_STRONG_LMC_DM_MASS_RMAX_MSUN,
            "dm_a": V1_STRONG_LMC_DM_A_KPC,
            "dm_rmax": B12_LMC_R200_KPC,
            "star_mass": V1_STRONG_LMC_STAR_MASS_MSUN,
            "star_rd": 1.7,
            "star_zd": 0.34,
            "star_rmax": 6.8,
            "star_zmax": 1.7,
        }
        smc_collisionless = {
            "dm_profile": "truncated_hernquist",
            "dm_mass_rmax": V1_STRONG_SMC_DM_MASS_RMAX_MSUN,
            "dm_a": V1_STRONG_SMC_DM_A_KPC,
            "dm_rmax": B12_SMC_R200_KPC,
            "star_mass": V1_STRONG_SMC_STAR_MASS_MSUN,
            "star_rd": 1.1,
            "star_zd": 0.22,
            "star_rmax": 4.4,
            "star_zmax": 1.1,
        }
    elif satellite_structure_model == "b12-literature":
        lmc_gas_mass = float(lmc_gas.get("cgm_mass_rmax", 0.0))
        smc_gas_mass = float(smc_gas.get("cgm_mass_rmax", 0.0))
        lmc_collisionless = {
            "dm_profile": "truncated_hernquist",
            "dm_mass_rmax": (
                B12_LMC_TOTAL_MASS_MSUN
                - B12_LMC_STAR_MASS_MSUN - lmc_gas_mass
            ),
            "dm_a": B12_LMC_DM_A_KPC,
            "dm_rmax": B12_LMC_R200_KPC,
            "star_mass": B12_LMC_STAR_MASS_MSUN,
            "star_rd": 1.7,
            "star_zd": 0.34,
            "star_rmax": 6.8,
            "star_zmax": 1.7,
        }
        smc_collisionless = {
            "dm_profile": "truncated_hernquist",
            "dm_mass_rmax": (
                B12_SMC_TOTAL_MASS_MSUN
                - B12_SMC_STAR_MASS_MSUN - smc_gas_mass
            ),
            "dm_a": B12_SMC_DM_A_KPC,
            "dm_rmax": B12_SMC_R200_KPC,
            "star_mass": B12_SMC_STAR_MASS_MSUN,
            "star_rd": 1.1,
            "star_zd": 0.22,
            "star_rmax": 4.4,
            "star_zmax": 1.1,
        }
    else:
        raise ValueError(
            f"未知 --satellite-structure-model={satellite_structure_model!r}。"
        )

    if min(
        lmc_collisionless["dm_mass_rmax"],
        smc_collisionless["dm_mass_rmax"],
    ) <= 0.0:
        raise RuntimeError("所选恒星/气体质量使 B12 总质量预算中的 DM 质量非正。")

    lmc = Satellite(name="LMC", **lmc_collisionless, **lmc_gas)
    smc = Satellite(name="SMC", **smc_collisionless, **smc_gas)
    return lmc, smc


def make_satellite_model_key(args) -> str:
    """用于阻止跨结构、跨气体半径复用 shooting offset。"""
    def vector_key(value) -> str:
        return ",".join(f"{float(x):.9g}" for x in np.asarray(value))

    fields = [
        "generator=v6_tg19_oprea_no_binary",
        "binary_pre_evolution_Gyr=0",
        "mutual_gravity_during_infall=1",
        "disk_orientation=insertion_orbital_angular_momentum",
        f"structure={args.satellite_structure_model}",
        f"gas={args.satellite_gas_model}",
        f"lookback={float(args.lookback_gyr):.9g}",
        f"df={int(args.dynamical_friction)}",
        f"recoil={int(args.host_recoil)}",
        f"lnL={float(args.coulomb_log):.9g}",
        f"lmc_r={vector_key(args.lmc_target_position)}",
        f"lmc_v={vector_key(args.lmc_target_velocity)}",
        f"smc_r={vector_key(args.smc_target_position)}",
        f"smc_v={vector_key(args.smc_target_velocity)}",
    ]
    if args.satellite_gas_model == "v1-safe-warm":
        fields.append(f"v1_cut_a={float(args.v1_gas_truncation_in_a):.9g}")
    elif args.satellite_gas_model == "c25-hot-corona":
        fields.extend((
            f"c25_mode={args.lmc_cgm_mode}",
            f"c25_rc={float(args.lmc_cgm_core_radius):.9g}",
            f"c25_rmax={float(args.lmc_cgm_rmax):.9g}",
        ))
    return "|".join(fields)


def equivalent_nfw_concentration(r200: float, hernquist_a: float) -> float:
    """按 B12/Springel+2005 映射反解与 Hernquist a 等效的 NFW 浓度。"""
    def mapped_a(concentration: float) -> float:
        shape = 2.0 * (
            math.log1p(concentration) - concentration/(1.0 + concentration)
        )
        return r200/concentration * math.sqrt(shape)

    low, high = 1.001, 200.0
    if not (mapped_a(high) <= hernquist_a <= mapped_a(low)):
        return float("nan")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if mapped_a(middle) > hernquist_a:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


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


def nfw_shape(x):
    """NFW 的无量纲累积质量 f(x)=ln(1+x)-x/(1+x)。"""
    value = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.log1p(value) - value / (1.0 + value)


def dm_mass_enclosed(r, sat: Satellite):
    """严格归一到 sat.dm_rmax 内 live DM 质量的截断球对称质量。"""
    value = np.maximum(np.asarray(r, dtype=np.float64), 0.0)
    limited = np.minimum(value, sat.dm_rmax)
    if sat.dm_profile == "truncated_hernquist":
        mass = hernquist_mass_enclosed(limited, sat.dm_analytic_mass, sat.dm_a)
    elif sat.dm_profile == "truncated_nfw":
        denominator = nfw_shape(sat.dm_rmax / sat.dm_a)
        mass = sat.dm_mass_rmax * nfw_shape(limited / sat.dm_a) / denominator
    else:
        raise ValueError(f"{sat.name}: 不支持 DM profile={sat.dm_profile!r}。")
    return np.minimum(mass, sat.dm_mass_rmax)


def exponential_disk_mass_enclosed(r, mass: float, rd: float):
    x = np.asarray(r, dtype=np.float64) / rd
    return mass * (1.0 - (1.0 + x) * np.exp(-x))


def disk_analytic_mass(live_mass: float, rd: float, rmax: float) -> float:
    """返回无限指数盘归一化，使 R<=rmax 的质量恰为 live_mass。"""
    retained = float(exponential_disk_mass_enclosed(rmax, 1.0, rd))
    if retained <= 0.0:
        raise ValueError("指数盘的 rd/rmax 必须为正。")
    return live_mass / retained


def truncated_exponential_disk_mass_enclosed(r, live_mass: float, rd: float,
                                              rmax: float):
    value = np.maximum(np.asarray(r, dtype=np.float64), 0.0)
    limited = np.minimum(value, rmax)
    analytic = disk_analytic_mass(live_mass, rd, rmax)
    return np.minimum(
        exponential_disk_mass_enclosed(limited, analytic, rd), live_mass
    )


def star_mass_enclosed(r, sat: Satellite):
    return truncated_exponential_disk_mass_enclosed(
        r, sat.star_mass, sat.star_rd, sat.star_rmax
    )


def gas_mass_enclosed(r, sat: Satellite, spherical_table=None):
    value = np.asarray(r, dtype=np.float64)
    if sat.cgm_mass_rmax <= 0.0:
        return np.zeros_like(value)
    if sat.gas_profile == "tg19_truncated_exponential_disk":
        return truncated_exponential_disk_mass_enclosed(
            value, sat.cgm_mass_rmax, sat.gas_rd, sat.cgm_rmax
        )
    if spherical_table is not None:
        return np.interp(
            value, spherical_table["r"], spherical_table["mass"],
            left=0.0, right=spherical_table["mass"][-1],
        )
    if sat.gas_profile == "v1_safe_warm_hernquist":
        retained = (sat.cgm_rmax / (sat.cgm_rmax + sat.cgm_rc)) ** 2
        analytic = sat.cgm_mass_rmax / retained
        return hernquist_mass_enclosed(
            np.minimum(value, sat.cgm_rmax), analytic, sat.cgm_rc
        )
    # 只在反向轨道尚未建立 beta table 时沿用 v4 的球平均近似。
    if sat.gas_profile == "c25_beta_hydrostatic":
        return sat.cgm_mass_rmax * np.minimum(
            np.maximum(value, 0.0) / sat.cgm_rmax, 1.0
        )
    raise ValueError(f"{sat.name}: 不支持 gas profile={sat.gas_profile!r}。")


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
    if hasattr(sat, "menc"):
        return sat.menc(s)  # pipeline extension: measured relaxed-particle profile
    value = dm_mass_enclosed(s, sat)
    value += star_mass_enclosed(s, sat)
    value += gas_mass_enclosed(s, sat)
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

def sample_dm_position_pairs(rng, n: int, sat: Satellite):
    if n % 2:
        raise ValueError("quiet-start DM 粒子数必须为偶数。")
    half = n // 2
    uniform = rng.random(half)
    if sat.dm_profile == "truncated_hernquist":
        retained = (sat.dm_rmax / (sat.dm_rmax + sat.dm_a)) ** 2
        y = np.sqrt(uniform * retained)
        radius = sat.dm_a * y / np.maximum(1.0 - y, 1.0e-14)
    elif sat.dm_profile == "truncated_nfw":
        rgrid = np.concatenate((
            [0.0], np.geomspace(sat.dm_a*1.0e-8, sat.dm_rmax, 50000)
        ))
        cdf = nfw_shape(rgrid/sat.dm_a)
        cdf /= cdf[-1]
        radius = np.interp(uniform, cdf, rgrid)
    else:
        raise ValueError(f"{sat.name}: 不支持 DM profile={sat.dm_profile!r}。")
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
    if sat.gas_profile == "tg19_truncated_exponential_disk":
        # 盘气体在柱坐标中单独采样；不能误当成球状 CGM。
        return None

    if sat.gas_profile == "v1_safe_warm_hernquist":
        # v1 的气体用 Hernquist 球；把无限尾截到用户选择的有限半径并重新
        # 归一化，保持声明的 live gas mass 全部位于截断半径内。
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

    enclosed = dm_mass_enclosed(r, sat) + star_mass_enclosed(r, sat) + mass
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
    if sat.dm_profile == "truncated_hernquist":
        rho_shape = 1.0 / (r * (r + sat.dm_a) ** 3)
    elif sat.dm_profile == "truncated_nfw":
        x = r / sat.dm_a
        rho_shape = 1.0 / (x * (1.0 + x)**2)
    else:
        raise ValueError(f"{sat.name}: 不支持 DM profile={sat.dm_profile!r}。")
    enclosed = (
        dm_mass_enclosed(r, sat)
        + star_mass_enclosed(r, sat)
        + gas_mass_enclosed(r, sat, cgm_table)
    )
    gravity = G_KPC_KMS2_MSUN * enclosed / r**2
    integral = -cumulative_trapezoid(
        (rho_shape * gravity)[::-1], r[::-1], initial=0.0
    )[::-1]
    sigma = np.sqrt(np.maximum(integral / rho_shape, 0.0))
    # 从 r 到截断半径积分球平均力，再接外部点质量势；对 NFW/Hernquist
    # 及两种气体结构都使用同一个自洽近似，而不套用 Hernquist 解析势。
    potential_depth = -cumulative_trapezoid(
        gravity[::-1], r[::-1], initial=0.0
    )[::-1]
    potential_depth += G_KPC_KMS2_MSUN * sat.live_mass / sat.dm_rmax
    vesc = np.sqrt(np.maximum(2.0 * potential_depth, 0.0))
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
    if sat.star_vertical_profile == "double_exponential":
        retained_z = 1.0 - np.exp(-sat.star_zmax/sat.star_zd)
        zabs = -sat.star_zd * np.log(
            1.0 - rng.random(half)*retained_z
        )
        z = zabs * rng.choice(np.array([-1.0, 1.0]), size=half)
    elif sat.star_vertical_profile == "sech2":
        zlim = np.tanh(sat.star_zmax / sat.star_zd)
        z = sat.star_zd * np.arctanh(rng.uniform(-zlim, zlim, half))
    else:
        raise ValueError(
            f"{sat.name}: 不支持 stellar vertical profile="
            f"{sat.star_vertical_profile!r}。"
        )
    pos_half = np.column_stack((radius*np.cos(phi), radius*np.sin(phi), z))
    return np.vstack((pos_half, -pos_half)), radius, phi


def build_disk_kinematic_table(sat: Satellite, cgm_table, n_grid: int = 8000):
    """Q=1.5 的近似 epicyclic 盘；足以作为待短时稳定性检查的 live IC。"""
    r = np.geomspace(1.0e-3, sat.star_rmax, n_grid)
    dm_vc2 = G_KPC_KMS2_MSUN * dm_mass_enclosed(r, sat) / r
    spherical_gas_vc2 = np.zeros_like(r)
    gas_disk_vc2 = np.zeros_like(r)
    if sat.gas_profile == "tg19_truncated_exponential_disk":
        gas_disk_vc2 = thin_exponential_disk_vc2(
            r,
            disk_analytic_mass(sat.cgm_mass_rmax, sat.gas_rd, sat.cgm_rmax),
            sat.gas_rd,
        )
    else:
        spherical_gas_vc2 = (
            G_KPC_KMS2_MSUN * gas_mass_enclosed(r, sat, cgm_table) / r
        )
    star_analytic_mass = disk_analytic_mass(
        sat.star_mass, sat.star_rd, sat.star_rmax
    )
    vc2 = (
        dm_vc2 + spherical_gas_vc2 + gas_disk_vc2
        + thin_exponential_disk_vc2(r, star_analytic_mass, sat.star_rd)
    )
    omega2 = vc2 / r**2
    kappa2 = np.maximum(r*np.gradient(omega2, r, edge_order=2) + 4.0*omega2, 1.0e-12)
    kappa = np.sqrt(kappa2)
    surface = (
        star_analytic_mass / (2.0*np.pi*sat.star_rd**2)
        * np.exp(-r/sat.star_rd)
    )
    sigma_r = np.maximum(1.5 * 3.36 * G_KPC_KMS2_MSUN * surface / kappa, 4.0)
    sigma_phi = np.minimum(sigma_r * kappa / (2.0*np.sqrt(omega2)), sigma_r)

    spherical = dm_mass_enclosed(r, sat)
    if sat.gas_profile != "tg19_truncated_exponential_disk":
        spherical += gas_mass_enclosed(r, sat, cgm_table)
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


def build_gas_disk_kinematic_table(sat: Satellite, n_grid: int = 8000):
    """TG19 1e4 K 指数气体盘的局部等温静水平衡与压力支撑近似。"""
    if sat.gas_profile != "tg19_truncated_exponential_disk":
        raise ValueError(f"{sat.name}: 当前不是 TG19 指数气体盘。")
    r = np.geomspace(max(1.0e-3, sat.gas_rd*1.0e-5), sat.cgm_rmax, n_grid)
    star_analytic = disk_analytic_mass(sat.star_mass, sat.star_rd, sat.star_rmax)
    gas_analytic = disk_analytic_mass(
        sat.cgm_mass_rmax, sat.gas_rd, sat.cgm_rmax
    )
    vc2 = G_KPC_KMS2_MSUN * dm_mass_enclosed(r, sat) / r
    vc2 += thin_exponential_disk_vc2(r, star_analytic, sat.star_rd)
    vc2 += thin_exponential_disk_vc2(r, gas_analytic, sat.gas_rd)

    sound_speed2 = (
        K_B_CGS * sat.cgm_outer_temperature
        / (sat.gas_mean_molecular_weight * M_P_CGS) / 1.0e10
    )
    sigma_star = (
        star_analytic / (2.0*np.pi*sat.star_rd**2) * np.exp(-r/sat.star_rd)
    )
    sigma_gas = (
        gas_analytic / (2.0*np.pi*sat.gas_rd**2) * np.exp(-r/sat.gas_rd)
    )
    nu2_spherical = (
        G_KPC_KMS2_MSUN * dm_mass_enclosed(r, sat) / np.maximum(r**3, 1.0e-30)
    )
    h_spherical = np.sqrt(sound_speed2 / np.maximum(nu2_spherical, 1.0e-30))
    h_sheet = sound_speed2 / np.maximum(
        np.pi * G_KPC_KMS2_MSUN * (sigma_star + sigma_gas), 1.0e-30
    )
    scale_height = 1.0 / np.sqrt(
        1.0/np.maximum(h_spherical, 1.0e-30)**2
        + 1.0/np.maximum(h_sheet, 1.0e-30)**2
    )
    scale_height = np.clip(scale_height, 0.05, 10.0)

    # 对 Sigma~exp(-R/Rd) 的一阶径向压力支撑。忽略 dH/dR 是这里最主要的
    # 近似，因此无二体预演化版本必须先做短时稳定性检查。
    mean_vphi2 = np.maximum(vc2 - sound_speed2*r/sat.gas_rd, 0.0)
    return {
        "r": r,
        "scale_height": scale_height,
        "mean_vphi": np.sqrt(mean_vphi2),
        "sound_speed2": sound_speed2,
    }


def sample_gas_disk_position_pairs(rng, n: int, sat: Satellite, table):
    if n % 2:
        raise ValueError("quiet-start 气体盘粒子数必须为偶数。")
    half = n // 2
    rgrid = np.linspace(0.0, sat.cgm_rmax, 30001)
    cdf = exponential_disk_mass_enclosed(rgrid, 1.0, sat.gas_rd)
    cdf /= cdf[-1]
    radius = np.interp(rng.random(half), cdf, rgrid)
    phi = rng.uniform(0.0, 2.0*np.pi, half)
    height = np.interp(radius, table["r"], table["scale_height"])
    z = np.clip(rng.normal(0.0, height), -4.0*height, 4.0*height)
    pos_half = np.column_stack((radius*np.cos(phi), radius*np.sin(phi), z))
    return (
        np.vstack((pos_half, -pos_half)),
        np.concatenate((radius, radius)),
        radius,
        phi,
    )


def sample_gas_disk_velocity_pairs(radius, phi, table):
    vphi = np.interp(radius, table["r"], table["mean_vphi"])
    vel_half = np.column_stack((-vphi*np.sin(phi), vphi*np.cos(phi), np.zeros_like(vphi)))
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

    dm_pos, dm_r = sample_dm_position_pairs(rng, n_dm, sat)
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

    if sat.gas_profile == "tg19_truncated_exponential_disk":
        n_gas = even_count_from_target(sat.cgm_mass_rmax, target_gas_mass)
        gas_kin = build_gas_disk_kinematic_table(sat)
        gas_pos, gas_radius, gas_half_radius, gas_phi = (
            sample_gas_disk_position_pairs(rng, n_gas, sat, gas_kin)
        )
        gas_internal_vel = sample_gas_disk_velocity_pairs(
            gas_half_radius, gas_phi, gas_kin
        )
        gas_pos = gas_pos @ rotation.T
        gas_internal_vel = gas_internal_vel @ rotation.T
        gas_direction = gas_pos / np.maximum(gas_radius[:, None], 1.0e-30)
        gas_mean_radial_velocity = float(np.mean(np.sum(
            gas_internal_vel * gas_direction, axis=1
        )))
        gas_bulk_velocity_residual = np.mean(gas_internal_vel, axis=0)
        gas_temp = np.full(n_gas, sat.cgm_outer_temperature, dtype=np.float64)
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
            "gas_scale_height_minmax_kpc": np.array([
                np.min(gas_kin["scale_height"]),
                np.max(gas_kin["scale_height"]),
            ]),
        })
    elif cgm_table is not None:
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


def load_base_shooting_offsets(path: str, expected_model_key: str):
    """只读取完全相同卫星模型上一轮 IC 的累计 12D shooting offset。"""
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
        stored_model_key = attrs.get("SATELLITE_MODEL_KEY")
        if stored_model_key is None:
            raise RuntimeError(
                f"{path} 没有 SATELLITE_MODEL_KEY；它来自旧结构，不能复用 shooting "
                "offset。请从当前模型 trial00 重新校准。"
            )
        if isinstance(stored_model_key, bytes):
            stored_model_key = stored_model_key.decode("utf-8")
        stored_model_key = str(stored_model_key)
        if stored_model_key != expected_model_key:
            raise RuntimeError(
                f"{path} 的 SATELLITE_MODEL_KEY={stored_model_key!r}，当前是 "
                f"{expected_model_key!r}。结构、气体截断或 lookback 改变后不能沿用 "
                "旧 offset；请从当前模型 trial00 重新校准。"
            )
        missing = [key for key in keys if key not in attrs]
        if missing:
            raise RuntimeError(f"{path} 缺少累计 shooting offset：{missing}")
        values = tuple(np.asarray(attrs[key], dtype=np.float64) for key in keys)
    if any(value.shape != (3,) or np.any(~np.isfinite(value)) for value in values):
        raise RuntimeError(f"{path} 的 shooting offset 必须是有限三维向量。")
    return values


