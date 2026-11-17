#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIZMO/MFM jet 气泡形态诊断：纯 yt 密度、温度与软 X-ray 图（v10.3-hsml-units）

本版专门用于观察 jet/bubble 的物理形态：

1. 完全不画速度箭头，也不构造任何速度投影；
2. 密度和 0.6--1.0 keV 软 X-ray 用 --disk-mask-half-height 筛选盘外
   gas 粒子；当前默认 0（保留全部），设为 5 才排除 |z-z_MW| < 5 kpc；
3. 粒子筛选发生在 yt 沉积之前，筛选后的临时 GIZMO 快照再由 yt
   原生 SPH/MFM 投影，绝不是在 PNG 上覆盖白带；
4. 三种物理量默认从 yt FRB 的有限正值全范围确定安全对数色标。
   密度/X-ray 仍可选择 auto 百分位或 fixed 固定色标；温度可选择 fixed。
5. 温度默认保留完整中心切片，以观察 jet 从盘面发射的结构；添加
   --mask-temperature-disk 后，温度也采用同一个 |z-z_MW| 筛选；
6. 三类 PNG 全部由 yt 原生保存；默认额外生成 1×3 连图，PIL 仅原尺寸
   粘贴已保存的图片，不重绘物理场、不缩放像素；--no-panel 可关闭连图。
7. 色条统一显示 lg(物理量/单位)，例如密度刻度 -28、-27；坐标 y/z
   仍为线性 kpc。FRB、加权温度和发射率投影仍在线性物理量上计算，
   不对 log(T) 求平均，也不对 log(发射率) 做视线积分。
   --density-zlim / --temperature-zlim / --xray-zlim 仍输入线性物理值。
8. mask=0 时直接使用原始 ds，不复制或筛选快照；mask>0 才创建盘外气体
   临时快照。视场默认 200 kpc，分辨率默认 800×800，与所附 v6 一致。
9. 密度和温度默认是 x=中心的 y-z 切片，X-ray 是沿 x 全深度投影。
   fov 仅限制图像宽高，未限制 X-ray 的视线深度；未添加 1/(4*pi)。
10. 优先使用 Metallicity_00/.02，不能把包含总金属和 He 的 11 列相加。
    每帧附带 render_settings JSON，记录中心、色标、筛选和时间。
11. 修复 yt 构造器在用户设置色标前提前渲染的风险：初始化阶段使用固定
    线性色标，正式保存前使用检查过的 LogNorm；物理插值始终由 yt 完成。
    若图像无有限正值则明确失败，绝不把像素补成假温度/发射率。
    报错包含阶段和完整 traceback，保存在 errors 目录；有失败帧退出码为 1。

12. 在 yt 建立字段之前，为原生/自动生成的 smoothing_length 注册长度单位；
    跳过 .hsml.hdf5 缓存；屏蔽银盘时保留完整快照的核长度，避免重新求邻居。

依赖：
    pip install "yt>=4.4" h5py numpy pillow

典型运行：
    python erosita_bubble_xray_yt_v10_1_native.py \
        --snapshot-dir ./output_sat --make-video --fps 8

只测试一帧：
    python erosita_bubble_xray_yt_v10_1_native.py \
        --snapshot-dir ./output_sat --start 29 --stop 29

注入离散度诊断仅在同时显式提供 --fixed-mdot-code、--wind-particle-mass-code、
--wind-f-accretion 和 --wind-spawn-number 时打印；未提供时跳过，不把历史值
当作当前运行配置。诊断不改写快照或 GIZMO 参数，不影响图像物理量。
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gizmo_compat import load_gizmo, discover_snapshots
import h5py
import numpy as np
import yt
import matplotlib
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
from yt.visualization.plot_window import AxisAlignedSlicePlot, AxisAlignedProjectionPlot
from yt.utilities.physical_constants import kboltz, mh


# =============================================================================
# 1. 用户最常修改的参数
# =============================================================================

DEFAULT_SNAPSHOT_DIR = "./output"
DEFAULT_OUTPUT_DIR = "figures_pure_yt_morphology_v10_3"
SNAPSHOT_PATTERN = "snapshot_*.hdf5"

# 沿 x 轴观察，所以图像横轴是 y、纵轴是 z。
VIEW_AXIS = "x"
FOV_KPC = 200.0
BUFFER_SIZE = 800  # 对齐 v6 未指定 buff_size 时的 yt 默认分辨率
OUTPUT_DPI = 180

# 排除中央银盘：保留 |z-z_MW| >= 指定半高的 gas；当前 0 表示不排除。
DISK_MASK_HALF_HEIGHT_KPC = 0.0

# 固定色标是自动色标失效时的回退值，也可由命令行显式启用。
DENSITY_ZLIM = (1.0e-28, 1.0e-22)       # g cm^-3
TEMPERATURE_ZLIM = (1.0e5, 3.0e8)      # K
XRAY_ZLIM = (1.0e-8, 1.0e-1)           # erg s^-1 cm^-2

DENSITY_CMAP = "viridis"
TEMPERATURE_CMAP = "magma"
XRAY_CMAP = "inferno"

# 自动色标只统计有限、严格为正的 yt FRB 像素，并在 log10 空间取百分位。
DENSITY_PERCENTILES = (1.0, 99.5)
XRAY_PERCENTILES = (5.0, 99.7)
DENSITY_MIN_DECADES = 3.0
XRAY_MIN_DECADES = 4.0

DEFAULT_TEMPERATURE_MODE = "slice"
DEFAULT_SLAB_HALF_DEPTH_KPC = 3.0

XRAY_E_MIN_KEV = 0.6
XRAY_E_MAX_KEV = 1.0
GIZMO_ZSUN_MASS_FRACTION = 0.02

# 默认以目录中编号最小的快照为 t_jet=0；也可用命令行覆盖。
JET_ON_TIME_CODE = None

# 注入参数不设历史默认值；只接受显式 CLI 输入，且仅影响日志诊断。

# 缺少卫星 ID 元数据时，宿主星回退筛选所用的 ID 上限。
HOST_STAR_ID_MAX = 1_000_000_000

# 临时快照只保留绘制密度和 X-ray 所需的字段，降低临时磁盘占用。
MASKED_GAS_FIELDS = {
    "Coordinates",
    "Masses",
    "Density",
    "SmoothingLength",
    "KernelMaxRadius",
    "ParticleIDs",
    "ParticleChildIDsNumber",
    "ParticleIDGenerationNumber",
    "InternalEnergy",
    "Temperature",
    "ElectronAbundance",
    "NeutralHydrogenAbundance",
}

yt.funcs.mylog.setLevel(30)
# 对数图中的零值/空白盘带会在 yt 遮罩前经过 unyt.log10；只屏蔽这一条
# 已知且无害的提示，其他 RuntimeWarning 仍照常显示。
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in log10",
    category=RuntimeWarning,
    module=r"unyt\.array",
)


# =============================================================================
# 2. 文件、元数据与 MW 中心
# =============================================================================

def snapshot_number(path: Path) -> int:
    """从 snapshot_029.hdf5 提取 29；无法识别时排到最后。"""
    match = re.search(r"snapshot_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def find_snapshot_files(snapshot_dir: Path) -> List[Path]:
    # yt writes snapshot_000.hsml.hdf5 caches alongside snapshots.
    # Read one entry per snapshot; .0 is the first part of a split snapshot.
    files = [p for p in snapshot_dir.glob(SNAPSHOT_PATTERN)
             if re.fullmatch(r"snapshot_\d+(?:\.0)?\.hdf5", p.name)]
    return sorted(files, key=snapshot_number)


def select_snapshot_files(
    files: Sequence[Path],
    start: Optional[int],
    stop: Optional[int],
    stride: int,
) -> List[Path]:
    selected = []
    for path in files:
        number = snapshot_number(path)
        if start is not None and number < start:
            continue
        if stop is not None and number > stop:
            continue
        selected.append(path)
    return selected[::stride]


def _metadata_attributes(handle: h5py.File, group_name: str) -> Dict[str, object]:
    if group_name not in handle:
        return {}
    return {str(key): value for key, value in handle[group_name].attrs.items()}


def collect_metadata(handle: h5py.File) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    for group_name in (
        "ReservedParticleIDs",
        "MagellanicOrbitReference",
        "MagellanicModelReference",
        "MWModelReference",
        "ModelNotes",
    ):
        metadata.update(_metadata_attributes(handle, group_name))
    return metadata


def _scalar_int(value: object) -> Optional[int]:
    array = np.asarray(value)
    if array.size != 1:
        return None
    try:
        return int(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None


def find_id_ranges(
    metadata: Dict[str, object], token: str
) -> List[Tuple[int, int, str]]:
    """寻找属性名中含 token 的 *_START / *_END 粒子 ID 范围。"""
    normalized = {str(key).upper(): value for key, value in metadata.items()}
    ranges: List[Tuple[int, int, str]] = []
    seen = set()
    for key, value in normalized.items():
        if token.upper() not in key or not re.search(r"(?:_ID)?_START$", key):
            continue
        end_key = re.sub(r"START$", "END", key)
        if end_key not in normalized:
            continue
        start = _scalar_int(value)
        end = _scalar_int(normalized[end_key])
        if start is None or end is None or end < start or (start, end) in seen:
            continue
        ranges.append((start, end, key.rsplit("_", 1)[0]))
        seen.add((start, end))
    return sorted(ranges)


def mask_in_ranges(ids: np.ndarray, ranges: Sequence[Tuple[int, int, str]]) -> np.ndarray:
    mask = np.zeros(len(ids), dtype=bool)
    for start, end, _ in ranges:
        mask |= (ids >= np.uint64(start)) & (ids <= np.uint64(end))
    return mask


def find_mw_center(
    ds,
    filename: Path,
    reference_metadata: Dict[str, object],
):
    """
    只确定 MW 中心，不再测量或使用任何速度参考系。

    优先顺序：MW_BH_ID -> 最大 BH_Mass -> 最大 PartType5 Masses ->
    第一颗 BH -> 排除 LMC/SMC 后的宿主星位置中位数 -> gas 位置中位数。
    """
    with h5py.File(filename, "r") as handle:
        metadata = dict(reference_metadata)
        metadata.update(collect_metadata(handle))

        if "PartType5" in handle and "Coordinates" in handle["PartType5"]:
            black_holes = handle["PartType5"]
            positions = np.asarray(black_holes["Coordinates"], dtype=np.float64)
            if len(positions):
                index = None
                wanted_id = _scalar_int(metadata.get("MW_BH_ID", 90_000_000))
                if wanted_id is not None and "ParticleIDs" in black_holes:
                    ids = np.asarray(black_holes["ParticleIDs"], dtype=np.uint64)
                    matches = np.flatnonzero(ids == np.uint64(wanted_id))
                    if matches.size:
                        index = int(matches[0])
                        method = f"MW BH ParticleID={wanted_id}"
                if index is None and "BH_Mass" in black_holes:
                    index = int(np.argmax(black_holes["BH_Mass"][:]))
                    method = "most-massive BH fallback"
                elif index is None and "Masses" in black_holes:
                    index = int(np.argmax(black_holes["Masses"][:]))
                    method = "most-massive PartType5 fallback"
                elif index is None:
                    index = 0
                    method = "first BH fallback"
                center = ds.arr(positions[index], "code_length").to("kpc")
                return center, method, metadata

        if "PartType4" in handle and "Coordinates" in handle["PartType4"]:
            stars = handle["PartType4"]
            positions = np.asarray(stars["Coordinates"], dtype=np.float64)
            keep = np.ones(len(positions), dtype=bool)
            if "ParticleIDs" in stars:
                ids = np.asarray(stars["ParticleIDs"], dtype=np.uint64)
                satellite_ranges = (
                    find_id_ranges(metadata, "LMC") + find_id_ranges(metadata, "SMC")
                )
                if satellite_ranges:
                    keep &= ~mask_in_ranges(ids, satellite_ranges)
                else:
                    keep &= ids < np.uint64(HOST_STAR_ID_MAX)
            if np.any(keep):
                center_code = np.median(positions[keep], axis=0)
                return ds.arr(center_code, "code_length").to("kpc"), \
                    "host-star median fallback", metadata

        if "PartType0" in handle and "Coordinates" in handle["PartType0"]:
            center_code = np.median(handle["PartType0/Coordinates"][:], axis=0)
            return ds.arr(center_code, "code_length").to("kpc"), \
                "gas median fallback", metadata

    raise RuntimeError("快照中没有可用于确定 MW 中心的 BH、恒星或 gas 坐标")


# =============================================================================
# 3. 物理字段：Kelvin 温度与 APEC 金属丰度
# =============================================================================




def load_yt_snapshot(path, cache_dir=None):
    """Use native GIZMO kernel IO before yt constructs its particle index."""
    return load_gizmo(path, cache_dir=cache_dir)

def ensure_kelvin_temperature_field(ds) -> str:
    """确保 ('gas','temperature') 具有 K 量纲，并返回诊断文字。"""
    temperature_field = ("gas", "temperature")
    available = set(ds.field_list) | set(ds.derived_field_list)

    if temperature_field in available:
        original_units = ds._get_field_info(temperature_field).units
        try:
            ds.quan(1.0, original_units).to("K")
            return f"原生 yt 温度场，单位={original_units}"
        except Exception:
            pass

    stored_temperature = ("PartType0", "Temperature")
    if stored_temperature in set(ds.field_list):

        def _stored_temperature_in_kelvin(field, data):
            raw = data[stored_temperature]
            if raw.units.is_dimensionless:
                return data.ds.arr(raw.to_value("dimensionless"), "K")
            return raw.to("K")

        ds.add_field(
            temperature_field,
            function=_stored_temperature_in_kelvin,
            sampling_type="local",
            units="K",
            display_name=r"T",
            force_override=True,
        )
        return "修复无量纲 PartType0/Temperature：按 Kelvin 解释"

    internal_energy_field = ("PartType0", "InternalEnergy")
    if internal_energy_field not in available:
        raise RuntimeError(
            "找不到有温度量纲的 temperature，也没有 PartType0/Temperature "
            "或 PartType0/InternalEnergy，无法重建 Kelvin 温度"
        )

    electron_field = ("PartType0", "ElectronAbundance")
    has_electron = electron_field in available

    def _temperature_from_internal_energy(field, data):
        specific_energy = data[internal_energy_field].to("erg/g")
        hydrogen_mass_fraction = 0.76
        if has_electron:
            electron_abundance = data[electron_field].to_value("dimensionless")
            mean_molecular_weight = 4.0 / (
                1.0
                + 3.0 * hydrogen_mass_fraction
                + 4.0 * hydrogen_mass_fraction * electron_abundance
            )
        else:
            mean_molecular_weight = 0.59
        return (
            (5.0 / 3.0 - 1.0)
            * mean_molecular_weight
            * mh
            * specific_energy
            / kboltz
        ).to("K")

    ds.add_field(
        temperature_field,
        function=_temperature_from_internal_energy,
        sampling_type="local",
        units="K",
        display_name=r"T",
        force_override=True,
    )
    electron_note = "使用 ElectronAbundance" if has_electron else "采用 mu=0.59"
    return f"由 PartType0/InternalEnergy 重建 Kelvin 温度（{electron_note}）"


def add_apec_metallicity_field(ds) -> None:
    """把 GIZMO 总金属质量分数转换成 APEC 所需的 Z/Z_sun。"""
    available = set(ds.field_list) | set(ds.derived_field_list)
    source_field = next((field for field in [
        ("PartType0", "Metallicity_00"), ("gas", "metallicity")
    ] if field in available), None)
    if source_field is None:
        raise RuntimeError(
            "yt 没有找到 ('gas','metallicity')；请检查 PartType0/Metallicity "
            "或 Metallicity_00"
        )

    def _apec_metallicity(field, data):
        z_mass_fraction = data[source_field]
        return (
            z_mass_fraction / GIZMO_ZSUN_MASS_FRACTION
        ) * data.ds.quan(1.0, "Zsun")

    ds.add_field(
        ("gas", "apec_metallicity"),
        function=_apec_metallicity,
        sampling_type="local",
        units="Zsun",
        display_name=r"Z/Z_\odot",
        force_override=True,
    )


# =============================================================================
# 4. 在粒子层排除 |z-z_MW| < z_cut 的银盘带
# =============================================================================

def _copy_attributes(source, target) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _copy_masked_particle_dataset(
    source_dataset,
    target_group,
    dataset_name: str,
    keep_mask: np.ndarray,
    chunk_size: int = 250_000,
):
    """分块复制粒子场，避免 h5py 对数百万离散索引执行低效点选择。"""
    output_shape = (int(np.count_nonzero(keep_mask)),) + source_dataset.shape[1:]
    target_dataset = target_group.create_dataset(
        dataset_name,
        shape=output_shape,
        dtype=source_dataset.dtype,
    )
    _copy_attributes(source_dataset, target_dataset)

    output_start = 0
    for input_start in range(0, len(keep_mask), chunk_size):
        input_stop = min(input_start + chunk_size, len(keep_mask))
        local_keep = keep_mask[input_start:input_stop]
        local_count = int(np.count_nonzero(local_keep))
        if local_count == 0:
            continue
        block = source_dataset[input_start:input_stop]
        target_dataset[output_start:output_start + local_count] = block[local_keep]
        output_start += local_count
    return target_dataset


def write_disk_masked_snapshot(
    source_path: Path,
    output_path: Path,
    source_ds,
    center,
    half_height_kpc: float,
) -> Dict[str, int]:
    """
    写出只含盘外 gas 的临时 GIZMO HDF5。

    这里筛选的是粒子中心；随后密度切片和 X-ray 投影都由 yt 重新执行
    SPH/MFM 核沉积。临时文件在每一帧保存完成后自动删除。
    """
    with h5py.File(source_path, "r") as source:
        if "Header" not in source or "PartType0" not in source:
            raise RuntimeError("快照缺少 Header 或 PartType0")
        if int(source["Header"].attrs.get("NumFilesPerSnapshot", 1)) != 1:
            raise RuntimeError("Particle disk masking currently requires a single-file snapshot; merge all parts first or use --disk-mask-half-height 0.")
        gas = source["PartType0"]
        if "Coordinates" not in gas:
            raise RuntimeError("PartType0 缺少 Coordinates")

        # 这里只读 z 分量；无需为筛选额外载入完整 N×3 坐标数组。
        z_code = np.asarray(gas["Coordinates"][:, 2], dtype=np.float64)
        z_kpc = source_ds.arr(
            z_code, "code_length"
        ).to_value("kpc")
        center_z_kpc = float(center[2].to_value("kpc"))
        if half_height_kpc > 0.0:
            keep = np.abs(z_kpc - center_z_kpc) >= half_height_kpc
        else:
            keep = np.ones(len(z_kpc), dtype=bool)
        kept_count = int(np.count_nonzero(keep))
        if kept_count == 0:
            raise RuntimeError(
                f"应用 |z-z_MW| >= {half_height_kpc:g} kpc 后没有剩余 gas 粒子"
            )

        with h5py.File(output_path, "x") as target:
            header_out = target.create_group("Header")
            _copy_attributes(source["Header"], header_out)

            counts = np.zeros(6, dtype=np.uint64)
            counts[0] = np.uint64(kept_count)
            for name in ("NumPart_ThisFile", "NumPart_Total"):
                old = np.asarray(source["Header"].attrs.get(name, counts))
                header_out.attrs[name] = counts.astype(old.dtype, copy=False)
            if "NumPart_Total_HighWord" in source["Header"].attrs:
                old_high = np.asarray(source["Header"].attrs["NumPart_Total_HighWord"])
                header_out.attrs["NumPart_Total_HighWord"] = np.zeros(
                    6, dtype=old_high.dtype
                )
            if "NumFilesPerSnapshot" in header_out.attrs:
                header_out.attrs["NumFilesPerSnapshot"] = 1

            # 保留单位、运行参数和自定义元数据，但不复制其他粒子类型。
            for key in source.keys():
                if key == "Header" or re.fullmatch(r"PartType\d+", key):
                    continue
                source.copy(key, target)

            gas_out = target.create_group("PartType0")
            _copy_attributes(gas, gas_out)
            copied_fields = []
            for name, dataset in gas.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                needed = name in MASKED_GAS_FIELDS or name.lower().startswith("metallicity")
                if not needed:
                    continue
                if dataset.ndim < 1 or dataset.shape[0] != len(keep):
                    continue
                _copy_masked_particle_dataset(dataset, gas_out, name, keep)
                copied_fields.append(name)

            if "Coordinates" not in copied_fields:
                raise RuntimeError("临时快照未能复制 PartType0/Coordinates")
            if not {"SmoothingLength", "KernelMaxRadius"}.intersection(copied_fields):
                # Preserve full-snapshot kernels when excluding the disk.
                # Recomputing neighbours after masking would broaden kernels
                # near the cut and change the plotted morphology.
                ad = source_ds.all_data()
                positions = ad["PartType0", "particle_position"].to_value("code_length")
                original_positions = gas["Coordinates"][:]
                if positions.shape != original_positions.shape or not np.allclose(
                    positions, original_positions, rtol=1e-12, atol=0.0
                ):
                    raise RuntimeError("Cannot verify yt/HDF5 particle order for smoothing-length transfer")
                h = ad["PartType0", "smoothing_length"].to_value("code_length")
                if h.shape != keep.shape or not np.all(np.isfinite(h) & (h > 0)):
                    raise RuntimeError("Invalid full-snapshot smoothing lengths")
                gas_out.create_dataset("SmoothingLength", data=h[keep])
                header_out.attrs["ErositaSmoothingLengthSource"] = "project_full_sample_reconstruction"
                ad.clear_data()
                del ad, positions, original_positions, h


    return {
        "total": int(len(keep)),
        "kept": kept_count,
        "removed": int(len(keep) - kept_count),
    }


# =============================================================================
# 5. 色标与三类原生 yt 图
# =============================================================================

class _SafeInitialNorm:
    """yt 4.4 构造器会先渲染；仅该次内部初始化使用有限线性色标。

    不改 yt 源码、数据或插值。正式保存前 configure_positive_log_color
    必须应用已检查的 LogNorm。此钩子针对 yt 4.4 的 NormHandler API。
    """

    def _get_default_handlers(self, field, default_display_units):
        norm_handler, colorbar_handler = super()._get_default_handlers(
            field, default_display_units
        )
        norm_handler.norm = Normalize(vmin=0.0, vmax=1.0)
        return norm_handler, colorbar_handler


class SafeSlicePlot(_SafeInitialNorm, AxisAlignedSlicePlot):
    pass


class SafeProjectionPlot(_SafeInitialNorm, AxisAlignedProjectionPlot):
    pass


def mark_render_stage(args, stage: str) -> None:
    args.render_stage = stage
    if not hasattr(args, "render_diagnostics"):
        args.render_diagnostics = {}
    print(f"  [阶段] {stage}", flush=True)

def make_time_label(ds, jet_on_time_gyr: float) -> str:
    absolute_time_gyr = ds.current_time.to_value("Gyr")
    jet_zero = ds.quan(jet_on_time_gyr, "Gyr")
    jet_age_myr = (ds.current_time - jet_zero).to_value("Myr")
    if abs(jet_age_myr) < 1.0e-8:
        jet_age_myr = 0.0
    return f"t = {absolute_time_gyr:.4f} Gyr;  t_jet = {jet_age_myr:.1f} Myr"


def configure_positive_log_color(
    plot,
    field,
    unit: str,
    cmap: str,
    scale_mode: str,
    fixed_limits: Tuple[float, float],
    percentiles: Tuple[float, float],
    minimum_decades: float,
    buff_size: int,
    diagnostics: Optional[dict] = None,
) -> Tuple[Tuple[float, float], str]:
    """正式对数渲染前检查 yt FRB；不对物理值取 log 或填充假数据。"""
    plot.set_buff_size((buff_size, buff_size))
    plot.set_unit(field, unit)
    plot.set_cmap(field, cmap)
    # 这里只读取 yt 原生切片/投影，仍在线性物理量上进行流体计算。
    pixels = np.ma.asarray(plot.frb[field].to_value(unit), dtype=np.float64)
    values = pixels.compressed()
    positive = values[np.isfinite(values) & (values > 0.0)]
    stats = {
        "field": list(field), "unit": unit, "pixels": int(pixels.size),
        "masked": int(np.count_nonzero(np.ma.getmaskarray(pixels))),
        "positive": int(positive.size), "zero": int(np.count_nonzero(values == 0)),
        "negative": int(np.count_nonzero(np.isfinite(values) & (values < 0))),
        "nonfinite": int(np.count_nonzero(~np.isfinite(values))),
        "positive_min": float(positive.min()) if positive.size else None,
        "positive_max": float(positive.max()) if positive.size else None,
        "requested_scale": scale_mode,
    }
    if diagnostics is not None:
        diagnostics[str(field)] = stats
    print(f"  [FRB诊断] {json.dumps(stats, ensure_ascii=False)}", flush=True)
    if not positive.size:
        raise ValueError(f"{field}: yt 图像没有有限正值，不能生成物理上有效的对数图；详见 FRB 诊断。")

    limits = tuple(float(value) for value in fixed_limits)
    note = "fixed"
    if scale_mode in ("yt", "auto"):
        log_values = np.log10(positive)
        if scale_mode == "yt":
            low_log, high_log = float(log_values.min()), float(log_values.max())
            note = "yt FRB finite-positive full range (safe log)"
        else:
            low_log, high_log = np.percentile(log_values, percentiles)
            if high_log - low_log < minimum_decades:
                low_log = high_log - minimum_decades
            note = f"auto p{percentiles[0]:g}-p{percentiles[1]:g}, N_positive={positive.size}"
        if high_log - low_log < 1e-8:
            low_log -= 0.5
            high_log += 0.5
            note += "; constant/nearly constant image: expanded display range"
        # 防止极端数值使 Matplotlib 色条变换上溢/下溢，不改变源像素。
        if low_log < -250 or high_log > 250 or high_log - low_log > 250:
            note += "; extreme numeric range: explicit fallback to fixed display limits"
        else:
            limits = (float(10.0**low_log), float(10.0**high_log))
    if not (np.isfinite(limits).all() and 0.0 < limits[0] < limits[1]):
        raise ValueError(f"{field}: 对数色标范围非法：{limits} {unit}")
    stats["chosen_limits"] = list(limits)
    stats["scale_note"] = note
    # 使用公开 set_norm API。之后不能再 set_log / set_zlim，否则会清除该 norm。
    plot.set_norm(field, LogNorm(vmin=limits[0], vmax=limits[1]))
    plot.set_font_size(12)
    print(f"  [色标] {field}: {limits} {unit}; {note}", flush=True)
    return limits, note


def make_density_plot(ds, center, time_label: str, args):
    field = ("gas", "density")
    mark_render_stage(args, "density: yt 切片构造")
    plot = SafeSlicePlot(
        ds,
        VIEW_AXIS,
        field,
        center=center,
        width=(args.fov, "kpc"),
        buff_size=(args.buff_size, args.buff_size),
    )
    plot.set_axes_unit("kpc")
    mark_render_stage(args, "density: FRB 与对数色标")
    limits, scale_note = configure_positive_log_color(
        plot,
        field,
        "g/cm**3",
        DENSITY_CMAP,
        args.density_color_scale,
        tuple(args.density_zlim),
        tuple(args.density_percentiles),
        args.density_min_decades,
        args.buff_size,
        args.render_diagnostics,
    )
    plot.set_background_color(field, "black")
    plot.annotate_title(
        rf"Density slice; $|z-z_{{MW}}|\geq {args.disk_mask_half_height:g}$ kpc | "
        f"{time_label}"
    )
    plot.set_colorbar_label(
        field, r"$\lg\!\left(\rho/[\mathrm{g\,cm^{-3}}]\right)$"
    )
    return plot, field, limits, scale_note


def make_temperature_plot(ds, center, time_label: str, args, disk_is_masked: bool):
    field = ("gas", "temperature")
    mark_render_stage(args, "temperature: yt 切片/投影构造")
    if args.temperature_mode == "slice":
        plot = SafeSlicePlot(
            ds,
            VIEW_AXIS,
            field,
            center=center,
            width=(args.fov, "kpc"),
            buff_size=(args.buff_size, args.buff_size),
        )
        title = "Temperature slice through MW center"
    else:
        half_fov = 0.5 * args.fov
        half_depth = args.slab_half_depth
        left_edge = center - ds.arr([half_depth, half_fov, half_fov], "kpc")
        right_edge = center + ds.arr([half_depth, half_fov, half_fov], "kpc")
        slab = ds.box(left_edge, right_edge)
        plot = SafeProjectionPlot(
            ds,
            VIEW_AXIS,
            field,
            weight_field=("gas", "density"),
            data_source=slab,
            center=center,
            width=(args.fov, "kpc"),
            buff_size=(args.buff_size, args.buff_size),
        )
        title = rf"Mass-weighted T in $|x-x_{{MW}}|<{half_depth:g}$ kpc"

    plot.set_axes_unit("kpc")
    mark_render_stage(args, "temperature: FRB 与对数色标")
    limits, scale_note = configure_positive_log_color(
        plot, field, "K", TEMPERATURE_CMAP, args.temperature_color_scale,
        tuple(args.temperature_zlim), (0.0, 100.0), 1.0, args.buff_size,
        args.render_diagnostics,
    )
    if disk_is_masked:
        title += rf"; $|z-z_{{MW}}|\geq {args.disk_mask_half_height:g}$ kpc"
    plot.annotate_title(f"{title} | {time_label}")
    plot.set_colorbar_label(field, r"$\lg(T/\mathrm{K})$")
    return plot, field, limits, scale_note


def make_xray_plot(ds, center, time_label: str, args):
    mark_render_stage(args, "xray: APEC 发射率场注册")
    added_fields = yt.add_xray_emissivity_field(
        ds,
        XRAY_E_MIN_KEV,
        XRAY_E_MAX_KEV,
        table_type="apec",
        metallicity=("gas", "apec_metallicity"),
        data_dir=args.apec_data_dir,
    )
    field = added_fields[0]
    mark_render_stage(args, "xray: yt 投影构造")
    plot = SafeProjectionPlot(
        ds,
        VIEW_AXIS,
        field,
        weight_field=None,
        center=center,
        width=(args.fov, "kpc"),
        buff_size=(args.buff_size, args.buff_size),
    )
    plot.set_axes_unit("kpc")
    mark_render_stage(args, "xray: FRB 与对数色标")
    limits, scale_note = configure_positive_log_color(
        plot,
        field,
        "erg/(s*cm**2)",
        XRAY_CMAP,
        args.xray_color_scale,
        tuple(args.xray_zlim),
        tuple(args.xray_percentiles),
        args.xray_min_decades,
        args.buff_size,
        args.render_diagnostics,
    )
    plot.set_background_color(field, "black")
    plot.annotate_title(
        rf"Intrinsic APEC {XRAY_E_MIN_KEV:.1f}-{XRAY_E_MAX_KEV:.1f} keV; "
        rf"$|z-z_{{MW}}|\geq {args.disk_mask_half_height:g}$ kpc | {time_label}"
    )
    plot.set_colorbar_label(
        field,
        r"$\lg\!\left[(\int\epsilon_X\,dl)/"
        r"(\mathrm{erg\,s^{-1}\,cm^{-2}})\right]$",
    )
    return plot, field, limits, scale_note


def format_lg_colorbars(plot) -> None:
    """保留 yt 的 LogNorm 和线性 FRB，只将色条数值格式化为 lg(Q/unit)。"""
    plot.render()
    for image_plot in plot.plots.values():
        colorbar = image_plot.cb
        if colorbar is None:
            continue
        vmin, vmax = float(colorbar.norm.vmin), float(colorbar.norm.vmax)
        if not (np.isfinite(vmin) and np.isfinite(vmax) and 0.0 < vmin < vmax):
            raise ValueError(f"lg 色标需要有限正值范围，收到 {vmin}, {vmax}")
        lo, hi = np.log10([vmin, vmax])
        ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        ticks = ticks[(ticks >= lo - 1e-10) & (ticks <= hi + 1e-10)]
        if ticks.size < 2:
            ticks = np.linspace(lo, hi, 5)
        colorbar.set_ticks(np.power(10.0, ticks))
        colorbar.formatter = FuncFormatter(
            lambda value, position: f"{np.log10(value):g}" if value > 0 else ""
        )
        colorbar.minorticks_off()
        colorbar.update_ticks()


def save_single_yt_plot(plot, output_path: Path, dpi: int) -> Path:
    """直接由 yt 保存单图；不做外部拼图或像素重采样。"""
    # 空白盘带、零发射率或无效温度会被对数色标自动遮罩；unyt 在遮罩前
    # 可能发出这一条无害的 log10 RuntimeWarning，只精确屏蔽该提示。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in log10",
            category=RuntimeWarning,
            module=r"unyt\.array",
        )
        format_lg_colorbars(plot)
        saved = plot.save(str(output_path), mpl_kwargs={"dpi": dpi})
    if not saved:
        raise RuntimeError(f"yt 没有返回输出文件：{output_path}")
    return Path(saved[0])


def save_panel(paths: Dict[str, Path], output: Path) -> Path:
    """仅把 yt 原生 PNG 原尺寸粘贴到白色画布，不重绘或缩放物理图。"""
    from PIL import Image

    images = []
    for name in ("density", "temperature", "xray"):
        with Image.open(paths[name]) as source:
            images.append(source.convert("RGB"))
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    panel = Image.new("RGB", (3*width, height), "white")
    for i, im in enumerate(images):
        panel.paste(im, (i*width+(width-im.width)//2, (height-im.height)//2))
    panel.save(output)
    return output


# =============================================================================
# 6. 注入参数诊断、单帧流程与视频
# =============================================================================

def print_injection_diagnostics(ds, args) -> Optional[dict]:
    inputs = {
        "fixed_mdot_code": args.fixed_mdot_code,
        "wind_particle_mass_code": args.wind_particle_mass_code,
        "wind_f_accretion": args.wind_f_accretion,
        "wind_spawn_number": args.wind_spawn_number,
    }
    if all(value is None for value in inputs.values()):
        print("\n[注入诊断跳过] 未提供本次运行的四个显式注入参数；不采用历史默认值。\n")
        return None
    if any(value is None for value in inputs.values()):
        raise ValueError("注入诊断需要同时显式提供四个参数。")

    mass_unit_msun = float(ds.mass_unit.to_value("Msun"))
    time_unit_yr = float(ds.time_unit.to_value("yr"))
    mdot_bh = inputs["fixed_mdot_code"] * mass_unit_msun / time_unit_yr
    particle_mass_msun = inputs["wind_particle_mass_code"] * mass_unit_msun
    fraction = inputs["wind_f_accretion"]
    mdot_wind = mdot_bh * (1.0 - fraction) / fraction
    event_mass = inputs["wind_spawn_number"] * particle_mass_msun
    interval_myr = event_mass / mdot_wind / 1.0e6 if mdot_wind > 0.0 else None
    result = dict(inputs, source="explicit_cli", mass_unit_msun=mass_unit_msun,
                  time_unit_yr=time_unit_yr, mdot_bh_msun_per_yr=mdot_bh,
                  wind_particle_mass_msun=particle_mass_msun,
                  mdot_wind_msun_per_yr=mdot_wind, event_mass_msun=event_mass,
                  event_interval_myr=interval_myr)

    print("\n========== jet 注入参数诊断（不修改快照） ==========")
    print("来源：显式 CLI 参数；不自动推断编译配置，也不是实测吸积/反馈率。")
    print(f"fixed-mdot-code       = {inputs['fixed_mdot_code']:g} code_mass/code_time")
    print(f"按当前快照单位换算：Mdot_BH = {mdot_bh:.6g} Msun/yr")
    print(f"wind-particle-mass-code = {inputs['wind_particle_mass_code']:g} code_mass")
    print(f"每个生成粒子的目标质量 = {particle_mass_msun:.6g} Msun / cell")
    print(f"wind-spawn-number     = {inputs['wind_spawn_number']:d} cell / event")
    print(f"若 f_accretion={fraction:g}，估算 Mdot_wind = {mdot_wind:.6g} Msun/yr")
    if interval_myr is None:
        print("估算生成事件间隔       = 不适用（给定参数下无风质量注入）")
    else:
        print(f"估算生成事件间隔       = {interval_myr:.6g} Myr")
    print("估算沿用 Mdot_wind=Mdot_BH*(1-f)/f；其适用性须与实际源码定义核对。")
    print("===================================================\n")
    return result


def render_one_snapshot(
    filename: Path,
    output_dir: Path,
    args,
) -> Dict[str, Path]:
    """保存三张原生 yt PNG；按需筛选盘外粒子，并额外原尺寸拼图。"""
    number = snapshot_number(filename)
    print(f"\n>> 正在处理 snapshot {number:03d}: {filename.name}")
    args.render_diagnostics = {}
    mark_render_stage(args, "加载快照 / MW 中心 / 温度场")
    ds = load_yt_snapshot(filename, cache_dir=output_dir / "yt_cache")
    center, center_method, _ = find_mw_center(
        ds, filename, args.reference_metadata
    )
    temperature_method = ensure_kelvin_temperature_field(ds)
    time_label = make_time_label(ds, args.jet_on_time_gyr_resolved)

    print(
        "  MW 中心："
        f"({center[0].to_value('kpc'):.4f}, "
        f"{center[1].to_value('kpc'):.4f}, "
        f"{center[2].to_value('kpc'):.4f}) kpc [{center_method}]"
    )
    print(f"  原快照温度字段：{temperature_method}")

    paths = {
        "density": output_dir / f"density_diskmasked_snap_{number:03d}.png",
        "temperature": output_dir / f"temperature_jet_snap_{number:03d}.png",
        "xray": output_dir / f"xray_apec_diskmasked_snap_{number:03d}.png",
    }

    # 默认温度保留完整盘面；若用户要求屏蔽盘区，则稍后使用过滤后的数据集。
    if not args.mask_temperature_disk:
        temperature_plot, _, temperature_limits, temperature_note = make_temperature_plot(
            ds, center, time_label, args, disk_is_masked=False
        )
        mark_render_stage(args, "temperature: 对数渲染 / 色条 / 保存 PNG")
        paths["temperature"] = save_single_yt_plot(
            temperature_plot, paths["temperature"], args.dpi
        )

    temp_parent = Path(args.yt_temp_dir) if args.yt_temp_dir else output_dir
    needs_mask = args.disk_mask_half_height > 0.0
    manager = (tempfile.TemporaryDirectory(
        prefix=f"yt_morphology_{number:03d}_", dir=temp_parent
    ) if needs_mask else nullcontext(None))
    with manager as temp_directory:
        mark_render_stage(args, "银盘筛选 / 绘图数据集")
        if needs_mask:
            masked_path = Path(temp_directory) / f"snapshot_{number:03d}_diskmasked.hdf5"
            mask_stats = write_disk_masked_snapshot(
                filename, masked_path, ds, center, args.disk_mask_half_height,
            )
            masked_ds = load_yt_snapshot(masked_path, cache_dir=Path(temp_directory) / "yt_cache")
            masked_center = masked_ds.arr(center.to_value("kpc"), "kpc")
        else:
            masked_ds, masked_center = ds, center
            with h5py.File(filename, "r") as handle:
                count = len(handle["PartType0/Coordinates"])
            mask_stats = {"total": count, "kept": count, "removed": 0}
            print("  不排除银盘：直接使用原始 yt dataset，不生成临时快照")
        print(
            f"  银盘筛选：保留 |z-z_MW| >= {args.disk_mask_half_height:g} kpc；"
            f"gas {mask_stats['kept']}/{mask_stats['total']}，"
            f"排除 {mask_stats['removed']}"
        )

        masked_temperature_method = ensure_kelvin_temperature_field(masked_ds)
        add_apec_metallicity_field(masked_ds)
        print(f"  绘图数据集温度字段：{masked_temperature_method}")

        density_plot, _, density_limits, density_note = make_density_plot(
            masked_ds, masked_center, time_label, args
        )
        mark_render_stage(args, "density: 对数渲染 / 色条 / 保存 PNG")
        paths["density"] = save_single_yt_plot(
            density_plot, paths["density"], args.dpi
        )
        print(
            "  密度色标："
            f"[{density_limits[0]:.4e}, {density_limits[1]:.4e}] g cm^-3 "
            f"[{density_note}]"
        )

        if args.mask_temperature_disk:
            temperature_plot, _, temperature_limits, temperature_note = make_temperature_plot(
                masked_ds,
                masked_center,
                time_label,
                args,
                disk_is_masked=needs_mask,
            )
            mark_render_stage(args, "temperature masked: 对数渲染 / 色条 / 保存 PNG")
            paths["temperature"] = save_single_yt_plot(
                temperature_plot, paths["temperature"], args.dpi
            )

        xray_plot, _, xray_limits, xray_note = make_xray_plot(
            masked_ds, masked_center, time_label, args
        )
        mark_render_stage(args, "xray: 对数渲染 / 色条 / 保存 PNG")
        paths["xray"] = save_single_yt_plot(
            xray_plot, paths["xray"], args.dpi
        )
        print(
            "  X-ray 色标："
            f"[{xray_limits[0]:.4e}, {xray_limits[1]:.4e}] erg s^-1 cm^-2 "
            f"[{xray_note}]"
        )

    if not args.no_panel:
        mark_render_stage(args, "三联图保存")
        paths["panel"] = save_panel(paths, output_dir / f"panel_snap_{number:03d}.png")
    settings = {
        "version": "v10.3-hsml-units", "snapshot": str(filename.resolve()),
        "yt_version": yt.__version__, "axis": VIEW_AXIS,
        "reader": ds._erosita_reader_info,
        "masked_reader": masked_ds._erosita_reader_info,
        "masked_kernels_preserved": bool(needs_mask),
        "center_kpc": center.to_value("kpc").tolist(), "center_method": center_method,
        "fov_kpc": args.fov, "buff_size": args.buff_size,
        "temperature_mode": args.temperature_mode,
        "temperature_field_method": temperature_method,
        "disk_mask_half_height_kpc": args.disk_mask_half_height,
        "mask_temperature_disk": args.mask_temperature_disk,
        "gas_particle_counts": mask_stats, "uses_original_dataset": not needs_mask,
        "density_color_scale": args.density_color_scale,
        "temperature_color_scale": args.temperature_color_scale,
        "xray_color_scale": args.xray_color_scale,
        "density_limits_g_cm3": list(density_limits),
        "temperature_limits_K": list(temperature_limits),
        "xray_limits_erg_s_cm2": list(xray_limits),
        "temperature_scale_note": temperature_note,
        "time_gyr": float(ds.current_time.to_value("Gyr")),
        "jet_on_time_gyr": args.jet_on_time_gyr_resolved,
        "xray_energy_keV": [XRAY_E_MIN_KEV, XRAY_E_MAX_KEV],
        "xray_definition": "integral epsilon dl, full LOS depth; no 1/(4pi), no absorption",
        "metallicity_definition": "GIZMO total metal mass fraction / 0.02; no element-column sum",
        "streamlines": False,
    }
    settings["render_diagnostics"] = args.render_diagnostics
    settings["plotter_version"] = "morphology-v10.3-hsml-units / 2026-10-30"
    settings["yt_version"] = yt.__version__
    settings["matplotlib_version"] = matplotlib.__version__
    settings_path = output_dir / f"render_settings_snap_{number:03d}.json"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  温度色标：{temperature_limits} K [{temperature_note}]")
    for label, path in paths.items():
        print(f"  已保存 [{label}]：{path}")
    return paths


def make_video(
    frames: Sequence[Path],
    output_dir: Path,
    fps: int,
    video_name: str,
) -> Optional[Path]:
    """用 ffmpeg 编码一类 PNG；单图分别编码，启用连图时另有 panel 视频。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[警告] 未找到 ffmpeg，PNG 已生成，但跳过视频。")
        return None
    if not frames:
        return None

    output_path = output_dir / video_name
    manifest_path = output_dir / f".ffmpeg_{Path(video_name).stem}_frames.txt"
    frame_duration = 1.0 / fps
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for frame in frames:
            escaped = str(frame.resolve()).replace("'", "'\\''")
            manifest.write(f"file '{escaped}'\n")
            manifest.write(f"duration {frame_duration:.12f}\n")
        escaped_last = str(frames[-1].resolve()).replace("'", "'\\''")
        manifest.write(f"file '{escaped_last}'\n")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-r",
        str(fps),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    print(f"\n>> 正在生成视频：{output_path}")
    try:
        subprocess.run(command, check=True)
    finally:
        manifest_path.unlink(missing_ok=True)
    print(f">> 视频完成：{output_path}")
    return output_path


# =============================================================================
# 7. 命令行与主程序
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "纯 yt 输出密度、温度和 APEC 软 X-ray；无速度箭头，密度/X-ray "
            "默认保留全部 gas、使用有限正值范围对数色标，并输出三联图。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=None, help="最小快照编号（含）")
    parser.add_argument("--stop", type=int, default=None, help="最大快照编号（含）")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fov", type=float, default=FOV_KPC, help="视场边长，kpc")
    parser.add_argument("--buff-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI)
    parser.add_argument("--version", action="version", version="morphology-v10.3-hsml-units / 2026-10-30")
    parser.add_argument("--no-panel", action="store_true", help="只保留三张 yt 单图，不额外拼成连图")
    parser.add_argument(
        "--reference-ic",
        default=None,
        help="可选 Stage-2 IC，用于读取 MW_BH_ID 和卫星 ID 元数据",
    )
    parser.add_argument(
        "--disk-mask-half-height",
        type=float,
        default=DISK_MASK_HALF_HEIGHT_KPC,
        help="密度和 X-ray 排除的中央 |z-z_MW| 半厚度，kpc；0 表示不排除",
    )
    parser.add_argument(
        "--mask-temperature-disk",
        action="store_true",
        help="让温度图也使用相同的银盘屏蔽；默认温度保留完整中心区",
    )
    parser.add_argument(
        "--temperature-mode",
        choices=("slice", "slab"),
        default=DEFAULT_TEMPERATURE_MODE,
    )
    parser.add_argument(
        "--slab-half-depth",
        type=float,
        default=DEFAULT_SLAB_HALF_DEPTH_KPC,
        help="温度 slab 沿视线方向半厚度，kpc",
    )
    parser.add_argument(
        "--temperature-color-scale", choices=("yt", "fixed"), default="yt",
        help="yt=有限正值全范围安全对数色标；fixed=采用 temperature-zlim",
    )
    parser.add_argument(
        "--temperature-zlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=TEMPERATURE_ZLIM,
        help="temperature-color-scale=fixed 时使用，输入线性 K 值",
    )

    parser.add_argument(
        "--density-color-scale",
        choices=("yt", "auto", "fixed"),
        default="yt",
        help="yt=有限正值全范围；auto=百分位范围；fixed=统一手动范围",
    )
    parser.add_argument(
        "--density-zlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DENSITY_ZLIM,
        help="density-color-scale=fixed 时使用，g cm^-3",
    )
    parser.add_argument(
        "--density-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DENSITY_PERCENTILES,
        help="auto 模式在 log10(正值像素) 中采用的百分位",
    )
    parser.add_argument(
        "--density-min-decades",
        type=float,
        default=DENSITY_MIN_DECADES,
        help="密度 auto 色标最小动态范围（dex）",
    )

    parser.add_argument(
        "--xray-color-scale",
        choices=("yt", "auto", "fixed"),
        default="yt",
        help="yt=有限正值全范围；auto=百分位范围；fixed=统一手动范围",
    )
    parser.add_argument(
        "--xray-zlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=XRAY_ZLIM,
        help="xray-color-scale=fixed 时使用，erg s^-1 cm^-2",
    )
    parser.add_argument(
        "--xray-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=XRAY_PERCENTILES,
        help="auto 模式在 log10(正值像素) 中采用的百分位",
    )
    parser.add_argument(
        "--xray-min-decades",
        type=float,
        default=XRAY_MIN_DECADES,
        help="X-ray auto 色标最小动态范围（dex）",
    )
    parser.add_argument("--apec-data-dir", default=None)
    parser.add_argument(
        "--yt-temp-dir",
        default=None,
        help="盘外 gas 临时 HDF5 的父目录；每帧完成后自动删除",
    )

    jet_time_group = parser.add_mutually_exclusive_group()
    jet_time_group.add_argument(
        "--jet-on-time",
        type=float,
        default=JET_ON_TIME_CODE,
        help="jet 开启绝对时刻，code_time",
    )
    jet_time_group.add_argument("--jet-on-time-gyr", type=float, default=None)
    jet_time_group.add_argument("--jet-on-snapshot", type=int, default=None)

    injection_group = parser.add_argument_group("仅用于日志的显式注入参数（四项须同时提供）")
    injection_group.add_argument(
        "--fixed-mdot-code", type=float, default=None,
        help="本次固定黑洞吸积率，code_mass/code_time；不自动读取编译配置",
    )
    injection_group.add_argument(
        "--wind-particle-mass-code", type=float, default=None,
        help="本次目标生成粒子质量，code_mass；按快照实际质量单位换算",
    )
    injection_group.add_argument(
        "--wind-f-accretion", type=float, default=None,
        help="本次吸积比例 f，用于估算风质量率 Mdot_BH*(1-f)/f",
    )
    injection_group.add_argument(
        "--wind-spawn-number", type=int, default=None,
        help="每次生成事件的粒子数量；仅用于日志诊断",
    )

    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--video-name", default="jet_morphology_yt_v10_1_native.mp4")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.stride < 1:
        parser.error("--stride 必须 >= 1")
    if args.fov <= 0.0:
        parser.error("--fov 必须 > 0")
    if args.buff_size < 64:
        parser.error("--buff-size 建议至少为 64")
    if args.disk_mask_half_height < 0.0:
        parser.error("--disk-mask-half-height 必须 >= 0")
    if args.disk_mask_half_height >= 0.5 * args.fov:
        parser.error("--disk-mask-half-height 必须小于视场半宽 fov/2")
    if args.slab_half_depth <= 0.0:
        parser.error("--slab-half-depth 必须 > 0")
    if args.density_min_decades <= 0.0 or args.xray_min_decades <= 0.0:
        parser.error("auto 色标最小动态范围必须 > 0")
    if args.fps < 1:
        parser.error("--fps 必须 >= 1")
    if args.yt_temp_dir is not None and not Path(args.yt_temp_dir).is_dir():
        parser.error("--yt-temp-dir 必须是已存在目录")

    injection_values = (
        args.fixed_mdot_code, args.wind_particle_mass_code,
        args.wind_f_accretion, args.wind_spawn_number,
    )
    if any(value is not None for value in injection_values):
        if any(value is None for value in injection_values):
            parser.error("注入诊断须同时提供 --fixed-mdot-code、--wind-particle-mass-code、"
                         "--wind-f-accretion、--wind-spawn-number")
        if not all(np.isfinite(value) for value in injection_values):
            parser.error("注入诊断参数必须为有限数值")
        if args.fixed_mdot_code < 0.0:
            parser.error("--fixed-mdot-code 必须 >= 0")
        if args.wind_particle_mass_code <= 0.0:
            parser.error("--wind-particle-mass-code 必须 > 0")
        if not 0.0 < args.wind_f_accretion <= 1.0:
            parser.error("--wind-f-accretion 必须满足 0 < f <= 1")
        if args.wind_spawn_number < 1:
            parser.error("--wind-spawn-number 必须 >= 1")

    for name, limits in (
        ("--density-zlim", args.density_zlim),
        ("--temperature-zlim", args.temperature_zlim),
        ("--xray-zlim", args.xray_zlim),
    ):
        if limits[0] <= 0.0 or limits[1] <= limits[0]:
            parser.error(f"{name} 必须满足 0 < MIN < MAX")
    for name, percentiles in (
        ("--density-percentiles", args.density_percentiles),
        ("--xray-percentiles", args.xray_percentiles),
    ):
        if not (0.0 <= percentiles[0] < percentiles[1] <= 100.0):
            parser.error(f"{name} 必须满足 0 <= LOW < HIGH <= 100")


def resolve_jet_zero_time(parser, args, all_files, selected_files, first_ds):
    if args.jet_on_snapshot is not None:
        matching = [
            path for path in all_files
            if snapshot_number(path) == args.jet_on_snapshot
        ]
        if not matching:
            parser.error(
                f"--jet-on-snapshot={args.jet_on_snapshot} 不在快照目录中"
            )
        reference_path = matching[0]
        reference_ds = first_ds if reference_path == selected_files[0] else load_yt_snapshot(reference_path)
        jet_zero = reference_ds.current_time.to("Gyr")
        source = f"snapshot_{args.jet_on_snapshot:03d} 的 Header/Time"
    elif args.jet_on_time_gyr is not None:
        jet_zero = first_ds.quan(args.jet_on_time_gyr, "Gyr")
        source = "--jet-on-time-gyr"
    elif args.jet_on_time is not None:
        jet_zero = first_ds.quan(args.jet_on_time, "code_time").to("Gyr")
        source = "--jet-on-time"
    else:
        reference_path = all_files[0]
        reference_ds = first_ds if reference_path == selected_files[0] else load_yt_snapshot(reference_path)
        jet_zero = reference_ds.current_time.to("Gyr")
        source = f"目录最早 snapshot_{snapshot_number(reference_path):03d}"
    return float(jet_zero.to_value("Gyr")), source


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    print("脚本版本：morphology-v10.3-hsml-units / 2026-10-30")
    print(f"运行文件：{Path(__file__).resolve()}")
    print(f"Python：{sys.executable}")
    print(f"依赖版本：yt={yt.__version__}, matplotlib={matplotlib.__version__}, numpy={np.__version__}")

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    if not snapshot_dir.is_dir():
        parser.error(f"快照目录不存在：{snapshot_dir}")
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = snapshot_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = find_snapshot_files(snapshot_dir)
    selected_files = select_snapshot_files(
        all_files, args.start, args.stop, args.stride
    )
    if not selected_files:
        parser.error(f"在 {snapshot_dir} 中找不到符合范围的 {SNAPSHOT_PATTERN}")

    args.reference_metadata = {}
    if args.reference_ic is not None:
        reference_ic = Path(args.reference_ic).expanduser().resolve()
        if not reference_ic.is_file():
            parser.error(f"--reference-ic 不存在：{reference_ic}")
        with h5py.File(reference_ic, "r") as handle:
            args.reference_metadata = collect_metadata(handle)
        print(f"已读取中心定位元数据：{reference_ic}")

    print(f"发现 {len(all_files)} 个快照，本次选择 {len(selected_files)} 个。")
    print(f"输出目录：{output_dir}")
    print("速度场：完全关闭；不读取、不投影、不绘制任何速度箭头")
    print(
        f"银盘排除：density / soft X-ray 使用 |z-z_MW| >= "
        f"{args.disk_mask_half_height:g} kpc"
    )
    print(
        "温度银盘排除："
        + ("开启" if args.mask_temperature_disk else "关闭（保留 jet 发射中心）")
    )
    print(
        f"色标：density={args.density_color_scale}, "
        f"temperature={args.temperature_color_scale}, X-ray={args.xray_color_scale}"
    )
    print("输出方式：三类图均由 yt 原生保存；" + ("关闭连图" if args.no_panel else "额外原尺寸拼成三联图"))
    if args.make_video and (
        args.density_color_scale != "fixed" or args.xray_color_scale != "fixed"
        or args.temperature_color_scale != "fixed"
    ):
        print(
            "[提示] 自动色标适合看每帧形态，但视频亮度可能随帧变化；"
            "定量比较时请切换 fixed 并给出统一 zlim。"
        )

    first_ds = load_yt_snapshot(selected_files[0])
    print_injection_diagnostics(first_ds, args)
    args.jet_on_time_gyr_resolved, jet_zero_source = resolve_jet_zero_time(
        parser, args, all_files, selected_files, first_ds
    )
    jet_zero_code = first_ds.quan(
        args.jet_on_time_gyr_resolved, "Gyr"
    ).to_value("code_time")
    print(
        "t_jet=0："
        f"{args.jet_on_time_gyr_resolved:.9f} Gyr = "
        f"{jet_zero_code:.9f} code_time [{jet_zero_source}]\n"
    )
    del first_ds

    rendered: Dict[str, List[Path]] = {
        "density": [],
        "temperature": [],
        "xray": [],
    }
    failed: List[Tuple[Path, str]] = []
    for filename in selected_files:
        args.render_stage = "开始快照"
        args.render_diagnostics = {}
        try:
            result = render_one_snapshot(filename, output_dir, args)
            for label, path in result.items():
                rendered.setdefault(label, []).append(path)
        except Exception as exc:
            failed.append((filename, str(exc)))
            trace = traceback.format_exc()
            error_dir = output_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            trace_path = error_dir / f"{filename.stem}.traceback.txt"
            trace_path.write_text(
                f"脚本: {Path(__file__).resolve()}\n"
                f"yt={yt.__version__}, matplotlib={matplotlib.__version__}\n"
                f"失败阶段: {args.render_stage}\n"
                f"FRB统计: {json.dumps(args.render_diagnostics, ensure_ascii=False)}\n\n"
                + trace, encoding="utf-8"
            )
            print(f"[错误] {filename.name} / {args.render_stage}: {exc}", flush=True)
            print(trace, file=sys.stderr, flush=True)
            print(f"完整报错已保存：{trace_path}", flush=True)
        finally:
            # 及时释放前一帧的 Matplotlib 图，避免批量绘图累积内存。
            import matplotlib.pyplot as plt
            plt.close("all")

    print("\n==================== 运行结果 ====================")
    print(f"成功：{len(rendered['density'])} 个快照（每帧 3 张 yt PNG，连图数={len(rendered.get('panel', []))}）")
    print(f"失败：{len(failed)} 帧")
    for filename, message in failed:
        print(f"  - {filename.name}: {message}")
    print(f"PNG 目录：{output_dir}")

    if args.make_video and rendered["density"]:
        try:
            base = Path(args.video_name)
            suffix = base.suffix or ".mp4"
            stem = base.stem if base.suffix else base.name
            for label, frames in rendered.items():
                make_video(
                    frames,
                    output_dir,
                    args.fps,
                    f"{label}_{stem}{suffix}",
                )
        except subprocess.CalledProcessError as exc:
            print(f"[警告] ffmpeg 返回错误码 {exc.returncode}；PNG 不受影响。")
    elif not args.make_video:
        print("视频未生成（默认关闭）；需要时添加 --make-video。")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
