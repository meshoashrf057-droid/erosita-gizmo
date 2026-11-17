#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIZMO/MFM jet 气泡形态诊断：纯 yt 密度、温度与软 X-ray 图（v10-lg）

本版专门用于观察 jet/bubble 的物理形态：

1. 完全不画速度箭头，也不构造任何速度投影；
2. 密度和 0.6--1.0 keV 软 X-ray 用 --disk-mask-half-height 筛选盘外
   gas 粒子；当前默认 0（保留全部），设为 5 才排除 |z-z_MW| < 5 kpc；
3. 粒子筛选发生在 yt 沉积之前，筛选后的临时 GIZMO 快照再由 yt
   原生 SPH/MFM 投影，绝不是在 PNG 上覆盖白带；
4. 密度和 X-ray 默认用正值像素的 log10 百分位自动色标，以免极端亮点
   压扁气泡轮廓；也可以切换回固定色标，便于跨快照定量比较；
5. 温度默认保留完整中心切片，以观察 jet 从盘面发射的结构；添加
   --mask-temperature-disk 后，温度也采用同一个 |z-z_MW| 筛选；
6. 三类 PNG 都由 yt 单独保存，不使用 Matplotlib/PIL/OpenCV 拼图。
7. 色条统一显示 lg(物理量/单位)，例如密度刻度 -28、-27；坐标 y/z
   仍为线性 kpc。FRB、加权温度和发射率投影仍在线性物理量上计算，
   不对 log(T) 求平均，也不对 log(发射率) 做视线积分。
   --density-zlim / --temperature-zlim / --xray-zlim 仍输入线性物理值。

依赖：
    pip install "yt>=4.4" h5py numpy

典型运行：
    python erosita_bubble_xray_yt_v10_log.py \
        --snapshot-dir ./output_sat --make-video --fps 8

只测试一帧：
    python erosita_bubble_xray_yt_v10_log.py \
        --snapshot-dir ./output_sat --start 29 --stop 29

重要：BH_DEBUG_FIX_MDOT_MBH 和 PRACTICAL_GAS_SPAWN_MSUN 只用于打印
注入离散度诊断，不会改写已有快照，也不会替代 GIZMO 源码参数。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import yt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from yt.utilities.physical_constants import kboltz, mh


# =============================================================================
# 1. 用户最常修改的参数
# =============================================================================

DEFAULT_SNAPSHOT_DIR = "./output_sat"
DEFAULT_OUTPUT_DIR = "figures_pure_yt_morphology_v10_log44"
SNAPSHOT_PATTERN = "snapshot_*.hdf5"

# 沿 x 轴观察，所以图像横轴是 y、纵轴是 z。
VIEW_AXIS = "x"
FOV_KPC = 44.0
BUFFER_SIZE = 1200
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

# 用户已确认的注入参数，仅用于日志诊断。
BH_DEBUG_FIX_MDOT_MBH = 0.0062          # code_mass / code_time
PRACTICAL_GAS_SPAWN_MSUN = 1000.0      # 每个新生 gas cell 的目标质量
BAL_F_ACCRETION = 0.5
BH_WIND_SPAWN_NUMBER = 2

# 缺少卫星 ID 元数据时，宿主星回退筛选所用的 ID 上限。
HOST_STAR_ID_MAX = 1_000_000_000

# 临时快照只保留绘制密度和 X-ray 所需的字段，降低临时磁盘占用。
MASKED_GAS_FIELDS = {
    "Coordinates",
    "Masses",
    "Density",
    "SmoothingLength",
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
    return sorted(snapshot_dir.glob(SNAPSHOT_PATTERN), key=snapshot_number)


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
            raw = data[stored_temperature].to_value("dimensionless")
            return data.ds.arr(raw, "K")

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
    if ("gas", "metallicity") not in available:
        raise RuntimeError(
            "yt 没有找到 ('gas','metallicity')；请检查 PartType0/Metallicity "
            "或 Metallicity_00"
        )

    def _apec_metallicity(field, data):
        z_mass_fraction = data[("gas", "metallicity")]
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

        with h5py.File(output_path, "w") as target:
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

    return {
        "total": int(len(keep)),
        "kept": kept_count,
        "removed": int(len(keep) - kept_count),
    }


# =============================================================================
# 5. 色标与三类原生 yt 图
# =============================================================================

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
) -> Tuple[Tuple[float, float], str]:
    """设置 yt 对数颜色归一化，并返回实际 zlim 与诊断说明。"""
    plot.set_buff_size((buff_size, buff_size))
    plot.set_unit(field, unit)
    plot.set_cmap(field, cmap)
    plot.set_log(field, True)

    limits = tuple(float(value) for value in fixed_limits)
    note = "fixed"
    if scale_mode == "auto":
        try:
            values = np.asarray(plot.frb[field].to_value(unit), dtype=np.float64)
            positive = values[np.isfinite(values) & (values > 0.0)]
            if positive.size < 16:
                raise RuntimeError(f"严格正值像素只有 {positive.size} 个")
            log_values = np.log10(positive)
            low_log, high_log = np.percentile(log_values, percentiles)
            if not np.isfinite(low_log) or not np.isfinite(high_log):
                raise RuntimeError("百分位得到非有限数值")
            if high_log - low_log < minimum_decades:
                low_log = high_log - minimum_decades
            limits = (10.0**low_log, 10.0**high_log)
            note = (
                f"auto p{percentiles[0]:g}-p{percentiles[1]:g}, "
                f"N_positive={positive.size}"
            )
        except Exception as exc:
            note = f"auto 失败，回退 fixed：{exc}"

    plot.set_zlim(field, limits[0], limits[1])
    plot.set_font_size(12)
    return limits, note


def make_density_plot(ds, center, time_label: str, args):
    field = ("gas", "density")
    plot = yt.SlicePlot(
        ds,
        VIEW_AXIS,
        field,
        center=center,
        width=(args.fov, "kpc"),
        buff_size=(args.buff_size, args.buff_size),
    )
    plot.set_axes_unit("kpc")
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
    if args.temperature_mode == "slice":
        plot = yt.SlicePlot(
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
        plot = yt.ProjectionPlot(
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
    plot.set_unit(field, "K")
    plot.set_cmap(field, TEMPERATURE_CMAP)
    plot.set_log(field, True)
    plot.set_zlim(field, args.temperature_zlim[0], args.temperature_zlim[1])
    plot.set_buff_size((args.buff_size, args.buff_size))
    plot.set_font_size(12)
    if disk_is_masked:
        title += rf"; $|z-z_{{MW}}|\geq {args.disk_mask_half_height:g}$ kpc"
    plot.annotate_title(f"{title} | {time_label}")
    plot.set_colorbar_label(field, r"$\lg(T/\mathrm{K})$")
    return plot, field


def make_xray_plot(ds, center, time_label: str, args):
    added_fields = yt.add_xray_emissivity_field(
        ds,
        XRAY_E_MIN_KEV,
        XRAY_E_MAX_KEV,
        table_type="apec",
        metallicity=("gas", "apec_metallicity"),
        data_dir=args.apec_data_dir,
    )
    field = added_fields[0]
    plot = yt.ProjectionPlot(
        ds,
        VIEW_AXIS,
        field,
        weight_field=None,
        center=center,
        width=(args.fov, "kpc"),
        buff_size=(args.buff_size, args.buff_size),
    )
    plot.set_axes_unit("kpc")
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


# =============================================================================
# 6. 注入参数诊断、单帧流程与视频
# =============================================================================

def print_injection_diagnostics(ds) -> None:
    mass_unit_msun = ds.mass_unit.to_value("Msun")
    time_unit_yr = ds.time_unit.to_value("yr")
    mdot_bh = BH_DEBUG_FIX_MDOT_MBH * mass_unit_msun / time_unit_yr
    if 0.0 < BAL_F_ACCRETION <= 1.0:
        mdot_wind = mdot_bh * (1.0 - BAL_F_ACCRETION) / BAL_F_ACCRETION
    else:
        mdot_wind = np.nan
    event_mass = BH_WIND_SPAWN_NUMBER * PRACTICAL_GAS_SPAWN_MSUN
    interval_myr = event_mass / mdot_wind / 1.0e6 if mdot_wind > 0.0 else np.nan

    print("\n========== jet 注入参数诊断（不修改快照） ==========")
    print(f"BH_DEBUG_FIX_MDOT_MBH = {BH_DEBUG_FIX_MDOT_MBH:g} code_mass/code_time")
    print(f"按当前单位换算：Mdot_BH = {mdot_bh:.6g} Msun/yr")
    print(f"PracticalGasSpawn      = {PRACTICAL_GAS_SPAWN_MSUN:.6g} Msun / cell")
    print(f"BH_WIND_SPAWN_NUMBER   = {BH_WIND_SPAWN_NUMBER:d} cell / event")
    print(f"若 BAL_f_accretion={BAL_F_ACCRETION:g}，估算 Mdot_wind = {mdot_wind:.6g} Msun/yr")
    print(f"估算生成事件间隔       = {interval_myr:.6g} Myr")
    print("注意：最后两项取决于源码中 BAL_f_accretion 的确切定义。")
    print("===================================================\n")


def render_one_snapshot(
    filename: Path,
    output_dir: Path,
    args,
) -> Dict[str, Path]:
    """分别保存密度、温度和软 X-ray 三张原生 yt PNG。"""
    number = snapshot_number(filename)
    print(f"\n>> 正在处理 snapshot {number:03d}: {filename.name}")

    ds = yt.load(str(filename))
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
        temperature_plot, _ = make_temperature_plot(
            ds, center, time_label, args, disk_is_masked=False
        )
        paths["temperature"] = save_single_yt_plot(
            temperature_plot, paths["temperature"], args.dpi
        )

    temp_parent = Path(args.yt_temp_dir) if args.yt_temp_dir else None
    with tempfile.TemporaryDirectory(
        prefix=f"yt_morphology_{number:03d}_", dir=temp_parent
    ) as temp_directory:
        masked_path = Path(temp_directory) / f"snapshot_{number:03d}_diskmasked.hdf5"
        mask_stats = write_disk_masked_snapshot(
            filename,
            masked_path,
            ds,
            center,
            args.disk_mask_half_height,
        )
        print(
            f"  银盘筛选：保留 |z-z_MW| >= {args.disk_mask_half_height:g} kpc；"
            f"gas {mask_stats['kept']}/{mask_stats['total']}，"
            f"排除 {mask_stats['removed']}"
        )

        masked_ds = yt.load(str(masked_path))
        masked_center = masked_ds.arr(center.to_value("kpc"), "kpc")
        masked_temperature_method = ensure_kelvin_temperature_field(masked_ds)
        add_apec_metallicity_field(masked_ds)
        print(f"  盘外临时快照温度字段：{masked_temperature_method}")

        density_plot, _, density_limits, density_note = make_density_plot(
            masked_ds, masked_center, time_label, args
        )
        paths["density"] = save_single_yt_plot(
            density_plot, paths["density"], args.dpi
        )
        print(
            "  密度色标："
            f"[{density_limits[0]:.4e}, {density_limits[1]:.4e}] g cm^-3 "
            f"[{density_note}]"
        )

        if args.mask_temperature_disk:
            temperature_plot, _ = make_temperature_plot(
                masked_ds,
                masked_center,
                time_label,
                args,
                disk_is_masked=True,
            )
            paths["temperature"] = save_single_yt_plot(
                temperature_plot, paths["temperature"], args.dpi
            )

        xray_plot, _, xray_limits, xray_note = make_xray_plot(
            masked_ds, masked_center, time_label, args
        )
        paths["xray"] = save_single_yt_plot(
            xray_plot, paths["xray"], args.dpi
        )
        print(
            "  X-ray 色标："
            f"[{xray_limits[0]:.4e}, {xray_limits[1]:.4e}] erg s^-1 cm^-2 "
            f"[{xray_note}]"
        )

    for label, path in paths.items():
        print(f"  已保存 [{label}]：{path}")
    return paths


def make_video(
    frames: Sequence[Path],
    output_dir: Path,
    fps: int,
    video_name: str,
) -> Optional[Path]:
    """用 ffmpeg 编码一类 PNG；三种物理量始终生成三个独立视频。"""
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
            "默认排除 |z-z_MW|<5 kpc 银盘带。"
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
        "--temperature-zlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=TEMPERATURE_ZLIM,
        help="温度固定对数色标，K",
    )

    parser.add_argument(
        "--density-color-scale",
        choices=("auto", "fixed"),
        default="auto",
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
        choices=("auto", "fixed"),
        default="auto",
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

    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--video-name", default="jet_morphology_yt_v10_log.mp4")
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
        reference_ds = first_ds if reference_path == selected_files[0] else yt.load(
            str(reference_path)
        )
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
        reference_ds = first_ds if reference_path == selected_files[0] else yt.load(
            str(reference_path)
        )
        jet_zero = reference_ds.current_time.to("Gyr")
        source = f"目录最早 snapshot_{snapshot_number(reference_path):03d}"
    return float(jet_zero.to_value("Gyr")), source


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    print("脚本版本：morphology-v10-log / 2026-10-05")

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
        f"temperature=fixed log, X-ray={args.xray_color_scale}"
    )
    print("输出方式：三类图分别由 yt 原生保存，不生成拼接三联图")
    if args.make_video and (
        args.density_color_scale == "auto" or args.xray_color_scale == "auto"
    ):
        print(
            "[提示] auto 色标适合看每帧形态，但视频亮度可能随帧变化；"
            "定量比较时请切换 fixed 并给出统一 zlim。"
        )

    first_ds = yt.load(str(selected_files[0]))
    print_injection_diagnostics(first_ds)
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
        try:
            result = render_one_snapshot(filename, output_dir, args)
            for label, path in result.items():
                rendered[label].append(path)
        except Exception as exc:
            failed.append((filename, str(exc)))
            print(f"[错误] {filename.name} 绘制失败：{exc}")

    print("\n==================== 运行结果 ====================")
    print(f"成功：{len(rendered['density'])} 个快照（每个快照 3 张 yt PNG）")
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


if __name__ == "__main__":
    main()
