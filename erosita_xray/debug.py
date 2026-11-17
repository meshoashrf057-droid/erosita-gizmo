#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GIZMO -> yt/APEC -> BH-centered ±22 kpc X-ray cube
-> Sun=(-8.2,0,0) -> HEALPix LOS -> 全天球

Metallicity = Metallicity_00 + ... + Metallicity_10
I_X = 1/(4*pi) * integral epsilon_X ds

输出：
1. standard all-sky
2. bubble-enhanced all-sky：仅改变色标统计区域，不改变数据
3. GC l-b zoom
"""

import os, glob, gc, h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import healpy as hp
import yt
from scipy.ndimage import map_coordinates

# =============================================================================
# 参数
# =============================================================================
SNAPSHOT_DIR = "./allsky"
OUTPUT_DIR = os.path.join(SNAPSHOT_DIR, "figures_allsky_xray")
GRID_DIR = os.path.join(SNAPSHOT_DIR, "allsky_xray_grids")
SNAPSHOT_GLOB = "snapshot_043.hdf5"
# SNAPSHOT_GLOB = "snapshot_043.hdf5"

E_MIN_KEV, E_MAX_KEV = 0.6, 1.0
XRAY_FIELD = ("gas", "xray_emissivity_0.6_1.0_keV")

GRID_RADIUS_KPC = 22.0
GRID_N = 256
GRID_DIMS = (GRID_N, GRID_N, GRID_N)
OVERWRITE_GRID = False

SUN_OFFSET_KPC = np.array([-8.2, 0.0, 0.0], dtype=np.float64)
LOS_MIN_KPC = 0.05
LOS_MAX_KPC = 35.0
N_LOS = GRID_N
NSIDE = 128
PIXEL_CHUNK = 4096
OVERWRITE_ALLSKY = False

CMAP = "inferno"

DPI = 220
STANDARD_PMIN, STANDARD_PMAX = 1.0, 99.5

# bubble-enhanced色标只由GC高纬区域确定
BUBBLE_COLOR_LON_HALF = 80.0
BUBBLE_COLOR_LAT_MIN = 10.0
BUBBLE_COLOR_LAT_MAX = 80.0
BUBBLE_PMIN, BUBBLE_PMAX = 2.0, 98.5

# GC zoom
GC_LON_HALF = 75.0
GC_LAT_HALF = 80.0
LOCAL_NX, LOCAL_NY = 1000, 900

MAKE_STANDARD = True
MAKE_BUBBLE_ENHANCED = True
MAKE_GC_ZOOM = True

KPC_TO_CM = 3.0856775814913673e21
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)
yt.funcs.mylog.setLevel(30)

# =============================================================================
# 金属丰度：与你的切片脚本完全相同
# =============================================================================
def _total_metallicity(field, data):
    total = data["PartType0", "Metallicity_00"].copy()
    for i in range(1, 11):
        total += data["PartType0", f"Metallicity_{i:02d}"]
    return total

yt.add_field(
    ("gas", "total_metallicity"),
    function=_total_metallicity,
    units="",
    sampling_type="cell"
)

# =============================================================================
# BH、文件名
# =============================================================================
def get_bh_center_and_time(filename):
    with h5py.File(filename, "r") as f:
        time = float(f["Header"].attrs.get("Time", np.nan))
        if "PartType5" not in f or "Coordinates" not in f["PartType5"]:
            raise RuntimeError(f"{filename} 缺少PartType5/Coordinates。")
        bh = np.asarray(f["PartType5"]["Coordinates"][0], dtype=np.float64)
    return bh, time

def stem(fn):
    return os.path.splitext(os.path.basename(fn))[0]

def grid_filename(fn):
    return os.path.join(
        GRID_DIR,
        f"{stem(fn)}_xray06_10_metalsum11_R{GRID_RADIUS_KPC:.0f}_grid{GRID_N}.hdf5"
    )

def npy_filename(fn):
    return os.path.join(
        OUTPUT_DIR,
        f"{stem(fn)}_xray06_10_metalsum11_R{GRID_RADIUS_KPC:.0f}_"
        f"nside{NSIDE}_los{LOS_MAX_KPC:.0f}.npy"
    )

def output_filename(fn, suffix):
    return os.path.join(
        OUTPUT_DIR,
        f"{stem(fn)}_xray06_10_metalsum11_R{GRID_RADIUS_KPC:.0f}_{suffix}.png"
    )

# =============================================================================
# yt：GIZMO -> APEC -> X-ray cube
# =============================================================================
def build_xray_grid(snapshot_file, grid_file):
    bh, time = get_bh_center_and_time(snapshot_file)
    print(f"  BH = {bh} kpc")
    print(f"  Time = {time:.6f} Gyr")

    ds = yt.load(snapshot_file)
    ds.force_periodicity(False)

    yt.add_xray_emissivity_field(
        ds, E_MIN_KEV, E_MAX_KEV,
        table_type="apec",
        metallicity=("gas", "total_metallicity")
    )

    center = ds.arr(bh, "kpc")
    radius = ds.quan(GRID_RADIUS_KPC, "kpc")
    dx = 2.0 * GRID_RADIUS_KPC / GRID_N

    print(
        f"  yt X-ray grid = {GRID_DIMS}, "
        f"±{GRID_RADIUS_KPC:.1f} kpc, dx={dx:.4f} kpc"
    )

    grid = ds.arbitrary_grid(
        center - radius, center + radius, GRID_DIMS
    )

    cube = np.asarray(
        grid[XRAY_FIELD].in_units("erg/s/cm**3").d,
        dtype=np.float32
    )
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    cube[cube < 0.0] = 0.0

    with h5py.File(grid_file, "w") as f:
        f.create_dataset(
            "xray_emissivity", data=cube,
            compression="gzip", compression_opts=1
        )
        f.attrs["units"] = "erg/s/cm**3"
        f.attrs["time"] = time
        f.attrs["grid_radius_kpc"] = GRID_RADIUS_KPC
        f.attrs["grid_n"] = GRID_N
        f.attrs["metallicity"] = "sum_Metallicity_00_to_10"
        f.attrs["bh_x_kpc"], f.attrs["bh_y_kpc"], f.attrs["bh_z_kpc"] = bh

    print(f"  保存cube：{grid_file}")
    del cube, grid, ds
    gc.collect()

def load_xray_grid(grid_file):
    with h5py.File(grid_file, "r") as f:
        cube = np.asarray(f["xray_emissivity"][...], dtype=np.float32)
        time = float(f.attrs["time"])
        radius = float(f.attrs["grid_radius_kpc"])
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    cube[cube < 0.0] = 0.0
    return cube, time, radius

# =============================================================================
# 三线性采样
# =============================================================================
def sample_cube(cube, points, radius):
    nx, ny, nz = cube.shape
    dx, dy, dz = 2*radius/nx, 2*radius/ny, 2*radius/nz

    ix = (points[:, 0] + radius) / dx - 0.5
    iy = (points[:, 1] + radius) / dy - 0.5
    iz = (points[:, 2] + radius) / dz - 0.5

    values = map_coordinates(
        cube, np.vstack((ix, iy, iz)),
        order=1, mode="constant", cval=0.0, prefilter=False
    )
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values[values < 0.0] = 0.0
    return values

# =============================================================================
# Sun -> HEALPix LOS
# =============================================================================
def make_allsky(cube, radius):
    npix = hp.nside2npix(NSIDE)
    pix = np.arange(npix, dtype=np.int64)
    vx, vy, vz = hp.pix2vec(NSIDE, pix, nest=False)
    directions = np.column_stack((vx, vy, vz)).astype(np.float64)

    los_kpc = np.linspace(
        LOS_MIN_KPC, LOS_MAX_KPC, N_LOS, dtype=np.float64
    )
    los_cm = los_kpc * KPC_TO_CM
    intensity = np.zeros(npix, dtype=np.float64)

    print(
        f"  HEALPix: NSIDE={NSIDE}, Npix={npix}, "
        f"LOS={LOS_MIN_KPC:.2f}-{LOS_MAX_KPC:.2f} kpc, "
        f"N_LOS={N_LOS}"
    )

    for i0 in range(0, npix, PIXEL_CHUNK):
        i1 = min(i0 + PIXEL_CHUNK, npix)
        d = directions[i0:i1]

        points = (
            SUN_OFFSET_KPC[None, None, :]
            + d[:, None, :] * los_kpc[None, :, None]
        )

        eps = sample_cube(
            cube, points.reshape(-1, 3), radius
        ).reshape(i1 - i0, N_LOS)

        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(eps, x=los_cm, axis=1)
        else:
            integral = np.trapz(eps, x=los_cm, axis=1)

        intensity[i0:i1] = integral / (4.0 * np.pi)
        print(f"    pixels {i0:6d}-{i1-1:6d} / {npix-1}")
        del points, eps, integral

    intensity = np.nan_to_num(
        intensity, nan=0.0, posinf=0.0, neginf=0.0
    )
    intensity[intensity < 0.0] = 0.0
    return intensity

# =============================================================================
# HEALPix坐标、色标
# =============================================================================
def lon_lat():
    npix = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix), nest=False)
    lon360 = np.degrees(phi)
    lon_signed = (lon360 + 180.0) % 360.0 - 180.0
    lat = 90.0 - np.degrees(theta)
    return lon360, lon_signed, lat

def get_standard_limits(hpxmap):
    good = hpxmap[np.isfinite(hpxmap) & (hpxmap > 0.0)]
    return (
        float(np.percentile(good, STANDARD_PMIN)),
        float(np.percentile(good, STANDARD_PMAX))
    )

def get_bubble_limits(hpxmap):
    _, lon, lat = lon_lat()
    region = (
        (np.abs(lon) <= BUBBLE_COLOR_LON_HALF)
        & (np.abs(lat) >= BUBBLE_COLOR_LAT_MIN)
        & (np.abs(lat) <= BUBBLE_COLOR_LAT_MAX)
        & np.isfinite(hpxmap)
        & (hpxmap > 0.0)
    )
    good = hpxmap[region]
    if good.size == 0:
        raise RuntimeError("bubble色标区域中没有正X-ray强度。")
    return (
        float(np.percentile(good, BUBBLE_PMIN)),
        float(np.percentile(good, BUBBLE_PMAX))
    )

def make_log_map(hpxmap, vmin, vmax):
    x = np.asarray(hpxmap, dtype=np.float64).copy()
    x[~np.isfinite(x) | (x <= 0.0)] = vmin
    x = np.clip(x, vmin, vmax)
    return np.log10(x)

def scientific_label(value):
    e = int(np.floor(np.log10(value)))
    m = value / 10.0**e
    return rf"${m:.1f}\times10^{{{e}}}$"

def add_colorbar(fig, vmin, vmax, rect):
    logmin, logmax = np.log10(vmin), np.log10(vmax)
    sm = ScalarMappable(
        norm=Normalize(vmin=logmin, vmax=logmax), cmap=CMAP
    )
    sm.set_array([])
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(np.log10(ticks))
    cbar.set_ticklabels([scientific_label(v) for v in ticks])
    cbar.set_label(
        r"$I_{0.6-1.0\,\mathrm{keV}}$ "
        r"$[\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}}]$",
        fontsize=10
    )
    cbar.ax.tick_params(labelsize=8)

def sky_text(lon, lat, text, fontsize=9):
    t = hp.projtext(
        lon, lat, text, lonlat=True,
        color="black", fontsize=fontsize,
        ha="center", va="center"
    )
    try:
        t.set_path_effects([
            pe.withStroke(linewidth=2.2, foreground="white")
        ])
    except Exception:
        pass

# =============================================================================
# Mollweide：standard / bubble enhanced
# =============================================================================
def plot_mollweide(hpxmap, time, output_png, vmin, vmax, title):
    logmap = make_log_map(hpxmap, vmin, vmax)

    plt.close("all")
    hp.mollview(
        logmap,
        nest=False,
        flip="astro",
        rot=(0, 0, 0),
        cmap=CMAP,
        min=np.log10(vmin),
        max=np.log10(vmax),
        cbar=False,
        title=title,
        xsize=1800
    )

    fig = plt.gcf()
    fig.set_size_inches(12.0, 7.0)

    hp.graticule(
        dpar=30, dmer=60,
        color="0.35", alpha=0.45, linewidth=0.7
    )

    # 与PPT一致：0--360°经度
    for lon, label in [
        (120, "120°"), (60, "60°"), (0, "0°"),
        (300, "300°"), (240, "240°")
    ]:
        sky_text(lon, 2.0, label, 10)

    for lat in [60, 30, 0, -30, -60]:
        sky_text(170.0, lat, f"{lat}°", 9)

    sky_text(0.0, -5.0, "GC", 8)

    fig.text(
        0.50, 0.165,
        r"Galactic longitude $l$ [deg]",
        ha="center", fontsize=10
    )
    fig.text(
        0.035, 0.52,
        r"Galactic latitude $b$ [deg]",
        rotation=90, va="center", fontsize=10
    )

    add_colorbar(fig, vmin, vmax, [0.30, 0.085, 0.40, 0.032])

    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  保存：{output_png}")
    print(f"  colorbar = {vmin:.4e} -- {vmax:.4e}")

# =============================================================================
# GC zoom
# =============================================================================
def healpix_to_lb_image(hpxmap):
    lon = np.linspace(-GC_LON_HALF, GC_LON_HALF, LOCAL_NX)
    lat = np.linspace(-GC_LAT_HALF, GC_LAT_HALF, LOCAL_NY)
    lon2d, lat2d = np.meshgrid(lon, lat)

    image = hp.get_interp_val(
        hpxmap, lon2d, lat2d,
        nest=False, lonlat=True
    )
    return np.asarray(image, dtype=np.float64), lon, lat

def plot_gc_zoom(hpxmap, time, output_png, vmin, vmax):
    image, lon, lat = healpix_to_lb_image(hpxmap)

    image[~np.isfinite(image) | (image <= 0.0)] = vmin
    image = np.clip(image, vmin, vmax)
    image = np.log10(image)

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    im = ax.imshow(
        image,
        origin="lower",
        extent=[
            -GC_LON_HALF, GC_LON_HALF,
            -GC_LAT_HALF, GC_LAT_HALF
        ],
        cmap=CMAP,
        vmin=np.log10(vmin),
        vmax=np.log10(vmax),
        interpolation="bilinear",
        aspect="auto"
    )

    # 天文学习惯：+l在左
    ax.set_xlim(GC_LON_HALF, -GC_LON_HALF)
    ax.set_ylim(-GC_LAT_HALF, GC_LAT_HALF)

    ax.set_xlabel(r"Galactic longitude $l$ [deg]", fontsize=11)
    ax.set_ylabel(r"Galactic latitude $b$ [deg]", fontsize=11)

    ax.set_xticks(np.arange(-60, 61, 20))
    ax.set_yticks(np.arange(-80, 81, 20))
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    ax.axvline(0.0, color="0.3", linewidth=0.8)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.axhline(
        BUBBLE_COLOR_LAT_MIN,
        color="0.35", linestyle="--", linewidth=0.7
    )
    ax.axhline(
        -BUBBLE_COLOR_LAT_MIN,
        color="0.35", linestyle="--", linewidth=0.7
    )

    ax.set_title(
        "eROSITA-like X-ray 0.6--1.0 keV\n"
        f"GC bubble-enhanced view | Time={time:.3f} Gyr"
    )

    sm = ScalarMappable(
        norm=Normalize(np.log10(vmin), np.log10(vmax)),
        cmap=CMAP
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.025)
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(np.log10(ticks))
    cbar.set_ticklabels([scientific_label(v) for v in ticks])
    cbar.set_label(
        r"$I_{0.6-1.0\,\mathrm{keV}}$ "
        r"$[\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}}]$"
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  保存：{output_png}")

# =============================================================================
# 南北bubble简单统计
# =============================================================================
def print_bubble_statistics(hpxmap):
    _, lon, lat = lon_lat()

    north = (
        (np.abs(lon) <= 60.0)
        & (lat >= 10.0) & (lat <= 70.0)
        & np.isfinite(hpxmap) & (hpxmap > 0.0)
    )
    south = (
        (np.abs(lon) <= 60.0)
        & (lat <= -10.0) & (lat >= -70.0)
        & np.isfinite(hpxmap) & (hpxmap > 0.0)
    )

    for name, mask in [("North", north), ("South", south)]:
        x = hpxmap[mask]
        if x.size:
            p50, p90, p99 = np.percentile(x, [50, 90, 99])
            print(
                f"  {name}: median={p50:.4e}, "
                f"P90={p90:.4e}, P99={p99:.4e}"
            )

# =============================================================================
# 单snapshot
# =============================================================================
def process_snapshot(snapshot_file):
    print("\n" + "="*78)
    print(f"处理：{snapshot_file}")

    grid_file = grid_filename(snapshot_file)
    npy_file = npy_filename(snapshot_file)

    # 已有HEALPix数据则只重新画图
    if not OVERWRITE_ALLSKY and os.path.exists(npy_file):
        hpxmap = np.asarray(np.load(npy_file), dtype=np.float64).reshape(-1)

        if hpxmap.size != hp.nside2npix(NSIDE):
            raise RuntimeError("已有.npy与当前NSIDE不匹配。")

        if os.path.exists(grid_file):
            with h5py.File(grid_file, "r") as f:
                time = float(f.attrs["time"])
        else:
            _, time = get_bh_center_and_time(snapshot_file)

        print(f"  使用已有HEALPix：{npy_file}")

    else:
        if OVERWRITE_GRID or not os.path.exists(grid_file):
            build_xray_grid(snapshot_file, grid_file)
        else:
            print(f"  使用已有cube：{grid_file}")

        cube, time, radius = load_xray_grid(grid_file)
        hpxmap = make_allsky(cube, radius)
        np.save(npy_file, hpxmap)
        print(f"  保存HEALPix：{npy_file}")
        del cube
        gc.collect()

    print_bubble_statistics(hpxmap)

    standard_vmin, standard_vmax = get_standard_limits(hpxmap)
    bubble_vmin, bubble_vmax = get_bubble_limits(hpxmap)

    if MAKE_STANDARD:
        plot_mollweide(
            hpxmap, time,
            output_filename(snapshot_file, "allsky_standard"),
            standard_vmin, standard_vmax,
            "eROSITA-like X-ray intensity 0.6--1.0 keV"
            f"   |   Time={time:.3f} Gyr"
        )

    if MAKE_BUBBLE_ENHANCED:
        plot_mollweide(
            hpxmap, time,
            output_filename(snapshot_file, "allsky_bubble_enhanced"),
            bubble_vmin, bubble_vmax,
            "eROSITA-like X-ray 0.6--1.0 keV | "
            "bubble-enhanced color scale"
            f"   |   Time={time:.3f} Gyr"
        )

    if MAKE_GC_ZOOM:
        plot_gc_zoom(
            hpxmap, time,
            output_filename(snapshot_file, "gc_zoom"),
            bubble_vmin, bubble_vmax
        )

    del hpxmap
    gc.collect()

# =============================================================================
# main
# =============================================================================
def main():
    snapshots = sorted(
        glob.glob(os.path.join(SNAPSHOT_DIR, SNAPSHOT_GLOB))
    )
    if not snapshots:
        raise FileNotFoundError(
            f"没有找到：{os.path.join(SNAPSHOT_DIR, SNAPSHOT_GLOB)}"
        )

    dx = 2.0 * GRID_RADIUS_KPC / GRID_N
    ds_los = (LOS_MAX_KPC - LOS_MIN_KPC) / (N_LOS - 1)

    print("脚本：YT-APEC-R22-bubble-diagnostic-v2")
    print(f"snapshot数 = {len(snapshots)}")
    print(f"yt={yt.__version__}, healpy={hp.__version__}")
    print("Metallicity = Metallicity_00 + ... + Metallicity_10")
    print(f"Grid = {GRID_N}^3, ±{GRID_RADIUS_KPC} kpc, dx={dx:.4f} kpc")
    print(f"LOS = {LOS_MIN_KPC}-{LOS_MAX_KPC} kpc, N={N_LOS}, ds={ds_los:.4f} kpc")
    print(f"Sun = {SUN_OFFSET_KPC} kpc, NSIDE={NSIDE}")
    print(f"OVERWRITE_GRID={OVERWRITE_GRID}")
    print(f"OVERWRITE_ALLSKY={OVERWRITE_ALLSKY}")

    for fn in snapshots:
        process_snapshot(fn)

    print("\n全部处理完成。")

if __name__ == "__main__":
    main()