#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【卫星期-极简纯净拼接版 - GIZMO/yt 平滑长度单位兼容版】
- 黑洞中心锁定
- 200x200 kpc 视界
- 动态时间标题
- PIL 1x3 无损拼接
- ★ 总金属丰度使用 Metallicity_00 / 0.02，不对 11 列求和
"""

import os
import re
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gizmo_compat import load_gizmo, discover_snapshots
import h5py
import numpy as np
import yt
from PIL import Image




def load_yt_snapshot(path, cache_dir=None):
    """Use native GIZMO kernel IO before yt constructs its particle index."""
    return load_gizmo(path, cache_dir=cache_dir)

# 屏蔽冗长日志
yt.funcs.mylog.setLevel(40)

# ========== 1. 基础设置 ==========
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot-dir", default="./output")
parser.add_argument("--output-dir", default=None)
parser.add_argument("--start", type=int)
parser.add_argument("--stop", type=int)
parser.add_argument("--buff-size", type=int, default=800)
parser.add_argument("--apec-data-dir", default=None)
args = parser.parse_args()
snapshot_dir = args.snapshot_dir
figure_dir = args.output_dir or os.path.join(snapshot_dir, "figures_panel_yt")
os.makedirs(figure_dir, exist_ok=True)
fov_x = fov_y = 200.0
plot_width = ((fov_x, "kpc"), (fov_y, "kpc"))
snap_files = [f.name for f in discover_snapshots(snapshot_dir)
              if (args.start is None or int(re.search(r"snapshot_(\d+)", f.name)[1]) >= args.start)
              and (args.stop is None or int(re.search(r"snapshot_(\d+)", f.name)[1]) <= args.stop)]
print(f"Python: {sys.executable}; yt={yt.__version__}")
print(f"🔥 发现 {len(snap_files)} 个快照，启动 v14.1 (平滑长度单位修复；Z=Metallicity_00/0.02) 渲染管线...")


# ★ 定义标量总金属丰度
GIZMO_ZSUN = 0.02

def _apec_metallicity(field, data):
    z_mass_fraction = data["PartType0", "Metallicity_00"]
    z_over_zsun = z_mass_fraction / GIZMO_ZSUN
    return z_over_zsun * data.ds.quan(1.0, "Zsun")

yt.add_field(
    ("gas", "apec_metallicity"),
    function=_apec_metallicity,
    units="Zsun",
    sampling_type="local"
)


def get_galaxy_center_and_time(filename):
    """优先使用黑洞坐标，否则用核心恒星质心"""
    with h5py.File(filename, 'r') as f:
        time = f['Header'].attrs['Time']

        if 'PartType5' in f and 'Coordinates' in f['PartType5'] and len(f['PartType5']['Coordinates']) > 0:
            bh_pos = f['PartType5']['Coordinates'][0]
            return bh_pos, time

        if 'PartType4' not in f:
            return None, time

        pos_s = f['PartType4']['Coordinates'][:]
        mass_s = f['PartType4']['Masses'][:]
        ids_s = f['PartType4']['ParticleIDs'][:]
        mw_mask = (ids_s < 1000000000)
        if np.sum(mw_mask) == 0:
            return None, time

        com_guess = np.average(pos_s[mw_mask], axis=0, weights=mass_s[mw_mask])
        dist = np.sqrt(np.sum((pos_s - com_guess) ** 2, axis=1))
        core_mask = mw_mask & (dist < 5.0)
        com = np.average(pos_s[core_mask], axis=0, weights=mass_s[core_mask]) if np.sum(core_mask) > 100 else com_guess
        return com, time


# ========== 2. 主渲染循环 ==========
for fname in snap_files:
    snap_num = int(fname.split('_')[1].split('.')[0])
    filename = os.path.join(snapshot_dir, fname)

    com, time = get_galaxy_center_and_time(filename)
    if com is None:
        continue

    ds = load_yt_snapshot(filename, cache_dir=Path(figure_dir) / "yt_cache")
    time = float(ds.current_time.to_value("Gyr"))
    print(f">> 正在渲染快照 {snap_num:03d} | Time = {time:.3f} Gyr ...")

    tmp_f1 = os.path.join(figure_dir, f'tmp_1.png')
    tmp_f2 = os.path.join(figure_dir, f'tmp_2.png')
    tmp_f3 = os.path.join(figure_dir, f'tmp_3.png')

    center = ds.arr(com, 'code_length').to('kpc')

    # ---- 图 1：密度切片 + 速度流线 ----
    p_den = yt.SlicePlot(ds, 'x', ('gas', 'density'), center=center, width=plot_width, buff_size=(args.buff_size,args.buff_size))
    p_den.set_cmap(('gas', 'density'), 'viridis')
    p_den.annotate_streamlines(('gas', 'velocity_y'), ('gas', 'velocity_z'), density=0.5, color='white')
    p_den.annotate_title(f"Density | Time = {time:.3f} Gyr")
    p_den.save(tmp_f1)

    # ---- 图 2：温度切片 ----
    p_temp = yt.SlicePlot(ds, 'x', ('gas', 'temperature'), center=center, width=plot_width, buff_size=(args.buff_size,args.buff_size))
    p_temp.set_cmap(('gas', 'temperature'), 'magma')
    p_temp.annotate_title(f"Temperature Slice | Time = {time:.3f} Gyr")
    p_temp.save(tmp_f2)

    # ---- 图 3：软 X 射线表面亮度（使用总金属质量分数） ----
    yt.add_xray_emissivity_field(ds, 0.6, 1.0, table_type='apec',
                                 metallicity=("gas", "apec_metallicity"), data_dir=args.apec_data_dir)
    p_xray = yt.ProjectionPlot(ds, 'x', "xray_emissivity_0.6_1.0_keV",
                               weight_field=None, center=center, width=plot_width, buff_size=(args.buff_size,args.buff_size))
    p_xray.set_cmap("xray_emissivity_0.6_1.0_keV", 'inferno')
    p_xray.set_background_color("xray_emissivity_0.6_1.0_keV", 'black')
    p_xray.annotate_title(f"eROSITA X-ray | Time = {time:.3f} Gyr")
    p_xray.save(tmp_f3)

    # ---- 拼接三连图 ----
    try:
        img1 = Image.open(tmp_f1)
        img2 = Image.open(tmp_f2)
        img3 = Image.open(tmp_f3)
        w, h = img1.size
        panel = Image.new('RGB', (w * 3, h), (255, 255, 255))
        panel.paste(img1, (0, 0))
        panel.paste(img2, (w, 0))
        panel.paste(img3, (w * 2, 0))
        panel.save(os.path.join(figure_dir, f'panel_snap_{snap_num:03d}.png'))
    except Exception as e:
        print(f"拼接失败: {e}")
    finally:
        for f in [tmp_f1, tmp_f2, tmp_f3]:
            if os.path.exists(f):
                os.remove(f)

print(f"\n🎉 v14.1 平滑长度单位兼容版执行成功！所有连图已保存至 {figure_dir}")
