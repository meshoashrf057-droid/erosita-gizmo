#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【类银河系终极鉴定仪表盘 v5.0】(纯 yt 渲染 + PIL 像素拼接版)
1. 彻底解耦：图1~图5 使用 yt 原生切片/粒子投影，图6 使用 matplotlib 1D 折线。
2. 无损画质：利用 PIL 横竖拼贴为 2x3 面板，杜绝 matplotlib 子图的轴线变形。
3. 纯净输出：执行完毕自动清理碎片图。
"""

import matplotlib

matplotlib.use('Agg')
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import yt
from PIL import Image, ImageDraw, ImageFont

# 屏蔽 yt 冗长日志
yt.funcs.mylog.setLevel(40)

# ========== 参数设置 ==========
snapshot_dir = './output'
figure_dir = os.path.join(snapshot_dir, 'figures_mw_validator_pure')
os.makedirs(figure_dir, exist_ok=True)

plot_width = (30, 'kpc')

snap_files = sorted([f for f in os.listdir(snapshot_dir) if f.startswith('snapshot_') and f.endswith('.hdf5')])
print(f"🔥 发现 {len(snap_files)} 个快照，启动【类银河系终极鉴定 - 纯 yt 拼接版】...")


def get_com_and_kinematics(filename):
    """读取质心，并提取计算旋转曲线所需的运动学数据"""
    with h5py.File(filename, 'r') as f:
        time = f['Header'].attrs['Time']
        pos_all, mass_all, ids_all = [], [], []

        for ptype in ['PartType0', 'PartType4']:
            if ptype in f:
                pos_all.append(f[ptype]['Coordinates'][:])
                mass_all.append(f[ptype]['Masses'][:])
                ids_all.append(f[ptype]['ParticleIDs'][:])

        if not pos_all: return np.array([500., 500., 500.]), time, None

        pos_cat = np.vstack(pos_all)
        mass_cat = np.concatenate(mass_all)
        ids_cat = np.concatenate(ids_all)

        guess_com = np.average(pos_cat, axis=0, weights=mass_cat)
        r_sph = np.sqrt(np.sum((pos_cat - guess_com) ** 2, axis=1))
        core_mask = (r_sph < 10.0) & (ids_cat < 1e9)

        if np.sum(core_mask) > 100:
            com = np.sum(mass_cat[core_mask, None] * pos_cat[core_mask], axis=0) / np.sum(mass_cat[core_mask])
        else:
            com = guess_com

        kinematics = {}
        for p_idx, p_name in zip(['PartType0', 'PartType4'], ['gas', 'star']):
            if p_idx in f and 'Velocities' in f[p_idx]:
                p_pos = f[p_idx]['Coordinates'][:] - com
                p_vel = f[p_idx]['Velocities'][:]

                R = np.sqrt(p_pos[:, 0] ** 2 + p_pos[:, 1] ** 2)
                v_phi = (p_pos[:, 0] * p_vel[:, 1] - p_pos[:, 1] * p_vel[:, 0]) / (R + 1e-6)
                kinematics[p_name] = {'R': R, 'Vphi': v_phi, 'z': p_pos[:, 2]}

    return com, time, kinematics


for fname in snap_files:
    snap_num = int(fname.split('_')[1].split('.')[0])
    print(f"\n>> 正在渲染快照 {snap_num:03d} ...")
    filename = os.path.join(snapshot_dir, fname)

    try:
        com, time, kinematics = get_com_and_kinematics(filename)
    except Exception as e:
        print(f"  读取失败: {e}");
        continue

    # 定义 6 张碎片图路径
    tmp_imgs = [os.path.join(figure_dir, f'tmp_{i}.png') for i in range(1, 7)]

    ds = yt.load(filename)
    center = ds.arr(com, 'kpc')


    # 添加自定义金属丰度场
    def _metal_z(field, data):
        if ('PartType0', 'Metallicity') in data.ds.field_list:
            met = data['PartType0', 'Metallicity']
            return met[:, 0] / 0.02 if len(met.shape) > 1 else met / 0.02
        else:
            return data['gas', 'metallicity'] / 0.02


    ds.add_field(('gas', 'metal_z'), function=_metal_z, sampling_type='particle', units='', force_override=True)

    # ==================== 面板 1: yt Face-on 恒星密度 ====================
    p1 = yt.ParticlePlot(ds, ('PartType4', 'particle_position_x'), ('PartType4', 'particle_position_y'),
                         ('PartType4', 'particle_mass'), center=center, width=plot_width)
    p1.set_cmap(('PartType4', 'particle_mass'), 'magma')
    p1.set_background_color(('PartType4', 'particle_mass'), 'black')
    p1.annotate_title('1. Face-on Stellar Mass')
    p1.save(tmp_imgs[0])

    # ==================== 面板 2: yt Edge-on 恒星密度 ====================
    p2 = yt.ParticlePlot(ds, ('PartType4', 'particle_position_y'), ('PartType4', 'particle_position_z'),
                         ('PartType4', 'particle_mass'), center=center, width=plot_width)
    p2.set_cmap(('PartType4', 'particle_mass'), 'magma')
    p2.set_background_color(('PartType4', 'particle_mass'), 'black')
    p2.annotate_title('2. Edge-on Stellar Mass')
    p2.save(tmp_imgs[1])

    # ==================== 面板 3: yt Edge-on 气体密度 ====================
    p3 = yt.SlicePlot(ds, 'x', ('gas', 'density'), center=center, width=plot_width)
    p3.set_cmap(('gas', 'density'), 'viridis')
    p3.set_zlim(('gas', 'density'), 1e-28, 1e-21)  # 对应原来 log10 的 -28 到 -21
    p3.set_background_color(('gas', 'density'), 'black')
    p3.annotate_title('3. Edge-on Gas Density')
    p3.save(tmp_imgs[2])

    # ==================== 面板 4: yt Edge-on 气体温度 ====================
    p4 = yt.SlicePlot(ds, 'x', ('gas', 'temperature'), center=center, width=plot_width)
    p4.set_cmap(('gas', 'temperature'), 'turbo')
    p4.set_zlim(('gas', 'temperature'), 1e1, 10 ** 6.5)  # 对应原来 log10 的 1.0 到 6.5
    p4.set_background_color(('gas', 'temperature'), 'black')
    p4.annotate_title('4. Edge-on Gas Temperature')
    p4.save(tmp_imgs[3])

    # ==================== 面板 5: yt Edge-on 气体金属丰度 ====================
    p5 = yt.SlicePlot(ds, 'x', ('gas', 'metal_z'), center=center, width=plot_width)
    p5.set_cmap(('gas', 'metal_z'), 'YlOrRd_r')
    p5.set_zlim(('gas', 'metal_z'), 1e-2, 1.0)
    p5.set_background_color(('gas', 'metal_z'), 'black')
    p5.annotate_title('5. Edge-on Gas Metallicity')
    p5.save(tmp_imgs[4])

    # ==================== 面板 6: matplotlib 旋转曲线 ====================
    fig, ax_rot = plt.subplots(figsize=(8, 7), dpi=150)
    if kinematics:
        R_bins = np.linspace(0.5, 20.0, 40)
        bin_centers = (R_bins[:-1] + R_bins[1:]) / 2

        if 'gas' in kinematics:
            z_mask = np.abs(kinematics['gas']['z']) < 2.0
            stat_gas, _, _ = binned_statistic(kinematics['gas']['R'][z_mask], kinematics['gas']['Vphi'][z_mask],
                                              statistic='median', bins=R_bins)
            ax_rot.plot(bin_centers, stat_gas, label=r'Gas $V_\phi$', color='blue', linewidth=3)

        if 'star' in kinematics:
            stat_star, _, _ = binned_statistic(kinematics['star']['R'], kinematics['star']['Vphi'], statistic='median',
                                               bins=R_bins)
            ax_rot.plot(bin_centers, stat_star, label=r'Star $V_\phi$', color='orange', linewidth=3)

        ax_rot.axhline(220, color='red', linestyle='--', label='MW Vel (220 km/s)')
        ax_rot.axvline(8.0, color='grey', linestyle=':', label='Solar Rad (8 kpc)')
        ax_rot.plot(8.0, 220, marker='*', color='red', markersize=15)

        ax_rot.set(xlim=(0, 20), ylim=(0, 300), xlabel='Radius R (kpc)', ylabel=r'Rotation Velocity $V_\phi$ (km/s)')
        ax_rot.set_title('6. Rotation Curve (Kinematic Proof)', fontsize=14, fontweight='bold')
        ax_rot.legend(loc='lower right')
        ax_rot.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(tmp_imgs[5])
    plt.close(fig)

    # ==================== 🌟 PIL 终极 2x3 原生拼接 ====================
    try:
        images = [Image.open(f) for f in tmp_imgs]

        # 统一强制对齐到 yt 图 1 的尺寸
        target_size = images[0].size
        images = [img.resize(target_size, Image.LANCZOS) for img in images]

        w, h = target_size
        # 创建 2行 x 3列 的大画布
        panel = Image.new('RGB', (w * 3, h * 2), (255, 255, 255))

        # 贴图
        panel.paste(images[0], (0, 0))  # (0,0) 第一排第一张
        panel.paste(images[1], (w, 0))  # (0,1) 第一排第二张
        panel.paste(images[2], (w * 2, 0))  # (0,2) 第一排第三张
        panel.paste(images[3], (0, h))  # (1,0) 第二排第一张
        panel.paste(images[4], (w, h))  # (1,1) 第二排第二张
        panel.paste(images[5], (w * 2, h))  # (1,2) 第二排第三张

        # 添加全局大标题
        draw = ImageDraw.Draw(panel)
        title_text = f"Milky Way Validation Dashboard | T = {time:.2f} Gyr"
        # 使用默认字体进行简单绘制 (如果你系统有特定字体，可用 ImageFont.truetype 加载)
        draw.text((w, 10), title_text, fill="black", align="center")

        panel_filename = os.path.join(figure_dir, f'mw_validator_snap{snap_num:03d}.png')
        panel.save(panel_filename)

    except Exception as e:
        print(f"拼接失败: {e}")
    finally:
        # 清理碎片垃圾图
        for f in tmp_imgs:
            if os.path.exists(f): os.remove(f)

print("\n🎉 孤立盘终极鉴定完成！所有连图已生成。")