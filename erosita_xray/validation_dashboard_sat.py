#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【卫星期-终极验证面板 v4.0】(纯 yt 渲染 + PIL 原生像素拼接版)
1. 彻底移除 matplotlib 的 subplots 嵌套逻辑，解放 yt 原生渲染引擎。
2. 图1、图2 完全使用 yt.ParticlePlot 和 yt.ProjectionPlot 独立生成。
3. 利用 PIL 将四张图横竖完美拼贴为 2x2 面板，绝无轴线干扰。
"""

import matplotlib

matplotlib.use('Agg')
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import yt
from PIL import Image

# 屏蔽 yt 冗长日志
yt.funcs.mylog.setLevel(40)

# ========== 参数设置 ==========
snapshot_dir = './output'
figure_dir = os.path.join(snapshot_dir, 'figures_validation_pure')
os.makedirs(figure_dir, exist_ok=True)

snap_files = sorted([f for f in os.listdir(snapshot_dir) if f.startswith('snapshot_') and f.endswith('.hdf5')])
print(f"🔥 发现 {len(snap_files)} 个快照，启动 v4.0 (纯 yt 渲染 + PIL 像素拼接) 管线...")

# 轨道历史全局记录
history_time, history_lmc_dist, history_lmc_vel = [], [], []


def extract_core_data(filename):
    with h5py.File(filename, 'r') as f:
        time = f['Header'].attrs['Time']
        if 'PartType4' not in f or 'PartType0' not in f: return None, time, None, np.nan, np.nan

        pos_s = f['PartType4']['Coordinates'][:]
        vel_s = f['PartType4']['Velocities'][:] if 'Velocities' in f['PartType4'] else np.zeros_like(pos_s)
        mass_s = f['PartType4']['Masses'][:]
        ids_s = f['PartType4']['ParticleIDs'][:]
        pos_g = f['PartType0']['Coordinates'][:]
        vel_g = f['PartType0']['Velocities'][:] if 'Velocities' in f['PartType0'] else np.zeros_like(pos_g)

        is_lmc_s = (ids_s >= 1000000000) & (ids_s < 2000000000)
        is_smc_s = (ids_s >= 2000000000) & (ids_s < 3000000000)
        mw_mask_s = ~(is_lmc_s | is_smc_s)

        com_guess = np.average(pos_s[mw_mask_s], axis=0, weights=mass_s[mw_mask_s])
        core_mask = mw_mask_s & (np.sqrt(np.sum((pos_s - com_guess) ** 2, axis=1)) < 10.0)
        if np.sum(core_mask) > 100:
            com = np.average(pos_s[core_mask], axis=0, weights=mass_s[core_mask])
            v_com = np.average(vel_s[core_mask], axis=0, weights=mass_s[core_mask])
        else:
            com, v_com = com_guess, np.zeros(3)

        p_pos_s = pos_s[mw_mask_s] - com
        p_vel_s = vel_s[mw_mask_s] - v_com
        R_star = np.sqrt(p_pos_s[:, 0] ** 2 + p_pos_s[:, 1] ** 2)
        Vphi_star = (p_pos_s[:, 0] * p_vel_s[:, 1] - p_pos_s[:, 1] * p_vel_s[:, 0]) / (R_star + 1e-6)

        p_pos_g = pos_g - com
        p_vel_g = vel_g - v_com
        disk_gas_mask = (np.abs(p_pos_g[:, 2]) < 2.0) & (np.sqrt(np.sum(p_pos_g ** 2, axis=1)) < 30.0)
        R_gas = np.sqrt(p_pos_g[disk_gas_mask][:, 0] ** 2 + p_pos_g[disk_gas_mask][:, 1] ** 2)
        Vphi_gas = (p_pos_g[disk_gas_mask][:, 0] * p_vel_g[disk_gas_mask][:, 1] - p_pos_g[disk_gas_mask][:, 1] *
                    p_vel_g[disk_gas_mask][:, 0]) / (R_gas + 1e-6)

        dist_lmc, vtot_lmc = np.nan, np.nan
        if np.sum(is_lmc_s) > 0:
            dist_lmc = np.linalg.norm(np.average(pos_s[is_lmc_s] - com, axis=0, weights=mass_s[is_lmc_s]))
            vtot_lmc = np.linalg.norm(np.average(vel_s[is_lmc_s] - v_com, axis=0, weights=mass_s[is_lmc_s]))

        kinematics = {'star_R': R_star, 'star_Vphi': Vphi_star, 'gas_R': R_gas, 'gas_Vphi': Vphi_gas}
    return com, time, kinematics, dist_lmc, vtot_lmc


for fname in snap_files:
    snap_num = int(fname.split('_')[1].split('.')[0])
    filename = os.path.join(snapshot_dir, fname)

    try:
        com, time, mw_kin, d_lmc, v_lmc = extract_core_data(filename)
        if com is None: continue
        history_time.append(time)
        history_lmc_dist.append(d_lmc)
        history_lmc_vel.append(v_lmc)
    except Exception as e:
        print(f"提取失败 {snap_num}: {e}");
        continue

    print(f">> 渲染快照 {snap_num:03d} [Time={time:.2f} Gyr] ...")

    # 定义四张独立碎片图的路径
    tmp_1 = os.path.join(figure_dir, 'tmp_1.png')
    tmp_2 = os.path.join(figure_dir, 'tmp_2.png')
    tmp_3 = os.path.join(figure_dir, 'tmp_3.png')
    tmp_4 = os.path.join(figure_dir, 'tmp_4.png')

    # ========== 图 1：纯 yt 微观恒星盘 (Face-on) ==========
    ds = yt.load(filename)
    center = ds.arr(com, 'kpc')

    # 使用纯 yt 的 ParticlePlot 绘制恒星表面密度
    p1 = yt.ParticlePlot(ds, ('PartType4', 'particle_position_x'), ('PartType4', 'particle_position_y'),
                         ('PartType4', 'particle_mass'), center=center, width=(30.0, 'kpc'))
    p1.set_cmap(('PartType4', 'particle_mass'), 'magma')
    p1.annotate_title('MW Core Integrity (Face-on Stellar)')
    p1.save(tmp_1)

    # ========== 图 2：纯 yt 宏观星流 (Edge-on Gas) ==========
    # 使用纯 yt 的 ProjectionPlot 沿着 X 轴投影气体密度
    p2 = yt.ProjectionPlot(ds, 'x', ('gas', 'density'), center=center, width=(600.0, 'kpc'))
    p2.set_cmap(('gas', 'density'), 'viridis')
    p2.annotate_title('Magellanic Stream (Edge-on Gas 600kpc)')
    p2.save(tmp_2)

    # ========== 图 3：独立 matplotlib 旋转曲线 ==========
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    R_bins = np.linspace(0.5, 20.0, 40)
    bin_c = (R_bins[:-1] + R_bins[1:]) / 2
    if len(mw_kin['star_R']) > 0:
        stat_s, _, _ = binned_statistic(mw_kin['star_R'], mw_kin['star_Vphi'], statistic='median', bins=R_bins)
        ax.plot(bin_c, stat_s, label=r'Stars $V_\phi$', color='darkorange', lw=3)
    if len(mw_kin['gas_R']) > 0:
        stat_g, _, _ = binned_statistic(mw_kin['gas_R'], mw_kin['gas_Vphi'], statistic='median', bins=R_bins)
        ax.plot(bin_c, stat_g, label=r'Gas $V_\phi$', color='cyan', lw=2, ls='-.')
    ax.axhline(220, color='red', ls='--');
    ax.axvline(8.0, color='grey', ls=':')
    ax.plot(8.0, 220, marker='*', color='red', ms=15)
    ax.set(xlim=(0, 20), ylim=(0, 350), xlabel='Radius R [kpc]', ylabel=r'$V_\phi$ [km/s]',
           title='MW Rotation Curve Validation')
    ax.grid(True, alpha=0.3);
    ax.legend(loc='lower right')
    fig.tight_layout();
    fig.savefig(tmp_3);
    plt.close(fig)

    # ========== 图 4：独立 matplotlib 轨道追踪 ==========
    fig, ax_dist = plt.subplots(figsize=(8, 7), dpi=150)
    ax_vel = ax_dist.twinx()
    ax_dist.plot(history_time, history_lmc_dist, 'b-', lw=3, label='LMC Dist')
    ax_vel.plot(history_time, history_lmc_vel, 'g-', lw=3, label='LMC Vel')

    target_dist, target_vel, vel_err = 49.66, 321.0, 24.0
    ax_dist.axhline(target_dist, color='blue', ls='--', alpha=0.5)
    ax_vel.axhline(target_vel, color='green', ls='--', alpha=0.5)
    ax_vel.fill_between([0.45, 2.05], target_vel - vel_err, target_vel + vel_err, color='green', alpha=0.1)

    ax_dist.set(xlim=(0.45, 2.05), xlabel='Time [Gyr]', ylabel='Distance [kpc]', title='LMC Orbital Target Tracking')
    ax_vel.set_ylabel('Velocity [km/s]')
    ax_dist.grid(True, alpha=0.3)
    fig.tight_layout();
    fig.savefig(tmp_4);
    plt.close(fig)

    # ========== 🌟 PIL 终极原生拼接 ==========
    # 强制所有图片拉伸至相同的高清分辨率，防止 yt 和 matplotlib 的尺寸差异
    try:
        img1, img2, img3, img4 = Image.open(tmp_1), Image.open(tmp_2), Image.open(tmp_3), Image.open(tmp_4)

        # 以 yt 输出的图片 1 的尺寸为标准基准 (例如 1000 x 800)
        target_size = img1.size
        img2 = img2.resize(target_size, Image.LANCZOS)
        img3 = img3.resize(target_size, Image.LANCZOS)
        img4 = img4.resize(target_size, Image.LANCZOS)

        w, h = target_size
        panel = Image.new('RGB', (w * 2, h * 2), (255, 255, 255))

        # 2x2 拼贴
        panel.paste(img1, (0, 0))
        panel.paste(img2, (w, 0))
        panel.paste(img3, (0, h))
        panel.paste(img4, (w, h))

        panel_filename = os.path.join(figure_dir, f'validation_dashboard_{snap_num:03d}.png')
        panel.save(panel_filename)

    except Exception as e:
        print(f"拼接失败: {e}")
    finally:
        # 清理垃圾，保持目录纯净
        for f in [tmp_1, tmp_2, tmp_3, tmp_4]:
            if os.path.exists(f): os.remove(f)

print("\n🎉 v4.0 (纯 yt 渲染 + PIL 拼接版) 执行成功！面板连图已保存。")