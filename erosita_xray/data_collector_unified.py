#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【孤立期/干涉期通用 - 终极数据采集版 v4.2 (单位统一 + 卫星修复)】

v4.2 关键修复：
1. ID 区间从 1e9 修正为 1e8（匹配 add_lmc_smc_gaswake_orbit_v3.py）
2. 所有质量从代码单位 (1e10 Msun) 统一转为 Msun
3. 时间统一转为 Gyr
4. 无恒星卫星的 shrinking_sphere_com 回退策略
5. LMC 束缚气体半径从 MagellanicModelReference 动态读取
6. BH 坐标转为相对质心系
"""

import h5py
import numpy as np
import os
import sys

# ========== 用户设置 ==========
snapshot_dir = './output_sat'
snap_base = 'snapshot'
snap_nums = list(range(80))
output_file = 'output_sat/disk_data_unified.csv'

R_disk_max, z_disk_max = 10.0, 2.0
R_center_max, z_center_max = 3.0, 1.0

# ========== 代码单位常量 ==========
# 来自 make_ic_live_mw_cgm_isolated_v3.py / add_lmc_smc_gaswake_orbit_v3.py
UNIT_MASS_MSUN = 1.0e10            # 代码质量单位 = 1e10 Msun
UNIT_LENGTH_KPC = 1.0              # 代码长度单位 = 1 kpc
UNIT_VELOCITY_KMS = 1.0            # 代码速度单位 = 1 km/s
UNIT_TIME_GYR = 0.977792222        # 代码时间单位 = 0.978 Gyr
INSERTION_TIME_CODE = 0.0          # Stage-2 IC 起始时间

# 体积冷却率转换：rho_code * cool_code -> erg/cm^3/s
COOL_UNIT_CGS = 2.192e-28


def load_snapshot(filename):
    """读取快照，所有物理量转为 Msun / Gyr / km/s。"""
    with h5py.File(filename, 'r') as f:
        time_code = f['Header'].attrs['Time']
        time_gyr = (time_code - INSERTION_TIME_CODE) * UNIT_TIME_GYR

        pos_list, mass_list, vel_list, ptype_list, id_list, age_list = [], [], [], [], [], []
        rho_list, cool_list, sfr_list = [], [], []

        for i in [0, 1, 4]:
            group = f'PartType{i}'
            if group not in f:
                continue
            pos = f[group]['Coordinates'][:]
            mass = f[group]['Masses'][:] * UNIT_MASS_MSUN      # 代码单位 -> Msun
            vel = f[group]['Velocities'][:] if 'Velocities' in f[group] else np.zeros_like(pos)
            ids = f[group]['ParticleIDs'][:].astype(np.uint64)
            ages = f[group]['StellarFormationTime'][:] if (
                    i == 4 and 'StellarFormationTime' in f[group]) else np.full(len(pos), -1.0)

            if i == 0:
                rho = f[group]['Density'][:] if 'Density' in f[group] else np.zeros(len(pos))
                cool = f[group]['CoolingRate'][:] if 'CoolingRate' in f[group] else np.zeros(len(pos))
                sfr = f[group]['StarFormationRate'][:] if 'StarFormationRate' in f[group] else np.zeros(len(pos))
            else:
                rho = np.zeros(len(pos)); cool = np.zeros(len(pos)); sfr = np.zeros(len(pos))

            pos_list.append(pos); mass_list.append(mass); vel_list.append(vel)
            ptype_list.append(np.full(len(pos), i, dtype=np.int32))
            id_list.append(ids); age_list.append(ages)
            rho_list.append(rho); cool_list.append(cool); sfr_list.append(sfr)

        # 卫星 ID 区间：优先从 MagellanicOrbitReference 动态读取
        sat_ranges = {
            'LMC_ID_START': 100_000_000, 'LMC_ID_END': 199_999_999,
            'SMC_ID_START': 200_000_000, 'SMC_ID_END': 299_999_999,
        }
        lmc_gas_rmax_kpc = 50.0
        if 'MagellanicOrbitReference' in f:
            ref = f['MagellanicOrbitReference'].attrs
            for key in sat_ranges:
                if key in ref:
                    sat_ranges[key] = int(ref[key])
            if 'LMC_GasRmax_kpc' in ref:
                lmc_gas_rmax_kpc = float(ref['LMC_GasRmax_kpc'])

        # BH 数据
        bh_data = {k: np.nan for k in ('mass','mdot','x','y','z','jx','jy','jz')}
        if 'PartType5' in f and 'Coordinates' in f['PartType5'] and len(f['PartType5']['Coordinates']) > 0:
            g5 = f['PartType5']
            bh_pos = g5['Coordinates'][0]
            bh_data['x'], bh_data['y'], bh_data['z'] = bh_pos[0], bh_pos[1], bh_pos[2]
            if 'BH_Mass' in g5:
                bh_data['mass'] = g5['BH_Mass'][0] * UNIT_MASS_MSUN
            if 'BH_Mdot' in g5:
                bh_data['mdot'] = g5['BH_Mdot'][0] * UNIT_MASS_MSUN / (UNIT_TIME_GYR * 1e9)
            if 'BH_Specific_AngMom' in g5:
                bj = g5['BH_Specific_AngMom'][0]
                bh_data['jx'], bh_data['jy'], bh_data['jz'] = bj[0], bj[1], bj[2]

        if not pos_list:
            return (time_gyr, None, None, None, None, None, None, None,
                    None, None, bh_data, sat_ranges, lmc_gas_rmax_kpc)

        return (time_gyr, np.vstack(pos_list), np.concatenate(mass_list),
                np.vstack(vel_list), np.concatenate(ptype_list),
                np.concatenate(id_list), np.concatenate(age_list),
                np.concatenate(rho_list), np.concatenate(cool_list),
                np.concatenate(sfr_list), bh_data, sat_ranges, lmc_gas_rmax_kpc)


def shrinking_sphere_com(pos, mass, vel, mask):
    """恒星收缩球迭代法。mask 为空时返回 NaN 而非零向量。"""
    if np.sum(mask) < 10:
        return np.full(3, np.nan), np.full(3, np.nan)

    com1 = np.average(pos[mask], axis=0, weights=mass[mask])
    dist1 = np.linalg.norm(pos - com1, axis=1)
    mask_20 = mask & (dist1 < 20.0)
    if np.sum(mask_20) < 50:
        return com1, np.average(vel[mask], axis=0, weights=mass[mask])

    com2 = np.average(pos[mask_20], axis=0, weights=mass[mask_20])
    dist2 = np.linalg.norm(pos - com2, axis=1)
    mask_5 = mask & (dist2 < 5.0)
    if np.sum(mask_5) < 100:
        mask_5 = mask_20

    com = np.average(pos[mask_5], axis=0, weights=mass[mask_5])
    vcom = np.average(vel[mask_5], axis=0, weights=mass[mask_5])
    return com, vcom


def disk_ellipse_params(pos_centered, mass, ptype, Rmax, zmax, part_types=(4,)):
    x, y, z = pos_centered[:, 0], pos_centered[:, 1], pos_centered[:, 2]
    mask = ((np.sqrt(x**2 + y**2) <= Rmax) & (np.abs(z) <= zmax)
            & np.isin(ptype, part_types))
    if np.sum(mask) < 50:
        return np.nan, np.nan, 0
    M = np.sum(mass[mask])
    cov = np.array([
        [np.sum(mass[mask]*x[mask]**2)/M, np.sum(mass[mask]*x[mask]*y[mask])/M],
        [np.sum(mass[mask]*x[mask]*y[mask])/M, np.sum(mass[mask]*y[mask]**2)/M],
    ])
    eigvals, eigvecs = np.linalg.eigh(cov)
    b_over_a = np.sqrt(max(eigvals[0], 0) / max(eigvals[1], 1e-10)) if eigvals[1] > 0 else np.nan
    pa = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return b_over_a, pa, int(np.sum(mask))


def main():
    print(f">> 开始扫描快照目录: {snapshot_dir}")
    snap_files = [(n, os.path.join(snapshot_dir, f"{snap_base}_{n:03d}.hdf5"))
                  for n in snap_nums
                  if os.path.exists(os.path.join(snapshot_dir, f"{snap_base}_{n:03d}.hdf5"))]
    if not snap_files:
        sys.exit("!! 未找到任何快照文件")

    header = ("snap,time_gyr,com_x,com_y,com_z,vcom_x,vcom_y,vcom_z,"
              "dist_lmc,dist_smc,lmc_vtot,lmc_bound_gas,"
              "disk_ba,bar_ba,mw_vphi_8kpc,new_star_mass,total_sfr,mean_vol_cool_erg_cm3_s,"
              "bh_mass,bh_mdot_msun_yr,bh_x,bh_y,bh_z,bh_jx,bh_jy,bh_jz,bh_offset\n")

    with open(output_file, 'w') as f_out:
        f_out.write(header)

        for snap_num, fname in snap_files:
            (time_gyr, pos_all, mass_all, vel_all, ptype_all, id_all,
             age_all, rho_all, cool_all, sfr_all, bh_data,
             sat_ranges, lmc_gas_rmax_kpc) = load_snapshot(fname)

            if pos_all is None:
                continue

            # ---- 卫星识别：使用动态 ID 区间 ----
            is_lmc = (id_all >= sat_ranges['LMC_ID_START']) & (id_all <= sat_ranges['LMC_ID_END'])
            is_smc = (id_all >= sat_ranges['SMC_ID_START']) & (id_all <= sat_ranges['SMC_ID_END'])
            mw_mask = ~(is_lmc | is_smc)
            mw_star_mask = mw_mask & (ptype_all == 4)

            # 首次运行时打印调试信息
            if snap_num == snap_files[0][0]:
                print(f"[debug] snap {snap_num}: N_LMC={int(is_lmc.sum())}, N_SMC={int(is_smc.sum())}")
                print(f"[debug] LMC ID range = [{sat_ranges['LMC_ID_START']}, {sat_ranges['LMC_ID_END']}]")
                print(f"[debug] SMC ID range = [{sat_ranges['SMC_ID_START']}, {sat_ranges['SMC_ID_END']}]")
                print(f"[debug] LMC gas Rmax = {lmc_gas_rmax_kpc} kpc")

            # ---- MW 质心 ----
            com, v_com = shrinking_sphere_com(pos_all, mass_all, vel_all, mw_star_mask)
            if not np.all(np.isfinite(com)):
                print(f"[warn] snap {snap_num}: MW COM 未收敛，跳过")
                continue
            pos_c = pos_all - com
            vel_c = vel_all - v_com

            # ---- LMC/SMC 追踪（带回退） ----
            dist_lmc = dist_smc = lmc_vtot = np.nan
            lmc_bound_gas = 0.0

            lmc_star_mask = is_lmc & (ptype_all == 4)
            if np.sum(lmc_star_mask) >= 10:
                com_lmc, v_lmc = shrinking_sphere_com(pos_c, mass_all, vel_c, lmc_star_mask)
            elif np.sum(is_lmc) >= 10:
                com_lmc = np.average(pos_c[is_lmc], axis=0, weights=mass_all[is_lmc])
                v_lmc = np.average(vel_c[is_lmc], axis=0, weights=mass_all[is_lmc])
            else:
                com_lmc = v_lmc = np.full(3, np.nan)

            if np.all(np.isfinite(com_lmc)):
                dist_lmc = np.linalg.norm(com_lmc)
                lmc_vtot = np.linalg.norm(v_lmc)
                lmc_gas_mask = is_lmc & (ptype_all == 0)
                if np.sum(lmc_gas_mask) > 0:
                    gas_dist = np.linalg.norm(pos_c[lmc_gas_mask] - com_lmc, axis=1)
                    lmc_bound_gas = np.sum(mass_all[lmc_gas_mask][gas_dist < 1.5 * lmc_gas_rmax_kpc])

            smc_star_mask = is_smc & (ptype_all == 4)
            if np.sum(smc_star_mask) >= 10:
                com_smc, _ = shrinking_sphere_com(pos_c, mass_all, vel_c, smc_star_mask)
            elif np.sum(is_smc) >= 10:
                com_smc = np.average(pos_c[is_smc], axis=0, weights=mass_all[is_smc])
            else:
                com_smc = np.full(3, np.nan)
            if np.all(np.isfinite(com_smc)):
                dist_smc = np.linalg.norm(com_smc)

            # ---- MW 结构与热力学 ----
            ba_disk, _, _ = disk_ellipse_params(pos_c[mw_mask], mass_all[mw_mask],
                                                ptype_all[mw_mask], R_disk_max, z_disk_max, (4,))
            ba_bar, _, _ = disk_ellipse_params(pos_c[mw_mask], mass_all[mw_mask],
                                               ptype_all[mw_mask], R_center_max, z_center_max, (4,))

            new_star_mask = mw_mask & (ptype_all == 4) & (age_all > 0)
            new_star_mass = np.sum(mass_all[new_star_mask]) if np.sum(new_star_mask) > 0 else 0.0

            # 8 kpc 旋转曲线
            solar_mask = mw_mask & (ptype_all == 4) & (np.abs(pos_c[:, 2]) < 2.0)
            vphi_8kpc = np.nan
            if np.sum(solar_mask) > 0:
                ps, vs = pos_c[solar_mask], vel_c[solar_mask]
                R_star = np.sqrt(ps[:, 0]**2 + ps[:, 1]**2)
                Vphi = (ps[:, 0]*vs[:, 1] - ps[:, 1]*vs[:, 0]) / (R_star + 1e-6)
                ring = (R_star > 7.5) & (R_star < 8.5)
                if np.sum(ring) > 10:
                    vphi_8kpc = np.median(Vphi[ring])

            # 恒星形成率与体积冷却率
            mw_gas_mask = (mw_mask & (ptype_all == 0)
                           & (np.sqrt(pos_c[:, 0]**2 + pos_c[:, 1]**2) <= 15.0)
                           & (np.abs(pos_c[:, 2]) <= 2.0))
            total_sfr = np.sum(sfr_all[mw_gas_mask]) if np.sum(mw_gas_mask) > 0 else 0.0
            cooling_mask = mw_gas_mask & (cool_all != 0)
            if np.sum(cooling_mask) > 0:
                mean_vol_cool = np.mean(rho_all[cooling_mask]
                                        * np.abs(cool_all[cooling_mask])) * COOL_UNIT_CGS
            else:
                mean_vol_cool = 0.0

            # ---- BH：转为相对质心系 ----
            bh_mass = bh_data['mass']
            bh_mdot = bh_data['mdot']
            if np.isfinite(bh_data['x']):
                bh_rel = np.array([bh_data['x'], bh_data['y'], bh_data['z']]) - com
                bh_x, bh_y, bh_z = bh_rel
                bh_offset = np.linalg.norm(bh_rel)
            else:
                bh_x = bh_y = bh_z = bh_offset = np.nan
            bh_jx, bh_jy, bh_jz = bh_data['jx'], bh_data['jy'], bh_data['jz']

            f_out.write(
                f"{snap_num},{time_gyr:.6f},{com[0]:.3f},{com[1]:.3f},{com[2]:.3f},"
                f"{v_com[0]:.3f},{v_com[1]:.3f},{v_com[2]:.3f},"
                f"{dist_lmc:.3f},{dist_smc:.3f},{lmc_vtot:.3f},{lmc_bound_gas:.6e},"
                f"{ba_disk:.4f},{ba_bar:.4f},{vphi_8kpc:.3f},{new_star_mass:.6e},"
                f"{total_sfr:.4f},{mean_vol_cool:.6e},"
                f"{bh_mass:.6e},{bh_mdot:.6e},{bh_x:.3f},{bh_y:.3f},{bh_z:.3f},"
                f"{bh_jx:.6e},{bh_jy:.6e},{bh_jz:.6e},{bh_offset:.4f}\n"
            )
            f_out.flush()

    print(f">> 数据采集完毕: {output_file}")


if __name__ == "__main__":
    main()