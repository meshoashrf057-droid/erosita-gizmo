#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【快照接力与卫星空投引擎 - 双精度不朽版】
1. 完美继承 0.5 Gyr 时间轴，不重置时间，根除超新星爆发错乱Bug！
2. 强制全局 Float64 双精度，根除内存粉碎与 NaN 毒药！
3. 黑洞 11 维亚网格状态 100% 继承！

功能：从已有的孤立银河系快照（例如运行了0.5 Gyr的snapshot_005.hdf5）出发，
      读取其所有粒子数据（气体、暗物质、恒星、黑洞），保持时间戳不变，
      重新居中并消除整体运动，然后在盒子内额外生成大麦哲伦云（LMC）和
      小麦哲伦云（SMC）两个卫星星系，最终合并输出为新的初始条件文件，
      用于继续模拟。
"""

import h5py
import numpy as np

# ==================== 1. 用户核心配置区 ====================
# 输入：已经演化了一段时间的GIZMO快照文件（本代码设计用于0.5 Gyr快照）
input_snapshot = 'snapshot_006.hdf5'  # 【请修改】指向 0.5 Gyr 的孤立盘快照
output_ic = './mw_with_lmc_smc_ic.hdf5'  # 输出带卫星星系的新初始条件文件

# 模拟盒子参数
boxsize = 1000.0  # 盒子边长 [kpc]
box_center = np.array([boxsize / 2.0, boxsize / 2.0, boxsize / 2.0], dtype=np.float64)  # 盒子中心坐标

# ==================== 卫星星系初始位置与速度 ====================
# 🎯 【1.5 Gyr 完美高能滑翔轨道】
# 大麦哲伦云 (LMC) 相对于银河系中心的位置偏移（将在盒子中心基础上叠加）
pos_LMC_offset = np.array([45.0, 350.0, -15.0], dtype=np.float64)   # [kpc]
vel_LMC = np.array([-11.27, -162.88, -31.19], dtype=np.float64)        # [km/s]

# 小麦哲伦云 (SMC) 相对于 LMC 的位置和速度（紧密跟随）
pos_SMC_offset = pos_LMC_offset + np.array([-10.0, 10.0, -5.0], dtype=np.float64)
vel_SMC = vel_LMC + np.array([-20.0, 30.0, -10.0], dtype=np.float64)

# ==================== 物理常数与单位转换 ====================
G = 43007.1  # 引力常数，单位制：1e10 Msun, kpc, km/s
gamma = 5.0 / 3.0  # 绝热指数（单原子理想气体）
mu = 0.61          # 平均分子量（适用于热电离气体）
k_B_cgs = 1.380649e-16   # 玻尔兹曼常数 [erg/K]
m_p_cgs = 1.6726219e-24  # 质子质量 [g]
# 将温度转换为GIZMO内部比内能单位的因子
# 内能 u = k_B * T / ((gamma-1) * mu * m_p) ，再除以1e10转为代码单位
factor_erg = k_B_cgs / ((gamma - 1) * mu * m_p_cgs) / 1e10  # [ (km/s)^2 / K ]
u_warm_gas = 50000.0 * factor_erg  # 卫星星系中暖气体的比内能（对应T≈5e4 K）

np.random.seed(42)  # 固定随机种子，保证可重复性


# ==================== 2. 卫星星系生成器 (强制 float64) ====================
def sample_hernquist(N, a, M_tot):
    """
    从 Hernquist 密度分布中抽样 N 个粒子，返回位置和等质量粒子的质量。
    参数：
        N     : 粒子数量
        a     : Hernquist 标长 [kpc]
        M_tot : 该组分的总质量 [1e10 Msun]
    返回：
        pos   : (N,3) 位置数组 [kpc]，dtype=float64
        mass  : (N,)  质量数组，每个粒子质量相同
    """
    u = np.random.random(N)
    # 变换抽样：r = a * sqrt(u) / (1 - sqrt(u))
    r = a * np.sqrt(u) / (1 - np.sqrt(u))
    phi = 2 * np.pi * np.random.random(N)            # 方位角
    theta = np.arccos(2 * np.random.random(N) - 1)   # 极角（余弦均匀分布）
    pos = np.zeros((N, 3), dtype=np.float64)         # 【修复】：显式指定float64
    pos[:, 0] = r * np.sin(theta) * np.cos(phi)
    pos[:, 1] = r * np.sin(theta) * np.sin(phi)
    pos[:, 2] = r * np.cos(theta)
    return pos, np.full(N, M_tot / N, dtype=np.float64)  # 等质量粒子


def generate_satellite(pos_init, vel_init, M_dm, M_gas, M_star, N_dict, a_dm, R_gas, Z_val, base_id):
    """
    生成一个卫星星系（包含暗物质晕、恒星、气体），并赋予整体位置和速度。
    参数：
        pos_init : (3,) 卫星星系中心在盒子中的初始位置 [kpc]
        vel_init : (3,) 卫星星系质心的初始速度 [km/s]
        M_dm, M_gas, M_star : 各组分总质量 [1e10 Msun]
        N_dict   : 字典，包含各组分粒子数，如 {'dm':1000, 'gas':500, 'star':300}
        a_dm     : 暗物质晕的 Hernquist 标长 [kpc]
        R_gas    : 气体盘的径向标长 [kpc]（也用于气体Hernquist分布）
        Z_val    : 卫星星系的整体金属丰度（质量分数）
        base_id  : 该卫星星系的起始粒子ID（用于保证全局ID唯一）
    返回：
        一个字典，键为GIZMO粒子类型字符串 '0','1','4'，值为包含对应粒子属性的字典。
    """
    # 1. 生成各组分相对于卫星质心的位置和质量
    pos_dm, mass_dm = sample_hernquist(N_dict['dm'], a_dm, M_dm)
    pos_star, mass_star = sample_hernquist(N_dict['star'], a_dm * 0.2, M_star)  # 恒星分布更集中
    pos_gas, mass_gas = sample_hernquist(N_dict['gas'], R_gas, M_gas)
    vel_gas = np.zeros((N_dict['gas'], 3), dtype=np.float64)

    # 2. 将相对位置平移到绝对位置（盒子内）
    pos_dm += pos_init
    pos_gas += pos_init
    pos_star += pos_init

    # 3. 速度：整体运动 + 随机弥散（各向同性高斯）
    vel_dm = vel_init + np.random.normal(0, 20.0, (N_dict['dm'], 3)).astype(np.float64)
    vel_star = vel_init + np.random.normal(0, 15.0, (N_dict['star'], 3)).astype(np.float64)
    vel_gas += vel_init + np.random.normal(0, 25.0, (N_dict['gas'], 3)).astype(np.float64)

    # 4. 金属丰度数组（11维，GIZMO格式）
    z_gas_11d = np.zeros((N_dict['gas'], 11), dtype=np.float64)
    z_gas_11d[:, 0] = Z_val      # 第0维：总金属量
    z_gas_11d[:, 1] = 0.24       # 第1维：氦丰度（固定值）
    z_gas_11d[:, 2:] = Z_val * 0.1  # 其余元素按比例缩放

    z_star_11d = np.zeros((N_dict['star'], 11), dtype=np.float64)
    z_star_11d[:, 0] = Z_val
    z_star_11d[:, 1] = 0.24
    z_star_11d[:, 2:] = Z_val * 0.1

    # 5. 构造返回字典，按粒子类型组织
    return {
        '0': {  # 气体 (PartType0)
            'pos': pos_gas,
            'vel': vel_gas,
            'mass': mass_gas,
            'id': np.arange(base_id, base_id + N_dict['gas'], dtype=np.uint64),
            'u': np.full(N_dict['gas'], u_warm_gas, dtype=np.float64),  # 比内能
            'z': z_gas_11d
        },
        '1': {  # 暗物质 (PartType1)
            'pos': pos_dm,
            'vel': vel_dm,
            'mass': mass_dm,
            'id': np.arange(base_id + N_dict['gas'],
                            base_id + N_dict['gas'] + N_dict['dm'], dtype=np.uint64)
        },
        '4': {  # 恒星 (PartType4)
            'pos': pos_star,
            'vel': vel_star,
            'mass': mass_star,
            'id': np.arange(base_id + N_dict['gas'] + N_dict['dm'],
                            base_id + sum(N_dict.values()), dtype=np.uint64),
            'z': z_star_11d,
            'age': np.random.uniform(-10.0, -2.0, N_dict['star']).astype(np.float64)  # 恒星形成时间（负值表示年龄）
        }
    }


# ==================== 3. 主程序 ====================
def main():
    print(f">>> 正在读取银河系快照 (继承时间轴): {input_snapshot} ...")
    # 初始化存储银河系数据的字典
    mw_data = {'0': {}, '1': {}, '4': {}, '5': {}}
    original_header_attrs = {}

    # 读取输入快照
    with h5py.File(input_snapshot, 'r') as f:
        time_snapshot = f['Header'].attrs['Time']  # 读取当前模拟时间，如0.5 Gyr
        print(f"    检测到当前系统时间为: {time_snapshot:.4f} Gyr")

        # 备份原始Header所有属性，便于后续继承
        for k, v in f['Header'].attrs.items():
            original_header_attrs[k] = v

        # 读取每种粒子类型的所有数据集
        for ptype in ['0', '1', '4', '5']:
            if f'PartType{ptype}' in f:
                grp = f[f'PartType{ptype}']
                for key in grp.keys():
                    mw_data[ptype][key] = grp[key][:]

                # 【防错乱修复】：原封不动保留年龄，不做任何减法！
                # 如果快照中恒星数据缺少 StellarFormationTime，则补充一个随机的古老年龄
                if ptype == '4':
                    if 'StellarFormationTime' not in grp:
                        mw_data['4']['StellarFormationTime'] = np.random.uniform(-10, -2,
                                                                                 len(grp['Masses'][:])).astype(
                            np.float64)

    print(">>> 正在进行坐标系重置...")
    # 利用银河系核球区恒星估计星系中心位置和质心速度
    # 先粗略筛选出ID较小的恒星（通常核球恒星ID较小）
    core_mask = mw_data['4']['ParticleIDs'] < 1e9
    # 用这些恒星的质量加权平均位置作为中心初猜
    com_guess = np.average(mw_data['4']['Coordinates'][core_mask], axis=0,
                           weights=mw_data['4']['Masses'][core_mask])
    # 计算到初猜中心的距离，进一步筛选出半径10 kpc内的核球恒星
    r_core = np.sqrt(np.sum((mw_data['4']['Coordinates'][core_mask] - com_guess) ** 2, axis=1))
    strict_core = core_mask & (r_core < 10.0)

    # 如果有足够多的核球恒星，就用它们精确计算中心位置和速度
    if np.sum(strict_core) > 100:
        com = np.average(mw_data['4']['Coordinates'][strict_core], axis=0,
                         weights=mw_data['4']['Masses'][strict_core])
        v_com = np.average(mw_data['4']['Velocities'][strict_core], axis=0,
                           weights=mw_data['4']['Masses'][strict_core])
    else:
        com = com_guess
        v_com = np.zeros(3, dtype=np.float64)

    # 将所有粒子（包括暗物质、气体、恒星、黑洞）平移到盒子中心，并减去整体运动速度
    for ptype in ['0', '1', '4', '5']:
        if 'Coordinates' in mw_data[ptype]:
            mw_data[ptype]['Coordinates'] = mw_data[ptype]['Coordinates'] - com + box_center
            mw_data[ptype]['Velocities'] = mw_data[ptype]['Velocities'] - v_com

    print(">>> 正在生成 LMC & SMC ...")
    # 生成大麦哲伦云
    lmc = generate_satellite(
        box_center + pos_LMC_offset,   # 绝对位置
        vel_LMC,                       # 整体速度
        15.0,   # 暗物质晕质量 [1e10 Msun]
        0.15,   # 气体质量
        0.27,   # 恒星质量
        {'dm': 15000, 'gas': 15000, 'star': 2700},  # 各组分粒子数
        8.0,    # 暗物质晕标长 a_dm [kpc]
        6.0,    # 气体盘标长 R_gas [kpc]
        0.006,  # 金属丰度 Z (约0.3 Z_sun)
        1000000000  # 起始粒子ID（10亿）
    )

    # 生成小麦哲伦云
    smc = generate_satellite(
        box_center + pos_SMC_offset,
        vel_SMC,
        1.5,    # 暗物质质量
        0.10,   # 气体质量
        0.03,   # 恒星质量
        {'dm': 1500, 'gas': 10000, 'star': 300},
        3.0,    # a_dm
        3.0,    # R_gas
        0.002,  # 金属丰度更低 (约0.1 Z_sun)
        2000000000  # 起始ID（20亿）
    )

    print(">>> 正在合并并写入...")
    with h5py.File(output_ic, 'w') as f:
        # 写入Header，继承原始属性
        header = f.create_group("Header")
        for k, v in original_header_attrs.items():
            header.attrs[k] = v

        # 【防错乱修复】：继承真实的模拟时间（如0.5 Gyr），并强制双精度标志
        header.attrs["Time"] = time_snapshot
        header.attrs["Flag_DoublePrecision"] = 1

        # 计算合并后的各类型粒子总数
        num_0 = len(mw_data['0']['Coordinates']) + len(lmc['0']['pos']) + len(smc['0']['pos'])
        num_1 = len(mw_data['1']['Coordinates']) + len(lmc['1']['pos']) + len(smc['1']['pos'])
        num_4 = len(mw_data['4']['Coordinates']) + len(lmc['4']['pos']) + len(smc['4']['pos'])
        num_5 = len(mw_data['5'].get('Coordinates', []))

        part_counts = np.array([num_0, num_1, 0, 0, num_4, num_5], dtype=np.uint32)
        header.attrs["NumPart_ThisFile"] = part_counts
        header.attrs["NumPart_Total"] = part_counts

        # 合并并写入气体、暗物质、恒星 (PartType0,1,4)
        for ptype in ['0', '1', '4']:
            grp = f.create_group(f"PartType{ptype}")
            # 垂直堆叠银河系数据和两个卫星星系的数据
            grp.create_dataset("Coordinates",
                               data=np.vstack([mw_data[ptype]['Coordinates'],
                                               lmc[ptype]['pos'],
                                               smc[ptype]['pos']]),
                               dtype=np.float64)
            grp.create_dataset("Velocities",
                               data=np.vstack([mw_data[ptype]['Velocities'],
                                               lmc[ptype]['vel'],
                                               smc[ptype]['vel']]),
                               dtype=np.float64)
            grp.create_dataset("ParticleIDs",
                               data=np.concatenate([mw_data[ptype]['ParticleIDs'],
                                                    lmc[ptype]['id'],
                                                    smc[ptype]['id']]),
                               dtype=np.uint64)
            grp.create_dataset("Masses",
                               data=np.concatenate([mw_data[ptype]['Masses'],
                                                    lmc[ptype]['mass'],
                                                    smc[ptype]['mass']]),
                               dtype=np.float64)

            # 气体额外数据集：内能和金属丰度
            if ptype == '0':
                grp.create_dataset("InternalEnergy",
                                   data=np.concatenate([mw_data['0']['InternalEnergy'],
                                                        lmc['0']['u'],
                                                        smc['0']['u']]),
                                   dtype=np.float64)
                grp.create_dataset("Metallicity",
                                   data=np.vstack([mw_data['0']['Metallicity'],
                                                   lmc['0']['z'],
                                                   smc['0']['z']]),
                                   dtype=np.float64)
            # 恒星额外数据集：金属丰度和形成时间
            elif ptype == '4':
                grp.create_dataset("Metallicity",
                                   data=np.vstack([mw_data['4']['Metallicity'],
                                                   lmc['4']['z'],
                                                   smc['4']['z']]),
                                   dtype=np.float64)
                grp.create_dataset("StellarFormationTime",
                                   data=np.concatenate([mw_data['4']['StellarFormationTime'],
                                                        lmc['4']['age'],
                                                        smc['4']['age']]),
                                   dtype=np.float64)

        # 写入黑洞 (PartType5)，原样继承（因为卫星星系中没有黑洞）
        if num_5 > 0:
            grp5 = f.create_group("PartType5")
            for key, val in mw_data['5'].items():
                if val.dtype.kind == 'f':   # 浮点数据强制转为float64
                    grp5.create_dataset(key, data=val, dtype=np.float64)
                else:                       # 整数（如ID）保持原类型
                    grp5.create_dataset(key, data=val)

    print(" 完美！时空连贯，双精度就绪！")


if __name__ == "__main__":
    main()