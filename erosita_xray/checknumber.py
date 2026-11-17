#!/usr/bin/env python3
"""
LMC 轨道 shooting - 2D 缩放迭代
每次跑完 1.5 Gyr 模拟后，从 disk_data_unified.csv 读出 achieved 值，
运行此脚本得到新的 v_init。
"""

import numpy as np

# ============ 当前初始条件（上一次运行的输入）============
v_init_current = np.array([-12.73, -183.98, -35.23])   # km/s
r_init_current = np.array([45.0, 350.0, -15.0])     # kpc

# ============ 当前终点结果（从 csv 读）============
dist_achieved = 51.844   # kpc
v_achieved = 392.313     # km/s

# ============ K13 目标 ============
dist_target = 49.66      # kpc
v_target = 321.0         # km/s

# ============ 缩放因子 ============
s_dist = dist_target / dist_achieved       # 0.876
s_vel  = v_target  / v_achieved            # 0.857

# 用几何平均（对非线性更稳健）
s = np.sqrt(s_dist * s_vel)                # ≈ 0.866

print(f"s_dist = {s_dist:.4f}")
print(f"s_vel  = {s_vel:.4f}")
print(f"s      = {s:.4f}")

# ============ 新初始速度 ============
v_init_new = v_init_current * s

print(f"\n当前 |v_init| = {np.linalg.norm(v_init_current):.1f} km/s")
print(f"新   |v_init| = {np.linalg.norm(v_init_new):.1f} km/s")
print(f"\n新 v_init = {v_init_new}")
print(f"\n替换 add_lmc_smc_v1.py 中的:")
print(f"  vel_LMC = np.array([{v_init_new[0]:.2f}, {v_init_new[1]:.2f}, {v_init_new[2]:.2f}])")