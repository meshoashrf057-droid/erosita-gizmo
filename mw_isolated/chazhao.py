#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 5-30 kpc 范围内 CGM 粒子的径向速度 (内流/外流)。
============================================================
本版本修复:
    1. 去掉 \\odot 转义错误
    2. 全部标签改为英文 (避免中文字体警告)
    3. 默认不按温度过滤, 先打印温度分布和质量预算
    4. 增加 "先诊断 CGM 在哪" 的功能

用法:
    python chazhao.py snapshot_007.hdf5
    python chazhao.py snapshot_007.hdf5 --temp_min 1e4 --temp_max 1e12
    python chazhao.py snapshot_007.hdf5 --plot
============================================================
"""

import argparse
import numpy as np
import h5py

# GIZMO 单位: 1e10 Msun, kpc, km/s
MSUN_PER_GYR = 0.9778e10   # (1e10 Msun) * km/s / kpc -> Msun/Gyr

# 内能 -> 温度 转换 (u 单位: (km/s)^2)
k_B = 1.380649e-16
m_p = 1.6726219e-24
mu = 0.61
gamma = 5.0 / 3.0
U_TO_T = (gamma - 1) * mu * m_p / k_B * 1e10   # u_code -> T [K]


def load_gas(fname):
    with h5py.File(fname, "r") as f:
        pos = f["PartType0/Coordinates"][:]
        vel = f["PartType0/Velocities"][:]
        m = f["PartType0/Masses"][:]

        if "Temperature" in f["PartType0"]:
            T = f["PartType0/Temperature"][:]
        else:
            u = f["PartType0/InternalEnergy"][:]
            T = u * U_TO_T

        if "PartType5" in f and len(f["PartType5/Coordinates"]) > 0:
            center = f["PartType5/Coordinates"][0]
        else:
            center = np.average(pos, axis=0, weights=m)

    return pos, vel, m, T, center


def print_temperature_budget(r, m, T, rmin, rmax):
    """打印 [rmin, rmax] 内的温度分布和质量预算。"""
    sel = (r >= rmin) & (r <= rmax)
    n_tot = sel.sum()
    M_tot = m[sel].sum() * 1e10

    print(f"\n===== 诊断: {rmin}-{rmax} kpc 内的温度/质量分布 =====")
    print(f"  总粒子数: {n_tot:,}")
    print(f"  总质量:   {M_tot:.3e} Msun")

    if n_tot == 0:
        print("  [警告] 该半径内没有任何气体粒子!")
        return

    T_r = T[sel]
    print(f"  温度分位: "
          f"1%={np.percentile(T_r, 1):.2e}  "
          f"10%={np.percentile(T_r, 10):.2e}  "
          f"50%={np.percentile(T_r, 50):.2e}  "
          f"90%={np.percentile(T_r, 90):.2e}  "
          f"99%={np.percentile(T_r, 99):.2e} K")

    print(f"\n  {'T range [K]':>22} {'N':>10} {'Mass [Msun]':>16} {'f_M':>8}")
    print("  " + "-" * 62)
    ranges = [(0, 1e4), (1e4, 1e5), (1e5, 3e5),
              (3e5, 1e6), (1e6, 1e7), (1e7, 1e12)]
    for lo, hi in ranges:
        mask = (T_r >= lo) & (T_r < hi)
        n_in = mask.sum()
        m_in = m[sel][mask].sum() * 1e10
        f_in = m_in / M_tot if M_tot > 0 else 0
        print(f"  [{lo:.0e}, {hi:.0e})".rjust(22)
              + f" {n_in:>10d} {m_in:>16.3e} {f_in:>8.3f}")


def print_radial_budget(r, m, T):
    """打印不同半径壳层内的质量预算 (全温度)。"""
    print(f"\n===== 诊断: 半径壳层质量预算 (全温度) =====")
    print(f"  {'R range [kpc]':>18} {'N':>10} {'Mass [Msun]':>16} "
          f"{'<T> [K]':>12}")
    print("  " + "-" * 62)
    edges = [0, 1, 3, 5, 10, 15, 20, 30, 50, 100, 200, 500]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (r >= lo) & (r < hi)
        n_in = mask.sum()
        if n_in == 0:
            continue
        m_in = m[mask].sum() * 1e10
        T_in = np.sum(m[mask] * T[mask]) / m[mask].sum()
        print(f"  [{lo:>4}, {hi:>4})".rjust(18)
              + f" {n_in:>10d} {m_in:>16.3e} {T_in:>12.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", help="GIZMO snapshot")
    p.add_argument("--rmin", type=float, default=5.0)
    p.add_argument("--rmax", type=float, default=30.0)
    p.add_argument("--temp_min", type=float, default=0.0,
                   help="温度下限 [K], 默认 0 (不过滤)")
    p.add_argument("--temp_max", type=float, default=1e12,
                   help="温度上限 [K], 默认 1e12 (不过滤)")
    p.add_argument("--nbins", type=int, default=25)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default="cgm_inflow.png")
    args = p.parse_args()

    # ---------- 读取 ----------
    print(f"Reading {args.snapshot} ...")
    pos, vel, m, T, center = load_gas(args.snapshot)
    print(f"  Gas particles: {len(pos):,}")
    print(f"  Center: {center}")

    # ---------- 相对坐标与径向速度 ----------
    rel = pos - center
    r = np.linalg.norm(rel, axis=1)
    r_safe = np.maximum(r, 1e-6)
    r_hat = rel / r_safe[:, None]
    v_r = np.sum(vel * r_hat, axis=1)

    # ---------- 先做诊断 (不做温度过滤) ----------
    print_temperature_budget(r, m, T, args.rmin, args.rmax)
    print_radial_budget(r, m, T)

    # ---------- 温度/半径筛选 ----------
    sel = (r >= args.rmin) & (r <= args.rmax)
    sel &= (T >= args.temp_min) & (T <= args.temp_max)

    n = sel.sum()
    print(f"\n===== 应用筛选 =====")
    print(f"  R in [{args.rmin}, {args.rmax}] kpc")
    print(f"  T in [{args.temp_min:.2e}, {args.temp_max:.2e}] K")
    print(f"  Selected: {n:,} / {len(pos):,}")

    if n == 0:
        print("\n[ERROR] No particles pass the selection.")
        print("        See temperature budget above to pick a sensible window.")
        return

    r_s = r[sel]
    m_s = m[sel]
    vr_s = v_r[sel]
    T_s = T[sel]
    M_tot = m_s.sum()

    # ---------- 全局统计 ----------
    vr_mean = np.sum(m_s * vr_s) / M_tot
    vr_std = np.sqrt(np.sum(m_s * (vr_s - vr_mean) ** 2) / M_tot)
    T_mean = np.sum(m_s * T_s) / M_tot

    m_in = m_s[vr_s < 0].sum()
    m_out = m_s[vr_s > 0].sum()
    f_in = m_in / M_tot

    print(f"\n===== Global statistics (mass-weighted) =====")
    print(f"  Total mass:        {M_tot*1e10:.3e} Msun")
    print(f"  <v_r>  = {vr_mean:+.2f} km/s   (+:outflow, -:inflow)")
    print(f"  sigma(v_r) = {vr_std:.2f} km/s")
    print(f"  <T>    = {T_mean:.3e} K")
    print(f"  Inflow  mass fraction: {f_in*100:.1f}%")
    print(f"  Outflow mass fraction: {(1-f_in)*100:.1f}%")

    # SNR
    if vr_std > 0:
        snr = abs(vr_mean) / (vr_std / np.sqrt(n))
        print(f"  |<v_r>| / (sigma/sqrt(N)) = {snr:.2f}")

    # ---------- 分径向 bin ----------
    bins = np.linspace(args.rmin, args.rmax, args.nbins + 1)
    rc = 0.5 * (bins[:-1] + bins[1:])
    dr = bins[1] - bins[0]
    idx = np.digitize(r_s, bins) - 1

    print(f"\n===== Radial bins =====")
    print(f"{'R':>8} {'N':>8} {'M[1e10]':>10} "
          f"{'<v_r>':>10} {'sig_vr':>10} {'f_in':>8} "
          f"{'<T>[K]':>11} {'Mdot_r[1e9/Gyr]':>18}")
    print("-" * 100)

    rows = []
    for i in range(args.nbins):
        mask = idx == i
        if mask.sum() == 0:
            print(f"{rc[i]:>8.2f} {0:>8d} {'---':>10}")
            rows.append((rc[i], 0, 0, 0, 0, 0, 0, 0))
            continue
        mb = m_s[mask].sum()
        vrb = np.sum(m_s[mask] * vr_s[mask]) / mb
        vrs = np.sqrt(np.sum(m_s[mask] * (vr_s[mask] - vrb) ** 2) / mb)
        fib = m_s[mask][vr_s[mask] < 0].sum() / mb
        Tb = np.sum(m_s[mask] * T_s[mask]) / mb
        Mdot = mb * vrb / dr * MSUN_PER_GYR

        rows.append((rc[i], mask.sum(), mb, vrb, vrs, fib, Tb, Mdot))
        print(f"{rc[i]:>8.2f} {mask.sum():>8d} {mb:>10.3f} "
              f"{vrb:>+10.2f} {vrs:>10.2f} {fib:>8.3f} "
              f"{Tb:>11.2e} {Mdot/1e9:>18.3e}")

    # ---------- 判据 ----------
    print(f"\n===== Interpretation =====")
    if vr_mean < -30:
        print(f"  <v_r> = {vr_mean:.1f} km/s -> significant inflow")
    elif vr_mean < -10:
        print(f"  <v_r> = {vr_mean:.1f} km/s -> weak inflow")
    elif vr_mean < 10:
        print(f"  <v_r> = {vr_mean:.1f} km/s -> near balance")
    else:
        print(f"  <v_r> = {vr_mean:.1f} km/s -> outflow dominated")

    if f_in > 0.7:
        print(f"  Inflow fraction {f_in*100:.1f}% -> collapse")
    elif f_in > 0.5:
        print(f"  Inflow fraction {f_in*100:.1f}% -> weak collapse")
    else:
        print(f"  Inflow fraction {f_in*100:.1f}% -> turbulent/outflow")

    # ---------- 绘图 ----------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            arr = np.array(rows)
            valid = arr[:, 2] > 0

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            ax = axes[0, 0]
            ax.errorbar(arr[valid, 0], arr[valid, 3],
                        yerr=arr[valid, 4],
                        fmt="o-", capsize=3, color="C0")
            ax.axhline(0, color="k", ls="--", lw=0.8)
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel(r"$\langle v_r \rangle$ [km/s]")
            ax.set_title("Radial velocity profile")
            ax.grid(alpha=0.3)

            ax = axes[0, 1]
            ax.plot(arr[valid, 0], arr[valid, 5] * 100, "o-",
                    color="C3", label="inflow")
            ax.plot(arr[valid, 0], (1 - arr[valid, 5]) * 100, "s-",
                    color="C2", label="outflow")
            ax.axhline(50, color="k", ls="--", lw=0.8)
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel("Mass fraction [%]")
            ax.set_title("In/outflow mass fraction")
            ax.legend()
            ax.grid(alpha=0.3)

            ax = axes[1, 0]
            ax.plot(arr[valid, 0], arr[valid, 7] / 1e9, "o-", color="C4")
            ax.axhline(0, color="k", ls="--", lw=0.8)
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel(r"$\dot{M}_r$ [$10^9$ M$_\odot$/Gyr]")
            ax.set_title("Radial mass flux (negative = inflow)")
            ax.grid(alpha=0.3)

            ax = axes[1, 1]
            ax.plot(arr[valid, 0], arr[valid, 6] / 1e6, "o-", color="C5")
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel(r"$\langle T \rangle$ [$10^6$ K]")
            ax.set_title("Mean temperature")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.out, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved: {args.out}")
        except ImportError:
            print("\n[Warning] matplotlib not available, skipping plot.")
        except Exception as e:
            print(f"\n[Warning] plotting failed: {e}")


if __name__ == "__main__":
    main()