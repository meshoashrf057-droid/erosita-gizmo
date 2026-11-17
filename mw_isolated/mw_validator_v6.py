#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孤立 live MW 诊断 v6.0：yt 原生图 + 总引力圆速度 + 2×3 连图。

运行（不修改快照）：
  python mw_validator_v6.py --snapshot-dir ./output
  python mw_validator_v6.py --snapshot-dir ./output --start 6 --stop 6
  python mw_validator_v6.py --snapshot-dir ./output --gas-tmax 0

依赖：pip install 'yt>=4.4' numpy scipy h5py matplotlib pillow

物理定义：
  黑线 Vc(R) = sqrt(R*d<Phi>/dR)，在当前恒星盘的中面计算。
  所有 PartType0..5 的质量都参与引力；不以球对称 GM(<r)/r 代替盘引力。
  对每个粒子的方位角做解析平均（椭圆积分环核），保留其柱半径和高度。
  使用可控的 Plummer 平滑：默认各类型 eps=0.1 kpc。这是平滑重建势，
  不是直接读取 GIZMO 的力，也不等同于其 spline/自适应软化。
  在 R_sun 额外计算 eps/2 和 2*eps，报告敏感度；这不是完整分辨率收敛测试。
  分量的有符号 Vc^2 可相加；实际 gas/star vphi 不能相加。
  气体实际旋转：默认 |z'|<1 kpc、T<3e4 K；--gas-tmax 0 可取消温度筛选。
  恒星实际旋转：默认 PartType2/3/4 中 |z'|<1 kpc、R>=2 kpc 的样本。
  未知出生标签时不能彻底分离核球，故图例标为 midplane stars。
  可用 --disk-star-types 和 --star-id-min/--star-id-max 限制运动学/定向样本。
  ID 边界为闭区间；不影响总引力或全体恒星形态图。
  实际旋转线为粒子数加权中位数，阴影为 16--84 百分位（不是均值误差）。
  CSV 另存质量加权平均、速度弥散和径向/垂向平均速度。

中心：恒星收缩球中心，BH 仅作为初始种子；扣除中心 5 kpc 恒星整体速度。
盘面：由恒星角动量确定，运动学、引力和 yt 图使用同一盘面。
流体密度、温度、金属丰度全部由 yt.SlicePlot 原生 SPH/MFM 切片绘制。
恒星图为 yt.ParticleProjectionPlot；仅一维曲线使用 matplotlib。
PIL 仅等比例排版拼图，不重算、插值或拼接流体物理场。

单位：优先从快照 Header/Parameters/Units 读取基本单位；缺失时明确打印
  默认 1 kpc、1 km/s、1e10 Msun。CLI 可覆盖。时间由 yt 转为 Gyr。
仅适用于非宇宙学、孤立快照；不假定粒子 ID<1e9 就是 MW。
支持 snapshot_NNN.hdf5 和 snapshot_NNN.0.hdf5 等分文件（由 yt 读取全部）。

输出：每帧 2×3 PNG、rotation.csv、diagnostics.json；跨帧 summary.csv。
参考点：McMillan 2011, arXiv:1102.4340，R0=8.29±0.16 kpc，Vc=239±5 km/s。
它是参考约束，不会用于缩放模拟速度或质量，也不是对所有半径的目标值。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import h5py
import numpy as np
from scipy.special import ellipkm1, ellipe
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yt

VERSION = 'mw-validator-v6.0 / 2026-10-07'
G = 4.30091727003628e-6  # kpc (km/s)^2 Msun^-1
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.98841586e33


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=VERSION)
    p.add_argument('--snapshot-dir', default='./output')
    p.add_argument('--output-dir', default=None, help='默认 SNAPSHOT_DIR/figures_mw_validator_v6')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=None, help='包含该快照编号')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--width-kpc', type=float, default=30.)
    p.add_argument('--stellar-depth-kpc', type=float, default=30.)
    p.add_argument('--buffer-size', type=int, default=800)
    p.add_argument('--dpi', type=int, default=140)
    p.add_argument('--keep-panels', action='store_true')
    p.add_argument('--center-mode', choices=['stars', 'bh'], default='stars')
    p.add_argument('--center-kpc', nargs=3, type=float)
    p.add_argument('--center-radius-kpc', type=float, default=1.)
    p.add_argument('--bulk-radius-kpc', type=float, default=5.)
    p.add_argument('--min-center-particles', type=int, default=128)
    p.add_argument('--fixed-z', action='store_true', help='固定原始 z 轴；默认随恒星盘定向')
    p.add_argument('--disk-star-types', nargs='+', type=int, default=[2, 3, 4])
    p.add_argument('--star-id-min', type=int)
    p.add_argument('--star-id-max', type=int)
    p.add_argument('--disk-height-kpc', type=float, default=1.)
    p.add_argument('--gas-tmax', type=float, default=3e4, help='旋转样本温度上限 K；0=不筛温度')
    p.add_argument('--kinematic-rmin-kpc', type=float, default=2.)
    p.add_argument('--rmin-kpc', type=float, default=.5)
    p.add_argument('--rmax-kpc', type=float, default=20.)
    p.add_argument('--n-bins', type=int, default=39)
    p.add_argument('--min-bin-particles', type=int, default=20)
    p.add_argument('--solar-radius-kpc', type=float, default=8.29)
    p.add_argument('--solar-velocity-kms', type=float, default=239.)
    p.add_argument('--solar-radius-error-kpc', type=float, default=.16)
    p.add_argument('--solar-velocity-error-kms', type=float, default=5.)
    p.add_argument('--force-softening-kpc', nargs=6, type=float, default=[.1]*6,
                   metavar=('GAS', 'DM', 'PT2', 'PT3', 'STARS', 'BH'),
                   help='诊断 Plummer 平滑长度，非自动读取 GIZMO softening')
    p.add_argument('--force-chunk-size', type=int, default=200000)
    p.add_argument('--unit-length-kpc', type=float, default=None)
    p.add_argument('--unit-velocity-kms', type=float, default=None)
    p.add_argument('--unit-mass-msun', type=float, default=None)
    p.add_argument('--zsun', type=float, default=.02)
    p.add_argument('--density-limits', nargs=2, type=float, default=[1e-28, 1e-21])
    p.add_argument('--temperature-limits', nargs=2, type=float, default=[1e1, 10**6.5])
    p.add_argument('--metallicity-limits', nargs=2, type=float, default=[1e-2, 1.])
    return p


def validate_args(p, a):
    for name in ['width_kpc', 'stellar_depth_kpc', 'center_radius_kpc', 'bulk_radius_kpc',
                 'disk_height_kpc', 'solar_radius_kpc', 'solar_velocity_kms', 'zsun']:
        if not np.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f'{name} 必须为有限正数')
    if not 0 < a.rmin_kpc < a.rmax_kpc or a.gas_tmax < 0 or a.kinematic_rmin_kpc < 0:
        p.error('半径范围或气体温度上限无效')
    if min(a.n_bins, a.stride, a.buffer_size, a.dpi, a.min_bin_particles,
           a.min_center_particles, a.force_chunk_size) < 1:
        p.error('粒子数、网格和步长必须为正整数')
    if any(t not in [2, 3, 4] for t in a.disk_star_types):
        p.error('--disk-star-types 仅支持恒星 PartType2/3/4')
    if any(not np.isfinite(e) or e <= 0 for e in a.force_softening_kpc):
        p.error('六个诊断平滑长度必须全部为有限正数')
    if a.solar_radius_error_kpc < 0 or a.solar_velocity_error_kms < 0:
        p.error('参考误差不能为负')
    if a.star_id_min is not None and a.star_id_max is not None and a.star_id_min > a.star_id_max:
        p.error('star-id-min 大于 star-id-max')
    for name in ['density_limits', 'temperature_limits', 'metallicity_limits']:
        lo, hi = getattr(a, name)
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            p.error(f'{name} 须输入正的线性物理值，且下限小于上限')
    for name in ['unit_length_kpc', 'unit_velocity_kms', 'unit_mass_msun']:
        val = getattr(a, name)
        if val is not None and (not np.isfinite(val) or val <= 0):
            p.error(f'{name} 无效')


def find_snapshots(directory):
    found = {}
    for path in Path(directory).glob('snapshot_*.hdf5'):
        m = re.fullmatch(r'snapshot_(\d+)(?:\.(\d+))?\.hdf5', path.name)
        if not m or m[2] not in (None, '0'):
            continue
        number = int(m[1])
        if number in found:
            raise ValueError(f'编号 {number} 同时存在单文件和分文件快照，请分目录处理')
        found[number] = path.resolve()
    return sorted(found.items())


def load_dataset(path, args):
    attrs = {}
    with h5py.File(path, 'r') as f:
        for name in ['Header', 'Parameters', 'Units']:
            if name in f:
                attrs.update(dict(f[name].attrs))
        if bool(attrs.get('ComovingIntegrationOn', 0)):
            raise ValueError('仅支持非宇宙学孤立模拟')
    keys = ['UnitLength_in_cm', 'UnitVelocity_in_cm_per_s', 'UnitMass_in_g']
    fallback = [KPC_CM, 1e5, 1e10*MSUN_G]
    overrides = [args.unit_length_kpc, args.unit_velocity_kms, args.unit_mass_msun]
    factors = [KPC_CM, 1e5, MSUN_G]
    units, source = {}, {}
    for key, default, value, factor in zip(keys, fallback, overrides, factors):
        units[key] = float(value*factor if value is not None else attrs.get(key, default))
        source[key] = 'CLI' if value is not None else ('snapshot' if key in attrs else 'explicit fallback')
        if not np.isfinite(units[key]) or units[key] <= 0:
            raise ValueError(f'无效单位 {key}={units[key]}')
    ds = yt.load(str(path), unit_base=units)
    if ds.cosmological_simulation:
        raise ValueError('yt 将此文件识别为宇宙学模拟；请核实 Header，未继续计算')
    # Gadget frontend 默认将孤立盒也标为 periodic；yt 4.4 的 public property 只读。
    ds._periodicity = (False, False, False)
    actual = dict(length_kpc=float(ds.quan(1, 'code_length').to_value('kpc')),
                  velocity_kms=float(ds.quan(1, 'code_velocity').to_value('km/s')),
                  mass_msun=float(ds.quan(1, 'code_mass').to_value('Msun')),
                  time_gyr=float(ds.quan(1, 'code_time').to_value('Gyr')))
    print('  基本单位:', actual, '; 来源:', source, flush=True)
    return ds, dict(cgs=units, source=source, actual=actual)


def available_field(ds, candidates):
    fields = set(ds.field_list) | set(ds.derived_field_list)
    return next((f for f in candidates if f in fields), None)


def read_particles(ds):
    """使用同一个 yt data object，保证温度、速度和 ID 的粒子顺序一致。"""
    ad = ds.all_data()
    particles = {}
    for i in range(6):
        t = f'PartType{i}'
        if not available_field(ds, [(t, 'particle_position_x')]):
            continue
        pos = np.column_stack([ad[t, 'particle_position_'+k].to_value('kpc') for k in 'xyz'])
        if len(pos) == 0:
            continue
        mass = np.asarray(ad[t, 'particle_mass'].to_value('Msun'), dtype=float)
        vel = np.column_stack([ad[t, 'particle_velocity_'+k].to_value('km/s') for k in 'xyz'])
        if not (np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(mass).all()):
            raise ValueError(f'{t} 存在非有限坐标/速度/质量')
        if np.any(mass < 0):
            raise ValueError(f'{t} 存在负质量')
        item = dict(pos=pos, mass=mass, vel=vel)
        ids_field = available_field(ds, [(t, 'particle_index'), (t, 'ParticleIDs')])
        item['ids'] = np.asarray(ad[ids_field]) if ids_field else None
        if i == 0:
            tf = available_field(ds, [(t, 'temperature'), ('gas', 'temperature')])
            if tf is None:
                raise ValueError('yt 无法提供气体温度；需要 InternalEnergy 等完整热力学字段')
            item['temp'] = np.asarray(ad[tf].to_value('K'))
            if item['temp'].shape != mass.shape:
                raise ValueError('气体温度与 PartType0 粒子顺序/数量不兼容')
        particles[i] = item
    ad.clear_data()
    return particles


def combine_stars(particles, types, id_min=None, id_max=None):
    pieces = []
    for i in sorted(set(types)):
        if i not in particles:
            continue
        d = particles[i]
        mask = d['mass'] > 0
        if id_min is not None or id_max is not None:
            if d['ids'] is None:
                raise ValueError('要求 ID 筛选，但快照缺少 ParticleIDs')
            if id_min is not None:
                mask &= d['ids'] >= id_min
            if id_max is not None:
                mask &= d['ids'] <= id_max
        if mask.any():
            pieces.append({k: d[k][mask] for k in ['pos', 'vel', 'mass']})
    if not pieces:
        raise ValueError('找不到符合选择的恒星样本；请检查 --disk-star-types 和 ID 范围')
    return {k: np.concatenate([d[k] for d in pieces]) for k in ['pos', 'vel', 'mass']}


def find_frame(particles, args):
    stars = combine_stars(particles, [2, 3, 4])
    pos, mass = stars['pos'], stars['mass']
    bh = particles.get(5)
    bh_seed = bh['pos'][np.argmax(bh['mass'])] if bh is not None else None
    center = np.average(pos, axis=0, weights=mass)
    if bh_seed is not None:
        center = bh_seed.copy()
    used_radius = None
    if args.center_kpc is not None:
        center = np.array(args.center_kpc, dtype=float)
        method = 'manual position'
    elif args.center_mode == 'bh':
        if bh_seed is None:
            raise ValueError('--center-mode bh 但没有 PartType5')
        method = 'most massive PartType5 position'
    else:
        method = 'shrinking stellar sphere (all stellar types)'
        radius = max(30., args.center_radius_kpc)
        for _ in range(100):
            m = np.linalg.norm(pos-center, axis=1) < radius
            if m.sum() < args.min_center_particles:
                if used_radius is None:
                    raise ValueError('中心附近恒星不足，无法可靠定心；可指定 --center-kpc')
                break
            center = np.average(pos[m], axis=0, weights=mass[m])
            used_radius = radius
            if radius <= args.center_radius_kpc*(1+1e-9):
                break
            radius = max(args.center_radius_kpc, radius*.8)
    core = np.linalg.norm(pos-center, axis=1) < args.bulk_radius_kpc
    if core.sum() < args.min_center_particles:
        raise ValueError('整体速度估计样本不足；请检查中心或增加 --bulk-radius-kpc')
    bulk = np.average(stars['vel'][core], axis=0, weights=mass[core])
    tracer = combine_stars(particles, args.disk_star_types, args.star_id_min, args.star_id_max)
    x, v, w = tracer['pos']-center, tracer['vel']-bulk, tracer['mass']
    normal = np.array([0., 0., 1.])
    if not args.fixed_z:
        m = (np.linalg.norm(x, axis=1) > 2.) & (np.linalg.norm(x, axis=1) < 15.)
        for _ in range(3):
            if m.sum() < args.min_bin_particles:
                raise ValueError('2--15 kpc 定向恒星不足，可用 --fixed-z 明确固定盘面')
            ang = np.sum(w[m, None]*np.cross(x[m], v[m]), axis=0)
            scale = np.sum(w[m]*np.linalg.norm(x[m], axis=1)*np.linalg.norm(v[m], axis=1))
            if np.linalg.norm(ang) < .01*max(scale, 1e-30):
                raise ValueError('恒星净角动量太弱，无法确定盘面；请检查样本或用 --fixed-z')
            normal = ang/np.linalg.norm(ang)
            z = x@normal
            r = np.sqrt(np.maximum(np.sum(x*x, axis=1)-z*z, 0))
            m = (r > 2.) & (r < 15.) & (np.abs(z) < 3.)
    anchor = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    ex = anchor - (anchor@normal)*normal
    ex /= np.linalg.norm(ex)
    ey = np.cross(normal, ex)
    basis = np.column_stack([ex, ey, normal])
    info = dict(center_kpc=center, bulk_velocity_kms=bulk, basis_columns=basis,
                disk_normal=normal, tilt_deg=float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))),
                center_method=method, center_final_radius_kpc=used_radius,
                bulk_star_count=int(core.sum()))
    return center, bulk, basis, tracer, info


def ring_vc2(radii, particle_r, particle_z, mass, epsilon, chunk=200000):
    """解析方位平均的 Plummer 粒子势，返回有符号 R*d<Phi>/dR。

    D=(R+a)^2+z^2+eps^2, q=(R-a)^2+z^2+eps^2, m=1-q/D.
    v2=G*M/[pi*sqrt(D)]*[K(m)+(R^2-a^2-z^2-eps^2)/q*E(m)].
    中心外的环可产生向外径向力，故单项或分量 v2 不保证为正。
    """
    out = np.zeros(len(radii), dtype=float)
    for start in range(0, len(mass), chunk):
        sl = slice(start, start+chunk)
        a = particle_r[sl]
        h2 = particle_z[sl]**2 + epsilon**2
        w = mass[sl]
        for j, R in enumerate(radii):
            D = (R+a)**2 + h2
            q = (R-a)**2 + h2
            p = np.clip(q/D, np.finfo(float).tiny, 1.)
            kernel = ellipkm1(p) + (R*R-a*a-h2)/q * ellipe(1-p)
            out[j] += G/np.pi*np.sum(w*kernel/np.sqrt(D))
    return out


def gravity_profile(particles, center, basis, radii, args):
    per_type = {}
    sensitivity = np.zeros(3)
    for i, d in particles.items():
        x = (d['pos']-center)@basis
        r, z = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
        eps = args.force_softening_kpc[i]
        print(f'  引力重建 PartType{i}: N={len(r):,}, eps={eps:g} kpc', flush=True)
        per_type[i] = ring_vc2(radii, r, z, d['mass'], eps, args.force_chunk_size)
        for j, factor in enumerate([.5, 1., 2.]):
            sensitivity[j] += ring_vc2([args.solar_radius_kpc], r, z, d['mass'], eps*factor,
                                      args.force_chunk_size)[0]
    zeros = np.zeros(len(radii))
    groups = dict(total=sum(per_type.values(), zeros.copy()), gas=per_type.get(0, zeros),
                  dm=per_type.get(1, zeros),
                  stars=sum([per_type.get(i, zeros) for i in [2, 3, 4]], zeros.copy()),
                  bh=per_type.get(5, zeros))
    return groups, per_type, sensitivity


def positive_sqrt(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.where(x >= 0, x, np.nan))


def cylindrical_kinematics(d, center, bulk, basis):
    x = (d['pos']-center)@basis
    v = (d['vel']-bulk)@basis
    R = np.hypot(x[:, 0], x[:, 1])
    den = np.where(R > 1e-10, R, np.nan)
    return dict(R=R, z=x[:, 2], mass=d['mass'],
                vphi=(x[:, 0]*v[:, 1]-x[:, 1]*v[:, 0])/den,
                vr=(x[:, 0]*v[:, 0]+x[:, 1]*v[:, 1])/den, vz=v[:, 2])


def summarize_sample(k, mask, minimum):
    mask = mask & np.isfinite(k['vphi']) & (k['mass'] > 0)
    out = dict(n=int(mask.sum()), mass_msun=float(k['mass'][mask].sum()))
    names = ['median_vphi_kms', 'p16_vphi_kms', 'p84_vphi_kms', 'mean_vphi_kms',
             'sigma_vphi_kms', 'mean_vr_kms', 'sigma_vr_kms', 'mean_vz_kms', 'sigma_vz_kms']
    out.update({name: float('nan') for name in names})
    if out['n'] < minimum:
        return out
    out.update(zip(names[:3], map(float, np.percentile(k['vphi'][mask], [50, 16, 84]))))
    w = k['mass'][mask]
    for key in ['vphi', 'vr', 'vz']:
        values = k[key][mask]
        mean = np.average(values, weights=w)
        out[f'mean_{key}_kms'] = float(mean)
        out[f'sigma_{key}_kms'] = float(np.sqrt(np.average((values-mean)**2, weights=w)))
    return out


def kinematic_profiles(particles, stars, center, bulk, basis, edges, args):
    profiles, solar = {}, {}
    for name, d in [('gas', particles.get(0)), ('stars', stars)]:
        if d is None:
            continue
        k = cylindrical_kinematics(d, center, bulk, basis)
        mask = (np.abs(k['z']) < args.disk_height_kpc) & (k['R'] >= args.kinematic_rmin_kpc)
        if name == 'gas' and args.gas_tmax > 0:
            mask &= np.isfinite(d['temp']) & (d['temp'] > 0) & (d['temp'] < args.gas_tmax)
        stats = [summarize_sample(k, mask & (k['R'] >= lo) & (k['R'] < hi), args.min_bin_particles)
                 for lo, hi in zip(edges[:-1], edges[1:])]
        profiles[name] = {key: np.array([s[key] for s in stats]) for key in stats[0]}
        half = (edges[1]-edges[0])/2
        solar[name] = summarize_sample(k, mask & (abs(k['R']-args.solar_radius_kpc) < half),
                                       args.min_bin_particles)
        solar[name]['annulus_halfwidth_kpc'] = float(half)
    return profiles, solar


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_lg_yt(plot, field, label, path, args, limits=None):
    """只格式化 yt 的色条；数据仍是线性物理量，投影/切片完全由 yt 完成。"""
    plot.set_log(field, True)
    if limits is not None:
        plot.set_zlim(field, *limits)
    plot.set_colorbar_label(field, label)
    plot.set_font_size(16)
    plot.set_buff_size(args.buffer_size)
    plot.set_figure_size(7)
    plot.render()
    cb = plot.plots[field].cb
    lo, hi = cb.mappable.norm.vmin, cb.mappable.norm.vmax
    if lo is not None and hi is not None and 0 < lo < hi:
        exponents = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(np.log10(lo), np.log10(hi))
        ticks = 10.**exponents
        ticks = ticks[(ticks >= lo*(1-1e-10)) & (ticks <= hi*(1+1e-10))]
        cb.set_ticks(ticks)
        cb.formatter = FuncFormatter(lambda value, _: f'{np.log10(value):g}' if value > 0 else '')
        cb.update_ticks()
        cb.ax.yaxis.set_minor_locator(NullLocator())
    plot.save(str(path), mpl_kwargs={'dpi': args.dpi})
    for q in plot.plots.values():
        plt.close(q.figure)


def stellar_dataset(stars, center, basis):
    x = (stars['pos']-center)@basis
    data = {('stars', 'particle_position_'+axis): x[:, j] for j, axis in enumerate('xyz')}
    data['stars', 'particle_mass'] = stars['mass']
    bound = max(1., np.max(np.abs(x))*1.01)
    return yt.load_particles(data, length_unit='kpc', mass_unit='Msun',
                             bbox=np.array([[-bound, bound]]*3), periodicity=(False, False, False))


def render_yt_panels(ds, particles, center, basis, paths, args):
    all_stars = combine_stars(particles, [2, 3, 4])
    sds = stellar_dataset(all_stars, center, basis)
    field = ('stars', 'particle_mass')
    for i, axis in enumerate([2, 0]):
        plot = yt.ParticleProjectionPlot(sds, axis, field, center=sds.arr([0, 0, 0], 'kpc'),
                                         width=(args.width_kpc, 'kpc'),
                                         depth=(args.stellar_depth_kpc, 'kpc'))
        plot.set_unit(field, 'Msun')
        plot.set_cmap(field, 'magma')
        plot.set_background_color(field, 'black')
        plot.set_axes_unit('kpc')
        plot.set_xlabel("x' (kpc)" if axis == 2 else "y' (kpc)")
        plot.set_ylabel("y' (kpc)" if axis == 2 else "z' (kpc)")
        plot.annotate_title(f'{i+1}. {"Face-on" if i == 0 else "Edge-on"} stellar mass')
        save_lg_yt(plot, field, r'$\lg(M_{\rm pixel}/M_\odot)$', paths[i], args)
    del sds
    if 0 not in particles:
        raise ValueError('快照没有气体，无法生成流体切片')
    mf = available_field(ds, [('PartType0', 'Metallicity_00'), ('PartType0', 'Metallicity'),
                             ('gas', 'metallicity')])
    if mf is None:
        raise ValueError('缺少金属丰度字段，不能生成金属图；未假造默认金属丰度')

    def metal_z(field, data):
        z = data[mf]
        if z.ndim == 2:
            z = z[:, 0]
        return z/args.zsun

    metal_field = ('gas', 'validator_metal_z')
    ds.add_field(metal_field, function=metal_z, sampling_type='particle', units='', force_override=True)
    configs = [
        (('gas', 'density'), 'g/cm**3', 'viridis', args.density_limits,
         r'$\lg[\rho/({\rm g\,cm^{-3}})]$', '3. Edge-on gas density'),
        (('gas', 'temperature'), 'K', 'turbo', args.temperature_limits,
         r'$\lg(T/{\rm K})$', '4. Edge-on gas temperature'),
        (metal_field, '', 'YlOrRd_r', args.metallicity_limits,
         r'$\lg(Z/Z_\odot)$', '5. Edge-on gas metallicity')]
    for i, (field, unit, cmap, limits, label, title) in enumerate(configs, 2):
        # yt 原生 off-axis SPH/MFM 切片；normal 是盘内 x'，north 是盘法向 z'。
        plot = yt.SlicePlot(ds, basis[:, 0], field, center=ds.arr(center, 'kpc'),
                            north_vector=basis[:, 2], width=(args.width_kpc, 'kpc'))
        plot.set_unit(field, unit)
        plot.set_axes_unit('kpc')
        plot.set_xlabel("y' (kpc)")
        plot.set_ylabel("z' (kpc)")
        plot.set_cmap(field, cmap)
        plot.set_background_color(field, 'black')
        plot.annotate_title(title)
        save_lg_yt(plot, field, label, paths[i], args, limits)


def rotation_plot(radii, gravity, profiles, solar_vc, path, args):
    fig, ax = plt.subplots(figsize=(8.4, 7.0), dpi=args.dpi)
    ax.plot(radii, positive_sqrt(gravity['total']), c='black', lw=2.8,
            label=r'Total $V_c$: reconstructed gravity')
    for name, color, label in [('dm', '#6f4ba0', 'DM gravity'), ('stars', '#99731b', 'All-star gravity'),
                                ('gas', '#268a73', 'Gas gravity')]:
        ax.plot(radii, positive_sqrt(gravity[name]), ':', c=color, lw=1.7, label=label)
    for name, color, label in [('gas', '#1976d2', r'Gas median $v_\phi$'),
                                ('stars', '#df6a16', r'Midplane stars median $v_\phi$')]:
        if name not in profiles:
            continue
        d = profiles[name]
        ax.fill_between(radii, d['p16_vphi_kms'], d['p84_vphi_kms'], color=color, alpha=.13)
        ax.plot(radii, d['median_vphi_kms'], '--', c=color, lw=2, label=label)
    ax.errorbar(args.solar_radius_kpc, args.solar_velocity_kms,
                xerr=args.solar_radius_error_kpc, yerr=args.solar_velocity_error_kms,
                fmt='*', ms=12, color='#c62828', capsize=3, zorder=10, label='McMillan 2011 reference')
    ax.axvline(args.solar_radius_kpc, c='gray', ls=':', lw=.8)
    ax.axhline(0, c='gray', lw=.6)
    ax.set(xlim=(0, args.rmax_kpc), xlabel="Disk radius R' (kpc)", ylabel='Velocity (km/s)')
    ax.set_ylim(bottom=min(-25., ax.get_ylim()[0]), top=max(300., ax.get_ylim()[1]))
    ax.set_title('6. Circular speed and tracer rotation', fontsize=15)
    ax.text(.02, .98, f'Reconstructed Vc({args.solar_radius_kpc:g} kpc) = {solar_vc:.1f} km/s',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', fontsize=9, framealpha=.9)
    temp_text = f'T < {args.gas_tmax:g} K' if args.gas_tmax > 0 else 'all gas temperatures'
    notes = (f'Tracers: |z\'| < {args.disk_height_kpc:g} kpc; R >= {args.kinematic_rmin_kpc:g} kpc; {temp_text}\n'
             'Shading: particle 16-84% range. Gravity: azimuth-averaged Plummer reconstruction.')
    fig.text(.5, .025, notes, ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .085, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def stitch(paths, destination, title, subtitle):
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert('RGB'))
    w, h = max(im.width for im in images), max(im.height for im in images)
    header = 110
    canvas = Image.new('RGB', (3*w, 2*h+header), 'white')
    for j, im in enumerate(images):
        # 等比例缩放 + 留白；不把原来的宽高比强制拉伸。
        tile = ImageOps.contain(im, (w, h), Image.Resampling.LANCZOS)
        canvas.paste(tile, ((j % 3)*w+(w-tile.width)//2, header+(j//3)*h+(h-tile.height)//2))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 30)
        small = ImageFont.truetype('DejaVuSans.ttf', 21)
    except OSError:
        font = ImageFont.load_default(size=30)
        small = ImageFont.load_default(size=21)
    draw.text((24, 15), title, fill='black', font=font)
    draw.text((24, 62), subtitle, fill='#333333', font=small)
    canvas.save(destination)


def process_snapshot(number, path, args, output):
    print(f'\n>> snapshot_{number:03d}: {path}', flush=True)
    ds, units = load_dataset(path, args)
    particles = read_particles(ds)
    center, bulk, basis, stars, frame = find_frame(particles, args)
    print(f'  中心={center}; 整体速度={bulk} km/s; 盘倾角={frame["tilt_deg"]:.3f} deg', flush=True)
    edges = np.linspace(args.rmin_kpc, args.rmax_kpc, args.n_bins+1)
    radii = (edges[:-1]+edges[1:])/2
    gravity, per_type, solar_v2 = gravity_profile(particles, center, basis, radii, args)
    profiles, solar_kin = kinematic_profiles(particles, stars, center, bulk, basis, edges, args)
    solar_vc = positive_sqrt(solar_v2)
    sensitivity_fraction = (float(np.max(np.abs(solar_vc[[0, 2]]/solar_vc[1]-1)))
                            if np.isfinite(solar_vc).all() and solar_vc[1] > 0 else float('nan'))
    print(f'  Vc(R={args.solar_radius_kpc:g} kpc): {solar_vc[1]:.3f} km/s; '
          f'eps/2, eps, 2eps = {solar_vc} km/s', flush=True)
    if np.isfinite(sensitivity_fraction) and sensitivity_fraction > .05:
        print('  注意：太阳半径圆速度对诊断平滑长度的敏感度超过 5%，需检查分辨率/平滑尺度。', flush=True)
    if np.any(gravity['total'] < 0):
        print('  注意：部分半径的平均径向引力向外，无实数圆速度；对应曲线留空，CSV 保留有符号 Vc^2。', flush=True)
    for name, s in solar_kin.items():
        print(f'  {name}: N={s["n"]}, median vphi={s["median_vphi_kms"]:.3f}, '
              f'mass-weighted mean={s["mean_vphi_kms"]:.3f} km/s', flush=True)
    stem = f'mw_validator_snap{number:03d}'
    rows = []
    for j, R in enumerate(radii):
        row = dict(radius_kpc=R)
        for name, values in gravity.items():
            row[f'{name}_vc2_kms2'] = values[j]
            row[f'{name}_vc_kms'] = positive_sqrt([values[j]])[0]
        for t, values in per_type.items():
            row[f'PartType{t}_vc2_kms2'] = values[j]
        for name, data in profiles.items():
            row.update({f'{name}_{k}': v[j] for k, v in data.items()})
        rows.append(row)
    write_csv(output/f'{stem}_rotation.csv', rows)
    time_gyr = float(ds.current_time.to_value('Gyr'))
    metadata = dict(version=VERSION, snapshot=str(path), time_gyr=time_gyr, units=units,
                    frame=frame, settings=vars(args),
                    counts={i: len(d['mass']) for i, d in particles.items()},
                    masses_msun={i: float(d['mass'].sum()) for i, d in particles.items()},
                    gravity_method='azimuth-averaged particle potential with diagnostic Plummer smoothing',
                    external_analytic_potentials_included=False,
                    gravity_component_rule='add signed vc2, not tracer velocities',
                    negative_total_vc2_bins=int(np.sum(gravity['total'] < 0)),
                    negative_component_vc2_bins={k: int(np.sum(v < 0)) for k, v in gravity.items()},
                    solar_radius_kpc=args.solar_radius_kpc,
                    solar_vc_softening_half_nominal_double_kms=solar_vc,
                    solar_softening_max_relative_difference=sensitivity_fraction,
                    solar_kinematics=solar_kin,
                    limitations=['No external analytic potential is reconstructed.',
                                 'Plummer reconstruction is not the exact GIZMO force kernel.',
                                 'Spatially selected stars can retain bulge contamination.',
                                 'Percentile bands are tracer scatter, not uncertainty in Vc.',
                                 'Negative vc2 means outward averaged force; curve masked, signed CSV retained.'])
    metadata_path = output/f'{stem}_diagnostics.json'
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(json_safe(metadata), f, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        with tempfile.TemporaryDirectory(prefix=f'{stem}_', dir=output) as tmp:
            panel_paths = [Path(tmp)/f'panel{i}.png' for i in range(1, 7)]
            render_yt_panels(ds, particles, center, basis, panel_paths, args)
            rotation_plot(radii, gravity, profiles, solar_vc[1], panel_paths[5], args)
            title = f'Isolated MW | snapshot {number:03d} | t = {time_gyr:.4f} Gyr | v6.0'
            subtitle = (f'Disk-aligned frame | tilt = {frame["tilt_deg"]:.2f} deg | '
                        f'Vc({args.solar_radius_kpc:g} kpc) = {solar_vc[1]:.2f} km/s '
                        f'| reference = {args.solar_velocity_kms:g} km/s | fluid maps: yt slices')
            stitch(panel_paths, output/f'{stem}.png', title, subtitle)
            if args.keep_panels:
                for j, source in enumerate(panel_paths, 1):
                    (output/f'{stem}_panel{j}.png').write_bytes(source.read_bytes())
    except Exception as exc:
        metadata['render_error'] = str(exc)
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(json_safe(metadata), f, ensure_ascii=False, indent=2, allow_nan=False)
        raise
    summary = dict(snapshot=number, time_gyr=time_gyr, vc_solar_kms=solar_vc[1],
                   vc_solar_eps_half_kms=solar_vc[0], vc_solar_eps_double_kms=solar_vc[2],
                   solar_softening_max_relative_difference=sensitivity_fraction,
                   vc_minus_reference_kms=solar_vc[1]-args.solar_velocity_kms,
                   disk_tilt_deg=frame['tilt_deg'])
    for name, s in solar_kin.items():
        summary.update({f'{name}_{k}': v for k, v in s.items()})
    print(f'  连图: {output/stem}.png', flush=True)
    del particles, stars, ds
    gc.collect()
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    print(VERSION, flush=True)
    print('总 Vc 为全部快照粒子的轴对称重建值；不包含额外解析外势。', flush=True)
    directory = Path(args.snapshot_dir).expanduser().resolve()
    if not directory.is_dir():
        parser.error(f'快照目录不存在: {directory}')
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else directory/'figures_mw_validator_v6'
    output.mkdir(parents=True, exist_ok=True)
    files = [(n, p) for n, p in find_snapshots(directory)
             if n >= args.start and (args.stop is None or n <= args.stop)][::args.stride]
    if not files:
        parser.error('没有找到符合范围的 snapshot_NNN[.0].hdf5')
    yt.funcs.mylog.setLevel(40)
    summary_path = output/'mw_validator_summary.csv'
    existing = {}
    if summary_path.exists():
        with summary_path.open(encoding='utf-8', newline='') as f:
            existing = {int(r['snapshot']): r for r in csv.DictReader(f)}
    errors = []
    for number, path in files:
        try:
            existing[number] = process_snapshot(number, path, args, output)
            write_csv(summary_path, [existing[k] for k in sorted(existing)])
        except Exception as exc:
            errors.append((number, str(exc)))
            print(f'  失败 snapshot_{number:03d}: {type(exc).__name__}: {exc}', flush=True)
            plt.close('all')
            gc.collect()
    if errors:
        print('以下快照未完成:', errors, flush=True)
        raise SystemExit(1)
    print(f'完成 {len(files)} 帧；连图和数据目录: {output}', flush=True)


if __name__ == '__main__':
    main()
