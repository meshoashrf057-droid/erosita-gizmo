#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读检查 GIZMO 非宇宙学 MW / LMC / SMC 快照中的气体金属丰度。

快速检查：
  python check_snapshot_metallicity_v1.py --snapshot-dir ./allsky
完整核查（使用本地 APEC 表；不生成三维网格）：
  python check_snapshot_metallicity_v1.py --snapshot-dir ./allsky \
    --reference-ic mw_relaxed_lmc_smc_tg19_trial02.hdf5 \
    --xray --apec-data-dir .
只检查指定文件，也可以包括孤立 IC / 不同阶段的快照：
  python check_snapshot_metallicity_v1.py --snapshots 'allsky/snapshot_02*.hdf5'

必需依赖 numpy、h5py；yt 可选，--xray 时必需。不修改 HDF5、IC 或绘图缓存。
默认 Zsun=0.02；直接读取 Metallicity[:,0]（总金属质量分数）。
第 1 列是 He，不将所有列相加当作总金属丰度。
支持 N、N×1、N×11 Metallicity，以及分文件 snapshot_NNN.i.hdf5。
自动识别 BH 中心；无 BH 时显式提示使用 BoxSize/2，可用 --center-kpc 覆盖。
几何与当前 allsky 对齐：BH±30 kpc 立方体、Sun=(-8.29,0,0)、LOS=0..60 kpc、
|z|>=5 kpc。所有统计都按粒子中心选择，未做核重建和视线积分。
默认检查原始 HDF5，并在 yt 可用时逐粒子核对 Metallicity_00。
修订 v1.1：允许重复父 ParticleIDs；优先使用父ID+ChildID+Generation组合，
仍不唯一时使用相同ID组内的三维坐标核验匹配，不假定 HDF5/yt 的排列相同。
--xray 额外用 yt/APEC 计算 sum(epsilon*m/rho) 和光度加权丰度。
没有 APEC 表时报告该步骤失败，仍保存原始统计；不下载或代用冷却表。
来源分类只用 snapshot/reference IC 的 ID 元数据，不凭丰度或位置猜来源。
新生成/改 ID 的粒子可能为 unassigned；ID 来源标签不代表混合后的金属来源。
--reference-ic 可给多个文件：提取来源 ID，且逐份输出原始统计用于阶段对比。

输出：metallicity_summary.csv、metallicity_audit.json、metallicity_audit.txt。
均写入 --output-dir（默认 ./allsky/metallicity_audit）。
数值有效、yt 读取一致，不等同于物理模型或观测匹配通过。
日期：2026-10-08。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import h5py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gizmo_units import resolve_units, require_noncosmological
from gizmo_compat import load_gizmo

KPC_CM = 3.0856775814913673e21
MSUN_G = 1.98847e33
MP = 1.67262192369e-24
KB = 1.380649e-16
GYR_S = 31557600.e9
ELEMENTS = ['Z_total', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
ORIGIN_KEYS = {'MW_CGM': 'MW_CGM', 'MW_ISM': 'MW_ISM',
               'LMC_gas': 'LMC_GAS', 'SMC_gas': 'SMC_GAS'}


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def clean(x):
    """合法 JSON：无数据为 null，不输出 NaN/Infinity。"""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, bytes):
        return x.decode(errors='replace')
    return x


def snapshot_parts(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, 'r') as f:
        nfiles = int(f['Header'].attrs.get('NumFilesPerSnapshot', 1))
    match = re.match(r'^(.*)\.(\d+)\.(hdf5|h5)$', path.name)
    if nfiles == 1 and not match:
        return [path]
    if match:
        base, _, ext = match.groups()
    else:
        base, ext = path.stem, path.suffix[1:]
    files = [path.with_name(f'{base}.{i}.{ext}') for i in range(nfiles)]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError(f'分文件快照不完整：{files}')
    return files


def discover(args):
    found = []
    if args.snapshots:
        for token in args.snapshots:
            matches = glob.glob(token)
            if not matches:
                raise FileNotFoundError(f'没有匹配文件：{token}')
            found.extend(Path(p) for p in matches)
    else:
        pattern = re.compile(r'^(snapshot|snap)_\d+(\.\d+)?\.(hdf5|h5)$')
        found = [p for p in Path(args.snapshot_dir).rglob('*')
                 if p.is_file() and pattern.match(p.name)]
    unique = {}
    for p in found:
        parts = snapshot_parts(p)
        unique[str(parts[0])] = parts
    if not unique:
        raise FileNotFoundError(f'未找到快照：{args.snapshot_dir}')
    return [unique[k] for k in sorted(unique, key=natural_key)]


def metadata(path):
    out = {}
    with h5py.File(path, 'r') as f:
        for name in ['Header', 'Parameters', 'MWModelReference',
                     'MagellanicOrbitReference', 'ReservedParticleIDs']:
            if name in f:
                out[name] = {k: clean(v) for k, v in f[name].attrs.items()}
    return out


def unit_values(meta, args):
    require_noncosmological(meta)
    spec = [('length_kpc', 'UnitLength_in_cm', KPC_CM),
            ('mass_msun', 'UnitMass_in_g', MSUN_G),
            ('velocity_kms', 'UnitVelocity_in_cm_per_s', 1e5)]
    overrides = {key: getattr(args, 'unit_'+name)*factor for name, key, factor in spec
                 if getattr(args, 'unit_'+name) is not None}
    cgs, source = resolve_units(meta, overrides=overrides)
    return ({name: cgs[key]/factor for name, key, factor in spec},
            {name: source[key] for name, key, factor in spec})


def read_arrays(parts, units):
    batches = []
    schemas = []
    for path in parts:
        with h5py.File(path, 'r') as f:
            if 'PartType0' not in f:
                continue
            g = f['PartType0']
            n = len(g['Coordinates'])
            if not n:
                continue
            d = {k: np.asarray(g[k]) for k in ['Coordinates', 'ParticleIDs']}
            if 'Metallicity' in g:
                z = np.asarray(g['Metallicity'], dtype=float)
                schema = {'dataset': 'PartType0/Metallicity', 'shape': list(z.shape),
                          'attrs': {k: clean(v) for k, v in g['Metallicity'].attrs.items()}}
                if z.ndim == 1:
                    z = z[:, None]
            elif 'Metallicity_00' in g:
                fields = sorted([k for k in g if re.fullmatch(r'Metallicity_\d\d', k)])
                z = np.column_stack([np.asarray(g[k], float) for k in fields])
                if fields != [f'Metallicity_{i:02d}' for i in range(len(fields))]:
                    raise ValueError('Metallicity_XX 列号不连续，拒绝猜测元素顺序。')
                schema = {'dataset': fields, 'shape': list(z.shape), 'attrs': {}}
            else:
                raise ValueError(f'{path}: 没有 Metallicity / Metallicity_00')
            if z.ndim != 2 or z.shape[0] != n or z.shape[1] < 1:
                raise ValueError(f'Metallicity 形状错误：{z.shape}，气体 N={n}')
            d['metal'] = z
            if 'Masses' in g:
                d['Masses'] = np.asarray(g['Masses'], float)
            else:
                m = float(f['Header'].attrs.get('MassTable', np.zeros(6))[0])
                if m <= 0:
                    raise ValueError('气体没有 Masses，MassTable[0] 也不为正。')
                d['Masses'] = np.full(n, m)
            for k in ['Temperature', 'InternalEnergy', 'Density', 'ElectronAbundance',
                      'ParticleChildIDsNumber', 'ParticleIDGenerationNumber']:
                if k in g:
                    d[k] = np.asarray(g[k])
            if d['Coordinates'].shape != (n, 3) or d['ParticleIDs'].shape != (n,):
                raise ValueError('Coordinates 或 ParticleIDs 形状不匹配。')
            for k, a in d.items():
                if k not in ('metal', 'Coordinates') and a.shape != (n,):
                    raise ValueError(f'{k} 形状不是 N：{a.shape}')
            schemas.append({'file': str(path), **schema})
            batches.append(d)
    if not batches:
        raise ValueError('没有 PartType0 气体粒子。')
    if len({d['metal'].shape[1] for d in batches}) != 1:
        raise ValueError('分文件 Metallicity 列数不一致。')
    fields = set.intersection(*(set(d) for d in batches))
    raw = {k: np.concatenate([d[k] for d in batches], axis=0) for k in fields}
    if any(set(d) != fields for d in batches):
        raise ValueError('分文件字段不一致，无法安全联合统计。')
    if not np.issubdtype(raw['ParticleIDs'].dtype, np.integer):
        raise ValueError('原始 ParticleIDs 应为整数。')
    raw['ParticleIDs'] = raw['ParticleIDs'].astype(np.uint64)
    raw['position_kpc'] = raw['Coordinates'].astype(float) * units['length_kpc']
    raw['mass_msun'] = raw['Masses'].astype(float) * units['mass_msun']
    if np.any(~np.isfinite(raw['position_kpc'])):
        raise ValueError('Coordinates 包含 NaN/Inf。')
    if np.any(~np.isfinite(raw['mass_msun'])) or np.any(raw['mass_msun'] <= 0):
        raise ValueError('粒子质量不是有限正数。')
    return raw, schemas


def get_center(parts, meta, units, args, notes):
    if args.center_kpc is not None:
        return np.array(args.center_kpc), 'CLI --center-kpc'
    requested = args.bh_id
    if requested is None:
        for group in meta.values():
            if 'MW_BH_ID' in group:
                requested = int(group['MW_BH_ID'])
                break
    pos, ids = [], []
    for path in parts:
        with h5py.File(path, 'r') as f:
            if 'PartType5' in f:
                pos.extend(np.asarray(f['PartType5/Coordinates'], float))
                ids.extend(np.asarray(f['PartType5/ParticleIDs'], np.uint64))
    if pos:
        pos = np.array(pos) * units['length_kpc']
        if requested is not None:
            hit = np.flatnonzero(np.array(ids, dtype=np.uint64) == requested)
            if len(hit) != 1:
                raise ValueError(f'BH ID={requested} 匹配数={len(hit)}')
            return pos[hit[0]], f'BH ParticleID={requested}'
        if len(pos) == 1:
            return pos[0], f'only BH ParticleID={int(ids[0])}'
        raise ValueError('多个 BH；请指定 --bh-id 或 --center-kpc。')
    if args.bh_id is not None:
        raise ValueError('指定了 --bh-id，但快照没有 BH。')
    box = np.asarray(meta['Header'].get('BoxSize', 0.), float)
    if np.all(box > 0):
        notes.append('没有 BH：中心使用 BoxSize/2；若星系不在盒子中心，请指定 --center-kpc。')
        return np.broadcast_to(box / 2., (3,)) * units['length_kpc'], 'BoxSize/2 fallback'
    raise ValueError('没有 BH 或有效 BoxSize，请指定 --center-kpc。')


def origin_ranges(meta, refs, notes):
    result = {}
    # 当前快照的元数据优先；各 reference 只补充缺少的范围。
    for source, m in [('snapshot', meta)] + refs:
        for group in ['ReservedParticleIDs', 'MagellanicOrbitReference']:
            attrs = m.get(group, {})
            for label, prefix in ORIGIN_KEYS.items():
                if prefix + '_ID_START' not in attrs or prefix + '_ID_END' not in attrs:
                    continue
                pair = [int(attrs[prefix + '_ID_START']), int(attrs[prefix + '_ID_END'])]
                if pair[1] < pair[0]:
                    continue
                if label in result and result[label]['range'] != pair:
                    notes.append(f'{label} ID 范围冲突：保留 {result[label]["source"]}，忽略 {source}。')
                else:
                    result.setdefault(label, {'range': pair, 'source': source + '/' + group})
    pairs = list(result.items())
    for i, (a, x) in enumerate(pairs):
        for b, y in pairs[i+1:]:
            if max(x['range'][0], y['range'][0]) <= min(x['range'][1], y['range'][1]):
                raise ValueError(f'来源 ID 区间重叠：{a}, {b}')
    return result


def raw_temperature(raw, units):
    if 'Temperature' in raw:
        return raw['Temperature'].astype(float), 'stored Temperature, interpreted as K'
    if 'InternalEnergy' in raw and 'ElectronAbundance' in raw:
        z = raw['metal'][:, 0]
        he = raw['metal'][:, 1] if raw['metal'].shape[1] == 11 else np.full(len(z), .24)
        x = 1. - he - z
        denom = x * (1. + raw['ElectronAbundance']) + he / 4.
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (2./3.) * raw['InternalEnergy'] * (units['velocity_kms']*1.e5)**2 * MP / (KB*denom)
        t[(x <= 0) | (denom <= 0)] = np.nan
        return t, 'approx: U, gamma=5/3, X=1-Y-Z, ne/nH; neglect metal particle count; Y=0.24 if no 11 columns'
    return None, 'unavailable (no stored Temperature or U+ElectronAbundance); no guessed temperature'


def masks_for(raw, center, args, ranges):
    xyz = raw['position_kpc'] - center
    radius = np.linalg.norm(xyz, axis=1)
    distance = np.linalg.norm(xyz - np.array(args.sun_kpc), axis=1)
    cube = np.all(np.abs(xyz) <= args.cube_radius_kpc, axis=1)
    view = cube & (distance >= args.los_kpc[0]) & (distance <= args.los_kpc[1])
    off = view & (np.abs(xyz[:, 2]) >= args.height_kpc)
    masks = {'all_gas': np.ones(len(xyz), bool), 'sphere_r_lt_200': radius < 200.,
             'cube_los_unmasked': view, 'cube_los_slab': view & ~off,
             'cube_los_retained': off, 'cube_los_north': off & (xyz[:, 2] >= 0),
             'cube_los_south': off & (xyz[:, 2] < 0)}
    known = np.zeros(len(xyz), bool)
    for label, info in ranges.items():
        lo, hi = info['range']
        select = (raw['ParticleIDs'] >= lo) & (raw['ParticleIDs'] <= hi)
        known |= select
        masks['origin_' + label] = select
        masks['retained_origin_' + label] = select & off
    masks['origin_unassigned'] = ~known
    masks['retained_origin_unassigned'] = ~known & off
    return masks


def distribution(values):
    values = np.asarray(values)
    x = values[np.isfinite(values)]
    out = {'N': len(values), 'nonfinite': int(np.count_nonzero(~np.isfinite(values))),
           'negative': int(np.count_nonzero(values < 0)), 'zero': int(np.count_nonzero(values == 0))}
    for key, q in [('min', 0), ('p01', 1), ('p16', 16), ('median', 50), ('p84', 84), ('p99', 99), ('max', 100)]:
        out[key] = float(np.percentile(x, q)) if len(x) else None
    return out


def region_stats(name, select, raw, args, temp=None, luminosity=None):
    z = raw['metal'][select, 0]
    mass = raw['mass_msun'][select]
    valid = np.isfinite(z) & (z >= 0) & (z <= 1)
    total_mass = float(mass.sum())
    vmass = float(mass[valid].sum())
    dist = distribution(z)
    row = {'region': name, 'N': int(select.sum()), 'mass_msun': total_mass,
           'N_invalid_Z': int((~valid).sum()), 'valid_Z_mass_fraction': vmass/total_mass if total_mass else None}
    row['N_zero_Z'] = int(np.count_nonzero(z == 0))
    row['zero_Z_number_fraction'] = float(np.count_nonzero(z == 0)/len(z)) if len(z) else None
    row['zero_Z_mass_fraction'] = float(mass[z == 0].sum()/vmass) if vmass else None
    for k in ['min', 'p01', 'median', 'p99', 'max']:
        row['raw_Z_' + k] = dist[k]
        row['Zsun_' + k] = dist[k] / args.zsun if dist[k] is not None else None
    row['Zsun_mass_weighted'] = float(np.sum(z[valid]*mass[valid])/vmass/args.zsun) if vmass else None
    row['metal_mass_msun_valid'] = float(np.sum(z[valid]*mass[valid]))
    for threshold in [.001, .01, .1]:
        label = f'{threshold:g}'
        row['mass_fraction_Zsun_lt_' + label] = float(mass[valid & (z/args.zsun < threshold)].sum()/vmass) if vmass else None
    if temp is not None:
        t = temp[select]
        tv = np.isfinite(t) & (t > 0)
        row['temperature_mass_weighted_K'] = float(np.average(t[tv], weights=mass[tv])) if tv.any() else None
    if luminosity is not None:
        lum = luminosity[select]
        good = valid & np.isfinite(lum) & (lum >= 0)
        lsum = float(lum[good].sum())
        row['N_invalid_luminosity'] = int((~good).sum())
        row['L_X_erg_s_valid'] = lsum
        row['Zsun_luminosity_weighted'] = float(np.sum(z[good]/args.zsun*lum[good])/lsum) if lsum else None
        row['L_fraction_Zsun_lt_0.01'] = float(lum[good & (z/args.zsun < .01)].sum()/lsum) if lsum else None
        row['brightest_particle_L_fraction'] = float(lum[good].max()/lsum) if lsum else None
    return row


def integer_ids(values, name):
    a = np.asarray(values)
    if np.issubdtype(a.dtype, np.floating) and (np.any(a > 2**53) or np.any(a != np.floor(a))):
        raise ValueError(f'{name} 浮点化后无法保证精确匹配。')
    if np.any(~np.isfinite(a)) or np.any(a < 0) or a.ndim != 1:
        raise ValueError(f'{name} 不是有效非负整数向量。')
    return a.astype(np.uint64)


def duplicate_id_summary(values):
    ids, counts = np.unique(values, return_counts=True)
    duplicate = counts > 1
    return {'N_particles': len(values), 'N_unique_parent_IDs': len(ids),
            'N_repeated_parent_IDs': int(duplicate.sum()),
            'N_particles_in_repeated_ID_groups': int(counts[duplicate].sum()),
            'N_extra_rows_beyond_unique_parent_IDs': int((counts[duplicate]-1).sum()),
            'max_multiplicity': int(counts.max()) if len(counts) else 0,
            'examples': [{'ParticleID': int(i), 'count': int(c)} for i, c in zip(ids[duplicate][:8], counts[duplicate][:8])]}


def record_keys(columns):
    """保留64位整数；不能将ID与坐标合并成浮点二维数组。"""
    keys = np.empty(len(columns[0]), dtype=[(f'k{i}', np.asarray(c).dtype) for i, c in enumerate(columns)])
    for i, column in enumerate(columns):
        keys[f'k{i}'] = column
    return keys


def unique_key_map(raw_columns, yt_columns):
    a, b = record_keys(raw_columns), record_keys(yt_columns)
    sa, sb = np.argsort(a, kind='stable'), np.argsort(b, kind='stable')
    if np.any(a[sa][1:] == a[sa][:-1]) or np.any(b[sb][1:] == b[sb][:-1]):
        return None
    if not np.array_equal(a[sa], b[sb]):
        raise ValueError('yt/HDF5 组合ID集合不同，拒绝猜测匹配。')
    idx = np.empty(len(a), dtype=np.int64)
    idx[sb] = sa
    return idx


def match_particle_rows(raw, ad, ds):
    """返回 yt 行 -> HDF5 行。重复父ID本身不是错误，不用于跨快照追踪。"""
    yt_ids = integer_ids(ad['PartType0', 'ParticleIDs'].d, 'yt ParticleIDs')
    raw_ids = integer_ids(raw['ParticleIDs'], 'HDF5 ParticleIDs')
    if len(yt_ids) != len(raw_ids):
        raise ValueError(f'yt/HDF5 粒子数不同：{len(yt_ids)}/{len(raw_ids)}')
    info = duplicate_id_summary(raw_ids)
    raw_columns, yt_columns = [raw_ids], [yt_ids]
    names = ['ParticleIDs']
    idx = unique_key_map(raw_columns, yt_columns)
    if idx is None:
        for key in ['ParticleChildIDsNumber', 'ParticleIDGenerationNumber']:
            if key in raw and ('PartType0', key) in ds.field_list:
                raw_columns.append(integer_ids(raw[key], 'HDF5 '+key))
                yt_columns.append(integer_ids(ad['PartType0', key].d, 'yt '+key))
                names.append(key)
        idx = unique_key_map(raw_columns, yt_columns)
    info['id_fields_used'] = names.copy()
    xyz_raw = np.asarray(raw['Coordinates'], dtype=float)
    xyz_yt = np.asarray(ad['PartType0', 'Coordinates'].to_value('code_length'), dtype=float)
    if xyz_yt.shape != xyz_raw.shape or np.any(~np.isfinite(xyz_yt)):
        raise ValueError('yt/HDF5 坐标形状不同或包含非有限值。')
    if idx is None:
        # 同一快照，不依赖猜测原始读取顺序；首先以精确坐标作为消歧信息。
        raw_keys = record_keys(raw_columns + [xyz_raw[:, i] for i in range(3)])
        yt_keys = record_keys(yt_columns + [xyz_yt[:, i] for i in range(3)])
        sr, sy = np.argsort(raw_keys, kind='stable'), np.argsort(yt_keys, kind='stable')
        if np.any(raw_keys[sr][1:] == raw_keys[sr][:-1]) or np.any(yt_keys[sy][1:] == yt_keys[sy][:-1]):
            raise ValueError('父/子/代际ID及坐标均重复，无法唯一匹配；未按任意顺序配对。')
        if not all(np.array_equal(r[sr], y[sy]) for r, y in zip(raw_columns, yt_columns)):
            raise ValueError('yt/HDF5 重复ID组的成员数不同。')
        idx = np.empty(len(sr), dtype=np.int64)
        idx[sy] = sr
        info['method'] = 'ID fields + within-snapshot Coordinates; coordinate equality verified'
    else:
        info['method'] = 'unique '+ '+'.join(names)
    # 不使用金属丰度作为匹配键，否则会把被审计的字段错误掩盖掉。
    close = np.isclose(xyz_raw[idx], xyz_yt, rtol=2.e-12, atol=2.e-12)
    if not np.all(close):
        raise ValueError('ID对应后的坐标核验失败；可能有读入/排序差异，停止逐粒子对照。')
    info['max_coordinate_difference_code_length'] = float(np.max(np.abs(xyz_raw[idx]-xyz_yt))) if len(idx) else 0.
    return idx, info


def yt_audit(path, raw, units, args):
    import yt
    yt.funcs.mylog.setLevel(40)
    base = dict(UnitLength_in_cm=units['length_kpc']*KPC_CM,
                UnitMass_in_g=units['mass_msun']*MSUN_G,
                UnitVelocity_in_cm_per_s=units['velocity_kms']*1e5)
    cache = str(Path(args.output_dir)/'yt_cache') if args.output_dir else None
    ds = load_gizmo(str(path), unit_base=base, cache_dir=cache)
    if ds.cosmological_simulation:
        raise ValueError('本工具只用于非宇宙学快照。')
    ds.force_periodicity(False)
    ad = ds.all_data()
    raw_field = ('PartType0', 'Metallicity_00')
    if raw_field not in ds.field_list:
        raw_field = ('PartType0', 'Metallicity')
    q = ad[raw_field]
    idx, matching = match_particle_rows(raw, ad, ds)
    expected = raw['metal'][idx, 0]
    yt_values = np.asarray(q.to_value('dimensionless'), float)
    if yt_values.shape != expected.shape:
        raise ValueError(f'yt 金属字段形状异常：{yt_values.shape}')
    ok = np.isclose(yt_values, expected, rtol=2.e-6, atol=1.e-12, equal_nan=True)
    correct = expected / args.zsun
    legacy = np.asarray((q/args.zsun * ds.quan(1., 'Zsun')).to_value('Zsun'), float)
    finite = np.isfinite(correct) & (correct > 0)
    u = {'length_kpc': float(ds.length_unit.to_value('kpc')),
         'mass_msun': float(ds.mass_unit.to_value('Msun')),
         'velocity_kms': float(ds.velocity_unit.to_value('km/s'))}
    units_match = all(np.isclose(u[k], units[k], rtol=5.e-4) for k in u)
    result = {'status': 'ok' if ok.all() and units_match else 'mismatch', 'yt_version': yt.__version__,
              'field': list(raw_field), 'field_units': str(q.units),
              'particle_matching': matching,
              'yt_units': u, 'yt_units_match_raw_audit': units_match,
              'N_raw_field_mismatch': int((~ok).sum()),
              'v4_formula': '(yt_field / Zsun_mass_fraction * ds.quan(1,"Zsun")).to_value("Zsun")',
              'v4_formula_over_raw_Zsun': distribution(legacy[finite]/correct[finite]),
              'yt_gas_metallicity': None}
    try:
        alias = ad['gas', 'metallicity']
        result['yt_gas_metallicity'] = {'units': str(alias.units),
            'values_as_stored_in_yt': distribution(alias.d),
            'to_Zsun': distribution(alias.to_value('Zsun')),
            'note': 'yt 默认 Zsun 约定未必是本项目 0.02；本脚本采用 raw/指定Zsun，不能据此重复换算。'}
    except Exception as exc:
        result['yt_gas_metallicity'] = {'error': str(exc)}
    lum_raw = None
    def compute_luminosity():
        if not ok.all() or not units_match:
            raise ValueError('yt 原始字段或基本单位与 HDF5 审计不一致；拒绝继续计算 APEC 光度。')
        if (np.any(~np.isfinite(raw['metal'])) or np.any(raw['metal'] < 0)
                or np.any(expected > 1)
                or (raw['metal'].shape[1] == 11 and np.any(raw['metal'][:, :2].sum(axis=1) > 1))):
            raise ValueError('原始丰度有非法值；只输出原始审计，不计算 APEC。')
        table_dir = Path(args.apec_data_dir or '.')
        table = table_dir / 'apec_emissivity_v3.h5'
        if not table.is_file():
            raise FileNotFoundError(f'--xray 需要本地 {table.resolve()}，请设置 --apec-data-dir。')

        def audit_metallicity(field, data):
            value = data[raw_field].to_value('dimensionless') / args.zsun
            return data.ds.arr(value, 'Zsun')

        ds.add_field(('gas', 'audit_apec_metallicity'), function=audit_metallicity,
                     sampling_type='local', units='Zsun', force_override=True)
        yt.add_xray_emissivity_field(ds, args.energy_kev[0], args.energy_kev[1],
            metallicity=('gas', 'audit_apec_metallicity'), table_type='apec',
            redshift=0., data_dir=str(table_dir.resolve()))
        ef = ('gas', f'xray_emissivity_{args.energy_kev[0]}_{args.energy_kev[1]}_keV')
        # Independent of allsky cache, grids, smoothing and sky pixelisation.
        eps = ad[ef].to_value('erg/s/cm**3')
        mass = ad['PartType0', 'Masses'].to_value('g') if ('PartType0', 'Masses') in ds.field_list else ad['gas', 'mass'].to_value('g')
        rho = ad['PartType0', 'Density'].to_value('g/cm**3')
        with np.errstate(divide='ignore', invalid='ignore'):
            lum = eps*mass/rho
        lum_raw = np.empty(len(idx), dtype=float)
        lum_raw[idx] = lum
        xray_result = {'status': 'ok', 'energy_keV': args.energy_kev,
            'apec_table': str(table.resolve()), 'definition': 'sum epsilon_X * m/rho; intrinsic, unabsorbed',
            'N_invalid_luminosity': int(np.count_nonzero(~np.isfinite(lum) | (lum < 0))),
            'temperature_field': 'gas/temperature (yt); stored Temperature registered in K'}
        return xray_result, lum_raw
    if args.xray:
        try:
            result['xray'], lum_raw = compute_luminosity()
        except Exception as exc:
            result['xray'] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            if result['status'] == 'ok':
                result['status'] = 'partial'
    return result, lum_raw


def compare_photometry(path, rows, args):
    directory = Path(args.photometry_dir or args.snapshot_dir)
    if not directory.exists():
        return []
    key = re.sub(r'\.\d+$', '', Path(path).stem)
    row = next(r for r in rows if r['region'] == 'cube_los_retained')
    comparisons = []
    for f in sorted(directory.rglob('*photometry.json')):
        try:
            p = json.loads(f.read_text())
            pkey = re.sub(r'\.\d+$', '', Path(p.get('snapshot', '')).stem)
            if pkey != key:
                continue
            signature = p['cache_signature']
            grid = signature['grid']
            same = (np.isclose(grid['grid_radius_kpc'], args.cube_radius_kpc)
                and np.isclose(signature['disk_mask_half_height_kpc'], args.height_kpc)
                and np.allclose(signature['sun_offset_kpc'], args.sun_kpc)
                and np.allclose(signature['los_range_kpc'], args.los_kpc)
                and np.allclose(grid['energy_keV'], args.energy_kev))
            pa = p.get('particle_audit', {})
            old_z = pa.get('retained_luminosity_weighted_means', {}).get('metallicity_Zsun')
            old_l = pa.get('groups', {}).get('retained', {}).get('luminosity_erg_s')
            new_z, new_l = row.get('Zsun_luminosity_weighted'), row.get('L_X_erg_s_valid')
            comparisons.append({'file': str(f), 'geometry_and_band_match': bool(same),
                'old_particle_L_erg_s': old_l, 'new_particle_L_erg_s': new_l,
                'old_Zsun_luminosity_weighted': old_z, 'new_Zsun_luminosity_weighted': new_z,
                'new_to_old_particle_L_ratio': new_l/old_l if same and new_l is not None and old_l else None,
                'note': 'Compare original particle audit, not LOS-smoothed L. Filename match alone does not prove identical snapshot bytes or centring.'})
        except Exception as exc:
            if key in f.name:
                comparisons.append({'file': str(f), 'error': str(exc)})
    return comparisons


def inspect(parts, args, refs, role):
    notes = []
    meta = metadata(parts[0])
    for g in ['Header', 'Parameters']:
        if meta.get(g, {}).get('ComovingIntegrationOn', 0):
            raise ValueError('ComovingIntegrationOn=1：此脚本只用于非宇宙学快照。')
    units, unit_sources = unit_values(meta, args)
    raw, schemas = read_arrays(parts, units)
    center, center_method = get_center(parts, meta, units, args, notes)
    ranges = origin_ranges(meta, refs, notes)
    masks = masks_for(raw, center, args, ranges)
    temp, temp_source = raw_temperature(raw, units)
    if temp is not None:
        masks['retained_T_ge_1e6'] = masks['cube_los_retained'] & np.isfinite(temp) & (temp >= 1.e6)
    columns = []
    ncol = raw['metal'].shape[1]
    for i in range(ncol):
        label = ELEMENTS[i] if ncol == 11 else ('Z_total' if i == 0 else f'column_{i}')
        columns.append({'column': i, 'label': label, **distribution(raw['metal'][:, i])})
    quality = {'parent_ID_summary': duplicate_id_summary(raw['ParticleIDs']),
               'nonfinite_metal_entries': int(np.count_nonzero(~np.isfinite(raw['metal']))),
               'negative_metal_entries': int(np.count_nonzero(raw['metal'] < 0)),
               'N_total_Z_gt_1': int(np.count_nonzero(raw['metal'][:, 0] > 1))}
    if ncol == 11:
        z, he = raw['metal'][:, 0], raw['metal'][:, 1]
        valid = np.isfinite(z) & (z > 0)
        quality['N_Z_plus_He_gt_1'] = int(np.count_nonzero(z+he > 1))
        quality['tracked_metals_sum_over_total_Z'] = distribution(raw['metal'][valid, 2:].sum(axis=1)/z[valid])
        quality['note'] = 'He 是列1；列2..10仅用于一致性参考，不加到列0。不要求所跟踪元素和严格等于总Z。'
    if not ranges:
        notes.append('缺少来源 ID 元数据：按空间正常统计；来源为 unassigned，可用 --reference-ic 补充。')
    yt_result, lum = {'status': 'disabled'}, None
    if args.yt_check != 'off' or args.xray:
        try:
            yt_result, lum = yt_audit(parts[0], raw, units, args)
        except ImportError as exc:
            yt_result = {'status': 'unavailable', 'error': str(exc)}
            notes.append('yt/APEC 对照未执行；原始 HDF5 检查已完成。')
        except Exception as exc:
            yt_result = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            notes.append('yt/APEC 对照失败；原始 HDF5 检查仍保留，错误见 yt_check。')
    rows = [region_stats(name, mask, raw, args, temp, lum) for name, mask in masks.items()]
    time_code = float(meta['Header'].get('Time', 0.))
    time_gyr = time_code * units['length_kpc']*KPC_CM/(units['velocity_kms']*1.e5)/GYR_S
    for row in rows:
        row.update({'snapshot': str(parts[0]), 'role': role, 'time_code': time_code, 'time_gyr': time_gyr})
    return {'snapshot': str(parts[0]), 'parts': [str(p) for p in parts], 'role': role,
        'time_code': time_code, 'time_gyr': time_gyr, 'units': units, 'unit_sources': unit_sources,
        'center_kpc': center, 'center_method': center_method, 'metallicity_schema': schemas,
        'origin_ranges': ranges, 'reference_abundance_metadata': meta.get('MWModelReference', {}),
        'temperature_source': temp_source, 'columns': columns, 'quality': quality,
        'yt_check': yt_result, 'regions': rows, 'notes': notes,
        'photometry_comparison': compare_photometry(parts[0], rows, args)}


def fmt(x):
    return 'N/A' if x is None else f'{x:.6g}'


def describe(report):
    lines = [f"\n{Path(report['snapshot']).name} [{report['role']}]",
             f"  文件：{report['snapshot']}",
             f"  中心：{report['center_method']} {report['center_kpc']} kpc",
             f"  单位：{report['units']}；来源：{report['unit_sources']}"]
    for row in report['regions']:
        if row['region'] in ['all_gas', 'cube_los_slab', 'cube_los_retained'] or row['region'].startswith('origin_'):
            lines.append(f"  {row['region']}: N={row['N']}, raw_Z中位={fmt(row['raw_Z_median'])}, "
                         f"Z/Zsun范围=[{fmt(row['Zsun_min'])}, {fmt(row['Zsun_max'])}], "
                         f"中位={fmt(row['Zsun_median'])}, 质量加权={fmt(row['Zsun_mass_weighted'])}, "
                         f"光度加权={fmt(row.get('Zsun_luminosity_weighted'))}, 零丰度粒子={row['N_zero_Z']}")
    yr = report['yt_check']
    lines.append(f"  yt逐粒子核对：{yr['status']}")
    if 'v4_formula_over_raw_Zsun' in yr:
        lines.append(f"  粒子匹配方式：{yr['particle_matching']['method']}")
        lines.append(f"  v4丰度公式/raw换算 的中位比值：{fmt(yr['v4_formula_over_raw_Zsun']['median'])}（应接近1）")
        lines.append(f"  yt/HDF5原始丰度不一致粒子数：{yr['N_raw_field_mismatch']}")
    if 'error' in yr:
        lines.append('  yt/APEC错误：' + yr['error'])
    if yr.get('xray', {}).get('status') == 'error':
        lines.append('  APEC错误（原始丰度/yt字段对照已保留）：' + yr['xray']['error'])
    q = report['quality']
    ids = q['parent_ID_summary']
    lines.append(f"  重复父ID组数={ids['N_repeated_parent_IDs']}，涉及粒子数={ids['N_particles_in_repeated_ID_groups']}（不据此判定模拟错误）")
    lines.append(f"  原始数据非法项：NaN/Inf={q['nonfinite_metal_entries']}, "
                 f"负丰度={q['negative_metal_entries']}, Z>1={q['N_total_Z_gt_1']}, "
                 f"Z+He>1={q.get('N_Z_plus_He_gt_1', 'N/A')}")
    for comp in report['photometry_comparison']:
        if 'error' not in comp:
            lines.append(f"  对照 {Path(comp['file']).name}: 几何/能段一致={comp['geometry_and_band_match']}; "
                         f"旧/新光度加权Z={fmt(comp['old_Zsun_luminosity_weighted'])}/{fmt(comp['new_Zsun_luminosity_weighted'])}")
    for note in report['notes']:
        lines.append('  提示：' + note)
    return lines


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snapshot-dir', default='./mw_lmc_smc_relaxed_trial00_000_run/production/output')
    p.add_argument('--snapshots', nargs='+', help='文件/通配符；设置后替代 --snapshot-dir 扫描')
    p.add_argument('--reference-ic', nargs='+', default=[], help='提取来源 ID，且对 IC 本身执行原始审计')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--zsun', type=float, default=.02)
    p.add_argument('--cube-radius-kpc', type=float, default=30.)
    p.add_argument('--height-kpc', type=float, default=5.)
    p.add_argument('--sun-kpc', nargs=3, type=float, default=[-8.29, 0., 0.])
    p.add_argument('--los-kpc', nargs=2, type=float, default=[0., 60.])
    p.add_argument('--center-kpc', nargs=3, type=float, default=None)
    p.add_argument('--bh-id', type=int, default=None)
    p.add_argument('--unit-length-kpc', type=float, default=None)
    p.add_argument('--unit-mass-msun', type=float, default=None)
    p.add_argument('--unit-velocity-kms', type=float, default=None)
    p.add_argument('--yt-check', choices=['auto', 'on', 'off'], default='auto')
    p.add_argument('--xray', action='store_true', help='yt/APEC 粒子光度及光度加权Z；需要本地APEC表')
    p.add_argument('--apec-data-dir', default='.')
    p.add_argument('--energy-kev', type=float, nargs=2, default=[.6, 1.])
    p.add_argument('--photometry-dir', default=None, help='旧 photometry.json 所在目录；默认递归查找 snapshot-dir')
    args = p.parse_args()
    if not 0 < args.zsun < 1 or args.cube_radius_kpc <= 0 or args.height_kpc < 0:
        p.error('zsun需在(0,1)，cube-radius>0，height>=0')
    if not 0 <= args.los_kpc[0] < args.los_kpc[1] or not 0 < args.energy_kev[0] < args.energy_kev[1]:
        p.error('LOS/能段范围无效')
    for v in [args.zsun, args.cube_radius_kpc, args.height_kpc, *args.sun_kpc, *args.los_kpc, *args.energy_kev]:
        if not np.isfinite(v):
            p.error('数值参数必须有限')
    return args


def main():
    args = arguments()
    snapshots = discover(args)
    references = [snapshot_parts(p) for p in args.reference_ic]
    ref_meta = [(str(p[0]), metadata(p[0])) for p in references]
    out = Path(args.output_dir or (Path(args.snapshot_dir)/'metallicity_audit')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = ['GIZMO 气体金属丰度检查 v1.1（支持重复父ID）；原始文件只读。',
           f'Z/Zsun = Metallicity[:,0] / {args.zsun:g}；不累加He和元素列。',
           f'立方体半边长={args.cube_radius_kpc:g} kpc；保留 |z|>={args.height_kpc:g} kpc。',
           '质量加权与光度加权不同。未指定 --xray 时，不会用质量加权冒充光度加权。',
           '不同来源ID标签只代表粒子原始所属组；富集、混合和新生成粒子需要结合模拟物理解释。']
    print('\n'.join(log), flush=True)
    reports, failures, seen = [], [], set()
    for role, collection in [('reference_IC', references), ('snapshot', snapshots)]:
        for parts in collection:
            if str(parts[0]) in seen:
                continue
            seen.add(str(parts[0]))
            print(f'\n读取 {parts[0]} ...', flush=True)
            try:
                # IC 未必含 Density/Temperature；仍做原始丰度与yt读取检查，不强求APEC。
                current = argparse.Namespace(**vars(args))
                if role == 'reference_IC':
                    current.xray = False
                report = inspect(parts, current, ref_meta, role)
                reports.append(clean(report))
                lines = describe(clean(report))
            except Exception as exc:
                error = {'snapshot': str(parts[0]), 'error': f'{type(exc).__name__}: {exc}'}
                failures.append(error)
                lines = [f"失败：{error['snapshot']}：{error['error']}"]
            print('\n'.join(lines), flush=True)
            log.extend(lines)
            payload = {'schema': 1, 'settings': vars(args), 'reports': reports, 'failures': failures,
                       'note': 'Raw Z mass fraction is authoritative input. Audit does not modify snapshots or claim physical validation.'}
            (out/'metallicity_audit.json').write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    rows = [row for report in reports for row in report['regions']]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'metallicity_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out/'metallicity_audit.txt').write_text('\n'.join(log)+'\n', encoding='utf-8')
    print(f'\n输出：{out}/metallicity_audit.json\n      {out}/metallicity_summary.csv\n      {out}/metallicity_audit.txt')
    requested_errors = any(r['yt_check']['status'] in ['error', 'mismatch', 'unavailable', 'partial']
                           for r in reports if args.yt_check == 'on' or (args.xray and r['role'] == 'snapshot'))
    invalid_raw = any(r['quality']['nonfinite_metal_entries'] or r['quality']['negative_metal_entries']
                      or r['quality']['N_total_Z_gt_1'] or r['quality'].get('N_Z_plus_He_gt_1', 0) for r in reports)
    return 1 if failures or requested_errors or invalid_raw else 0


if __name__ == '__main__':
    sys.exit(main())
