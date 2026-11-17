#!/usr/bin/env python3
"""Restore IC metals and stellar formation times into a ZERO-TIME GIZMO snapshot.

For the user's non-cosmological, 11-species, PartType4 stellar model only.
No GIZMO source changes. Requires numpy and h5py. Never modifies inputs.

prepare: validates the IC and same-build snapshot_000, copies the entire snapshot,
         restores Metallicity and StellarFormationTime by unique ParticleIDs,
         fixes Flag_Metals=11, and optionally writes smoke/production parameters.
verify:  compares an early mode-2 output against the restored input by IDs.

IMPORTANT
* The template MUST be the initial snapshot at Header/Time=0, from the SAME IC
  and SAME executable/physics configuration. No evolved snapshot is accepted.
* This does NOT repair a simulation already evolved with incorrect abundances.
* Keep COOL_MOLECFRAC=5 for the audited configuration. The uploaded init.c has a
  separate missing-loop defect in its COOL_MOLECFRAC_NONEQM mode-2 branch; do not
  assume this workflow validates that non-equilibrium branch.
* All other snapshot fields are retained. Temperature/cooling-rate diagnostics
  in the prepared file are template values, NOT recalculated scientific results.
  Use the newly evolved GIZMO output for diagnostics.
* Mode 2 uses Header/Time and infers the next snapshot number from the input
  filename. This tool requires a basename ending in _000.hdf5; output starts at
  index 001 (or later), not necessarily at time exactly zero.
* No universal completeness claim is made for a different GIZMO build.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def inventory(f):
    h = f['Header'].attrs
    need(int(h['NumFilesPerSnapshot']) == 1, '仅支持单文件快照。')
    counts = np.array([len(f[f'PartType{p}/ParticleIDs'])
                       if f'PartType{p}' in f else 0 for p in range(6)])
    for key in ('NumPart_ThisFile', 'NumPart_Total'):
        need(np.array_equal(counts, h[key]), f'{key} 与实际粒子数不符。')
    need(not np.any(h.get('NumPart_Total_HighWord', np.zeros(6))),
         '不支持超过单文件计数范围的输入。')
    need(counts[0] > 0, '没有气体粒子。')
    need(counts[2] == counts[3] == 0, '本工具要求所有恒星均为 PartType4。')
    for key in ('Omega0', 'Omega_Matter', 'OmegaLambda', 'Omega_Lambda'):
        if key in h:
            need(float(h[key]) == 0, '本工具仅适用于非宇宙学模型。')
    return counts


def id_map(source, target):
    a, b = source['ParticleIDs'][:], target['ParticleIDs'][:]
    need(a.dtype.kind in 'iu' and b.dtype.kind in 'iu', 'ParticleIDs 必须为整数。')
    need(a.ndim == b.ndim == 1, 'ParticleIDs 必须是一维数组。')
    sa, sb = np.argsort(a), np.argsort(b)
    need(len(a) == len(b) and np.array_equal(a[sa], b[sb]),
         f'{source.name}: 粒子 ID 集合不一致。')
    need(len(a) < 2 or np.all(a[sa][1:] != a[sa][:-1]),
         f'{source.name}: 存在重复父 ID；本工具拒绝猜测对应关系。')
    mapping = np.empty(len(a), dtype=np.int64)
    mapping[sb] = sa
    return mapping


def masses(f, p):
    group = f[f'PartType{p}']
    if 'Masses' in group:
        return group['Masses'][:]
    mass = float(f['Header'].attrs['MassTable'][p])
    need(mass > 0, f'PartType{p}: 缺少粒子质量。')
    return np.full(len(group['ParticleIDs']), mass)


def close(a, b, name, rtol=1e-7, atol=1e-8):
    need(a.shape == b.shape, f'{name}: 形状不同。')
    need(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f'{name}: 非法数值。')
    difference = float(np.max(np.abs(a - b))) if a.size else 0.
    need(np.allclose(a, b, rtol=rtol, atol=atol),
         f'{name}: 差异超限，max_abs={difference:.6g}；可能不是同一份 IC 的起始快照。')


def metals(group):
    z = group['Metallicity'][:]
    need(z.shape == (len(group['ParticleIDs']), 11), f'{group.name}: 要求 N×11 丰度。')
    need(np.all(np.isfinite(z)) and np.all(z >= 0), f'{group.name}: 丰度存在负值/NaN。')
    need(np.all(z[:, 0] + z[:, 1] <= 1 + 1e-10), f'{group.name}: Z+He>1。')
    return z


def write_dataset(group, key, data):
    attributes = dict(group[key].attrs) if key in group else {}
    if key in group:
        del group[key]
    ds = group.create_dataset(key, data=data)
    for key, value in attributes.items():
        ds.attrs[key] = value


def parameter_text(path, updates):
    lines = Path(path).read_text().splitlines()
    values = {}
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(('#', '%')):
            fields = stripped.split()
            if len(fields) >= 2:
                need(fields[0] not in values, f'参数重复: {fields[0]}')
                values[fields[0]] = fields[1]
    need(float(values.get('ComovingIntegrationOn', '-1')) == 0,
         '参数文件必须明确 ComovingIntegrationOn=0。')
    need(float(values.get('InitGasTemp', '-1')) == 0, '要求 InitGasTemp=0。')
    need(float(values.get('TimeBegin', '-1')) == 0, '本工具要求 TimeBegin=0。')
    need(int(values.get('ICFormat', '-1')) == int(values.get('SnapFormat', '-1')) == 3,
         '要求 ICFormat=SnapFormat=3。')
    need(int(values.get('NumFilesPerSnapshot', '-1')) == 1, '要求单文件快照。')
    need(float(values.get('TimeMax', '0')) > 0, 'TimeMax 必须大于零。')
    need(set(updates) <= set(values), f'参数文件缺少键: {set(updates) - set(values)}')
    for i, line in enumerate(lines):
        words = line.strip().split()
        if words and words[0] in updates:
            lines[i] = f'{words[0]:45s} {updates[words[0]]}'
    return '\n'.join(lines) + '\n'


def prepare(args):
    original, template, output = map(lambda p: Path(p).resolve(),
                                     (args.original_ic, args.template_snapshot, args.output))
    need(output.name.endswith('_000.hdf5'), '输出名必须以 _000.hdf5 结尾。')
    need(not output.exists(), f'输出已存在，拒绝覆盖: {output}')
    parameter_files = []
    if args.param:
        for kind in ('smoke', 'production'):
            par = output.with_name(output.stem + f'.{kind}.param')
            directory = output.with_name(output.stem + f'_{kind}_output')
            need(not par.exists(), f'参数文件已存在: {par}')
            need(not directory.exists(), f'输出目录已存在: {directory}')
            updates = {'InitCondFile': str(output.with_suffix('')),
                       'OutputDir': str(directory) + '/', 'TimeBegin': '0',
                       'TimeOfFirstSnapshot': '0', 'OutputListOn': '0',
                       'SnapshotFileBase': 'snapshot'}
            if kind == 'smoke':
                updates.update(TimeMax='0.0001', TimeBetSnapshot='0.00005',
                               MaxSizeTimestep='0.00001')
            need(' ' not in str(output) and ' ' not in str(directory),
                 'GIZMO 参数路径请不要包含空格。')
            parameter_files.append((par, parameter_text(args.param, updates), directory))

    maps = {}
    with h5py.File(original, 'r') as src, h5py.File(template, 'r') as snap:
        n = inventory(src)
        need(np.array_equal(n, inventory(snap)), 'IC 与模板粒子数不同。')
        for f in (src, snap):
            t = float(f['Header'].attrs['Time'])
            need(np.isfinite(t) and abs(t) <= 1e-12,
                 f'{f.filename}: Header/Time={t}；拒绝用已经演化的快照补写丰度。')
        for key in ('BoxSize', 'HubbleParam', 'UnitLength_in_cm',
                    'UnitMass_in_g', 'UnitVelocity_in_cm_per_s'):
            if key in src['Header'].attrs and key in snap['Header'].attrs:
                need(np.allclose(src['Header'].attrs[key], snap['Header'].attrs[key],
                                 rtol=1e-8, atol=0), f'单位/头信息不同: {key}')
        for p in range(6):
            if not n[p]:
                continue
            a, b = src[f'PartType{p}'], snap[f'PartType{p}']
            index = maps[p] = id_map(a, b)
            for field in ('Coordinates', 'Velocities'):
                close(a[field][:][index], b[field][:], f'{a.name}/{field}', atol=1e-6)
            close(masses(src, p)[index], masses(snap, p), f'{a.name}/Masses', atol=0)
            if p in (0, 4):
                metals(a)
                metals(b)
            if p == 4:
                age = a['StellarFormationTime'][:]
                need(age.shape == (n[p],) and np.all(np.isfinite(age)) and np.all(age <= 0),
                     '初始恒星形成时间必须有限且不晚于 t=0。')
                need('StellarFormationTime' in b, '模板缺少恒星形成时间字段。')
                print(f'原始恒星形成时间范围（code time）: [{age.min():.8g}, {age.max():.8g}]')
            if p == 5:
                need('BH_Mass' in b and np.all(b['BH_Mass'][:] > 0), '模板缺少有效 BH_Mass。')
                if 'BH_Mass' in a:
                    close(a['BH_Mass'][:][index], b['BH_Mass'][:], 'BH_Mass', atol=0)
        close(src['PartType0/InternalEnergy'][:][maps[0]], snap['PartType0/InternalEnergy'][:],
              'InternalEnergy', rtol=1e-6, atol=1e-8)
        for field in ('Density', 'SmoothingLength', 'ElectronAbundance'):
            data = snap[f'PartType0/{field}'][:]
            need(data.shape == (n[0],) and np.all(np.isfinite(data)), f'模板 {field} 非法。')
            need(np.all(data >= 0 if field == 'ElectronAbundance' else data > 0),
                 f'模板 {field} 超出允许范围。')
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.metal_restart_', suffix='.hdf5', dir=output.parent)
        os.close(fd)
        try:
            shutil.copyfile(template, temporary)
            with h5py.File(temporary, 'r+') as dst:
                dst['Header'].attrs['Flag_Metals'] = np.int32(11)
                for p, index in maps.items():
                    a, b = src[f'PartType{p}'], dst[f'PartType{p}']
                    if p in (0, 4):
                        write_dataset(b, 'Metallicity', np.asarray(a['Metallicity'][:][index], dtype='f8'))
                        if p == 4:
                            write_dataset(b, 'StellarFormationTime',
                                          np.asarray(a['StellarFormationTime'][:][index], dtype='f8'))
                    # Preserve optional origin labels absent from GIZMO outputs.
                    for key in ('GasComponent', 'StellarComponent'):
                        if key in a:
                            write_dataset(b, key, a[key][:][index])
                for key in src:
                    if key != 'Header' and not key.startswith('PartType') and key not in dst:
                        src.copy(key, dst)
                dst.attrs['MetalRestartOriginalIC'] = str(original)
                dst.attrs['MetalRestartTemplateSnapshot'] = str(template)
                dst.attrs['MetalRestartRequiredMode'] = np.int32(2)
                dst.attrs['MetalRestartNote'] = 'Restored by unique IDs; diagnostics inherited, not recomputed.'
                # Exact check of every restored species before publishing the file.
                for p in (0, 4):
                    if n[p]:
                        need(np.array_equal(src[f'PartType{p}/Metallicity'][:][maps[p]],
                                            dst[f'PartType{p}/Metallicity'][:]), '写后丰度校验失败。')
                if n[4]:
                    need(np.array_equal(src['PartType4/StellarFormationTime'][:][maps[4]],
                                        dst['PartType4/StellarFormationTime'][:]), '写后形成时间校验失败。')
            # Exclusive hard-link publication prevents accidental output replacement.
            os.link(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print(f'已生成: {output}\nFlag_Metals=11；气体/恒星丰度及恒星形成时间已按 ID 恢复。')
    for par, content, directory in parameter_files:
        with par.open('x') as handle:
            handle.write(content)
        directory.mkdir()
        print(f'参数: {par}\n输出目录: {directory}')
    print('使用相同 GIZMO 可执行文件，以“参数文件 2”启动；不要在 2 后加快照编号。')
    print('先跑 smoke，再检查最早输出。未执行真实 GIZMO；启动核验仍需在你的机器上完成。')


def verify(args):
    good = True
    with h5py.File(args.input, 'r') as src, h5py.File(args.snapshot, 'r') as dst:
        ns, nd = inventory(src), inventory(dst)
        elapsed = float(dst['Header'].attrs['Time']) - float(src['Header'].attrs['Time'])
        need(0 <= elapsed <= args.max_elapsed_code + 1e-12,
             f'elapsed_code={elapsed:g} 超出起始测试范围；请选 smoke 的最早输出。')
        need(np.array_equal(ns, nd), '粒子数已变化；不能做本工具的逐粒子启动核验。')
        need(int(dst['Header'].attrs['Flag_Metals']) == 11, '输出 Flag_Metals 不是 11。')
        print(f'elapsed_code={elapsed:.8g} (约 {elapsed * 0.977792222:.8g} Gyr)')
        for p in (0, 4):
            if not ns[p]:
                continue
            a, b = src[f'PartType{p}'], dst[f'PartType{p}']
            index = id_map(a, b)
            za, zb = metals(a)[index], metals(b)
            for k, name in enumerate(('Z', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe')):
                match = np.isclose(za[:, k], zb[:, k], rtol=1e-6, atol=1e-12)
                lost = np.count_nonzero((za[:, k] > 0) & (zb[:, k] == 0))
                print(f'PartType{p} {name:2s}: 不一致={np.count_nonzero(~match)}, 正值变零={lost}')
                good = good and bool(np.all(match))
            if p == 4:
                match = np.isclose(a['StellarFormationTime'][:][index], b['StellarFormationTime'][:],
                                   rtol=1e-7, atol=1e-10)
                print(f'恒星形成时间不一致={np.count_nonzero(~match)}')
                good = good and bool(np.all(match))
        if ns[5]:
            index = id_map(src['PartType5'], dst['PartType5'])
            close(src['PartType5/BH_Mass'][:][index], dst['PartType5/BH_Mass'][:], 'BH_Mass', atol=0)
    if not good:
        print('未通过“初始字段保留”核验。少量差异可能来自真实富集/混合；大批正值变零则提示初始化问题。')
        return 2
    print('PASS：全部 11 列丰度及初始恒星形成时间保留（给定容差内）。')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='从原始 IC 和零时刻完整快照制作模式 2 输入')
    p.add_argument('--original-ic', required=True)
    p.add_argument('--template-snapshot', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--param', help='原始参数文件；生成独立 smoke 和 production 参数副本')
    p = sub.add_parser('verify', help='按 ID 核对模式 2 的最早输出')
    p.add_argument('--input', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--max-elapsed-code', type=float, default=0.0001)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args)
            return 0
        return verify(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
