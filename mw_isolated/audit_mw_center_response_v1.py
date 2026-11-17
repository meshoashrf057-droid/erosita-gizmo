#!/usr/bin/env python3
"""只读核对 MW BH 与固定恒星中心的响应；不修改 shooting 残差或验收标准。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np

VERSION = "mw-center-response-audit-v1.0"
IDENTITY_FIELDS = ("ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber")
COORDS = ("x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_keys(group, fields=None, mask=None):
    """与原验证器一样使用完整整数身份；不悄悄丢弃 child/generation 字段。"""
    present = tuple(name for name in IDENTITY_FIELDS if name in group)
    fields = tuple(fields or present)
    require(bool(fields) and fields[0] == "ParticleIDs" and present == fields,
            f"恒星身份字段不一致：要求 {fields}，实际 {present}")
    arrays = []
    for field in fields:
        a = np.asarray(group[field][:])
        require(a.ndim == 1 and a.dtype.kind in "ui" and np.all(a >= 0),
                f"{field} 必须是一维非负整数")
        arrays.append(a if mask is None else a[mask])
    require(len({len(a) for a in arrays}) == 1, "身份列长度不同")
    keys = np.empty(len(arrays[0]), dtype=[(field, np.uint64) for field in fields])
    for field, a in zip(fields, arrays):
        keys[field] = a
    require(len(np.unique(keys)) == len(keys), "存在重复完整恒星身份，无法安全匹配")
    return keys


def shrinking_core_state(pos, vel, mass, core_radius_kpc, min_particles=40):
    """原验证器的 shrinking-core 算法；仅增加输入有效性检查。"""
    pos, vel, mass = map(lambda x: np.asarray(x, dtype=float), (pos, vel, mass))
    require(pos.shape == vel.shape == (len(mass), 3), "位置/速度/质量形状不一致")
    require(all(np.all(np.isfinite(x)) for x in (pos, vel, mass)) and np.all(mass > 0),
            "核心输入含非法数值或非正质量")
    require(np.isfinite(core_radius_kpc) and core_radius_kpc > 0, "孔径必须有限且为正")
    require(len(pos) >= min_particles, f"只有 {len(pos)} 颗示踪星，需要 >= {min_particles}")
    center = np.average(pos, axis=0, weights=mass)
    mask = np.ones(len(pos), dtype=bool)
    radius = float("inf")
    for _ in range(30):
        dist = np.linalg.norm(pos - center, axis=1)
        active = dist[mask]
        if active.size <= 2 * min_particles:
            break
        radius = float(np.quantile(active, 0.80))
        new_mask = dist <= radius
        if new_mask.sum() < min_particles:
            break
        mask = new_mask
        center = np.average(pos[mask], axis=0, weights=mass[mask])
        if radius <= core_radius_kpc:
            break
    dist = np.linalg.norm(pos - center, axis=1)
    final = dist <= core_radius_kpc
    if final.sum() < min_particles:
        final = mask
    center = np.average(pos[final], axis=0, weights=mass[final])
    velocity = np.average(vel[final], axis=0, weights=mass[final])
    rms = math.sqrt(float(np.average(np.sum((pos[final] - center) ** 2, axis=1), weights=mass[final])))
    return center, velocity, int(final.sum()), rms


def check_header(f):
    h = f["Header"].attrs
    require(int(h.get("NumFilesPerSnapshot", 1)) == 1, "不支持分块快照")
    if "NumPart_ThisFile" in h and "NumPart_Total" in h:
        total = np.asarray(h["NumPart_Total"], dtype=np.uint64)
        total += np.asarray(h.get("NumPart_Total_HighWord", np.zeros(6)), dtype=np.uint64) << np.uint64(32)
        require(np.array_equal(total, h["NumPart_ThisFile"]), "分块快照或粒子计数不一致")
    for group in (f["Header"], f.get("Parameters")):
        if group is not None and "ComovingIntegrationOn" in group.attrs:
            require(int(group.attrs["ComovingIntegrationOn"]) == 0, "只支持非宇宙学快照")
    require(abs(float(h.get("Redshift", 0.0))) < 1e-12, "不支持非零红移快照")
    require(np.isfinite(float(h["Time"])), "快照时间非法")


def star_arrays(f):
    s = f["PartType4"]
    p, v = np.asarray(s["Coordinates"][:], float), np.asarray(s["Velocities"][:], float)
    if "Masses" in s:
        m = np.asarray(s["Masses"][:], float)
    else:
        m = np.full(len(p), float(f["Header"].attrs["MassTable"][4]))
    require(m.shape == (len(p),) and p.shape == v.shape == (len(p), 3), "恒星场形状错误")
    require(all(np.all(np.isfinite(a)) for a in (p, v, m)) and np.all(m > 0), "恒星场非法")
    return p, v, m


def frozen_aperture_velocity(p, v, m, center, radius=10.0):
    mask = np.linalg.norm(p - center, axis=1) <= radius
    require(np.count_nonzero(mask) >= 100, "固定 MW 恒星速度孔径内少于 100 颗星")
    return {"radius_kpc": radius, "nstar": int(mask.sum()),
            "velocity_kms": np.average(v[mask], axis=0, weights=m[mask]).tolist()}


def record_reference(record):
    o, pr = record["orbit"], record["provenance"]
    path = Path(pr["reference_ic"])
    actual_sha = sha256_file(path)
    require(actual_sha == pr["reference_ic_sha256"] == o["output_ic_sha256"], f"{path}: IC 指纹不一致")
    sig = pr["measurement_signature"]
    require(sig["algorithm"] == "fixed-original-star-identities-shrinking-core-v4" and
            sig["tracer_policy"] == "original-stars" and
            sig["host_frame"] == "BH-position+MW-inner10kpc-mass-weighted-star-velocity-v3",
            "不支持此测量算法；必须使用既定 original-stars 验证结果")
    require(sig["lmc_core_radius_kpc"] == 8.0 and sig["smc_core_radius_kpc"] == 3.0, "需要原有 8/3 kpc 卫星孔径")
    require(pr["target_covered"] and pr["shooting_usable"] and pr["evaluation_method"] == "exact target snapshot",
            "需要准确目标快照，不能使用时间插值")
    sources = pr["target_snapshot_sources"]
    require(len(sources) == 1 and sources[0]["weight"] == 1.0, "需要单个准确目标快照")
    require(o["model_key"] == pr["model_key"] and o["trial_label"] == pr["trial_label"] == record["trial"], "模型或 trial 标记不一致")
    require(abs(o["stage2_target_code"] - pr["target_time_code"]) < 1e-10, "目标时间不一致")
    units = o["units"]
    require(all(np.isfinite(units[k]) and units[k] > 0 for k in ("length_kpc", "velocity_kms", "mass_msun", "time_gyr")), "单位非法")
    with h5py.File(path, "r") as f:
        check_header(f)
        g = f["MagellanicOrbitReference"]
        require(abs(float(f["Header"].attrs["Time"]) - o["stage2_start_code"]) < 1e-10, "IC 起始时间不一致")
        require(abs(float(g.attrs["PresentTargetTime_Code"]) - pr["target_time_code"]) < 1e-10, "IC 目标时间不一致")
        keys = identity_keys(f["PartType4"], sig["identity_fields"])
        origins = np.asarray(f["MergeParticleOrigins/PartType4"][:])
        require(origins.shape == (len(keys),) and np.all(np.isin(origins, (0, 1, 2))), "IC 恒星来源标签非法")
        reference = {"mw_keys": keys[origins == 0], "fields": sig["identity_fields"],
                     "bh_id": int(g.attrs["MW_BH_ID"]), "ic_path": str(path), "ic_sha256": actual_sha,
                     "snapshot_path": sources[0]["file"], "units": units, "ranges": {}, "sat_keys": {}}
        require(len(reference["mw_keys"]) >= 100, "IC 的 MW 恒星过少")
        ids = np.asarray(f["PartType4/ParticleIDs"][:], np.uint64)
        for sat in ("lmc", "smc"):
            label = sat.upper()
            reference["ranges"][sat] = [int(g.attrs[label + "_ID_START"]), int(g.attrs[label + "_ID_END"])]
            lo, hi = int(g.attrs[label + "_OLD_STAR_ID_START"]), int(g.attrs[label + "_OLD_STAR_ID_END"])
            mask = (ids >= lo) & (ids <= hi)
            reference["sat_keys"][sat] = keys[mask]
            require(mask.sum() == pr["reference_tracer_counts"][sat], f"{sat}: 原始示踪数量不一致")
        return reference


def inspect_trial(record, reference):
    units = reference["units"]
    length, speed = units["length_kpc"], units["velocity_kms"]
    scale = np.array([length] * 3 + [speed] * 3)
    sources = record["provenance"]["target_snapshot_sources"]
    history = [h for h in record["history"] if Path(h["file"]) == Path(reference["snapshot_path"])]
    require(len(history) == 1, "目标快照在历史记录中缺失或重复")
    h = history[0]
    with h5py.File(reference["snapshot_path"], "r") as f:
        check_header(f)
        time = float(f["Header"].attrs["Time"])
        require(abs(time - record["provenance"]["target_time_code"]) < 1e-10 and
                abs(time - sources[0]["time_code"]) < 1e-10 and abs(time - float(h["time_code"])) < 1e-10,
                "目标快照时间与记录不符")
        keys = identity_keys(f["PartType4"], reference["fields"])
        p, v, m = star_arrays(f)
        bh = f["PartType5"]
        matches = np.flatnonzero(np.asarray(bh["ParticleIDs"][:], np.uint64) == reference["bh_id"])
        require(len(matches) == 1, "指定 MW BH 必须唯一，不能回退到第一颗 BH")
        bp = np.asarray(bh["Coordinates"][matches[0]], float)
        bv = np.asarray(bh["Velocities"][matches[0]], float)
        require(bp.shape == bv.shape == (3,) and np.all(np.isfinite(bp)) and np.all(np.isfinite(bv)), "BH 场非法")
        ids = np.asarray(f["PartType4/ParticleIDs"][:], np.uint64)
        all_mw = np.ones(len(ids), bool)
        for lo, hi in reference["ranges"].values():
            all_mw &= ~((ids >= lo) & (ids <= hi))
        official_core = all_mw & (np.linalg.norm(p - bp, axis=1) <= 10.0)
        require(official_core.sum() >= 100, "无法复现原有 MW 内 10 kpc 恒星速度")
        hv = np.average(v[official_core], axis=0, weights=m[official_core])
        official = np.r_[bp, hv]
        expected = np.array([float(h["host_" + k]) for k in COORDS])
        error = float(np.max(np.abs((official - expected) * scale)))
        require(error <= 1e-6, f"原有 MW 测量不能复现，最大差 {error}")
        fixed = np.isin(keys, reference["mw_keys"])
        require(fixed.sum() == len(reference["mw_keys"]), "固定 MW 恒星身份有缺失；停止以避免样本变化混淆")
        fp, fv, fm = p[fixed] * length, v[fixed] * speed, m[fixed] * units["mass_msun"]
        result = {"trial": record["trial"], "reference_ic": reference["ic_path"],
                  "reference_ic_sha256": reference["ic_sha256"], "snapshot": reference["snapshot_path"],
                  "time_code": time, "units": units, "bh_id": reference["bh_id"],
                  "bh_position_kpc": (bp * length).tolist(), "bh_velocity_kms": (bv * speed).tolist(),
                  "official_host_velocity_kms": (hv * speed).tolist(), "official_host_nstar": int(official_core.sum()),
                  "official_reproduction_max_abs_physical": error,
                  "fixed_mw_reference_count": len(reference["mw_keys"]), "fixed_mw_matched_count": int(fixed.sum()),
                  "fixed_mw_matched_fraction": float(fixed.sum() / len(reference["mw_keys"])),
                  "fixed_mw_velocity_10kpc_about_bh": frozen_aperture_velocity(fp, fv, fm, bp * length),
                  "stellar_centers": {}, "satellites": {}}
        for radius in (2.0, 5.0, 10.0):
            center, velocity, count, rms = shrinking_core_state(fp, fv, fm, radius, 100)
            result["stellar_centers"][str(int(radius))] = {
                "radius_kpc": radius, "position_kpc": center.tolist(), "core_velocity_kms": velocity.tolist(),
                "nstar": count, "rms_kpc": rms, "bh_minus_stellar_kpc": (bp * length - center).tolist(),
                "fixed_mw_velocity_10kpc": frozen_aperture_velocity(fp, fv, fm, center)}
        for sat, radius in (("lmc", 8.0), ("smc", 3.0)):
            mask = np.isin(keys, reference["sat_keys"][sat])
            require(mask.sum() == len(reference["sat_keys"][sat]), f"{sat}: 固定恒星有缺失")
            center, velocity, count, rms = shrinking_core_state(p[mask], v[mask], m[mask], radius,
                                                              min(40, max(10, int(mask.sum()) // 4)))
            relative = np.r_[center - bp, velocity - hv]
            expected = np.array([float(h[sat + "_" + k]) for k in COORDS])
            error = float(np.max(np.abs((relative - expected) * scale)))
            require(error <= 1e-6, f"{sat}: 原有测量不能复现，最大差 {error}")
            result["satellites"][sat] = {"position_box_kpc": (center * length).tolist(),
                "velocity_box_kms": (velocity * speed).tolist(), "relative_official_6d": (relative * scale).tolist(),
                "nstar": count, "rms_kpc": rms * length, "reproduction_max_abs_physical": error}
        return result


def compare_trials(a, b):
    delta = lambda key: (np.asarray(b[key]) - np.asarray(a[key])).tolist()
    out = {"from": a["trial"], "to": b["trial"], "bh_position_delta_kpc": delta("bh_position_kpc"),
           "bh_velocity_delta_kms": delta("bh_velocity_kms"),
           "official_host_velocity_delta_kms": delta("official_host_velocity_kms"), "apertures": {}}
    for radius in ("2", "5", "10"):
        sa, sb = a["stellar_centers"][radius], b["stellar_centers"][radius]
        dcenter = np.asarray(sb["position_kpc"]) - sa["position_kpc"]
        dvel = np.asarray(sb["fixed_mw_velocity_10kpc"]["velocity_kms"]) - sa["fixed_mw_velocity_10kpc"]["velocity_kms"]
        row = {"stellar_center_delta_kpc": dcenter.tolist(), "stellar_velocity10_delta_kms": dvel.tolist(),
               "bh_minus_stellar_delta_kpc": (np.asarray(sb["bh_minus_stellar_kpc"]) - sa["bh_minus_stellar_kpc"]).tolist(),
               "satellites": {}}
        for sat in ("lmc", "smc"):
            p = np.asarray(b["satellites"][sat]["position_box_kpc"]) - a["satellites"][sat]["position_box_kpc"]
            v = np.asarray(b["satellites"][sat]["velocity_box_kms"]) - a["satellites"][sat]["velocity_box_kms"]
            row["satellites"][sat] = {"box_position_delta_kpc": p.tolist(), "box_velocity_delta_kms": v.tolist(),
                "relative_bh_position_delta_kpc": (p - out["bh_position_delta_kpc"]).tolist(),
                "relative_stellar_position_delta_kpc": (p - dcenter).tolist(),
                "relative_stellar_velocity10_delta_kms": (v - dvel).tolist()}
        out["apertures"][radius] = row
    return out


def audit_pair(history_path, trial_names=("trial10", "trial11")):
    require(len(trial_names) == 2 and trial_names[0] != trial_names[1], "需要两个不同的 trial")
    payload = json.loads(Path(history_path).read_text())
    selected = []
    for name in trial_names:
        matches = [t for t in payload["trials"] if t["trial"] == name]
        require(len(matches) == 1, f"{name}: 记录缺失或重复")
        selected.append(matches[0])
    a, b = selected
    for key in ("model_key", "source_sha256", "source_files", "model_key_inputs", "rotation_matrices", "units", "stage2_start_code", "stage2_target_code"):
        require(a["orbit"][key] == b["orbit"][key], f"两轮 {key} 不一致")
    require(a["provenance"]["measurement_signature"] == b["provenance"]["measurement_signature"], "两轮测量定义不同")
    references = [record_reference(t) for t in selected]
    require(references[0]["bh_id"] == references[1]["bh_id"], "两轮 MW BH 身份不同")
    require(np.array_equal(np.sort(references[0]["mw_keys"]), np.sort(references[1]["mw_keys"])), "两轮固定 MW 恒星身份不同")
    for sat in ("lmc", "smc"):
        require(np.array_equal(np.sort(references[0]["sat_keys"][sat]), np.sort(references[1]["sat_keys"][sat])), "两轮卫星固定恒星身份不同")
    measurements = [inspect_trial(t, ref) for t, ref in zip(selected, references)]
    return {"version": VERSION, "history_json": str(Path(history_path).resolve()), "history_sha256": sha256_file(history_path),
            "notes": ["原始 HDF5 与 CSV 全部只读；没有改变观测目标、shooting 测量或验收标准。",
                      "固定 MW 星群指插入时已经存在的 MW 恒星，不包含之后新形成的恒星。",
                      "原验证器直接把 code 坐标标为 kpc；复现检查将记录和实测两侧均乘同一个审计单位因子。",
                      "卫星原测量使用原有 8/3 code-length 孔径；独立 MW 诊断使用物理 2/5/10 kpc 孔径。",
                      "孔径差异不等于统计误差；BH 与恒星中心的偏移不自动证明数值噪声或重定位错误。",
                      "非周期非宇宙学单文件模型；不折回坐标；不运行 GIZMO。"],
            "trials": measurements, "comparison": compare_trials(*measurements)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-json", required=True, type=Path)
    parser.add_argument("--trials", nargs=2, default=["trial10", "trial11"])
    parser.add_argument("--output", type=Path, default=Path("mw_center_response_10_11.json"))
    args = parser.parse_args()
    require(not args.output.exists(), f"输出已存在，不覆盖：{args.output}")
    print(VERSION + "：只读核对 IC 指纹及终点测量，然后比较 BH 与恒星中心。", flush=True)
    result = audit_pair(args.history_json, args.trials)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    c = result["comparison"]
    print("BH 位置变化 [x,y,z] kpc:", c["bh_position_delta_kpc"])
    print("孔径[kpc]  恒星中心 Δx[kpc]  BH−恒星偏移 Δx[kpc]")
    for radius, row in c["apertures"].items():
        print(f"{radius:>8}  {row['stellar_center_delta_kpc'][0]:>17.7f}  {row['bh_minus_stellar_delta_kpc'][0]:>23.7f}")
    print("若 BH 与多个孔径的恒星中心共同移动，支持宿主整体位移的贡献；若相对偏移改变，需进一步检查，不能直接称为噪声。")
    print("未修改任何 shooting 残差或通过标准。输出：", args.output.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
