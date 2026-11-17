#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读 shooting 历史：比较已施加初态、静态预测和真实终点。

只使用 Python 标准库。读取已有 orbit.json、残差 CSV、provenance 和 history。
不读取大型 HDF5，不修改 IC、快照、param、残差或目标容差，不生成下一轮 IC。
输出一个小 JSON，供检查停滞分量和后续响应校准使用。
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ORDER = [(s, c) for s in ("LMC", "SMC") for c in ("x", "y", "z", "vx", "vy", "vz")]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(values, label):
    if not isinstance(values, list) or len(values) != 12:
        raise ValueError(f"{label}: 必须是 12 维列表")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label}: 包含非有限值")
    return result


def close(a, b):
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-7) for x, y in zip(a, b))


def difference(a, b):
    return [x-y for x, y in zip(a, b)]


def norm(r, sigma):
    return math.sqrt(sum((x/s)**2 for x, s in zip(r, sigma)))


def read_trial(root, trial):
    stem = f"mw_lmc_smc_relaxed_{trial}_000"
    orbit_path = root / (stem + ".orbit.json")
    folder = root / (stem + "_run") / "production/output/magellanic_validation_fixed"
    csv_path = folder / "magellanic_present_6d.csv"
    provenance_path = csv_path.with_suffix(".provenance.json")
    orbit = json.loads(orbit_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if orbit.get("trial_label") != trial or provenance.get("trial_label") != trial:
        raise ValueError(f"{trial}: trial_label 不匹配；检查是否把另一轮结果写入本轮目录")
    if provenance.get("schema_version") != 1:
        raise ValueError(f"{trial}: 不支持的 provenance schema")
    if provenance.get("residual_csv_sha256") != sha256(csv_path):
        raise ValueError(f"{trial}: CSV 校验值与 provenance 不匹配")
    if Path(provenance.get("reference_ic", "")).name != stem + ".hdf5":
        raise ValueError(f"{trial}: reference IC 文件名与本轮不同")
    if not provenance.get("reference_ic_sha256"):
        raise ValueError(f"{trial}: provenance 缺少 reference IC 指纹")
    if orbit.get("output_ic_sha256") != provenance["reference_ic_sha256"]:
        raise ValueError(f"{trial}: orbit 记录的 IC 指纹与 provenance 不一致")
    if provenance.get("model_key") != orbit.get("model_key") or not orbit.get("model_key"):
        raise ValueError(f"{trial}: orbit / provenance 模型标识不一致")
    if provenance.get("target_covered") is not True or provenance.get("shooting_usable") is not True:
        raise ValueError(f"{trial}: 未覆盖目标时刻，不能当作有效 shooting 终点")
    if provenance.get("stellar_tracer_policy") != "original-stars":
        raise ValueError(f"{trial}: 需要 original-stars 残差")
    if not provenance.get("measurement_signature"):
        raise ValueError(f"{trial}: 缺少测量定义标识")
    if not math.isclose(float(orbit["stage2_target_code"]), float(provenance["target_time_code"]), abs_tol=1e-7, rel_tol=0):
        raise ValueError(f"{trial}: 目标时刻不一致")
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keyed = {(r["satellite"].upper(), r["component"].lower()): r for r in rows}
    if len(rows) != 12 or set(keyed) != set(ORDER):
        raise ValueError(f"{trial}: CSV 需要恰好 12 个不重复的卫星/分量")
    columns = {k: vector([keyed[q][k] for q in ORDER], f"{trial}/{k}")
               for k in ("simulated", "observed", "residual", "sigma", "zscore")}
    if any(s <= 0 for s in columns["sigma"]):
        raise ValueError(f"{trial}: sigma 必须为正")
    if not close(columns["residual"], difference(columns["simulated"], columns["observed"])):
        raise ValueError(f"{trial}: residual != simulated-observed")
    if not close(columns["zscore"], [r/s for r, s in zip(columns["residual"], columns["sigma"])]):
        raise ValueError(f"{trial}: zscore != residual/sigma")
    for i, q in enumerate(ORDER):
        expected_unit = "kpc" if i % 6 < 3 else "km/s"
        if keyed[q]["unit"] != expected_unit:
            raise ValueError(f"{trial}: {q} 单位不是 {expected_unit}")
    initial = vector(orbit["applied_initial_12d"], f"{trial}/applied_initial")
    nominal = vector(orbit["nominal_initial_12d"], f"{trial}/nominal_initial")
    offset = vector(orbit["cumulative_offset_12d"], f"{trial}/offset")
    if not close(initial, [x+d for x, d in zip(nominal, offset)]):
        raise ValueError(f"{trial}: applied != nominal + cumulative_offset")
    history_path = folder / "magellanic_6d_history.csv"
    history = []
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8-sig") as f:
            history = list(csv.DictReader(f))
    meta_keys = ("trial_label", "model_key", "merger_revision", "output_ic_sha256", "source_sha256", "source_files",
                 "source_centers_kpc", "source_bulk_kms", "rotation_matrices", "units",
                 "stage2_start_code", "stage2_target_code", "lookback_gyr", "model_key_inputs",
                 "applied_initial_12d", "nominal_initial_12d", "cumulative_offset_12d",
                 "previous_offset_12d", "this_iteration_increment_12d", "manual_increment_12d",
                 "shooting_base_ic", "shooting_alpha", "shooting_diagnostics",
                 "shooting_residual_provenance")
    return dict(trial=trial, orbit_file=str(orbit_path), orbit_sha256=sha256(orbit_path),
                residual_csv=str(csv_path), provenance=provenance,
                orbit={k: orbit.get(k) for k in meta_keys}, **columns,
                normalized_l2=norm(columns["residual"], columns["sigma"]),
                history=history)


def compare(previous, current):
    a, b = previous["orbit"], current["orbit"]
    if Path(b.get("shooting_base_ic") or "").name != f"mw_lmc_smc_relaxed_{previous['trial']}_000.hdf5":
        raise ValueError(f"{current['trial']}: 实际 shooting base 不是 {previous['trial']}")
    dx = difference(b["applied_initial_12d"], a["applied_initial_12d"])
    if not close(dx, vector(b["this_iteration_increment_12d"], "increment")):
        raise ValueError(f"{current['trial']}: 初态差与记录的本轮增量不一致")
    source = b.get("shooting_residual_provenance") or {}
    for key in ("residual_csv_sha256", "reference_ic_sha256", "measurement_signature"):
        if source.get(key) != previous["provenance"].get(key):
            raise ValueError(f"{current['trial']}: 修正所用旧残差的 {key} 与当前 {previous['trial']} 不符")
    diag = b.get("shooting_diagnostics") or {}
    predicted = vector(diag.get("predicted_residual_12d"), "predicted_residual_12d")
    dr = difference(current["residual"], previous["residual"])
    dp = difference(predicted, previous["residual"])
    sigma = current["sigma"]
    return {
        "from": previous["trial"], "to": current["trial"],
        "initial_step_12d": dx, "actual_residual_change_12d": dr,
        "predicted_residual_12d": predicted, "predicted_residual_change_12d": dp,
        "actual_minus_predicted_12d": difference(current["residual"], predicted),
        "previous_norm": previous["normalized_l2"], "actual_norm": current["normalized_l2"],
        "predicted_norm": norm(predicted, sigma),
        "alpha": b.get("shooting_alpha"), "step_factor": diag.get("applied_lambda"),
        "components": [{"satellite": q[0], "component": q[1],
                        "predicted_delta_sigma": dp[i]/sigma[i],
                        "actual_delta_sigma": dr[i]/sigma[i],
                        "prediction_error_sigma": (dr[i]-dp[i])/sigma[i]}
                       for i, q in enumerate(ORDER)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--trials", nargs="+", default=["trial04", "trial05", "trial06"])
    p.add_argument("--output", type=Path, default=Path("shooting_response_audit_04_06.json"))
    args = p.parse_args()
    if len(args.trials) < 2 or len(set(args.trials)) != len(args.trials):
        p.error("至少给两轮、按实际先后排列且不重复")
    if args.output.suffix.lower() != ".json" or args.output.exists():
        p.error("输出必须是尚不存在的新 JSON 文件，避免覆盖已有文件")
    trials = [read_trial(args.root.resolve(), name) for name in args.trials]
    baseline = trials[0]
    for t in trials[1:]:
        for key in ("model_key", "measurement_signature", "target_time_code", "stellar_tracer_policy"):
            if t["provenance"][key] != baseline["provenance"][key]:
                raise ValueError(f"{t['trial']}: {key} 改变，不能直接比较")
        if not close(t["observed"], baseline["observed"]) or not close(t["sigma"], baseline["sigma"]):
            raise ValueError(f"{t['trial']}: 目标或容差改变")
    steps = [compare(a, b) for a, b in zip(trials, trials[1:])]
    result = dict(schema_version=1,
        coordinate_order=[f"{s}_{c}" for s, c in ORDER],
        notes=["源文件全部只读；没有生成修正 IC。",
               "只核对小文件及记录的 IC 指纹，未重新读取 HDF5 校验其字节。",
               "若干实测方向不能唯一确定完整 12x12 live Jacobian。",
               "各轮 history 为原 CSV 字符串记录，方便保留完整数值。"],
        trials=trials, steps=steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print("小文件一致性检查通过（未重新读取 HDF5）。")
    print("trial       ||r/sigma||      LMC y/sigma      SMC y/sigma     relative_y_error[kpc]")
    for t in trials:
        print(f"{t['trial']:10s} {t['normalized_l2']:11.6f} {t['zscore'][1]:16.6f}"
              f" {t['zscore'][7]:16.6f} {t['residual'][7]-t['residual'][1]:24.6f}")
    for step in steps:
        print(f"\n{step['from']} -> {step['to']}：预测/实际 y 残差变化（sigma）")
        for i in (1, 7):
            c = step["components"][i]
            print(f"  {c['satellite']}: predicted={c['predicted_delta_sigma']:+.6f},"
                  f" actual={c['actual_delta_sigma']:+.6f},"
                  f" mismatch={c['prediction_error_sigma']:+.6f}")
    print(f"\n导出完成：{args.output.resolve()}\n请提供此 JSON；不需要上传大型快照。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
