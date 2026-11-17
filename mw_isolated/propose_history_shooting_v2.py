#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从当前静态势 + 已有 live 历史，计算受限的 shooting 候选（不写 IC）。

与现有 merger 并用；不替换 backend、merger、validator 或物理参数。
允许以历史中已测残差最小的一轮为基准，同时保留其后拒绝步的响应。
历史保持原顺序，按文件名和 SHA 核对每一步的实际父轮次。
只读三个源快照和 base IC。只写一个新的 proposal JSON，不启动 GIZMO。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from history_response_v2 import propose_candidates

VERSION = "history-response-proposal-v2.0-best-anchor / 2026-10-14"
ORDER = [f"{s}_{c}" for s in ("LMC", "SMC")
         for c in ("x", "y", "z", "vx", "vy", "vz")]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def vec(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (12,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 应为有限 12D 向量")
    return a


def same(a, b, name, atol=1e-7):
    if not np.allclose(a, b, atol=atol, rtol=1e-9):
        raise ValueError(f"{name} 不一致，不能混用不同轮次/模型的响应")


def resolve_parent_index(trials, index):
    """用生成时保存的 base 文件名和 residual provenance SHA 定位更早父轮次。"""
    orbit = trials[index]["orbit"]
    source = orbit.get("shooting_residual_provenance") or {}
    base_name = Path(orbit["shooting_base_ic"]).name
    base_sha = source.get("reference_ic_sha256")
    matches = [i for i in range(index)
               if Path(trials[i]["provenance"]["reference_ic"]).name == base_name
               and trials[i]["provenance"]["reference_ic_sha256"] == base_sha]
    if len(matches) != 1:
        raise ValueError("历史 base IC 文件名及 SHA256 必须唯一匹配更早父轮次；不能重排历史")
    return matches[0]


def validate_history(payload):
    """重新计算关键关系，不直接相信导出器预计算的 steps/norm。"""
    if payload.get("schema_version") != 1 or payload.get("coordinate_order") != ORDER:
        raise ValueError("history schema / 12D 顺序不符")
    trials = payload.get("trials", [])
    if len(trials) < 3:
        raise ValueError("需要至少三轮，即两个实测初态方向")
    if len({t["trial"] for t in trials}) != len(trials):
        raise ValueError("历史 trial 重复")
    first = trials[0]
    sigma = vec(first["sigma"], "sigma")
    if np.any(sigma <= 0):
        raise ValueError("sigma 必须为正")
    observed = vec(first["observed"], "observed")
    initials, residuals = [], []
    for i, t in enumerate(trials):
        o, p = t["orbit"], t["provenance"]
        if t["trial"] != o["trial_label"] or t["trial"] != p["trial_label"]:
            raise ValueError("trial 标签错配")
        for key in ("model_key", "source_sha256", "rotation_matrices", "units",
                    "stage2_start_code", "stage2_target_code", "model_key_inputs"):
            if o[key] != first["orbit"][key]:
                raise ValueError(f"历史 {key} 改变")
        for key in ("model_key", "measurement_signature", "target_time_code", "stellar_tracer_policy"):
            if p[key] != first["provenance"][key]:
                raise ValueError(f"历史测量 {key} 改变")
        if p.get("schema_version") != 1 or p.get("stellar_tracer_policy") != "original-stars":
            raise ValueError("需要 schema=1、original-stars 的 provenance")
        if not p.get("measurement_signature") or p.get("model_key") != o["model_key"]:
            raise ValueError("历史测量定义/模型标识缺失或不匹配")
        if p.get("target_covered") is not True or p.get("shooting_usable") is not True:
            raise ValueError("历史中有未覆盖目标时刻的结果")
        same(p["target_time_code"], o["stage2_target_code"], "目标时间")
        if not o.get("output_ic_sha256") or o["output_ic_sha256"] != p.get("reference_ic_sha256"):
            raise ValueError("历史 orbit / provenance 的 IC 指纹不同")
        same(vec(t["sigma"], "sigma"), sigma, "sigma")
        same(vec(t["observed"], "observed"), observed, "观测目标")
        r = vec(t["residual"], "residual")
        same(r, vec(t["simulated"], "simulated")-observed, "simulated-observed")
        same(vec(t["zscore"], "zscore"), r/sigma, "zscore")
        x = vec(o["applied_initial_12d"], "applied initial")
        same(x, vec(o["nominal_initial_12d"], "nominal") + vec(o["cumulative_offset_12d"], "offset"), "初态累计")
        if i:
            parent_index = resolve_parent_index(trials, i)
            previous = trials[parent_index]
            q = previous["orbit"]
            same(o["previous_offset_12d"], q["cumulative_offset_12d"], "上一轮累计 offset")
            same(x-initials[parent_index], vec(o["this_iteration_increment_12d"], "increment")
                 + vec(o["manual_increment_12d"], "manual"), "实际增量")
            source = o.get("shooting_residual_provenance") or {}
            for key in ("residual_csv_sha256", "reference_ic_sha256", "measurement_signature"):
                if source.get(key) != previous["provenance"].get(key):
                    raise ValueError(f"历史修正来源 {key} 错配")
            diag = o.get("shooting_diagnostics") or {}
            same(vec(diag["base_initial_12d"], "diag base"), initials[parent_index], "诊断 base")
            same(vec(diag["residual_12d"], "diag residual"), residuals[parent_index], "诊断输入残差")
            same(vec(diag["sigma_12d"], "diag sigma"), sigma, "诊断 sigma")
        initials.append(x)
        residuals.append(r)
    return np.asarray(initials), np.asarray(residuals), sigma, observed


def select_anchor(payload, base_path, base_sha256, residuals, sigma):
    """按实际文件名和 SHA 选择历史锚点；拒绝从更差的已测轮次继续。"""
    trials = payload["trials"]
    matches = [i for i, t in enumerate(trials)
               if Path(t["provenance"]["reference_ic"]).name == Path(base_path).name
               and t["provenance"]["reference_ic_sha256"] == base_sha256]
    if len(matches) != 1:
        raise ValueError("base IC 文件名及 SHA256 必须唯一匹配历史中的同一轮")
    index = matches[0]
    norms = np.linalg.norm(np.asarray(residuals) / sigma, axis=1)
    best_norm = float(np.min(norms))
    tolerance = 1e-10 * max(1.0, best_norm)
    best_indices = [i for i, n in enumerate(norms) if n <= best_norm+tolerance]
    if index not in best_indices:
        labels = ", ".join(trials[i]["trial"] for i in best_indices)
        raise ValueError(f"当前 base 不是已有历史中残差范数最小的轮次；请使用 {labels}，保留其他轮次作为响应测量")
    report = dict(anchor_history_index=index, anchor_trial=trials[index]["trial"],
                  best_trial_indices=best_indices,
                  best_trials=[trials[i]["trial"] for i in best_indices],
                  best_residual_norm=best_norm,
                  history_trial_labels=[t["trial"] for t in trials],
                  history_residual_norms=norms.tolist(),
                  history_order_preserved=True,
                  note="按相同测量定义下已测残差范数选基准；这是确定性选择规则，不估计随机噪声或保证下一轮改善。")
    return index, report


def evaluate_live_steps(payload, residuals, sigma):
    """比较保存的预报与实测，不把 surrogate 接受误称为 live 接受。"""
    trials = payload["trials"]
    evaluations = []
    for i in range(1, len(trials)):
        parent_index = resolve_parent_index(trials, i)
        previous, current = trials[parent_index], trials[i]
        diagnostics = current["orbit"]["shooting_diagnostics"]
        method = diagnostics.get("method")
        is_probe = bool(diagnostics.get("is_measurement_probe", False)) or "probe" in str(method or "").lower()
        predicted = vec(diagnostics["predicted_residual_12d"], "历史预测残差") / sigma
        old_z, actual_z = residuals[parent_index]/sigma, residuals[i]/sigma
        loss_before = float(0.5*np.dot(old_z, old_z))
        loss_predicted = float(0.5*np.dot(predicted, predicted))
        loss_actual = float(0.5*np.dot(actual_z, actual_z))
        predicted_reduction = loss_before-loss_predicted
        actual_reduction = loss_before-loss_actual
        denominator_floor = 1e-12*max(1.0, loss_before)
        ratio = actual_reduction/predicted_reduction if predicted_reduction > denominator_floor else None
        evaluations.append(dict(
            base_trial=previous["trial"], base_history_index=parent_index,
            evaluated_trial=current["trial"], evaluated_history_index=i,
            method=method, is_measurement_probe=is_probe,
            outcome_role="measurement_outcome_not_descent_acceptance" if is_probe else "shooting_step_improvement_check",
            loss_definition="0.5 * sum((residual/sigma)^2)",
            loss_before=loss_before, predicted_loss=loss_predicted, actual_loss=loss_actual,
            predicted_loss_reduction=predicted_reduction,
            actual_loss_reduction=actual_reduction,
            actual_to_predicted_reduction_ratio=ratio,
            actual_loss_improved=bool(actual_reduction > 0),
            accepted_as_improvement=None if is_probe else bool(actual_reduction > 0),
            note=("受控探针用于测量响应；损失增减是测量结果，不用于拒绝探针，也未承诺下降。"
                  if is_probe else "按已测损失下降的启发式接受判断；未估计数值/随机噪声，不是置信度或成功概率。")))
    return evaluations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-dir", type=Path, default=Path("."), help="含现有 merger/backend 的目录")
    p.add_argument("--base-ic", type=Path, required=True)
    p.add_argument("--base-param", type=Path, required=True)
    p.add_argument("--history-json", type=Path, required=True)
    p.add_argument("--residual-csv", type=Path, help="默认 base IC 配套 production/output/magellanic_validation_fixed")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--max-position-step-kpc", type=float, default=2.)
    p.add_argument("--max-velocity-step-kms", type=float, default=5.)
    p.add_argument("--output", type=Path, default=Path("trial08_shooting_proposal.json"))
    args = p.parse_args()
    print(f"脚本版本：{VERSION}", flush=True)
    if args.output.exists() or args.output.suffix.lower() != ".json":
        p.error("输出必须为尚不存在的 JSON，以免覆盖已有结果")
    base = args.base_ic.expanduser().resolve(strict=True)
    history_path = args.history_json.expanduser().resolve(strict=True)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    initials, residuals, sigma, observed = validate_history(payload)
    print("核对 base IC 与历史指纹；全部输入只读。", flush=True)
    base_hash = digest(base)
    anchor_index, selection = select_anchor(payload, base, base_hash, residuals, sigma)
    anchor = payload["trials"][anchor_index]
    live_step_evaluation = evaluate_live_steps(payload, residuals, sigma)
    print(f"基准：{anchor['trial']}（历史索引 {anchor_index}）；保留全部 {len(initials)} 轮原顺序响应。", flush=True)
    for step in live_step_evaluation:
        ratio = step["actual_to_predicted_reduction_ratio"]
        ratio_text = "N/A" if ratio is None else f"{ratio:.6f}"
        role = "探针测量结果" if step["is_measurement_probe"] else "shooting 步"
        print(f"  {step['base_trial']} → {step['evaluated_trial']} [{role}]: 实测损失变化收益={step['actual_loss_reduction']:.6f}, 预测收益比={ratio_text}, 损失改善={step['actual_loss_improved']}", flush=True)
    pipeline = args.pipeline_dir.expanduser().resolve(strict=True)
    sys.path.insert(0, str(pipeline))
    import merge_relaxed_mw_lmc_smc as m
    from shooting_local import forward_state
    audit = m.read_merge_audit(base)
    for key in ("model_key", "source_sha256", "rotation_matrices", "model_key_inputs"):
        if audit[key] != anchor["orbit"][key]:
            raise ValueError(f"本地 base IC 的 {key} 与导出历史不同")
    same(audit["applied_initial_12d"], initials[anchor_index], "本地 base 初态")
    key = audit["model_key_inputs"]
    if digest(m.b.__file__) != key["backend_sha256"]:
        raise ValueError("tg19_backend.py 已变化；请使用产生这些历史 IC 的原后端")
    targets = np.asarray(key["targets"], dtype=float).reshape(12)
    same(targets, observed, "base IC 目标")
    csv_path = args.residual_csv or base.with_suffix("").with_name(base.stem+"_run") / "production/output/magellanic_validation_fixed/magellanic_present_6d.csv"
    live, provenance = m.checked_shooting_residual(str(csv_path), str(base), audit["model_key"],
        targets.reshape(4, 3), sigma, audit["stage2_target_code"], require_provenance=True)
    same(live, residuals[anchor_index], "本地基准 residual")
    for key_name in ("residual_csv_sha256", "measurement_signature"):
        if provenance[key_name] != anchor["provenance"][key_name]:
            raise ValueError(f"本地基准 residual 的 {key_name} 与导出不同")
    _, _, values = m.read_param(args.base_param)
    raw_units, units = m.param_units(values)
    for unit, value in units.items():
        same(value, audit["units"][unit], f"单位 {unit}", atol=1e-10)
    sources = []
    for path, expected in zip(audit["source_files"], audit["source_sha256"]):
        path = Path(path).expanduser().resolve(strict=True)
        print(f"读取原弛豫末态：{path}", flush=True)
        if digest(path) != expected:
            raise ValueError(f"源快照 SHA256 改变：{path}")
        sources.append(m.read_snapshot(path, raw_units))
    if len(sources) != 3:
        raise ValueError("需要 MW/LMC/SMC 三个源快照")
    centers = np.asarray(audit["source_centers_kpc"], dtype=float)
    host = m.host_profile(sources[0], centers[0], units)
    satellites = [m.MeasuredSatellite(s, c, units) for s, c in zip(sources[1:], centers[1:])]
    kwargs = dict(profile=host, lmc=satellites[0], smc=satellites[1],
        lookback_gyr=audit["lookback_gyr"], use_dynamical_friction=bool(key["df"]),
        include_host_recoil=bool(key["recoil"]), coulomb_log=key["coulomb_log"])
    del sources
    cache = {}
    def forward(x):
        state = vec(x, "forward input")
        fingerprint = state.tobytes()
        if fingerprint not in cache:
            n = len(cache)+1
            print(f"  静态积分 {n}（不是 GIZMO 演化）", flush=True)
            cache[fingerprint] = forward_state(state, kwargs)
        return cache[fingerprint].copy()
    # 与历史保存的静态 endpoint 对照，防止意外重建了另一个质量模型。
    closure = []
    for i, trial in enumerate(payload["trials"]):
        predicted = forward(initials[i])
        old_endpoint = vec(trial["orbit"]["shooting_diagnostics"]["proposed_model_endpoint_12d"], "stored model endpoint")
        error = predicted-old_endpoint
        block_norms = np.linalg.norm(error.reshape(4, 3), axis=1)
        if np.max(block_norms) > .01:
            raise ValueError(f"重建静态模型与 {trial['trial']} 保存的终态差异过大：{block_norms}")
        closure.append(dict(trial=trial["trial"], block_norms=block_norms.tolist()))
    print("计算完整静态差分响应，并校准历史已测方向；其余方向仍依赖静态先验。", flush=True)
    result = propose_candidates(initials[anchor_index], live, sigma, initials, residuals, forward,
        alpha=args.alpha, max_position_step=args.max_position_step_kpc,
        max_velocity_step=args.max_velocity_step_kms, anchor_index=anchor_index)
    report = dict(schema_version=1, purpose="candidate_only_not_live_convergence",
        script_version=VERSION, python_version=sys.version, numpy_version=np.__version__,
        code_sha256={"proposal_cli": digest(__file__),
                     "history_response": digest(Path(__file__).with_name("history_response_v2.py")),
                     "merger": digest(m.__file__),
                     "backend": digest(m.b.__file__)},
        base_ic=str(base), base_ic_sha256=base_hash, model_key=audit["model_key"],
        history_json=str(history_path), history_json_sha256=digest(history_path),
        source_sha256=audit["source_sha256"], model_key_inputs=key,
        target_time_code=audit["stage2_target_code"], coordinate_order=ORDER,
        measurement_signature=provenance["measurement_signature"],
        base_initial_12d=initials[anchor_index].tolist(), live_residual_12d=live.tolist(),
        anchor_selection=selection, live_history_evaluation=live_step_evaluation,
        sigma_12d=sigma.tolist(), static_reconstruction_check=closure,
        static_integrations=len(cache), result=result,
        warning="只校准历史覆盖的方向；候选包含未测方向。所有预测均不是 live 收敛证明。未写 IC，未启动 GIZMO。")
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(f"\n完成：{args.output.resolve()}")
    selected = result.get("selected")
    if selected:
        print("可供审查的候选：")
        step = vec(selected["increment_12d"], "selected step").reshape(4, 3)
        for label, value in zip(("LMC dpos kpc", "LMC dvel km/s", "SMC dpos kpc", "SMC dvel km/s"), step):
            print(f"  {label}: {value}")
        print(f"  校准模型预测 norm={selected['predicted_residual_norm']:.6f}, max|r/sigma|={selected['max_abs_normalized']:.6f}")
        print(f"  相同候选的原静态预测 norm={selected['static_predicted_residual_norm']:.6f}")
        print(f"  不同历史正则化下最差预测 norm={selected['worst_across_ridge_residual_norm']:.6f}")
        predicted_z = vec(selected["predicted_residual_12d"], "predicted residual")/sigma
        print(f"  LMC 预测 r/sigma={predicted_z[:6]}")
        print(f"  SMC 预测 r/sigma={predicted_z[6:]}")
        print(f"  初态方向落在历史子空间外的比例={selected['outside_history_fraction']:.3f}")
        if not selected["passes_all_ridge_acceptance_criteria"]:
            print("  注意：有正则化网格不支持全部预测改善条件，先审查方向敏感性。")
    else:
        print("本次没有通过预测条件的候选；保留报告检查，不生成不合格 IC。")
    print("请提供本 JSON，先核对不同正则化下方向是否稳定，再安排 live 校准。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
