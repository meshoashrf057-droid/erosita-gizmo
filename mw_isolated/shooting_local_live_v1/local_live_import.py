#!/usr/bin/env python3
"""Read a bounded trial10 live-response candidate; never optimize or rescale it.

The proposal is an audit record, not a source of authority: all four ICs and
their validator products are re-read before accepting the recorded increment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from local_live_evidence import read_evidence, same_evidence, validate_probe
from local_live_params import assert_base_param_compatible
from local_live_binary import select_binary
from local_live_build_review import validate_review


SCHEMA = "local-live-proposal-v2"
COORDINATE_ORDER = [f"{sat}_{c}" for sat in ("LMC", "SMC")
                    for c in ("x", "y", "z", "vx", "vy", "vz")]
ANCHOR_IC_SHA256 = "914ca086c68e1e5d04d6aff53ee3284cd67092241e330bd5ee3dd6c9224a9579"
COORDINATE_INDICES = (0, 1, 7)
PROBE_DEFINITION = (("trial11", 0), ("trial12", 1), ("trial13", 7))
PROBE_STEP_KPC = 0.5
MAX_STEP_KPC = 0.5
TARGET_MAX = 2.8
MAX_CONDITION = 1000.0


def _fail(message):
    raise ValueError("局部 live 候选校验失败：" + message)


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        _fail(f"{label} 不是有效数值数组：{exc}")
    if result.shape != shape or not np.all(np.isfinite(result)):
        _fail(f"{label} 必须是形状 {shape} 的有限数值数组")
    return result


def _same(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-8):
        _fail(f"{label} 不一致；请勿混用轮次或修改候选")


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        _fail(f"{label} 缺少有效 SHA256")
    return value.lower()


def _file(path, label):
    if not isinstance(path, (str, Path)) or not str(path):
        _fail(f"{label} 缺少路径")
    try:
        result = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"{label} 无法读取：{exc}")
    if not result.is_file():
        _fail(f"{label} 必须是文件")
    return result


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fresh_evidence(record, trial):
    if not isinstance(record, dict) or record.get("trial") != trial:
        _fail(f"{trial} 缺少正确的实测证据")
    if not isinstance(record.get("work_dir"), str) or not record["work_dir"]:
        _fail(f"{trial} 缺少证据工作目录")
    if record.get("validation_subdir") != "magellanic_validation_fixed":
        _fail(f"{trial} 必须使用 original-stars 的 magellanic_validation_fixed 结果")
    fresh = read_evidence(record["work_dir"], trial, record["validation_subdir"])
    same_evidence(record, fresh)
    return fresh


def load_local_live_step(path, base_initial, residual, sigma, base_ic_path,
                         base_ic_sha256, source_residual_path,
                         source_residual_sha256, model_key):
    """Return the already solved 12D increment and t10-only evidence metadata.

    No numerical optimization is performed. The increment is applied once, with
    no alpha factor. The original merger still checks its source snapshots and
    applies this increment to the cumulative initial state of trial10.
    """
    path = _file(path, "候选 JSON")
    try:
        raw = path.read_bytes()
        proposal = json.loads(raw)
    except (OSError, ValueError) as exc:
        _fail(f"无法读取候选 JSON：{exc}")
    if not isinstance(proposal, dict) or proposal.get("schema") != SCHEMA:
        _fail("候选 schema 不匹配")
    if proposal.get("coordinate_order") != COORDINATE_ORDER:
        _fail("12D 坐标顺序不匹配")
    if proposal.get("base_trial") != "trial10":
        _fail("仅允许以 trial10 为差分及应用基准")
    if proposal.get("alpha_applied") is not False:
        _fail("候选必须明确声明 alpha_applied=false；本模式不乘 alpha")
    if proposal.get("target_max") != TARGET_MAX:
        _fail("预测最大归一化残差上限固定为 2.8")
    _same(_array(proposal.get("step_caps_kpc"), (3,), "三坐标限步"),
          np.full(3, MAX_STEP_KPC), "三坐标限步")
    if not isinstance(model_key, str) or not model_key or proposal.get("shooting_model_key") != model_key:
        _fail("shooting_model_key 不一致")
    if not isinstance(proposal.get("solver_report"), dict):
        _fail("缺少求解诊断 solver_report")
    if proposal["solver_report"].get("ready_for_live_validation") is not True:
        _fail("求解结果仅供诊断，不可生成新的 live 候选")

    if proposal.get("history_dependency") is not False:
        _fail("本版候选必须明确 history_dependency=false；不接受旧历史管线的提案")
    if "history_json_path" in proposal or "history_json_sha256" in proposal:
        _fail("本版提案不得依赖旧历史 JSON；请用 t10-only 管线重新生成")
    if proposal.get("param_comparison_policy") != "effective-production-v1":
        _fail("候选缺少 effective-production-v1 参数核对策略；请用配套版本重新生成提案")

    actual_base = _file(base_ic_path, "当前基准 IC")
    actual_residual = _file(source_residual_path, "当前残差 CSV")
    for key, actual, supplied, label in (
        ("base_ic", actual_base, base_ic_sha256, "基准 IC"),
        ("source_residual_csv", actual_residual, source_residual_sha256, "残差 CSV"),
    ):
        if _file(proposal.get(key + "_path"), label) != actual:
            _fail(label + " 路径不一致")
        actual_hash = _file_hash(actual)
        if actual_hash != _hash(supplied, label + " 调用者 SHA256") or actual_hash != _hash(proposal.get(key + "_sha256"), label + " 候选 SHA256"):
            _fail(label + " 内容指纹不一致")
    if _hash(base_ic_sha256, "基准 IC SHA256") != ANCHOR_IC_SHA256:
        _fail("不是已确认的 trial10 基准 IC")

    current_initial = _array(base_initial, (12,), "当前基准初态")
    current_residual = _array(residual, (12,), "当前基准残差")
    current_sigma = _array(sigma, (12,), "当前 sigma")
    if np.any(current_sigma <= 0):
        _fail("sigma 必须全部大于零")
    for key, actual in (("base_initial_12d", current_initial),
                        ("current_residual_12d", current_residual),
                        ("sigma_12d", current_sigma)):
        _same(_array(proposal.get(key), (12,), key), actual, key)

    anchor = _fresh_evidence(proposal.get("anchor_evidence"), "trial10")
    build_review = validate_review(proposal.get("build_review"), anchor)
    # 用本批实际执行的同目录 param 副本核对内容；原路径可能已用于高分辨率配置。
    # 不要求历史文件仍存在，但保留原运行 cwd，避免相对冷却表路径含义改变。
    execution_param = _file(proposal.get("execution_base_param_path"), "实际执行 param")
    execution_param_hash = _file_hash(execution_param)
    if execution_param_hash != _hash(proposal.get("execution_base_param_sha256"), "实际执行 param SHA256"):
        _fail("实际执行 param 已改变；请重新核对本批已冻结的执行文件")
    historical_param = anchor.get("base_param_path")
    if not isinstance(historical_param, str) or not historical_param:
        _fail("trial10 缺少原始 param 路径记录")
    if execution_param.parent != Path(historical_param).expanduser().resolve().parent:
        _fail("实际执行 param 必须与 trial10 原 param 位于同一目录，保持相对路径的运行含义")
    param_compatibility = assert_base_param_compatible(anchor, execution_param)
    recorded_binary = proposal.get("execution_gizmo")
    if not isinstance(recorded_binary, dict):
        _fail("缺少本批 execution_gizmo 可执行文件记录")
    execution_gizmo = select_binary(anchor, _file(recorded_binary.get("path"), "实际执行 GIZMO"),
                                    build_review=build_review)
    if execution_gizmo["sha256"] != _hash(recorded_binary.get("sha256"), "实际执行 GIZMO SHA256"):
        _fail("实际执行 GIZMO 指纹与提案记录不同")
    if _hash(anchor.get("ic_sha256"), "trial10 IC SHA256") != ANCHOR_IC_SHA256:
        _fail("锚点证据不是已确认的 trial10 IC")
    if _file(anchor.get("ic_path"), "锚点 IC") != actual_base:
        _fail("锚点 IC 与应用基准不同")
    if _file(anchor.get("residual_csv_path"), "锚点残差 CSV") != actual_residual:
        _fail("锚点残差与应用残差不同")
    if anchor.get("model_key") != model_key:
        _fail("锚点模型与当前模型不同")
    for key, expected in (("applied_initial_12d", current_initial),
                          ("residual_12d", current_residual),
                          ("sigma_12d", current_sigma)):
        _same(_array(anchor.get(key), (12,), "锚点 " + key), expected, "锚点 " + key)

    records = proposal.get("probe_evidence")
    if not isinstance(records, list) or len(records) != 3 or not all(isinstance(r, dict) for r in records):
        _fail("必须提供 trial11、trial12、trial13 的三条实测证据")
    by_trial = {r.get("trial"): r for r in records}
    if set(by_trial) != {t for t, _ in PROBE_DEFINITION}:
        _fail("实测轮次缺失、重复或错误")
    columns = []
    evidence_hashes = [ANCHOR_IC_SHA256]
    for trial, index in PROBE_DEFINITION:
        probe = _fresh_evidence(by_trial[trial], trial)
        validate_probe(anchor, probe, index, PROBE_STEP_KPC, build_review=build_review)
        if _file(probe.get("base_param_path"), trial + " 实际 param") != execution_param:
            _fail(trial + " 的 param 路径与本批实际执行 param 不同")
        if _hash(probe.get("base_param_sha256"), trial + " 实际 param SHA256") != execution_param_hash:
            _fail(trial + " 的实际 param 指纹与本批执行文件不同")
        if _file(probe.get("gizmo_path"), trial + " 实际 GIZMO") != Path(execution_gizmo["path"]):
            _fail(trial + " 的 GIZMO 路径与本批实际执行文件不同")
        if _hash(probe.get("gizmo_sha256"), trial + " 实际 GIZMO SHA256") != execution_gizmo["sha256"]:
            _fail(trial + " 的 GIZMO 指纹与本批实际执行文件不同")
        # Recheck elementary invariants here as well as in the shared audit.
        expected_step = np.zeros(12)
        expected_step[index] = PROBE_STEP_KPC
        measured = _array(probe.get("residual_12d"), (12,), trial + " 残差")
        _same(_array(probe.get("applied_initial_12d"), (12,), trial + " 初态"),
              current_initial + expected_step, trial + " 同基点初态")
        _same(_array(probe.get("sigma_12d"), (12,), trial + " sigma"), current_sigma, trial + " sigma")
        evidence_hashes.append(_hash(probe.get("ic_sha256"), trial + " IC SHA256"))
        columns.append((measured - current_residual) / current_sigma / PROBE_STEP_KPC)
    if len(set(evidence_hashes)) != 4:
        _fail("四份不同 IC 的 SHA256 不得重复")
    response = np.column_stack(columns)
    _same(_array(proposal.get("measured_response_normalized_12x3"), (12, 3), "实测响应矩阵"),
          response, "重新计算的实测响应矩阵")
    singular = np.linalg.svd(response * MAX_STEP_KPC, compute_uv=False)
    rank = int(np.linalg.matrix_rank(response * MAX_STEP_KPC))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if rank != 3 or not np.isfinite(condition) or condition > MAX_CONDITION:
        _fail("三坐标响应秩不足或病态；不得应用该候选")

    step = _array(proposal.get("proposed_step_12d"), (12,), "候选增量")
    _same(_array(proposal["solver_report"].get("delta"), (3,), "求解器增量"),
          step[list(COORDINATE_INDICES)], "求解器与候选增量")
    mask = np.ones(12, dtype=bool)
    mask[list(COORDINATE_INDICES)] = False
    if np.any(step[mask] != 0):
        _fail("只允许改变 LMC_x、LMC_y、SMC_y；其他位置及全部速度保持不变")
    if np.any(np.abs(step[list(COORDINATE_INDICES)]) > MAX_STEP_KPC + 1e-12):
        _fail("每个位置坐标增量不得超过 0.5 kpc")
    if np.linalg.norm(step[:3]) > np.sqrt(2) * MAX_STEP_KPC + 1e-12:
        _fail("LMC 总位置改变量超出本次信赖范围")
    candidate = current_initial + step
    _same(_array(proposal.get("candidate_initial_12d"), (12,), "候选初态"), candidate, "候选初态")
    predicted = current_residual + current_sigma * (response @ step[list(COORDINATE_INDICES)])
    _same(_array(proposal.get("predicted_residual_12d"), (12,), "预测残差"), predicted, "预测残差")
    current_z = current_residual / current_sigma
    predicted_z = predicted / current_sigma
    _same(_array(proposal["solver_report"].get("projected_normalized_residual"), (12,), "求解器预测"),
          predicted_z, "求解器与重新计算的预测")
    current_max = float(np.max(np.abs(current_z)))
    predicted_max = float(np.max(np.abs(predicted_z)))
    if predicted_max > TARGET_MAX + 1e-9 or predicted_max >= current_max - 1e-10:
        _fail("候选没有在限制步长内将全部预测分量降至 2.8 sigma 以内")

    diagnostics = {
        "method": "local-live-joint", "purpose": "same-anchor-bounded-live-candidate",
        "coordinate_order": COORDINATE_ORDER, "proposal_path": str(path),
        "proposal_sha256": hashlib.sha256(raw).hexdigest(),
        "base_trial": "trial10", "fit_anchor_trial": "trial10", "history_dependency": False,
        "execution_base_param_path": str(execution_param),
        "execution_base_param_sha256": execution_param_hash,
        "execution_param_compatibility": param_compatibility,
        "execution_gizmo": execution_gizmo,
        "build_review": build_review,
        "base_ic_path": str(actual_base), "base_ic_sha256": base_ic_sha256,
        "model_key": model_key, "base_initial_12d": current_initial.tolist(),
        "current_residual_12d": current_residual.tolist(), "residual_12d": current_residual.tolist(),
        "sigma_12d": current_sigma.tolist(), "candidate_initial_12d": candidate.tolist(),
        "predicted_residual_12d": predicted.tolist(),
        "predicted_normalized_residual_12d": predicted_z.tolist(),
        "current_residual_norm": float(np.linalg.norm(current_z)),
        "predicted_residual_norm": float(np.linalg.norm(predicted_z)),
        "current_max_abs_normalized_residual": current_max,
        "predicted_max_abs_normalized_residual": predicted_max,
        "proposed_step_12d": step.tolist(), "step12": step.tolist(), "proposed_step": step.tolist(),
        "applied_block_norms": {
            "LMC_position_kpc": float(np.linalg.norm(step[:3])), "LMC_velocity_kms": 0.0,
            "SMC_position_kpc": float(np.linalg.norm(step[6:9])), "SMC_velocity_kms": 0.0,
        },
        "applied_lambda": 1.0, "alpha_applied": False,
        "alpha_applied_by_importer": False, "importer_increment_scale": 1.0,
        "status": "joint_candidate_requires_live_validation",
        "static_prediction_used_for_acceptance": False,
        "measured_response_normalized_12x3": response.tolist(),
        "response_singular_values_scaled": singular.tolist(), "response_rank": rank,
        "response_condition": condition, "target_max": TARGET_MAX,
        "model_uncertainty": (
            "Three one-sided finite differences measured about trial10. The joint "
            "response, opposite-sided derivatives and repeatability are not measured. "
            "Predicted componentwise passage is not a live GIZMO validation."
        ),
        "proposal": proposal,
    }
    if build_review is not None:
        diagnostics["model_uncertainty"] += (
            " The anchor endpoint is from the original executable and the probes "
            "use a reviewed same-source/options rebuild. Endpoint numerical "
            "equivalence has not been measured; reuse of the old baseline is an "
            "explicit assumption, not an additional live calibration."
        )
        diagnostics["baseline_reuse_assumption"] = "reviewed-rebuild-endpoint-equivalence-unmeasured"
    return step.copy(), diagnostics
