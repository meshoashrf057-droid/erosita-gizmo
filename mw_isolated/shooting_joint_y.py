#!/usr/bin/env python3
"""读取两条实测 y 响应产生的有界联合候选；只校验，绝不再次优化或乘 alpha。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


COORDINATE_ORDER = [f"{sat}_{c}" for sat in ("LMC", "SMC")
                    for c in ("x", "y", "z", "vx", "vy", "vz")]
SCHEMA = "measured-joint-y-proposal-v1"


def _fail(message):
    raise ValueError("联合 y 候选校验失败：" + message)


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        _fail(f"{label} 不是有效数值数组：{exc}")
    if result.shape != shape or not np.all(np.isfinite(result)):
        _fail(f"{label} 必须为有限数值，形状应为 {shape}")
    return result


def _same(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-8):
        _fail(f"{label} 不一致；请勿混用轮次或修改候选")


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        _fail(f"{label} 缺少有效 SHA256")
    return value.lower()


def _basename(value, label):
    if not isinstance(value, (str, Path)) or not str(value):
        _fail(f"{label} 缺少路径")
    return Path(value).name


def load_joint_y_step(path, base_initial, residual, sigma, base_ic_path,
                      base_ic_sha256, source_residual_path, source_residual_sha256,
                      model_key):
    """返回 ``(固定的 12D 增量, 可写入 AuditJSON 的诊断)``。

    调用者应先运行既有 provenance 检查，再提供实际 IC/CSV 的 SHA256。
    文件路径允许搬迁，但文件名、内容指纹、状态和模型必须一致。
    这里的实测响应是一侧有限差分近似，并不是完整 live Jacobian。
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
        proposal = json.loads(raw)
    except (OSError, ValueError) as exc:
        _fail(f"无法读取候选 JSON：{exc}")
    if not isinstance(proposal, dict) or proposal.get("schema") != SCHEMA:
        _fail("候选 schema 不匹配")
    if proposal.get("coordinate_order", COORDINATE_ORDER) != COORDINATE_ORDER:
        _fail("12D 坐标顺序不匹配")
    if proposal.get("base_trial") != "trial08" or proposal.get("fit_anchor_trial") != "trial06":
        _fail("本候选必须以 trial08 为应用基准，以 trial06 为差分基准")
    if not isinstance(model_key, str) or not model_key or proposal.get("shooting_model_key") != model_key:
        _fail("shooting_model_key 不匹配")
    for proposed_key, live_value, label in (
        ("base_ic_sha256", base_ic_sha256, "基准 IC SHA256"),
        ("source_residual_csv_sha256", source_residual_sha256, "残差 CSV SHA256"),
    ):
        if _hash(proposal.get(proposed_key), label) != _hash(live_value, label):
            _fail(label + " 不匹配")
    for proposed_key, live_value, label in (
        ("base_ic_path", base_ic_path, "基准 IC"),
        ("source_residual_csv_path", source_residual_path, "残差 CSV"),
    ):
        if _basename(proposal.get(proposed_key), label) != _basename(live_value, label):
            _fail(label + " 文件名不匹配")
    anchor_hash = _hash(proposal.get("fit_anchor_base_ic_sha256"), "差分基准 IC SHA256")
    if anchor_hash == _hash(base_ic_sha256, "基准 IC SHA256"):
        _fail("trial06 与 trial08 的 IC 指纹不能相同")

    current_initial = _array(base_initial, (12,), "当前基准初态")
    current_residual = _array(residual, (12,), "当前基准残差")
    current_sigma = _array(sigma, (12,), "当前 sigma")
    if np.any(current_sigma <= 0):
        _fail("sigma 必须全部大于零")
    for key, current in (("base_initial_12d", current_initial),
                         ("current_residual_12d", current_residual),
                         ("sigma_12d", current_sigma)):
        _same(_array(proposal.get(key), (12,), key), current, key)
    anchor_initial = _array(proposal.get("fit_anchor_initial_12d"), (12,), "差分基准初态")
    anchor_residual = _array(proposal.get("fit_anchor_residual_12d"), (12,), "差分基准残差")
    step = _array(proposal.get("proposed_step_12d"), (12,), "候选增量")
    outside_y = np.ones(12, dtype=bool)
    outside_y[[1, 7]] = False
    if np.any(step[outside_y] != 0):
        _fail("只允许修改 LMC_y 与 SMC_y")
    if np.any(np.abs(step[[1, 7]]) > 3.0 + 1e-12):
        _fail("每个 y 位置增量不得超过 3 kpc")

    offsets = _array(proposal.get("total_y_offsets_from_fit_anchor_kpc"), (2,), "相对差分基准的总 y 偏移")
    # 总偏移同样保持在原差分基准附近，而不是通过重复导入绕过限步。
    if np.any(np.abs(offsets) > 3.0 + 1e-12):
        _fail("相对 trial06 的总 y 偏移不得超过 3 kpc")
    expected_candidate = anchor_initial.copy()
    expected_candidate[[1, 7]] += offsets
    candidate_initial = current_initial + step
    _same(candidate_initial, expected_candidate, "候选初态与累计偏移")
    if "candidate_initial_12d" in proposal:
        _same(_array(proposal["candidate_initial_12d"], (12,), "候选初态"),
              candidate_initial, "候选初态")

    evidence = proposal.get("probe_evidence")
    if not isinstance(evidence, list) or len(evidence) != 2:
        _fail("必须提供 trial08 和 trial09 的两条实测证据")
    if not all(isinstance(item, dict) for item in evidence):
        _fail("实测证据格式错误")
    by_trial = {item.get("trial"): item for item in evidence}
    if set(by_trial) != {"trial08", "trial09"}:
        _fail("实测证据轮次缺失、重复或错误")
    columns = []
    probe_hashes = []
    for trial, coordinate, displacement in (("trial08", 1, 1.0), ("trial09", 7, -1.0)):
        item = by_trial[trial]
        probe_hash = _hash(item.get("ic_sha256"), trial + " IC SHA256")
        probe_hashes.append(probe_hash)
        applied = _array(item.get("applied_initial_12d"), (12,), trial + " 初态")
        measured = _array(item.get("residual_12d"), (12,), trial + " 残差")
        source_step = _array(item.get("source_step_12d"), (12,), trial + " 来源增量")
        expected_step = np.zeros(12)
        expected_step[coordinate] = displacement
        _same(source_step, expected_step, trial + " 独立测试增量")
        _same(applied, anchor_initial + expected_step, trial + " 初态来源")
        if trial == "trial08":
            if probe_hash != _hash(base_ic_sha256, "基准 IC SHA256"):
                _fail("trial08 实测证据与应用基准 IC 指纹不匹配")
            _same(applied, current_initial, "trial08 与当前基准初态")
            _same(measured, current_residual, "trial08 与当前基准残差")
        columns.append((measured - anchor_residual) / (current_sigma * displacement))
    if len({anchor_hash, *probe_hashes}) != 3:
        _fail("三个不同 IC 的指纹不能重复")
    response = np.column_stack(columns)
    supplied_response = _array(proposal.get("measured_response_normalized_12x2"),
                               (12, 2), "实测归一化响应矩阵")
    _same(response, supplied_response, "实测响应矩阵（包括 trial09 负步长符号）")
    predicted = anchor_residual + current_sigma * (response @ offsets)
    supplied_prediction = _array(proposal.get("predicted_residual_12d"), (12,), "预测残差")
    _same(predicted, supplied_prediction, "两条实测响应的仿射预测")
    current_z = current_residual / current_sigma
    predicted_z = predicted / current_sigma
    current_norm = float(np.linalg.norm(current_z))
    predicted_norm = float(np.linalg.norm(predicted_z))
    current_max = float(np.max(np.abs(current_z)))
    predicted_max = float(np.max(np.abs(predicted_z)))
    if predicted_norm >= current_norm - 1e-10 or predicted_max >= current_max - 1e-10:
        _fail("候选必须同时降低 trial08 的预测总残差范数与最大绝对分量")

    diagnostics = {
        "method": "measured-joint-y",
        "purpose": "bounded-joint-live-candidate",
        "coordinate_order": COORDINATE_ORDER,
        "proposal_path": str(path.resolve()),
        "proposal_sha256": hashlib.sha256(raw).hexdigest(),
        "base_trial": "trial08", "fit_anchor_trial": "trial06",
        "base_ic_path": str(Path(base_ic_path).expanduser().resolve()),
        "base_ic_sha256": base_ic_sha256,
        "model_key": model_key,
        "base_initial_12d": current_initial.tolist(),
        "current_residual_12d": current_residual.tolist(),
        "residual_12d": current_residual.tolist(),
        "sigma_12d": current_sigma.tolist(),
        "candidate_initial_12d": candidate_initial.tolist(),
        "predicted_residual_12d": predicted.tolist(),
        "predicted_normalized_residual_12d": predicted_z.tolist(),
        "current_normalized_norm": current_norm,
        "predicted_normalized_norm": predicted_norm,
        "current_residual_norm": current_norm,
        "predicted_residual_norm": predicted_norm,
        "current_max_abs_normalized_residual": current_max,
        "predicted_max_abs_normalized_residual": predicted_max,
        "proposed_step_12d": step.tolist(), "step12": step.tolist(),
        "proposed_step": step.tolist(),
        "applied_block_norms": {
            "LMC_position_kpc": float(abs(step[1])), "LMC_velocity_kms": 0.0,
            "SMC_position_kpc": float(abs(step[7])), "SMC_velocity_kms": 0.0,
        },
        "applied_lambda": 1.0,
        "status": "joint_candidate_requires_live_validation",
        "static_prediction_used_for_acceptance": False,
        "alpha_applied": False, "alpha_applied_by_importer": False,
        "importer_increment_scale": 1.0,
        "measured_response_normalized_12x2": response.tolist(),
        "fit_anchor_initial_12d": anchor_initial.tolist(),
        "fit_anchor_residual_12d": anchor_residual.tolist(),
        "total_y_offsets_from_fit_anchor_kpc": offsets.tolist(),
        "model_uncertainty": (
            "One-sided finite-difference affine prediction only; positive SMC y is "
            "extrapolated from a negative 1 kpc probe. Joint nonlinear response and "
            "simulation noise have not been measured. Predicted tolerance passage "
            "is not a live GIZMO validation."
        ),
        "proposal": proposal,
    }
    return step.copy(), diagnostics
