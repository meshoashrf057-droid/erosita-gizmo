"""Bounded one-coordinate perturbations for measuring a live 12D response.

The returned step is in physical kpc / km s^-1, never scaled by alpha or
optimized against the static model. No simulation files are read or written.
"""
from __future__ import annotations

import numpy as np

COORDINATES = tuple(f"{sat}_{axis}" for sat in ("LMC", "SMC")
                    for axis in ("x", "y", "z", "vx", "vy", "vz"))
POSITION_LIMIT_KPC = 2.0
VELOCITY_LIMIT_KMS = 5.0


def make_probe_step(coordinate: str, step: float) -> tuple[np.ndarray, dict]:
    """Return exactly one nonzero entry, enforcing fixed physical bounds."""
    if coordinate not in COORDINATES:
        raise ValueError(f"probe-coordinate 必须为 {', '.join(COORDINATES)} 之一。")
    if isinstance(step, (bool, np.bool_)) or np.ndim(step) != 0:
        raise ValueError("probe-step 必须为有限非零实数。")
    try:
        value = float(step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probe-step 必须为有限非零实数。") from exc
    if not np.isfinite(value) or value == 0:
        raise ValueError("probe-step 必须为有限非零实数。")
    index = COORDINATES.index(coordinate)
    position = index % 6 < 3
    limit = POSITION_LIMIT_KPC if position else VELOCITY_LIMIT_KMS
    unit = "kpc" if position else "km/s"
    if abs(value) > limit:
        raise ValueError(f"live-probe 单坐标限制为 |step| <= {limit:g} {unit}；不会自动缩放。")
    increment = np.zeros(12, dtype=float)
    increment[index] = value
    norms = {}
    for i, sat in enumerate(("LMC", "SMC")):
        norms[sat] = dict(position_kpc=float(np.linalg.norm(increment[6*i:6*i+3])),
                          velocity_kms=float(np.linalg.norm(increment[6*i+3:6*i+6])))
    audit = dict(method="live-probe", purpose="live-response-measurement",
                 probe_coordinate=coordinate, probe_coordinate_index=index,
                 coordinate_order=list(COORDINATES), probe_step=value,
                 probe_step_unit=unit, hard_position_limit_kpc=POSITION_LIMIT_KPC,
                 hard_velocity_limit_kms=VELOCITY_LIMIT_KMS,
                 proposed_step=increment.tolist(), proposed_step_12d=increment.tolist(),
                 applied_step_norms=norms,
                 applied_block_norms=dict(LMC_position_kpc=norms["LMC"]["position_kpc"],
                     LMC_velocity_kms=norms["LMC"]["velocity_kms"],
                     SMC_position_kpc=norms["SMC"]["position_kpc"],
                     SMC_velocity_kms=norms["SMC"]["velocity_kms"]),
                 applied_lambda=1.0,
                 alpha_applied=False, alpha_applied_by_importer=False,
                 importer_increment_scale=1.0, static_prediction_used_for_acceptance=False)
    return increment, audit


def probe_static_diagnostic(base_initial, increment, residual, sigma, forward):
    """Record the static response without accepting/rejecting a live probe.

    The prediction is diagnostic, including when it worsens the residual. A
    finite static result is still needed for compatibility with the response
    exporter; integration failure raises before writing an IC.
    """
    values = []
    for name, value in (("base_initial", base_initial), ("increment", increment),
                        ("residual", residual), ("sigma", sigma)):
        array = np.asarray(value, dtype=float)
        if array.shape != (12,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} 必须为有限 12D 向量。")
        values.append(array)
    base, step, r, sig = values
    if np.any(sig <= 0):
        raise ValueError("sigma 必须全部为正。")
    diagnostic = dict(base_initial_12d=base.tolist(), candidate_initial_12d=(base+step).tolist(),
                      current_residual_12d=r.tolist(), residual_12d=r.tolist(), sigma_12d=sig.tolist(),
                      current_residual_norm=float(np.linalg.norm(r/sig)),
                      predicted_residual_norm=None, predicted_residual_12d=None,
                      model_endpoint_12d=None, proposed_model_endpoint_12d=None,
                      static_diagnostic_status="not_evaluated")
    try:
        fbase = np.asarray(forward(base.copy()), dtype=float)
        if fbase.shape != (12,) or not np.all(np.isfinite(fbase)):
            raise ValueError("静态 base 终点不是有限 12D 向量。")
        diagnostic["model_endpoint_12d"] = fbase.tolist()
        fprobe = np.asarray(forward((base+step).copy()), dtype=float)
        if fprobe.shape != (12,) or not np.all(np.isfinite(fprobe)):
            raise ValueError("静态 probe 终点不是有限 12D 向量。")
        predicted = r + (fprobe-fbase)
        diagnostic.update(proposed_model_endpoint_12d=fprobe.tolist(),
                          predicted_residual_12d=predicted.tolist(),
                          predicted_residual_norm=float(np.linalg.norm(predicted/sig)),
                          static_diagnostic_status="ok")
    except Exception as exc:
        raise ValueError("静态诊断积分失败，无法写出与历史响应导出工具兼容的有限终点；"
                         "未生成 IC。这是元数据完整性检查，并非静态残差下降门槛。"
                         f" 原因: {type(exc).__name__}: {exc}") from exc
    return diagnostic
