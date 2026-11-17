"""Validate and apply an already damped history-shooting proposal exactly once.

The importer checks the live residual/base identity and replays only two static
integrations. It does not rerun the proposal optimizer or rescale its increment.
Static consistency and history-model acceptance are not live convergence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ORDER = [f"{sat}_{name}" for sat in ("LMC", "SMC")
         for name in ("x", "y", "z", "vx", "vy", "vz")]
BLOCK_NAMES = ("LMC_position_kpc", "LMC_velocity_kms",
               "SMC_position_kpc", "SMC_velocity_kms")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _vector(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (12,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: require a finite 12-vector")
    return result.copy()


def _close(actual, expected, label, atol=1e-8):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if (actual.shape != expected.shape or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(expected))
            or not np.allclose(actual, expected, rtol=1e-10, atol=atol)):
        raise ValueError(f"{label}: proposal and current inputs are inconsistent")


def _positive(value, label):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{label}: must be positive and finite")
    return result


def _norms(vector, sigma):
    z = vector / sigma
    return float(np.linalg.norm(z)), float(np.max(np.abs(z)))


def _block_norms(vector):
    return [float(np.linalg.norm(vector[i:i+3])) for i in range(0, 12, 3)]


def _replay_close(actual, stored, label):
    errors = _block_norms(_vector(actual, label) - _vector(stored, label))
    if any(error >= 0.01 for error in errors):
        raise ValueError(f"{label}: static replay differs by >=0.01 kpc or km/s: {errors}")
    return dict(zip(BLOCK_NAMES, errors))


def _better(predicted, current, sigma):
    z, old = predicted / sigma, current / sigma
    norm, maximum = _norms(predicted, sigma)
    old_norm, old_maximum = _norms(current, sigma)
    tolerance = 1e-12 * max(1.0, old_norm)
    failing = np.abs(old) > 3.0
    return bool(norm < old_norm-tolerance and maximum < old_maximum-tolerance
                and np.all(np.abs(z[failing]) < np.abs(old[failing])-tolerance))


def load_proposal_step(path, base_ic, base_initial, residual, sigma, model_key,
                       source_sha256, model_key_inputs, target_time_code,
                       residual_provenance, backend_sha256, forward, alpha,
                       max_position_step_kpc, max_velocity_step_kms):
    """Return the stored increment unchanged, after identity and response checks.

``alpha`` and both caps must match the proposal's generating options; they are
checked, never applied a second time. ``forward`` operates in kpc and km/s.
The creating merger SHA is retained for provenance, not compared with this
new importer: adding this mode intentionally changes the merger's file bytes.
"""
    path = Path(path).expanduser().resolve(strict=True)
    proposal = json.loads(path.read_text(encoding="utf-8"))
    if proposal.get("schema_version") != 1:
        raise ValueError("Unsupported proposal schema_version")
    if proposal.get("purpose") != "candidate_only_not_live_convergence":
        raise ValueError("Unexpected proposal purpose")
    if proposal.get("coordinate_order") != ORDER:
        raise ValueError("Proposal coordinate_order is not LMC r,v then SMC r,v")
    result = proposal.get("result", {})
    if result.get("status") != "candidate_requires_live_validation":
        raise ValueError("Proposal has no accepted candidate requiring live validation")
    selected = result.get("selected")
    if not isinstance(selected, dict) or selected.get("accepted") is not True:
        raise ValueError("Proposal selected candidate is missing or rejected")
    candidates = result.get("candidates", [])
    matches = [item for item in candidates if item.get("id") == selected.get("id")]
    # Producer versions may append this explanatory field to either/both copies.
    selected_core = {k: v for k, v in selected.items() if k != "selection_note"}
    matching_core = ({k: v for k, v in matches[0].items() if k != "selection_note"}
                     if len(matches) == 1 else None)
    if len(matches) != 1 or selected_core != matching_core:
        raise ValueError("Selected candidate does not match its unique candidates entry")
    if selected.get("passes_all_ridge_acceptance_criteria") is not True:
        raise ValueError("Selected candidate did not pass all ridge acceptance criteria")

    base = _vector(base_initial, "base_initial")
    residual = _vector(residual, "residual")
    sigma = _vector(sigma, "sigma")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    actual_base_sha = _sha(Path(base_ic).expanduser().resolve(strict=True))
    if proposal.get("base_ic_sha256") != actual_base_sha:
        raise ValueError("Proposal base_ic_sha256 does not match the actual base IC")
    if proposal.get("model_key") != model_key:
        raise ValueError("Proposal model_key does not match")
    if proposal.get("source_sha256") != _plain(source_sha256):
        raise ValueError("Proposal source_sha256 does not match")
    if proposal.get("model_key_inputs") != _plain(model_key_inputs):
        raise ValueError("Proposal model_key_inputs do not match")
    if proposal.get("code_sha256", {}).get("backend") != backend_sha256:
        raise ValueError("Proposal backend SHA does not match")
    _close(proposal["target_time_code"], target_time_code, "target_time_code", atol=1e-7)
    if not isinstance(residual_provenance, dict) or residual_provenance.get("status") != "verified":
        raise ValueError("A verified residual provenance is required for history-proposal")
    if residual_provenance.get("reference_ic_sha256") != actual_base_sha:
        raise ValueError("Residual provenance base IC SHA does not match")
    if residual_provenance.get("model_key") != model_key:
        raise ValueError("Residual provenance model_key does not match")
    if residual_provenance.get("target_covered") is not True:
        raise ValueError("Residual provenance does not cover the target epoch")
    _close(residual_provenance["target_time_code"], target_time_code,
           "residual target_time_code", atol=1e-7)
    signature = proposal.get("measurement_signature")
    if not signature or signature != residual_provenance.get("measurement_signature"):
        raise ValueError("Proposal and residual measurement_signature differ")
    for key, value in (("base_initial_12d", base), ("live_residual_12d", residual),
                       ("sigma_12d", sigma)):
        _close(_vector(proposal[key], key), value, key)

    stored_diag = result["diagnostics"]
    for key, value in (("base_initial_12d", base), ("residual_12d", residual),
                       ("sigma_12d", sigma)):
        _close(_vector(stored_diag[key], key), value, "diagnostics."+key)
    alpha = _positive(alpha, "alpha")
    if alpha > 1:
        raise ValueError("alpha must be <=1")
    pos_cap = _positive(max_position_step_kpc, "position cap")
    vel_cap = _positive(max_velocity_step_kms, "velocity cap")
    for key, value in (("alpha", alpha), ("max_position_step_kpc", pos_cap),
                       ("max_velocity_step_kms", vel_cap)):
        _close(stored_diag[key], value, key)
    increment = _vector(selected["increment_12d"], "selected increment")
    if not np.any(increment):
        raise ValueError("Selected proposal increment is zero")
    _close(_vector(selected["candidate_initial_12d"], "candidate initial"),
           base+increment, "candidate_initial=base+increment")
    norms = _block_norms(increment)
    if any(n > (pos_cap if i % 2 == 0 else vel_cap)*(1+1e-10)
           for i, n in enumerate(norms)):
        raise ValueError("Stored increment exceeds its position/velocity cap")
    _close(selected["applied_block_norms"], norms, "applied_block_norms")
    scale = _vector(stored_diag["initial_scale_12d"], "initial scale")
    if np.any(scale <= 0):
        raise ValueError("Initial scale must be positive")
    old_norm, old_max = _norms(residual, sigma)
    _close(stored_diag["current_residual_norm"], old_norm, "current norm")
    _close(stored_diag["current_max_abs_normalized"], old_max, "current max")
    failing = np.flatnonzero(np.abs(residual/sigma) > 3.0).tolist()
    if stored_diag["currently_failing_component_indices"] != failing:
        raise ValueError("Stored failing component indices are inconsistent")

    endpoint = _vector(forward(base.copy()), "base static endpoint")
    new_endpoint = _vector(forward(base+increment), "candidate static endpoint")
    prior = residual + new_endpoint - endpoint
    stored_prior = _vector(selected["static_predicted_residual_12d"], "stored static prediction")
    replay_errors = {"static": _replay_close(prior, stored_prior, "static prediction")}
    for key, value in zip(("static_predicted_residual_norm", "static_max_abs_normalized"),
                          _norms(stored_prior, sigma)):
        _close(selected[key], value, key)
    models = stored_diag.get("defect_models", [])
    cross = selected.get("cross_ridge_predictions", [])
    if not models or len(models) != len(cross):
        raise ValueError("Missing or incomplete cross-ridge model predictions")
    ridge_values = [float(model["ridge_relative"]) for model in models]
    if not all(np.isfinite(x) and x >= 0 for x in ridge_values) or len(set(ridge_values)) != len(models):
        raise ValueError("Invalid or duplicate defect-model ridge values")
    replay_predictions, chosen = [], None
    for model in models:
        ridge = float(model["ridge_relative"])
        entries = [item for item in cross if float(item["ridge_relative"]) == ridge]
        if len(entries) != 1:
            raise ValueError("Cross-ridge prediction must match one unique defect model")
        record = entries[0]
        correction = np.asarray(model["correction_scaled_matrix"], dtype=float)
        if correction.shape != (12, 12) or not np.all(np.isfinite(correction)):
            raise ValueError("Defect correction matrix must be finite 12x12")
        predicted = prior + sigma*(correction @ (increment/scale))
        stored_predicted = _vector(record["predicted_residual_12d"], "stored ridge prediction")
        replay_errors[str(ridge)] = _replay_close(predicted, stored_predicted, f"ridge {ridge}")
        for key, value in zip(("predicted_residual_norm", "max_abs_normalized"),
                              _norms(stored_predicted, sigma)):
            _close(record[key], value, f"ridge {ridge} {key}")
        if record.get("acceptance_criteria_met") is not True or not _better(predicted, residual, sigma):
            raise ValueError(f"Replayed ridge {ridge} does not improve norm/max/failing components")
        norm, maximum = _norms(predicted, sigma)
        replay_predictions.append(dict(ridge_relative=ridge,
            predicted_residual_12d=predicted.tolist(), predicted_residual_norm=norm,
            max_abs_normalized=maximum, acceptance_criteria_met=True))
        if ridge == float(selected["ridge_relative"]):
            chosen = predicted
            _close(selected["predicted_residual_12d"], stored_predicted, "selected ridge prediction")
            _close(selected["predicted_residual_norm"], record["predicted_residual_norm"], "selected norm")
            _close(selected["max_abs_normalized"], record["max_abs_normalized"], "selected max")
    if chosen is None:
        raise ValueError("Selected ridge is absent from defect models")
    _close(selected["worst_across_ridge_residual_norm"],
           max(item["predicted_residual_norm"] for item in cross), "worst ridge norm")
    _close(selected["worst_across_ridge_max_abs_normalized"],
           max(item["max_abs_normalized"] for item in cross), "worst ridge max")
    norm, maximum = _norms(chosen, sigma)
    diagnostics = dict(method="history-proposal", status="applied_verified_stored_candidate",
        warning="Already-damped increment applied once. Predictions are not live convergence.",
        base_initial_12d=base.tolist(), residual_12d=residual.tolist(), sigma_12d=sigma.tolist(),
        alpha=alpha, alpha_applied_by_importer=False,
        max_position_step_kpc=pos_cap, max_velocity_step_kms=vel_cap,
        model_endpoint_12d=endpoint.tolist(), proposed_model_endpoint_12d=new_endpoint.tolist(),
        proposed_step_12d=increment.tolist(), applied_block_norms=dict(zip(BLOCK_NAMES, norms)),
        applied_lambda=1.0, importer_increment_scale=1.0,
        predicted_residual_12d=chosen.tolist(), predicted_residual_norm=norm,
        current_residual_norm=old_norm, max_abs_normalized=maximum,
        static_predicted_residual_12d=prior.tolist(),
        static_predicted_residual_norm=_norms(prior, sigma)[0],
        cross_ridge_predictions=replay_predictions, static_replay_errors=replay_errors,
        proposal_path=str(path), proposal_sha256=_sha(path), proposal_candidate_id=selected["id"],
        proposal_base_ic_sha256=actual_base_sha, proposal_report=proposal)
    return increment.copy(), diagnostics
