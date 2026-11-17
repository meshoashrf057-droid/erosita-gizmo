#!/usr/bin/env python3
"""Constrained local shooting proposals; predictions are not live convergence.

``J[i,j]`` is d(residual_i / sigma_i)/d(initial_parameter_j), measured
with the same initial anchor, target time, units, and tracer definition.
This module cannot verify those provenance requirements; the caller must.
Only componentwise initial-step bounds are imposed. No satellite-vector
norm bound, observational covariance, or simulation-noise model is implied.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


CONDITION_LIMIT = 1.0e3


def propose_step(z_base, J, step_scales, max_scaled_step=1.0,
                 target_max=2.8, regularization=1.0e-3):
    """Return a JSON-serializable bounded minimax proposal and diagnostics.

    Parameters
    ----------
    z_base : array-like, shape (12,)
        Actual endpoint residuals divided componentwise by their tolerances.
    J : array-like, shape (12, n), 1 <= n <= 12
        Local response per physical initial coordinate, e.g. sigma/kpc.
    step_scales : array-like, shape (n,)
        Positive physical component-step scales. The actual component bounds
        are +/- ``step_scales * max_scaled_step``.
    max_scaled_step, target_max : positive finite float
        Trust-box multiplier and desired maximum normalized residual.
    regularization : nonnegative finite float
        Small residual-norm weight in the secondary minimum-step objective.

    First solve min(t), |z + J delta| <= t within the physical component
    bounds. Then minimize 0.5*||u||^2 + regularization*0.5*||z+A u||^2,
    retaining the first optimum to numerical tolerance, where u is the
    step divided by its actual bound and A = J * actual_bound. This second
    convex problem selects a unique small step among near-minimax solutions.

    Malformed/nonfinite inputs raise ValueError; optimizer failures raise
    RuntimeError. Rank deficiency, condition > 1000, insufficient predicted
    improvement, or failure to reach target return ready=False. A ready
    result only authorizes consideration of a new live validation run; it
    is not evidence that any real endpoint passes its acceptance test.
    """
    z = np.asarray(z_base, dtype=float)
    jac = np.asarray(J, dtype=float)
    scales = np.asarray(step_scales, dtype=float)
    if z.shape != (12,):
        raise ValueError("z_base must have shape (12,), including BOTH satellites")
    if jac.ndim != 2 or jac.shape[0] != 12 or not 1 <= jac.shape[1] <= 12:
        raise ValueError("J must have shape (12, n), with 1 <= n <= 12")
    if scales.shape != (jac.shape[1],):
        raise ValueError("step_scales must contain one value per J column")
    if not all(np.all(np.isfinite(a)) for a in (z, jac, scales)):
        raise ValueError("residuals, J, and step_scales must all be finite")
    if np.any(scales <= 0):
        raise ValueError("step_scales must be strictly positive")
    controls = np.asarray([max_scaled_step, target_max, regularization], dtype=float)
    if not np.all(np.isfinite(controls)):
        raise ValueError("solver controls must be finite")
    max_scaled_step, target_max, regularization = map(float, controls)
    if max_scaled_step <= 0 or target_max <= 0 or regularization < 0:
        raise ValueError("step/target must be positive; regularization nonnegative")

    bound = scales * max_scaled_step
    A = jac * bound[None, :]
    if not np.all(np.isfinite(bound)) or not np.all(np.isfinite(A)):
        raise ValueError("physical step bounds or scaled response overflow")
    n = jac.shape[1]
    singular = np.linalg.svd(A, compute_uv=False)
    rank_tol = np.finfo(float).eps * max(A.shape) * singular[0]
    rank = int(np.count_nonzero(singular > rank_tol))
    condition = float(singular[0] / singular[-1]) if rank == n else None
    base_max = float(np.max(np.abs(z)))
    base_l2 = float(np.linalg.norm(z))

    # [A,-1] [u,t] <= -z and [-A,-1] [u,t] <= z.
    lhs = np.vstack((np.column_stack((A, -np.ones(12))),
                     np.column_stack((-A, -np.ones(12)))))
    rhs = np.concatenate((-z, z))
    objective = np.zeros(n + 1)
    objective[-1] = 1.0
    lp = linprog(objective, A_ub=lhs, b_ub=rhs,
                 bounds=[(-1.0, 1.0)] * n + [(0.0, None)], method="highs")
    if not lp.success or lp.x is None or not np.all(np.isfinite(lp.x)):
        raise RuntimeError(f"minimax optimization failed: {lp.message}")
    optimum = float(lp.x[-1])
    minimax_slack = 1.0e-8 * max(1.0, base_max, optimum)
    allowed = optimum + minimax_slack
    constraint = LinearConstraint(A, -allowed - z, allowed - z)

    def objective2(u):
        pred = z + A @ u
        return 0.5 * (u @ u + regularization * (pred @ pred))

    def gradient2(u):
        return u + regularization * A.T @ (z + A @ u)

    qp = minimize(objective2, lp.x[:-1], jac=gradient2, method="SLSQP",
                  bounds=Bounds(-np.ones(n), np.ones(n)),
                  constraints=[constraint],
                  options={"ftol": 1.0e-12, "maxiter": 1000})
    if not qp.success or not np.all(np.isfinite(qp.x)):
        raise RuntimeError(f"minimum-step refinement failed: {qp.message}")
    u = np.asarray(qp.x, dtype=float)
    delta = bound * u
    pred = z + jac @ delta
    pred_max = float(np.max(np.abs(pred)))
    if np.max(np.abs(u)) > 1.0 + 1.0e-8 or pred_max > allowed + 1.0e-7:
        raise RuntimeError("optimizer returned an infeasible bounded minimax step")

    reasons = []
    if rank < n:
        reasons.append("response_columns_rank_deficient")
    elif condition > CONDITION_LIMIT:
        reasons.append("response_columns_ill_conditioned")
    if base_max <= target_max:
        reasons.append("base_already_within_prediction_target")
    if pred_max >= target_max:
        reasons.append("predicted_max_does_not_reach_target")
    if base_max - pred_max <= 1.0e-8 * max(1.0, base_max):
        reasons.append("no_resolved_predicted_minimax_improvement")
    ready = not reasons
    return {
        "prediction_kind": "local_linear_prediction_not_live_validation",
        "ready_for_live_validation": ready,
        "status": "candidate_for_live_validation" if ready else "diagnostic_only",
        "reasons": reasons,
        "delta": delta.tolist(),
        "normalized_step": u.tolist(),
        "projected_normalized_residual": pred.tolist(),
        "projected_max_abs": pred_max,
        "projected_l2": float(np.linalg.norm(pred)),
        "base_max_abs": base_max,
        "base_l2": base_l2,
        "predicted_minimax_improvement": base_max - pred_max,
        "rank": rank,
        "number_of_columns": n,
        "singular_values": singular.tolist(),
        "rank_tolerance": float(rank_tol),
        "condition_number": condition,
        "condition_limit": CONDITION_LIMIT,
        "minimax_optimum": optimum,
        "minimax_numerical_slack": minimax_slack,
        "bounds_physical": bound.tolist(),
        "step_scales": scales.tolist(),
        "max_scaled_step": max_scaled_step,
        "target_max": target_max,
        "regularization": regularization,
        "componentwise_bounds_only": True,
        "simulation_noise_assessed": False,
    }
