"""Static response plus regularized, history-constrained defect calibration.

This module does not generate ICs or claim live convergence. A finite set of
live secants constrains only their sampled subspace; other directions retain
the static-model prior. Dependencies: NumPy only.
"""
from __future__ import annotations

import numpy as np

INITIAL_SCALE = np.tile([20.0, 20.0, 20.0, 30.0, 30.0, 30.0], 2)
RIDGE_RELATIVE = (0.001, 0.01, 0.1)
DAMPING_RELATIVE = (0.001, 0.01, 0.1)
FD_PHYSICAL_EPSILON = 0.02


def _vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (12,) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite array with shape (12,)")
    return x.copy()


def _history(value, name):
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or x.shape[1] != 12 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite array with shape (N,12), N >= 3")
    return x.copy()


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _norms(z):
    norm = float(np.linalg.norm(z))
    maximum = float(np.max(np.abs(z)))
    if not np.isfinite(norm) or not np.isfinite(maximum):
        raise ValueError("Nonfinite normalized residual norm")
    return norm, maximum


def _block_norms(x):
    return [float(np.linalg.norm(x[i:i+3])) for i in range(0, 12, 3)]


def _better(predicted_z, current_z, failing):
    norm, maximum = _norms(predicted_z)
    old_norm, old_max = _norms(current_z)
    tol = 1e-12 * max(1.0, old_norm)
    return bool(norm < old_norm-tol and maximum < old_max-tol and
                np.all(np.abs(predicted_z[failing]) < np.abs(current_z[failing])-tol))


def propose_candidates(base, residual, sigma, history_initials, history_residuals,
                       forward, alpha=0.3, max_position_step=2.0,
                       max_velocity_step=5.0, anchor_index=None):
    """Return a JSON-ready report of bounded proposals, never a new IC.

All physical vectors use [LMC position kpc, LMC velocity km/s, SMC position,
SMC velocity]. History rows must retain their provenance order. The anchor
may be any row matching ``base`` / ``residual``; it must not be moved to the
end of the history. If ``anchor_index`` is omitted a unique matching row is
required. ``forward`` accepts/returns physical 12-vectors
and evaluates the *static* orbit model. The caller must establish consistent
particle inputs, target, time, units, and tracer definition across the history.

For scaled history directions U and scaled live-minus-static endpoint changes
E, C = E @ regularized_pinv(U). Each singular reciprocal is
s/(s**2 + (ridge_relative*smax(U))**2). Current scaled static Jacobian A is
obtained by central physical differences of 0.02 kpc or 0.02 km/s. A damped
least-squares solve uses G=A+C and damping=damping_relative*smax(G).

Acceptance is only a surrogate prediction: total normalized L2 and maximum
absolute residual must decrease, as must every currently >3-sigma component.
Selection minimizes worst predicted L2 across the three ridge models, then
worst maximum absolute residual, considering only candidates that satisfy
the acceptance criteria under every ridge model. New live evolution is
still necessary.
    """
    base = _vector(base, "base")
    residual = _vector(residual, "residual")
    sigma = _vector(sigma, "sigma")
    hx = _history(history_initials, "history_initials")
    hr = _history(history_residuals, "history_residuals")
    if hx.shape != hr.shape:
        raise ValueError("Initial-state and residual histories must have identical shapes")
    if np.any(sigma <= 0):
        raise ValueError("sigma entries must be positive")
    matching = [i for i in range(len(hx))
                if np.allclose(hx[i], base, rtol=0, atol=1e-8)
                and np.allclose(hr[i], residual, rtol=0, atol=1e-8)]
    if anchor_index is None:
        if len(matching) != 1:
            raise ValueError("Anchor inference requires exactly one matching initial/residual history row")
        anchor_index = matching[0]
    elif (isinstance(anchor_index, (bool, np.bool_)) or
          not isinstance(anchor_index, (int, np.integer)) or
          not 0 <= int(anchor_index) < len(hx)):
        raise ValueError("anchor_index must be an integer within the history")
    anchor_index = int(anchor_index)
    if anchor_index not in matching:
        raise ValueError("Anchor initial/residual history row does not match base/residual")
    alpha = float(alpha)
    caps = (float(max_position_step), float(max_velocity_step))
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0,1]")
    if any(not np.isfinite(x) or x <= 0 for x in caps):
        raise ValueError("Position and velocity step caps must be positive and finite")
    z = residual/sigma
    current_norm, current_max = _norms(z)
    failing = np.flatnonzero(np.abs(z) > 3.0)
    diagnostics = dict(
        method="static-prior-plus-regularized-history-defect",
        anchor_history_index=anchor_index,
        history_order="unaltered provenance order; adjacent secants include rejected live steps",
        warning="Proposal only: sampled live directions do not determine a full live Jacobian; new live validation is required.",
        base_initial_12d=base, residual_12d=residual, sigma_12d=sigma,
        initial_scale_12d=INITIAL_SCALE,
        initial_scale_units="kpc for positions; km/s for velocities",
        current_residual_norm=current_norm, current_max_abs_normalized=current_max,
        currently_failing_component_indices=failing,
        alpha=alpha, max_position_step_kpc=caps[0], max_velocity_step_kms=caps[1],
        finite_difference_physical_epsilon=FD_PHYSICAL_EPSILON,
        finite_difference_units="0.02 kpc for positions; 0.02 km/s for velocities",
        regularization_definition="s/(s^2+lambda^2), lambda=relative*smax; dimensionless scaled matrices",
        ridge_relative_grid=RIDGE_RELATIVE, damping_relative_grid=DAMPING_RELATIVE)
    if not np.any(residual):
        return _plain(dict(status="zero_residual", selected=None, candidates=[], diagnostics=diagnostics))

    # Columns of U are differences of scaled INITIAL states, not endpoint sigmas.
    u_history = (np.diff(hx, axis=0)/INITIAL_SCALE).T
    left, singular, right_t = np.linalg.svd(u_history, full_matrices=False)
    rank_tolerance = max(float(singular[0])*1e-10, np.finfo(float).tiny)
    rank = int(np.sum(singular > rank_tolerance))
    if rank < 2:
        raise ValueError(f"History has only {rank} independent initial-state direction(s); at least 2 required")
    basis = left[:, :rank]

    def evaluate(state, label):
        return _vector(forward(state.copy()), label)

    static_history = np.array([evaluate(x, f"static history endpoint {i}") for i, x in enumerate(hx)])
    base_static = static_history[anchor_index].copy()
    live_deltas = np.diff(hr, axis=0).T/sigma[:, None]
    static_deltas = np.diff(static_history, axis=0).T/sigma[:, None]
    defects = live_deltas-static_deltas
    static_jacobian = np.empty((12, 12))
    for i in range(12):
        delta = np.zeros(12); delta[i] = FD_PHYSICAL_EPSILON
        plus = evaluate(base+delta, f"static +difference {i}")
        minus = evaluate(base-delta, f"static -difference {i}")
        static_jacobian[:, i] = ((plus-minus)/(2*FD_PHYSICAL_EPSILON)) * INITIAL_SCALE[i]/sigma
    if not np.all(np.isfinite(static_jacobian)):
        raise ValueError("Static scaled Jacobian is not finite")
    models = []
    for ridge in RIDGE_RELATIVE:
        ridge_parameter = ridge*singular[0]
        factors = singular/(singular**2+ridge_parameter**2)
        inverse = (right_t.T*factors)@left.T
        correction = defects@inverse
        models.append(dict(ridge_relative=ridge, ridge_parameter=float(ridge_parameter), correction=correction))
    diagnostics.update(
        history_count=len(hx), history_rank=rank, history_scaled_singular_values=singular,
        history_rank_tolerance=rank_tolerance,
        history_condition_number=float(singular[0]/singular[rank-1]),
        unmeasured_initial_dimension=12-rank,
        history_scaled_initial_steps=u_history, history_scaled_live_changes=live_deltas,
        history_scaled_static_changes=static_deltas, history_scaled_defects=defects,
        history_static_endpoints_12d=static_history, static_scaled_jacobian=static_jacobian,
        defect_models=[dict(ridge_relative=m["ridge_relative"], ridge_parameter=m["ridge_parameter"],
                           correction_scaled_matrix=m["correction"],
                           history_defect_fit_error_norms=np.linalg.norm(m["correction"]@u_history-defects, axis=0))
                       for m in models])
    # Show how much of the task can be addressed inside sampled live directions.
    coefficients = np.linalg.lstsq(live_deltas, -z, rcond=None)[0]
    subspace_z = z+live_deltas@coefficients
    diagnostics["historical_subspace_least_squares"] = dict(
        coefficients=coefficients, predicted_residual_norm=_norms(subspace_z)[0],
        max_abs_normalized=_norms(subspace_z)[1], predicted_residual_12d=subspace_z*sigma,
        note="Unbounded linear history-only diagnostic, not a validated candidate")

    candidates = []
    for model in models:
        correction = model["correction"]
        response = static_jacobian+correction
        jl, js, jrt = np.linalg.svd(response, full_matrices=False)
        for damping_relative in DAMPING_RELATIVE:
            damping = float(damping_relative*js[0])
            candidate = dict(id=f"candidate_{len(candidates)+1:02d}",
                ridge_relative=model["ridge_relative"], ridge_parameter=model["ridge_parameter"],
                damping_relative=damping_relative, damping_parameter=damping,
                response_scaled_singular_values=js, accepted=False,
                increment_12d=None, predicted_residual_12d=None,
                static_predicted_residual_12d=None, predicted_residual_norm=None,
                max_abs_normalized=None, outside_history_fraction=None, attempts=[])
            if js[0] <= np.finfo(float).tiny:
                candidate["reason"] = "zero_response_matrix"
                candidates.append(candidate)
                continue
            factors = js/(js**2+damping**2)
            raw_u = -alpha*(jrt.T@(factors*(jl.T@z)))
            raw_increment = INITIAL_SCALE*raw_u
            cap_factor = 1.0
            for block, norm in enumerate(_block_norms(raw_increment)):
                if norm > caps[block % 2]:
                    cap_factor = min(cap_factor, caps[block % 2]/norm)
            candidate.update(raw_increment_12d=raw_increment,
                             raw_block_norms=_block_norms(raw_increment), cap_factor=cap_factor)
            for backtracks in range(5):
                factor = cap_factor*0.5**backtracks
                scaled_step = raw_u*factor
                increment = INITIAL_SCALE*scaled_step
                try:
                    endpoint = evaluate(base+increment, "candidate static endpoint")
                    prior_z = z+(endpoint-base_static)/sigma
                    predicted_z = prior_z+correction@scaled_step
                    predicted_norm, predicted_max = _norms(predicted_z)
                except (ValueError, RuntimeError) as exc:
                    candidate["attempts"].append(dict(backtracks=backtracks, step_factor=factor, error=str(exc)))
                    continue
                accepted = _better(predicted_z, z, failing)
                candidate["attempts"].append(dict(backtracks=backtracks, step_factor=factor,
                    predicted_residual_norm=predicted_norm, max_abs_normalized=predicted_max,
                    accepted=accepted))
                step_norm = float(np.linalg.norm(scaled_step))
                outside = float(np.linalg.norm(scaled_step-basis@(basis.T@scaled_step))/step_norm) if step_norm else 0.0
                outside = float(np.clip(outside, 0.0, 1.0))
                cross_model = []
                for other in models:
                    other_z = prior_z+other["correction"]@scaled_step
                    other_norm, other_max = _norms(other_z)
                    cross_model.append(dict(ridge_relative=other["ridge_relative"],
                        predicted_residual_norm=other_norm, max_abs_normalized=other_max,
                        predicted_residual_12d=other_z*sigma,
                        acceptance_criteria_met=_better(other_z,z,failing)))
                candidate.update(accepted=accepted, increment_12d=increment,
                    candidate_initial_12d=base+increment,
                    predicted_residual_12d=predicted_z*sigma,
                    static_predicted_residual_12d=prior_z*sigma,
                    predicted_residual_norm=predicted_norm, max_abs_normalized=predicted_max,
                    static_predicted_residual_norm=_norms(prior_z)[0],
                    static_max_abs_normalized=_norms(prior_z)[1],
                    outside_history_fraction=outside, applied_block_norms=_block_norms(increment),
                    step_factor=factor, backtracks=backtracks,
                    cross_ridge_predictions=cross_model,
                    worst_across_ridge_residual_norm=max(p["predicted_residual_norm"] for p in cross_model),
                    worst_across_ridge_max_abs_normalized=max(p["max_abs_normalized"] for p in cross_model),
                    passes_all_ridge_acceptance_criteria=all(p["acceptance_criteria_met"] for p in cross_model))
                if accepted:
                    break
            if not candidate["accepted"]:
                candidate["reason"] = "no_step_meets_all_surrogate_acceptance_criteria"
            candidates.append(candidate)

    accepted = [c for c in candidates if c["accepted"] and c.get("passes_all_ridge_acceptance_criteria", False)]
    for candidate in candidates:
        if candidate["accepted"] and not candidate.get("passes_all_ridge_acceptance_criteria", False):
            candidate["selection_excluded_reason"] = "at_least_one_ridge_model_fails_acceptance_criteria"
    selected = min(accepted, key=lambda c:(c["worst_across_ridge_residual_norm"],
                                          c["worst_across_ridge_max_abs_normalized"],
                                          c["max_abs_normalized"])) if accepted else None
    if selected is not None:
        selected["selection_note"] = "Best bounded proposal by worst predicted norm across ridge models; live validation still required."
    if accepted:
        directions = np.array([c["increment_12d"]/INITIAL_SCALE for c in accepted])
        norms = np.linalg.norm(directions,axis=1)
        cosine = (directions@directions.T)/(norms[:,None]*norms[None,:])
        cosine = np.clip(cosine,-1.,1.)
        pairs = cosine[np.triu_indices(len(accepted),1)]
        stability = dict(candidate_ids=[c["id"] for c in accepted], pairwise_cosine_matrix=cosine,
                         minimum_pairwise_cosine=float(np.min(pairs)) if pairs.size else None,
                         median_pairwise_cosine=float(np.median(pairs)) if pairs.size else None,
                         note="Agreement among static-prior candidates does not measure live uncertainty.")
    else:
        stability = dict(candidate_ids=[], pairwise_cosine_matrix=[],
                         minimum_pairwise_cosine=None, median_pairwise_cosine=None)
    diagnostics.update(accepted_candidate_count=len(accepted), grid_direction_stability=stability,
                       selection_rule="require all ridge acceptance criteria; minimize worst residual L2 across ridge models, then worst max absolute normalized residual")
    return _plain(dict(status="candidate_requires_live_validation" if selected else "no_acceptable_candidate",
                       selected=selected, candidates=candidates, diagnostics=diagnostics))
