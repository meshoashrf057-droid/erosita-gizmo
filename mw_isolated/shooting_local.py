"""Local static-surrogate shooting steps; these do not predict live convergence.

The historical backend is intentionally unchanged, so existing model hashes and
base ICs remain usable. State vectors use [LMC r, LMC v, SMC r, SMC v], in kpc and
km/s. Only the ODE's internal velocities use kpc/Gyr.
"""
from __future__ import annotations

import numpy as np
import tg19_backend as b

VELOCITY_COLUMNS = np.array([3, 4, 5, 9, 10, 11])
BLOCK_NAMES = ("LMC_position_kpc", "LMC_velocity_kms",
               "SMC_position_kpc", "SMC_velocity_kms")


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (12,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 12-vector in kpc and km/s")
    return array.copy()


def _block_norms(value):
    return {name: float(np.linalg.norm(value[3*i:3*i+3]))
            for i, name in enumerate(BLOCK_NAMES)}


def _weighted_norm(value, sigma):
    result = float(np.linalg.norm(value / sigma))
    if not np.isfinite(result):
        raise ValueError("Scaled shooting norm is not finite")
    return result


def forward_state(initial12_kpc_kms, orbit_kwargs):
    """Advance the exact historical static ODE for ``lookback_gyr``.

``orbit_kwargs`` is the merger's original dictionary for
``b.integrate_magellanic_orbit``. The ODE below matches that function, including
extended satellite masses, dynamical friction, and the MW recoil term.
    """
    initial = _vector(initial12_kpc_kms, "initial state")
    duration = float(orbit_kwargs["lookback_gyr"])
    coulomb_log = float(orbit_kwargs["coulomb_log"])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("lookback_gyr must be positive and finite")
    if not np.isfinite(coulomb_log) or coulomb_log < 0:
        raise ValueError("coulomb_log must be nonnegative and finite")
    profile, lmc, smc = (orbit_kwargs[key] for key in ("profile", "lmc", "smc"))
    use_df = orbit_kwargs["use_dynamical_friction"]
    use_recoil = orbit_kwargs["include_host_recoil"]
    initial[VELOCITY_COLUMNS] *= b.KMS_TO_KPC_GYR

    def rhs(_time, state):
        rl, vl, rs, vs = state[:3], state[3:6], state[6:9], state[9:12]
        delta = rs - rl
        separation = float(np.linalg.norm(delta))
        if separation < 1.0e-6:
            raise RuntimeError("Static forward integration: LMC/SMC centres coincide")
        al = b.host_acceleration(rl, profile)
        al += (b.G_KPC3_MSUN_GYR2 * b.satellite_enclosed_mass(separation, smc)
               * delta / separation**3)
        a_s = b.host_acceleration(rs, profile)
        a_s += (-b.G_KPC3_MSUN_GYR2 * b.satellite_enclosed_mass(separation, lmc)
                * delta / separation**3)
        if use_recoil:
            r_l = max(float(np.linalg.norm(rl)), 1.0e-6)
            r_s = max(float(np.linalg.norm(rs)), 1.0e-6)
            a_mw = b.G_KPC3_MSUN_GYR2 * (
                b.satellite_enclosed_mass(r_l, lmc) * rl / r_l**3
                + b.satellite_enclosed_mass(r_s, smc) * rs / r_s**3)
            al -= a_mw
            a_s -= a_mw
        if use_df:
            al += b.dynamical_friction_acceleration(
                rl, vl, lmc.live_mass, profile, coulomb_log)
            a_s += b.dynamical_friction_acceleration(
                rs, vs, smc.live_mass, profile, coulomb_log)
        return np.concatenate((vl, al, vs, a_s))

    result = b.solve_ivp(rhs, (-duration, 0.0), initial, method="DOP853",
                         rtol=2.0e-10, atol=1.0e-11,
                         max_step=min(0.002, duration / 100.0))
    if not result.success:
        raise RuntimeError(f"Static forward integration failed: {result.message}")
    final = result.y[:, -1].copy()
    final[VELOCITY_COLUMNS] /= b.KMS_TO_KPC_GYR
    return _vector(final, "static endpoint")


def propose_local_step(base, residual12, inverse_callable, forward_callable,
                       sigma12, alpha, max_position_step_kpc=20.0,
                       max_velocity_step_kms=30.0, max_backtracks=8):
    """Return a bounded correction and auditable static-model diagnostics.

The callables accept and return finite 12-vectors in kpc and km/s. Residuals
are measured live endpoints minus the desired endpoints. The correction is
``B(S(base)-alpha*residual)-B(S(base))``. A single scalar preserves its
12-dimensional direction while enforcing both satellites' position/velocity
step caps. Backtracking tests a *static-model prediction*, not a live GIZMO run.

The 0.01 kpc / 0.01 km/s closure checks test numerical forward/inverse consistency;
they are not observational tolerances or evidence of physical model accuracy.
    """
    base = _vector(base, "base initial state")
    residual = _vector(residual12, "live residual")
    sigma = _vector(sigma12, "sigma")
    if np.any(sigma <= 0):
        raise ValueError("All shooting sigma values must be positive")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("shooting alpha must be finite and in (0, 1]")
    caps = (float(max_position_step_kpc), float(max_velocity_step_kms))
    if any(not np.isfinite(cap) or cap <= 0 for cap in caps):
        raise ValueError("Position and velocity step caps must be positive and finite")
    if (isinstance(max_backtracks, bool) or
            not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0):
        raise ValueError("max_backtracks must be a nonnegative integer")
    current_norm = _weighted_norm(residual, sigma)
    zero = np.zeros(12)
    diag = dict(method="local_static_defect_correction", alpha=alpha,
                warning="Model-predicted improvement is not a live convergence guarantee.",
                base_initial_12d=base.tolist(), residual_12d=residual.tolist(),
                sigma_12d=sigma.tolist(), current_residual_norm=current_norm,
                max_position_step_kpc=caps[0], max_velocity_step_kms=caps[1],
                closure_position_tolerance_kpc=0.01,
                closure_velocity_tolerance_kms=0.01)
    if not np.any(residual):
        diag.update(status="zero_residual", model_endpoint_12d=None,
                    proposed_model_endpoint_12d=None,
                    inverse_base_12d=None, closure_12d=None, closure_block_norms=None,
                    raw_step_12d=zero.tolist(), proposed_step_12d=zero.tolist(),
                    raw_block_norms=_block_norms(zero), applied_block_norms=_block_norms(zero),
                    raw_step_scaled_norm=0.0, applied_step_scaled_norm=0.0,
                    cap_lambda=1.0, applied_lambda=1.0, backtracks=0,
                    predicted_residual_12d=zero.tolist(), predicted_residual_norm=0.0,
                    model_prediction_attempts=[])
        return zero, diag

    endpoint = _vector(forward_callable(base.copy()), "base static endpoint")
    inverse_base = _vector(inverse_callable(endpoint.copy()), "inverse base state")
    closure = inverse_base - base
    closure_norms = _block_norms(closure)
    if any(value > 0.01 for value in closure_norms.values()):
        raise RuntimeError("Static forward/inverse closure exceeds 0.01 kpc or "
                           f"0.01 km/s; check units and ODE consistency: {closure_norms}")
    candidate = _vector(inverse_callable(endpoint - alpha * residual), "inverse candidate")
    raw = _vector(candidate - inverse_base, "raw shooting increment")
    norms = _block_norms(raw)
    cap_lambda = 1.0
    for i, norm in enumerate(norms.values()):
        if norm > caps[i % 2]:
            cap_lambda = min(cap_lambda, caps[i % 2] / norm)
    attempts = []
    for backtracks in range(max_backtracks + 1):
        factor = cap_lambda * (0.5 ** backtracks)
        increment = factor * raw
        try:
            new_endpoint = _vector(forward_callable(base + increment), "proposed static endpoint")
            predicted = _vector(residual + new_endpoint - endpoint, "predicted live residual")
            predicted_norm = _weighted_norm(predicted, sigma)
            attempts.append(dict(applied_lambda=factor, predicted_residual_norm=predicted_norm))
        except (RuntimeError, ValueError) as error:
            attempts.append(dict(applied_lambda=factor, error=str(error)))
            continue
        if predicted_norm < current_norm:
            diag.update(status="accepted_static_prediction", model_endpoint_12d=endpoint.tolist(),
                        proposed_model_endpoint_12d=new_endpoint.tolist(),
                        inverse_base_12d=inverse_base.tolist(), closure_12d=closure.tolist(),
                        closure_block_norms=closure_norms,
                        raw_step_12d=raw.tolist(), proposed_step_12d=increment.tolist(),
                        raw_block_norms=norms, applied_block_norms=_block_norms(increment),
                        raw_step_scaled_norm=_weighted_norm(raw, sigma),
                        applied_step_scaled_norm=_weighted_norm(increment, sigma),
                        cap_lambda=cap_lambda, applied_lambda=factor, backtracks=backtracks,
                        predicted_residual_12d=predicted.tolist(),
                        predicted_residual_norm=predicted_norm,
                        model_prediction_attempts=attempts)
            return increment, diag
    raise RuntimeError("No bounded shooting step decreases the static-model predicted "
                       f"weighted residual after {max_backtracks} backtracks. "
                       f"Current norm={current_norm:.9g}; attempts={attempts}. "
                       "Do not launch a new live trial from this rejected proposal.")
