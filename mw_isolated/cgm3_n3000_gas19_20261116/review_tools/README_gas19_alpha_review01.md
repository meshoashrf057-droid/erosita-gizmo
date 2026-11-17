# Gas19 single alpha=.3 adapter — source prepared only

Current family is gas19 187738...; the earlier gas-rich family's pass, residuals and live Jacobian are not inputs. No proposal, IC, runtime or simulation was executed by preparing these sources.

The original shooting_local.propose_local_step is unchanged:
delta = B(S(base) - .3 * measured_residual) - B(S(base)).
It retains full12D static forward/inverse closure0.01 native units, per-cloud position20kpc/velocity30km/s caps, scalar cap/backtracking, and static weighted D12 improvement. It uses newly read gas19 source mass profiles; the host may be read only from its same-SHA verified archive if its original local path is gone. This does not require live finite-difference probes or claim live convergence.

Wait for successful full-duration resume02, canonical true-target endpoint gate and immutable pre-gate status matching report.production_status_sha256. A real12/12 PASS stops immediately. Actual valid FAIL enables one alpha01. If max<=4, the default near-target eligibility is retained; a provenance JSON must explicitly bind this exact gate SHA and the user's current single-alpha instruction:
schema=gas19-single-alpha-user-routing/v1; authorized=true; allow_near_alpha=true; family_key=current; base_ic_sha256=81c53...; gate_sha256=actual; alpha=0.3; scope='single gas19 alpha01 after this exact failed endpoint'.

Run each necessary operation once from the existing attached visible terminal with complete logs, with fixed Python. Existing outputs cause a stop, never an implicit repeat. No smoke is created or represented as passed.

Set N to /home/xingdi/gizmo-runs/mw_isolated/cgm3_n3000_gas19_20261116 and PY to /home/xingdi/gizmo-runs/.venv/bin/python.

1. $PY -B -u $N/review_tools/gas19_alpha_review01.py propose --base-run $N/calibration01 --trial-dir $N/alpha01 --gate-result $N/calibration01/endpoint_review/gate_result.json --status-file $N/calibration01/production_resume02/status.json --gate-status-file $N/recovery_20261117_02/pre_gate_status.json
   Add --near-authorization $N/provenance/ACTUAL_BOUND_AUTHORIZATION.json only if needed.
2. Independently review the actual proposal; static prediction is not a live success.
3. $PY -B -u $N/review_tools/gas19_alpha_review01.py apply --proposal $N/alpha01_proposal.json --trial-dir $N/alpha01
   Apply freshly recomputes the static proposal and requires exact equality, then changes only one native-unit rigid12D initial translation on an independent IC copy.
4. $PY -B -u $N/review_tools/gas19_alpha_runtime_review01.py prepare --proposal $N/alpha01_proposal.json --trial-dir $N/alpha01
   Exhaustive particle/Header/attribute/identity/observation checks, same parameter bytes, all50 independent dependency copies, MPI6/OMP1/nice19/flag2, explicit no-smoke readiness.
5. Independently review actual readiness/clearance, then the ONLY COMPLETE PRODUCTION ENTRY:
   $PY -B -u $N/review_tools/gas19_alpha_production_review01.py --trial-dir $N/alpha01
   The thin source-reviewed wrapper holds the same existing global lease as resume02, then calls unchanged scripts/production_and_gate.py. That existing controller runs start_production.sh -> script PTY -> drive_stage/native+validator, then automatically gates the true endpoint and writes accepted/outcome. Do not dispatch start_production.sh or drive_stage directly as the overall workflow.

Original sources, source HDFs, old scientific gates and the active resumed run are untouched.

Compute lock order: alpha ODE/IC/runtime operations acquire existing old-campaign global compute.flock, then gas19 local compute.flock; production wrapper acquires only global, while unchanged drive_stage acquires local. No lock or source was deleted/replaced and no simulation ran during this preparation.
