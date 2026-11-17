#!/usr/bin/env python3
"""Read-only default gate for the t13 -> jet handoff.

Exit codes: 0=pass, 2=waiting, 3=orbit_failed, 4=invalid_evidence.
Optional reports/cache must stay in RUN/followup_checks.
No simulation input, runner, snapshot, or validation result is modified.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

import h5py

VERSION = "cpp-orbit-correction-gate-v1"
SATELLITES = ("lmc", "smc")
COMPONENTS = ("x", "y", "z", "vx", "vy", "vz")
ORDER = [f"{s}_{c}" for s in SATELLITES for c in COMPONENTS]
UNITS = ("kpc", "kpc", "kpc", "km/s", "km/s", "km/s")
SIGNATURE = {
    "algorithm": "fixed-original-star-identities-shrinking-core-v4",
    "tracer_policy": "original-stars",
    "identity_fields": ["ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber"],
    "host_frame": "BH-position+MW-inner10kpc-mass-weighted-star-velocity-v3",
    "lmc_core_radius_kpc": 8.0,
    "smc_core_radius_kpc": 3.0,
}

class EvidenceError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise EvidenceError(message)

def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}: not numeric")
    value = float(value)
    require(math.isfinite(value), f"{name}: nonfinite")
    return value

def close(a, b, name, *, atol=1e-10, rtol=1e-12):
    require(math.isclose(number(a, name), number(b, name), abs_tol=atol, rel_tol=rtol),
            f"{name}: inconsistent")

def vector(value, name):
    require(isinstance(value, list) and len(value) == 6, f"{name}: expected 6 values")
    return [number(v, name) for v in value]

def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)

def load_json(path):
    try:
        value = json.loads(Path(path).read_text())
        def finite_json(item):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("JSON contains a nonfinite number")
            if isinstance(item, dict):
                for child in item.values(): finite_json(child)
            elif isinstance(item, list):
                for child in item: finite_json(child)
        finite_json(value)
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"{path}: cannot read JSON: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value

def resolved(value, expected, name):
    require(isinstance(value, str), f"{name}: missing path")
    require(Path(value).expanduser().resolve() == Path(expected).resolve(), f"{name}: wrong path")

def file_stamp(path):
    stat = Path(path).stat()
    return {
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev, "inode": stat.st_ino,
    }

def inside_checks(run, destination):
    destination = Path(destination).expanduser().resolve()
    root = (Path(run).resolve() / "followup_checks").resolve()
    require(destination == root or root in destination.parents,
            "Reports and hash caches must be inside RUN/followup_checks")
    return destination

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("x") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

class Hashes:
    """Hash each unchanged file at most once per invocation; disk cache is opt-in."""
    def __init__(self, run, cache_dir=None):
        self.memory = {}
        self.entries = {}
        self.cache_path = None
        if cache_dir is not None:
            self.cache_path = inside_checks(run, cache_dir) / "jet_gate_hash_cache.json"
            if self.cache_path.exists():
                try:
                    cached = load_json(self.cache_path)
                    if cached.get("version") == VERSION:
                        entries = cached.get("entries", {})
                        self.entries = entries if isinstance(entries, dict) else {}
                except EvidenceError:
                    pass

    def digest(self, path):
        path = Path(path).resolve()
        key = str(path)
        before = file_stamp(path)
        found = self.memory.get(key) or self.entries.get(key)
        if isinstance(found, dict) and found.get("stat") == before:
            digest = found.get("sha256", "")
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                self.memory[key] = found
                return digest
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        require(before == file_stamp(path), f"{path}: file changed while hashing")
        self.memory[key] = {"stat": before, "sha256": digest.hexdigest()}
        return digest.hexdigest()

    def save(self):
        if self.cache_path is not None and self.memory:
            atomic_json(self.cache_path, {"version": VERSION, "entries": self.memory})

def validate_12d(summary, provenance, reference):
    """Pure semantic validation. Returns evidence plus the scientific orbit decision."""
    epoch = summary.get("present_epoch")
    require(isinstance(epoch, dict), "summary.present_epoch missing")
    require(epoch.get("target_time_bracketed") is True, "target time is not covered")
    require(provenance.get("target_covered") is True, "provenance target_covered is not true")
    require(provenance.get("shooting_usable") is True, "provenance shooting_usable is not true")
    require(epoch.get("evaluation_method") == "exact target snapshot",
            "Jet input must be the exact final target snapshot, not an interpolation/nearest snapshot")
    require(provenance.get("evaluation_method") == epoch["evaluation_method"],
            "Evaluation methods disagree")
    close(epoch.get("nsigma_threshold"), 3.0, "nsigma_threshold", atol=0, rtol=0)
    for record in (summary, provenance):
        require(record.get("measurement_signature") == SIGNATURE, "Measurement signature changed")
        require(record.get("stellar_tracer_policy") == "original-stars", "Wrong stellar tracer policy")
    require(summary.get("coordinate_order") == ["x_kpc", "y_kpc", "z_kpc", "vx_kms", "vy_kms", "vz_kms"],
            "Summary coordinate order differs")
    require(provenance.get("coordinate_order") == ORDER, "Provenance coordinate order differs")
    target = number(reference["target_time_code"], "reference target time")
    tolerance = 1e-8 * max(1.0, abs(target))
    close(epoch.get("target_time_code"), target, "summary target time", atol=tolerance, rtol=0)
    close(provenance.get("target_time_code"), target, "provenance target time", atol=tolerance, rtol=0)
    for current in (summary.get("unit_time_gyr_used"), provenance.get("unit_time_gyr")):
        close(current, reference["unit_time_gyr"], "time units", atol=0, rtol=1e-12)
    zscores = []
    failures = []
    values = {}
    for sat in SATELLITES:
        vals = {kind: vector(epoch.get(f"{sat}_{kind}_6d"), f"{sat}_{kind}_6d")
                for kind in ("simulated", "observed", "sigma", "residual", "normalized_residual")}
        for index, component in enumerate(COMPONENTS):
            prefix = f"{sat}_{component}"
            sigma = vals["sigma"][index]
            require(sigma > 0, prefix + ": sigma must be positive")
            close(vals["observed"][index], reference[sat]["observed"][index], prefix + " observed")
            close(sigma, reference[sat]["sigma"][index], prefix + " sigma", atol=0, rtol=1e-12)
            residual = vals["simulated"][index] - vals["observed"][index]
            close(vals["residual"][index], residual, prefix + " residual")
            zscore = residual / sigma
            close(vals["normalized_residual"][index], zscore, prefix + " zscore")
            zscores.append(zscore)
            if abs(zscore) > 3:
                failures.append({"coordinate": prefix, "zscore": zscore})
        computed = all(abs(z) <= 3 for z in zscores[-6:])
        require(epoch.get(f"{sat}_6d_pass") is computed, f"{sat}_6d_pass disagrees with numeric residuals")
        values[sat] = vals
    return {
        "orbit_pass": not failures,
        "coordinate_order": ORDER,
        "normalized_residual_12d": zscores,
        "max_abs_sigma": max(abs(z) for z in zscores),
        "threshold": 3.0,
        "failed_coordinates": failures,
        "values": values,
        "additional_diagnostics_not_used_as_gate": {
            key: epoch.get(key) for key in (
                "overall_pass", "scalar_velocity_pass", "scalar_velocity_required",
                "satellite_structure_pass", "gas_timestep_pass", "science_quality_pass",
                "science_quality_required", "lmc_tracer_fraction", "smc_tracer_fraction")
        },
    }

def validate_csv(path, decision):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 12, "Residual CSV must contain exactly 12 rows")
    seen = set()
    for row in rows:
        sat = row.get("satellite", "").lower()
        component = row.get("component", "")
        require(sat in SATELLITES and component in COMPONENTS, "Invalid CSV coordinate")
        key = (sat, component)
        require(key not in seen, "Duplicate CSV coordinate")
        seen.add(key)
        index = COMPONENTS.index(component)
        require(row.get("unit") == UNITS[index], "CSV coordinate unit differs")
        values = decision["values"][sat]
        for name, source in (("simulated", "simulated"), ("observed", "observed"),
                             ("sigma", "sigma"), ("residual", "residual"),
                             ("zscore", "normalized_residual")):
            try:
                actual = float(row[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceError("Invalid numeric CSV field") from exc
            close(actual, values[source][index], f"CSV {sat}_{component}/{name}")
        require(row.get("pass", "").lower() in ("true", "false"), "Invalid CSV pass value")
        require((row["pass"].lower() == "true") == (abs(values["normalized_residual"][index]) <= 3),
                "CSV pass does not match numeric residual")
    require(seen == {(s, c) for s in SATELLITES for c in COMPONENTS}, "Missing CSV coordinate")

def check_run(run, cache_dir=None):
    run = Path(run).expanduser().resolve()
    result = {
        "version": VERSION, "run_directory": str(run),
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": "invalid_evidence", "reasons": [],
    }
    hashes = Hashes(run, cache_dir)
    try:
        status_path = run / "production/status.json"
        if not status_path.exists():
            result.update(state="waiting", reasons=["Production status not yet available"])
            return result
        status = load_json(status_path)
        result["production_state"] = status.get("state")
        if status.get("state") in ("starting", "running", "exited", "validating"):
            if status.get("returncode") not in (None, 0):
                raise EvidenceError("Simulation exited unsuccessfully")
            result.update(state="waiting", reasons=["Waiting for simulation completion"])
            return result
        require(status.get("state") == "completed", "Simulation did not complete at its target")
        require(status.get("returncode") == 0, "Completed run has nonzero/missing return code")
        if "orbit_validation_returncode" not in status:
            result.update(state="waiting", reasons=["Simulation finished; waiting for orbit validation"])
            return result
        require(status["orbit_validation_returncode"] == 0, "Orbit validator failed")
        folder = run / "orbit_validation"
        summary_path = folder / "magellanic_validation_summary.json"
        provenance_path = folder / "magellanic_present_6d.provenance.json"
        csv_path = folder / "magellanic_present_6d.csv"
        ref_path = run / ("orbit_000.hdf5" if (run / "orbit_000.hdf5").is_file() else "t13_cpp_000.hdf5")
        summary, provenance = load_json(summary_path), load_json(provenance_path)
        audit = load_json(ref_path.with_suffix(".orbit.json"))
        for record in (summary, provenance):
            resolved(record.get("reference_ic"), ref_path, "reference_ic")
        resolved(summary.get("shooting_provenance"), provenance_path, "shooting_provenance")
        resolved(provenance.get("residual_csv"), csv_path, "residual_csv")
        ref_sha = hashes.digest(ref_path)
        require(ref_sha == provenance.get("reference_ic_sha256"), "Reference IC SHA256 mismatch")
        require(ref_sha == audit.get("output_ic_sha256"), "Reference IC differs from its build audit")
        csv_sha = hashes.digest(csv_path)
        require(csv_sha == provenance.get("residual_csv_sha256"), "Residual CSV SHA256 mismatch")
        with h5py.File(ref_path, "r") as ref:
            orbit = ref["MagellanicOrbitReference"]
            model = ref["MagellanicModelReference"]
            reference = {
                "target_time_code": float(orbit.attrs["PresentTargetTime_Code"]),
                "unit_time_gyr": float(orbit.attrs["UnitTime_Gyr"]),
            }
            require(reference["unit_time_gyr"] > 0, "Reference time unit must be positive")
            require(provenance.get("model_key") == text(model.attrs["SATELLITE_MODEL_KEY"]),
                    "Provenance model_key differs")
            require(provenance.get("trial_label") == text(model.attrs["TRIAL_LABEL"]),
                    "Provenance trial_label differs")
            for sat in SATELLITES:
                cap = sat.upper()
                reference[sat] = {
                    "observed": [float(x) for x in orbit[cap+"_Position_kpc"][-1]] +
                                [float(x) for x in orbit[cap+"_Velocity_kms"][-1]],
                    "sigma": [float(x) for x in orbit[cap+"_Current_PositionSigma_kpc"][...]] +
                             [float(x) for x in orbit[cap+"_Current_VelocitySigma_kms"][...]],
                }
        target = number(reference["target_time_code"], "target time")
        tolerance = 1e-8 * max(1.0, abs(target))
        for value in (audit.get("stage2_target_code"), status.get("target_time"),
                      status.get("final_snapshot_time")):
            close(value, target, "Run/IC target time", atol=tolerance, rtol=0)
        decision = validate_12d(summary, provenance, reference)
        validate_csv(csv_path, decision)
        snapshots = []
        for path in (run / "production/output").iterdir():
            if re.fullmatch(r"snapshot_[0-9]+\.hdf5", path.name):
                with h5py.File(path, "r") as snapshot:
                    stamp = number(float(snapshot["Header"].attrs["Time"]), "snapshot time")
                snapshots.append((stamp, path.resolve()))
        require(bool(snapshots), "No production snapshots")
        snapshots.sort(key=lambda item: (item[0], item[1].name))
        final_time, final_path = snapshots[-1]
        close(final_time, target, "Last snapshot is not at target", atol=tolerance, rtol=0)
        sources = provenance.get("target_snapshot_sources")
        require(isinstance(sources, list) and len(sources) == 1,
                "Final handoff needs one exact endpoint snapshot")
        resolved(sources[0].get("file"), final_path, "Endpoint snapshot")
        close(sources[0].get("time_code"), final_time, "Endpoint source time", atol=tolerance, rtol=0)
        close(sources[0].get("weight"), 1, "Endpoint source weight", atol=1e-12, rtol=0)
        with h5py.File(final_path, "r") as final:
            header = final["Header"].attrs
            require(int(header.get("NumFilesPerSnapshot", 1)) == 1, "Split endpoint requires explicit handling")
        final_stat = file_stamp(final_path)
        evidence_time = min(x.stat().st_mtime_ns for x in (summary_path, provenance_path, csv_path))
        require(max(final_stat["mtime_ns"], final_stat["ctime_ns"]) <= evidence_time,
                "Endpoint snapshot was changed after the validation evidence was written; rerun validation")
        final_sha = hashes.digest(final_path)
        for path in (ref_path, csv_path, final_path):
            require(hashes.memory[str(path.resolve())]["stat"] == file_stamp(path),
                    f"{path}: changed while evaluating gate")
        result.update({
            "state": "pass" if decision["orbit_pass"] else "orbit_failed",
            "reasons": [] if decision["orbit_pass"] else ["At least one of the 12 coordinates exceeds 3 sigma"],
            "target_time_code": target, "reference_ic": str(ref_path), "reference_ic_sha256": ref_sha,
            "residual_csv": str(csv_path), "residual_csv_sha256": csv_sha,
            "endpoint_snapshot": {
                "path": str(final_path), "sha256": final_sha,
                "time_code": final_time, **final_stat,
            },
            "coordinate_order": decision["coordinate_order"],
            "normalized_residual_12d": decision["normalized_residual_12d"],
            "max_abs_sigma": decision["max_abs_sigma"],
            "nsigma_threshold": 3.0,
            "failed_coordinates": decision["failed_coordinates"],
            "additional_diagnostics_not_used_as_gate": decision["additional_diagnostics_not_used_as_gate"],
            "failed_snapshot_count": summary.get("failed_snapshot_count"),
            "measurement_signature": SIGNATURE,
            "handoff_instruction": "Copy only the endpoint bound by this SHA256; verify copied bytes before editing the copy.",
        })
        hashes.save()
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as exc:
        result.update(state="invalid_evidence", reasons=[str(exc)])
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="Optional JSON report inside RUN/followup_checks")
    parser.add_argument("--cache-dir", type=Path, help="Optional SHA cache inside RUN/followup_checks; default is read-only")
    args = parser.parse_args(argv)
    try:
        report = check_run(args.run_dir, args.cache_dir)
        if args.output is not None:
            output = inside_checks(args.run_dir, args.output)
            require(output.suffix == ".json", "Gate report filename must end in .json")
            require(output.name != "jet_gate_hash_cache.json", "Gate report cannot overwrite hash cache")
            atomic_json(output, report)
    except (EvidenceError, OSError) as exc:
        report = {"version": VERSION, "state": "invalid_evidence", "reasons": [str(exc)]}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return {"pass": 0, "waiting": 2, "orbit_failed": 3, "invalid_evidence": 4}[report["state"]]

if __name__ == "__main__":
    raise SystemExit(main())
