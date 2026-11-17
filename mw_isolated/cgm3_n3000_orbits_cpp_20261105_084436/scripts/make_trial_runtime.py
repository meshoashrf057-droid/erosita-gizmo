#!/usr/bin/env python3
"""Prepare a new orbit trial runtime; never start GIZMO or create readiness."""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import shutil

from run_trial import BINARY_PATH, BINARY_SHA256, PROJECT_PYTHON, sha256, read_json, utc

TEMPLATE = Path("/home/xingdi/gizmo-runs/mw_isolated/shooting_cpp_20261031_082656/local01_lmc_x_p05")
RUNNER = Path(__file__).resolve().with_name("drive_stage.py")
SHARED = {"InitCondFile": "../orbit_000", "OutputDir": "./output/", "MaxMemSize": "1000", "CpuTimeBetRestartFile": "900"}
SMOKE_DURATION = 1e-5


def parse_params(text):
    values = {}
    for line in text.splitlines():
        fields = line.split("%", 1)[0].split()
        if len(fields) >= 2:
            if fields[0] in values:
                raise ValueError(f"Duplicate parameter: {fields[0]}")
            values[fields[0]] = fields[1]
    return values


def changed_params(text, replacements):
    original = parse_params(text)
    missing = set(replacements) - set(original)
    if missing:
        raise ValueError(f"Missing template parameter(s): {sorted(missing)}")
    lines = []
    for line in text.splitlines(keepends=True):
        fields = line.split("%", 1)[0].split()
        if fields and fields[0] in replacements:
            comment = (" %" + line.split("%", 1)[1].rstrip("\n")) if "%" in line else ""
            line = f"{fields[0]:<42} {replacements[fields[0]]}{comment}\n"
        lines.append(line)
    result = "".join(lines)
    actual = parse_params(result)
    for key, value in original.items():
        if key not in replacements and actual[key] != value:
            raise AssertionError(f"Unexpected parameter change: {key}")
    return result


def launcher(trial, mode):
    title = re.sub(r"[^A-Za-z0-9_.-]", "_", trial.name) + "_" + mode + "_CPP"
    q = shlex.quote
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        'if [[ ! -t 0 || ! -t 1 ]]; then echo "Launch this script in a visible Ubuntu terminal." >&2; exit 2; fi',
        f"cd {q(str(trial))}",
        'export GIZMO_VISIBLE_TTY="$(tty)"',
        f'log={q(mode + "_terminal_session.log")}',
        'if ! (set -o noclobber; : > "$log") 2>/dev/null; then echo "Terminal log already exists; refusing duplicate launch." >&2; exit 2; fi',
        'exec > >(tee -a "$log") 2>&1',
        f"printf '\\033]0;{title}\\007'",
        f"printf '{title}: MPI=6, nice=19, RestartFlag=2\\n'",
        "printf 'Run directory: %s\\n' \"$PWD\"",
        "printf 'Visible terminal: %s\\n' \"$GIZMO_VISIBLE_TTY\"",
        f"printf '%s\\n' \"$GIZMO_VISIBLE_TTY\" > {q(mode + '_terminal.tty')}",
        "printf 'Start UTC: '; date -u",
        f"{q(PROJECT_PYTHON)} -u {q(str(RUNNER))} --trial-dir {q(str(trial))} --mode {mode}",
        "result=$?",
        "printf '\\nProcess returned %s\\n' \"$result\"",
        f"if [[ -f {mode}/status.json ]]; then cat {mode}/status.json; fi",
        'exit "$result"',
        "",
    ])


def prepare(trial, template=None):
    trial = Path(trial).resolve()
    template = Path(template or TEMPLATE).resolve()
    if not trial.is_dir():
        raise ValueError("Create the trial directory and validated orbit_000 IC/sidecar first")
    reserved = ("smoke", "production", "run_configuration.json", "runtime_preparation.lock",
                "start_smoke.sh", "start_production.sh", "production_ready.json",
                "orbit_validation", "orbit_validation.log", "smoke_terminal_session.log",
                "production_terminal_session.log")
    for name in reserved:
        if (trial / name).exists():
            raise ValueError(f"Refusing to overwrite existing runtime artifact: {trial / name}")
    audit = read_json(trial / "orbit_000.orbit.json")
    ic_hash = sha256(trial / "orbit_000.hdf5")
    if ic_hash != audit["output_ic_sha256"]:
        raise ValueError("Trial IC differs from its sidecar")
    previous = read_json(template / "run_configuration.json")
    if previous["binary"] != {"path": BINARY_PATH, "sha256": BINARY_SHA256}:
        raise ValueError("Template does not use the pinned C++ binary")
    if sha256(BINARY_PATH) != BINARY_SHA256:
        raise ValueError("Pinned C++ executable changed")
    if not Path(previous["dynamic_library_path"]).is_dir():
        raise ValueError("Required existing HDF5 library directory is missing")
    parameter_source = template / "production" / "stage2.param"
    text = parameter_source.read_text()
    params = parse_params(text)
    old_start, old_target = float(params["TimeBegin"]), float(params["TimeMax"])
    start, target = float(audit["stage2_start_code"]), float(audit["stage2_target_code"])
    family_overrides=dict(TimeBegin=format(start,'.17g'),TimeMax=format(target,'.17g'),
        TimeOfFirstSnapshot=format(start,'.17g'),CritPhysDensity="3000",
        MaxKernelRadius="1e56",MinSizeTimestep="1e-12")
    text=changed_params(text,family_overrides)
    params=parse_params(text)
    family=audit["model_key_inputs"]["production_parameters"]
    for key in family_overrides:
        if float(params[key])!=float(family[key]):
            raise ValueError("New-family parameter differs: "+key)
    if not (math.isfinite(start) and math.isfinite(target) and target > start):
        raise ValueError("Invalid template time range")
    if not math.isclose(start, float(audit["stage2_start_code"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("IC start time disagrees with fixed production template")
    if not math.isclose(target, float(audit["stage2_target_code"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("IC target disagrees with fixed production template")
    if params["NumFilesPerSnapshot"] != "1" or params["SnapshotFileBase"] != "snapshot":
        raise ValueError("This runner requires single-file snapshot_NNN.hdf5 outputs")
    dependencies = [template / "production" / "TREECOOL"]
    dependencies.extend(sorted((template / "production" / "spcool_tables").glob("spcool_*")))
    if not dependencies[0].is_file() or not any(p.is_file() and re.fullmatch(r"spcool_\d+", p.name) for p in dependencies[1:]):
        raise ValueError("TREECOOL or spcool_tables cooling inputs are missing")
    # Only actual numeric tables are runtime inputs; omit macOS/Windows metadata files.
    dependencies = [dependencies[0]] + [p for p in dependencies[1:] if p.is_file() and re.fullmatch(r"spcool_\d+", p.name)]
    parameter_text = {
        "production": changed_params(text, SHARED),
        "smoke": changed_params(text, dict(SHARED, TimeMax=format(start + SMOKE_DURATION, ".17g"),
            TimeOfFirstSnapshot=format(start + 1e-8, ".17g"), TimeBetSnapshot=format((start + SMOKE_DURATION) - (start + 1e-8), ".17g"),
            MaxSizeTimestep="1e-6", MinSizeTimestep="1e-12", TimeLimitCPU="1800")),
    }
    # Exclusive persistent marker deliberately blocks reusing a partially prepared directory.
    with (trial / "runtime_preparation.lock").open("x") as lock:
        lock.write(json.dumps({"prepared_utc": utc(), "ic_sha256": ic_hash}) + "\n")
    records = []
    hashes = {}
    for mode in ("smoke", "production"):
        sub = trial / mode
        sub.mkdir()
        (sub / "output").mkdir()
        (sub / "spcool_tables").mkdir()
        (sub / "stage2.param").write_text(parameter_text[mode])
        hashes[mode] = sha256(sub / "stage2.param")
        for source in dependencies:
            relative = source.relative_to(template / "production")
            destination = sub / relative
            before_hash = sha256(source)
            shutil.copy2(source, destination)
            if sha256(destination) != before_hash:
                raise ValueError(f"Cooling dependency copy mismatch: {source}")
            records.append({"mode": mode, "relative_path": str(destination.relative_to(trial)),
                            "source": str(source), "sha256": before_hash})
        script = trial / f"start_{mode}.sh"
        script.write_text(launcher(trial, mode))
        script.chmod(0o755)
    config = {
        "schema": "isolated-cpp-orbit-runtime-v1", "created_utc": utc(),
        "trial_dir": str(trial), "parameter_source": str(parameter_source),
        "parameter_source_sha256": sha256(parameter_source), "shared_overrides": SHARED,
        "family_parameter_overrides": family_overrides,
        "physical_inheritance": "CritPhysDensity3000, MaxKernelRadius1e56 and MinSizeTimestep1e-12 follow chosen n3000 host; original satellite source snapshots and other physical parameters unchanged; snapshot interval0.1code inherited from orbit template",
        "smoke_only": {"duration_code": SMOKE_DURATION, "TimeOfFirstSnapshot_offset_code": 1e-8,
                       "TimeBetSnapshot": (start + SMOKE_DURATION) - (start + 1e-8), "MaxSizeTimestep": 1e-6,
                       "MinSizeTimestep": 1e-12, "TimeLimitCPU": 1800},
        "ic_sha256": ic_hash, "sidecar_sha256": sha256(trial / "orbit_000.orbit.json"),
        "param_sha256": hashes, "production_target_code": target, "production_start_code": start,
        "restart_flag": 2, "mpi_tasks": 6, "omp_threads": 1, "nice": 19,
        "per_rank_memory_MiB": 1000, "smoke_timeout_seconds": 2100,
        "dynamic_library_path": previous["dynamic_library_path"],
        "binary": {"path": BINARY_PATH, "sha256": BINARY_SHA256},
        "runtime_dependencies": records,
        "readiness_required": ["status=pass", "ic_sha256", "sidecar_sha256", "production_param_sha256",
                               "smoke_param_sha256", "binary_sha256"],
    }
    with (trial / "run_configuration.json").open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = prepare(args.trial_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Refused: {exc}\n")
    print(json.dumps({"trial_dir": config["trial_dir"], "ic_sha256": config["ic_sha256"],
                      "param_sha256": config["param_sha256"], "status": "prepared",
                      "note": "No simulation launched; readiness requires a reviewed successful smoke run."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
