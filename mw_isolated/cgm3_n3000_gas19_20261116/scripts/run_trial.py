#!/usr/bin/env python3
"""Run one isolated C++ orbit trial. Never resume or overwrite an earlier run."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading

PROJECT_PYTHON = "/home/xingdi/gizmo-runs/.venv/bin/python"
VALIDATOR = "/home/xingdi/gizmo-runs/mw_isolated/validation_dashboard_sat_6d_gaswake_v3.py"
BINARY_PATH = "/home/xingdi/gizmo-public-gizmo-cpp/GIZMO"
BINARY_SHA256 = "07154d9677d2e897a7dc557e37f9565d0be835a5e612943870cc5021b137f343"
SNAPSHOT_RE = re.compile(r"snapshot_(\d+)\.hdf5")
TIME_TOLERANCE = 1e-7


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_status(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def require_hash(path, expected, description):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{description} changed: {path}; expected {expected}, got {actual}")
    return actual


def preflight(trial, mode):
    if mode != "production": raise ValueError("User disabled all future smoke simulations")
    trial = Path(trial).resolve()
    config = read_json(trial / "run_configuration.json")
    if config["binary"] != {"path": BINARY_PATH, "sha256": BINARY_SHA256}:
        raise ValueError("Only the pinned existing C++ executable is allowed")
    if config["mpi_tasks"] != 6 or config["restart_flag"] != 2 or config["nice"] != 19:
        raise ValueError("Required execution settings are MPI=6, nice=19, RestartFlag=2")
    require_hash(BINARY_PATH, BINARY_SHA256, "GIZMO executable")
    require_hash(trial / "orbit_000.hdf5", config["ic_sha256"], "Trial IC")
    require_hash(trial / "orbit_000.orbit.json", config["sidecar_sha256"], "IC sidecar")
    audit = read_json(trial / "orbit_000.orbit.json")
    if audit["output_ic_sha256"] != config["ic_sha256"]:
        raise ValueError("IC and sidecar hash disagree")
    if not math.isclose(float(audit["stage2_target_code"]),
                        float(config["production_target_code"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("IC sidecar and runtime target disagree")
    sub = trial / mode
    require_hash(sub / "stage2.param", config["param_sha256"][mode], "Run parameters")
    for item in config["runtime_dependencies"]:
        if item["mode"] == mode:
            require_hash(trial / item["relative_path"], item["sha256"], "Cooling dependency")
    if any((sub / "output").iterdir()):
        raise ValueError("Output directory is not empty; no overwrite or implicit restart")
    for reserved in ("launch.lock", "gizmo_run.log", "status.json", "status.json.partial"):
        if (sub / reserved).exists():
            raise ValueError(f"Previous run artifact exists: {sub / reserved}")
    if mode == "production":
        for reserved in ("orbit_validation", "orbit_validation.log"):
            if (trial / reserved).exists():
                raise ValueError(f"Previous validation artifact exists: {trial / reserved}")
        ready = read_json(trial / "production_ready.json")
        if ready.get("status") != "pass":
            raise ValueError("Production preflight has not passed")
        expected = {
            "ic_sha256": config["ic_sha256"],
            "production_param_sha256": config["param_sha256"]["production"],
            "binary_sha256": config["binary"]["sha256"],
            "sidecar_sha256": config["sidecar_sha256"],
        }
        for key, value in expected.items():
            if ready.get(key) != value:
                raise ValueError(f"Production readiness is stale or unbound: {key}")
        policy_path=trial.parent/"provenance/user_no_smoke_policy.json"
        policy=read_json(policy_path)
        if policy.get("smoke_required") is not False or ready.get("smoke_required") is not False:
            raise ValueError("Explicit direct-production user policy missing")
        if ready.get("user_no_smoke_policy_sha256") != sha256(policy_path):
            raise ValueError("Direct-production authorization binding changed")
    return trial, config


def last_snapshot(output):
    """Inspect only complete single-file snapshot names, sorted by physical time."""
    import h5py
    candidates = []
    for path in Path(output).iterdir():
        match = SNAPSHOT_RE.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            header = handle["Header"].attrs
            time_code = float(header["Time"])
            nfiles = int(header.get("NumFilesPerSnapshot", 1))
        if not math.isfinite(time_code) or nfiles != 1:
            raise ValueError(f"Invalid or unsupported multi-file snapshot: {path}")
        candidates.append((time_code, int(match.group(1)), path))
    if not candidates:
        return None, None
    time_code, _, path = max(candidates)
    return path, time_code


class ProcessInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def stream_process(command, cwd, env, log_path, timeout, on_start):
    """Mirror combined stdout/stderr to the terminal and an exclusive log."""
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   bufsize=1, stdin=subprocess.DEVNULL, start_new_session=True)
        old_handlers = {}
        def interrupted(signum, frame):
            raise ProcessInterrupted(signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        failures = []
        def mirror():
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except (BrokenPipeError, OSError):
                        pass
            except Exception as exc:
                failures.append(exc)
                stop_group(process)
        thread = threading.Thread(target=mirror, daemon=True)
        thread.start()
        try:
            on_start(process.pid)
            returncode = process.wait(timeout=timeout)
        except BaseException:
            stop_group(process)
            raise
        finally:
            thread.join(timeout=10)
            if not thread.is_alive():
                process.stdout.close()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        if thread.is_alive():
            raise RuntimeError("Output stream did not close after process exit")
        if failures:
            raise RuntimeError(f"Failed to record complete process output: {failures[0]}")
    return returncode


def run(trial, mode):
    trial, config = preflight(trial, mode)
    from stage_safety import verify_before_run
    verify_before_run(trial, mode)
    sub = trial / mode
    with (sub / "launch.lock").open("x") as lock:
        lock.write(json.dumps({"wrapper_pid": os.getpid(), "started_utc": utc()}) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               LD_LIBRARY_PATH=config["dynamic_library_path"],
               PYTHONPYCACHEPREFIX=str(trial / "pycache"),
               MPLCONFIGDIR=str(trial / "mpl"), TMPDIR=str(trial / "tmp"))
    for name in ("pycache", "mpl", "tmp"):
        (trial / name).mkdir(exist_ok=True)
    command = ["/usr/bin/mpirun", "-np", "6", "/usr/bin/nice", "-n", "19",
               BINARY_PATH, "stage2.param", "2"]
    status = {
        "state": "starting", "mode": mode, "wrapper_pid": os.getpid(),
        "started_utc": utc(), "command": command, "cwd": str(sub),
        "visible_terminal": os.environ.get("GIZMO_VISIBLE_TTY"),
        "ic_sha256": config["ic_sha256"], "param_sha256": config["param_sha256"][mode],
        "sidecar_sha256": config["sidecar_sha256"], "binary_sha256": BINARY_SHA256,
        "simulation_returncode": None, "returncode": None, "wrapper_returncode": None, "orbit_validation_returncode": None,
    }
    def save():
        write_status(sub / "status.json", status)
    def started(pid):
        status.update(state="running", mpi_pid=pid, process_group=pid)
        save()
    save()
    print(json.dumps(status), flush=True)
    phase = "simulation"
    try:
        rc = stream_process(command, sub, env, sub / "gizmo_run.log",
                            config["smoke_timeout_seconds"] if mode == "smoke" else None, started)
        status.update(simulation_returncode=rc, returncode=rc, simulation_ended_utc=utc())
        if rc != 0:
            status.update(state="simulation_failed", wrapper_returncode=rc if rc > 0 else 128 - rc, ended_utc=utc())
            save()
            return rc if rc > 0 else 128 - rc
        if mode == "smoke":
            status.update(state="smoke_completed", wrapper_returncode=0, ended_utc=utc(),
                          note="Execution passed; explicit field-preservation review is still required.")
            save()
            return 0
        final_path, final_time = last_snapshot(sub / "output")
        target = float(config["production_target_code"])
        status.update(final_snapshot=str(final_path) if final_path else None,
                      final_snapshot_time=final_time, target_time=target)
        if final_time is None or abs(final_time - target) > TIME_TOLERANCE:
            status.update(state="stopped_before_target", wrapper_returncode=3, ended_utc=utc())
            save()
            return 3
        phase = "validation"
        command = [PROJECT_PYTHON, VALIDATOR, "--snapshot-dir", str(sub / "output"),
                   "--reference-ic", str(trial / "orbit_000.hdf5"),
                   "--output-dir", str(trial / "orbit_validation")]
        status.update(state="validating", validation_started_utc=utc(), validation_command=command)
        save()
        def validation_started(pid):
            status.update(validator_pid=pid, validation_process_group=pid)
            save()
        rc = stream_process(command, trial, env, trial / "orbit_validation.log",
                            None, validation_started)
        status.update(orbit_validation_returncode=rc,
                      state="completed" if rc == 0 else "validation_failed",
                      wrapper_returncode=rc if rc >= 0 else 128 - rc, ended_utc=utc(),
                      orbit_validation_note="Execution completion does not imply the scientific 12D sigma gate passes.")
        save()
        return rc if rc >= 0 else 128 - rc
    except subprocess.TimeoutExpired:
        status.update(state="timeout", failed_phase=phase, wrapper_returncode=124, ended_utc=utc())
        save()
        return 124
    except ProcessInterrupted as exc:
        status.update(state="interrupted", failed_phase=phase, wrapper_returncode=128 + exc.signum, ended_utc=utc())
        save()
        return 128 + exc.signum
    except Exception as exc:
        status.update(state="failed", failed_phase=phase, wrapper_returncode=1,
                      error=f"{type(exc).__name__}: {exc}", ended_utc=utc())
        save()
        raise
    finally:
        print(json.dumps(status), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.trial_dir, args.mode)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
